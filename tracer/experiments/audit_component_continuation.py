"""Separate immediate intervention value from common-controller continuation."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from tracer.experiments.closed_loop_protocol import audit_candidate_protocol, prepare_manifest
from tracer.experiments.eval_component_factorial import FACTORIAL_POLICIES, _local_sources, audit_shared_emissions
from tracer.experiments.eval_persistent_closed_loop import _evaluate_policy, _load_model, _realized_total_cost
from tracer.experiments.finetune_h3_response_ranking import _adapter_from_config, _make_spec
from tracer.experiments.formal_manifest import sha256_file
from tracer.response.closed_loop import PlanningCostConfig


def first_eligible_disagreements(first, second, continuation_windows):
    if len(first) != len(second):
        raise ValueError("policy episodes must align")
    points = []
    for episode, (first_rows, second_rows) in enumerate(zip(first, second)):
        if len(first_rows) != len(second_rows):
            raise ValueError("policy windows must align")
        for window in range(max(0, len(first_rows) - continuation_windows)):
            left, right = first_rows[window], second_rows[window]
            if left["state_hash"] != right["state_hash"] or left["candidate_hash"] != right["candidate_hash"]:
                raise ValueError("interventions require identical factual states and candidates")
            if left["selected_candidate"] != right["selected_candidate"]:
                points.append({"episode": episode, "window": window, "first_action": left["selected_candidate"], "second_action": right["selected_candidate"]})
                break
    return points


def continuation_seed_sequences(spec, points, windows):
    split_offset = 0 if spec.split == "id" else 7_000_003
    return [[spec.seed + split_offset + point["episode"] * 104729 + (point["window"] + 1 + window) * 1009 for point in points] for window in range(windows)]


def summarize_interventions(records, episodes):
    output = {"episodes": episodes, "interventions": len(records), "coverage": len(records) / episodes}
    for metric in ("immediate_cost_difference", "continuation_cost_difference", "combined_cost_difference"):
        values = np.zeros(episodes)
        for record in records:
            values[record["episode"]] = record[metric]
        bootstrap = np.random.default_rng(924627).integers(0, episodes, size=(4000, episodes))
        output[metric] = {"all_scene_mean": float(values.mean()), "intervened_scene_mean": float(np.mean([record[metric] for record in records])) if records else None, "case_bootstrap_ci95": np.quantile(values[bootstrap].mean(1), [.025, .975]).tolist()}
    output["immediate_improvement_lost"] = sum(record["immediate_cost_difference"] < 0 <= record["combined_cost_difference"] for record in records)
    return output


def compare_continuation(trace, first, second, shared, windows):
    points = first_eligible_disagreements(first["window_records"], second["window_records"], windows)
    spec, planning = shared["spec"], shared["planning_cost"]
    if not points:
        return {"summary": summarize_interventions([], shared["episodes"]), "records": [], "continuation_audits": []}
    branches, immediate_costs, immediate_collisions = [], [], []
    for action_key in ("first_action", "second_action"):
        states, costs, collisions, previous_plans = [], [], [], []
        for point in points:
            reference = trace[point["window"]][point["episode"]]
            simulator = copy.deepcopy(reference.simulator)
            outcome = simulator.step(reference.pool, point[action_key], planning)
            states.append(simulator)
            costs.append(_realized_total_cost(outcome, spec, planning))
            collisions.append(outcome.collision)
            previous_plans.append(reference.pool.robot_candidates[point[action_key]].detach().clone())
        result = _evaluate_policy(
            **{name: value for name, value in shared.items() if name not in {"spec", "episodes"}},
            policy=FACTORIAL_POLICIES[0], spec=replace(spec, windows=windows), episodes=len(points),
            simulator_factory=lambda _spec, index: copy.deepcopy(states[index]),
            candidate_seed_sequences=continuation_seed_sequences(spec, points, windows),
            initial_previous_plans=previous_plans if shared.get("retain_previous_plan", False) else None,
        )
        branches.append(result)
        immediate_costs.append(costs)
        immediate_collisions.append(collisions)
    records, audits = [], []
    for index, point in enumerate(points):
        first_value = first["window_records"][point["episode"]][point["window"]]["selected_true_cost"]
        second_value = second["window_records"][point["episode"]][point["window"]]["selected_true_cost"]
        if not np.allclose([first_value, second_value], [immediate_costs[0][index], immediate_costs[1][index]], atol=1e-6, rtol=1e-6):
            raise ValueError("intervention prefix costs do not reproduce the original counterfactuals")
        for left, right in zip(branches[0]["window_records"][index], branches[1]["window_records"][index]):
            if left["candidate_seed"] != right["candidate_seed"] or left["environment_rng_hash"] != right["environment_rng_hash"]:
                raise ValueError("continuation candidate and environment random streams must be paired")
            retained = bool(shared.get("retain_previous_plan", False))
            if retained and any(record["retained_plan_candidate"] < 0 for record in (left, right)):
                raise ValueError("continuation must preserve the intervened plan from its first window")
            audits.append({"episode": point["episode"], "original_window": point["window"], "continuation_window": left["window"], "candidate_seed": left["candidate_seed"], "paired_environment_rng": True, "retained_plan_present": retained})
        immediate_difference = immediate_costs[0][index] - immediate_costs[1][index]
        continuation_difference = branches[0]["episode_records"][index]["cumulative_executed_cost"] - branches[1]["episode_records"][index]["cumulative_executed_cost"]
        records.append({**point, "immediate_cost_difference": immediate_difference, "continuation_cost_difference": continuation_difference, "combined_cost_difference": immediate_difference + continuation_difference, "first_collision": bool(immediate_collisions[0][index] or branches[0]["episode_records"][index]["episode_collision"]), "second_collision": bool(immediate_collisions[1][index] or branches[1]["episode_records"][index]["episode_collision"])})
    return {"summary": summarize_interventions(records, shared["episodes"]), "records": records, "continuation_audits": audits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--emission-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--data-seed", type=int, default=924613)
    parser.add_argument("--continuation-windows", type=int, default=2)
    parser.add_argument("--splits", nargs="+", choices=["id", "ood"], default=["id", "ood"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw["simulation"]["data_seed"] = args.data_seed
    if args.continuation_windows < 1 or args.continuation_windows >= raw["simulation"]["windows"]:
        raise ValueError("continuation must fit inside the original episode")
    model = _load_model(Path(args.checkpoint), args.device)
    emission = _load_model(Path(args.emission_checkpoint), args.device)
    adapter = _adapter_from_config(raw, args.device) if raw["candidate_source"]["type"] == "sage" else None
    inputs = [Path(args.checkpoint).resolve(), Path(args.emission_checkpoint).resolve(), Path(args.config).resolve()]
    if adapter is not None:
        inputs.append(Path(raw["candidate_source"]["checkpoint"]).resolve())
    output = Path(args.output_dir)
    manifest = prepare_manifest(output, {"arguments": vars(args), "config": raw, "scope": "first eligible action disagreement on a shared Uniform factual stream, then the same Uniform-Joint continuation; post-hoc diagnostic not full-policy superiority"}, inputs, _local_sources(), False)
    results = {}
    for split in args.splits:
        spec = _make_spec(raw, split)
        shared = dict(model=model, emission_model=emission, spec=spec, episodes=args.episodes, device=args.device,
                      planning_cost=PlanningCostConfig(**raw["planning"]), likelihood_scale=.05,
                      candidate_adapter=adapter, prefix_objective=True, response_gate_min=1., response_gate_max=1.,
                      retain_previous_plan=bool(raw.get("candidate_source", {}).get("retain_previous_plan", False)))
        trace, policies = [], {}
        policies[FACTORIAL_POLICIES[0].name] = _evaluate_policy(**shared, policy=FACTORIAL_POLICIES[0], captured_reference_trace=trace, evaluation_mode="same_state")
        for policy in FACTORIAL_POLICIES[1:]:
            arguments = {**shared, "response_gate_min": float(policy.name.endswith("-joint")), "response_gate_max": float(policy.name.endswith("-joint"))}
            policies[policy.name] = _evaluate_policy(**arguments, policy=policy, reference_trace=trace, evaluation_mode="same_state")
        audit = audit_candidate_protocol({"checkpoint": {split: policies}}, list(policies), "same_state")
        emissions = audit_shared_emissions(policies)
        contrasts = {}
        for name, comparator in [("joint", "persistent-additive"), ("memory", "reset-joint")]:
            contrasts[name] = compare_continuation(trace, policies["persistent-joint"], policies[comparator], shared, args.continuation_windows)
            print(split, name, contrasts[name]["summary"], flush=True)
        results[split] = {"state_audit": audit, "emission_audit": emissions, "contrasts": contrasts}
        (output / f"{split}.json").write_text(json.dumps(results[split], indent=2, allow_nan=False) + "\n")
    changed = [filename for group in ("sources", "checkpoints") for filename, digest in manifest[group].items() if sha256_file(Path(filename)) != digest]
    summary = {"status": "COMPLETED" if not changed else "INVALIDATED", "changed_inputs": changed, "results": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if changed:
        raise RuntimeError("continuation audit source changed")


if __name__ == "__main__":
    main()
