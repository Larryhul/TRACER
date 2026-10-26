"""Check whether known simulator modes can improve matched component decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sage.experiments.closed_loop_protocol import audit_candidate_protocol, prepare_manifest
from sage.experiments.eval_component_factorial import _local_sources
from sage.experiments.eval_persistent_closed_loop import Policy, _evaluate_policy, _load_model, _realized_total_cost
from sage.experiments.finetune_h3_response_ranking import _adapter_from_config, _make_spec
from sage.experiments.formal_manifest import sha256_file
from sage.response.closed_loop import PlanningCostConfig


def simulator_mode_information(trace, factual, planning):
    records = []
    for window_index, windows in enumerate(trace):
        for episode_index, record in enumerate(windows):
            simulator = record.simulator
            mode = simulator.mode
            costs = []
            try:
                for candidate_mode in range(3):
                    simulator.mode = candidate_mode
                    outcomes = simulator.counterfactuals(record.pool, planning)
                    costs.append([_realized_total_cost(outcome, simulator.spec, planning) for outcome in outcomes])
            finally:
                simulator.mode = mode
            cost_array = np.asarray(costs)
            factual_row = factual["window_records"][episode_index][window_index]
            support = np.asarray(factual_row["planning_admissible_mask"], dtype=bool)
            if not support.any():
                records.append({"episode": episode_index, "window": window_index, "status": "no_common_support"})
                continue
            known_selected = int(np.where(support, cost_array[mode], np.inf).argmin())
            uniform_selected = int(np.where(support, cost_array.mean(axis=0), np.inf).argmin())
            actual_costs = -np.asarray(factual_row["ground_truth_candidate_utility"])
            if not np.allclose(actual_costs, cost_array[mode], rtol=0, atol=1e-6):
                raise ValueError("simulator mode intervention changed factual outcomes")
            records.append({
                "episode": episode_index, "window": window_index, "status": "matched",
                "known_selected": known_selected, "uniform_selected": uniform_selected,
                "known_mode_gain": float(cost_array[mode, uniform_selected] - cost_array[mode, known_selected]),
                "mode_specific_cost_spread": float(np.ptp(cost_array - cost_array.mean(axis=-1, keepdims=True), axis=0).mean()),
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--data-seed", type=int, default=924613)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix-objective", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw = yaml.safe_load(Path(args.config).read_text())
    raw["simulation"]["data_seed"] = args.data_seed
    model = _load_model(Path(args.checkpoint), args.device)
    if model.config.dt != raw["simulation"]["dt"]:
        raise ValueError("information audit must preserve the trained timestep")
    adapter = _adapter_from_config(raw, args.device)
    output = Path(args.output_dir)
    manifest = prepare_manifest(output, {
        "arguments": vars(args), "config": raw,
        "claim_boundary": "known simulator mode under the learned model, not an optimal simulator oracle",
    }, [Path(args.checkpoint).resolve(), Path(raw["candidate_source"]["checkpoint"]).resolve()], _local_sources(), False)
    summaries = {}
    for split in ("id", "ood"):
        arguments = dict(
            model=model, spec=_make_spec(raw, split), episodes=args.episodes,
            device=args.device, planning_cost=PlanningCostConfig(**raw["planning"]),
            likelihood_scale=0.05, candidate_adapter=adapter,
            response_gate_min=1.0, response_gate_max=1.0,
            evaluation_mode="same_state", prefix_objective=args.prefix_objective,
            retain_previous_plan=bool(raw.get("candidate_source", {}).get("retain_previous_plan", False)),
        )
        trace = []
        uniform = _evaluate_policy(**arguments, policy=Policy("uniform", "uniform", True, response_gate=True), captured_reference_trace=trace)
        known = _evaluate_policy(**arguments, policy=Policy("known", "oracle", True, response_gate=True), reference_trace=trace)
        policies = {"uniform": uniform, "known": known}
        audit = audit_candidate_protocol({"checkpoint": {split: policies}}, list(policies), "same_state")
        changes = 0
        windows = 0
        for uniform_rows, known_rows in zip(uniform["window_records"], known["window_records"]):
            for uniform_row, known_row in zip(uniform_rows, known_rows):
                changes += known_row["belief_action_flip"]
                windows += 1
        differences = np.asarray([
            known_row["realized_total_cost"] - uniform_row["realized_total_cost"]
            for uniform_row, known_row in zip(uniform["episode_records"], known["episode_records"])
        ])
        generator = np.random.default_rng(614203)
        samples = generator.integers(0, args.episodes, size=(4000, args.episodes))
        summaries[split] = {
            "audit": audit, "known_mode_executed_action_changes": changes,
            "windows": windows, "known_minus_uniform_cost": float(differences.mean()),
            "case_bootstrap_ci95": np.quantile(differences[samples].mean(axis=1), [0.025, 0.975]).tolist(),
        }
        simulator_records = simulator_mode_information(trace, uniform, arguments["planning_cost"])
        comparable = [record for record in simulator_records if record["status"] == "matched"]
        summaries[split]["simulator_mode_information"] = {
            "matched_windows": len(comparable),
            "known_mode_changes": sum(record["known_selected"] != record["uniform_selected"] for record in comparable),
            "mean_gain": float(np.mean([record["known_mode_gain"] for record in comparable])) if comparable else None,
            "scope": "optimistic finite-pool diagnostic with hidden preferred velocity shared by both; not deployable performance",
        }
        (output / f"{split}_simulator_mode_information.json").write_text(json.dumps(simulator_records, indent=2))
        (output / f"{split}.json").write_text(json.dumps(policies, allow_nan=True))
        print(split, summaries[split], flush=True)
    changed = [path for group in ("sources", "checkpoints") for path, digest in manifest[group].items() if sha256_file(Path(path)) != digest]
    result = {"status": "COMPLETED" if not changed else "INVALIDATED", "changed_inputs": changed, "results": summaries}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if changed:
        raise RuntimeError("information audit inputs changed")


if __name__ == "__main__":
    main()
