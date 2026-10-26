import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sage.experiments.closed_loop_protocol import prepare_manifest
from sage.experiments.eval_component_factorial import _local_sources
from sage.experiments.eval_persistent_closed_loop import _realized_total_cost
from sage.experiments.finetune_h3_response_ranking import _make_spec
from sage.experiments.formal_manifest import sha256_file
from sage.response.closed_loop import CandidatePool, PlanningCostConfig, SyntheticClosedLoopSimulator
from sage.response.closed_loop_contract import observable_state_sha256, task_goals, validate_candidate_state
from sage.response.coordinated_candidates import complete_joint_speed_candidates
from sage.response.persistent_models import stationary_reference_candidates


GROUP_FIELDS = ("history", "current", "candidates", "all_mode_responses", "all_mode_references", "all_mode_costs")


def response_labels(simulator, pool, planning, num_modes):
    validate_candidate_state(simulator, pool)
    rng_before = copy.deepcopy(simulator.rng.bit_generator.state)
    mode_before = simulator.mode
    noise = simulator.rng.normal(0., simulator.spec.response_noise, size=(simulator.spec.horizon, 2)).astype(np.float32)
    simulator.rng.bit_generator.state = rng_before
    references = stationary_reference_candidates(pool.robot_current, simulator.spec.horizon).numpy()
    responses, reference_responses, costs = [], [], []
    try:
        for mode in range(num_modes):
            simulator.mode = mode
            responses.append(simulator._response_batch(pool.robot_current.numpy(), pool.robot_candidates.numpy(), noise=noise))
            reference_responses.append(simulator._response_batch(pool.robot_current.numpy(), references, noise=noise))
            costs.append([_realized_total_cost(outcome, simulator.spec, planning) for outcome in simulator.counterfactuals(pool, planning)])
    finally:
        simulator.mode = mode_before
        simulator.rng.bit_generator.state = rng_before
    return {
        "all_mode_responses": torch.from_numpy(np.asarray(responses, dtype=np.float32)),
        "all_mode_references": torch.from_numpy(np.asarray(reference_responses, dtype=np.float32)),
        "all_mode_costs": torch.tensor(costs, dtype=torch.float32),
    }


def relabel_observed_states(data, trace, spec, planning, speed_scales, goal_speed):
    if any(name not in data for name in GROUP_FIELDS):
        raise ValueError("complete grouped response supervision required")
    episodes, windows = data["history"].shape[:2]
    if windows != spec.windows:
        raise ValueError("recorded and configured windows must match")
    rows = {(int(row["episode"]), int(row["window"])): row for row in trace}
    expected = {(episode, window) for episode in range(episodes) for window in range(windows)}
    if len(rows) != len(trace) or set(rows) != expected:
        raise ValueError("factual trace must cover every scene/window exactly once")
    original = {name: data[name].clone() for name in GROUP_FIELDS}
    completed = {name: [[] for _ in range(episodes)] for name in GROUP_FIELDS}
    audit = []
    for episode in range(episodes):
        simulator = SyntheticClosedLoopSimulator(spec, episode)
        for window in range(windows):
            row = rows[(episode, window)]
            if observable_state_sha256(simulator) != row["state_hash"]:
                raise ValueError("source factual state does not match replay")
            rng_hash = hashlib.sha256(json.dumps(simulator.rng.bit_generator.state, sort_keys=True).encode()).hexdigest()
            if rng_hash != row["environment_rng_hash"]:
                raise ValueError("source factual random stream does not match replay")
            torch.testing.assert_close(torch.from_numpy(simulator.history), data["history"][episode, window], rtol=0, atol=0)
            current, candidates = data["current"][episode, window], data["candidates"][episode, window]
            pool = CandidatePool(current, candidates, torch.zeros(current.shape[:2]))
            num_modes = data["all_mode_responses"].shape[2]
            factual_labels = response_labels(simulator, pool, planning, num_modes)
            for name, values in factual_labels.items():
                torch.testing.assert_close(values, data[name][episode, window], rtol=1e-5, atol=1e-6)
            proposal = complete_joint_speed_candidates(pool, task_goals(simulator)-simulator.robot_positions,
                spec.dt, goal_speed, speed_scales, spec.execute_steps, stop_at_goal=spec.task_type == "fixed_goal")
            labels = response_labels(simulator, proposal, planning, num_modes)
            for name, value in {"history": data["history"][episode, window], "current": proposal.robot_current,
                                "candidates": proposal.robot_candidates, **labels}.items():
                completed[name][episode].append(value)
            executed = int(row["executed_candidate"])
            if not 0 <= executed < len(candidates):
                raise ValueError("recorded factual execution is outside the original pool")
            outcome = simulator.step(pool, executed, planning)
            response_hash = hashlib.sha256(outcome.response[:spec.execute_steps].numpy().tobytes()).hexdigest()
            if response_hash != row["executed_response_hash"]:
                raise ValueError("relabeling changed the factual execution stream")
            audit.append({"episode": episode, "window": window, "state_hash": row["state_hash"],
                          "executed_response_hash": response_hash, "candidate_budget": len(proposal.robot_candidates)})
    completed = {name: torch.stack([torch.stack(values) for values in scene_values]) for name, scene_values in completed.items()}
    if any(completed[name].shape != original[name].shape for name in GROUP_FIELDS):
        raise ValueError("matched candidate relabeling changed supervision budget")
    return original, completed, audit


def validate_relabel_protocol(source, target):
    for raw in (source, target):
        if raw["simulation"].get("task_type", "entity_approach") != "entity_approach":
            raise ValueError("candidate relabeling currently supports entity-approach cost only")
    source_spec, target_spec = vars(_make_spec(source, "id")), vars(_make_spec(target, "id"))
    for spec in (source_spec, target_spec):
        spec.pop("seed")
    if source_spec != target_spec or source["planning"] != target["planning"]:
        raise ValueError("relabeling cannot change simulator cadence or planning cost")
    if not target["candidate_source"].get("joint_speed_completion"):
        raise ValueError("explicit joint candidate completion required")
    for key in ("goal_speed", "goal_direction", "task_prefix_steps"):
        if source["candidate_source"].get(key) != target["candidate_source"].get(key):
            raise ValueError(f"candidate relabeling changed shared observable rule: {key}")


def main():
    parser = argparse.ArgumentParser(description="Re-label fixed factual training states with equal-budget observable joint candidates")
    parser.add_argument("--config", required=True)
    parser.add_argument("--coverage-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    coverage, output = Path(args.coverage_dir).resolve(), Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    target = yaml.safe_load(config_path.read_text())
    source_path, summary_path = coverage / "run_manifest.json", coverage / "summary.json"
    source = json.loads(source_path.read_text())["config"]
    summary = json.loads(summary_path.read_text())
    if source.get("schema") != "component-policy-coverage-v1" or summary.get("status") != "COMPLETED" or summary.get("changed_inputs"):
        raise ValueError("completed unchanged source policy coverage required")
    validate_relabel_protocol(source["config"], target)
    inputs = [config_path, source_path, summary_path]
    for split, profile in (("train", "uniform"), ("validation", "factorial")):
        data_path = coverage / profile / f"{split}_rollouts.pt"
        if summary["trajectory_sha256"].get(str(data_path)) != sha256_file(data_path):
            raise ValueError("source trajectory hash mismatch")
        inputs.extend([data_path, coverage / profile / f"{split}_trace.json"])
    manifest = prepare_manifest(output, {
        "schema": "component-candidate-relabel-v1", "config": target, "source_config": source["config"],
        "arguments": vars(args), "source_arguments": source["arguments"],
        "all_mode_supervision": True,
        "state_scope": "Original Uniform factual training states; no claim of completed-candidate on-policy occupancy.",
        "label_scope": "Common-noise offline all-mode simulator labels only. Completed candidate targets are NOT an executed evidence stream and cannot fit online belief as factual observations.",
        "validation_scope": "Same 32 validation states represented once per original/completed candidate family, equal family weights. Duplicate contexts are not independent scenes; validation is a training loss, not a significance test.",
        "budget": "Equal original/completed training scene, window, candidate and epoch budgets; no outcome-selected candidates or scene filtering."},
        inputs, [Path(__file__).resolve(), *_local_sources()], False)
    datasets, audits = {}, {}
    for split, profile in (("train", "uniform"), ("validation", "factorial")):
        data = torch.load(coverage / profile / f"{split}_rollouts.pt", weights_only=True, map_location="cpu")
        trace = json.loads((coverage / profile / f"{split}_trace.json").read_text())
        scene_config = copy.deepcopy(source["config"])
        scene_config["simulation"]["data_seed"] = source["arguments"][f"{split}_data_seed"]
        spec = _make_spec(scene_config, "id")
        original, completed, audit = relabel_observed_states(data, trace, spec, PlanningCostConfig(**target["planning"]),
            target["candidate_source"]["joint_speed_completion"], target["candidate_source"]["goal_speed"])
        datasets[split], audits[split] = {"original": original, "completed": completed}, audit
        print(json.dumps({"split": split, "factual_scenes": len(data["history"]), "audited_windows": len(audit)}), flush=True)
    validation = {name: torch.cat([datasets["validation"][family][name] for family in ("original", "completed")], dim=0) for name in GROUP_FIELDS}
    digests = {}
    for family in ("original", "completed"):
        directory = output / family
        directory.mkdir()
        for split, data in (("train", datasets["train"][family]), ("validation", validation)):
            path = directory / f"{split}_rollouts.pt"
            torch.save(data, path)
            digests[str(path)] = sha256_file(path)
    first = output / "original/validation_rollouts.pt"
    second = output / "completed/validation_rollouts.pt"
    second.write_bytes(first.read_bytes())
    digests[str(second)] = sha256_file(second)
    (output / "factual_replay_audit.json").write_text(json.dumps(audits, indent=2)+"\n")
    changed = [path for category in ("sources", "checkpoints") for path, digest in manifest[category].items() if sha256_file(Path(path)) != digest]
    result = {"schema": "component-candidate-relabel-v1", "status": "INVALIDATED" if changed else "COMPLETED", "changed_inputs": changed,
              "trajectory_sha256": digests, "shared_validation_sha256": sha256_file(first),
              "training_independent_scenes": len(datasets["train"]["original"]["history"]),
              "validation_independent_scenes": len(datasets["validation"]["original"]["history"]),
              "validation_context_repetitions": 2, "factual_replay_audit_sha256": sha256_file(output / "factual_replay_audit.json")}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    if changed:
        raise RuntimeError("candidate relabeling sources changed during collection")


if __name__ == "__main__":
    main()
