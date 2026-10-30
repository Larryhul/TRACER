"""Compare equal-budget Uniform and six-policy factual-state coverage."""

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from tracer.experiments.closed_loop_protocol import prepare_manifest
from tracer.experiments.eval_component_factorial import FACTORIAL_POLICIES, _local_sources
from tracer.experiments.eval_persistent_closed_loop import _evaluate_policy, _load_model
from tracer.experiments.finetune_h3_response_ranking import _adapter_from_config, _make_spec
from tracer.experiments.formal_manifest import sha256_file
from tracer.experiments.train_component_residual import trace_to_rollouts
from tracer.response.closed_loop import PlanningCostConfig, SyntheticClosedLoopSimulator


@torch.no_grad()
def collect_policy_coverage(raw, adapter, model, emission, episodes, device, profile):
    if profile not in {"uniform", "factorial"} or episodes < 1:
        raise ValueError("positive episodes and a supported coverage profile are required")
    spec = _make_spec(raw, "id")
    planning = PlanningCostConfig(**raw["planning"])
    trace = [[None] * episodes for _ in range(spec.windows)]
    policy_indices = torch.arange(episodes) % len(FACTORIAL_POLICIES) if profile == "factorial" else torch.zeros(episodes, dtype=torch.long)
    audit = []
    for policy_index, policy in enumerate(FACTORIAL_POLICIES):
        indices = torch.where(policy_indices == policy_index)[0].tolist()
        if not indices:
            continue
        selected_trace = []
        candidate_seeds = [[spec.seed + episode * 104729 + window * 1009 for episode in indices] for window in range(spec.windows)]
        weight = float(policy.name.endswith("-joint"))
        result = _evaluate_policy(
            model, policy, spec, len(indices), device, planning, .05,
            emission_model=emission, candidate_adapter=adapter,
            response_gate_min=weight, response_gate_max=weight, prefix_objective=True,
            retain_previous_plan=bool(raw.get("candidate_source", {}).get("retain_previous_plan", False)),
            simulator_factory=lambda current_spec, index: SyntheticClosedLoopSimulator(current_spec, indices[index]),
            candidate_seed_sequences=candidate_seeds, evaluation_mode="same_state", captured_reference_trace=selected_trace,
        )
        for window in range(spec.windows):
            for local_index, episode in enumerate(indices):
                trace[window][episode] = selected_trace[window][local_index]
                row = result["window_records"][local_index][window]
                audit.append({"episode": episode, "window": window, "behavior": policy.name,
                              "state_hash": row["state_hash"], "candidate_hash": row["candidate_hash"],
                              "candidate_seed": row["candidate_seed"], "environment_rng_hash": row["environment_rng_hash"],
                              "executed_candidate": row["executed_candidate"], "executed_response_hash": row["executed_response_hash"]})
    batch = trace_to_rollouts(trace, spec, planning, model.config.num_modes, all_mode_supervision=True)
    batch["behavior_policy_index"] = policy_indices
    return batch, sorted(audit, key=lambda row: (row["episode"], row["window"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--emission-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-episodes", type=int, default=96)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--train-data-seed", type=int, default=924601)
    parser.add_argument("--validation-data-seed", type=int, default=924607)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.train_data_seed == args.validation_data_seed or min(args.train_episodes, args.validation_episodes) < 6:
        raise ValueError("disjoint data seeds and at least six scenes per split are required")
    torch.set_num_threads(4)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model = _load_model(Path(args.checkpoint), args.device)
    emission = _load_model(Path(args.emission_checkpoint), args.device)
    adapter = _adapter_from_config(raw, args.device) if raw["candidate_source"]["type"] == "sage" else None
    output = Path(args.output_dir)
    inputs = [Path(args.checkpoint).resolve(), Path(args.emission_checkpoint).resolve(), Path(args.config).resolve()]
    if adapter is not None:
        inputs.append(Path(raw["candidate_source"]["checkpoint"]).resolve())
    manifest = prepare_manifest(output, {"schema": "component-policy-coverage-v1", "all_mode_supervision": True, "arguments": vars(args), "config": raw,
                                       "assignment": "episode index modulo six, fixed before outcomes; Uniform control uses identical initial scenes and noise streams",
                                       "validation": "shared six-policy validation data for both training variants",
                                       "label_boundary": "all-mode cloned futures are offline training labels only, never controller or belief inputs"}, inputs, _local_sources(), False)
    controls = {}
    for split, profile, episodes, data_seed in [
        ("train", "uniform", args.train_episodes, args.train_data_seed),
        ("train", "factorial", args.train_episodes, args.train_data_seed),
        ("validation", "factorial", args.validation_episodes, args.validation_data_seed),
    ]:
        configuration = copy.deepcopy(raw)
        configuration["simulation"]["data_seed"] = data_seed
        batch, audit = collect_policy_coverage(configuration, adapter, model, emission, episodes, args.device, profile)
        directory = output / profile
        directory.mkdir(exist_ok=True)
        torch.save(batch, directory / f"{split}_rollouts.pt")
        (directory / f"{split}_trace.json").write_text(json.dumps(audit, indent=2) + "\n")
        if split == "train":
            controls[profile] = audit
        print(split, profile, "scenes", episodes, "policy_counts", torch.bincount(batch["behavior_policy_index"], minlength=6).tolist(), flush=True)
    first = {(row["episode"], row["window"]): row for row in controls["uniform"]}
    for row in controls["factorial"]:
        reference = first[row["episode"], row["window"]]
        if row["candidate_seed"] != reference["candidate_seed"] or row["environment_rng_hash"] != reference["environment_rng_hash"]:
            raise ValueError("coverage controls must share candidate and environment streams")
        if row["window"] == 0 and any(row[name] != reference[name] for name in ("state_hash", "candidate_hash")):
            raise ValueError("coverage controls must start at identical physical states and candidates")
    (output / "uniform/validation_rollouts.pt").symlink_to(output.resolve() / "factorial/validation_rollouts.pt")
    changed = [path for group in ("sources", "checkpoints") for path, digest in manifest[group].items() if sha256_file(Path(path)) != digest]
    payload = {"status": "INVALIDATED" if changed else "COMPLETED", "changed_inputs": changed, "paired_training_scenes": args.train_episodes,
               "shared_validation_sha256": sha256_file(output / "factorial/validation_rollouts.pt"),
               "trajectory_sha256": {str(path): sha256_file(path) for path in sorted(output.glob("*/*_rollouts.pt"))}}
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    if changed:
        raise RuntimeError("coverage collection inputs changed")


if __name__ == "__main__":
    main()
