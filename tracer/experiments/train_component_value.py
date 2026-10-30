"""Fit a common observable-state continuation head from factual episode returns."""

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from tracer.experiments.calibrate_component_memory import component_protocol_id
from tracer.experiments.closed_loop_protocol import prepare_manifest
from tracer.experiments.eval_component_factorial import FACTORIAL_POLICIES, _local_sources
from tracer.experiments.eval_persistent_closed_loop import _evaluate_policy, _load_model
from tracer.experiments.finetune_h3_response_ranking import _adapter_from_config, _make_spec
from tracer.experiments.formal_manifest import sha256_file
from tracer.response.closed_loop import PlanningCostConfig
from tracer.response.continuation_value import FrozenContinuationValue, observable_value_features


def collect_value_targets(raw, model, adapter, episodes, device):
    spec = _make_spec(raw, "id")
    trace = []
    evaluation = _evaluate_policy(model, FACTORIAL_POLICIES[0], spec, episodes, device, PlanningCostConfig(**raw["planning"]), .05,
                                  candidate_adapter=adapter, prefix_objective=True, response_gate_min=1., response_gate_max=1.,
                                  evaluation_mode="same_state", captured_reference_trace=trace)
    features, modes, returns, remaining = [], [], [], []
    for episode in range(episodes):
        costs = torch.tensor([row["executed_total_cost"] for row in evaluation["window_records"][episode]], dtype=torch.float64)
        suffix_returns = costs.flip(0).cumsum(0).flip(0)
        for window in range(spec.windows):
            reference = trace[window][episode]
            features.append(observable_value_features(torch.from_numpy(reference.simulator.history), reference.pool.robot_current[0], spec.windows - window, spec.windows, spec.dt))
            modes.append(reference.simulator.mode)
            returns.append(suffix_returns[window])
            remaining.append(spec.windows - window)
    return {"features": torch.stack(features).double(), "modes": torch.tensor(modes), "returns": torch.stack(returns), "remaining": torch.tensor(remaining).double()}, evaluation


def fit_value_head(data, num_modes, regularization):
    center = data["features"].mean(0)
    scale = data["features"].std(0).clamp_min(1e-4)
    normalized = (data["features"] - center) / scale
    design = torch.cat([torch.ones_like(normalized[:, :1]), normalized], -1)
    target = data["returns"] / data["remaining"]
    penalty = torch.eye(design.shape[-1], dtype=design.dtype) * regularization
    penalty[0, 0] = 0.
    coefficients = []
    for mode in range(num_modes):
        active = data["modes"] == mode
        if not active.any():
            raise ValueError("factual value training must observe every response mode")
        matrix = design[active]
        coefficients.append(torch.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ target[active]))
    return center, scale, torch.stack(coefficients)


def value_error(model, data):
    features = data["features"][:, None].expand(-1, model.coefficients.shape[0], -1)
    rates = model.mode_values(features, 1)
    selected = rates[torch.arange(len(rates)), data["modes"]]
    return float((selected - data["returns"] / data["remaining"]).square().mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-episodes", type=int, default=192)
    parser.add_argument("--validation-episodes", type=int, default=64)
    parser.add_argument("--train-data-seed", type=int, default=924601)
    parser.add_argument("--validation-data-seed", type=int, default=924607)
    parser.add_argument("--regularizations", type=float, nargs="+", default=[.01, .1, 1., 10., 100.])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.train_data_seed == args.validation_data_seed or min(args.regularizations) <= 0:
        raise ValueError("disjoint scene seeds and positive regularization required")
    torch.set_num_threads(4)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if raw["candidate_source"].get("retain_previous_plan", False) or raw["simulation"].get("task_type", "entity_approach") != "entity_approach":
        raise ValueError("value pilot requires memoryless candidate generation and the entity-approach task")
    checkpoint = Path(args.checkpoint).resolve()
    model = _load_model(checkpoint, args.device)
    adapter = _adapter_from_config(raw, args.device)
    output = Path(args.output_dir)
    manifest = prepare_manifest(output, {"arguments": vars(args), "config": raw,
                                       "target": "actual remaining executed cost under frozen Uniform-Joint continuation, normalized per remaining window",
                                       "mode_boundary": "simulator mode labels fit offline mode-specific heads; deployment marginalizes current factual belief and never receives a true mode",
                                       "selection": "validation mean squared per-window return prediction; no test or Persistent-minus-Reset selection"},
                                [checkpoint, Path(args.config).resolve(), Path(raw["candidate_source"]["checkpoint"]).resolve()], _local_sources(), False)
    datasets = {}
    for split, episodes, seed in [("train", args.train_episodes, args.train_data_seed), ("validation", args.validation_episodes, args.validation_data_seed)]:
        configuration = copy.deepcopy(raw)
        configuration["simulation"]["data_seed"] = seed
        data, evaluation = collect_value_targets(configuration, model, adapter, episodes, args.device)
        datasets[split] = data
        torch.save(data, output / f"{split}.pt")
        (output / f"{split}_rollout.json").write_text(json.dumps(evaluation, allow_nan=True))
        print(split, "factual scenes", episodes, "state targets", len(data["modes"]), flush=True)
    trials = []
    for regularization in args.regularizations:
        center, scale, coefficients = fit_value_head(datasets["train"], model.config.num_modes, regularization)
        head = FrozenContinuationValue(center, scale, coefficients, raw["simulation"]["windows"], model.config.dt, raw["simulation"]["execute_steps"])
        trials.append((value_error(head, datasets["validation"]), regularization, head))
    best_error, regularization, head = min(trials, key=lambda value: (value[0], -value[1]))
    payload = {"center": head.center, "scale": head.scale, "coefficients": head.coefficients, "total_windows": head.total_windows, "dt": head.dt, "execute_steps": head.execute_steps,
               "regularization": regularization, "planning_checkpoint_sha256": sha256_file(checkpoint), "component_protocol_id": component_protocol_id(raw, True),
               "training_data_seed": args.train_data_seed, "validation_data_seed": args.validation_data_seed}
    torch.save(payload, output / "value.pt")
    changed = [path for group in ("sources", "checkpoints") for path, digest in manifest[group].items() if sha256_file(Path(path)) != digest]
    summary = {"status": "INVALIDATED" if changed else "COMPLETED", "changed_inputs": changed, "selected_regularization": regularization, "validation_mse": best_error,
               "grid": [{"regularization": penalty, "validation_mse": error} for error, penalty, _ in trials], "value_sha256": sha256_file(output / "value.pt")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary, flush=True)
    if changed:
        raise RuntimeError("value training inputs changed")


if __name__ == "__main__":
    main()
