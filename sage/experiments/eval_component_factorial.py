"""Separate memory and joint mean residuals with shared factual emissions."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from sage.experiments.closed_loop_protocol import audit_candidate_protocol, prepare_manifest
from sage.experiments.eval_persistent_closed_loop import Policy, _evaluate_policy, _load_model
from sage.experiments.finetune_h3_response_ranking import _adapter_from_config, _make_spec
from sage.experiments.formal_manifest import sha256_file
from sage.response.closed_loop import PlanningCostConfig


FACTORIAL_POLICIES = tuple(
    Policy(f"{memory}-{response}", memory, True, response_gate=True)
    for memory in ("uniform", "reset", "persistent")
    for response in ("joint", "additive")
)

CONTRASTS = {
    "memory_at_joint": {"persistent-joint": 1, "reset-joint": -1},
    "memory_at_additive": {"persistent-additive": 1, "reset-additive": -1},
    "joint_at_persistent": {"persistent-joint": 1, "persistent-additive": -1},
    "joint_at_reset": {"reset-joint": 1, "reset-additive": -1},
    "joint_at_uniform": {"uniform-joint": 1, "uniform-additive": -1},
    "belief_at_joint": {"persistent-joint": 1, "uniform-joint": -1},
    "memory_joint_interaction": {
        "persistent-joint": 1, "persistent-additive": -1,
        "reset-joint": -1, "reset-additive": 1,
    },
}

METRICS = (
    "realized_total_cost", "oracle_regret", "task_progress", "collision_rate",
    "decision_prefix_nll", "decision_mode_nll", "decision_mode_accuracy",
    "planning_prefix_nll", "planning_prefix_ade",
    "episode_collision", "cumulative_executed_cost",
)


def residual_weight(policy_name: str, joint_weight: float) -> float:
    if not np.isfinite(joint_weight) or not 0.0 <= joint_weight <= 1.0:
        raise ValueError("joint residual weight must lie in [0, 1]")
    if policy_name not in {policy.name for policy in FACTORIAL_POLICIES}:
        raise ValueError(f"unknown factorial policy: {policy_name}")
    return 0.0 if policy_name.endswith("-additive") else float(joint_weight)


def audit_shared_emissions(policies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    for memory in ("uniform", "reset", "persistent"):
        joint = policies[f"{memory}-joint"]["window_records"]
        additive = policies[f"{memory}-additive"]["window_records"]
        if len(joint) != len(additive):
            raise ValueError("factorial policies must contain the same episodes")
        for joint_episode, additive_episode in zip(joint, additive):
            if len(joint_episode) != len(additive_episode):
                raise ValueError("factorial policies must contain the same windows")
            for joint_row, additive_row in zip(joint_episode, additive_episode):
                for key in ("state_hash", "executed_response_hash", "executed_candidate"):
                    if joint_row[key] != additive_row[key]:
                        raise ValueError(f"factual evidence differs: {key}")
                for key in ("decision_mode_probabilities", "posterior_mode_probabilities"):
                    if not np.allclose(joint_row[key], additive_row[key], rtol=0, atol=1e-7):
                        raise ValueError(f"planning residual changed inference: {key}")
                if not np.isclose(joint_row["decision_prefix_nll"], additive_row["decision_prefix_nll"], rtol=0, atol=1e-7):
                    raise ValueError("planning residual changed emission likelihood")
                checked += 1
    return {"status": "pass", "checked_emission_pairs": checked}


def summarize_factorial(
    runs: dict[str, dict[str, Any]], resamples: int = 10000, seed: int = 614203,
    additional_metrics: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not runs or resamples < 1:
        raise ValueError("nonempty runs and positive resamples are required")
    metrics = METRICS + additional_metrics
    if len(metrics) != len(set(metrics)):
        raise ValueError("factorial metric names must be unique")
    result: dict[str, Any] = {}
    splits = list(next(iter(runs.values())))
    for split in splits:
        arrays: dict[str, dict[str, np.ndarray]] = {}
        episode_ids = None
        for policy in FACTORIAL_POLICIES:
            arrays[policy.name] = {}
            for metric in metrics:
                seed_values = []
                for seed_run in runs.values():
                    rows = seed_run[split][policy.name]["episode_records"]
                    identifiers = [row["episode"] for row in rows]
                    if len(set(identifiers)) != len(identifiers):
                        raise ValueError("duplicate episode identifiers")
                    if episode_ids is None:
                        episode_ids = identifiers
                    if identifiers != episode_ids:
                        raise ValueError("scene identities must align across seeds and policies")
                    seed_values.append([float(row[metric]) for row in rows])
                values = np.asarray(seed_values, dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(f"nonfinite factorial outcome: {metric}")
                arrays[policy.name][metric] = values
        assert episode_ids is not None
        generator = np.random.default_rng(seed)
        bootstrap = generator.integers(0, len(episode_ids), size=(resamples, len(episode_ids)))
        contrasts = {}
        for name, coefficients in CONTRASTS.items():
            contrasts[name] = {}
            for metric in metrics:
                differences = sum(
                    weight * arrays[policy][metric] for policy, weight in coefficients.items()
                )
                case_means = differences.mean(axis=0)
                interval = np.quantile(case_means[bootstrap].mean(axis=1), [0.025, 0.975])
                contrasts[name][metric] = {
                    "mean_difference": float(case_means.mean()),
                    "case_bootstrap_ci95": interval.tolist(),
                    "per_model_seed": dict(zip(runs, differences.mean(axis=1).tolist())),
                }
        result[split] = {
            "independent_scenes": len(episode_ids), "model_seeds": len(runs),
            "policy_means": {
                policy: {metric: float(values.mean()) for metric, values in metrics.items()}
                for policy, metrics in arrays.items()
            },
            "contrasts": contrasts,
            "difference_convention": "first minus second; negative is better for cost/NLL",
            "bootstrap_unit": "scene, after averaging the fixed model seeds",
        }
    return result


def _local_sources() -> list[Path]:
    return sorted({
        Path(module.__file__).resolve() for name, module in sys.modules.items()
        if (name.startswith("sage.") or name == "__main__") and getattr(module, "__file__", None)
        and str(module.__file__).endswith(".py")
    } | {Path(__file__).resolve()})


def run_factorial(args: argparse.Namespace) -> dict[str, Any]:
    raw = yaml.safe_load(Path(args.config).read_text())
    raw["simulation"]["data_seed"] = args.data_seed
    raw["simulation"]["episodes"] = args.episodes
    retain_previous_plan = bool(getattr(args, "retain_previous_plan", False) or raw.get("candidate_source", {}).get("retain_previous_plan", False))
    if retain_previous_plan:
        raw["candidate_source"]["retain_previous_plan"] = True
    if args.calibration_template:
        from sage.experiments.calibrate_component_memory import component_protocol_id, load_component_calibration
    value_template = getattr(args, "continuation_value_template", None)
    noise_template = getattr(args, "emission_noise_template", None)
    if noise_template:
        from sage.experiments.component_emission_protocol import load_emission_noise
    value_calibration_template = getattr(args, "continuation_calibration_template", None)
    if value_calibration_template and not value_template:
        raise ValueError("continuation calibration requires a value checkpoint")
    if value_template:
        from sage.experiments.component_value_protocol import load_value_artifact, load_value_weight

        if not args.prefix_objective or retain_previous_plan:
            raise ValueError("continuation value requires prefix scoring without plan retention")
    if args.residual_weight_calibration_template:
        from sage.experiments.calibrate_component_residual_weight import load_residual_weight

        if not args.prefix_objective:
            raise ValueError("residual reliability calibration requires prefix objective")
    torch.set_num_threads(args.num_threads)
    device = args.device
    planning = PlanningCostConfig(**raw["planning"])
    source_type = raw.get("candidate_source", {}).get("type", "synthetic")
    if source_type not in {"sage", "synthetic"}:
        raise ValueError("factorial evaluation supports sage or synthetic candidates")
    adapter = _adapter_from_config(raw, device) if source_type == "sage" else None
    checkpoint_paths = [Path(args.checkpoint_template.format(seed=seed)).resolve() for seed in args.seeds]
    checkpoints = list(checkpoint_paths)
    if noise_template:
        for seed in args.seeds:
            noise_path = Path(noise_template.format(seed=seed)).resolve()
            checkpoints.extend([noise_path, noise_path.parent / "run_manifest.json"])
    if value_template:
        for seed in args.seeds:
            value_path = Path(value_template.format(seed=seed)).resolve()
            checkpoints.extend([value_path, value_path.parent / "summary.json", value_path.parent / "run_manifest.json"])
            if value_calibration_template:
                calibration_path = Path(value_calibration_template.format(seed=seed)).resolve()
                checkpoints.extend([calibration_path, calibration_path.parent / "run_manifest.json"])
    if args.calibration_template:
        checkpoints.extend(Path(args.calibration_template.format(seed=seed)).resolve() for seed in args.seeds)
        checkpoints.extend(Path(args.calibration_template.format(seed=seed)).resolve().parent / "run_manifest.json" for seed in args.seeds)
    if args.emission_checkpoint_template:
        checkpoints.extend(Path(args.emission_checkpoint_template.format(seed=seed)).resolve() for seed in args.seeds)
    if args.residual_weight_calibration_template:
        checkpoints.extend(Path(args.residual_weight_calibration_template.format(seed=seed)).resolve() for seed in args.seeds)
    if adapter is not None:
        checkpoints.append(Path(raw["candidate_source"]["checkpoint"]).resolve())
    protocol = {
        "config": raw, "arguments": vars(args),
        "emissions": "same ungated full-joint model, executed factual prefix only",
        "uncertainty": "same elementwise maximum of additive/joint standard deviations",
        "gate": "constant numeric weight; independent of belief and observations",
        "additive_scope": "same-weights joint-mean-residual ablation, not a retrained matched baseline",
        "same_state_scope": "counterfactual one-step decisions on a shared Uniform-Joint rollout",
        "closed_loop_scope": "independent policy rollouts with paired initial states and random streams",
        "claim_boundary": "synthetic component attribution; not external navigation or universal superiority",
    }
    output = Path(args.output_dir)
    manifest = prepare_manifest(output, protocol, checkpoints, _local_sources(), resume=False)
    modes = {mode: {} for mode in args.evaluation_modes}
    audits: dict[str, Any] = {}
    effective_settings: dict[str, Any] = {}
    names = [policy.name for policy in FACTORIAL_POLICIES]
    for seed, checkpoint_path in zip(args.seeds, checkpoint_paths):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = _load_model(checkpoint_path, device, expected_dt=raw["simulation"]["dt"]).eval()
        if model.training_dt != float(raw["simulation"]["dt"]):
            raise ValueError("component evaluation must preserve the checkpoint training timestep")
        emission_checkpoint = Path(args.emission_checkpoint_template.format(seed=seed)).resolve() if args.emission_checkpoint_template else None
        emission_model = _load_model(emission_checkpoint, device) if emission_checkpoint else None
        noise_path = Path(noise_template.format(seed=seed)).resolve() if noise_template else None
        noise_model = load_emission_noise(noise_path, emission_checkpoint or checkpoint_path, raw, args.data_seed) if noise_path else None
        value_model = None
        value_weight = getattr(args, "continuation_value_weight", 1.)
        if value_template:
            value_path = Path(value_template.format(seed=seed)).resolve()
            value_model, _ = load_value_artifact(value_path, checkpoint_path, raw, args.data_seed)
            if value_calibration_template:
                value_weight = load_value_weight(value_calibration_template.format(seed=seed), value_path, raw, args.data_seed)
        for mode in modes:
            modes[mode][str(seed)] = {}
        for split in args.splits:
            spec = _make_spec(raw, split)
            joint_weight = args.joint_weight
            if args.residual_weight_calibration_template:
                joint_weight = load_residual_weight(args.residual_weight_calibration_template.format(seed=seed), checkpoint_path, raw, args.data_seed)
            likelihood_scale = args.likelihood_scale
            forgetting_rate = args.persistent_forgetting_rate
            if args.calibration_template:
                likelihood_scale = load_component_calibration(
                    args.calibration_template.format(seed=seed), checkpoint_path, spec, args.data_seed,
                    prefix_objective=args.prefix_objective,
                    emission_checkpoint=emission_checkpoint,
                    protocol_id=component_protocol_id(raw, args.prefix_objective),
                    joint_weight=joint_weight,
                    continuation_value_checkpoint=value_path if value_template else None,
                    continuation_value_weight=value_weight if value_template else 0.,
                    emission_noise_checkpoint=noise_path,
                )
                calibration = json.loads(Path(args.calibration_template.format(seed=seed)).read_text())
                forgetting_rate = float(calibration["selected"].get("persistent_forgetting_rate", 0.0))
            effective_settings[str(seed)] = {
                "likelihood_scale": likelihood_scale,
                "persistent_forgetting_rate": forgetting_rate,
                "persistent_label": "Reset-equivalent" if forgetting_rate == 1.0 else ("Persistent" if forgetting_rate == 0.0 else "Tempered-Persistent"),
                "planning_checkpoint": str(checkpoint_path),
                "emission_checkpoint": str(emission_checkpoint) if emission_checkpoint else str(checkpoint_path),
                "emission_noise_checkpoint": str(noise_path) if noise_path else None,
                "joint_weight": joint_weight, "prefix_objective": args.prefix_objective,
                "retain_previous_plan": retain_previous_plan,
                "continuation_value_checkpoint": str(value_path) if value_template else None,
                "continuation_value_weight": value_weight if value_template else 0.,
            }
            shared = dict(
                model=model, spec=spec, episodes=args.episodes, device=device,
                planning_cost=planning, likelihood_scale=likelihood_scale,
                candidate_adapter=adapter,
                prefix_objective=args.prefix_objective,
                persistent_forgetting_rate=forgetting_rate,
                emission_model=emission_model,
                retain_previous_plan=retain_previous_plan,
                continuation_value_model=value_model,
                continuation_value_weight=value_weight,
                emission_noise_model=noise_model,
            )
            trace: list = []
            reference = _evaluate_policy(
                **shared, policy=FACTORIAL_POLICIES[0], captured_reference_trace=trace,
                evaluation_mode="same_state",
                response_gate_min=joint_weight, response_gate_max=joint_weight,
            )
            for mode in modes:
                policy_results = {}
                for policy in FACTORIAL_POLICIES:
                    weight = residual_weight(policy.name, joint_weight)
                    if policy.name == "uniform-joint":
                        result = copy.deepcopy(reference)
                        for rows in result["window_records"]:
                            for row in rows:
                                row["evaluation_mode"] = mode
                    else:
                        result = _evaluate_policy(
                            **shared, policy=policy, evaluation_mode=mode,
                            reference_trace=trace if mode == "same_state" else None,
                            response_gate_min=weight, response_gate_max=weight,
                        )
                    policy_results[policy.name] = result
                    print(f"seed={seed} split={split} mode={mode} policy={policy.name} "
                          f"cost={result['realized_total_cost']['mean']:.6f}", flush=True)
                modes[mode][str(seed)][split] = policy_results
                protocol_audit = audit_candidate_protocol({str(seed): {split: policy_results}}, names, mode)
                if mode == "same_state":
                    protocol_audit["emissions"] = audit_shared_emissions(policy_results)
                audits[f"{seed}/{split}/{mode}"] = protocol_audit
                (output / f"seed{seed}_{split}_{mode}.json").write_text(json.dumps(policy_results, allow_nan=True))
    changed_sources = [path for path, digest in manifest["sources"].items() if sha256_file(Path(path)) != digest]
    changed_checkpoints = [path for path, digest in manifest["checkpoints"].items() if sha256_file(Path(path)) != digest]
    summary = {
        "status": "COMPLETED" if not changed_sources and not changed_checkpoints else "INVALIDATED",
        "phase": args.phase, "changed_sources": changed_sources,
        "changed_checkpoints": changed_checkpoints, "audits": audits,
        "effective_settings": effective_settings,
        "results": {mode: summarize_factorial(
            runs, args.bootstrap_resamples,
            additional_metrics=("terminal_goal_distance", "terminal_goal_success")
            if raw["simulation"].get("task_type") == "fixed_goal" else (),
        ) for mode, runs in modes.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if summary["status"] != "COMPLETED":
        raise RuntimeError("factorial evaluation inputs changed during execution")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-template", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027])
    parser.add_argument("--splits", nargs="+", choices=["id", "ood"], default=["id", "ood"])
    parser.add_argument("--evaluation-modes", nargs="+", choices=["same_state", "closed_loop"], default=["same_state", "closed_loop"])
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--phase", choices=["validation", "confirmation"], default="validation")
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--likelihood-scale", type=float, default=0.05)
    parser.add_argument("--calibration-template")
    parser.add_argument("--prefix-objective", action="store_true")
    parser.add_argument("--retain-previous-plan", action="store_true")
    parser.add_argument("--continuation-value-template")
    parser.add_argument("--continuation-value-weight", type=float, default=1.)
    parser.add_argument("--continuation-calibration-template")
    parser.add_argument("--persistent-forgetting-rate", type=float, default=0.0)
    parser.add_argument("--emission-checkpoint-template")
    parser.add_argument("--emission-noise-template")
    parser.add_argument("--residual-weight-calibration-template")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    residual_weight("uniform-joint", args.joint_weight)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("model seeds must be unique")
    run_factorial(args)


if __name__ == "__main__":
    main()
