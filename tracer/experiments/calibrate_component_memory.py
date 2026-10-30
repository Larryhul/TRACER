"""Calibrate memory on a single shared, validation-only factual rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from tracer.belief.filter import apply_belief_forgetting, categorical_bayes_update, uniform_log_belief
from tracer.experiments.closed_loop_protocol import prepare_manifest
from tracer.experiments.eval_component_factorial import FACTORIAL_POLICIES, _local_sources
from tracer.experiments.eval_persistent_closed_loop import _evaluate_policy, _load_model
from tracer.experiments.finetune_h3_response_ranking import _adapter_from_config, _make_spec
from tracer.experiments.formal_manifest import sha256_file
from tracer.response.closed_loop import PlanningCostConfig


def component_protocol_id(raw, prefix_objective):
    simulation = vars(_make_spec(raw, "id")).copy()
    simulation.pop("seed")
    simulation.pop("split")
    candidate = dict(raw["candidate_source"])
    for name in ("checkpoint_scope", "cache_candidate_pools", "shared_candidate_pools"):
        candidate.pop(name, None)
    protocol = {
        "simulation": simulation, "planning": vars(PlanningCostConfig(**raw["planning"])),
        "candidate_source": candidate, "prefix_objective": bool(prefix_objective),
    }
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()


def memory_decision_beliefs(likelihoods: torch.Tensor, scale: float, forgetting_rate: float = 0.0) -> dict[str, torch.Tensor]:
    if likelihoods.ndim != 3 or likelihoods.shape[-1] < 2 or not torch.isfinite(likelihoods).all():
        raise ValueError("finite (episodes, windows, modes) likelihoods required")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("likelihood scale must be positive")
    uniform = torch.full_like(likelihoods, -np.log(likelihoods.shape[-1]))
    centered = likelihoods - likelihoods.amax(dim=-1, keepdim=True)
    current = uniform_log_belief(likelihoods.shape[-1], likelihoods.shape[0], device=likelihoods.device, dtype=likelihoods.dtype)
    before = []
    for window in range(likelihoods.shape[1]):
        before.append(current)
        prior = apply_belief_forgetting(current, forgetting_rate) if forgetting_rate else current
        current = categorical_bayes_update(prior, centered[:, window] * scale)
    persistent = torch.stack(before, dim=1)
    reset = uniform.clone()
    reset[:, 1:] = torch.log_softmax(centered[:, :-1] * scale, dim=-1)
    return {"uniform": uniform, "reset": reset, "persistent": persistent}


def score_scales(likelihoods: torch.Tensor, scales: list[float], coordinates: int, forgetting_rates=None) -> list[dict]:
    if likelihoods.shape[1] < 2 or coordinates < 1 or not scales:
        raise ValueError("at least two windows, one scale, and positive coordinates required")
    records = []
    for scale in scales:
        for forgetting_rate in ([0.0] if forgetting_rates is None else forgetting_rates):
            beliefs = memory_decision_beliefs(likelihoods, scale, forgetting_rate)
            scores = {
                name: float((-torch.logsumexp(prior + likelihoods, dim=-1)[:, 1:] / coordinates).mean())
                for name, prior in beliefs.items()
            }
            records.append({
                "likelihood_scale": float(scale), "persistent_forgetting_rate": float(forgetting_rate),
                "selection_score": 0.5 * (scores["reset"] + scores["persistent"]),
                "decision_response_nll": scores,
                "memory_nll_difference": scores["persistent"] - scores["reset"],
            })
    return records


def load_component_calibration(path, checkpoint, spec, evaluation_data_seed, prefix_objective=False, emission_checkpoint=None, protocol_id=None, joint_weight=1.0, continuation_value_checkpoint=None, continuation_value_weight=0., emission_noise_checkpoint=None):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema") != "component-memory-factual-v1" or record.get("status") != "COMPLETED":
        raise ValueError("completed component memory calibration required")
    if record.get("phase") != "validation":
        raise ValueError("memory calibration must use validation data")
    if record["checkpoint_sha256"] != sha256_file(Path(checkpoint)):
        raise ValueError("memory calibration checkpoint mismatch")
    expected_emission = sha256_file(Path(emission_checkpoint)) if emission_checkpoint else None
    if record.get("emission_checkpoint_sha256") != expected_emission:
        raise ValueError("memory calibration emission checkpoint mismatch")
    expected_noise = sha256_file(Path(emission_noise_checkpoint)) if emission_noise_checkpoint else None
    if record.get("emission_noise_sha256") != expected_noise:
        raise ValueError("memory calibration correlated emission mismatch")
    if record["data_seed"] == evaluation_data_seed:
        raise ValueError("calibration and evaluation scenes must be separate")
    if record.get("prefix_objective", False) != prefix_objective:
        raise ValueError("calibration decision objective mismatch")
    if record.get("joint_weight", 1.0) != joint_weight:
        raise ValueError("calibration residual weight mismatch")
    expected_value = sha256_file(Path(continuation_value_checkpoint)) if continuation_value_checkpoint else None
    if record.get("continuation_value_sha256") != expected_value or record.get("continuation_value_weight", 0.) != continuation_value_weight:
        raise ValueError("calibration continuation value mismatch")
    if protocol_id is not None:
        recorded_protocol = record.get("component_protocol_id")
        if recorded_protocol is None:
            original = json.loads((Path(path).parent / "run_manifest.json").read_text(encoding="utf-8"))["config"]
            recorded_protocol = component_protocol_id(original["config"], original["arguments"].get("prefix_objective", False))
        if recorded_protocol != protocol_id:
            raise ValueError("calibration candidate/simulator/planning protocol mismatch")
    for name in ("dt", "execute_steps", "num_robots", "horizon"):
        if record[name] != getattr(spec, name):
            raise ValueError(f"memory calibration {name} mismatch")
    scale = float(record["selected"]["likelihood_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid calibrated likelihood scale")
    return scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--response-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-seed", type=int, default=924609)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix-objective", action="store_true")
    parser.add_argument("--forgetting-rates", type=float, nargs="+", default=[0.0])
    parser.add_argument("--reuse-factual-dir")
    parser.add_argument("--emission-checkpoint")
    parser.add_argument("--emission-noise-checkpoint")
    parser.add_argument("--residual-weight-calibration")
    parser.add_argument("--continuation-value-checkpoint")
    parser.add_argument("--continuation-value-weight", type=float, default=0.)
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw["simulation"]["data_seed"] = args.data_seed
    spec = _make_spec(raw, "id")
    checkpoint = Path(args.response_checkpoint).resolve()
    model = _load_model(checkpoint, args.device, expected_dt=spec.dt).eval()
    if model.training_dt != spec.dt:
        raise ValueError("component calibration must preserve the checkpoint training timestep")
    emission_checkpoint = Path(args.emission_checkpoint).resolve() if args.emission_checkpoint else None
    emission_model = _load_model(emission_checkpoint, args.device) if emission_checkpoint else None
    adapter = _adapter_from_config(raw, args.device)
    output = Path(args.output_dir)
    inputs = [checkpoint, Path(raw["candidate_source"]["checkpoint"]).resolve()]
    noise_checkpoint = Path(args.emission_noise_checkpoint).resolve() if args.emission_noise_checkpoint else None
    noise_model = None
    if noise_checkpoint:
        from tracer.experiments.component_emission_protocol import load_emission_noise

        noise_model = load_emission_noise(noise_checkpoint, emission_checkpoint or checkpoint, raw, args.data_seed)
        inputs.extend([noise_checkpoint, noise_checkpoint.parent / "run_manifest.json"])
    value_model = None
    value_checkpoint = Path(args.continuation_value_checkpoint).resolve() if args.continuation_value_checkpoint else None
    if value_checkpoint:
        from tracer.experiments.component_value_protocol import load_value_artifact

        value_model, _ = load_value_artifact(value_checkpoint, checkpoint, raw, args.data_seed)
        inputs.extend([value_checkpoint, value_checkpoint.parent / "summary.json", value_checkpoint.parent / "run_manifest.json"])
    elif args.continuation_value_weight != 0.:
        raise ValueError("nonzero continuation weight requires a value checkpoint")
    joint_weight = 1.0
    if args.residual_weight_calibration:
        if not args.prefix_objective:
            raise ValueError("residual reliability calibration requires prefix objective")
        from tracer.experiments.calibrate_component_residual_weight import load_residual_weight

        joint_weight = load_residual_weight(args.residual_weight_calibration, checkpoint, raw, args.data_seed)
        inputs.append(Path(args.residual_weight_calibration).resolve())
    if emission_checkpoint:
        inputs.append(emission_checkpoint)
    if args.reuse_factual_dir:
        previous_path = Path(args.reuse_factual_dir) / "calibration.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous["status"] != "COMPLETED" or previous["phase"] != "validation":
            raise ValueError("reused factual stream must be a completed validation calibration")
        if previous.get("emission_noise_sha256") != (sha256_file(noise_checkpoint) if noise_checkpoint else None):
            raise ValueError("reused factual stream correlated emission mismatch")
        for name, expected in {"checkpoint_sha256": sha256_file(checkpoint), "data_seed": args.data_seed, "episodes": args.episodes, "prefix_objective": args.prefix_objective, "dt": spec.dt, "execute_steps": spec.execute_steps, "num_robots": spec.num_robots, "horizon": spec.horizon}.items():
            if previous.get(name) != expected:
                raise ValueError(f"reused factual stream {name} mismatch")
        if previous.get("emission_checkpoint_sha256") != (sha256_file(emission_checkpoint) if emission_checkpoint else None):
            raise ValueError("reused factual stream emission checkpoint mismatch")
        if previous.get("joint_weight", 1.0) != joint_weight:
            raise ValueError("reused factual stream residual weight mismatch")
        if previous.get("continuation_value_sha256") != (sha256_file(value_checkpoint) if value_checkpoint else None) or previous.get("continuation_value_weight", 0.) != args.continuation_value_weight:
            raise ValueError("reused factual stream continuation value mismatch")
        previous_manifest = Path(args.reuse_factual_dir) / "run_manifest.json"
        original = json.loads(previous_manifest.read_text(encoding="utf-8"))["config"]
        if component_protocol_id(original["config"], original["arguments"].get("prefix_objective", False)) != component_protocol_id(raw, args.prefix_objective):
            raise ValueError("reused factual stream candidate/simulator/planning protocol mismatch")
        inputs += [previous_path, previous_manifest, Path(args.reuse_factual_dir) / "factual_rollout.json"]
    manifest = prepare_manifest(output, {
        "config": raw, "arguments": vars(args),
        "selection": "mean of Persistent and Reset pre-update factual prediction NLL",
        "policy": "fixed full-Joint Uniform controller, no mode labels used for selection",
    }, inputs, _local_sources(), False)
    if args.reuse_factual_dir:
        result = json.loads((Path(args.reuse_factual_dir) / "factual_rollout.json").read_text(encoding="utf-8"))
    else:
        result = _evaluate_policy(
            model, FACTORIAL_POLICIES[0], spec, args.episodes, args.device,
            PlanningCostConfig(**raw["planning"]), likelihood_scale=0.05,
            candidate_adapter=adapter, response_gate_min=joint_weight, response_gate_max=joint_weight,
            prefix_objective=args.prefix_objective,
            emission_model=emission_model,
            retain_previous_plan=bool(raw.get("candidate_source", {}).get("retain_previous_plan", False)),
            continuation_value_model=value_model,
            continuation_value_weight=args.continuation_value_weight,
            emission_noise_model=noise_model,
        )
    likelihoods = torch.tensor([
        [row["executed_mode_log_likelihood"] for row in rows]
        for rows in result["window_records"]
    ], dtype=torch.float64)
    scores = score_scales(likelihoods, args.scales, 2 * spec.execute_steps, args.forgetting_rates)
    selected = min(scores, key=lambda row: row["selection_score"])
    changed_sources = [path for path, digest in manifest["sources"].items() if sha256_file(Path(path)) != digest]
    changed_inputs = [path for path, digest in manifest["checkpoints"].items() if sha256_file(Path(path)) != digest]
    calibration = {
        "schema": "component-memory-factual-v1", "phase": "validation",
        "status": "COMPLETED" if not changed_sources and not changed_inputs else "INVALIDATED",
        "checkpoint_sha256": sha256_file(checkpoint), "data_seed": args.data_seed,
        "emission_checkpoint_sha256": sha256_file(emission_checkpoint) if emission_checkpoint else None,
        "emission_noise_sha256": sha256_file(noise_checkpoint) if noise_checkpoint else None,
        "component_protocol_id": component_protocol_id(raw, args.prefix_objective),
        "episodes": args.episodes, "dt": spec.dt, "execute_steps": spec.execute_steps,
        "num_robots": spec.num_robots, "horizon": spec.horizon,
        "prefix_objective": args.prefix_objective,
        "joint_weight": joint_weight,
        "continuation_value_sha256": sha256_file(value_checkpoint) if value_checkpoint else None,
        "continuation_value_weight": args.continuation_value_weight,
        "selected": selected, "scores": scores,
        "changed_sources": changed_sources, "changed_inputs": changed_inputs,
    }
    (output / "factual_rollout.json").write_text(json.dumps(result, allow_nan=True))
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2, allow_nan=False) + "\n")
    print(json.dumps(calibration, indent=2), flush=True)
    if calibration["status"] != "COMPLETED":
        raise RuntimeError("memory calibration inputs changed")


if __name__ == "__main__":
    main()
