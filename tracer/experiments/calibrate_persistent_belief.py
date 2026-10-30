from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from tracer.belief.filter import run_categorical_filter
from tracer.experiments.formal_manifest import sha256_file
from tracer.experiments.train_persistent_trajectory_response import _flatten_entities, _make_dataset, _move
from tracer.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    executed_prefix_mode_log_likelihood,
)


DEFAULT_SCALES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_belief_calibration(
    calibration: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    execute_steps: int,
    dt: float,
    num_robots: int,
) -> dict[str, Any]:
    """Reject a calibration from a different checkpoint or execution protocol."""

    if not isinstance(calibration, dict):
        raise ValueError("belief calibration must be a JSON object")
    for field, expected in (
        ("phase", "validation"),
        ("split", "val"),
    ):
        if calibration.get(field) != expected:
            raise ValueError(f"calibration requires {field}={expected}")
    selection_metric = calibration.get("selection_metric")
    if selection_metric not in {"decision_mode_nll", "decision_response_nll"}:
        raise ValueError("calibration selection_metric must be decision_mode_nll or decision_response_nll")
    if selection_metric == "decision_response_nll" and calibration.get("protocol") != "external-factual-prefix-cadence-v1":
        raise ValueError("response-NLL calibration requires the external factual cadence protocol")
    expected_contract = {
        "execute_steps": _positive_integer(execute_steps, "execute_steps"),
        "dt": _positive_number(dt, "dt"),
        "num_robots": _positive_integer(num_robots, "num_robots"),
    }
    for field, expected in expected_contract.items():
        observed = (
            _positive_number(calibration.get(field), field)
            if field == "dt"
            else _positive_integer(calibration.get(field), field)
        )
        if observed != expected:
            raise ValueError(f"calibration {field} mismatch: {observed} != {expected}")
    if calibration.get("checkpoint_sha256") != sha256_file(Path(checkpoint_path)):
        raise ValueError("calibration checkpoint_sha256 mismatch")
    selected = calibration.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("calibration requires a selected record")
    _positive_number(selected.get("likelihood_scale"), "selected.likelihood_scale")
    score = selected.get("decision_mode_nll")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or score < 0.0
    ):
        raise ValueError("selected.decision_mode_nll must be finite and non-negative")
    if selection_metric == "decision_response_nll":
        response_score = selected.get(selection_metric)
        if isinstance(response_score, bool) or not isinstance(response_score, (int, float)) or not math.isfinite(response_score):
            raise ValueError("selected.decision_response_nll must be finite")
    return calibration


def load_belief_calibration(
    path: str | Path,
    checkpoint_path: str | Path,
    *,
    execute_steps: int,
    dt: float,
    num_robots: int,
) -> dict[str, Any]:
    """Load a validated artifact; consume selected.likelihood_scale on success."""

    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_belief_calibration(
        calibration,
        checkpoint_path,
        execute_steps=execute_steps,
        dt=dt,
        num_robots=num_robots,
    )


def calibration_scores(likelihoods, modes, active, scales):
    if not scales or any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("calibration scales must be finite and positive")
    if not torch.isfinite(likelihoods).all():
        raise ValueError("validation likelihoods must be finite")
    if likelihoods.ndim != 3 or likelihoods.shape[-1] < 2:
        raise ValueError("likelihoods must have shape (tracks, windows, modes)")
    if modes.shape != likelihoods.shape[:1] or active.shape != likelihoods.shape[:2]:
        raise ValueError("modes and active mask must align with likelihoods")
    active = active.bool()
    mask = active.bool().clone()
    mask[:, 0] = False
    if not mask.any():
        raise ValueError("calibration requires active decisions after window zero")
    labels = modes[:, None].expand_as(mask)
    targets = torch.nn.functional.one_hot(labels, likelihoods.shape[-1])
    records = []
    for scale in scales:
        before, after = run_categorical_filter(likelihoods * scale, observation_mask=active)
        probability = before.exp()
        records.append({
            "likelihood_scale": float(scale),
            "decision_mode_nll": float(-before.gather(-1, labels[..., None]).squeeze(-1)[mask].mean()),
            "decision_mode_brier": float((probability - targets).square().sum(-1)[mask].mean()),
            "decision_mode_accuracy": float((before.argmax(-1) == labels)[mask].float().mean()),
            "decision_confidence": float(probability.max(-1).values[mask].mean()),
            "posterior_mode_nll": float(-after.gather(-1, labels[..., None]).squeeze(-1)[active].mean()),
            "decisions": int(mask.sum()),
        })
    return records


def _calibration_prediction(model, batch):
    evidence = {}
    if model.config.evidence_conditioning:
        for name in ("evidence_candidates", "evidence_responses", "evidence_mask"):
            value = batch.get(name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"evidence_conditioning=True requires {name} in calibration data; "
                    "refusing calibration without evidence"
                )
            evidence[name] = value
    return model(
        batch["entity_history"],
        batch["robot_current"],
        batch["robot_candidates"],
        neighbor_history=batch.get("neighbor_history"),
        neighbor_mask=batch.get("neighbor_mask"),
        **evidence,
    )


@torch.no_grad()
def calibrate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    device: str = "cpu",
    scales: list[float] | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Collect validation prefixes once and select by pre-observation mode NLL."""

    config_path, checkpoint_path, output_path = map(
        Path, (config_path, checkpoint_path, output_path)
    )
    config_sha256 = sha256_file(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evaluation = raw.get("evaluation", {})
    if evaluation.get("phase") != "validation":
        raise ValueError("temperature selection requires evaluation.phase=validation")
    if evaluation.get("split", "val") != "val":
        raise ValueError("temperature selection requires evaluation.split=val")
    if evaluation.get("selection_metric", "decision_mode_nll") != "decision_mode_nll":
        raise ValueError("temperature selection requires selection_metric=decision_mode_nll")
    if output_path.exists():
        raise ValueError("calibration output exists; use a new path")
    if scales is None:
        scales = list(raw.get("calibration", {}).get("scales", DEFAULT_SCALES))
    if not scales:
        raise ValueError("calibration requires finite and positive scales")
    scales = [_positive_number(scale, "likelihood_scale") for scale in scales]
    training = raw.get("training", {})
    torch.set_num_threads(_positive_integer(training.get("num_threads", 4), "num_threads"))
    batch_size = _positive_integer(
        batch_size if batch_size is not None else training.get("batch_size", 16), "batch_size"
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = PersistentTrajectoryModelConfig(**checkpoint["model_config"])
    dataset_config = raw["dataset"]
    dt = _positive_number(dataset_config["dt"], "dt")
    robots = _positive_integer(dataset_config["num_robots"], "num_robots")
    steps = _positive_integer(dataset_config["execute_steps"], "execute_steps")
    episodes = _positive_integer(dataset_config["val_episodes"], "val_episodes")
    if model_config.dt != dt or model_config.num_robots != robots:
        raise ValueError("checkpoint and calibration data dt/num_robots must agree")
    for field in ("history", "horizon"):
        if dataset_config.get(field, getattr(model_config, field)) != getattr(model_config, field):
            raise ValueError(f"checkpoint and calibration data {field} must agree")
    if steps > model_config.horizon:
        raise ValueError("execute_steps must lie within the model horizon")
    model = BeliefConditionedTeamResponseModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = _make_dataset(raw, "val", int(dataset_config["data_seed"]))
    likelihoods, modes, masks = [], [], []
    for raw_batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        batch = _flatten_entities(_move(raw_batch, device))
        prediction = _calibration_prediction(model, batch)
        likelihoods.append(executed_prefix_mode_log_likelihood(
            prediction["mean"], prediction["log_std"], batch["target_future_vel"], steps,
        ).cpu())
        modes.append(batch["mode"].cpu())
        masks.append(batch["active_mask"].cpu())
    likelihoods, modes, masks = torch.cat(likelihoods), torch.cat(modes), torch.cat(masks)
    records = calibration_scores(likelihoods, modes, masks, scales)
    selected = min(records, key=lambda record: (record["decision_mode_nll"], record["likelihood_scale"]))
    result = {
        "schema_version": 1,
        "phase": "validation",
        "split": "val",
        "selection_metric": "decision_mode_nll",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "model_seed": checkpoint.get("seed"),
        "config": raw,
        "config_sha256": config_sha256,
        "execute_steps": steps,
        "dt": dt,
        "num_robots": robots,
        "num_modes": model_config.num_modes,
        "evidence_conditioning": model_config.evidence_conditioning,
        "data_seed": int(dataset_config["data_seed"]),
        "episodes": episodes,
        "tracks": int(likelihoods.shape[0]),
        "active_observations": int(masks.sum()),
        "decision_mask": "active and window > 0",
        "likelihood_reduction": "sum over executed prefix time and xy coordinates",
        "scales": scales,
        "records": records,
        "selected": selected,
    }
    if sha256_file(config_path) != config_sha256:
        raise ValueError("calibration config changed during evaluation")
    validate_belief_calibration(
        result, checkpoint_path, execute_steps=steps, dt=dt, num_robots=robots
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Calibrate H2 decision belief on validation prefixes only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scales", nargs="+", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    result = calibrate_checkpoint(
        args.config, args.checkpoint, args.output,
        device=args.device, scales=args.scales, batch_size=args.batch_size,
    )
    print(json.dumps(result["selected"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
