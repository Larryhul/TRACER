"""Audit recorded-action dependence without retraining or belief updates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sage.data.external_response import ExternalResponseEpisodeDataset, stable_entity_id
from sage.experiments.run_external_response_benchmark import _sha256, _validate_protocol_arrays
from sage.experiments.train_persistent_trajectory_response import (
    _flatten_entities,
    _move,
    constant_velocity_target,
)
from sage.response.metrics import trajectory_errors
from sage.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    stationary_reference_candidates,
    trajectory_mode_log_likelihood,
)


def prediction_metrics(mean: torch.Tensor, target: torch.Tensor, dt: float) -> dict[str, np.ndarray]:
    ade, fde = trajectory_errors(mean.reshape(-1, mean.shape[-2], 2), target.reshape(-1, target.shape[-2], 2), dt)
    return {
        "ade": ade.reshape(target.shape[:2]).cpu().numpy(),
        "fde": fde.reshape(target.shape[:2]).cpu().numpy(),
    }


def summarize_recordings(values: np.ndarray, mask: np.ndarray, recordings: np.ndarray) -> dict:
    if values.shape != mask.shape or recordings.shape != values.shape[:1]:
        raise ValueError("values, mask and recording IDs must align")
    per_recording = {}
    for recording in np.unique(recordings):
        selected = mask & (recordings == recording)[:, None]
        if selected.any():
            per_recording[str(recording)] = float(values[selected].mean())
    return {
        "window_weighted_mean": float(values[mask].mean()) if mask.any() else None,
        "recording_macro_mean": float(np.mean(list(per_recording.values()))) if per_recording else None,
        "count": int(mask.sum()),
        "per_recording": per_recording,
    }


def load_model(path: Path, device: str) -> BeliefConditionedTeamResponseModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def audit(
    data_path: Path,
    run_dir: Path,
    device: str = "cpu",
    batch_size: int = 64,
    velocity_residual: bool = False,
) -> dict:
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    split_manifest = json.loads(data_path.with_suffix(".json").read_text())
    split = split_manifest["split"]
    data_hash = _sha256(data_path)
    if metadata["input_sha256"][split] != data_hash:
        raise ValueError("data hash does not match checkpoint run metadata")
    model = load_model(run_dir / "response.pt", device)
    history_model = load_model(run_dir / "history_only_response.pt", device)
    if model.config != history_model.config:
        raise ValueError("response/history-only model protocols do not match")
    if bool(metadata.get("velocity_residual", False)) != velocity_residual:
        raise ValueError("audit velocity-residual flag does not match run metadata")
    with np.load(data_path) as arrays:
        _validate_protocol_arrays(dict(arrays), data_path, history=model.config.history, horizon=model.config.horizon, windows=metadata["windows"])
        modes = np.zeros(arrays["entity_id"].shape, dtype=np.int64)
    dataset = ExternalResponseEpisodeDataset(data_path, modes)
    entity_to_recording = {
        stable_entity_id(recording, track): recording
        for recording in split_manifest["recording_ids"]
        for track in range(1, 64)
    }
    metrics = {}
    all_recordings, all_active, all_near, all_moving = [], [], [], []
    sensitivities = []
    for raw_batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch = _flatten_entities(_move(raw_batch, device))
        target = batch["target_future_vel"]
        target_offset = torch.zeros_like(target)
        if velocity_residual:
            target_offset = constant_velocity_target(
                batch["entity_history"], model.config.horizon, model.config.dt
            )
        model_target = target - target_offset
        reference = stationary_reference_candidates(batch["robot_current"], model.config.horizon)
        predictions = {}
        for name, predictor, candidates in (
            ("recorded_action", model, batch["robot_candidates"]),
            ("same_weights_stationary_action", model, reference),
            ("trained_history_only", history_model, reference),
        ):
            output = predictor(
                batch["entity_history"], batch["robot_current"], candidates,
                neighbor_history=batch.get("neighbor_history"), neighbor_mask=batch.get("neighbor_mask"),
            )
            mean = output["mean"] + target_offset.unsqueeze(-3)
            mean = mean.mean(dim=2)
            predictions[name] = mean
            measured = prediction_metrics(mean, target, model.config.dt)
            likelihood = trajectory_mode_log_likelihood(output["mean"], output["log_std"], model_target)
            measured["nll"] = (-(torch.logsumexp(likelihood, dim=-1) - math.log(model.config.num_modes)) / (model.config.horizon * 2)).cpu().numpy()
            for metric, value in measured.items():
                metrics.setdefault(name, {}).setdefault(metric, []).append(value)
        velocity = (batch["entity_history"][..., -1, :] - batch["entity_history"][..., -2, :]) / model.config.dt
        for name, mean in (
            ("constant_velocity", velocity[..., None, :].expand_as(target)),
            ("stationary_pedestrian", torch.zeros_like(target)),
        ):
            for metric, value in prediction_metrics(mean, target, model.config.dt).items():
                metrics.setdefault(name, {}).setdefault(metric, []).append(value)
        sensitivities.append(prediction_metrics(predictions["recorded_action"], predictions["same_weights_stationary_action"], model.config.dt)["ade"])
        active = batch["active_mask"].cpu().numpy().copy()
        active[:, 0] = False
        all_active.append(active)
        all_near.append(batch["high_interaction"].cpu().numpy())
        displacement = batch["robot_candidates"][..., -1, :2] - batch["robot_current"][..., :2]
        all_moving.append((torch.linalg.vector_norm(displacement, dim=-1).amax(dim=-1) > .05).cpu().numpy())
        recordings = []
        for entity, present in zip(batch["entity_id"].cpu().tolist(), batch["active_mask"].any(dim=-1).cpu().tolist()):
            if present and entity not in entity_to_recording:
                raise ValueError(f"unmapped active entity ID {entity}")
            recordings.append(entity_to_recording.get(entity, "inactive"))
        all_recordings.extend(recordings)
    recordings = np.asarray(all_recordings)
    active = np.concatenate(all_active)
    masks = {
        "all": active,
        "near_robot": active & np.concatenate(all_near),
        "moving_robot": active & np.concatenate(all_moving),
    }
    result = {
        "protocol": "frozen-weights-uniform-mode-action-ablation-v1",
        "split": split,
        "seed": metadata["model_seed"],
        "data_sha256": data_hash,
        "checkpoint_sha256": {name: _sha256(run_dir / name) for name in ("response.pt", "history_only_response.pt")},
        "dt": model.config.dt,
        "velocity_residual": velocity_residual,
        "horizon_seconds": model.config.dt * model.config.horizon,
        "claim_boundary": "observational input-dependence audit; stationary-action replacement can be out of distribution; no causal effect identification",
        "aggregation": "windows 1 onwards; overlapping windows are correlated; macro means weight each recording equally; no significance claim",
        "moving_robot_definition": "recorded robot endpoint displacement > 0.05 m; descriptive future-based stratum, not a deployable selector",
        "strata": {},
    }
    concatenated = {name: {metric: np.concatenate(values) for metric, values in group.items()} for name, group in metrics.items()}
    sensitivity = np.concatenate(sensitivities)
    for stratum, mask in masks.items():
        result["strata"][stratum] = {
            name: {metric: summarize_recordings(values, mask, recordings) for metric, values in group.items()}
            for name, group in concatenated.items()
        }
        result["strata"][stratum]["same_weights_prediction_change_m"] = summarize_recordings(sensitivity, mask, recordings)
    old = json.loads((run_dir / f"{split}_metrics.json").read_text())
    for name, original in (("recorded_action", "no-belief"), ("trained_history_only", "history-only")):
        for metric, old_key in (("nll", "predictive_trajectory_nll"), ("ade", "trajectory_ade"), ("fde", "trajectory_fde")):
            current = result["strata"]["all"][name][metric]["window_weighted_mean"]
            if not np.isclose(current, old[original][old_key], rtol=1e-4, atol=1e-5):
                raise ValueError(f"replay differs from saved metrics: {name}/{metric}: {current} vs {old[original][old_key]}")
    result["saved_metric_replay_verified"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--velocity-residual", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    results = [
        audit(args.data, directory, args.device, velocity_residual=args.velocity_residual)
        for directory in args.run_dir
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(results)} frozen-model audits to {args.output}")


if __name__ == "__main__":
    main()
