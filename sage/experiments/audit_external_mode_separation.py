from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sage.experiments.audit_external_candidate_belief import finite_json
from sage.experiments.train_rolling_external_belief import matched_mode_probe_groups, predict_all_modes, sha256
from sage.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig


def mode_effect_metrics(predicted, target, log_std):
    if predicted.shape != target.shape or log_std.shape != target.shape or target.ndim != 4:
        raise ValueError("mode effect audit requires matching (groups, modes, horizon, coordinates) tensors")
    predicted_effect = predicted - predicted.mean(1, keepdim=True)
    target_effect = target - target.mean(1, keepdim=True)
    informative = target.std(1, unbiased=False).norm(dim=-1).mean(-1) > .02
    result = {}
    for name, mask in (("all", torch.ones_like(informative)), ("informative", informative), ("invariant", ~informative)):
        selected = int(mask.sum())
        record = {"groups": selected}
        if selected:
            residual = target[mask, :, None] - predicted[mask, None]
            uncertainty = log_std[mask, None]
            likelihood = -(.5 * residual.square() * (-2 * uncertainty).exp() + uncertainty).sum((-1, -2))
            labels = torch.arange(target.shape[1])[None].expand(selected, -1)
            record.update({
                "true_spread": float(target[mask].std(1, unbiased=False).mean()),
                "predicted_spread": float(predicted[mask].std(1, unbiased=False).mean()),
                "effect_rmse": float((predicted_effect[mask] - target_effect[mask]).square().mean().sqrt()),
                "zero_effect_rmse": float(target_effect[mask].square().mean().sqrt()),
                "conditional_rmse": float((predicted[mask] - target[mask]).square().mean().sqrt()),
                "mode_recovery_accuracy": float((likelihood.argmax(-1) == labels).float().mean()),
            })
        result[name] = record
    return result


@torch.no_grad()
def audit(checkpoint_path, probe_dir, device, batch_size):
    checkpoint_path, probe_dir = Path(checkpoint_path), Path(probe_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    manifest = json.loads((probe_dir / "manifest.json").read_text())
    if manifest["phase"] != "validation":
        raise ValueError("mode separation must be audited on held-out interventions")
    with np.load(probe_dir / "val.npz") as loaded:
        data = dict(loaded)
    train_cases = set(checkpoint["provenance"]["case_splits"]["train"])
    val_cases = set(checkpoint["provenance"]["case_splits"]["val"])
    if set(data["case"]) & train_cases or not set(data["case"]) <= val_cases:
        raise ValueError("mode audit requires checkpoint validation cases, never training cases")
    config = PersistentTrajectoryModelConfig(**checkpoint["model_config"])
    if not np.allclose(config.mode_speed_values, manifest["mode_speed_values"], atol=1e-6, rtol=0):
        raise ValueError("mode audit coordinates must match checkpoint conditioning")
    model = BeliefConditionedTeamResponseModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    groups = matched_mode_probe_groups(data, config.num_modes)
    rows = groups[:, 0]
    chunks = [predict_all_modes(model, data, rows[start:start + batch_size], device)[0]
              for start in range(0, len(rows), batch_size)]
    mean = torch.cat([chunk["mean"].cpu() for chunk in chunks])
    log_std = torch.cat([chunk["log_std"].cpu() for chunk in chunks])
    target = torch.as_tensor(data["target"][groups])
    return {"checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": sha256(checkpoint_path),
            "probe_data_sha256": sha256(probe_dir / "val.npz"), "phase": "validation",
            "case_count": len(set(data["case"])), "factual": mode_effect_metrics(mean[:, :, :1], target[:, :, :1], log_std[:, :, :1]),
            "post_delay": mode_effect_metrics(mean[:, :, 1:], target[:, :, 1:], log_std[:, :, 1:]),
            "boundary": "held-out same-state speed interventions, not a factual history or closed-loop result"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or args.batch_size < 1:
        raise ValueError("mode audit requires a fresh output and positive batch size")
    torch.set_num_threads(2)
    results = [audit(path, args.probe_dir, args.device, args.batch_size) for path in args.checkpoints]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(finite_json(results), indent=2, allow_nan=False) + "\n")
    for result in results:
        print(json.dumps(finite_json(result)), flush=True)


if __name__ == "__main__":
    main()
