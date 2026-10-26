from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sage.experiments.train_rolling_external_response import batch_data
from sage.response.external_checkpoint_adapter import ExternalResponseCheckpointAdapter, delayed_response_prediction


def paired_response_effects(data, predicted):
    groups = {}
    for row, identity in enumerate(zip(data["case"], data["human"], data["window"])):
        if not data.get("branch_projected", np.zeros(len(predicted), dtype=bool))[row]:
            groups.setdefault(tuple(identity), []).append(row)
    observed, inferred = [], []
    for rows in groups.values():
        for offset, first in enumerate(rows):
            for second in rows[offset + 1:]:
                observed.append(data["target"][first] - data["target"][second])
                inferred.append(predicted[first] - predicted[second])
    if not observed:
        return {"pairs": 0}
    observed, inferred = np.asarray(observed), np.asarray(inferred)
    observed_flat, inferred_flat = observed[:, 1:].ravel(), inferred[:, 1:].ravel()
    correlation = float(np.corrcoef(observed_flat, inferred_flat)[0, 1]) if observed_flat.std() > 1e-12 and inferred_flat.std() > 1e-12 else None
    return {"pairs": len(observed), "true_effect_mse": float(np.mean(observed ** 2)),
            "predicted_effect_mse": float(np.mean(inferred ** 2)),
            "effect_error_mse": float(np.mean((inferred - observed) ** 2)),
            "post_delay_effect_correlation": correlation,
            "true_first_step_effect_max": float(np.abs(observed[:, 0]).max()),
            "predicted_first_step_effect_max": float(np.abs(inferred[:, 0]).max()),
            "boundary": "paired observed response differences among unprojected cloned branches; not a navigation causal gain"}


@torch.no_grad()
def evaluate(checkpoint, data, batch_size=128):
    adapter = ExternalResponseCheckpointAdapter.from_checkpoint(checkpoint, "cpu", expected_dt=.25)
    predictions, log_stds = [], []
    for start in range(0, len(data["target"]), batch_size):
        batch = batch_data(data, np.arange(start, min(start + batch_size, len(data["target"]))), "cpu")
        output = delayed_response_prediction(adapter.predictor, batch, adapter.response_delay_steps)
        predictions.append(output["mean"].cpu().numpy())
        log_stds.append(output["log_std"].cpu().numpy())
    mean, log_std = np.concatenate(predictions), np.concatenate(log_stds)
    residual = mean - data["target"]
    likelihood = .5 * residual ** 2 * np.exp(-2 * log_std) + log_std + .5 * np.log(2 * np.pi)
    errors = np.linalg.norm(np.cumsum(residual * .25, axis=1), axis=-1)
    subsets = {"all": np.ones(len(mean), dtype=bool), "early_windows": data["window"] < 24,
               "late_windows": data["window"] >= 24, "selected_branch": data.get("selected_branch", np.ones(len(mean), dtype=bool))}
    metrics = {}
    for name, selected in subsets.items():
        if selected.any():
            metrics[name] = {"rows": int(selected.sum()), "nll": float(likelihood[selected].mean()),
                             "ade": float(errors[selected].mean()), "prefix_mse": float((residual[selected, 0] ** 2).mean()),
                             "prefix_nll": float(likelihood[selected, 0].mean()),
                             "mean_velocity_std": float(np.exp(log_std[selected]).mean())}
    return {"checkpoint": str(checkpoint), "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "metrics": metrics, "response_effects": paired_response_effects(data, mean)}


def main():
    parser = argparse.ArgumentParser(description="Compare response error and paired action effects at fixed live-controller states.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = Path(args.output)
    if output.exists():
        raise ValueError("response audit results already exist")
    with np.load(args.data) as loaded:
        data = dict(loaded)
    result = {"phase": "development_diagnostic", "data": args.data,
              "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
              "models": [evaluate(path, data) for path in args.checkpoints]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
