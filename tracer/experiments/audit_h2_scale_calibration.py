"""Validation-only temperature audit for the H2 persistent belief filter.

The existing formal runner is intentionally left unchanged.  This script
loads frozen checkpoints, evaluates a scale grid on validation episodes, and
then applies separately selected scales to test/OOD.  It reports posterior
calibration and decision-time calibration from raw rows rather than selecting
only by predictive trajectory NLL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from tracer.experiments.eval_persistent_trajectory_checkpoints import _load_checkpoint_models
from tracer.experiments.train_persistent_trajectory_response import (
    _evaluate,
    _make_dataset,
    _set_seed,
)


def ece_from_rows(rows: list[dict[str, Any]], prefix: str = "posterior") -> float:
    selected = [
        row
        for row in rows
        if int(row["active"]) and int(row["window"]) > 0
    ]
    if not selected:
        raise ValueError("no post-initial active rows")
    confidence = [float(row[f"{prefix}_confidence"]) for row in selected]
    correct = [float(row[f"{prefix}_mode_correct"]) for row in selected]
    error = 0.0
    total = len(selected)
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        bucket = [
            position
            for position, value in enumerate(confidence)
            if (lower <= value < upper)
            or (index == 9 and lower <= value <= upper)
        ]
        if bucket:
            error += len(bucket) / total * abs(
                statistics.fmean(confidence[position] for position in bucket)
                - statistics.fmean(correct[position] for position in bucket)
            )
    return float(error)


def _brier_from_rows(rows: list[dict[str, Any]], prefix: str = "posterior") -> float:
    selected = [
        row
        for row in rows
        if int(row["active"]) and int(row["window"]) > 0
    ]
    if not selected:
        raise ValueError("no post-initial active rows")
    field = f"{prefix}_brier"
    if any(field not in row for row in selected):
        raise ValueError("full multiclass Brier requires re-evaluated probability-vector scores")
    return float(statistics.fmean(float(row[field]) for row in selected))


def _record(
    metrics: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    split: str,
    seed: int,
    scale: float,
) -> dict[str, Any]:
    persistent_rows = [row for row in rows if row["method"] == "persistent"]
    persistent = metrics["persistent"]
    return {
        "split": split,
        "seed": seed,
        "scale": scale,
        "posterior_ece": ece_from_rows(persistent_rows, "posterior"),
        "decision_ece": ece_from_rows(persistent_rows, "decision"),
        "posterior_brier": _brier_from_rows(persistent_rows, "posterior"),
        "decision_brier": _brier_from_rows(persistent_rows, "decision"),
        "posterior_mode_nll": float(persistent["mode_nll"]),
        "decision_mode_nll": float(persistent["decision_mode_nll"]),
        "predictive_trajectory_nll": float(
            persistent["predictive_trajectory_nll"]
        ),
        "trajectory_ade": float(persistent["trajectory_ade"]),
    }


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _select(rows: list[dict[str, Any]], metric: str) -> float:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        grouped.setdefault(float(row["scale"]), []).append(float(row[metric]))
    return min(grouped, key=lambda scale: statistics.fmean(grouped[scale]))


def run(
    root: Path,
    config_path: Path,
    checkpoint_root: Path,
    output_dir: Path,
    scales: tuple[float, ...],
    seeds: tuple[int, ...],
    prefix: int,
    device: str,
    batch_size: int,
    limit_episodes: int,
) -> dict[str, Any]:
    if not scales or any(scale <= 0.0 or not math.isfinite(scale) for scale in scales):
        raise ValueError("scales must be finite and positive")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw["dataset"]["execute_steps"] = prefix
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        _set_seed(seed)
        seed_dir = root / checkpoint_root / f"seed{seed}"
        response_model, recurrent_model, metadata = _load_checkpoint_models(seed_dir, device)
        data_seed = int(metadata["data_seed"])
        split_configs = {
            "val": _make_dataset(raw, "val", data_seed),
            str(metadata["id_split"]): _make_dataset(raw, str(metadata["id_split"]), data_seed),
            "ood": _make_dataset(
                raw,
                "ood",
                data_seed,
                action_policy=str(raw["dataset"].get("ood_action_policy", "correlated")),
            ),
        }
        if limit_episodes > 0:
            split_configs = {
                split: _slice_dataset(dataset, limit_episodes)
                for split, dataset in split_configs.items()
            }
        for scale in scales:
            for split, dataset in split_configs.items():
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                metrics, raw_rows = _evaluate(
                    response_model,
                    recurrent_model,
                    loader,
                    device=device,
                    dt=float(raw["dataset"].get("dt", 0.4)),
                    likelihood_scale=scale,
                    confidence_threshold=float(
                        raw.get("evaluation", {}).get("confidence_threshold", 0.8)
                    ),
                    execute_steps=prefix,
                    split_name=split,
                )
                rows.append(_record(metrics, raw_rows, split, seed, scale))
        del response_model, recurrent_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    validation_rows = [row for row in rows if row["split"] == "val"]
    selected = {
        "posterior_ece": _select(validation_rows, "posterior_ece"),
        "posterior_brier": _select(validation_rows, "posterior_brier"),
        "posterior_mode_nll": _select(validation_rows, "posterior_mode_nll"),
        "predictive_trajectory_nll": _select(
            validation_rows, "predictive_trajectory_nll"
        ),
    }
    summary: list[dict[str, Any]] = []
    for split in sorted({str(row["split"]) for row in rows}):
        for scale in scales:
            values = [
                row
                for row in rows
                if row["split"] == split and float(row["scale"]) == scale
            ]
            summary.append(
                {
                    "split": split,
                    "scale": scale,
                    "seed_count": len(values),
                    **{
                        metric: _mean_sd([float(row[metric]) for row in values])
                        for metric in (
                            "posterior_ece",
                            "decision_ece",
                            "posterior_brier",
                            "decision_brier",
                            "posterior_mode_nll",
                            "decision_mode_nll",
                            "predictive_trajectory_nll",
                            "trajectory_ade",
                        )
                    },
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "SUPPLEMENTARY_VALIDATION_ONLY_SCALE_AUDIT",
        "protocol": {
            "config": str(config_path),
            "checkpoint_root": str(checkpoint_root),
            "prefix": prefix,
            "scales": list(scales),
            "model_seeds": list(seeds),
            "selection_split": "val",
            "selection_metrics": list(selected),
            "limit_episodes": limit_episodes,
        },
        "selected_scales": selected,
        "summary": summary,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# H2 Validation-Only Likelihood Scale Calibration",
        "",
        f"Frozen checkpoints, executed prefix {prefix}, selection on validation only. Lower is better for every metric.",
        "",
        "## Selected scales",
        "",
    ]
    for metric, scale in selected.items():
        lines.append(f"- `{metric}`: `{scale:g}`")
    lines.extend(
        [
            "",
            "## Aggregate validation/test/OOD metrics",
            "",
            "| Split | Scale | Posterior ECE | Posterior Brier | Posterior NLL | Predictive NLL | ADE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['split']} | {row['scale']:.4f} | "
            f"{row['posterior_ece']['mean']:.6f} | "
            f"{row['posterior_brier']['mean']:.6f} | "
            f"{row['posterior_mode_nll']['mean']:.6f} | "
            f"{row['predictive_trajectory_nll']['mean']:.6f} | "
            f"{row['trajectory_ade']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "This is a supplementary calibration audit, not a replacement for the frozen formal H2 result.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved_summary={output_dir / 'summary.json'}", flush=True)
    return payload


def _slice_dataset(dataset: Any, limit: int) -> Any:
    if len(dataset) <= limit:
        return dataset
    from torch.utils.data import Subset

    return Subset(dataset, range(limit))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/persistent_trajectory_formal.yaml")
    parser.add_argument(
        "--checkpoint-root",
        default="experiments/formal_v2/clean_chain_submission_20260904/h2",
    )
    parser.add_argument("--output-dir", default="reports/h2_scale_calibration_20260906")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.0125, 0.025, 0.05, 0.1, 0.2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--prefix", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-episodes", type=int, default=0)
    args = parser.parse_args()
    run(
        Path(args.root).resolve(),
        Path(args.config),
        Path(args.checkpoint_root),
        Path(args.output_dir),
        tuple(args.scales),
        tuple(args.seeds),
        args.prefix,
        args.device,
        args.batch_size,
        args.limit_episodes,
    )


if __name__ == "__main__":
    main()
