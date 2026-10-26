"""Evaluate frozen H2 checkpoints with calibrated likelihood scales at a long prefix."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import torch
import yaml

from sage.experiments.eval_persistent_trajectory_checkpoints import _load_checkpoint_models
from sage.experiments.train_persistent_trajectory_response import _evaluate, _make_dataset, _set_seed


DEFAULT_SCALES = (0.0125, 0.025, 0.05, 0.1, 0.2)
DEFAULT_SEEDS = (2027, 2028, 2029, 2030, 2031)
CONTRASTS = (
    ("reset", "reset_minus_persistent_nll"),
    ("shuffled-id", "shuffled_id_minus_persistent_nll"),
    ("recurrent", "recurrent_minus_persistent_nll"),
)


def _record(
    metrics: dict[str, dict[str, Any]],
    split: str,
    scale: float,
    seed: int,
    prefix: int,
) -> list[dict[str, Any]]:
    persistent = metrics["persistent"]
    rows = []
    for baseline, contrast in CONTRASTS:
        rows.append(
            {
                "prefix": prefix,
                "split": split,
                "scale": scale,
                "seed": seed,
                "contrast": contrast,
                "persistent_nll": persistent["predictive_trajectory_nll"],
                "nll_mean": metrics[baseline]["predictive_trajectory_nll"]
                - persistent["predictive_trajectory_nll"],
                "ade_mean": metrics[baseline]["trajectory_ade"] - persistent["trajectory_ade"],
            }
        )
    return rows


def run(
    root: Path,
    config_path: Path,
    checkpoint_root: Path,
    output_dir: Path,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    prefix: int = 6,
    device: str = "cuda",
    batch_size: int = 8,
) -> dict[str, Any]:
    if prefix < 1:
        raise ValueError("prefix must be positive")
    if not scales or any(scale <= 0.0 for scale in scales):
        raise ValueError("scales must be positive")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset_cfg = raw["dataset"]
    evaluation_cfg = raw.get("evaluation", {})
    raw["dataset"]["execute_steps"] = prefix
    rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for seed in seeds:
        _set_seed(seed)
        seed_dir = root / checkpoint_root / f"seed{seed}"
        response_model, recurrent_model, metadata = _load_checkpoint_models(seed_dir, device)
        data_seed = metadata["data_seed"]
        id_dataset = _make_dataset(raw, metadata["id_split"], data_seed)
        ood_dataset = _make_dataset(
            raw,
            "ood",
            data_seed,
            action_policy=str(dataset_cfg.get("ood_action_policy", "correlated")),
        )
        loader_kwargs = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}
        loaders = {
            "id": torch.utils.data.DataLoader(id_dataset, **loader_kwargs),
            "ood": torch.utils.data.DataLoader(ood_dataset, **loader_kwargs),
        }
        seed_scale_rows: list[dict[str, Any]] = []
        for scale in scales:
            for split, loader in loaders.items():
                metrics, _ = _evaluate(
                    response_model,
                    recurrent_model,
                    loader,
                    device=device,
                    dt=float(dataset_cfg.get("dt", 0.4)),
                    likelihood_scale=scale,
                    confidence_threshold=float(evaluation_cfg.get("confidence_threshold", 0.8)),
                    execute_steps=prefix,
                    split_name=split,
                )
                scale_rows = _record(metrics, split, scale, seed, prefix)
                rows.extend(scale_rows)
                seed_scale_rows.extend(scale_rows)
        selected_scale = min(
            scales,
            key=lambda scale: next(
                row["persistent_nll"]
                for row in seed_scale_rows
                if row["scale"] == scale
                and row["split"] == "id"
                and row["contrast"] == "reset_minus_persistent_nll"
            ),
        )
        selected.extend(
            row for row in seed_scale_rows if row["scale"] == selected_scale
        )
        print(f"h2_prefix_scale_completed seed={seed} selected_scale={selected_scale:g}", flush=True)
        del response_model, recurrent_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary: list[dict[str, Any]] = []
    for scale in scales:
        for split in ("id", "ood"):
            for _, contrast in CONTRASTS:
                values = [
                    float(row["nll_mean"])
                    for row in rows
                    if row["scale"] == scale
                    and row["split"] == split
                    and row["contrast"] == contrast
                ]
                summary.append(
                    {
                        "prefix": prefix,
                        "scale": scale,
                        "split": split,
                        "contrast": contrast,
                        "nll_mean": statistics.fmean(values),
                        "nll_sd_across_seeds": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "n_seeds": len(values),
                    }
                )

    payload = {
        "status": "SUPPLEMENTARY_LONG_PREFIX_SCALE_SENSITIVITY",
        "protocol": {
            "prefix": prefix,
            "scales": list(scales),
            "model_seeds": list(seeds),
            "selection_rule": "minimum ID reset-minus-persistent predictive NLL per model seed",
            "interpretation": "post hoc stability diagnostic; not a Formal-v1 gate",
        },
        "summary": summary,
        "selected_per_seed": selected,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# H2 Long-Prefix Likelihood-Scale Sensitivity",
        "",
        f"Frozen H2 checkpoints are evaluated at executed prefix {prefix} across the disclosed scale grid. Positive NLL contrasts favor Persistent.",
        "",
        "| Scale | Split | Contrast | Mean | SD across seeds | n |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scale']:.4f} | {row['split']} | {row['contrast']} | "
            f"{row['nll_mean']:.6f} | {row['nll_sd_across_seeds']:.6f} | {row['n_seeds']} |"
        )
    lines.extend(
        [
            "",
            "The per-seed scale is selected using ID predictive NLL only; this diagnostic does not convert the long-prefix result into a confirmatory claim.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_summary={output_dir / 'summary.json'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/persistent_trajectory_formal.yaml")
    parser.add_argument("--checkpoint-root", default="experiments/formal_v1/h2/persistent_trajectory_suite_fast")
    parser.add_argument("--output-dir", default="reports/h2_prefix_scale_sensitivity_2026-08-23")
    parser.add_argument("--scales", nargs="+", type=float, default=list(DEFAULT_SCALES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--prefix", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
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
    )


if __name__ == "__main__":
    main()
