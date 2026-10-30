"""Evaluate frozen H1 response models across held-out response families."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tracer.experiments.eval_h1_paired_pilot import (
    _average_seeds,
    _evaluate_run,
    _load_config,
    _summarize_cell,
)


MODES = ("none", "additive", "nonlinear")
SEEDS = (2027, 2028, 2029, 2030, 2031)


def run(
    root: Path,
    checkpoint_root: Path,
    output_dir: Path,
    config_path: Path = Path("configs/h1_paired_formal.yaml"),
    modes: tuple[str, ...] = MODES,
    seeds: tuple[int, ...] = SEEDS,
    data_seed: int = 328731,
    device: str = "cuda",
    bootstrap_resamples: int = 5000,
) -> dict[str, Any]:
    raw = _load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for cell_index, (train_mode, eval_mode) in enumerate(
        (pair for pair in ((train, eval_) for train in modes for eval_ in modes))
    ):
            records: list[dict[str, Any]] = []
            for model_seed in seeds:
                run_dir = root / checkpoint_root / train_mode / f"seed{model_seed}"
                if not run_dir.exists():
                    raise FileNotFoundError(run_dir)
                print(
                    f"lofo train_mode={train_mode} eval_mode={eval_mode} seed={model_seed}",
                    flush=True,
                )
                seed_records, seed_diagnostics = _evaluate_run(
                    raw,
                    run_dir,
                    eval_mode,
                    model_seed,
                    data_seed,
                    device,
                    "test",
                    f"lofo_train_{train_mode}",
                    True,
                    False,
                )
                records.extend(seed_records)
                diagnostics[f"{train_mode}->{eval_mode}/seed{model_seed}"] = seed_diagnostics
            averaged = _average_seeds(records)
            summary = _summarize_cell(
                averaged,
                data_seed + cell_index * 104729,
                bootstrap_resamples,
            )
            for contrast, values in summary.items():
                cells.append(
                    {
                        "train_mode": train_mode,
                        "eval_mode": eval_mode,
                        "contrast": contrast,
                        "mean": values["mean"],
                        "sd": values["sd"],
                        "ci95_low": values["mean_ci95"][0],
                        "ci95_high": values["mean_ci95"][1],
                        "dz": values["dz"],
                        "n_base_scenes": values["n_base_scenes"],
                        "model_seed_count": len(seeds),
                    }
                )

    payload = {
        "status": "SUPPLEMENTARY_LOFO",
        "protocol": {
            "config": str(config_path),
            "checkpoint_root": str(checkpoint_root),
            "train_modes": list(modes),
            "eval_modes": list(modes),
            "model_seeds": list(seeds),
            "test_data_seed": data_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "interpretation": "response-family transfer diagnostic; separate from Formal-v1 gates",
        },
        "cells": cells,
        "diagnostic_keys": sorted(diagnostics),
    }
    (output_dir / "lofo_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "lofo_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(cells[0]) if cells else []
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cells)
    lines = [
        "# H1 Response-Family Leave-One-Family-Out Diagnostic",
        "",
        "Each row trains the response model on one frozen synthetic response family and evaluates it on another family. This is supplementary transfer evidence, not a replacement for the Formal-v1 paired-intervention gate.",
        "",
        "| Train family | Test family | Contrast | Mean | 95% CI | dz |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in cells:
        lines.append(
            f"| {row['train_mode']} | {row['eval_mode']} | {row['contrast']} | "
            f"{row['mean']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | {row['dz']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Positive matched-additive-minus-joint NLL favors the joint model; the diagonal cells are within-family reference results and off-diagonal cells are the leave-one-family-out diagnostic.",
            "The response families are controlled simulator mechanisms, not real human behavior categories.",
        ]
    )
    (output_dir / "lofo_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_summary={output_dir / 'lofo_summary.json'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--checkpoint-root", default="experiments/pilot/h1_paired_rescue_stage1_v2/checkpoints")
    parser.add_argument("--output-dir", default="reports/response_family_lofo_2026-08-21")
    parser.add_argument("--config", default="configs/h1_paired_formal.yaml")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--data-seed", type=int, default=328731)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(
        Path(args.root).resolve(),
        Path(args.checkpoint_root),
        Path(args.output_dir),
        Path(args.config),
        tuple(args.modes),
        tuple(args.seeds),
        args.data_seed,
        args.device,
        args.bootstrap_resamples,
    )


if __name__ == "__main__":
    main()
