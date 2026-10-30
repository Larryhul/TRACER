"""Compute ECE and predictive interval coverage from calibration-ready H2 rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("reset", "persistent", "recurrent", "shuffled-id", "no-belief", "oracle")
NUM_BINS = 10


def _read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def _ece(rows: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    bins: list[list[dict[str, Any]]] = [[] for _ in range(NUM_BINS)]
    for row in rows:
        confidence = min(max(float(row["posterior_confidence"]), 0.0), 1.0)
        bins[min(int(confidence * NUM_BINS), NUM_BINS - 1)].append(row)
    total = len(rows)
    if total == 0:
        return float("nan"), []
    records = []
    ece = 0.0
    for index, bucket in enumerate(bins):
        if not bucket:
            continue
        confidence = statistics.fmean(float(row["posterior_confidence"]) for row in bucket)
        accuracy = statistics.fmean(float(row["posterior_mode_correct"]) for row in bucket)
        gap = abs(confidence - accuracy)
        ece += len(bucket) / total * gap
        records.append(
            {
                "bin": index,
                "lower": index / NUM_BINS,
                "upper": (index + 1) / NUM_BINS,
                "count": len(bucket),
                "mean_confidence": confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )
    return ece, records


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("active") != "1" or row.get("main_evaluation") != "1":
            continue
        grouped[(row["_seed"], row["split"], row["method"])].append(row)
    seed_records = []
    for (seed, split, method), seed_rows in sorted(grouped.items()):
        ece, bins = _ece(seed_rows)
        seed_records.append(
            {
                "seed": int(seed),
                "split": split,
                "method": method,
                "ece": ece,
                "num_rows": len(seed_rows),
                "coverage_1sigma": statistics.fmean(
                    float(row["predictive_1sigma_coordinate_coverage"]) for row in seed_rows
                ),
                "coverage_2sigma": statistics.fmean(
                    float(row["predictive_2sigma_coordinate_coverage"]) for row in seed_rows
                ),
                "bins": bins,
            }
        )
    aggregate: list[dict[str, Any]] = []
    for split in ("id", "ood"):
        for method in METHODS:
            values = [
                row
                for row in seed_records
                if row["split"] == split and row["method"] == method
            ]
            if not values:
                continue
            aggregate.append(
                {
                    "split": split,
                    "method": method,
                    "num_seeds": len(values),
                    "ece_mean": statistics.fmean(row["ece"] for row in values),
                    "ece_std": statistics.stdev(row["ece"] for row in values) if len(values) > 1 else 0.0,
                    "coverage_1sigma_mean": statistics.fmean(row["coverage_1sigma"] for row in values),
                    "coverage_1sigma_std": statistics.stdev(row["coverage_1sigma"] for row in values) if len(values) > 1 else 0.0,
                    "coverage_2sigma_mean": statistics.fmean(row["coverage_2sigma"] for row in values),
                    "coverage_2sigma_std": statistics.stdev(row["coverage_2sigma"] for row in values) if len(values) > 1 else 0.0,
                }
            )
    return {"seed_records": seed_records, "aggregate": aggregate}


def _write_csv(path: Path, aggregate: list[dict[str, Any]]) -> None:
    fields = (
        "split",
        "method",
        "num_seeds",
        "ece_mean",
        "ece_std",
        "coverage_1sigma_mean",
        "coverage_1sigma_std",
        "coverage_2sigma_mean",
        "coverage_2sigma_std",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# H2 Calibration Diagnostics",
        "",
        "Rows are restricted to active post-initial windows. ECE uses ten equal-width bins over posterior maximum confidence. Coverage is coordinate-level empirical coverage of the mixture predictive mean +/- one or two mixture standard deviations.",
        "",
        "| Split | Method | ECE | 1-sigma coverage | 2-sigma coverage | Seeds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['split']} | {row['method']} | {row['ece_mean']:.4f} +/- {row['ece_std']:.4f} | "
            f"{row['coverage_1sigma_mean']:.4f} +/- {row['coverage_1sigma_std']:.4f} | "
            f"{row['coverage_2sigma_mean']:.4f} +/- {row['coverage_2sigma_std']:.4f} | {row['num_seeds']} |"
        )
    lines.extend(
        [
            "",
            "Reference values for a calibrated scalar Gaussian are approximately 0.6827 and 0.9545, but the reported coverage is an empirical diagnostic of the model's mixture-scale output rather than a formal guarantee.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path: Path, payload: dict[str, Any]) -> None:
    methods = [method for method in METHODS if any(row["method"] == method for row in payload["aggregate"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=220)
    x = list(range(len(methods)))
    width = 0.36
    colors = {"id": "#276FBF", "ood": "#D1495B"}
    for offset, split in ((-width / 2, "id"), (width / 2, "ood")):
        rows = {row["method"]: row for row in payload["aggregate"] if row["split"] == split}
        for axis, metric, title in (
            (axes[0], "ece_mean", "ECE (lower is better)"),
            (axes[1], "coverage_2sigma_mean", "2-sigma coverage"),
        ):
            values = [rows[method][metric] for method in methods]
            error_metric = "ece_std" if metric == "ece_mean" else "coverage_2sigma_std"
            axis.bar(
                [value + offset for value in x],
                values,
                width=width,
                yerr=[rows[method][error_metric] for method in methods],
                capsize=2,
                color=colors[split],
                label=split.upper(),
            )
    axes[0].set_ylabel("Mean +/- seed SD")
    axes[1].axhline(0.9545, color="#555555", linestyle=":", linewidth=1, label="Gaussian reference")
    for axis in axes:
        axis.set_xticks(x, methods, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze calibration-ready H2 raw rows.")
    parser.add_argument("--raw", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = [Path(path) for path in args.raw]
    rows = []
    for path in paths:
        seed = path.parent.name.removeprefix("seed")
        seed_rows = _read_rows([path])
        for row in seed_rows:
            row["_seed"] = seed
        rows.extend(seed_rows)
    payload = _summarize(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "calibration.csv", payload["aggregate"])
    _write_markdown(output_dir / "calibration.md", payload)
    _plot(output_dir / "h2_calibration.png", payload)
    print(f"saved_calibration={output_dir / 'calibration.json'}")
    print(f"saved_table={output_dir / 'calibration.csv'}")
    print(f"saved_report={output_dir / 'calibration.md'}")
    print(f"saved_figure={output_dir / 'h2_calibration.png'}")


if __name__ == "__main__":
    main()
