"""Summarize probability diagnostics from the frozen H2 evaluation outputs.

The archived Formal-v1 metrics contain proper scoring rules and posterior
entropy, but not per-observation confidence vectors.  This script therefore
does not manufacture ECE or interval-coverage estimates from aggregate data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("no-belief", "reset", "persistent", "recurrent", "shuffled-id", "oracle")
SCALAR_METRICS = (
    "mode_nll",
    "mode_brier",
    "final_mode_nll",
    "final_mode_brier",
    "final_mode_accuracy",
    "posterior_entropy",
    "final_posterior_entropy",
    "windows_to_confidence",
)
WINDOW_METRICS = ("entropy_by_window", "true_probability_by_window")


def _seed_dirs(root: Path) -> list[Path]:
    paths = sorted(
        (path.parent for path in root.glob("seed*/metrics.json")),
        key=lambda path: int(path.name.removeprefix("seed")),
    )
    if not paths:
        raise FileNotFoundError(f"no seed*/metrics.json files under {root}")
    return paths


def _mean_std(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": statistics.fmean(finite),
        "std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "n": len(finite),
    }


def summarize(root: Path) -> dict[str, Any]:
    scalar_values: dict[tuple[str, str, str], list[float]] = {}
    window_values: dict[tuple[str, str, str, int], list[float]] = {}
    seed_names: list[str] = []
    for seed_dir in _seed_dirs(root):
        seed_names.append(seed_dir.name)
        for split, filename in (("id", "metrics.json"), ("ood", "ood_metrics.json")):
            payload = json.loads((seed_dir / filename).read_text(encoding="utf-8"))
            for method, metrics in payload.items():
                for metric in SCALAR_METRICS:
                    if metric in metrics:
                        scalar_values.setdefault((split, method, metric), []).append(
                            float(metrics[metric])
                        )
                for metric in WINDOW_METRICS:
                    values = metrics.get(metric, [])
                    for window, value in enumerate(values):
                        window_values.setdefault(
                            (split, method, metric, window), []
                        ).append(float(value))

    scalar_summary = [
        {
            "split": split,
            "method": method,
            "metric": metric,
            **_mean_std(values),
            "seed_values": values,
        }
        for (split, method, metric), values in sorted(scalar_values.items())
    ]
    window_summary = [
        {
            "split": split,
            "method": method,
            "metric": metric,
            "window": window,
            **_mean_std(values),
            "seed_values": values,
        }
        for (split, method, metric, window), values in sorted(window_values.items())
    ]
    return {
        "source": str(root),
        "seed_dirs": seed_names,
        "num_seeds": len(seed_names),
        "scalar_metrics": scalar_summary,
        "window_metrics": window_summary,
        "limitations": [
            "The frozen raw schema does not contain per-observation maximum predicted confidence.",
            "Standard confidence-binned ECE cannot be reconstructed from aggregate metrics.",
            "Predictive interval coverage requires archived predictive scale and target-level residuals.",
        ],
    }


def _lookup(summary: dict[str, Any], split: str, method: str, metric: str) -> dict[str, Any]:
    for row in summary["scalar_metrics"]:
        if row["split"] == split and row["method"] == method and row["metric"] == metric:
            return row
    raise KeyError((split, method, metric))


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("split", "method", "metric", "mean", "std", "n")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["scalar_metrics"]:
            writer.writerow({field: row[field] for field in fields})


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# H2 Formal-v1 Probabilistic Diagnostics",
        "",
        f"Source: `{summary['source']}`; independent seeds: {summary['num_seeds']}.",
        "The table reports seed-level mean +/- standard deviation. Lower is better for NLL/Brier/entropy; higher is better for accuracy.",
        "",
        "| Split | Method | Final NLL | Final Brier | Final accuracy | Final entropy | Windows to confidence |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    methods = [method for method in METHODS if any(row["method"] == method for row in summary["scalar_metrics"])]
    for split in ("id", "ood"):
        for method in methods:
            values = {
                metric: _lookup(summary, split, method, metric)
                for metric in (
                    "final_mode_nll",
                    "final_mode_brier",
                    "final_mode_accuracy",
                    "final_posterior_entropy",
                    "windows_to_confidence",
                )
            }
            lines.append(
                "| {split} | {method} | {nll:.4f} +/- {nll_std:.4f} | {brier:.4f} +/- {brier_std:.4f} | "
                "{accuracy:.4f} +/- {accuracy_std:.4f} | {entropy:.4f} +/- {entropy_std:.4f} | "
                "{windows:.3f} +/- {windows_std:.3f} |".format(
                    split=split,
                    method=method,
                    nll=values["final_mode_nll"]["mean"],
                    nll_std=values["final_mode_nll"]["std"],
                    brier=values["final_mode_brier"]["mean"],
                    brier_std=values["final_mode_brier"]["std"],
                    accuracy=values["final_mode_accuracy"]["mean"],
                    accuracy_std=values["final_mode_accuracy"]["std"],
                    entropy=values["final_posterior_entropy"]["mean"],
                    entropy_std=values["final_posterior_entropy"]["std"],
                    windows=values["windows_to_confidence"]["mean"],
                    windows_std=values["windows_to_confidence"]["std"],
                )
            )
    lines.extend(
        [
            "",
            "## Reporting Boundary",
            "",
            "The archived Formal-v1 CSV stores true-class probabilities, aggregate Brier/NLL, and entropy, but not the full predicted probability vector for each observation. Consequently, standard max-confidence ECE and reliability diagrams cannot be reconstructed from this snapshot. Predictive interval coverage is also not available because predictive scales and target-level residuals were not archived.",
            "",
            "These missing diagnostics are implementation follow-ups, not silently substituted by the table above.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _window_rows(summary: dict[str, Any], split: str, method: str, metric: str) -> list[dict[str, Any]]:
    return [
        row
        for row in summary["window_metrics"]
        if row["split"] == split and row["method"] == method and row["metric"] == metric
    ]


def plot(summary: dict[str, Any], output: Path) -> None:
    methods = [method for method in METHODS if any(row["method"] == method for row in summary["scalar_metrics"])]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), dpi=220)
    colors = {"id": "#276FBF", "ood": "#D1495B"}
    x = list(range(len(methods)))
    width = 0.36
    for offset, split in ((-width / 2, "id"), (width / 2, "ood")):
        brier = [_lookup(summary, split, method, "final_mode_brier") for method in methods]
        axes[0].bar(
            [value + offset for value in x],
            [row["mean"] for row in brier],
            width=width,
            yerr=[row["std"] for row in brier],
            capsize=2,
            color=colors[split],
            label=split.upper(),
        )
        nll = [_lookup(summary, split, method, "final_mode_nll") for method in methods]
        axes[1].bar(
            [value + offset for value in x],
            [row["mean"] for row in nll],
            width=width,
            yerr=[row["std"] for row in nll],
            capsize=2,
            color=colors[split],
        )
    axes[0].set_title("Final mode Brier")
    axes[1].set_title("Final mode NLL")
    axes[0].set_ylabel("Mean +/- seed SD")
    for ax in axes[:2]:
        ax.set_xticks(x, methods, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    for split in ("id", "ood"):
        for method, color in (("reset", "#4C956C"), ("persistent", "#F4A259"), ("recurrent", "#7B2CBF")):
            rows = _window_rows(summary, split, method, "entropy_by_window")
            if not rows:
                continue
            rows.sort(key=lambda row: row["window"])
            windows = [row["window"] for row in rows]
            means = [row["mean"] for row in rows]
            stds = [row["std"] for row in rows]
            linestyle = "-" if split == "id" else "--"
            axes[2].plot(windows, means, color=color, linestyle=linestyle, label=f"{method} {split.upper()}")
            axes[2].fill_between(
                windows,
                [mean - std for mean, std in zip(means, stds)],
                [mean + std for mean, std in zip(means, stds)],
                color=color,
                alpha=0.08,
            )
    axes[2].set_title("Posterior entropy by window")
    axes[2].set_xlabel("Window")
    axes[2].set_ylabel("Entropy")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=6, ncol=2, frameon=False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize frozen H2 probabilistic diagnostics.")
    parser.add_argument("--root", required=True, help="Formal H2 suite root containing seed*/metrics.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary = summarize(Path(args.root))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "summary.csv", summary)
    _write_markdown(output_dir / "summary.md", summary)
    plot(summary, output_dir / "h2_probabilistic_diagnostics.png")
    print(f"saved_summary={output_dir / 'summary.json'}")
    print(f"saved_table={output_dir / 'summary.csv'}")
    print(f"saved_report={output_dir / 'summary.md'}")
    print(f"saved_figure={output_dir / 'h2_probabilistic_diagnostics.png'}")


if __name__ == "__main__":
    main()
