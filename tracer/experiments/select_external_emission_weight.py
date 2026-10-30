"""Select an external emission weight from validation-only results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


SEEDS = (2027, 2031, 2039)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_sd(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _load_weight(
    weight: float,
    root: Path,
    *,
    baseline: bool,
) -> dict[str, Any]:
    rows = []
    metadata = []
    for seed in SEEDS:
        directory = root / f"seed{seed}" if baseline else root / f"weight{weight}" / f"seed{seed}"
        rows.append(_read(directory / "val_metrics.json"))
        metadata.append(_read(directory / "run_metadata.json"))
        if not baseline and (directory / "test_metrics.json").exists():
            raise ValueError(f"validation sweep unexpectedly contains test metrics: {directory}")
    if any(record["git"]["dirty"] for record in metadata):
        raise ValueError(f"weight {weight} contains a dirty run")
    clustering_seeds = {record["mode_clustering_seed"] for record in metadata}
    input_hashes = {json.dumps(record["input_sha256"], sort_keys=True) for record in metadata}
    mode_counts = {tuple(record["mode_counts"]["val"]) for record in metadata}
    if len(clustering_seeds) != 1 or len(input_hashes) != 1 or len(mode_counts) != 1:
        raise ValueError(f"weight {weight} has inconsistent data provenance")

    def values(method: str, metric: str) -> list[float]:
        return [float(row[method][metric]) for row in rows]

    persistent_nll = values("persistent", "predictive_trajectory_nll")
    reset_nll = values("reset", "predictive_trajectory_nll")
    shuffled_nll = values("shuffled-id", "predictive_trajectory_nll")
    persistent_accuracy = values("persistent", "final_mode_accuracy")
    shuffled_accuracy = values("shuffled-id", "final_mode_accuracy")
    reset_gain = [reset - persistent for reset, persistent in zip(reset_nll, persistent_nll)]
    shuffled_gain = [
        shuffled - persistent
        for shuffled, persistent in zip(shuffled_nll, persistent_nll)
    ]
    identity_accuracy_gain = [
        persistent - shuffled
        for persistent, shuffled in zip(persistent_accuracy, shuffled_accuracy)
    ]
    return {
        "weight": weight,
        "seeds": list(SEEDS),
        "commit": metadata[0]["git"]["commit"],
        "validation_only": not baseline,
        "persistent_nll": _mean_sd(persistent_nll),
        "persistent_accuracy": _mean_sd(persistent_accuracy),
        "reset_minus_persistent_nll": _mean_sd(reset_gain),
        "shuffled_minus_persistent_nll": _mean_sd(shuffled_gain),
        "persistent_minus_shuffled_accuracy": _mean_sd(identity_accuracy_gain),
    }


def select(baseline_root: Path, sweep_root: Path) -> dict[str, Any]:
    records = [_load_weight(0.1, baseline_root, baseline=True)]
    for directory in sorted(sweep_root.glob("weight*")):
        weight = float(directory.name.removeprefix("weight"))
        records.append(_load_weight(weight, sweep_root, baseline=False))
    records.sort(key=lambda record: record["weight"])
    eligible = [
        record
        for record in records
        if record["reset_minus_persistent_nll"]["mean"] > 0.0
        and record["shuffled_minus_persistent_nll"]["mean"] > 0.0
    ]
    if not eligible:
        raise ValueError("no weight passes both predictive-NLL controls")
    selected = min(eligible, key=lambda record: record["persistent_nll"]["mean"])
    return {
        "dataset": "thor_magni",
        "protocol": "retrospective-validation-only-emission-weight-audit-v1",
        "selection_rule": [
            "Require mean reset-minus-persistent predictive NLL > 0.",
            "Require mean shuffled-ID-minus-persistent predictive NLL > 0.",
            "Among eligible weights, minimize absolute persistent predictive NLL.",
        ],
        "records": records,
        "eligible_weights": [record["weight"] for record in eligible],
        "selected_weight": selected["weight"],
        "decision": (
            "Retain the existing weight 0.1. Larger weights improve pseudo-mode "
            "classification but worsen the primary response predictive NLL. No new "
            "test evaluation is authorized by this audit."
        ),
        "claim_boundary": (
            "This retrospective audit follows validation metrics only but occurred after "
            "the baseline test run; it does not convert the external benchmark into new "
            "confirmatory evidence."
        ),
    }


def _report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# THOR-MAGNI Emission-Weight Validation Audit",
        "",
        "## Frozen Rule",
        "",
        *[f"1. {rule}" for rule in summary["selection_rule"]],
        "",
        "The primary quantity follows the paper's H2 claim: response predictive likelihood, "
        "not standalone pseudo-mode classification accuracy.",
        "",
        "## Validation Results",
        "",
        "| Weight | Persistent NLL | Persistent acc. (%) | Reset-Persistent NLL | "
        "Shuffled-Persistent NLL | Persistent-Shuffled acc. (pp) | Eligible |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    eligible = set(summary["eligible_weights"])
    for record in summary["records"]:
        lines.append(
            f"| {record['weight']:.1f} | {record['persistent_nll']['mean']:.6f} | "
            f"{record['persistent_accuracy']['mean'] * 100:.3f} | "
            f"{record['reset_minus_persistent_nll']['mean']:.6f} | "
            f"{record['shuffled_minus_persistent_nll']['mean']:.6f} | "
            f"{record['persistent_minus_shuffled_accuracy']['mean'] * 100:.3f} | "
            f"{'yes' if record['weight'] in eligible else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Selected weight: **{summary['selected_weight']:.1f}**.",
            "",
            summary["decision"],
            "",
            summary["claim_boundary"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _figure(summary: dict[str, Any], path: Path) -> None:
    records = summary["records"]
    weights = [record["weight"] for record in records]
    nll = [record["persistent_nll"]["mean"] for record in records]
    accuracy = [record["persistent_accuracy"]["mean"] * 100.0 for record in records]
    identity = [
        record["persistent_minus_shuffled_accuracy"]["mean"] * 100.0
        for record in records
    ]
    fig, axis = plt.subplots(figsize=(6.8, 4.0))
    secondary = axis.twinx()
    axis.plot(weights, nll, "o-", color="#c96c4b", label="Persistent NLL")
    secondary.plot(
        weights,
        accuracy,
        "s-",
        color="#287271",
        label="Persistent accuracy",
    )
    secondary.plot(
        weights,
        identity,
        "^-",
        color="#5c4d7d",
        label="Identity accuracy gain",
    )
    axis.axvline(summary["selected_weight"], color="#333333", linestyle="--")
    axis.set_xscale("log")
    axis.set_xticks(weights, [str(weight) for weight in weights])
    axis.set_xlabel("Mode discrimination weight")
    axis.set_ylabel("Validation predictive NLL")
    secondary.set_ylabel("Accuracy / gain (percentage points)")
    handles_a, labels_a = axis.get_legend_handles_labels()
    handles_b, labels_b = secondary.get_legend_handles_labels()
    axis.legend(handles_a + handles_b, labels_a + labels_b, loc="center right")
    axis.set_title("THOR-MAGNI validation trade-off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        default="experiments/external_response_benchmark_formal_v2/thor_magni",
    )
    parser.add_argument(
        "--sweep-root",
        default="experiments/external_emission_sweep_val",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/external_response_benchmark_20260901",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = select(Path(args.baseline_root), Path(args.sweep_root))
    (output_dir / "emission_sweep.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _report(summary, output_dir / "emission_sweep.md")
    _figure(summary, output_dir / "emission_sweep.png")
    print(f"selected weight={summary['selected_weight']}")


if __name__ == "__main__":
    main()
