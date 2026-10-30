"""Aggregate multi-seed external response/belief benchmark results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats


DATASETS = ("thor_magni", "huron_sacson")
METRICS = (
    "predictive_trajectory_nll",
    "trajectory_ade",
    "trajectory_fde",
    "final_mode_accuracy",
    "final_mode_nll",
)
COMPARISONS = {
    "action_vs_history_nll": (
        "predictive_trajectory_nll",
        "history-only",
        "no-belief",
    ),
    "action_vs_history_ade": ("trajectory_ade", "history-only", "no-belief"),
    "persistent_vs_reset_nll": (
        "predictive_trajectory_nll",
        "reset",
        "persistent",
    ),
    "persistent_vs_reset_accuracy": (
        "final_mode_accuracy",
        "persistent",
        "reset",
    ),
    "persistent_vs_shuffled_accuracy": (
        "final_mode_accuracy",
        "persistent",
        "shuffled-id",
    ),
    "recurrent_vs_persistent_accuracy": (
        "final_mode_accuracy",
        "recurrent",
        "persistent",
    ),
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, Any]:
    count = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if count > 1 else 0.0
    sem = std / math.sqrt(count) if count > 1 else 0.0
    critical = float(scipy_stats.t.ppf(0.975, count - 1)) if count > 1 else 0.0
    return {
        "values": values,
        "count": count,
        "mean": mean,
        "std": std,
        "ci95_low": mean - critical * sem,
        "ci95_high": mean + critical * sem,
    }


def _paired(values: list[float]) -> dict[str, Any]:
    result = _stats(values)
    if len(values) > 1 and not np.allclose(values, values[0]):
        test = scipy_stats.ttest_1samp(values, popmean=0.0)
        p_value = float(test.pvalue)
    elif values and values[0] == 0.0:
        p_value = 1.0
    else:
        p_value = 0.0
    result.update(
        {
            "wins": sum(value > 0.0 for value in values),
            "ties": sum(value == 0.0 for value in values),
            "losses": sum(value < 0.0 for value in values),
            "paired_t_p_value": p_value,
        }
    )
    return result


def _validate_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    reference = records[0]
    fields = {
        "commit": [record["git"]["commit"] for record in records],
        "dirty": [record["git"]["dirty"] for record in records],
        "mode_clustering_seed": [record["mode_clustering_seed"] for record in records],
        "input_sha256": [record["input_sha256"] for record in records],
        "test_mode_counts": [record["mode_counts"]["test"] for record in records],
    }
    checks = {
        "single_commit": len(set(fields["commit"])) == 1,
        "all_clean": not any(fields["dirty"]),
        "fixed_mode_clustering_seed": len(set(fields["mode_clustering_seed"])) == 1,
        "identical_inputs": all(
            value == fields["input_sha256"][0] for value in fields["input_sha256"]
        ),
        "identical_test_modes": all(
            value == fields["test_mode_counts"][0]
            for value in fields["test_mode_counts"]
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"provenance validation failed: {checks}")
    return {
        "checks": checks,
        "commit": fields["commit"][0],
        "mode_clustering_seed": fields["mode_clustering_seed"][0],
        "input_sha256": fields["input_sha256"][0],
        "test_mode_counts": fields["test_mode_counts"][0],
        "identity_shuffle_available": reference["identity_shuffle_available"],
        "episodes": {
            split: int(reference[f"{split}_episodes"])
            for split in ("train", "val", "test")
        },
    }


def aggregate(root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "protocol": "observational-single-robot-response-belief-v1",
        "result_root": str(root),
        "datasets": {},
    }
    for dataset in DATASETS:
        directories = sorted((root / dataset).glob("seed*"))
        if not directories:
            raise FileNotFoundError(f"no seed directories under {root / dataset}")
        metadata = [_read(directory / "run_metadata.json") for directory in directories]
        seeds = [int(record["model_seed"]) for record in metadata]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate model seeds for {dataset}")
        split_results = {
            split: [_read(directory / f"{split}_metrics.json") for directory in directories]
            for split in ("val", "test")
        }
        methods = sorted(set.intersection(*(set(result) for result in split_results["test"])))
        aggregates: dict[str, Any] = {}
        for split, results in split_results.items():
            aggregates[split] = {
                method: {
                    metric: _stats([float(result[method][metric]) for result in results])
                    for metric in METRICS
                }
                for method in methods
            }
        comparisons: dict[str, Any] = {}
        for name, (metric, positive_method, negative_method) in COMPARISONS.items():
            if positive_method not in methods or negative_method not in methods:
                continue
            values = [
                float(result[positive_method][metric])
                - float(result[negative_method][metric])
                for result in split_results["test"]
            ]
            comparisons[name] = {
                "metric": metric,
                "positive_means": positive_method,
                "negative_means": negative_method,
                **_paired(values),
            }
        summary["datasets"][dataset] = {
            "seeds": seeds,
            "provenance": _validate_metadata(metadata),
            "aggregates": aggregates,
            "comparisons": comparisons,
            "result_sha256": {
                directory.name: {
                    split: _sha256(directory / f"{split}_metrics.json")
                    for split in ("val", "test")
                }
                for directory in directories
            },
        }
    return summary


def _mean_sd(record: dict[str, Any], percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    return f"{record['mean'] * scale:.4f} +/- {record['std'] * scale:.4f}"


def _write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# External Observational Response/Belief Benchmark",
        "",
        "## Scope",
        "",
        "This benchmark is a single-robot observational degeneration of the current paper. "
        "It evaluates held-out-recording response prediction and recording-local identity "
        "continuity. It does not establish multi-robot counterfactual effects, causality, "
        "psychological traits, or long-term persistence.",
        "",
        "## Reproducibility Audit",
        "",
        "| Dataset | Seeds | Train/Val/Test episodes | Commit | Clean | "
        "Fixed labels | Identity shuffle |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for dataset, record in summary["datasets"].items():
        provenance = record["provenance"]
        episodes = provenance["episodes"]
        lines.append(
            f"| {dataset} | {len(record['seeds'])} | "
            f"{episodes['train']}/{episodes['val']}/{episodes['test']} | "
            f"`{provenance['commit'][:7]}` | yes | yes | "
            f"{'yes' if provenance['identity_shuffle_available'] else 'no'} |"
        )
    lines.extend(["", "## Test Results", ""])
    for dataset, record in summary["datasets"].items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                "Mean +/- sample SD across model seeds.",
                "",
                "| Method | Predictive NLL | ADE | FDE | Final mode accuracy (%) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        test = record["aggregates"]["test"]
        preferred = (
            "history-only",
            "no-belief",
            "reset",
            "persistent",
            "shuffled-id",
            "recurrent",
            "oracle",
        )
        for method in preferred:
            if method not in test:
                continue
            metrics = test[method]
            lines.append(
                f"| {method} | {_mean_sd(metrics['predictive_trajectory_nll'])} | "
                f"{_mean_sd(metrics['trajectory_ade'])} | "
                f"{_mean_sd(metrics['trajectory_fde'])} | "
                f"{_mean_sd(metrics['final_mode_accuracy'], percent=True)} |"
            )
        lines.extend(
            [
                "",
                "| Paired gain (positive favors first method) | Mean | 95% CI | "
                "Wins/Ties/Losses | p |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, comparison in record["comparisons"].items():
            lines.append(
                f"| {name} | {comparison['mean']:.6f} | "
                f"[{comparison['ci95_low']:.6f}, {comparison['ci95_high']:.6f}] | "
                f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']} | "
                f"{comparison['paired_t_p_value']:.4g} |"
            )
        lines.append("")
    thor = summary["datasets"]["thor_magni"]["comparisons"]
    huron = summary["datasets"]["huron_sacson"]["comparisons"]
    lines.extend(
        [
            "## Submission Decision",
            "",
            f"- THOR identity continuity: persistent beats reset in "
            f"{thor['persistent_vs_reset_accuracy']['wins']}/5 seeds and shuffled identity in "
            f"{thor['persistent_vs_shuffled_accuracy']['wins']}/5 seeds.",
            f"- Recurrent baseline: recurrent beats explicit persistent mode accuracy in "
            f"{thor['recurrent_vs_persistent_accuracy']['wins']}/5 THOR seeds; this is an open "
            "method gap, not a result to hide.",
            f"- Action conditioning: action beats history-only NLL in "
            f"{huron['action_vs_history_nll']['wins']}/5 HuRoN seeds but "
            f"{thor['action_vs_history_nll']['wins']}/5 THOR seeds. The external evidence is "
            "dataset-dependent and supports conditional association only.",
            "- HuRoN has 19 single-entity test episodes and no valid shuffled-identity control; "
            "treat it as a small transfer check, not primary evidence.",
            "- Confidence intervals and paired t-tests use five model seeds, not five independent "
            "datasets or participant cohorts; treat them as optimization-variability diagnostics.",
            "",
            "## Next Gate",
            "",
            "A validation-only emission-discrimination audit retained the original weight 0.1 "
            "because stronger classification weights worsened absolute predictive NLL. No test "
            "rerun was authorized. The remaining method gap is the stronger learned recurrent "
            "mode accuracy. This external benchmark cannot replace the controlled multi-robot "
            "closed-loop experiments required for the core joint-action claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_figure(summary: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    thor = summary["datasets"]["thor_magni"]
    methods = ["reset", "persistent", "shuffled-id", "recurrent"]
    values = [thor["aggregates"]["test"][method]["final_mode_accuracy"] for method in methods]
    means = [record["mean"] * 100.0 for record in values]
    errors = [
        (record["ci95_high"] - record["mean"]) * 100.0 for record in values
    ]
    axes[0].bar(
        methods,
        means,
        yerr=errors,
        capsize=4,
        color=["#7d8590", "#287271", "#c96c4b", "#5c4d7d"],
    )
    axes[0].set_ylabel("Final mode accuracy (%)")
    axes[0].set_title("THOR-MAGNI identity continuity")
    axes[0].tick_params(axis="x", rotation=18)
    for dataset, color in (("thor_magni", "#287271"), ("huron_sacson", "#c96c4b")):
        comparison = summary["datasets"][dataset]["comparisons"]["action_vs_history_nll"]
        x = 0 if dataset == "thor_magni" else 1
        jitter = np.linspace(-0.08, 0.08, comparison["count"])
        axes[1].scatter(x + jitter, comparison["values"], color=color, alpha=0.75)
        axes[1].errorbar(
            x,
            comparison["mean"],
            yerr=[
                [comparison["mean"] - comparison["ci95_low"]],
                [comparison["ci95_high"] - comparison["mean"]],
            ],
            fmt="o",
            color="black",
            capsize=5,
        )
    axes[1].axhline(0.0, color="#333333", linewidth=1)
    axes[1].set_xticks([0, 1], ["THOR-MAGNI", "HuRoN"])
    axes[1].set_ylabel("NLL gain: history-only - action")
    axes[1].set_title("Recorded-action conditional association")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default="experiments/external_response_benchmark_formal_v2",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/external_response_benchmark_20260901",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(Path(args.input_root))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(summary, output_dir / "summary.md")
    _write_figure(summary, output_dir / "external_benchmark.png")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
