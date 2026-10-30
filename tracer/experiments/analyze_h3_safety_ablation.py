"""Paired analysis of the probabilistic safety-filter ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tracer.experiments.paired_statistics import blocked_paired_bootstrap


POLICIES = (
    "task-only",
    "no-belief-joint",
    "reset-additive",
    "reset-joint",
    "persistent-additive",
    "persistent-joint",
    "oracle-joint",
)
METRICS = (
    "realized_total_cost",
    "task_progress",
    "collision_rate",
    "safety_cost",
    "human_cost",
    "min_distance",
)


def _read_metrics(path: Path) -> dict[tuple[int, str, str, int], dict[str, float]]:
    values: dict[tuple[int, str, str, int], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seed"]), row["split"], row["policy"], int(row["episode"]))
            values[key] = {
                name: float(value)
                for name, value in row.items()
                if name not in {"seed", "split", "policy", "episode"}
            }
    if not values:
        raise ValueError(f"empty episode metrics: {path}")
    return values


def _read_runs(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paired_metric(
    filtered: dict[tuple[int, str, str, int], dict[str, float]],
    unfiltered: dict[tuple[int, str, str, int], dict[str, float]],
    split: str,
    policy: str,
    metric: str,
) -> dict[str, object]:
    # The explicit key construction below keeps the pairing visible and avoids
    # accidentally comparing different episode sets after a partial run.
    keys = sorted(
        key
        for key in filtered
        if key[1] == split and key[2] == policy
    )
    by_seed: dict[int, dict[int, float]] = {}
    for seed, _, _, episode in keys:
        other = (seed, split, policy, episode)
        if other not in unfiltered:
            raise ValueError(f"missing paired no-filter record: {other}")
        by_seed.setdefault(seed, {})[episode] = (
            unfiltered[other][metric] - filtered[(seed, split, policy, episode)][metric]
        )
    import numpy as np

    stable_offset = sum(ord(char) for char in f"{split}:{policy}:{metric}")
    return blocked_paired_bootstrap(
        by_seed,
        np.random.default_rng(20270810 + stable_offset),
        10_000,
    )


def _selection_flip_rate(
    filtered_runs: dict[str, Any],
    unfiltered_runs: dict[str, Any],
    split: str,
    policy: str,
) -> float:
    first: list[int] = []
    second: list[int] = []
    for seed in sorted(filtered_runs, key=int):
        filtered = filtered_runs[seed][split][policy]["selected_candidates"]
        unfiltered = unfiltered_runs[seed][split][policy]["selected_candidates"]
        if len(filtered) != len(unfiltered):
            raise ValueError(f"selection record lengths differ for {seed}/{split}/{policy}")
        first.extend(int(row[0]) for row in filtered)
        second.extend(int(row[0]) for row in unfiltered)
    return sum(left != right for left, right in zip(first, second, strict=True)) / len(first)


def analyze(filtered_root: Path, unfiltered_root: Path, output_dir: Path) -> dict[str, Any]:
    filtered_metrics = _read_metrics(filtered_root / "episode_metrics.csv")
    unfiltered_metrics = _read_metrics(unfiltered_root / "episode_metrics.csv")
    filtered_runs = _read_runs(filtered_root / "runs.json")
    unfiltered_runs = _read_runs(unfiltered_root / "runs.json")
    if set(filtered_metrics) != set(unfiltered_metrics):
        raise ValueError("filtered and no-filter episode keys differ")
    records = []
    for split in ("id", "ood"):
        for policy in POLICIES:
            for metric in METRICS:
                record = _paired_metric(
                    filtered_metrics,
                    unfiltered_metrics,
                    split,
                    policy,
                    metric,
                )
                records.append(
                    {
                        "split": split,
                        "policy": policy,
                        "contrast": "no_filter_minus_filtered",
                        "metric": metric,
                        **record,
                    }
                )
    payload = {
        "analysis_status": "SECONDARY_SAFETY_ABLATION",
        "independent_unit": "paired episode",
        "model_seed_handling": "average within episode, then bootstrap episodes",
        "filtered_root": str(filtered_root),
        "unfiltered_root": str(unfiltered_root),
        "selection_flip_rate": {
            split: {
                policy: _selection_flip_rate(
                    filtered_runs, unfiltered_runs, split, policy
                )
                for policy in POLICIES
            }
            for split in ("id", "ood")
        },
        "contrasts": records,
        "claim_boundary": (
            "Safety-filter ablation under the current TRACER/SAGE synthetic protocol; "
            "no hard-safety or real-human causal claim."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# H3 Safety-Filter Ablation",
        "",
        "Positive values are no-filter minus filtered; for cost and collision metrics, positive values favor the filtered system.",
        "",
        "| Split | Policy | Metric | Mean | 95% CI |",
        "|---|---|---|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['split']} | {record['policy']} | {record['metric']} | "
            f"{float(record['mean']):.6f} | "
            f"[{float(record['bootstrap_ci95_low']):.6f}, "
            f"{float(record['bootstrap_ci95_high']):.6f}] |"
        )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-root", required=True)
    parser.add_argument("--unfiltered-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = analyze(Path(args.filtered_root), Path(args.unfiltered_root), Path(args.output_dir))
    print(json.dumps({"status": "complete", "records": len(payload["contrasts"])}, sort_keys=True))


if __name__ == "__main__":
    main()
