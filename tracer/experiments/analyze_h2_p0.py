from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tracer.experiments.paired_statistics import (
    blocked_paired_bootstrap,
    validate_formal_episode_analysis,
    validate_formal_episode_records,
)


PRIMARY_CONTRASTS = (
    ("reset-minus-persistent", "reset", "persistent"),
    ("recurrent-minus-persistent", "recurrent", "persistent"),
    ("shuffled-id-minus-persistent", "shuffled-id", "persistent"),
)
METRICS = (
    "predictive_trajectory_nll",
    "trajectory_ade",
    "trajectory_fde",
)


def _read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path_index, path in enumerate(paths):
        match = re.fullmatch(r"seed(\d+)", path.parent.name)
        training_seed = int(match.group(1)) if match else path_index
        with path.open(newline="", encoding="utf-8") as handle:
            seed_rows = list(csv.DictReader(handle))
        for row in seed_rows:
            row["_training_seed"] = str(training_seed)
        rows.extend(seed_rows)
    if not rows:
        raise ValueError("raw_metrics.csv is empty")
    return rows


def _episode_metric(
    rows: list[dict[str, str]],
    *,
    reentry_only: bool = False,
) -> dict[tuple[int, str, int, str, str], float]:
    grouped: dict[tuple[int, str, int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["main_evaluation"] != "1":
            continue
        if reentry_only and not (
            int(row["entity_slot"]) == 0 and int(row["window"]) == 4
        ):
            continue
        split = row["split"]
        training_seed = int(row["_training_seed"])
        episode = int(row["episode_id"])
        method = row["method"]
        for metric in METRICS:
            grouped[(training_seed, split, episode, method, metric)].append(
                float(row[metric])
            )
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def _final_accuracy(
    rows: list[dict[str, str]],
) -> dict[tuple[int, str, int, str], float]:
    last: dict[tuple[int, str, int, str, int], tuple[int, float]] = {}
    for row in rows:
        if row["active"] != "1":
            continue
        key = (
            int(row["_training_seed"]),
            row["split"],
            int(row["episode_id"]),
            row["method"],
            int(row["entity_slot"]),
        )
        value = (int(row["window"]), float(row["posterior_mode_correct"]))
        if key not in last or value[0] > last[key][0]:
            last[key] = value
    grouped: dict[tuple[int, str, int, str], list[float]] = defaultdict(list)
    for (training_seed, split, episode, method, _), (_, correct) in last.items():
        grouped[(training_seed, split, episode, method)].append(correct)
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def _contrast_records(
    episode_metrics: dict[tuple[int, str, int, str, str], float],
    final_accuracy: dict[tuple[int, str, int, str], float],
    bootstrap_samples: int,
    seed: int,
    scope: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    splits = sorted({key[1] for key in episode_metrics})
    for split in splits:
        units = sorted(
            {(key[0], key[2]) for key in episode_metrics if key[1] == split}
        )
        for contrast_name, baseline, persistent in PRIMARY_CONTRASTS:
            for metric in METRICS:
                differences: dict[int, dict[int, float]] = defaultdict(dict)
                for training_seed, episode in units:
                    differences[training_seed][episode] = (
                        episode_metrics[
                            (training_seed, split, episode, baseline, metric)
                        ]
                        - episode_metrics[
                            (training_seed, split, episode, persistent, metric)
                        ]
                    )
                records.append(
                    {
                        "scope": scope,
                        "split": split,
                        "contrast": contrast_name,
                        "metric": metric,
                        **blocked_paired_bootstrap(
                            differences, rng, bootstrap_samples
                        ),
                    }
                )
            if scope == "all-post-initial":
                accuracy_difference: dict[int, dict[int, float]] = defaultdict(dict)
                for training_seed, episode in units:
                    accuracy_difference[training_seed][episode] = (
                        final_accuracy[
                            (training_seed, split, episode, persistent)
                        ]
                        - final_accuracy[
                            (training_seed, split, episode, baseline)
                        ]
                    )
                records.append(
                    {
                        "scope": scope,
                        "split": split,
                        "contrast": f"persistent-minus-{baseline}",
                        "metric": "final_mode_accuracy",
                        **blocked_paired_bootstrap(
                            accuracy_difference, rng, bootstrap_samples
                        ),
                    }
                )
    return records


def _window0_sanity(rows: list[dict[str, str]]) -> dict[str, float]:
    methods = ("no-belief", "reset", "persistent", "shuffled-id", "recurrent")
    values: dict[tuple[int, str, int, str], float] = {}
    for row in rows:
        if int(row["window"]) == 0 and row["active"] == "1":
            key = (
                int(row["_training_seed"]),
                row["split"],
                int(row["episode_id"]),
                row["method"],
            )
            values.setdefault(key, 0.0)
            values[key] += float(row["predictive_trajectory_nll"])
    maximum_difference = 0.0
    for training_seed, split, episode, _ in values:
        method_values = [
            values[(training_seed, split, episode, method)] for method in methods
        ]
        maximum_difference = max(
            maximum_difference, max(method_values) - min(method_values)
        )
    return {"max_episode_sum_nll_difference": maximum_difference}


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_markdown(
    path: Path,
    records: list[dict[str, Any]],
    analysis_status: str,
) -> None:
    formal = analysis_status == "formal-confirmatory"
    lines = [
        "# H2 Formal-v1 Episode-Level Paired Analysis"
        if formal
        else "# H2 P0 Episode-Level Paired Analysis",
        "",
        (
            "This is the frozen Formal-v1 confirmatory analysis."
            if formal
            else "This is a post-pilot exploratory analysis, not a preregistered confirmatory test."
        ),
        "The independent unit is the episode; entity-window observations are aggregated within episode. Model seeds are nuisance blocks: contrasts are averaged over seeds within each episode, then episodes are bootstrapped.",
        "Positive contrasts favor persistent belief for NLL/ADE/FDE and favor persistent in the accuracy rows.",
        "",
        "| Scope | Split | Contrast | Metric | Mean | 95% bootstrap CI | dz |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for record in records:
        dz = float(record["paired_dz"])
        dz_text = f"{dz:.3f}" if math.isfinite(dz) else "NA"
        lines.append(
            "| {scope} | {split} | {contrast} | {metric} | {mean:.6f} | "
            "[{low:.6f}, {high:.6f}] | {dz} |".format(
                scope=record["scope"],
                split=record["split"],
                contrast=record["contrast"],
                metric=record["metric"],
                mean=float(record["mean"]),
                low=float(record["bootstrap_ci95_low"]),
                high=float(record["bootstrap_ci95_high"]),
                dz=dz_text,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze H2 P0 at the episode level.")
    parser.add_argument("--raw", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20270810)
    parser.add_argument(
        "--analysis-status",
        choices=("post-pilot-exploratory", "formal-confirmatory"),
        default="post-pilot-exploratory",
    )
    parser.add_argument("--formal-manifest", default=None)
    args = parser.parse_args()

    raw_paths = [Path(path) for path in args.raw]
    formal_context = None
    if args.analysis_status == "formal-confirmatory":
        if args.formal_manifest is None:
            raise RuntimeError("formal H2 analysis requires --formal-manifest")
        formal_context = validate_formal_episode_analysis(
            Path(args.formal_manifest),
            "H2",
            raw_paths,
            args.bootstrap_samples,
            args.seed,
        )
    elif args.formal_manifest is not None:
        raise RuntimeError("exploratory analysis must not claim a formal manifest")

    rows = _read_rows(raw_paths)
    episode_metrics = _episode_metric(rows)
    reentry_metrics = _episode_metric(rows, reentry_only=True)
    final_accuracy = _final_accuracy(rows)
    records = _contrast_records(
        episode_metrics,
        final_accuracy,
        args.bootstrap_samples,
        args.seed,
        "all-post-initial",
    )
    records.extend(
        _contrast_records(
            reentry_metrics,
            final_accuracy,
            args.bootstrap_samples,
            args.seed + 1,
            "entity0-reentry-window4",
        )
    )
    if formal_context is not None:
        validate_formal_episode_records(
            records,
            formal_context["config"],
            "H2",
            {int(row["_training_seed"]) for row in rows},
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "paired_episode_contrasts.csv", records)
    payload = {
        "analysis_status": (
            "FORMAL_CONFIRMATORY"
            if args.analysis_status == "formal-confirmatory"
            else "POST_PILOT_EXPLORATORY"
        ),
        "independent_unit": "paired episode",
        "model_seed_handling": "average within episode, then bootstrap episodes",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "window0_sanity": _window0_sanity(rows),
        "contrasts": records,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(output_dir / "analysis.md", records, args.analysis_status)
    print(f"saved_analysis={output_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
