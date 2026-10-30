from __future__ import annotations

import argparse
import csv
import json
import math
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


CONTRASTS = (
    ("reset-minus-persistent", "reset-joint", "persistent-joint", "realized_total_cost"),
    ("persistent-minus-reset", "persistent-joint", "reset-joint", "task_progress"),
    ("reset-minus-persistent", "reset-joint", "persistent-joint", "human_cost"),
    ("persistent-minus-reset", "persistent-joint", "reset-joint", "min_distance"),
    ("reset-minus-persistent", "reset-joint", "persistent-joint", "collision_rate"),
    ("additive-minus-joint", "persistent-additive", "persistent-joint", "realized_total_cost"),
    ("joint-minus-additive", "persistent-joint", "persistent-additive", "task_progress"),
    ("reset-additive-minus-reset-joint", "reset-additive", "reset-joint", "realized_total_cost"),
    ("reset-minus-persistent", "reset-joint", "persistent-joint", "oracle_regret"),
    ("no-belief-minus-persistent", "no-belief-joint", "persistent-joint", "oracle_regret"),
    ("task-only-minus-persistent", "task-only", "persistent-joint", "oracle_regret"),
    ("task-only-minus-persistent", "task-only", "persistent-joint", "realized_total_cost"),
)

FACTORIAL_INTERACTIONS = (
    (
        "belief-by-response-interaction",
        "realized_total_cost",
        ("reset-additive", "reset-joint", "persistent-additive", "persistent-joint"),
    ),
)

RECURRENT_CONTRASTS = (
    (
        "recurrent-minus-persistent",
        "recurrent-joint",
        "persistent-joint",
        "realized_total_cost",
    ),
    (
        "recurrent-minus-persistent",
        "recurrent-joint",
        "persistent-joint",
        "task_progress",
    ),
    (
        "recurrent-minus-persistent",
        "recurrent-joint",
        "persistent-joint",
        "human_cost",
    ),
    (
        "recurrent-minus-persistent",
        "recurrent-joint",
        "persistent-joint",
        "oracle_regret",
    ),
)


def _read_episode_metrics(path: Path) -> dict[tuple[int, str, str, int], dict[str, float]]:
    values: dict[tuple[int, str, str, int], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["seed"]),
                row["split"],
                row["policy"],
                int(row["episode"]),
            )
            values[key] = {
                name: float(value)
                for name, value in row.items()
                if name not in {"seed", "split", "policy", "episode"}
            }
    if not values:
        raise ValueError("episode_metrics.csv is empty")
    return values


def _selection_changes(runs: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    comparisons = {
        "persistent-vs-reset": ("persistent-joint", "reset-joint"),
        "joint-vs-additive": ("persistent-joint", "persistent-additive"),
        "persistent-vs-uniform": ("persistent-joint", "no-belief-joint"),
        "persistent-vs-oracle": ("persistent-joint", "oracle-joint"),
        "persistent-vs-task-only": ("persistent-joint", "task-only"),
    }
    has_recurrent = any(
        "recurrent-joint" in split_run
        for seed_run in runs.values()
        for split_run in seed_run.values()
    )
    if has_recurrent:
        comparisons["recurrent-vs-persistent"] = (
            "recurrent-joint",
            "persistent-joint",
        )
    results: dict[str, dict[str, float | int]] = {}
    splits = sorted(
        {
            split
            for seed_run in runs.values()
            for split in seed_run
        }
    )
    for split in splits:
        results[split] = {}
        for name, (first, second) in comparisons.items():
            first_selected = []
            second_selected = []
            for seed in sorted(runs, key=int):
                policies = runs[seed][split]
                first_selected.extend(policies[first]["selected_candidates"])
                second_selected.extend(policies[second]["selected_candidates"])
            changes = [
                int(first_episode[0] != second_episode[0])
                for first_episode, second_episode in zip(
                    first_selected, second_selected, strict=True
                )
            ]
            first_divergence = []
            for first_episode, second_episode in zip(
                first_selected, second_selected, strict=True
            ):
                differences = [
                    window
                    for window, (first_index, second_index) in enumerate(
                        zip(first_episode, second_episode, strict=True)
                    )
                    if first_index != second_index
                ]
                first_divergence.append(differences[0] if differences else None)
            diverged = [window for window in first_divergence if window is not None]
            results[split][name] = {
                "episodes": len(changes),
                "first_window_selection_change_rate": statistics.fmean(changes),
                "rollout_first_divergence_rate": len(diverged) / len(first_divergence),
                "mean_first_divergence_window_when_changed": (
                    statistics.fmean(diverged) if diverged else None
                ),
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Episode-level paired analysis for H3 P0.")
    parser.add_argument("--episode-metrics", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20270810)
    parser.add_argument(
        "--analysis-status",
        choices=(
            "post-pilot-exploratory",
            "secondary-robustness",
            "formal-confirmatory",
        ),
        default="post-pilot-exploratory",
    )
    parser.add_argument("--formal-manifest", default=None)
    args = parser.parse_args()

    episode_metrics_path = Path(args.episode_metrics)
    runs_path = Path(args.runs)
    formal_context = None
    if args.analysis_status == "formal-confirmatory":
        if args.formal_manifest is None:
            raise RuntimeError("formal H3 analysis requires --formal-manifest")
        formal_context = validate_formal_episode_analysis(
            Path(args.formal_manifest),
            "H3",
            [episode_metrics_path, runs_path],
            args.bootstrap_samples,
            args.seed,
        )
    elif (
        args.analysis_status == "post-pilot-exploratory"
        and args.formal_manifest is not None
    ):
        raise RuntimeError("exploratory analysis must not claim a formal manifest")

    values = _read_episode_metrics(episode_metrics_path)
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    splits = sorted({key[1] for key in values})
    contrasts = list(CONTRASTS)
    if any(key[2] == "recurrent-joint" for key in values):
        contrasts.extend(RECURRENT_CONTRASTS)
    for split in splits:
        units = sorted({(key[0], key[3]) for key in values if key[1] == split})
        for contrast, first, second, metric in contrasts:
            differences: dict[int, dict[int, float]] = defaultdict(dict)
            for seed, episode in units:
                differences[seed][episode] = (
                    values[(seed, split, first, episode)][metric]
                    - values[(seed, split, second, episode)][metric]
                )
            records.append(
                {
                    "split": split,
                    "contrast": contrast,
                    "metric": metric,
                    **blocked_paired_bootstrap(
                        differences, rng, args.bootstrap_samples
                    ),
                }
            )
        for contrast, metric, policies in FACTORIAL_INTERACTIONS:
            reset_additive, reset_joint, persistent_additive, persistent_joint = policies
            differences = defaultdict(dict)
            for seed, episode in units:
                differences[seed][episode] = (
                    values[(seed, split, reset_additive, episode)][metric]
                    - values[(seed, split, reset_joint, episode)][metric]
                    - values[(seed, split, persistent_additive, episode)][metric]
                    + values[(seed, split, persistent_joint, episode)][metric]
                )
            records.append(
                {
                    "split": split,
                    "contrast": contrast,
                    "metric": metric,
                    **blocked_paired_bootstrap(
                        differences, rng, args.bootstrap_samples
                    ),
                }
            )
    if formal_context is not None:
        validate_formal_episode_records(
            records,
            formal_context["config"],
            "H3",
            {key[0] for key in values},
        )

    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    selection_changes = _selection_changes(runs)
    policy_means: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    policy_metrics = (
        "task_progress",
        "human_cost",
        "min_distance",
        "collision_rate",
        "realized_total_cost",
        "oracle_regret",
        "oracle_hit_rate",
        "candidate_rank_spearman",
        "candidate_rank_kendall",
        "task_feasible_candidate_count",
        "task_feasible_candidate_fraction",
        "candidate_true_cost_range",
    )
    if all("runtime_ms_per_window" in record for record in values.values()):
        policy_metrics += ("runtime_ms_per_episode", "runtime_ms_per_window")
    for split in splits:
        policies = sorted({key[2] for key in values if key[1] == split})
        units = sorted({(key[0], key[3]) for key in values if key[1] == split})
        for policy in policies:
            policy_means[split][policy] = {
                metric: statistics.fmean(
                    values[(seed, split, policy, episode)][metric]
                    for seed, episode in units
                )
                for metric in policy_metrics
            }

    checks: dict[str, bool] = {}
    for split in splits:
        persistent = policy_means[split]["persistent-joint"]
        reset = policy_means[split]["reset-joint"]
        task_only = policy_means[split]["task-only"]
        task_floor = 0.98 * reset["task_progress"] if reset["task_progress"] > 0.0 else 0.0
        checks[f"{split}_persistent_task_progress_positive"] = persistent["task_progress"] > 0.0
        checks[f"{split}_persistent_task_noninferior_to_reset_2pct"] = (
            persistent["task_progress"] >= task_floor
        )
        checks[f"{split}_persistent_collision_not_worse_by_0_02"] = (
            persistent["collision_rate"] <= reset["collision_rate"] + 0.02
        )
        checks[f"{split}_persistent_changes_first_selection_vs_reset"] = (
            selection_changes[split]["persistent-vs-reset"][
                "rollout_first_divergence_rate"
            ]
            > 0.0
        )
        checks[f"{split}_at_least_two_task_feasible_candidates"] = (
            persistent["task_feasible_candidate_count"] >= 2.0
        )
        checks[f"{split}_candidate_true_cost_range_positive"] = (
            persistent["candidate_true_cost_range"] > 0.01
        )
        checks[f"{split}_predicted_candidate_ranking_positive"] = (
            persistent["candidate_rank_spearman"] > 0.0
        )
        checks[f"{split}_task_only_leaves_oracle_regret"] = (
            task_only["oracle_regret"] > 0.0
        )

    payload = {
        "analysis_status": (
            "FORMAL_CONFIRMATORY"
            if args.analysis_status == "formal-confirmatory"
            else (
                "SECONDARY_ROBUSTNESS"
                if args.analysis_status == "secondary-robustness"
                else "POST_PILOT_EXPLORATORY"
            )
        ),
        "independent_unit": "paired episode",
        "model_seed_handling": "average within episode, then bootstrap episodes",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "selection_divergence": selection_changes,
        "policy_means": policy_means,
        "contrasts": records,
        "gate": {
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "rule_status": (
                "formal operational integrity checks; not an automatic gate"
                if args.analysis_status == "formal-confirmatory"
                else (
                    "secondary robustness operational checks; not a confirmatory gate"
                    if args.analysis_status == "secondary-robustness"
                    else "post-pilot operational smoke rule; not preregistered"
                )
            ),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "paired_episode_contrasts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    formal = args.analysis_status == "formal-confirmatory"
    lines = [
        (
            "# H3 Formal-v1 Episode-Level Paired Analysis"
            if formal
            else (
                "# H3 Secondary Robustness Episode-Level Paired Analysis"
                if args.analysis_status == "secondary-robustness"
                else "# H3 P0 Episode-Level Paired Analysis"
            )
        ),
        "",
        (
            "This is the frozen Formal-v1 confirmatory analysis."
            if formal
            else (
                "This is a secondary robustness analysis, not a confirmatory test."
                if args.analysis_status == "secondary-robustness"
                else "This is a post-pilot exploratory analysis, not a preregistered confirmatory test."
            )
        ),
        "The independent unit is the paired episode. Model seeds are nuisance blocks: contrasts are averaged over seeds within each episode, then episodes are bootstrapped. For each policy pair, only the first differing selection is interpreted: rollout states are identical before that divergence.",
        "For pairwise rows, positive values favor the method named first in the contrast. The factorial interaction is reported as (reset-additive - reset-joint) - (persistent-additive - persistent-joint).",
        "",
        "| Split | Contrast | Metric | Mean | 95% bootstrap CI | dz |",
        "|---|---|---|---:|---:|---:|",
    ]
    for record in records:
        dz = float(record["paired_dz"])
        dz_text = f"{dz:.3f}" if math.isfinite(dz) else "NA"
        lines.append(
            f"| {record['split']} | {record['contrast']} | {record['metric']} | "
            f"{float(record['mean']):.6f} | "
            f"[{float(record['bootstrap_ci95_low']):.6f}, {float(record['bootstrap_ci95_high']):.6f}] | "
            f"{dz_text} |"
        )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
