from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tracer.experiments.collect_external_planner_supervision import file_hash
from tracer.experiments.summarize_executable_value_navigation import paired_navigation


EXTRA_METRICS = ("success", "stagnation_time", "human_velocity_variation", "terminal_goal_distance")


def assess_reliable_navigation(parent, candidate, minimum_scenes=32):
    report = paired_navigation(parent, candidate)
    missing = sorted({name for row in parent + candidate for name in EXTRA_METRICS if name not in row})
    if missing:
        return {**report, "reliability_status": "INCOMPLETE_TELEMETRY", "missing_metrics": missing,
                "development_screen_passed": False, "independent_confirmation_required": True}
    indexed = {(int(row["seed"]), int(row["case"])): row for row in parent}
    cases = sorted({int(row["case"]) for row in candidate})
    case_indices = np.random.default_rng(970910).integers(0, len(cases), (10000, len(cases)))
    for name in EXTRA_METRICS:
        previous = np.asarray([float(indexed[(int(row["seed"]), int(row["case"]))][name]) for row in candidate])
        current = np.asarray([float(row[name]) for row in candidate])
        if not np.isfinite(previous).all() or not np.isfinite(current).all():
            raise ValueError("reliability navigation cannot treat missing or nonfinite observations as safe")
        differences = current - previous
        case_means = np.asarray([differences[[int(row["case"]) == case for row in candidate]].mean() for case in cases])
        report["metrics"][name] = {"parent": float(previous.mean()), "candidate": float(current.mean()),
                                    "candidate_minus_parent": float(case_means.mean()),
                                    "case_bootstrap_ci95": np.quantile(case_means[case_indices].mean(axis=1), [.025, .975]).tolist()}
    previous_shortfall = np.asarray([
        float(indexed[(int(row["seed"]), int(row["case"]))]["terminal_goal_distance"])
        * (1. - float(indexed[(int(row["seed"]), int(row["case"]))]["success"]))
        for row in candidate
    ])
    current_shortfall = np.asarray([
        float(row["terminal_goal_distance"]) * (1. - float(row["success"])) for row in candidate
    ])
    shortfall_differences = current_shortfall - previous_shortfall
    shortfall_case_means = np.asarray([
        shortfall_differences[[int(row["case"]) == case for row in candidate]].mean() for case in cases
    ])
    report["metrics"]["goal_shortfall"] = {
        "parent": float(previous_shortfall.mean()), "candidate": float(current_shortfall.mean()),
        "candidate_minus_parent": float(shortfall_case_means.mean()),
        "case_bootstrap_ci95": np.quantile(
            shortfall_case_means[case_indices].mean(axis=1), [.025, .975]
        ).tolist(),
    }
    tiers = (
        (("collision_free_success", -1), ("collision", 1), ("wall_collision", 1)),
        (("success", -1), ("timeout", 1), ("stagnation_time", 1)),
        (("goal_shortfall", 1), ("human_velocity_variation", 1)),
        (("total_time", 1),),
    )
    first_improvement, earlier_tiers_not_worse = None, True
    for index, tier in enumerate(tiers):
        differences = [direction * report["metrics"][name]["candidate_minus_parent"] for name, direction in tier]
        if any(value > 1e-10 for value in differences):
            earlier_tiers_not_worse = False
            break
        if any(value < -1e-10 for value in differences):
            first_improvement = index
            break
    passed = (report["paired_scenes"] >= minimum_scenes and report["observed_safety_not_worse"]
              and earlier_tiers_not_worse and first_improvement is not None)
    report.update({"reliability_status": "COMPLETED", "minimum_development_scenes": minimum_scenes,
                   "first_improved_priority_tier": first_improvement,
                   "development_screen_passed": bool(passed), "independent_confirmation_required": True,
                   "zero_observed_harm_scene_upper95": 1. - .05 ** (1. / len(cases)) if report["observed_safety_not_worse"] else None,
                   "human_metric_scope": "Observed mean-human velocity total variation, not causal disturbance or subjective comfort.",
                   "progress_metric_scope": "Goal shortfall is zero after simulator-declared success; terminal position inside the success radius is not ranked as extra task progress. Raw task_progress remains descriptive.",
                   "assessment_source_sha256": file_hash(Path(__file__).resolve()),
                   "boundary": "Ordered development screen, not significance or noninferiority proof. Zero empirical harm is not zero population risk; scene upper bound assumes independent sampled scenes."})
    return report


def main():
    parser = argparse.ArgumentParser(description="Assess actual repeated replanning by safety, task completion, progress/human perturbation, then time.")
    parser.add_argument("--parent-csv", nargs="+", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.candidate_dir).resolve()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "COMPLETED" or summary.get("changed_required_sources") or summary.get("changed_sources"):
        raise ValueError("reliability navigation requires completed frozen inputs")
    sources = [Path(name).resolve() for name in args.parent_csv]
    candidate_path = directory / "episodes.csv"
    parent = [row for path in sources for row in csv.DictReader(path.open())]
    candidate = list(csv.DictReader(candidate_path.open()))
    report = assess_reliable_navigation(parent, candidate)
    report["input_hashes"] = {str(path): file_hash(path) for path in sources + [candidate_path, directory / "summary.json"]}
    with Path(args.output).open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
