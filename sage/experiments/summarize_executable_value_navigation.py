from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sage.experiments.collect_external_planner_supervision import file_hash


def paired_navigation(parent, candidate):
    indexed = {(int(row["seed"]), int(row["case"])): row for row in parent}
    keys = [(int(row["seed"]), int(row["case"])) for row in candidate]
    if len(indexed) != len(parent) or len(set(keys)) != len(keys) or set(indexed) != set(keys):
        raise ValueError("navigation comparison requires unique identical seed/case sets")
    for row, key in zip(candidate, keys):
        for field in ("initial_scene_hash", "environment_dt", "robot_kinematics", "robot_initial_heading", "policy"):
            if row[field] != indexed[key][field]:
                raise ValueError(f"navigation pair differs on {field}")
    metrics = {}
    scenes = sorted({key[1] for key in keys})
    for metric in ("collision_free_success", "collision", "wall_collision", "timeout", "total_time", "task_progress"):
        previous = np.asarray([float(indexed[key][metric]) for key in keys])
        current = np.asarray([float(row[metric]) for row in candidate])
        difference = current - previous
        scene_differences = np.asarray([difference[[key[1] == case for key in keys]].mean() for case in scenes])
        resamples = np.random.default_rng(519817).integers(0, len(scenes), size=(10000, len(scenes)))
        metrics[metric] = {
            "parent": float(previous.mean()), "candidate": float(current.mean()),
            "candidate_minus_parent": float(scene_differences.mean()),
            "case_bootstrap_ci95": np.quantile(scene_differences[resamples].mean(1), [.025, .975]).tolist(),
        }
    worsened = [key[1] for row, key in zip(candidate, keys) if
                float(row["collision_free_success"]) < float(indexed[key]["collision_free_success"]) or
                float(row["collision"]) > float(indexed[key]["collision"]) or
                float(row["wall_collision"]) > float(indexed[key]["wall_collision"]) or
                float(row["timeout"]) > float(indexed[key]["timeout"])]
    return {
        "paired_scenes": len(scenes), "metrics": metrics, "worsened_safety_or_success_cases": sorted(set(worsened)),
        "observed_safety_not_worse": not worsened,
        "time_directionally_improved": metrics["total_time"]["candidate_minus_parent"] < 0.,
        "time_case_interval_negative": metrics["total_time"]["case_bootstrap_ci95"][1] < 0.,
        "value_action_changes": sum(int(row.get("executable_value_action_changes", 0)) for row in candidate),
        "boundary": "Paired development navigation after model selection; not untouched confirmation or a safety/noninferiority guarantee.",
    }


def main():
    parser = argparse.ArgumentParser(description="Audit exact-case executable-value navigation against the frozen parent.")
    parser.add_argument("--parent-csv", nargs="+", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.candidate_dir).resolve()
    summary = json.loads((directory / "summary.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if summary["status"] != "COMPLETED" or summary["changed_required_sources"]:
        raise ValueError("candidate navigation must complete with unchanged required inputs")
    if protocol["phase"] != "validation" or not protocol["navigation_only"] or not protocol["executable_value_model"]:
        raise ValueError("this summary requires actual value-head validation navigation")
    sources = [Path(path).resolve() for path in args.parent_csv]
    parent = [row for path in sources for row in csv.DictReader(path.open())]
    candidate_path = directory / "episodes.csv"
    candidate = list(csv.DictReader(candidate_path.open()))
    report = paired_navigation(parent, candidate)
    report["input_hashes"] = {str(path): file_hash(path) for path in sources + [candidate_path, directory / "summary.json", directory / "protocol.json"]}
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("paired navigation summary already exists")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key:value for key,value in report.items() if key != "input_hashes"}, indent=2))


if __name__ == "__main__":
    main()
