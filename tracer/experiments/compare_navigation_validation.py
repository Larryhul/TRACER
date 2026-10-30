from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("success", "collision_free_success", "collision", "wall_collision", "timeout", "task_progress", "total_time")


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def compare(official_dir, tracer_dir, baseline="sicnav-np", tracer_policy="tracer-persistent"):
    official_rows = [row for row in read_rows(official_dir / "episode_metrics.csv") if row["policy"] == baseline]
    official = {int(row["case"]): row for row in official_rows}
    if len(official) != len(official_rows):
        raise ValueError("baseline cases must not contain duplicate rows")
    rows = [row for row in read_rows(tracer_dir / "episode_metrics.csv") if row["policy"] == tracer_policy]
    cases = sorted(official)
    if not rows or {int(row["case"]) for row in rows} != set(cases):
        raise ValueError("navigation comparison requires exactly paired nonempty case sets")
    seeds = sorted({int(row["seed"]) for row in rows})
    pairs = {(int(row["case"]), int(row["seed"])) for row in rows}
    if len(pairs) != len(rows) or pairs != {(case, seed) for case in cases for seed in seeds}:
        raise ValueError("navigation comparison requires a complete nonduplicated case-by-seed matrix")
    official_metadata = json.loads((official_dir / "metadata.json").read_text(encoding="utf-8"))
    tracer_metadata = json.loads((tracer_dir / "metadata.json").read_text(encoding="utf-8"))
    parameters = tracer_metadata["parameters"]
    if official_metadata["parameters"]["max_steps"] != min(parameters["max_steps"], parameters["windows"] * parameters["execute_steps"]):
        raise ValueError("navigation step budgets differ")
    if official_metadata["external_repository"]["commit"] != tracer_metadata["external_repository"]["commit"]:
        raise ValueError("navigation simulator revisions differ")
    motion_equal = True
    for row in rows:
        reference = official[int(row["case"])]
        for key in ("initial_scene_hash", "environment_dt"):
            if row[key] != reference[key]:
                raise ValueError(f"navigation case {row['case']} has different {key}")
        motion_equal &= row["robot_kinematics"] == reference["robot_kinematics"]
        if row["robot_kinematics"] == "unicycle" and reference["robot_kinematics"] == "unicycle":
            motion_equal &= abs(float(row["robot_initial_heading"]) - float(reference["robot_initial_heading"])) < 1e-7
    paired = {}
    generator = np.random.default_rng(968731)
    draws = generator.integers(len(cases), size=(10000, len(cases)))
    for metric in METRICS:
        reference = np.array([float(official[case][metric]) for case in cases])
        observed = np.array([np.mean([float(row[metric]) for row in rows if int(row["case"]) == case]) for case in cases])
        if not np.isfinite(reference).all() or not np.isfinite(observed).all():
            raise ValueError("paired navigation outcomes must be finite")
        delta = observed - reference
        paired[metric] = {"official": float(reference.mean()), "tracer": float(observed.mean()),
                          "tracer_minus_official": float(delta.mean()),
                          "descriptive_case_bootstrap_ci95": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()}
    return {"phase": "development_validation", "baseline": baseline, "tracer_policy": tracer_policy, "case_count": len(cases),
            "cases": cases, "tracer_model_seeds": seeds,
            "initial_scene_dt_budget_audit": "pass", "native_kinematics_and_heading_equal": bool(motion_equal),
            "motion_limit_config_sha256": tracer_metadata["trajectory_contract"].get("motion_policy_sha256"),
            "noninferiority_established": False,
            "inference_boundary": "descriptive validation; no prespecified margin, fresh-test claim, or finite-sample non-inferiority test",
            "belief_boundary": tracer_metadata["response_contract"]["mode_semantics"],
            "metrics": paired}


def main():
    parser = argparse.ArgumentParser(description="Audit case pairing and report descriptive external navigation gaps.")
    parser.add_argument("--official-dir", required=True)
    parser.add_argument("--tracer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", default="sicnav-np")
    parser.add_argument("--tracer-policy", default="tracer-persistent")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("comparison output exists; choose a new path")
    result = compare(Path(args.official_dir), Path(args.tracer_dir), args.baseline, args.tracer_policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
