from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from sage.experiments.collect_external_planner_supervision import file_hash
from sage.experiments.assess_reliable_navigation import assess_reliable_navigation
from sage.experiments.external_value_policy import value_policy_digest, value_policy_stages


def confirmation_cases(start, count, case_splits, excluded_cases):
    if count < 64 or start < 0:
        raise ValueError("confirmation requires a prespecified nonnegative block of at least64 cases")
    cases = list(range(start, start + count))
    excluded = set(excluded_cases) | set().union(*(set(values) for values in case_splits.values()))
    if excluded.intersection(cases):
        raise ValueError("confirmation cases overlap disclosed development/test/previous-confirmation cases")
    return cases, sorted(excluded)


def confirmation_reference_model(model):
    value_policy_stages(model)
    reference = model.get("reference_value_model")
    expected = model.get("training_behavior_value_sha256")
    if expected != value_policy_digest(reference):
        raise ValueError("confirmation reference differs from the frozen training behavior")
    return reference


def confirmation_acceptance(report, parent, candidate, minimum_improved_scenes=5):
    indexed = {(int(row["seed"]), int(row["case"])): row for row in parent}
    keys = [(int(row["seed"]), int(row["case"])) for row in candidate]
    if len(indexed) != len(parent) or len(set(keys)) != len(keys) or set(indexed) != set(keys):
        raise ValueError("confirmation acceptance requires identical unique paired scenes")
    human_deltas = np.asarray([
        float(row["human_velocity_variation"]) - float(indexed[key]["human_velocity_variation"])
        for row, key in zip(candidate, keys)
    ])
    if not np.isfinite(human_deltas).all() or len(human_deltas) < 2:
        raise ValueError("confirmation acceptance requires finite paired human-variation differences")
    changed_cases = sorted({
        key[1] for row, key in zip(candidate, keys)
        if int(float(row.get("executable_value_action_changes", 0))) > 0
    })
    added_stagnation_cases = sorted({
        key[1] for row, key in zip(candidate, keys)
        if float(row["stagnation_time"]) > float(indexed[key]["stagnation_time"]) + 1e-10
    })
    improved_human_scenes = int((human_deltas < -1e-10).sum())
    leave_best_out = float(np.delete(human_deltas, np.argmin(human_deltas)).mean())
    human_interval = report["metrics"]["human_velocity_variation"]["case_bootstrap_ci95"]
    return {
        "changed_action_scenes": changed_cases,
        "added_stagnation_cases": added_stagnation_cases,
        "improved_human_variation_scenes": improved_human_scenes,
        "worsened_human_variation_scenes": int((human_deltas > 1e-10).sum()),
        "leave_best_scene_out_human_variation_difference": leave_best_out,
        "primary_acceptance_passed": bool(
            report["observed_safety_not_worse"]
            and not added_stagnation_cases
            and report["metrics"]["goal_shortfall"]["candidate_minus_parent"] <= 1e-10
            and human_interval[1] < 0.
            and len(changed_cases) >= minimum_improved_scenes
            and improved_human_scenes >= minimum_improved_scenes
            and leave_best_out < 0.
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Freeze one executable value head before a paired untouched navigation confirmation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--value-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-start", type=int, required=True)
    parser.add_argument("--cases", type=int, required=True)
    parser.add_argument("--excluded-cases", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_path = Path(args.value_model).resolve()
    model = json.loads(model_path.read_text())
    cases, excluded = confirmation_cases(args.case_start, args.cases, model["case_splits"], args.excluded_cases)
    reference_model = confirmation_reference_model(model)
    reference_model_path = None
    if reference_model is not None:
        reference_model_path = output / "reference_value_model.json"
        reference_model_path.write_text(json.dumps(reference_model, indent=2, allow_nan=False) + "\n")
    protected = [Path(args.checkpoint).resolve(), Path(args.calibration).resolve(),
                 Path(args.source_metadata).resolve(), model_path]
    if reference_model_path is not None:
        protected.append(reference_model_path)
    protected += list(Path(__file__).resolve().parents[1].rglob("*.py"))
    hashes = {str(path): file_hash(path) for path in protected}
    selection = {
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cases": cases, "excluded_cases": sorted(excluded), "value_model": str(model_path),
        "reference_value_model": str(reference_model_path) if reference_model_path is not None else None,
        "reference_value_sha256": value_policy_digest(reference_model),
        "checkpoint": str(Path(args.checkpoint).resolve()), "source_hashes": hashes,
        "acceptance": {
            "no_added_failure_collision_or_timeout_scene": True,
            "no_added_stagnation_scene": True,
            "goal_shortfall_mean_not_worse": True,
            "human_variation_case_bootstrap_upper_below_zero": True,
            "minimum_changed_and_human_improved_scenes": 5,
            "human_variation_mean_negative_after_removing_best_scene": True,
            "elapsed_time_role": "secondary report only; cannot override a higher-priority failure",
        },
        "boundary": "One frozen new confirmation after previously disclosed development and failed response-ranking confirmations. No interim tuning or automatic promotion. Safety/task completion, stagnation, goal shortfall and human perturbation precede elapsed time.",
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    script = Path(__file__).with_name("eval_external_information_value.py")
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    parent_extra = (["--executable-value-model", str(reference_model_path)]
                    if reference_model_path is not None else [])
    for name, extra in (("parent", parent_extra),
                        ("candidate", ["--executable-value-model", str(model_path)])):
        command = [sys.executable, "-u", str(script), "--checkpoint", args.checkpoint,
                   "--calibration", args.calibration, "--source-metadata", args.source_metadata,
                   "--output-dir", str(output / name), "--confirmation-case-start", str(args.case_start),
                   "--confirmation-cases", str(args.cases), "--navigation-only", "--device", args.device, *extra]
        stream = (output / f"{name}.log").open("w")
        status = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT).returncode
        stream.close()
        if status:
            raise RuntimeError(f"confirmation {name} navigation failed: {status}")
    episodes = {}
    protocols = {}
    for name in ("parent", "candidate"):
        directory = output / name
        summary = json.loads((directory / "summary.json").read_text())
        protocol = json.loads((directory / "protocol.json").read_text())
        if summary["status"] != "COMPLETED" or summary["changed_required_sources"]:
            raise ValueError("confirmation inputs changed during navigation")
        if protocol["phase"] != "confirmation" or protocol["evaluation_cases"] != cases:
            raise ValueError("confirmation evaluation differs from the prespecified case block")
        protocols[name] = protocol
        episodes[name] = list(csv.DictReader((directory / "episodes.csv").open()))
        if len(episodes[name]) != len(cases) or sorted(int(row["case"]) for row in episodes[name]) != cases:
            raise ValueError("confirmation episode records must exactly cover every frozen case")
    permitted = {"output_dir", "belief_selection_audit", "planner_sensitivity_audit"}
    differences = [name for name in set(protocols["parent"]["parameters"]) | set(protocols["candidate"]["parameters"])
                   if name not in permitted and protocols["parent"]["parameters"].get(name) != protocols["candidate"]["parameters"].get(name)]
    if differences:
        raise ValueError(f"confirmation parent and candidate controllers differ: {differences}")
    report = assess_reliable_navigation(episodes["parent"], episodes["candidate"], minimum_scenes=args.cases)
    report.update(confirmation_acceptance(report, episodes["parent"], episodes["candidate"]))
    report["boundary"] = selection["boundary"]
    report["selection_sha256"] = file_hash(output / "selection.json")
    changed = [path for path, digest in hashes.items() if not Path(path).exists() or file_hash(path) != digest]
    report["status"] = "SOURCE_CHANGED" if changed else "COMPLETED"
    report["changed_protected_inputs"] = changed
    if changed:
        report["primary_acceptance_passed"] = False
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
