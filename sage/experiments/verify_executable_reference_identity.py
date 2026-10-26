from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sage.experiments.collect_external_planner_supervision import file_hash
from sage.experiments.summarize_executable_value_navigation import paired_navigation


def verify_reference_identity(parent, candidate):
    indexed = {(int(row["seed"]), int(row["case"])): row for row in parent}
    candidate_keys = [(int(row["seed"]), int(row["case"])) for row in candidate]
    if len(indexed) != len(parent) or len(set(candidate_keys)) != len(candidate_keys):
        raise ValueError("identity audit requires unique parent and candidate scenes")
    try:
        paired_parent = [indexed[key] for key in candidate_keys]
    except KeyError as error:
        raise ValueError("identity audit candidate has no matching parent scene") from error
    report = paired_navigation(paired_parent, candidate)
    nonzero = {
        name: values["candidate_minus_parent"]
        for name, values in report["metrics"].items()
        if abs(values["candidate_minus_parent"]) > 1e-10
    }
    if nonzero:
        raise ValueError(f"zero refinement changed navigation outcomes: {nonzero}")
    for parent_row, candidate_row in zip(paired_parent, candidate):
        parent_details = json.loads(parent_row["executable_value_stage_action_change_details"])
        candidate_details = json.loads(candidate_row["executable_value_stage_action_change_details"])
        if len(parent_details) != 1 or len(candidate_details) != 2:
            raise ValueError("identity audit requires one reference stage and one added refinement stage")
        if parent_details[0] != candidate_details[0]:
            raise ValueError("zero refinement changed the frozen behavior action sequence")
        if candidate_details[1] or int(candidate_row["executable_value_action_changes"]) != 0:
            raise ValueError("zero refinement must abstain from every action change")
    report.update({
        "status": "COMPLETED",
        "reference_action_sequences_identical": True,
        "refinement_action_changes": 0,
        "scope": "Executable contract identity audit on disclosed development scenes; not performance evidence.",
    })
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-csv", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(args.parent_csv).resolve(), Path(args.candidate_csv).resolve()]
    parent = list(csv.DictReader(paths[0].open()))
    candidate = list(csv.DictReader(paths[1].open()))
    report = verify_reference_identity(parent, candidate)
    report["input_hashes"] = {str(path): file_hash(path) for path in paths}
    with Path(args.output).resolve().open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
