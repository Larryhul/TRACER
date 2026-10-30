import csv
import json

import pytest

from tracer.experiments.compare_navigation_validation import METRICS, compare


def fixtures(tmp_path, missing=False, duplicate=False):
    official, tracer = tmp_path / "official", tmp_path / "tracer"
    official.mkdir()
    tracer.mkdir()
    parameters = {"max_steps": 60, "windows": 60, "execute_steps": 1}
    metadata = {"parameters": parameters, "external_repository": {"commit": "same"},
                "trajectory_contract": {"motion_policy_sha256": "frozen"},
                "response_contract": {"mode_semantics": "test"}}
    for directory in (official, tracer):
        (directory / "metadata.json").write_text(json.dumps(metadata))
    baseline, observed = [], []
    for case in (1, 2):
        common = {**{metric: 1. for metric in METRICS}, "case": case,
                  "initial_scene_hash": str(case), "environment_dt": .25,
                  "robot_kinematics": "unicycle", "robot_initial_heading": 0.}
        baseline.append({**common, "policy": "sicnav-np", "seed": 0})
        for seed in (2027, 2031):
            if not (missing and case == 2 and seed == 2031):
                observed.append({**common, "policy": "tracer-persistent", "seed": seed})
            observed.append({**common, "policy": "tracer-reset", "seed": seed, "success": 0.})
    if duplicate:
        observed.append(observed[0])
    for directory, rows in ((official, baseline), (tracer, observed)):
        with (directory / "episode_metrics.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    return official, tracer


def test_comparison_never_pools_different_belief_policies(tmp_path):
    official, tracer = fixtures(tmp_path)
    result = compare(official, tracer)
    assert result["metrics"]["success"]["tracer"] == 1.
    assert not result["noninferiority_established"]
    reset = compare(official, tracer, tracer_policy="tracer-reset")
    assert reset["metrics"]["success"]["tracer"] == 0.


@pytest.mark.parametrize("missing,duplicate", [(True, False), (False, True)])
def test_comparison_requires_complete_case_seed_matrix(tmp_path, missing, duplicate):
    official, tracer = fixtures(tmp_path, missing, duplicate)
    with pytest.raises(ValueError, match="case-by-seed"):
        compare(official, tracer)
