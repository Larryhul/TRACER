import importlib.util
import csv
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "baseline_recovery", Path(__file__).resolve().parents[1] / "scripts/resume_external_baseline_comparison.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def episode(policy, case=88, seed=None):
    row = {
        "policy": policy, "case": str(case), "initial_scene_hash": "same-scene",
        "environment_dt": "0.25", "robot_initial_heading": "1.57",
        "robot_kinematics": "unicycle", "success": "1", "collision_free_success": "1",
        "collision": "0", "wall_collision": "0", "timeout": "0", "task_progress": "0.9",
        "total_time": "4", "min_distance": "0.6",
    }
    if seed is not None:
        row["seed"] = str(seed)
    return row


def validate(tracer, official):
    return MODULE.paired_grid(tracer, official, (88,), (2027, 2031), ("tracer-persistent",), ("sicnav-p",))


def test_complete_grid_is_case_paired():
    tracer = [episode("tracer-persistent", seed=seed) for seed in (2027, 2031)]
    actual, official = validate(tracer, [episode("sicnav-p")])
    assert len(actual) == 2
    assert len(official) == 1


@pytest.mark.parametrize("field,value", [
    ("initial_scene_hash", "different"), ("environment_dt", "0.4"),
    ("robot_initial_heading", "0"), ("task_progress", "nan"), ("collision", "1"),
])
def test_mismatched_or_invalid_episode_rejected(field, value):
    tracer = [episode("tracer-persistent", seed=seed) for seed in (2027, 2031)]
    tracer[0][field] = value
    with pytest.raises(ValueError):
        validate(tracer, [episode("sicnav-p")])


def test_missing_or_duplicate_seed_rejected():
    row = episode("tracer-persistent", seed=2027)
    with pytest.raises(ValueError, match="Missing"):
        validate([row], [episode("sicnav-p")])
    with pytest.raises(ValueError, match="Duplicate"):
        validate([row, row], [episode("sicnav-p")])


def test_bootstrap_counts_scenes_not_seed_rows():
    result = MODULE.bootstrap([0.1, 0.2], samples=1000)
    assert result["case_count"] == 2
    assert result["mean"] == pytest.approx(0.15)


def test_holonomic_reference_not_rejected_as_scene_mismatch():
    tracer = [episode("tracer-persistent", seed=seed) for seed in (2027, 2031)]
    reference = episode("sicnav-p")
    reference["robot_kinematics"] = "holonomic"
    tracer[0]["robot_kinematics"] = "holonomic"
    tracer[0]["robot_initial_heading"] = "0"
    reference["robot_initial_heading"] = "0"
    tracer[1]["robot_kinematics"] = "holonomic"
    tracer[1]["robot_initial_heading"] = "0"
    validate(tracer, [reference])


def test_holonomic_heading_is_not_compared_to_unicycle():
    tracer = [episode("tracer-persistent", seed=seed) for seed in (2027, 2031)]
    reference = episode("orca")
    reference["robot_kinematics"] = "holonomic"
    reference["robot_initial_heading"] = "0"
    MODULE.paired_grid(tracer, [episode("sicnav-p"), reference], (88,), (2027, 2031),
                       ("tracer-persistent",), ("sicnav-p", "orca"))


def test_archived_control_parameters_are_preserved(tmp_path):
    original = {"static_route_weight": 2.0, "swept_human_guard": False,
                "align_route_before_motion": True, "belief_calibration_path": "seed{seed}/calibration.json",
                "human_cost_weight": 0.6, "risk_weight": 0.5}
    restored = MODULE.controller_parameters({"parameters": original}, tmp_path)
    assert all(restored[name] == value for name, value in original.items())
    assert restored["policies"] == ["tracer-persistent"]
    assert restored["belief_update_stride"] == 1


def make_run(directory, rows):
    directory.mkdir(parents=True)
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with (directory / "episode_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"external_repository": {"commit": "frozen"}, "parameters": {"max_steps": 60, "case_start": 88}}
    MODULE.write_json(directory / "metadata.json", metadata)
    MODULE.write_json(directory / "manifest.json", {
        "status": "COMPLETED", "artifacts": {name: MODULE.digest(directory / name) for name in ("episode_metrics.csv", "metadata.json")},
    })


def test_full_summary_uses_case_average_and_separate_kinematics(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "SEEDS", (2027, 2031))
    monkeypatch.setattr(MODULE, "CASES", (88, 89))
    monkeypatch.setattr(MODULE, "OFFICIAL_POLICIES", ("sicnav-p", "official-orca"))
    tracer = [episode("tracer-persistent", case=case, seed=seed) for case in (88, 89) for seed in (2027, 2031)]
    tracer[0]["total_time"] = "6"
    official = [episode(policy, case=case) for case in (88, 89) for policy in ("sicnav-p", "official-orca")]
    for row in official:
        if row["policy"] == "official-orca":
            row["robot_kinematics"] = "holonomic"
            row["robot_initial_heading"] = "0"
    make_run(tmp_path / "tracer", tracer)
    make_run(tmp_path / "official", official)
    MODULE.write_json(tmp_path / "selection.json", {"test": True})
    result = MODULE.summarize(tmp_path, tmp_path / "official")
    assert result["independent_scenes"] == 2
    assert result["tracer_records"] == 4
    assert result["policies"]["official-orca"]["comparison_role"] == "holonomic reference"
    contrast = result["persistent_minus_official"]["sicnav-p"]["total_time"]
    assert contrast["mean"] == pytest.approx(0.5)
    assert contrast["case_count"] == 2
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "COMPLETED"


def test_artifact_change_rejected(tmp_path):
    make_run(tmp_path / "official", [episode("sicnav-p")])
    (tmp_path / "official/episode_metrics.csv").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        MODULE.verify_artifacts(tmp_path / "official")
