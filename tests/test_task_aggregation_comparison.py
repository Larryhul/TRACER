import copy
import json

import pytest

from scripts.run_final_socialgym2_confirmation import _sha256, _snapshot_source, _verify_snapshot
from scripts.run_socialgym2_task_aggregation_comparison import adoption_decision, paired_metrics, validated_profile, verify_run


def episode():
    return {
        "metrics": {
            "seed": 2861, "scenario": "open", "episode": 0,
            "initial_state_hash": "same-state", "initial_scene_snapshot": "same-scene",
            "scenario_hash": "same-scenario", "initial_collision": 0,
            "team_success": 0, "collision_free_team_success": 0, "collision": 0,
            "task_progress": 0.7, "stagnation_window_rate": 0.2, "robot_success_fraction": 0.5,
            "belief_action_change_rate": 0, "fallback_rate": 0,
        },
        "budget_exhausted_before_team_success": True, "robot_halt_fraction": 0.3,
    }


@pytest.mark.parametrize("field", ["initial_state_hash", "initial_scene_snapshot", "scenario_hash", "initial_collision"])
def test_comparison_rejects_unmatched_scene(field):
    baseline = episode()
    candidate = copy.deepcopy(baseline)
    candidate["metrics"][field] = "different"
    with pytest.raises(ValueError, match="unmatched"):
        paired_metrics([baseline], [candidate], 1)


def test_comparison_rejects_partial_or_duplicate_data():
    with pytest.raises(ValueError, match="complete"):
        paired_metrics([episode()], [], 1)
    with pytest.raises(ValueError, match="complete"):
        paired_metrics([], [], 0)
    with pytest.raises(ValueError, match="duplicate"):
        paired_metrics([episode(), episode()], [episode()], 1)


def test_comparison_reports_gains_and_new_collisions_separately():
    baseline, candidate = episode(), episode()
    candidate["metrics"].update(team_success=1, collision=1)
    result = paired_metrics([baseline], [candidate], 1)
    assert result["metrics"]["team_success"]["delta"] == 1
    assert result["metrics"]["collision_free_team_success"]["delta"] == 0
    assert result["new_collision_scenes"] == [[2861, "open", 0]]
    assert result["initially_clean_collision_counts"] == {"baseline": 0, "candidate": 1}


def test_verifier_rejects_incomplete_run(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "FAILED", "records": 8}))
    with pytest.raises(ValueError, match="incomplete"):
        verify_run(tmp_path, 8)


def test_verifier_rejects_changed_artifact(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "status": "COMPLETED", "records": 8, "artifacts": {"metadata.json": "wrong-hash"},
    }))
    (tmp_path / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_run(tmp_path, 8)


def test_source_snapshot_is_isolated_from_concurrent_edits(tmp_path):
    origin = tmp_path / "origin"
    (origin / "sage").mkdir(parents=True)
    (origin / "sage/policy.py").write_text("value = 1\n")
    frozen = tmp_path / "frozen"
    manifest = _snapshot_source(origin, frozen)
    (origin / "sage/policy.py").write_text("value = 2\n")
    _verify_snapshot(frozen, manifest)
    assert (frozen / "sage/policy.py").read_text() == "value = 1\n"
    (frozen / "sage/policy.py").write_text("value = 3\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _verify_snapshot(frozen, manifest)


def test_adoption_requires_both_maps_and_collision_free_team_gain():
    baseline, candidate = episode(), episode()
    result = paired_metrics([baseline], [candidate], 1)
    assert not adoption_decision({"open": result})["adopt"]
    assert not adoption_decision({"open": result, "intersection": result})["adopt"]
    candidate["metrics"].update(team_success=1, collision_free_team_success=1)
    improved = paired_metrics([baseline], [candidate], 1)
    assert adoption_decision({"open": improved, "intersection": result})["adopt"]
    improved["new_collision_scenes"] = [[2861, "open", 0]]
    assert not adoption_decision({"open": improved, "intersection": result})["adopt"]


def test_adoption_does_not_hide_one_map_regression_in_pooled_gain():
    baseline, candidate = episode(), episode()
    unchanged = paired_metrics([baseline], [candidate], 1)
    candidate["metrics"].update(team_success=1, collision_free_team_success=1)
    improved = paired_metrics([baseline], [candidate], 1)
    regressed = copy.deepcopy(unchanged)
    regressed["metrics"]["task_progress"]["delta"] = -0.02
    assert not adoption_decision({"open": improved, "intersection": regressed})["adopt"]
    regressed = copy.deepcopy(unchanged)
    regressed["metrics"]["collision_free_team_success"]["delta"] = -0.1
    assert not adoption_decision({"open": improved, "intersection": regressed})["adopt"]


def test_profile_requires_frozen_sources_and_passing_evidence(tmp_path):
    origin = tmp_path / "origin"
    (origin / "sage").mkdir(parents=True)
    (origin / "sage/policy.py").write_text("value = 1\n")
    frozen = tmp_path / "frozen"
    _snapshot_source(origin, frozen)
    comparison_path = tmp_path / "comparison.json"
    baseline, candidate = episode(), episode()
    candidate["metrics"].update(team_success=1, collision_free_team_success=1)
    result = paired_metrics([baseline], [candidate], 1)
    comparison_path.write_text(json.dumps({"open": result, "intersection": result}))
    profile = {
        "schema": "socialgym2-selected-navigation-profile-v1",
        "candidate": "lower_tail25_release", "frozen_source": str(frozen),
        "source_manifest_sha256": _sha256(frozen / "source_manifest.json"),
        "comparison": str(comparison_path), "comparison_sha256": _sha256(comparison_path),
        "input_hashes": {"response": "same-input"},
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    (tmp_path / "protocol.json").write_text(json.dumps({
        "source_manifest_sha256": profile["source_manifest_sha256"],
        "input_hashes": profile["input_hashes"], "conditions": {"lower_tail25_release": {}},
    }))
    assert validated_profile(profile_path) == profile
    comparison_path.write_text("{}")
    with pytest.raises(ValueError, match="comparison changed"):
        validated_profile(profile_path)
