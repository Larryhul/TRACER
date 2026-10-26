import json
from types import SimpleNamespace

import numpy as np
import pytest

from sage.experiments.socialgym2_execution_diagnostics import (
    ExecutionDiagnostics,
    load_vector_map,
    observed_tick,
    point_segment_distances,
    sample_planned_prefix,
)
from sage.experiments.analyze_socialgym2_execution import analyze


def response(position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), collision=False, humans=()):
    return SimpleNamespace(
        robot_poses=np.asarray(position), robot_vels=np.asarray(velocity),
        goal_pose=np.array([2.0, 0.0, 0.0]), local_target=np.array([1.0, 0.0, 0.0]),
        human_poses=[np.asarray(human) for human in humans],
        human_vels=[np.zeros(3) for _ in humans],
        collision=collision, done=False, success=False, robot_state=0, follow_target=-1,
    )


def test_planned_prefix_uses_physical_time_and_ignores_unexecuted_suffix():
    initial = np.zeros((2, 2))
    planned = np.array([[[0.4, 0], [100, 0]], [[0, 0.8], [0, -100]]])
    positions, velocities = sample_planned_prefix(initial, planned, 0.1, 0.4)
    np.testing.assert_allclose(positions, [[0.1, 0], [0, 0.2]])
    np.testing.assert_allclose(velocities, [[1, 0], [0, 2]])
    changed = planned.copy()
    changed[:, 1] = 9999
    for first, second in zip(sample_planned_prefix(initial, planned, 0.4, 0.4), sample_planned_prefix(initial, changed, 0.4, 0.4)):
        np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError, match="beyond"):
        sample_planned_prefix(initial, planned, 1.0, 0.4)


def test_tick_rotates_velocities_and_retains_partial_observation_unknowns():
    value = response(position=(1, 2, np.pi / 2), velocity=(0.5, 0, 0.2), collision=True, humans=((0, 0, 0),))
    result = observed_tick([value], True, np.empty((0, 2, 2)))
    np.testing.assert_allclose(result["robot_world_velocities"], [[0, 0.5]], atol=1e-10)
    assert result["human_observations"] == []
    assert result["collisions"][0]["official_unclassified"]
    assert result["collisions"][0]["geometry_labels"] == []
    np.testing.assert_array_equal(value.robot_poses, [1, 2, np.pi / 2])


def test_tick_separates_robot_human_robot_robot_and_static_geometry():
    walls = np.array([[[0.2, -1], [0.2, 1]]])
    values = [response(collision=True, humans=((0, 0.3, 0),)), response(position=(-0.3, 0, 0), collision=True)]
    result = observed_tick(values, True, walls)
    assert set(result["collisions"][0]["geometry_labels"]) == {
        "robot_human_observed", "robot_robot", "robot_static_front_segment",
    }
    assert not result["collisions"][0]["official_unclassified"]
    assert result["collisions"][1]["geometry_labels"] == ["robot_robot"]


def test_geometric_overlap_does_not_overwrite_official_collision():
    values = [response(), response(position=(0.3, 0, 0))]
    result = observed_tick(values, True, np.empty((0, 2, 2)))
    assert result["collisions"][0]["geometry_labels"] == ["robot_robot"]
    assert result["collisions"][0]["official_collision"] is False


def test_vector_map_and_degenerate_segments(tmp_path):
    path = tmp_path / "map.txt"
    path.write_text("0, 0, 0, 0\n1, 0, 1, 1\n")
    walls = load_vector_map(path)
    np.testing.assert_allclose(point_segment_distances([[0, 1]], walls), [[1, 1]])
    path.write_text("0, 0, nan, 1\n")
    with pytest.raises(ValueError, match="finite"):
        load_vector_map(path)


def make_diagnostics(tmp_path):
    map_path = tmp_path / "map.txt"
    map_path.write_text("10, -1, 10, 1\n")
    diagnostics = ExecutionDiagnostics(tmp_path, map_path, 0.4, 0.025, True)
    diagnostics.begin_episode({"episode": 0, "policy": "tracer-persistent", "seed": 1, "scenario": "test", "initial_state_hash": "test"}, [response()], [])
    diagnostics.begin_window(0, 0.0, np.array([[[0.4, 0.0], [0.8, 0]]]), np.array([[[1.0, 0.0], [1, 0]]]), [0], [], {"fallback": False, "min_violation_probability": 0.0})
    return diagnostics


def test_short_window_uses_actual_elapsed_time_and_records_no_fabricated_update(tmp_path):
    diagnostics = make_diagnostics(tmp_path)
    diagnostics.record_tick(0.025, [response(position=(0.01, 0, 0), velocity=(0.4, 0, 0))])
    diagnostics.record_tick(0.05, [response(position=(0.02, 0, 0), velocity=(0.4, 0, 0))])
    diagnostics.end_window(0.05, [], [])
    diagnostics.end_episode({"team_success": 0.0, "initial_collision": 0.0, "collision": 0.0, "terminal_reached": False, "executed_sim_ticks": 2})
    diagnostics.close(True)
    records = [json.loads(line) for line in (tmp_path / "execution_diagnostics.jsonl").read_text().splitlines()]
    update = next(record for record in records if record["kind"] == "update")
    assert update["complete_model_step"] is False
    assert update["evidence_rows"] == []
    np.testing.assert_allclose(update["executed_mean_robot_velocity"], [[0.4, 0]])
    ticks = [record for record in records if record["kind"] == "tick"]
    np.testing.assert_allclose(ticks[-1]["position_error"], [0.03])
    manifest = json.loads((tmp_path / "execution_diagnostics_manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["counts"]["tick"] == 2
    analysis = analyze(tmp_path)
    assert analysis["episodes"][0]["short_windows"] == 1
    assert analysis["episodes"][0]["budget_exhausted_before_team_success"]
    assert analysis["robot_tick_error_strata_descriptive"]["go"]["position_error"]["count"] == 2
    with (tmp_path / "execution_diagnostics.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="manifest"):
        analyze(tmp_path)


def test_duplicate_tick_and_duplicate_evidence_are_rejected(tmp_path):
    diagnostics = make_diagnostics(tmp_path)
    diagnostics.record_tick(0.025, [response()])
    with pytest.raises(ValueError, match="duplicate"):
        diagnostics.record_tick(0.025, [response()])
    with pytest.raises(ValueError, match="more than once"):
        diagnostics.end_window(0.025, [], [{"track_id": 1}, {"track_id": 1}])
    diagnostics.close(False)
    manifest = json.loads((tmp_path / "execution_diagnostics_manifest.json").read_text())
    assert manifest["status"] == "INTERRUPTED"


def test_unfinished_stream_cannot_claim_completion(tmp_path):
    diagnostics = make_diagnostics(tmp_path)
    with pytest.raises(ValueError, match="unfinished"):
        diagnostics.close(True)


def test_waypoint_angle_uses_robot_local_frame(tmp_path):
    map_path = tmp_path / "map.txt"
    map_path.write_text("10, -1, 10, 1\n")
    diagnostics = ExecutionDiagnostics(tmp_path, map_path, 0.4, 0.025, True)
    value = response(position=(5, 5, np.pi / 2))
    value.goal_pose = np.array([5, 8, 0])
    value.local_target = np.array([1, 0, 0])
    diagnostics.begin_episode({"episode": 0}, [value], [])
    diagnostics.begin_window(0, 0, np.array([[[5, 5.4]]]), np.array([[[0, 1]]]), [0], [], {"fallback": False, "min_violation_probability": 0})
    diagnostics.record_tick(0.025, [value])
    diagnostics.end_window(0.025, [], [])
    diagnostics.end_episode({})
    diagnostics.close(True)
    rows = [json.loads(line) for line in (tmp_path / "execution_diagnostics.jsonl").read_text().splitlines()]
    tick = next(row for row in rows if row["kind"] == "tick")
    np.testing.assert_allclose(tick["waypoint_goal_angle"], [0], atol=1e-10)
    assert tick["waypoint_target_valid"] == [True]
