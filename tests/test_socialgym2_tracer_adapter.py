from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from sage.experiments.run_socialgym2_tracer_loop import (
    HumanTrack,
    _action_lattice,
    _candidate_geometry,
    _episode_seed,
    _executed_socialgym_action,
    _external_source_hashes,
    _pose2df,
    _executed_candidate_geometry,
    _initial_state_hash,
    _project_sage_order,
    _response_predictions,
    _scene_from_responses,
    _track_history,
    _update_tracks,
)


def test_socialgym2_response_uses_all_robots_and_pair_interactions() -> None:
    calls: list[bool] = []

    class SpyModel:
        config = SimpleNamespace(num_modes=3, horizon=2)

        def __call__(self, history, current, candidates, *, include_pair_interactions, robot_mask):
            calls.append(bool(include_pair_interactions))
            batch = history.shape[0]
            return {
                "mean": torch.zeros((batch, 1, 3, 2, 2), dtype=torch.float32),
                "log_std": torch.zeros((batch, 1, 3, 2, 2), dtype=torch.float32),
            }

    tracks = [
        HumanTrack(
            track_id=0,
            position=np.zeros(2, dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            history=[np.zeros(2, dtype=np.float32) for _ in range(2)],
        )
    ]
    robot_positions = np.zeros((3, 2), dtype=np.float32)
    robot_velocities = np.zeros((3, 2), dtype=np.float32)
    candidate_positions = np.zeros((4, 3, 2, 2), dtype=np.float32)
    candidate_velocities = np.zeros_like(candidate_positions)

    means, log_stds, references, current = _response_predictions(
        SpyModel(),
        tracks,
        robot_positions,
        robot_velocities,
        candidate_positions,
        candidate_velocities,
        history=2,
        dt=0.4,
        device="cpu",
    )

    assert calls == [True, True]
    assert means.shape == (1, 4, 3, 2, 2)
    assert log_stds.shape == means.shape
    assert references.shape == means.shape
    assert current.shape == (1, 4, 3, 4)


def test_socialgym2_action_lattice_is_exhaustive_and_stable() -> None:
    lattice = _action_lattice(3)
    assert lattice.shape == (8, 3)
    assert {tuple(row) for row in lattice} == {
        (a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)
    }
    order = _project_sage_order(np.zeros((8, 3), dtype=np.float32), lattice)
    assert sorted(order.tolist()) == list(range(8))


def test_socialgym2_projected_candidates_have_fixed_kinematics() -> None:
    positions = np.zeros((3, 2), dtype=np.float32)
    velocities = np.zeros((3, 2), dtype=np.float32)
    goals = np.ones((3, 2), dtype=np.float32)
    positions_out, velocities_out = _candidate_geometry(
        positions, velocities, goals, _action_lattice(3), 12, 0.4
    )
    assert positions_out.shape == (8, 3, 12, 2)
    assert velocities_out.shape == positions_out.shape
    halt = np.all(_action_lattice(3) == 1, axis=1).nonzero()[0][0]
    assert np.allclose(velocities_out[halt], 0.0)


def test_socialgym2_tracker_preserves_identity_through_missed_observation() -> None:
    tracks: list[HumanTrack] = []
    next_id = _update_tracks(
        tracks,
        [(np.array([0.0, 0.0], dtype=np.float32), np.array([0.2, 0.0], dtype=np.float32))],
        history=8,
        dt=0.4,
        next_id=0,
        max_missed=3,
    )
    assert next_id == 1
    track_id = tracks[0].track_id
    _update_tracks(tracks, [], 8, 0.4, next_id, 3)
    assert tracks[0].track_id == track_id
    assert tracks[0].missed == 1
    assert _track_history(tracks[0], 8, 0.4).shape == (8, 2)


def test_socialgym2_response_is_converted_from_robot_to_world_frame() -> None:
    response = SimpleNamespace(
        robot_poses=np.array([1.0, 2.0, np.pi / 2], dtype=np.float32),
        robot_vels=np.zeros(3, dtype=np.float32),
        goal_pose=np.array([4.0, 2.0, 0.0], dtype=np.float32),
        human_poses=[np.array([1.0, 0.0, 0.0], dtype=np.float32), np.zeros(3, dtype=np.float32)],
        human_vels=[np.array([0.5, 0.0, 0.0], dtype=np.float32), np.zeros(3, dtype=np.float32)],
    )
    robots, _, _, observations = _scene_from_responses([response], partial_observation=True)
    assert np.allclose(robots[0], [1.0, 2.0])
    assert len(observations) == 1
    assert np.allclose(observations[0][0], [1.0, 3.0], atol=1e-6)
    assert np.allclose(observations[0][1], [0.0, 0.5], atol=1e-6)


def test_socialgym2_initial_state_hash_is_order_invariant_for_humans() -> None:
    robots = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    velocities = np.zeros_like(robots)
    goals = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
    observations = [
        (np.array([2.0, 2.0], dtype=np.float32), np.array([0.1, 0.0], dtype=np.float32)),
        (np.array([1.0, 1.0], dtype=np.float32), np.array([0.0, 0.1], dtype=np.float32)),
    ]
    first = _initial_state_hash(robots, velocities, goals, observations)
    second = _initial_state_hash(robots, velocities, goals, list(reversed(observations)))
    changed = _initial_state_hash(robots, velocities, goals + 0.01, observations)
    assert first == second
    assert first != changed


def test_socialgym2_episode_seed_stream_is_disjoint() -> None:
    assert _episode_seed(2027, 0) == 2027000
    assert _episode_seed(2027, 15) != _episode_seed(2028, 0)


def test_socialgym2_external_action_projection_is_explicit() -> None:
    from sage.experiments.run_socialgym2_tracer_loop import EXTERNAL_ACTION_TO_BINARY

    assert EXTERNAL_ACTION_TO_BINARY == {0: 0, 1: 1, 2: 0, 3: 0}

    class Pose:
        def __init__(self) -> None:
            self.x = None
            self.y = None
            self.theta = None

    message = _pose2df(np.asarray([1.0, -2.0, 0.25], dtype=np.float32), Pose)
    assert (message.x, message.y, message.theta) == (1.0, -2.0, 0.25)
    assert _external_source_hashes(SimpleNamespace(), None) == {}


def test_socialgym2_external_action_is_not_overwritten_by_binary_lattice() -> None:
    ordered_actions = _action_lattice(3)
    external_action = np.asarray([0, 2, 3], dtype=np.int64)

    executed = _executed_socialgym_action(
        "social-ref", external_action, ordered_actions, selected=-1
    )

    assert np.array_equal(executed, external_action)
    assert np.array_equal(
        _executed_socialgym_action("tracer-persistent", None, ordered_actions, selected=2),
        ordered_actions[2],
    )


def test_socialgym2_executed_geometry_uses_observed_transition() -> None:
    positions = np.zeros((2, 2), dtype=np.float32)
    next_positions = np.asarray([[0.4, 0.0], [0.0, -0.2]], dtype=np.float32)
    candidate_positions, candidate_velocities = _executed_candidate_geometry(
        positions, next_positions, horizon=3, dt=0.4
    )
    assert candidate_positions.shape == (2, 3, 2)
    assert candidate_velocities.shape == candidate_positions.shape
    assert np.allclose(candidate_positions[:, 0], next_positions)
    assert np.allclose(candidate_velocities[:, 0], next_positions / 0.4)
