import numpy as np
import pytest

from sage.experiments.socialgym2_joint_geometry import (
    add_observed_human_clearances,
    constant_twist_prefix,
    executable_candidate_prefix,
    joint_prefix_clearances,
    observable_route_directions,
    select_with_joint_geometry,
    stagnation_adjusted_costs,
    stagnation_recovery_mask,
)


def test_constant_twist_tracks_body_frame_rotation_and_halt():
    poses = np.array([[1, 2, np.pi / 2], [0, 0, 0]])
    velocities = np.array([[1, 0, 0], [1, 0, 1]])
    positions, angles = constant_twist_prefix(poses, velocities, [[0, 0], [1, 1]], 4, 0.1)
    np.testing.assert_allclose(positions[0, 0, -1], [1, 2.4])
    np.testing.assert_allclose(positions[0, 1, -1], [np.sin(0.4), 1 - np.cos(0.4)])
    np.testing.assert_allclose(angles[0, 1, -1], 0.4)
    np.testing.assert_allclose(positions[1], np.broadcast_to(poses[:, None, :2], (2, 4, 2)))


def test_constant_twist_never_invents_nominal_speed_at_rest():
    positions, _ = constant_twist_prefix(np.zeros((3, 3)), np.zeros((3, 3)), [[0, 0, 0]], 16, 0.025)
    np.testing.assert_array_equal(positions, 0)


def test_executable_prefix_models_observed_go_startup_and_immediate_halt():
    positions, velocities = executable_candidate_prefix(
        robot_positions=np.zeros((1, 2)),
        robot_headings=np.zeros(1),
        robot_world_velocities=np.zeros((1, 2)),
        robot_angular_velocities=np.zeros(1),
        local_targets=np.array([[2.0, 0.0]]),
        goals=np.array([[3.0, 0.0]]),
        actions=np.array([[0], [1]]),
        horizon=3,
        dt=0.4,
    )
    np.testing.assert_allclose(positions[0, 0, 0], [0.39375, 0.0], atol=1e-6)
    np.testing.assert_allclose(velocities[0, 0, 0], [0.984375, 0.0], atol=1e-6)
    assert positions[0, 0, -1, 0] > 0.0
    np.testing.assert_array_equal(positions[1], 0.0)


def test_executable_prefix_matches_tick_rate_startup_acceleration():
    positions, _ = executable_candidate_prefix(
        robot_positions=np.zeros((1, 2)),
        robot_headings=np.zeros(1),
        robot_world_velocities=np.zeros((1, 2)),
        robot_angular_velocities=np.zeros(1),
        local_targets=np.array([[1.5, 0.0]]),
        goals=np.array([[3.0, 0.0]]),
        actions=np.array([[0]]),
        horizon=16,
        dt=0.025,
    )
    np.testing.assert_allclose(positions[0, 0, -1], [0.39375, 0.0], atol=1e-6)


def test_executable_prefix_first_step_matches_measured_twist_with_turning():
    positions, _ = executable_candidate_prefix(
        robot_positions=np.array([[1.0, 2.0]]),
        robot_headings=np.array([np.pi / 2]),
        robot_world_velocities=np.array([[1.0, 0.0]]),
        robot_angular_velocities=np.array([1.0]),
        local_targets=np.array([[0.0, 3.0]]),
        goals=np.array([[3.0, 2.0]]),
        actions=np.array([[0]]),
        horizon=2,
        dt=0.1,
    )
    expected = constant_twist_prefix(
        np.array([[1.0, 2.0, np.pi / 2]]),
        np.array([[0.0, -1.0, 1.0]]),
        [[0]],
        1,
        0.1,
    )[0][0, 0, 0]
    np.testing.assert_allclose(positions[0, 0, 0], expected, atol=1e-6)


def test_executable_prefix_measured_duration_is_resolution_invariant():
    common = dict(
        robot_positions=np.array([[1.0, 2.0]]),
        robot_headings=np.array([0.4]),
        robot_world_velocities=np.array([[0.8, -0.2]]),
        robot_angular_velocities=np.array([0.7]),
        local_targets=np.array([[1.5, 0.3]]),
        goals=np.array([[4.0, 3.0]]),
        actions=np.array([[0]]),
        measured_prefix_duration=0.4,
    )
    coarse, _ = executable_candidate_prefix(horizon=1, dt=0.4, **common)
    fine, _ = executable_candidate_prefix(horizon=16, dt=0.025, **common)
    np.testing.assert_allclose(fine[0, 0, -1], coarse[0, 0, 0], atol=1e-6)


def test_swept_relative_motion_detects_collision_between_ticks():
    initial = np.array([[-1, 0], [1, 0]])
    candidates = np.array([[[[1, 0]], [[-1, 0]]]])
    result = joint_prefix_clearances(initial, candidates, np.empty((0, 2, 2)))
    np.testing.assert_allclose(result["robot_clearance"], [-0.4])
    assert not result["feasible"][0]
    assert result["terminal_robot_clearance"][0] > 0


def test_static_sweep_detects_thin_wall_crossing():
    initial = np.array([[-1, 0]])
    candidates = np.array([[[[1, 0]]]])
    walls = np.array([[[0, -1], [0, 1]]])
    result = joint_prefix_clearances(initial, candidates, walls)
    assert result["static_clearance"][0] < 0
    assert not result["feasible"][0]


def test_geometry_does_not_assume_all_halt_is_safe():
    poses = np.array([[0, 0, 0], [0.3, 0, 0]])
    candidates, _ = constant_twist_prefix(poses, np.zeros_like(poses), [[1, 1]], 16, 0.025)
    result = joint_prefix_clearances(poses[:, :2], candidates, np.empty((0, 2, 2)))
    assert not result["feasible"][0]


def test_initial_robot_overlap_admits_only_net_separation_recovery():
    initial = np.array([[0.0, 0.0], [0.38, 0.0]])
    candidates = np.array([
        [[[0.0, 0.0]], [[0.48, 0.0]]],
        [[[0.0, 0.0]], [[0.38, 0.0]]],
        [[[0.1, 0.0]], [[0.38, 0.0]]],
    ])
    result = joint_prefix_clearances(initial, candidates, np.empty((0, 2, 2)))
    np.testing.assert_array_equal(result["feasible"], [True, False, False])
    np.testing.assert_array_equal(result["recovering_initial_overlap"], [True, False, False])


def test_margin_only_overlap_allows_nonworsening_halt_or_separation():
    initial = np.array([[0.0, 0.0], [0.45, 0.0]])
    candidates = np.array([
        [[[0.0, 0.0]], [[0.45, 0.0]]],
        [[[0.0, 0.0]], [[0.55, 0.0]]],
        [[[0.1, 0.0]], [[0.45, 0.0]]],
    ])
    result = joint_prefix_clearances(
        initial,
        candidates,
        np.empty((0, 2, 2)),
        robot_separation=0.5,
        hard_robot_separation=0.4,
    )
    np.testing.assert_array_equal(result["feasible"], [True, True, False])


def test_observed_human_sweep_rejects_crossing_go_and_keeps_halt():
    initial = np.array([[0.0, 0.0]])
    candidates = np.array([
        [[[0.3, 0.0]]],
        [[[0.0, 0.0]]],
    ])
    geometry = joint_prefix_clearances(initial, candidates, np.empty((0, 2, 2)))
    result = add_observed_human_clearances(
        geometry,
        initial,
        candidates,
        human_positions=[[0.6, 0.0]],
        human_velocities=[[-0.25, 0.0]],
        dt=0.4,
    )
    np.testing.assert_array_equal(result["human_feasible"], [False, True])
    np.testing.assert_array_equal(result["feasible"], [False, True])


def test_observed_human_margin_overlap_requires_nonworsening_motion():
    initial = np.array([[0.0, 0.0]])
    candidates = np.array([
        [[[0.1, 0.0]]],
        [[[-0.1, 0.0]]],
        [[[0.0, 0.0]]],
    ])
    geometry = joint_prefix_clearances(initial, candidates, np.empty((0, 2, 2)))
    result = add_observed_human_clearances(
        geometry,
        initial,
        candidates,
        human_positions=[[0.45, 0.0]],
        human_velocities=[[0.0, 0.0]],
        dt=0.4,
        robot_human_separation=0.5,
        hard_robot_human_separation=0.4,
    )
    np.testing.assert_array_equal(result["human_feasible"], [False, True, True])


def test_stagnation_penalty_only_discourages_halt_for_stalled_robots():
    result = stagnation_adjusted_costs(
        [0.0, 0.1, 0.2],
        [[1, 0], [0, 1], [0, 0]],
        [True, False],
        1.0,
    )
    np.testing.assert_allclose(result, [1.0, 0.1, 0.2])


def test_tiny_local_target_falls_back_to_stable_goal_direction():
    result = observable_route_directions(
        [[0.0, 0.0]],
        [np.pi / 2],
        [[0.001, -0.001]],
        [[2.0, 0.0]],
    )
    np.testing.assert_allclose(result, [[1.0, 0.0]])


def test_joint_geometry_is_translation_invariant():
    poses = np.array([[0, 0, 0], [1, 1, 0]])
    velocities = np.array([[1, 0, 0], [0.1, 0, 0]])
    candidates, _ = constant_twist_prefix(poses, velocities, [[0, 0], [0, 1]], 16, 0.025)
    walls = np.array([[[3, 0], [3, 1]]])
    original = joint_prefix_clearances(poses[:, :2], candidates, walls)
    moved = joint_prefix_clearances(poses[:, :2] + 10, candidates + 10, walls + 10)
    for key in original:
        np.testing.assert_allclose(original[key], moved[key])


def test_invalid_joint_action_cannot_create_motion():
    with pytest.raises(ValueError, match="GoAlone/Halt"):
        constant_twist_prefix(np.zeros((2, 3)), np.zeros((2, 3)), [[0, 2]], 16, 0.025)


def test_geometric_filter_keeps_response_cost_ranking_inside_shared_safe_set():
    geometry = {"feasible": np.array([False, True, True])}
    selected, fallback = select_with_joint_geometry([-10, 1, 2], [True, True, True], [0, 0, 0], geometry)
    assert (selected, fallback) == (1, False)
    selected, fallback = select_with_joint_geometry([-10, 1, 2], [True, False, True], [0, 0.2, 0], geometry)
    assert (selected, fallback) == (2, False)


def test_robust_fallback_prefers_physically_safe_all_halt():
    geometry = {
        "feasible": np.array([False, False]),
        "hard_feasible": np.array([False, True]),
        "robot_clearance": np.array([-0.2, -0.05]),
        "static_clearance": np.array([1.0, 1.0]),
        "terminal_robot_clearance": np.array([0.3, -0.05]),
    }
    selected, fallback = select_with_joint_geometry(
        [0.0, 0.2], [True, True], [0.0, 0.0], geometry, [[0, 0], [1, 1]],
    )
    assert (selected, fallback) == (1, True)


def test_stagnation_recovery_only_admits_hard_safe_nonworsening_go():
    geometry = {
        "feasible": np.array([False, True, False]),
        "hard_feasible": np.array([True, True, True]),
        "initial_robot_clearance": np.array([-0.05, -0.05, -0.05]),
        "terminal_robot_clearance": np.array([0.01, -0.05, -0.08]),
        "initial_static_clearance": np.array([0.1, 0.1, 0.1]),
        "terminal_static_clearance": np.array([0.1, 0.1, 0.1]),
    }
    actions = np.array([[0, 1], [1, 1], [0, 1]])
    np.testing.assert_array_equal(
        stagnation_recovery_mask(geometry, actions, [True, False]),
        [True, False, False],
    )
    selected, fallback = select_with_joint_geometry(
        [0.2, 0.0, -1.0],
        [True, True, True],
        [0.0, 0.0, 0.0],
        geometry,
        actions,
        [True, False],
    )
    assert (selected, fallback) == (0, False)


def test_stagnation_recovery_can_release_hard_safe_approach_explicitly():
    geometry = {
        "feasible": np.array([False, True]),
        "hard_feasible": np.array([True, True]),
        "initial_robot_clearance": np.array([0.2, 0.2]),
        "terminal_robot_clearance": np.array([0.05, 0.2]),
    }
    actions = np.array([[0, 1], [1, 1]])
    np.testing.assert_array_equal(
        stagnation_recovery_mask(geometry, actions, [True, False]),
        [False, False],
    )
    np.testing.assert_array_equal(
        stagnation_recovery_mask(
            geometry,
            actions,
            [True, False],
            allow_hard_safe_approach=True,
        ),
        [True, False],
    )


def test_stagnation_recovery_can_rank_hard_safe_candidates_by_clearance():
    geometry = {
        "feasible": np.array([False, False]),
        "hard_feasible": np.array([True, True]),
        "robot_clearance": np.array([-0.20, 0.02]),
        "terminal_robot_clearance": np.array([0.10, 0.02]),
        "human_clearance": np.array([1.0, 1.0]),
    }
    actions = np.array([[0, 1], [0, 1]])
    selected, fallback = select_with_joint_geometry(
        [-10.0, 0.0],
        [True, True],
        [0.0, 0.0],
        geometry,
        actions,
        [True, False],
        allow_hard_safe_approach=True,
        recovery_safety_first=True,
    )
    assert (selected, fallback) == (1, False)


def test_stagnation_recovery_keeps_cost_choice_inside_robust_robot_margin():
    geometry = {
        "feasible": np.array([False, False]),
        "hard_feasible": np.array([True, True]),
        "robot_clearance": np.array([0.02, 0.20]),
        "terminal_robot_clearance": np.array([0.02, 0.20]),
        "human_clearance": np.array([1.0, 1.0]),
    }
    actions = np.array([[0, 1], [0, 1]])
    selected, fallback = select_with_joint_geometry(
        [-10.0, 0.0],
        [True, True],
        [0.0, 0.0],
        geometry,
        actions,
        [True, False],
        allow_hard_safe_approach=True,
        recovery_safety_first=True,
    )
    assert (selected, fallback) == (0, False)


def test_stagnation_recovery_can_require_robust_robot_clearance():
    geometry = {
        "feasible": np.array([False, False]),
        "hard_feasible": np.array([True, True]),
        "robot_clearance": np.array([-0.01, 0.01]),
        "initial_robot_clearance": np.array([0.2, 0.2]),
        "terminal_robot_clearance": np.array([0.05, 0.05]),
    }
    actions = np.array([[0, 1], [0, 1]])
    np.testing.assert_array_equal(
        stagnation_recovery_mask(
            geometry,
            actions,
            [True, False],
            allow_hard_safe_approach=True,
            require_robust_robot_clearance=True,
        ),
        [False, True],
    )


def test_stagnation_recovery_can_require_robot_nonworsening_only():
    geometry = {
        "feasible": np.array([False, False]),
        "hard_feasible": np.array([True, True]),
        "initial_robot_clearance": np.array([0.1, -0.1]),
        "terminal_robot_clearance": np.array([0.0, -0.05]),
    }
    actions = np.array([[0, 1], [0, 1]])
    np.testing.assert_array_equal(
        stagnation_recovery_mask(
            geometry,
            actions,
            [True, False],
            allow_hard_safe_approach=True,
            require_nonworsening_robot_clearance=True,
        ),
        [False, True],
    )


def test_stagnation_recovery_uses_global_robot_clearance_floor():
    geometry = {
        "feasible": np.array([False, False]),
        "hard_feasible": np.array([True, True]),
        "robot_clearance": np.array([-0.06, -0.04]),
    }
    actions = np.array([[0, 1], [0, 1]])
    np.testing.assert_array_equal(
        stagnation_recovery_mask(
            geometry,
            actions,
            [True, False],
            allow_hard_safe_approach=True,
            robot_clearance_floor=-0.05,
        ),
        [False, True],
    )


def test_fallback_selects_lower_risk_without_assuming_halt_is_safe():
    geometry = {"feasible": np.array([True, False, True])}
    assert select_with_joint_geometry([0, -10, 2], [False] * 3, [0.1, 0, 0.2], geometry) == (0, True)


def test_initial_overlap_fallback_prefers_separating_option():
    geometry = {
        "feasible": np.array([False, False]),
        "robot_clearance": np.array([-0.1, -0.1]),
        "terminal_robot_clearance": np.array([0.1, -0.1]),
        "static_clearance": np.array([1, 1]),
    }
    assert select_with_joint_geometry([1, 0], [True, True], [0, 0], geometry) == (0, True)
