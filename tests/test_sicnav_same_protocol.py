import numpy as np
import pytest
from collections import namedtuple

from sage.experiments.run_sicnav_same_protocol import _case_means, _normalise
from sage.experiments.run_sicnav_tracer_loop import (
    _candidate_pool,
    _augment_with_safe_recovery_candidates,
    _counterfactual_candidate_costs,
    _executor_safe_mask,
    _kinematic_safe_mask,
    _kinematic_surface_clearance,
    _rank_correlation,
    _ranking_oracle_index,
    _ranking_top_k_hit,
    _project_sage_speed_limit,
    _sage_dynamic_safe_mask,
    _sage_hybrid_candidate_pool,
    _static_obstacle_safe_mask,
    _validate_executable_trajectory_contract,
)


class _Agent:
    radius = 0.3

    def __init__(self, position, velocity):
        self._position = position
        self._velocity = velocity

    def get_position(self):
        return self._position

    def get_velocity(self):
        return self._velocity


class _Robot(_Agent):
    v_pref = 1.0

    def get_goal_position(self):
        return (2.0, 0.0)


_Action = namedtuple("_Action", ["vx", "vy"])


class _IdentityExecutor:
    def constrain_agent_action_exact(self, agent, action):
        return action


def test_normalise_adds_source_and_fills_common_fields():
    rows = _normalise(
        [
            {
                "policy": "official-orca",
                "case": "3",
                "steps": "60",
                "success": "1",
                "task_progress": "0.75",
                "min_distance": "0.55",
            }
        ],
        "official",
    )

    assert rows[0]["source"] == "official"
    assert rows[0]["case"] == 3
    assert rows[0]["steps"] == 60
    assert rows[0]["success"] == 1.0
    assert rows[0]["task_progress"] == 0.75
    assert rows[0]["collision"] != rows[0]["collision"]


def test_normalise_maps_native_sicnav_policy_names_to_paired_labels():
    rows = _normalise(
        [{"policy": "sicnav-np", "case": "3", "steps": "60", "success": "1"}],
        "official",
    )

    assert rows[0]["policy"] == "official-sicnav-np"


def test_case_means_average_model_seed_blocks():
    rows = _normalise(
        [
            {"policy": "tracer-persistent", "seed": "2027", "case": "0", "steps": "60", "task_progress": "0.4"},
            {"policy": "tracer-persistent", "seed": "2031", "case": "0", "steps": "60", "task_progress": "0.6"},
            {"policy": "tracer-persistent", "seed": "2027", "case": "1", "steps": "60", "task_progress": "0.2"},
        ],
        "tracer",
    )

    assert _case_means(rows, "tracer-persistent", "task_progress") == {0: 0.5, 1: 0.2}


def test_kinematic_safe_mask_rejects_body_tube_and_supports_disabled_mode():
    robot = _Agent((0.0, 0.0), (0.0, 0.0))
    human = _Agent((1.0, 0.0), (-0.2, 0.0))
    candidates = np.asarray(
        [
            [[0.1, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]],
            [[0.1, 1.0, 0.0, 0.0], [0.2, 1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    assert _kinematic_safe_mask(candidates, robot, [human], 1.0, 0.1).tolist() == [False, True]
    assert _kinematic_safe_mask(candidates, robot, [human], 1.0, -1.0).tolist() == [True, True]
    assert _kinematic_safe_mask(candidates, robot, [human], 1.0, 0.1, 1).tolist() == [True, True]


def test_kinematic_surface_clearance_orders_unsafe_recovery_controls():
    robot = _Agent((0.0, 0.0), (0.0, 0.0))
    human = _Agent((0.5, 0.0), (0.0, 0.0))
    candidates = np.asarray(
        [
            [[0.1, 0.0, 0.4, 0.0]],
            [[-0.1, 0.0, -0.4, 0.0]],
        ],
        dtype=np.float32,
    )

    clearance = _kinematic_surface_clearance(
        candidates, robot, [human], 0.25, prefix_steps=1
    )
    assert clearance[1] > clearance[0]
    assert np.all(clearance < 0.1)


def test_candidate_pool_can_include_an_external_proposal_without_changing_shape():
    robot = _Robot((0.0, 0.0), (0.0, 0.0))
    human = _Agent((1.0, 0.0), (0.0, 0.0))
    _, _, global_candidates = _candidate_pool(
        robot,
        [human],
        num_candidates=4,
        horizon=3,
        dt=0.25,
        rng=np.random.default_rng(7),
        proposal_velocity=np.asarray([0.0, 0.7], dtype=np.float32),
    )

    assert global_candidates.shape == (4, 3, 4)
    np.testing.assert_allclose(global_candidates[1, :, 2:], [[0.0, 0.7]] * 3)
    np.testing.assert_allclose(global_candidates[1, :, :2], [[0.0, 0.175], [0.0, 0.35], [0.0, 0.525]])


def test_executor_safe_mask_preserves_projected_unicycle_heading(monkeypatch):
    from sage.experiments import run_sicnav_tracer_loop as runner

    robot = type("UnicycleRobot", (), {"kinematics": "unicycle"})()
    candidates = np.asarray(
        [[[-0.05, 0.0, -0.2, 0.0], [-0.10, 0.0, -0.2, 0.0]]],
        dtype=np.float32,
    )
    recorded_headings = []

    def fake_velocity_action(robot, action_type, velocity, target_heading=None):
        recorded_headings.append(target_heading)
        return _Action(float(velocity[0]), float(velocity[1]))

    monkeypatch.setattr(runner, "_velocity_action", fake_velocity_action)
    monkeypatch.setattr(
        runner,
        "_action_velocity",
        lambda robot, action: np.asarray([action.vx, action.vy], dtype=np.float32),
    )

    mask = _executor_safe_mask(
        _IdentityExecutor(),
        robot,
        _Action,
        candidates,
        candidate_headings=np.zeros((1, 2), dtype=np.float64),
    )

    assert mask.tolist() == [True]
    assert recorded_headings == [0.0]


def test_unicycle_ranking_audit_rejects_velocity_only_replay():
    with pytest.raises(ValueError, match="heading-aware action histories"):
        _counterfactual_candidate_costs(
            None,
            0,
            None,
            None,
            [],
            np.zeros((1, 1, 4), dtype=np.float32),
            1,
            0.25,
            type("Planning", (), {"task_steps": 1})(),
            motion_profile="sicnav-unicycle",
        )


def test_sage_hybrid_keeps_generator_horizon_when_response_horizon_is_shorter(monkeypatch):
    robot = _Robot((0.0, 0.0), (0.0, 0.0))
    human = _Agent((1.0, 0.0), (0.0, 0.0))
    sage_global = np.zeros((12, 12, 4), dtype=np.float32)
    sage_global[..., 2] = 0.4
    sage_global[..., 0] = np.arange(1, 13, dtype=np.float32)[None] * 0.1
    sage_relative = sage_global[None].copy()
    sage_relative[..., 0] -= 1.0

    def fake_sage_pool(*args, **kwargs):
        current = np.asarray([[-1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        return current, sage_relative, sage_global, None

    monkeypatch.setattr(
        "sage.experiments.run_sicnav_tracer_loop._sage_candidate_pool",
        fake_sage_pool,
    )
    _, relative, global_candidates, _ = _sage_hybrid_candidate_pool(
        object(),
        [[np.zeros(2, dtype=np.float32)]],
        robot,
        [human],
        num_candidates=16,
        horizon=6,
        dt=0.25,
        sample_seed=7,
        rng=np.random.default_rng(7),
    )

    assert relative.shape == (1, 16, 12, 4)
    assert global_candidates.shape == (16, 12, 4)
    _validate_executable_trajectory_contract(global_candidates, np.zeros(2), 0.25, 1)


def test_recovery_candidates_are_appended_only_when_native_safe_support_is_empty():
    robot = _Robot((0.0, 0.0), (0.0, 0.0))
    human = _Agent((0.5, 0.0), (0.0, 0.0))
    native = np.zeros((1, 4, 4), dtype=np.float32)
    native[..., 2] = 1.0
    native[..., 0] = np.arange(1, 5, dtype=np.float32)[None] * 0.25
    current = np.asarray([[-0.5, 0.0, 0.0, 0.0]], dtype=np.float32)
    relative = native[None].copy()
    relative[..., 0] -= 0.5

    _, augmented_relative, augmented, added = _augment_with_safe_recovery_candidates(
        current,
        relative,
        native,
        _IdentityExecutor(),
        robot,
        [human],
        [],
        _Action,
        dt=0.25,
        kinematic_margin=0.0,
        static_margin=0.0,
        recovery_margin=0.05,
        max_recovery_candidates=4,
        direction_count=32,
    )

    assert 1 <= added <= 4
    assert augmented.shape == (1 + added, 4, 4)
    assert augmented_relative.shape == (1, 1 + added, 4, 4)
    np.testing.assert_array_equal(augmented[0], native[0])
    _validate_executable_trajectory_contract(augmented, np.zeros(2), 0.25, 1)
    assert _kinematic_safe_mask(
        augmented[1:], robot, [human], 0.25, 0.05, prefix_steps=1
    ).all()

    safe_native = augmented[1:2]
    safe_relative = augmented_relative[:, 1:2]
    _, unchanged_relative, unchanged, second_added = _augment_with_safe_recovery_candidates(
        current,
        safe_relative,
        safe_native,
        _IdentityExecutor(),
        robot,
        [human],
        [],
        _Action,
        dt=0.25,
        kinematic_margin=0.0,
        static_margin=0.0,
        recovery_margin=0.05,
        max_recovery_candidates=4,
        direction_count=32,
    )
    assert second_added == 0
    np.testing.assert_array_equal(unchanged, safe_native)
    np.testing.assert_array_equal(unchanged_relative, safe_relative)


def test_static_obstacle_safe_mask_rejects_swept_path_and_supports_disabled_mode():
    robot = _Agent((0.0, 0.0), (0.0, 0.0))
    candidates = np.asarray(
        [
            [[0.5, 0.0, 0.0, 0.0], [1.5, 0.0, 0.0, 0.0]],
            [[0.5, 1.0, 0.0, 0.0], [1.5, 1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    obstacle = [[(1.0, -0.2), (1.0, 0.2)]]
    assert _static_obstacle_safe_mask(candidates, robot, obstacle, 0.0).tolist() == [False, True]
    assert _static_obstacle_safe_mask(candidates, robot, obstacle, -1.0).tolist() == [True, True]
    assert _static_obstacle_safe_mask(candidates, robot, obstacle, 0.0, 1).tolist() == [True, True]


def test_ranking_audit_reports_top_one_and_top_three_hits():
    predicted = np.asarray([0.1, 0.2, 0.3, 0.4])
    realized = np.asarray([0.4, 0.1, 0.2, 0.3])
    assert _ranking_top_k_hit(predicted, realized, 1) == 0.0
    assert _ranking_top_k_hit(predicted, realized, 3) == 1.0
    spearman, kendall = _rank_correlation(
        np.asarray([0.1, 0.1, 0.3]), np.asarray([0.2, 0.2, 0.1])
    )
    assert np.isfinite(spearman)
    assert np.isfinite(kendall)


def test_ranking_oracle_excludes_candidates_outside_decision_support():
    decision_scores = np.asarray([0.2, np.inf, 0.3])
    realized = np.asarray([0.4, 0.0, 0.1])

    assert _ranking_oracle_index(decision_scores, realized) == 2
    assert _ranking_top_k_hit(decision_scores, realized, 1) == 0.0


def test_executable_trajectory_contract_checks_world_position_integration():
    candidates = np.asarray(
        [[[0.25, 0.0, 1.0, 0.0], [0.5, 0.0, 1.0, 0.0]]],
        dtype=np.float32,
    )
    _validate_executable_trajectory_contract(candidates, np.zeros(2), 0.25, 1)
    invalid = candidates.copy()
    invalid[0, 1, 0] += 0.1
    try:
        _validate_executable_trajectory_contract(invalid, np.zeros(2), 0.25, 1)
    except ValueError as error:
        assert "integration error" in str(error)
    else:
        raise AssertionError("invalid trajectory should fail the executable contract")


def test_sage_speed_projection_preserves_integrated_control_contract():
    robot = _Robot((0.0, 0.0), (0.0, 0.0))
    candidates = np.asarray(
        [[[2.0, 0.0, 2.0, 0.0], [4.0, 0.0, 2.0, 0.0]]],
        dtype=np.float32,
    )
    projected = _project_sage_speed_limit(candidates, robot, 0.25, 1.0)
    np.testing.assert_allclose(projected[..., 2:], [[[1.0, 0.0], [1.0, 0.0]]])
    _validate_executable_trajectory_contract(projected, np.zeros(2), 0.25, 1)


def test_sage_dynamic_mask_uses_candidate_conditioned_entity_future():
    robot = _Agent((0.0, 0.0), (0.0, 0.0))
    human = _Agent((2.0, 0.0), (0.0, 0.0))
    candidates = np.asarray(
        [
            [[0.25, 0.0, 1.0, 0.0], [0.5, 0.0, 1.0, 0.0]],
            [[0.25, 1.0, 1.0, 0.0], [0.5, 1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    entity_future = np.asarray(
        [
            [[[0.45, 0.0], [0.55, 0.0]]],
            [[[2.0, 0.0], [2.0, 0.0]]],
        ],
        dtype=np.float32,
    )
    assert _sage_dynamic_safe_mask(candidates, entity_future, robot, [human], 0.0, 2).tolist() == [False, True]
