import numpy as np
import pytest

from sage.experiments.run_socialgym2_tracer_loop import (
    _action_lattice,
    _candidate_geometry,
    _completed_robot_action_lattice,
)
from sage.experiments.socialgym2_joint_geometry import joint_prefix_clearances


def test_completed_action_support_keeps_all_unfinished_choices():
    actions = _action_lattice(3)
    original = actions.copy()
    selected = _completed_robot_action_lattice(actions, np.array([True, False, True]))
    np.testing.assert_array_equal(selected, [[1, 0, 1], [1, 1, 1]])
    np.testing.assert_array_equal(actions, original)


def test_no_completed_robots_preserves_support_and_order():
    actions = _action_lattice(3)[::-1]
    np.testing.assert_array_equal(_completed_robot_action_lattice(actions, np.zeros(3, dtype=bool)), actions)


def test_all_completed_robots_have_one_joint_halt():
    np.testing.assert_array_equal(
        _completed_robot_action_lattice(_action_lattice(3), np.ones(3, dtype=bool)),
        [[1, 1, 1]],
    )


@pytest.mark.parametrize("actions,completed", [
    ([[0, 0]], [True, False]), ([[2, 1]], [False, False]), ([[0, 1]], [True]),
])
def test_completed_action_support_rejects_invalid_support(actions, completed):
    with pytest.raises(ValueError):
        _completed_robot_action_lattice(np.array(actions), np.array(completed))


def test_completed_robot_stays_in_joint_geometry_as_stationary_obstacle():
    current = np.array([[0.0, 0.0], [-0.8, 0.0]], dtype=np.float32)
    actions = _completed_robot_action_lattice(_action_lattice(2), np.array([True, False]))
    positions, velocities = _candidate_geometry(
        current, np.zeros_like(current), np.zeros(2), np.zeros(2),
        np.array([[2.0, 0.0], [2.0, 0.0]]), np.array([[0.0, 0.0], [2.0, 0.0]]),
        actions, 3, 0.4, "executable",
    )
    np.testing.assert_array_equal(positions[:, 0], 0)
    np.testing.assert_array_equal(velocities[:, 0], 0)
    assert np.any(positions[0, 1] != current[1])
    geometry = joint_prefix_clearances(current, positions, np.empty((0, 2, 2)))
    assert not geometry["feasible"][0]
    assert geometry["feasible"][1]
