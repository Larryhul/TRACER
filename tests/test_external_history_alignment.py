from types import SimpleNamespace

import numpy as np

from tracer.experiments.run_sicnav_tracer_loop import _histories


def _human(position, velocity):
    return SimpleNamespace(get_position=lambda: position, get_velocity=lambda: velocity)


def test_cold_history_is_chronological_and_ends_at_current_state():
    position = np.array([2.0, -1.0], dtype=np.float32)
    velocity = np.array([0.4, -0.8], dtype=np.float32)
    observed = [[position.copy()]]
    result = _histories(observed, [_human(position, velocity)], 8, 0.25)
    np.testing.assert_allclose(np.diff(result[0], axis=0), np.tile(velocity * 0.25, (7, 1)), atol=2e-7)
    np.testing.assert_array_equal(result[0, -1], position)
    repeated = _histories(observed, [_human(position, velocity)], 8, 0.25)
    np.testing.assert_array_equal(repeated, result)


def test_partial_history_prepends_before_oldest_observation():
    positions = [np.array([1.0, 0.0]), np.array([1.25, 0.0])]
    result = _histories([positions.copy()], [_human(positions[-1], [99.0, 99.0])], 4, 0.25)
    np.testing.assert_allclose(result[0], [[0.5, 0.0], [0.75, 0.0], [1.0, 0.0], [1.25, 0.0]])


def test_empty_stationary_history_has_no_fabricated_motion():
    result = _histories([[]], [_human([1.0, 2.0], [0.0, 0.0])], 4, 0.25)
    np.testing.assert_array_equal(result[0], [[1.0, 2.0]] * 4)
