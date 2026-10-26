import numpy as np
import pytest

from sage.response.unicycle_candidates import UnicycleLimits, braking_after_prefix, project_unicycle_candidates


def test_unicycle_projection_enforces_clock_acceleration_and_turn_rate():
    candidates = np.zeros((2, 12, 4), dtype=np.float32)
    candidates[0, :, 2] = 1.0
    candidates[1, :, 3] = 1.0
    limits = UnicycleLimits()
    projected = project_unicycle_candidates(candidates, [0., -1.5], [0., 0.], np.pi / 2, .25, limits)
    speeds = np.linalg.norm(projected[..., 2:], axis=-1)
    assert speeds.max() <= limits.max_speed + 1e-6
    assert np.diff(speeds, axis=1, prepend=np.zeros((2, 1))).max() <= limits.acceleration * .25 + 1e-6
    headings = np.arctan2(projected[..., 3], projected[..., 2])
    changes = (np.diff(headings, axis=1, prepend=np.full((2, 1), np.pi / 2)) + np.pi) % (2 * np.pi) - np.pi
    assert np.abs(changes).max() <= limits.max_yaw_rate * .25 + 1e-6
    np.testing.assert_allclose(projected[..., :2], np.array([0., -1.5]) + np.cumsum(projected[..., 2:] * .25, axis=1), atol=1e-6)


def test_unicycle_projection_brakes_instead_of_instantaneously_stopping():
    candidates = np.zeros((1, 5, 4), dtype=np.float32)
    projected = project_unicycle_candidates(candidates, [0., 0.], [.9, 0.], 0., .25, UnicycleLimits())
    assert projected[0, 0, 2] == pytest.approx(.525)
    assert projected[0, -1, 2] == pytest.approx(0.0)


def test_unicycle_can_reverse_from_rest_without_spinning_body():
    candidates = np.zeros((1, 8, 4), dtype=np.float32)
    candidates[..., 2] = -1.0
    projected, headings = project_unicycle_candidates(candidates, [0., 0.], [0., 0.], 0., .25,
                                                     UnicycleLimits(), return_headings=True)
    assert projected[0, 0, 2] == pytest.approx(-.125)
    np.testing.assert_allclose(headings, 0.0, atol=1e-6)


def test_unicycle_never_reverses_signed_speed_while_moving():
    candidates = np.zeros((1, 8, 4), dtype=np.float32)
    candidates[..., 2] = -1.0
    projected, headings = project_unicycle_candidates(candidates, [0., 0.], [.8, 0.], 0., .25,
                                                     UnicycleLimits(), return_headings=True)
    signed = (projected[..., 2:] * np.stack([np.cos(headings), np.sin(headings)], -1)).sum(-1)[0]
    assert not np.any(signed[:-1] * signed[1:] < -1e-6)


def test_braking_envelope_extends_prefix_until_physically_stopped():
    candidates = np.zeros((2, 2, 4))
    candidates[0, :, 0] = [.225, .45]
    candidates[0, :, 2] = .9
    candidates[1, :, 0] = [-.225, -.45]
    candidates[1, :, 2] = -.9
    stopping = braking_after_prefix(candidates, .25, 1.5)
    np.testing.assert_allclose(stopping[0, :, 2], [.9, .525, .15, 0.])
    np.testing.assert_allclose(stopping[1, :, 2], [-.9, -.525, -.15, 0.])
    np.testing.assert_allclose(stopping[..., :2], np.cumsum(stopping[..., 2:] * .25, axis=1))
    speeds = np.linalg.norm(stopping[..., 2:], axis=-1)
    assert np.diff(speeds, axis=1).min() >= -1.5 * .25 - 1e-8


def test_braking_envelope_rejects_safe_first_step_with_unsafe_stop():
    from types import SimpleNamespace
    from sage.experiments.run_sicnav_tracer_loop import _static_obstacle_safe_mask
    robot = SimpleNamespace(radius=.25, get_position=lambda: (0., 0.))
    candidate = np.array([[[.225, 0., .9, 0.]]])
    wall = [[(.6, -1.), (.6, 1.)]]
    assert _static_obstacle_safe_mask(candidate, robot, wall, .001).all()
    stopping = braking_after_prefix(candidate, .25, 1.5)
    assert not _static_obstacle_safe_mask(stopping, robot, wall, .001).any()


def test_braking_envelope_keeps_multistep_prefix_and_handles_zero_speed():
    candidates = np.zeros((1, 3, 4))
    candidates[0, :, 0] = [.1, .2, .3]
    candidates[0, :, 2] = .4
    stopping = braking_after_prefix(candidates, .25, 1.5, execute_steps=2)
    np.testing.assert_array_equal(stopping[:, :2], candidates[:, :2])
    assert stopping[0, -1, 2] == 0
    stationary = braking_after_prefix(np.zeros((1, 1, 4)), .25, 1.5)
    assert np.all(stationary == 0)
