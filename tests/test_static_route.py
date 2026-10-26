import numpy as np

from sage.response.static_route import static_route
from sage.response.unicycle_candidates import UnicycleLimits


def test_static_route_uses_doorway_instead_of_walking_into_wall():
    obstacles = [[[-1, -4], [-1, 4]], [[1, -4], [1, 4]], [[-1, 0], [-.5, 0]], [[.5, 0], [1, 0]]]
    route = static_route(obstacles, [0., 1.5], .25)
    start = np.array([-.65, -.5])
    paths = route.candidates(start, 24, .25, [.4, .8])
    np.testing.assert_allclose(paths[..., :2], start + np.cumsum(paths[..., 2:] * .25, axis=1), atol=1e-6)
    near_door = np.abs(paths[..., 1]) < .2
    assert np.all(np.abs(paths[..., 0][near_door]) < .25)
    assert route.distance_at(paths[:, -1, :2]).max() < route.distance_at(start)
    np.testing.assert_allclose(paths[1, -1, :2], route.goal, atol=1e-6)


def test_static_route_cache_and_stopped_goal_are_consistent():
    obstacles = [[[-1., -1.], [-1., 1.]]]
    route = static_route(obstacles, [0., 0.], .25)
    assert static_route(obstacles, [0., 0.], .25) is route
    assert np.isfinite(route.distance_at([[0., 0.], [.5, 0.]])).all()
    np.testing.assert_allclose(route.candidates([0., 0.], 6, .25, [.5]), 0.0)


def test_feedback_route_obeys_unicycle_motion_and_reaches_nearby_goal():
    route = static_route([[[-1., -1.], [-1., 1.]]], [0., 0.], .25)
    states, headings = route.unicycle_candidates([0., -.4], [0., 0.], np.pi / 2, 16, .25, [.7], UnicycleLimits())
    assert np.linalg.norm(states[0, -1, :2]) < .03
    np.testing.assert_allclose(states[..., :2], np.array([0., -.4]) + np.cumsum(states[..., 2:] * .25, axis=1), atol=1e-6)
    rotations = (np.diff(headings, axis=1, prepend=[[np.pi / 2]]) + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(rotations)) <= np.pi / 12 + 1e-6


def test_aligned_route_turns_in_place_before_moving_without_breaking_limits():
    route = static_route([[[-1., -1.], [-1., 1.]]], [0., 1.], .25)
    states, headings = route.unicycle_candidates([0., 0.], [0., 0.], 0., 20, .25, [.7], UnicycleLimits(), align_before_motion=True)
    np.testing.assert_array_equal(states[0, :4, 2:], 0.)
    assert headings[0, 0] > 0
    assert np.linalg.norm(states[0, -1, :2] - [0., 1.]) < .2
    speed = np.linalg.norm(states[..., 2:], axis=-1)
    assert np.diff(speed, axis=1, prepend=[[0.]]).max() <= .125 + 1e-6
    rotations = (np.diff(headings, axis=1, prepend=[[0.]]) + np.pi) % (2 * np.pi) - np.pi
    assert np.abs(rotations).max() <= np.pi / 12 + 1e-6


def test_aligned_route_brakes_before_turning_when_initially_moving():
    route = static_route([[[-1., -1.], [-1., 1.]]], [0., 1.], .25)
    states, headings = route.unicycle_candidates([0., 0.], [.9, 0.], 0., 20, .25, [.7], UnicycleLimits(), align_before_motion=True)
    assert states[0, 0, 2] == np.float32(.525)
    assert headings[0, 0] == 0.
    np.testing.assert_allclose(states[..., :2], np.cumsum(states[..., 2:] * .25, axis=1), atol=1e-6)
