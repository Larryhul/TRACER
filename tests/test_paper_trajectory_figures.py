from __future__ import annotations

import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from sage.experiments.plot_reference_trajectories import compact_grid, integrate, path_markers, scene_limits, square_limits
from sage.experiments.replay_paper_trajectories import environment_arrays, write_json


def test_integrate_keeps_initial_position_and_physical_time():
    path = integrate(np.asarray([2.0, 3.0]), np.asarray([[2.0, 0.0], [0.0, -4.0]]), 0.25)
    np.testing.assert_allclose(path, [[2.0, 3.0], [2.5, 3.0], [2.5, 2.0]])


def test_compact_grid_has_four_square_panels():
    figure, axes = compact_grid()
    assert axes.shape == (2, 2)
    assert all(axis.get_box_aspect() == 1 for axis in axes.flat)
    assert figure.subplotpars.wspace <= 0.10
    assert figure.subplotpars.hspace <= 0.16
    plt.close(figure)


def test_square_limits_preserve_data_extent():
    left, right, bottom, top = square_limits((-1, 2, -4, 5), padding=0.3)
    np.testing.assert_allclose(right - left, top - bottom)
    assert left < -1 and right > 2 and bottom < -4 and top > 5


def test_humans_are_hollow_and_robots_filled():
    figure, axis = plt.subplots()
    points = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    path_markers(axis, points, "blue", solid=False)
    path_markers(axis, points, "orange", solid=True)
    assert len(axis.collections[0].get_facecolors()) == 0
    assert len(axis.collections[1].get_facecolors()) == len(points)
    plt.close(figure)


def test_scene_limits_include_outlying_humans_and_goal():
    episode = {
        "robot": np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        "humans": np.asarray([[[3.0, -7.0]]]),
        "goal": np.asarray([-4.0, 8.0]),
    }
    left, right, bottom, top = scene_limits([episode])
    assert left < -4.0 and right > 3.0 and bottom < -7.0 and top > 8.0


def test_environment_arrays_exclude_warmup_and_include_terminal():
    def agent(position):
        return SimpleNamespace(position=position, get_position=lambda: position,
                               get_goal_position=lambda: (0.0, 2.0), radius=0.3)

    environment = SimpleNamespace(
        states=[[agent((0.0, -9.0)), [agent((1.0, -9.0))], []],
                [agent((0.0, 0.0)), [agent((1.0, 0.0))], []]],
        robot=agent((0.0, 0.25)), humans=[agent((1.0, 0.25))],
        static_obstacles=[], time_step=0.25,
    )
    arrays = environment_arrays(environment, 1)
    np.testing.assert_allclose(arrays["robot"], [[0.0, 0.0], [0.0, 0.25]])
    np.testing.assert_allclose(arrays["time"], [0.0, 0.25])
    assert arrays["humans"].shape == (2, 1, 2)


def test_json_maps_missing_metrics_to_null(tmp_path):
    path = tmp_path / "metrics.json"
    write_json(path, {"valid": 1.0, "missing": float("nan"), "nested": [float("inf")]})
    assert json.loads(path.read_text()) == {"valid": 1.0, "missing": None, "nested": [None]}
