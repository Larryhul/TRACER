from __future__ import annotations

from copy import deepcopy
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sage.experiments import plot_reference_trajectories as common
from sage.experiments.plot_multirobot_h3 import (
    POLICIES,
    align_humans,
    draw_panel,
    load_episodes,
    marker_indices,
    plot_matched_scenes,
    same_scene,
    sampled_gap,
    select_episode,
    unique_humans,
)


def make_episode(policy="tracer-persistent", episode_id=0):
    return {
        "initial": {"policy": policy, "episode": episode_id, "episode_seed": 2461000 + episode_id,
                    "initial_state_hash": "scene", "scenario": "envs/scenario/open"},
        "robot": np.asarray([[[0., 0.], [2., 0.], [4., 0.]], [[0., 4.], [2., 4.], [4., 4.]]]),
        "humans": np.asarray([[[10., 0.], [12., 0.]], [[10., 4.], [12., 4.]]]),
        "goals": np.asarray([[0., 4.], [2., 4.], [4., 4.]]),
        "time": np.asarray([0., 0.25]), "min_gap": 1.6,
        "terminal": {"team_success": 1., "collision_free_team_success": 1.,
                     "initial_collision": 0., "collision": 0.},
    }


def test_eight_humans_are_not_counted_three_times():
    observation = {"human_observations": [
        {"source_robot": robot, "position": [human + robot * 1e-7, 1.]}
        for robot in range(3) for human in range(8)
    ]}
    assert unique_humans(observation, expected=8).shape == (8, 2)


def test_same_position_humans_remain_distinct_with_complete_view():
    observation = {"human_observations": [{"source_robot": 0, "position": [0., 1.]}] * 8}
    assert unique_humans(observation, expected=8).shape == (8, 2)


def test_missing_humans_are_not_invented():
    with pytest.raises(ValueError, match="no complete observation"):
        unique_humans({"human_observations": [{"source_robot": 0, "position": [0., 1.]}]}, expected=8)


def test_assignment_tracks_eight_people_without_factorial_search():
    initial = np.column_stack([np.arange(8), np.zeros(8)])
    aligned = align_humans([initial, (initial + [0.1, 0])[::-1], (initial + [0.2, 0])[::-1]])
    np.testing.assert_allclose(aligned[-1], initial + [0.2, 0])
    with pytest.raises(ValueError, match="human count"):
        align_humans([initial, initial[:2]])


def test_sampled_gap_checks_robot_robot_and_robot_human_at_same_time():
    episode = make_episode()
    episode["robot"][1, 1] = episode["robot"][1, 0] + [0.2, 0.]
    gaps, contact = sampled_gap(episode)
    assert gaps.min() == pytest.approx(-0.2)
    assert contact[0] == 1
    episode["humans"][0, 0] = episode["robot"][0, 0] + [0.1, 0.]
    gaps, contact = sampled_gap(episode)
    assert gaps.min() == pytest.approx(-0.3)
    assert contact[0] == 0


def test_marker_sampling_keeps_terminal_without_interpolating_positions():
    times = np.arange(24) * 0.025
    indices = marker_indices(times)
    np.testing.assert_array_equal(indices, [0, 10, 20, 23])


def test_matched_scenes_refuse_extra_or_missing_cases(tmp_path):
    with pytest.raises(ValueError, match="two sources"):
        plot_matched_scenes([tmp_path / "first", tmp_path / "second", tmp_path / "third"], [0, 0], tmp_path)


def test_mismatched_scene_is_rejected():
    first = make_episode()
    second = deepcopy(first)
    second["humans"][0, 0, 0] += 0.02
    with pytest.raises(ValueError, match="initial scene mismatch"):
        same_scene([first, second])


def test_selection_excludes_initial_collision_and_retains_all_rows():
    episodes = {(policy, episode_id): make_episode(policy, episode_id)
                for policy in POLICIES for episode_id in range(2)}
    episodes[(POLICIES[0], 0)]["terminal"]["initial_collision"] = 1.
    result = select_episode(episodes)
    assert result["selected"]["episode"] == 1
    assert len(result["rows"]) == 8
    del episodes[(POLICIES[1], 1)]
    with pytest.raises(ValueError, match="missing paired policy"):
        select_episode(episodes)


def test_panel_uses_reference_markers_legend_and_full_episode_time():
    episode = make_episode()
    figure, axis = plt.subplots()
    draw_panel(axis, episode, (-1, 13, -5, 9))
    assert len(axis.lines) == 0
    assert len(axis.collections[0].get_facecolors()) == 0
    assert len(axis.collections[2].get_facecolors()) == 2
    np.testing.assert_allclose(axis.collections[2].get_facecolors()[-1, :3], [1., 165 / 255, 0.])
    assert [label.get_text() for label in axis.get_legend().get_texts()] == ["Robot", "Goal", "Human"]
    assert "Nav Time: 0.25s" in [label.get_text() for label in axis.texts]
    assert common.ORANGE == "#FFA500"
    plt.close(figure)


def test_loader_preserves_policy_identity_and_rejects_incomplete_data(tmp_path):
    path = tmp_path / "diagnostics.jsonl"
    records = []
    for policy in POLICIES[:2]:
        episode = make_episode(policy)
        for step, kind in enumerate(("initial", "tick")):
            records.append({**episode["initial"], "kind": kind, "time": step * 0.25,
                            "observation": {"robot_poses": episode["robot"][step].tolist(),
                                            "goals": episode["goals"].tolist(),
                                            "human_observations": [{"source_robot": 0, "position": position}
                                                                   for position in episode["humans"][step].tolist()]}})
        records.append({**episode["initial"], "kind": "episode", "metrics": episode["terminal"]})
    path.write_text("\n".join(json.dumps(record) for record in records))
    loaded = load_episodes(path, expected_humans=2)
    assert set(loaded) == {(policy, 0) for policy in POLICIES[:2]}
    path.write_text("\n".join(json.dumps(record) for record in records[:-1]))
    with pytest.raises(ValueError, match="incomplete episode"):
        load_episodes(path, expected_humans=2)
