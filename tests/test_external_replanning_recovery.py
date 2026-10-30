import copy

import numpy as np
import pytest

from tracer.experiments.recover_external_replanning import require_equal, verify_completed_episode


def recovery_episode():
    return {"case": 1, "seed": 2027, "initial_scene_hash": "same", "policy": "tracer-persistent",
            "robot_kinematics": "unicycle", "steps": 8, "windows": 8, "total_time": 2.,
            "success": 1., "collision_free_success": 1., "collision": 0., "wall_collision": 0.,
            "timeout": 0., "environment_timeout": 0., "task_progress": .95}


def test_recovery_preserves_completed_episode_and_parent_remaining_value():
    replay = recovery_episode()
    original = {key: str(value) for key, value in replay.items()}
    group = {"window": 2, "parent_selected": 3, "branches": [{"candidate": 3,
             "first_window_verified": True, "elapsed_steps": 6, "elapsed_time": 1.5,
             "terminal_goal_distance": .25, "success": 1.}]}
    verify_completed_episode(original, replay, [group], np.array([1.75, 0.]), np.array([2., 0.]), .25, 1)
    with pytest.raises(ValueError, match="initial_scene_hash"):
        verify_completed_episode(original, {**replay, "initial_scene_hash": "other"}, [group], [1.75, 0.], [2., 0.], .25, 1)
    with pytest.raises(ValueError, match="collision"):
        verify_completed_episode(original, {**replay, "collision": 1.}, [group], [1.75, 0.], [2., 0.], .25, 1)
    altered = copy.deepcopy(group)
    altered["branches"][0]["elapsed_time"] = 1.
    with pytest.raises(ValueError, match="remaining time"):
        verify_completed_episode(original, replay, [altered], [1.75, 0.], [2., 0.], .25, 1)


def test_recovery_rejects_missing_parent_and_changed_terminal_geometry():
    replay = recovery_episode()
    group = {"window": 0, "parent_selected": 1, "branches": []}
    with pytest.raises(ValueError, match="one verified factual parent"):
        verify_completed_episode(replay, replay, [group], [1.75, 0.], [2., 0.], .25, 1)
    with pytest.raises(ValueError, match="geometry"):
        require_equal(np.array([1., 2.]), np.array([[1., 2.]]), "geometry")
    require_equal([np.inf, -np.inf], [np.inf, -np.inf], "support")
