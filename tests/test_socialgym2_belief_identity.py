from types import SimpleNamespace

import numpy as np
import pytest

from sage.experiments.run_socialgym2_tracer_loop import HumanTrack, _scene_from_responses, _update_beliefs


def _track(track_id):
    return HumanTrack(track_id=track_id, position=np.zeros(2, dtype=np.float32),
                      velocity=np.array([1.0, 0.0], dtype=np.float32),
                      observed_velocity=np.array([1.0, 0.0], dtype=np.float32))


def test_belief_follows_prediction_identity_after_deletion_and_reordering():
    means = np.zeros((2, 1, 2, 1, 2), dtype=np.float32)
    means[0, 0, :, 0, 0] = [1.0, -1.0]
    means[1, 0, :, 0, 0] = [-1.0, 1.0]
    survivor, new_track = _track(22), _track(33)
    _update_beliefs([new_track, survivor], means, np.zeros_like(means), 0, "tracer-persistent",
                   SimpleNamespace(device="cpu", likelihood_scale=1.0), track_ids=[11, 22])
    assert survivor.belief.argmax() == 1
    assert new_track.belief is None


def test_duplicate_prediction_ids_are_rejected():
    means = np.zeros((2, 1, 2, 1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="uniquely align"):
        _update_beliefs([_track(11)], means, means, 0, "tracer-persistent",
                       SimpleNamespace(device="cpu", likelihood_scale=1.0), track_ids=[11, 11])


def test_same_robot_nearby_humans_are_not_merged():
    response = SimpleNamespace(robot_poses=[0.0, 0.0, 0.0], robot_vels=[0.0, 0.0, 0.0],
                               goal_pose=[4.0, 0.0, 0.0], human_poses=[[2.0, 0.0], [2.5, 0.0]],
                               human_vels=[[1.0, 0.0], [1.0, 0.0]])
    assert len(_scene_from_responses([response])[3]) == 2
    assert len(_scene_from_responses([response, response])[3]) == 2


def test_robot_body_velocity_is_rotated_to_world_frame():
    response = SimpleNamespace(robot_poses=[0.0, 0.0, np.pi / 2], robot_vels=[1.0, 0.0, 0.0],
                               goal_pose=[0.0, 4.0, 0.0], human_poses=[], human_vels=[])
    np.testing.assert_allclose(_scene_from_responses([response])[1][0], [0.0, 1.0], atol=1e-6)
