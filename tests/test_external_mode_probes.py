from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sage.experiments.train_rolling_external_belief import matched_mode_effect_loss, matched_mode_probe_groups, validate_mode_probe_data
from sage.experiments.train_rolling_external_response import branch_response


def _probe_data():
    return {"case": np.array([112, 112]), "mode": np.array([0, 1]), "preferred_speed": np.array([.5, 1.5])}


def test_mode_probes_are_training_only_and_coordinate_matched():
    manifest = {"phase": "train", "source_native_train_sha256": "train-hash", "mode_speed_values": [.5, 1.5]}
    data = _probe_data()
    validate_mode_probe_data(data, manifest, "train-hash", {112}, {136}, [.5, 1.5])
    with pytest.raises(ValueError, match="matching native training"):
        validate_mode_probe_data(data, manifest, "other-hash", {112}, {136}, [.5, 1.5])
    data["case"][0] = 136
    with pytest.raises(ValueError, match="validation or test"):
        validate_mode_probe_data(data, manifest, "train-hash", {112}, {136}, [.5, 1.5])


def test_matched_mode_effect_does_not_reward_common_baseline_fit():
    target = torch.tensor([0., 1., 2.])[:, None, None].expand(-1, 2, 2)
    prediction = (target + 5.).clone().requires_grad_()
    torch.testing.assert_close(matched_mode_effect_loss(prediction, target, 3), torch.tensor(0.))
    collapsed = torch.ones_like(target).requires_grad_()
    loss = matched_mode_effect_loss(collapsed, target, 3)
    assert loss > 0
    loss.backward()
    assert collapsed.grad[0].sum() > 0 and collapsed.grad[2].sum() < 0


def test_mode_groups_reject_mismatched_robot_actions():
    data = {**_probe_data(), "window": np.zeros(2), "human": np.zeros(2), "branch": np.zeros(2),
            "history": np.zeros((2, 3, 2)), "current": np.zeros((2, 1, 4)), "candidates": np.zeros((2, 1, 2, 4))}
    np.testing.assert_array_equal(matched_mode_probe_groups(data, 2), [[0, 1]])
    data["candidates"][1, 0, 0, 0] = 1.
    with pytest.raises(ValueError, match="identical observed"):
        matched_mode_probe_groups(data, 2)


def test_mode_probe_label_matches_intervened_speed():
    manifest = {"phase": "train", "source_native_train_sha256": "train-hash", "mode_speed_values": [.5, 1.5]}
    data = _probe_data()
    data["preferred_speed"][0] = .8
    with pytest.raises(ValueError, match="intervened preferred speeds"):
        validate_mode_probe_data(data, manifest, "train-hash", {112}, {136}, [.5, 1.5])


class _Agent:
    def __init__(self, speed):
        self.v_pref = speed
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity


class _Environment:
    time_step = .25

    def __init__(self):
        self.robot = _Agent(1.)
        self.humans = [_Agent(.8), _Agent(1.2)]

    def constrain_agent_action_exact(self, robot, action):
        return action

    def step(self, action):
        self.robot.velocity = action
        self.robot.position += action * self.time_step
        for human in self.humans:
            human.velocity = np.array([human.v_pref, 0.])


def test_mode_intervention_changes_only_cloned_focal_human(monkeypatch):
    module = "sage.experiments.train_rolling_external_response"
    monkeypatch.setattr(f"{module}._velocity_action", lambda robot, kind, velocity, heading: velocity)
    monkeypatch.setattr(f"{module}._action_velocity", lambda robot, action: action)
    env = _Environment()
    _, _, responses = branch_response(env, np.zeros((2, 2)), np.zeros(2), SimpleNamespace,
                                      preferred_speeds={0: .4})
    np.testing.assert_allclose(responses[:, 0, 0], .4)
    np.testing.assert_allclose(responses[:, 1, 0], 1.2)
    assert [human.v_pref for human in env.humans] == [.8, 1.2]
    np.testing.assert_array_equal(env.humans[0].velocity, np.zeros(2))
    with pytest.raises(ValueError, match="positive finite"):
        branch_response(env, np.zeros((2, 2)), np.zeros(2), SimpleNamespace, preferred_speeds={0: -1.})
