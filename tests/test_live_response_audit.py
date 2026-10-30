import numpy as np
import pytest

from tracer.experiments.live_response_audit import choose_audit_branches, collect_live_responses, save_response_audit
from tracer.experiments.pack_live_response_data import partition_records
from tracer.experiments.eval_live_response_models import paired_response_effects


class Agent:
    def __init__(self, position, velocity):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
    def get_position(self):
        return self.position.copy()
    def get_velocity(self):
        return self.velocity.copy()


class Environment:
    time_step = .25
    def __init__(self):
        self.robot = Agent([0, 0], [0, 0])
        self.humans = [Agent([1, 0], [.1, 0]), Agent([2, 0], [.2, 0])]
    def constrain_agent_action_exact(self, robot, action):
        return action
    def step(self, action):
        for human in self.humans:
            human.velocity += self.robot.velocity * .1
            human.position += human.velocity * self.time_step
        self.robot.velocity = action.copy()
        self.robot.position += action * self.time_step
        np.random.random()


def test_response_audit_includes_selected_branch_without_duplicates():
    result = choose_audit_branches(8, 5, 6, 7)
    assert result[0] == 5 and len(set(result)) == 6
    np.testing.assert_array_equal(result, choose_audit_branches(8, 5, 6, 7))
    assert len(choose_audit_branches(2, 1, 6, 7)) == 2


def test_live_response_audit_preserves_state_rng_and_factual_first_step(tmp_path):
    env = Environment()
    candidates = np.zeros((2, 3, 4), dtype=np.float32)
    candidates[0, :, 2] = .4
    candidates[1, :, 2] = -.4
    histories = np.stack([np.tile(human.position, (4, 1)) for human in env.humans]).astype(np.float32)
    random_before = np.random.get_state()
    data = collect_live_responses(env, candidates, None, histories, 1, 2, 7, 5, 0, None,
                                  lambda robot, action_type, velocity, heading: velocity.copy())
    np.testing.assert_array_equal(env.robot.position, [0., 0.])
    random_after = np.random.get_state()
    assert random_before[0] == random_after[0] and random_before[2:] == random_after[2:]
    np.testing.assert_array_equal(random_before[1], random_after[1])
    assert data["target"].shape == (4, 3, 2)
    np.testing.assert_array_equal(data["target"][:2, 0], data["target"][2:, 0])
    assert not np.allclose(data["target"][:2, 1:], data["target"][2:, 1:])
    assert data["selected_branch"].tolist() == [True, True, False, False]
    path = save_response_audit(tmp_path, 7, 5, "test", [data])
    assert path.exists()
    with pytest.raises(ValueError, match="already exists"):
        save_response_audit(tmp_path, 7, 5, "test", [data])


def test_audit_records_actual_velocity_after_internal_executor_changes():
    class DoubleConstraintEnvironment(Environment):
        def constrain_agent_action_exact(self, robot, action):
            return action * .5
        def step(self, action):
            return super().step(self.constrain_agent_action_exact(self.robot, action))
    env = DoubleConstraintEnvironment()
    candidates = np.zeros((1, 3, 4), dtype=np.float32)
    candidates[..., 2] = .4
    histories = np.stack([np.tile(human.position, (4, 1)) for human in env.humans]).astype(np.float32)
    data = collect_live_responses(env, candidates, None, histories, 0, 1, 7, 5, 0, None,
                                  lambda robot, action_type, velocity, heading: velocity.copy())
    assert data["branch_projected"].all()
    np.testing.assert_allclose(data["candidates"][..., 2], .1)


def test_live_response_packing_keeps_cases_disjoint_and_requires_factual_branch():
    records = {"case": np.array([1, 1, 2, 2]), "human": np.zeros(4, dtype=int),
               "window": np.zeros(4, dtype=int), "selected_branch": np.array([True, False, True, False])}
    splits = partition_records(records, [1], [2])
    assert set(splits["train"]["case"]) == {1} and set(splits["val"]["case"]) == {2}
    with pytest.raises(ValueError, match="disjoint"):
        partition_records(records, [1], [1, 2])
    with pytest.raises(ValueError, match="exactly"):
        partition_records(records, [1], [3])
    records["selected_branch"][2] = False
    with pytest.raises(ValueError, match="factual selected branch"):
        partition_records(records, [1], [2])


def test_response_effect_audit_excludes_projected_and_different_state_pairs():
    target = np.zeros((4, 3, 2))
    target[1, 1:] = .5
    target[2, 1:] = 20
    data = {"case": np.ones(4), "human": np.zeros(4), "window": np.array([0, 0, 0, 6]),
            "target": target, "branch_projected": np.array([False, False, True, False])}
    metrics = paired_response_effects(data, target.copy())
    assert metrics["pairs"] == 1
    assert metrics["effect_error_mse"] == 0
    assert metrics["true_first_step_effect_max"] == 0
