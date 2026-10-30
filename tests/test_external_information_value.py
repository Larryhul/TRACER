import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tracer.experiments.eval_external_information_value import summarize_information
from tracer.experiments.external_information_value import (
    DecisionInformationAudit,
    factual_candidate_costs,
    ranking_information,
    replay_candidates,
    simulator_snapshot,
    true_speed_log_beliefs,
)
from tracer.response.closed_loop import PlanningCostConfig
from tracer.response.unicycle_candidates import UnicycleLimits


class Agent:
    def __init__(self, position, velocity, goal, heading=0.0):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.theta = heading
        self.v_pref = 1.0
        self.radius = 0.25
        self.kinematics = "unicycle"

    def get_position(self):
        return self.position.copy()

    def get_velocity(self):
        return self.velocity.copy()

    def get_goal_position(self):
        return self.goal.copy()


class Environment:
    time_step = 0.25

    def __init__(self, stop_after=100):
        self.robot = Agent([0.0, 0.0], [0.0, 0.2], [0.0, 10.0], np.pi / 2)
        self.humans = [Agent([1.0, 0.0], [0.1, 0.0], [4.0, 0.0])]
        self.global_time = 0.0
        self.steps = 0
        self.stop_after = stop_after

    def constrain_agent_action_exact(self, robot, action):
        return action

    def step(self, action):
        if self.steps >= self.stop_after:
            raise RuntimeError("stepped beyond terminal state")
        for human in self.humans:
            human.velocity += self.robot.velocity * 0.1
            human.position += human.velocity * self.time_step
        self.robot.theta += action.r
        self.robot.velocity = action.v * np.asarray([np.cos(self.robot.theta), np.sin(self.robot.theta)])
        self.robot.position += self.robot.velocity * self.time_step
        self.global_time += self.time_step
        self.steps += 1
        np.random.random()
        random.random()
        torch.rand(1)
        return None, 0.0, self.steps >= self.stop_after, {}

    def close(self):
        pass


def action_converter(robot, action_type, velocity, target_heading):
    heading = float(target_heading)
    return SimpleNamespace(v=float(velocity @ np.asarray([np.cos(heading), np.sin(heading)])),
                           r=float((heading - robot.theta + np.pi) % (2 * np.pi) - np.pi))


def example_paths():
    paths = np.zeros((3, 3, 4))
    paths[0, :, 3] = 0.3
    paths[1, :, 3] = -0.3
    paths[2, :, 3] = 0.0
    paths[..., :2] = np.cumsum(paths[..., 2:] * 0.25, axis=1)
    headings = np.full((3, 3), np.pi / 2)
    headings[2] += 0.1 * np.arange(1, 4)
    return paths, headings


def test_replay_preserves_delayed_response_reverse_motion_heading_state_and_rng():
    env = Environment()
    paths, headings = example_paths()
    before = simulator_snapshot(env)
    numpy_before, python_before, torch_before = np.random.get_state(), random.getstate(), torch.random.get_rng_state()
    replay = replay_candidates(env, paths, headings, None, limits=UnicycleLimits(), action_converter=action_converter)
    np.testing.assert_array_equal(simulator_snapshot(env), before)
    assert env.global_time == 0.0 and env.steps == 0
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
    assert np.random.get_state()[2:] == numpy_before[2:] and random.getstate() == python_before
    assert torch.equal(torch_before, torch.random.get_rng_state())
    np.testing.assert_allclose(replay["snapshots"][0, 0, 1, 2:4], [0.1, 0.02])
    np.testing.assert_allclose(replay["snapshots"][:, 0, 1, 2:4], np.tile([0.1, 0.02], (3, 1)))
    assert not np.allclose(replay["snapshots"][0, 1, 1, 2:4], replay["snapshots"][1, 1, 1, 2:4])
    np.testing.assert_allclose(replay["snapshots"][1, :, 0, 3], -0.3)
    np.testing.assert_allclose(replay["snapshots"][2, :, 0, 4], headings[2])
    assert replay["motion_valid"].all() and not replay["projected"].any()


def test_replay_stops_at_terminal_and_pads_absorbing_states():
    paths, headings = example_paths()
    replay = replay_candidates(Environment(stop_after=1), paths, headings, None, action_converter=action_converter)
    np.testing.assert_array_equal(replay["active"], np.tile([True, False, False], (3, 1)))
    np.testing.assert_array_equal(replay["snapshots"][:, 1:, :, 2:4], 0.0)
    np.testing.assert_allclose(replay["snapshots"][:, 2, :, :2], replay["snapshots"][:, 0, :, :2])


def test_replay_requires_unicycle_headings():
    paths, _ = example_paths()
    with pytest.raises(ValueError, match="explicit"):
        replay_candidates(Environment(), paths, None, None, action_converter=action_converter)


def test_replay_records_executor_projection_and_rejects_infeasible_actual_braking():
    class ProjectedEnvironment(Environment):
        def constrain_agent_action_exact(self, robot, action):
            return SimpleNamespace(v=0.0, r=action.r)

    env = ProjectedEnvironment()
    env.robot.velocity = np.asarray([0.0, 1.0])
    paths, headings = example_paths()
    paths[..., 3] = 1.0
    replay = replay_candidates(env, paths, headings, None, limits=UnicycleLimits(), action_converter=action_converter)
    assert replay["projected"].all() and not replay["motion_valid"].any()
    np.testing.assert_array_equal(env.robot.velocity, [0.0, 1.0])


def test_replay_restores_rng_when_a_clone_fails():
    class BrokenEnvironment(Environment):
        def step(self, action):
            super().step(action)
            raise RuntimeError("clone failure")

    env = BrokenEnvironment()
    paths, headings = example_paths()
    numpy_before, python_before, torch_before = np.random.get_state(), random.getstate(), torch.random.get_rng_state()
    with pytest.raises(RuntimeError, match="clone failure"):
        replay_candidates(env, paths, headings, None, action_converter=action_converter)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
    assert np.random.get_state()[2:] == numpy_before[2:] and random.getstate() == python_before
    assert torch.equal(torch_before, torch.random.get_rng_state()) and env.steps == 0


def test_true_mode_uses_training_thresholds_without_changing_observed_beliefs():
    humans = [SimpleNamespace(v_pref=value) for value in [0.5, 0.9, 1.2, 1.7]]
    beliefs, labels = true_speed_log_beliefs(humans, [0.9, 1.2], 3, "cpu")
    assert labels.tolist() == [0, 1, 2, 2]
    torch.testing.assert_close(beliefs.exp().sum(-1), torch.ones(4))
    torch.testing.assert_close(beliefs.exp().max(-1).values, torch.ones(4))
    with pytest.raises(ValueError, match="thresholds"):
        true_speed_log_beliefs(humans, [1.2, 0.9], 3, "cpu")


def test_factual_cost_matches_realized_positions_and_original_penalty():
    paths, headings = example_paths()
    replay = replay_candidates(Environment(), paths, headings, None, action_converter=action_converter)
    replay["robot_radius"] = 0.25
    planning = PlanningCostConfig(task_steps=1, risk_weight=0.5, smoothness_weight=0.1)
    penalty = np.asarray([0.2, 0.3, 0.4])
    factual = factual_candidate_costs(replay, 0.25, planning, penalty)
    snapshots = replay["snapshots"]
    distance = np.linalg.norm(snapshots[:, :, 0, :2] - snapshots[:, :, 1, :2], axis=-1).min(-1)
    safety = np.exp(-distance / planning.comfort_scale)
    human = np.square(snapshots[:, :, 1, 2:4] - replay["reference_snapshots"][None, :, 1, 2:4]).sum(-1).mean(-1)
    expected = (-snapshots[:, 0, 0, 1] / (0.25 * planning.task_speed_scale)
                + planning.safety_weight * safety + planning.human_weight * human
                + planning.risk_weight * (distance < planning.collision_margin)
                + planning.smoothness_weight * np.linalg.norm(snapshots[:, 0, 0, 2:4] - [0.0, 0.2], axis=-1)
                + penalty)
    np.testing.assert_allclose(factual["total"], expected, atol=1e-6)


def test_prefix_regret_does_not_count_tail_only_information_as_action_value():
    snapshots = np.zeros((3, 3, 2, 8))
    snapshots[2, 0, 0, 3] = 0.4
    factual = np.asarray([5.0, 1.0, 3.0])
    scores = {name: np.arange(3.0) for name in ("uniform", "persistent", "true_mode")}
    selected = {"uniform": 0, "persistent": 0, "true_mode": 1}
    result = ranking_information(scores, factual, snapshots, np.ones(3, bool), selected, 1)
    assert result["uniform_regret"] == 4.0 and result["uniform_prefix_regret"] == 0.0
    assert result["oracle_action_change_vs_uniform"] == 0 and result["true_mode_action_change_vs_uniform"] == 0
    factual[2] = 0.5
    result = ranking_information(scores, factual, snapshots, np.ones(3, bool), selected, 1)
    assert result["uniform_prefix_regret"] == 0.5 and result["oracle_action_change_vs_uniform"] == 1
    result = ranking_information(scores, factual, snapshots, np.asarray([True, True, False]), selected, 1)
    assert result["uniform_prefix_regret"] == 0.0 and result["oracle_selected"] == 1


def test_goal_terminal_mask_excludes_post_goal_human_effects():
    env = Environment()
    env.robot.goal = np.asarray([0.0, 0.075])
    paths, headings = example_paths()
    replay = replay_candidates(env, paths, headings, None, action_converter=action_converter)
    replay["robot_radius"] = 0.01
    planning = PlanningCostConfig(task_steps=1)
    full = factual_candidate_costs(replay, 0.25, planning, np.zeros(3))
    terminal = factual_candidate_costs(replay, 0.25, planning, np.zeros(3), True)
    assert full["human"][0] > 0 and terminal["human"][0] == 0.0


def test_live_prefix_verification_rejects_robot_or_human_transition_mismatch(tmp_path):
    paths, headings = example_paths()
    env = Environment()
    replay = replay_candidates(env, paths, headings, None, action_converter=action_converter)
    audit = DecisionInformationAudit(tmp_path, [0.9, 1.2])
    audit.records.append({})
    audit.pending = replay["snapshots"][0, :1]
    with pytest.raises(ValueError, match="differs"):
        audit.verify_execution(env, 1)
    env.step(action_converter(env.robot, None, paths[0, 0, 2:], headings[0, 0]))
    audit.verify_execution(env, 1)
    assert audit.pending is None and audit.records[-1]["live_prefix_verified"]
    assert (tmp_path / "windows.jsonl").exists()


def test_case_bootstrap_does_not_treat_windows_as_independent_cases():
    row = {"status": "matched", "live_prefix_verified": True, "case": 1,
           "uniform_cost": 2.0, "persistent_cost": 1.0, "true_mode_cost": 1.0, "oracle_cost": 0.0,
           "uniform_prefix_regret": 1.0}
    for name in ("uniform", "persistent", "true_mode", "oracle"):
        row[f"{name}_supported"] = True
        row[f"{name}_action_change_vs_uniform"] = 0
    other = {**row, "case": 2, "persistent_cost": 3.0}
    result = summarize_information([copy.deepcopy(row) for _ in range(20)] + [other])
    assert result["independent_cases"] == 2
    assert result["cost_comparisons"]["persistent"]["mean_cost_reduction_vs_uniform"] == 0.0
