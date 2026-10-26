from __future__ import annotations

import torch

from sage.response.closed_loop import ClosedLoopSpec, PlanningCostConfig
from sage.response.independent_social_force import (
    IndependentSocialForceDataset,
    IndependentSocialForceDatasetSpec,
    MultiEntityIndependentSocialForceSimulator,
    MultiEntityIndependentSocialForceDataset,
    MultiEntitySocialForceDatasetSpec,
    IndependentSocialForceSimulator,
)


def _spec() -> ClosedLoopSpec:
    return ClosedLoopSpec(
        windows=3,
        history=5,
        horizon=6,
        num_robots=3,
        num_candidates=8,
        execute_steps=2,
        dt=0.3,
        response_noise=0.08,
        scene_family="crossing",
        seed=2027,
    )


def test_social_force_counterfactual_replay_is_exact_and_nontrivial() -> None:
    first = IndependentSocialForceSimulator(_spec(), 4)
    second = IndependentSocialForceSimulator(_spec(), 4)
    first_pool = first.candidate_pool()
    second_pool = second.candidate_pool()
    first_outcomes = first.counterfactuals(first_pool, PlanningCostConfig())
    second_outcomes = second.counterfactuals(second_pool, PlanningCostConfig())
    assert torch.equal(first_pool.robot_candidates, second_pool.robot_candidates)
    for left, right in zip(first_outcomes, second_outcomes):
        assert torch.equal(left.response, right.response)
    assert not torch.equal(first_outcomes[0].response, first_outcomes[1].response)


def test_social_force_counterfactual_matches_executed_step() -> None:
    simulator = IndependentSocialForceSimulator(_spec(), 1)
    pool = simulator.candidate_pool()
    history = simulator.history.copy()
    outcomes = simulator.counterfactuals(pool, PlanningCostConfig())
    assert (history == simulator.history).all()
    realized = simulator.step(pool, 3, PlanningCostConfig())
    assert torch.equal(realized.response, outcomes[3].response)
    assert simulator.window == 1


def test_social_force_dataset_matches_tracer_contract() -> None:
    dataset = IndependentSocialForceDataset(
        IndependentSocialForceDatasetSpec(
            num_episodes=2,
            windows=3,
            history=5,
            horizon=6,
            num_robots=3,
            num_candidates=8,
            execute_steps=2,
        )
    )
    sample = dataset[0]
    assert sample["entity_history"].shape == (3, 5, 2)
    assert sample["robot_current"].shape == (3, 3, 4)
    assert sample["robot_candidates"].shape == (3, 3, 6, 4)
    assert sample["target_future_vel"].shape == (3, 6, 2)
    assert sample["active_mask"].all()


def test_multi_entity_social_force_has_shared_actions_and_reentry() -> None:
    spec = ClosedLoopSpec(
        windows=8,
        history=5,
        horizon=6,
        num_robots=3,
        num_candidates=8,
        execute_steps=2,
        dt=0.3,
        response_noise=0.08,
        scene_family="crossing",
        seed=2027,
    )
    first = MultiEntityIndependentSocialForceSimulator(
        spec, 4, num_entities=6, active_entities=3
    )
    second = MultiEntityIndependentSocialForceSimulator(
        spec, 4, num_entities=6, active_entities=3
    )
    first_pool = first.candidate_pool()
    second_pool = second.candidate_pool()
    first_outcomes = first.counterfactuals(first_pool, PlanningCostConfig())
    second_outcomes = second.counterfactuals(second_pool, PlanningCostConfig())
    assert torch.equal(first_pool.robot_candidates, second_pool.robot_candidates)
    for left, right in zip(first_outcomes, second_outcomes):
        assert torch.equal(left.response, right.response)
    assert first_outcomes[0].response.shape == (6, 6, 2)
    assert not torch.equal(first_outcomes[0].response, first_outcomes[1].response)
    assert (first.active_mask() == second.active_mask()).all()
    first.step(first_pool, 2, PlanningCostConfig())
    assert first.window == 1
    assert first.histories.shape == (6, 5, 2)


def test_multi_entity_social_force_dataset_matches_persistent_contract() -> None:
    dataset = MultiEntityIndependentSocialForceDataset(
        MultiEntitySocialForceDatasetSpec(
            num_episodes=2,
            windows=8,
            history=5,
            horizon=6,
            num_robots=3,
            num_entities=6,
            active_entities=3,
            num_candidates=8,
            execute_steps=2,
        )
    )
    sample = dataset[0]
    assert sample["entity_history"].shape == (6, 8, 5, 2)
    assert sample["robot_current"].shape == (6, 8, 3, 4)
    assert sample["robot_candidates"].shape == (6, 8, 3, 6, 4)
    assert sample["target_future_vel"].shape == (6, 8, 6, 2)
    assert sample["active_mask"].shape == (6, 8)
    assert sample["active_mask"].sum(dim=0).tolist() == [3] * 8
    assert not bool(sample["presence_mask"][0, 2:4].any())
    assert bool(sample["presence_mask"][0, 4:8].all())
