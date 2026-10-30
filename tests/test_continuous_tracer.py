import numpy as np
import torch

from tracer.response.closed_loop import ClosedLoopSpec, PlanningCostConfig
from tracer.response.independent_social_force import (
    MultiEntityIndependentSocialForceSimulator,
)
from tracer.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
)
from tracer.response.joint_cost import (
    JointCandidateCostConfig,
    JointCandidateCostModel,
)


def test_continuous_conditioning_returns_grid_and_prior():
    model = BeliefConditionedTeamResponseModel(
        PersistentTrajectoryModelConfig(
            history=8,
            horizon=12,
            num_robots=3,
            architecture="tracer",
            conditioning="continuous",
        )
    )
    coordinates = torch.linspace(0.0, 2.0, 5)[None, None]
    output = model(
        torch.randn(2, 1, 8, 2),
        torch.randn(2, 1, 3, 4),
        torch.randn(2, 1, 3, 12, 4),
        condition_coordinates=coordinates.expand(2, 1, 5),
    )
    assert output["mean"].shape == (2, 1, 5, 12, 2)
    assert output["log_std"].shape == (2, 1, 5, 12, 2)
    assert output["coordinate_prior"].shape == (2, 1, 2)


def test_evidence_conditioning_accepts_only_causal_prefix_tensor_contract():
    model = BeliefConditionedTeamResponseModel(
        PersistentTrajectoryModelConfig(
            history=8,
            horizon=12,
            num_robots=3,
            architecture="tracer",
            conditioning="continuous",
            evidence_conditioning=True,
        )
    )
    output = model(
        torch.randn(2, 1, 8, 2),
        torch.randn(2, 1, 3, 4),
        torch.randn(2, 1, 3, 12, 4),
        condition_coordinates=torch.tensor([[[1.0]], [[0.5]]]),
        evidence_candidates=torch.randn(2, 1, 3, 3, 12, 4),
        evidence_responses=torch.randn(2, 1, 3, 12, 2),
        evidence_mask=torch.tensor([[[False, False, False]], [[True, False, False]]]),
    )
    assert output["mean"].shape == (2, 1, 1, 12, 2)


def test_entity_coupling_changes_counterfactual_response():
    spec = ClosedLoopSpec(num_robots=3, num_candidates=8, split="id", seed=123)
    independent = MultiEntityIndependentSocialForceSimulator(
        spec, 0, 6, 3, entity_interaction_strength=0.0
    )
    coupled = MultiEntityIndependentSocialForceSimulator(
        spec, 0, 6, 3, entity_interaction_strength=0.8
    )
    independent_response = independent.counterfactuals(
        independent.candidate_pool(), PlanningCostConfig()
    )[0].response.numpy()
    coupled_response = coupled.counterfactuals(
        coupled.candidate_pool(), PlanningCostConfig()
    )[0].response.numpy()
    assert np.abs(independent_response - coupled_response).mean() > 1e-3


def test_joint_candidate_cost_is_entity_permutation_invariant():
    model = JointCandidateCostModel(
        JointCandidateCostConfig(history=8, horizon=12, num_robots=3)
    ).eval()
    history = torch.randn(2, 4, 8, 2)
    current = torch.randn(2, 5, 4, 3, 4)
    candidates = torch.randn(2, 5, 4, 3, 12, 4)
    active = torch.tensor([[True, True, True, False], [True, True, True, True]])
    first = model(history, current, candidates, active)
    permutation = torch.tensor([2, 0, 3, 1])
    second = model(
        history[:, permutation],
        current[:, :, permutation],
        candidates[:, :, permutation],
        active[:, permutation],
    )
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)


def test_planning_cost_can_penalize_risk_and_first_action_jump():
    from tracer.response.closed_loop import PlanningCostConfig, expected_candidate_costs

    mode_means = torch.zeros(2, 1, 2, 2)
    reference = torch.zeros_like(mode_means)
    current = torch.zeros(2, 1, 4)
    candidates = torch.zeros(2, 1, 2, 4)
    candidates[0, 0, :, 0] = 0.66
    candidates[1, 0, :, 0] = 1.0
    candidates[0, 0, :, 2] = 0.1
    candidates[1, 0, :, 2] = 1.0
    log_stds = torch.full_like(mode_means, -0.7)
    baseline = expected_candidate_costs(
        mode_means,
        torch.ones(1),
        current,
        candidates,
        reference,
        0.25,
        PlanningCostConfig(
            task_weight=0.0,
            safety_weight=0.0,
            human_weight=0.0,
            task_steps=2,
            violation_probability_alpha=0.99,
        ),
        log_stds,
    )
    risk_regularized = expected_candidate_costs(
        mode_means,
        torch.ones(1),
        current,
        candidates,
        reference,
        0.25,
        PlanningCostConfig(
            task_weight=0.0,
            safety_weight=0.0,
            human_weight=0.0,
            task_steps=2,
            violation_probability_alpha=0.99,
            risk_weight=1.0,
        ),
        log_stds,
    )
    smooth_regularized = expected_candidate_costs(
        mode_means,
        torch.ones(1),
        current,
        candidates,
        reference,
        0.25,
        PlanningCostConfig(
            task_weight=0.0,
            safety_weight=0.0,
            human_weight=0.0,
            task_steps=2,
            smoothness_weight=1.0,
        ),
        log_stds,
    )
    assert float(baseline["total"][0]) == float(baseline["total"][1])
    assert float(risk_regularized["total"][0]) > float(risk_regularized["total"][1])
    assert float(smooth_regularized["total"][0]) < float(smooth_regularized["total"][1])


def test_planning_cost_accepts_explicit_robot_task_direction():
    from tracer.response.closed_loop import PlanningCostConfig, expected_candidate_costs

    mode_means = torch.zeros(2, 1, 2, 2)
    reference = torch.zeros_like(mode_means)
    current = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]
    )
    candidates = torch.zeros(2, 1, 2, 4)
    # Candidate 0 moves toward the focal human; candidate 1 moves toward the
    # external robot goal (+y).  The legacy implicit direction chooses 0.
    candidates[0, 0, :, :2] = torch.tensor([[0.9, 0.0], [0.75, 0.0]])
    candidates[1, 0, :, :2] = torch.tensor([[1.0, 0.1], [1.0, 0.25]])
    config = PlanningCostConfig(
        task_weight=1.0, safety_weight=0.0, human_weight=0.0, task_steps=2
    )
    legacy = expected_candidate_costs(
        mode_means, torch.ones(1), current, candidates, reference, 0.25, config
    )
    goal = expected_candidate_costs(
        mode_means,
        torch.ones(1),
        current,
        candidates,
        reference,
        0.25,
        config,
        task_direction=torch.tensor([0.0, 1.0]),
    )
    assert int(legacy["total"].argmin()) == 0
    assert int(goal["total"].argmin()) == 1


def test_soft_risk_weight_is_active_without_hard_risk_filter():
    from tracer.response.closed_loop import PlanningCostConfig, expected_candidate_costs

    mode_means = torch.zeros(2, 1, 2, 2)
    reference = torch.zeros_like(mode_means)
    current = torch.zeros(2, 1, 4)
    candidates = torch.zeros(2, 1, 2, 4)
    candidates[0, 0, :, :2] = 0.1
    candidates[1, 0, :, :2] = 1.0
    costs = expected_candidate_costs(
        mode_means,
        torch.ones(1),
        current,
        candidates,
        reference,
        0.25,
        PlanningCostConfig(
            task_weight=0.0,
            safety_weight=0.0,
            human_weight=0.0,
            collision_margin=0.65,
            risk_weight=1.0,
        ),
        mode_log_stds=torch.full_like(mode_means, -4.0),
    )
    assert float(costs["violation_probability"][0]) > float(
        costs["violation_probability"][1]
    )
    assert float(costs["total"][0]) > float(costs["total"][1])
    assert bool(costs["safe_mask"].all())
