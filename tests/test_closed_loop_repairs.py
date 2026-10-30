from argparse import Namespace

import numpy as np
import pytest
import torch

from tracer.belief.filter import adaptive_categorical_update
from tracer.experiments.eval_persistent_closed_loop import Policy, _evaluate_policy
from tracer.experiments.run_sicnav_tracer_loop import _update_belief, _response_predictions
from tracer.response.planning_support import (
    aggregate_entity_costs,
    diverse_candidate_indices,
    swept_surface_clearance,
    goal_return_trajectories,
)
from tracer.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig
from tracer.response.closed_loop import (
    ClosedLoopSpec,
    _independent_trajectory_noise,
    PlanningCostConfig,
    aggregate_robot_progress,
    expected_candidate_costs,
)


def test_adaptive_surprise_is_independent_of_update_temperature():
    prior = torch.tensor([0.999, 0.001]).log()
    likelihood = torch.tensor([-20.0, -1.0])
    cold = adaptive_categorical_update(prior, likelihood, likelihood_scale=0.05)
    warm = adaptive_categorical_update(prior, likelihood, likelihood_scale=1.0)
    torch.testing.assert_close(cold[1], warm[1])
    torch.testing.assert_close(cold[2], warm[2])
    assert bool(cold[3]) and bool(warm[3])
    assert warm[0].exp()[1] > cold[0].exp()[1]


def test_external_change_point_uses_raw_evidence():
    args = Namespace(likelihood_scale=0.05, change_point_threshold=5.0)
    posterior, changed = _update_belief(
        torch.tensor([0.999, 0.001]).log(), torch.tensor([-20.0, -1.0]),
        "tracer-change-point", args,
    )
    assert changed
    torch.testing.assert_close(posterior, torch.tensor([-1.0, -0.05]).log_softmax(-1))


def test_adaptive_persistent_closed_loop_converts_stored_log_belief_for_planning():
    torch.manual_seed(19)
    spec = ClosedLoopSpec(
        windows=2,
        history=4,
        horizon=4,
        num_robots=2,
        num_candidates=6,
        execute_steps=1,
        dt=0.4,
        response_noise=0.0,
        seed=19,
    )
    model = BeliefConditionedTeamResponseModel(
        PersistentTrajectoryModelConfig(
            architecture="tracer",
            history=4,
            horizon=4,
            num_robots=2,
            hidden_dim=16,
            num_heads=2,
            dropout=0.0,
            dt=0.4,
        )
    ).eval()
    result = _evaluate_policy(
        model,
        Policy("adaptive-persistent-joint", "adaptive-persistent", True),
        spec,
        episodes=1,
        device="cpu",
        planning_cost=PlanningCostConfig(task_steps=1, response_steps=1),
        likelihood_scale=0.05,
    )
    assert np.isfinite(result["belief_entropy"]["mean"])
    assert np.isfinite(result["final_belief_entropy"]["mean"])


def test_swept_guard_catches_between_tick_crossing():
    clearance = swept_surface_clearance(
        np.array([[[1.0, 0.0, 2.0, 0.0]]]), np.array([-1.0, 0.0]),
        np.array([[0.0, -1.0]]), np.array([[0.0, 2.0]]), np.array([0.6]), 1.0,
    )
    assert clearance.shape == (1, 1)
    assert clearance[0, 0] == pytest.approx(-0.6)


def test_swept_guard_is_translation_invariant():
    candidates = np.array([[[1.0, 1.0, 1.0, 1.0]]])
    original = swept_surface_clearance(candidates, np.zeros(2), np.array([[2.0, 1.0]]), np.zeros((1, 2)), np.array([0.6]), 1.0)
    shifted = candidates.copy()
    shifted[..., :2] += 10.0
    translated = swept_surface_clearance(shifted, np.ones(2) * 10, np.array([[12.0, 11.0]]), np.zeros((1, 2)), np.array([0.6]), 1.0)
    np.testing.assert_allclose(original, translated)


def test_goal_return_preserves_first_action_and_integrates_every_step():
    candidates = np.zeros((2, 12, 4), dtype=np.float32)
    candidates[:, :, 3] = 0.5
    paths = goal_return_trajectories(candidates, np.zeros(2), np.array([3.0, 0.0]), 0.25, 1.0)
    np.testing.assert_allclose(paths[:, 0, 2:], candidates[:, 0, 2:])
    np.testing.assert_allclose(paths[..., :2], np.cumsum(paths[..., 2:] * 0.25, axis=1), atol=1e-6)
    assert paths[0, -1, 0] > 1.0
    assert np.linalg.norm(paths[..., 2:], axis=-1).max() <= 1.0 + 1e-6


def test_diversity_retains_task_best_and_distinct_controls():
    trajectories = torch.zeros(4, 1, 3, 4)
    trajectories[0, ..., 2] = 1.0
    trajectories[1, ..., 2] = 0.99
    trajectories[2, ..., 3] = 1.0
    trajectories[3, ..., 3] = -1.0
    selected = diverse_candidate_indices(trajectories, torch.tensor([1.0, 0.99, 0.8, 0.8]), 3, 0.6)
    assert selected.tolist() == [0, 2, 3]


def test_diversity_uses_relative_positions_as_well_as_controls():
    trajectories = torch.zeros(4, 1, 3, 4)
    trajectories[0, 0, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    trajectories[1, 0, :, 0] = torch.tensor([1.0, 1.99, 2.98])
    trajectories[2, 0, :, 1] = torch.tensor([1.0, 2.0, 3.0])
    trajectories[3, 0, :, 1] = torch.tensor([-1.0, -2.0, -3.0])
    selected = diverse_candidate_indices(
        trajectories, torch.tensor([1.0, 0.99, 0.8, 0.8]), 3, 0.6
    )
    assert selected.tolist() == [0, 2, 3]


def test_crowd_risk_does_not_disappear_among_safe_neighbors():
    def cost(risk):
        return {name: torch.tensor([risk if name in {"safety", "violation_probability"} else 0.0]) for name in ("task", "safety", "human", "smoothness", "violation_probability")}

    danger = cost(0.8)
    safe = cost(0.0)
    single, _ = aggregate_entity_costs([danger], 1, 2, 0.6, 0, 0.5, "union-bound")
    crowd, risk = aggregate_entity_costs([danger, safe, safe], 1, 2, 0.6, 0, 0.5, "union-bound")
    torch.testing.assert_close(single, crowd)
    assert risk.item() == pytest.approx(0.8)


def test_evidence_ignores_padding_and_preserves_hidden_state_dimensions():
    torch.manual_seed(21)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", hidden_dim=16, num_heads=2, dropout=0.0,
        evidence_conditioning=True,
    )).eval()
    actions = torch.randn(2, 1, 3, 2, 4, 4)
    responses = torch.randn(2, 1, 3, 4, 2)
    mask = torch.tensor([[[True, False, True]], [[False, False, False]]])
    original = model._evidence_context(actions, responses, mask, 2, 1, torch.device("cpu"), torch.float32)
    changed_actions = actions.clone()
    changed_responses = responses.clone()
    changed_actions[~mask] = 1000.0
    changed_responses[~mask] = -1000.0
    changed = model._evidence_context(changed_actions, changed_responses, mask, 2, 1, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(original, changed)
    torch.testing.assert_close(original[1], torch.zeros(16))
    assert original[0].std() > 0.01


def test_enhanced_response_is_robot_permutation_invariant_and_differentiable():
    torch.manual_seed(7)
    config = PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=3, num_robots=3,
        hidden_dim=16, num_heads=2, graph_layers=1, dropout=0.0,
        geometry_features=True, temporal_pooling="attention",
        candidate_uncertainty=True, set_interactions=True,
    )
    model = BeliefConditionedTeamResponseModel(config).eval()
    history = torch.randn(2, 1, 4, 2)
    current = torch.randn(2, 1, 3, 4)
    candidates = torch.randn(2, 1, 3, 3, 4)
    first = model(history, current, candidates)
    order = torch.tensor([2, 0, 1])
    reordered = model(history, current[:, :, order], candidates[:, :, order])
    torch.testing.assert_close(first["mean"], reordered["mean"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(first["log_std"], reordered["log_std"], atol=1e-5, rtol=1e-5)
    altered = model(history, current, candidates + 1.0)
    assert not torch.allclose(first["log_std"], altered["log_std"])
    (first["mean"].square().mean() + first["log_std"].square().mean()).backward()
    assert model.candidate_std_decoder[-1].weight.grad.abs().sum() > 0


def test_zero_initialized_interaction_residual_starts_as_additive_response():
    torch.manual_seed(19)
    config = PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=3, num_robots=3,
        hidden_dim=16, num_heads=2, graph_layers=1, dropout=0.0,
        candidate_uncertainty=True, set_interactions=True,
        zero_init_interaction_residual=True,
    )
    model = BeliefConditionedTeamResponseModel(config).eval()
    history = torch.randn(2, 1, 4, 2)
    current = torch.randn(2, 1, 3, 4)
    candidates = torch.randn(2, 1, 3, 3, 4)
    with torch.no_grad():
        additive = model(history, current, candidates, include_pair_interactions=False)
        joint = model(history, current, candidates, include_pair_interactions=True)
    torch.testing.assert_close(additive["mean"], joint["mean"])
    assert model.pair_decoder[-1].weight.abs().sum() == 0
    assert model.set_decoder[-1].weight.abs().sum() == 0


def test_neighbor_geometry_distinguishes_equal_motion_at_different_locations():
    torch.manual_seed(11)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=3, num_robots=1,
        hidden_dim=16, num_heads=2, graph_layers=1, dropout=0.0, geometry_features=True,
    )).eval()
    history = torch.zeros(1, 1, 4, 2)
    current = torch.ones(1, 1, 1, 4)
    candidates = torch.ones(1, 1, 1, 3, 4)
    near = torch.ones(1, 1, 1, 4, 2)
    first = model(history, current, candidates, neighbor_history=near)
    distant = model(history, current, candidates, neighbor_history=near + 20.0)
    assert not torch.allclose(first["mean"], distant["mean"])


def test_risk_noise_matches_independent_temporal_likelihood():
    samples = _independent_trajectory_noise(2048, 6)
    assert samples.shape == (2048, 6, 2)
    correlation = torch.corrcoef(torch.stack([samples[:, 0, 0], samples[:, 1, 0]]))[0, 1]
    assert abs(correlation) < 0.05
    torch.testing.assert_close(samples, _independent_trajectory_noise(2048, 6))


def test_attention_evidence_cannot_read_unexecuted_suffix():
    torch.manual_seed(8)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", hidden_dim=16, num_heads=2, dropout=0.0,
        evidence_conditioning=True, evidence_pooling="attention", evidence_prefix_steps=2,
    )).eval()
    actions = torch.randn(1, 1, 2, 2, 4, 4)
    responses = torch.randn(1, 1, 2, 4, 2)
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    original = model._evidence_context(actions, responses, mask, 1, 1, torch.device("cpu"), torch.float32)
    actions[..., 2:, :] = 1000.0
    responses[..., 2:, :] = -1000.0
    changed = model._evidence_context(actions, responses, mask, 1, 1, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(original, changed)


def test_ranking_dataset_stores_only_executed_prefix_for_every_entity():
    from tracer.experiments.run_continuous_ranking_pilot import CounterfactualRankingDataset

    args = Namespace(
        windows=2, history=4, horizon=4, num_robots=2, num_entities=4, active_entities=2,
        num_candidates=6, execute_steps=1, dt=0.3, data_seed=913,
        response_noise=0.08, entity_max_speed=2.2, scene_family="crossing",
        entity_interaction_strength=0.8, entity_interaction_decay=0.72,
        entity_interaction_radius=0.85, collision_margin=0.65, risk_alpha=0.3,
        risk_samples=8,
    )
    sample = CounterfactualRankingDataset(args, "train", 1)[0]
    assert sample["executed_responses"].shape == (2, 4, 4, 2)
    assert torch.count_nonzero(sample["executed_responses"][:, :, 1:]) == 0
    assert torch.all(sample["executed_responses"][:, :, 0].abs().sum(dim=-1) > 0)


def test_reference_predictions_are_reused_per_entity_without_changing_values():
    torch.manual_seed(33)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=3, num_robots=1,
        hidden_dim=16, num_heads=2, graph_layers=1, dropout=0.0,
    )).eval()
    histories = np.random.default_rng(33).normal(size=(2, 4, 2)).astype(np.float32)
    current = np.ones((2, 4), dtype=np.float32)
    candidates = np.ones((2, 5, 3, 4), dtype=np.float32)
    _, _, references, _, _ = _response_predictions(model, histories, current, candidates, "cpu")
    _, _, single, _, _ = _response_predictions(model, histories, current, candidates[:, :1], "cpu")
    torch.testing.assert_close(references, single.expand_as(references))


def test_goal_cost_scale_matches_robot_speed_not_human_location():
    means = torch.zeros(2, 1, 2, 2)
    current = torch.zeros(2, 1, 4)
    current[..., 0] = -10.0
    candidates = torch.zeros(2, 1, 2, 4)
    candidates[..., 0] = -10.0
    candidates[0, 0, :, 1] = torch.tensor([0.25, 0.5])
    candidates[1, 0, :, 1] = torch.tensor([-0.25, -0.5])
    result = expected_candidate_costs(
        means, torch.ones(1), current, candidates, means, 0.25,
        PlanningCostConfig(task_speed_scale=1.0, safety_weight=0, human_weight=0, task_steps=2),
        task_direction=torch.tensor([0.0, 1.0]),
    )
    torch.testing.assert_close(result["task"], torch.tensor([-1.0, 1.0]))


def test_lower_tail_robot_progress_emphasizes_the_slowest_robot():
    progress = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]])
    mean = aggregate_robot_progress(progress, "mean")
    lower_tail = aggregate_robot_progress(progress, "lower-tail", min_blend=0.5)
    torch.testing.assert_close(mean, torch.tensor([1.0, 1.0 / 3.0]))
    torch.testing.assert_close(lower_tail, torch.tensor([1.0, -1.0 / 3.0]))


def test_lower_tail_robot_progress_ignores_finished_robot_when_masked():
    progress = torch.tensor([[1.0, -10.0], [2.0, -20.0]])
    active = torch.tensor([True, False])
    result = aggregate_robot_progress(progress, "lower-tail", active, min_blend=0.5)
    torch.testing.assert_close(result, torch.tensor([1.0, 2.0]))


def test_crowdnav_fallback_maximizes_clearance_and_respects_official_support():
    from tracer.experiments.run_crowdnav_tracer import _select_candidate

    beliefs = np.zeros((1, 1), dtype=np.float32)
    means = np.zeros((1, 3, 1, 2, 2), dtype=np.float32)
    paths = np.array([[[0.1, 0.0]] * 2, [[0.4, 0.0]] * 2, [[0.3, 0.0]] * 2], dtype=np.float32)
    arguments = dict(
        beliefs=beliefs, means=means, world_position=paths,
        robot_position=np.zeros(2), robot_goal=np.array([1.0, 0.0]),
        human_positions=np.zeros((1, 2)), dt=0.25, collision_margin=0.65,
    )
    selected, _, diagnostics = _select_candidate(**arguments)
    assert selected == 1
    assert diagnostics["safety_fallback"] == 1.0
    selected, _, _ = _select_candidate(**arguments, official_safe=np.array([True, False, True]))
    assert selected == 2


def test_crowdnav_checks_third_executed_step_not_xy_dimension():
    from tracer.experiments.run_crowdnav_tracer import _select_candidate

    paths = np.array([[[2.0, 0.0], [2.0, 0.0], [0.1, 0.0]]], dtype=np.float32)
    _, _, diagnostic = _select_candidate(
        np.zeros((1, 1)), np.zeros((1, 1, 1, 3, 2)), paths,
        np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.zeros((1, 2)),
        0.25, 0.65, task_steps=3,
    )
    assert diagnostic["predicted_min_distance"] == pytest.approx(0.1)
    assert diagnostic["safety_fallback"] == 1.0
