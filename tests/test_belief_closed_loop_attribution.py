import numpy as np
import pytest
import torch

from tracer.experiments import eval_persistent_closed_loop as loop
from tracer.belief.adaptive_persistent import AdaptivePersistentBelief, AdaptivePersistentBeliefConfig
from tracer.belief.filter import categorical_bayes_update
from tracer.response.closed_loop import ClosedLoopSpec, PlanningCostConfig
from tracer.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
)


@pytest.fixture
def small_loop():
    torch.manual_seed(91)
    spec = ClosedLoopSpec(
        windows=3, history=4, horizon=4, num_robots=2,
        num_candidates=6, execute_steps=1, dt=0.4, seed=91,
    )
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=4, num_robots=2,
        hidden_dim=16, num_heads=2, dropout=0.0, dt=0.4,
    )).eval()
    return dict(
        model=model, spec=spec, episodes=2, device="cpu",
        planning_cost=PlanningCostConfig(task_steps=1, response_steps=1),
        likelihood_scale=0.2,
    )


@pytest.mark.parametrize("gated", [False, True])
def test_zero_hazard_adaptive_preserves_independent_episode_decision_beliefs(small_loop, gated):
    persistent = loop._evaluate_policy(
        **small_loop, policy=loop.Policy("persistent", "persistent", True, response_gate=gated),
    )
    adaptive = loop._evaluate_policy(
        **small_loop,
        policy=loop.Policy("adaptive", "adaptive-persistent", True, response_gate=gated),
        adaptive_base_hazard=0.0, adaptive_max_hazard=0.0,
    )
    assert persistent["selected_candidates"] == adaptive["selected_candidates"]
    for original_rows, adaptive_rows in zip(persistent["window_records"], adaptive["window_records"]):
        previous_posterior = np.full(3, 1 / 3)
        for original, current in zip(original_rows, adaptive_rows):
            np.testing.assert_allclose(current["decision_mode_probabilities"], previous_posterior, atol=1e-6)
            np.testing.assert_allclose(current["decision_mode_probabilities"], original["decision_mode_probabilities"], atol=1e-6)
            np.testing.assert_allclose(current["posterior_mode_probabilities"], original["posterior_mode_probabilities"], atol=1e-6)
            previous_posterior = current["posterior_mode_probabilities"]
    assert adaptive["mode_accuracy_by_window"] == adaptive["posterior_mode_accuracy_by_window"]
    assert len(adaptive["decision_mode_accuracy_by_window"]) == small_loop["spec"].windows


def test_gate_does_not_rewrite_calibrated_emission_likelihood(small_loop):
    reference_trace = []
    original = loop._evaluate_policy(
        **small_loop, policy=loop.Policy("persistent", "persistent", True),
        evaluation_mode="same_state", captured_reference_trace=reference_trace,
    )
    gated = loop._evaluate_policy(
        **small_loop, policy=loop.Policy("gated", "persistent", True, response_gate=True),
        evaluation_mode="same_state", reference_trace=reference_trace,
        response_gate_max=0.0,
    )
    for original_rows, gated_rows in zip(original["window_records"], gated["window_records"]):
        for original_row, gated_row in zip(original_rows, gated_rows):
            assert original_row["executed_response_hash"] == gated_row["executed_response_hash"]
            np.testing.assert_allclose(original_row["posterior_mode_probabilities"], gated_row["posterior_mode_probabilities"], atol=1e-6)
            assert gated_row["decision_prefix_nll"] == pytest.approx(original_row["decision_prefix_nll"])
            assert gated_row["belief_evidence_source"] == "ungated_mode_conditional_response"


def test_uniform_counterfactual_recomputes_belief_dependent_gate(small_loop, monkeypatch):
    scoring_calls = []
    original_score = loop.expected_candidate_costs

    def capture_score(mode_means, belief, *arguments, **keywords):
        scoring_calls.append((mode_means.clone(), belief.clone()))
        return original_score(mode_means, belief, *arguments, **keywords)

    monkeypatch.setattr(loop, "expected_candidate_costs", capture_score)
    result = loop._evaluate_policy(
        **small_loop, policy=loop.Policy("gated", "persistent", True, response_gate=True),
        response_gate_belief_threshold=0.1,
        response_gate_uncertainty_threshold=100.0,
    )
    differing_gates = 0
    for start in range(0, len(scoring_calls), 4):
        current, mixture, uniform, prefix = scoring_calls[start:start + 4]
        torch.testing.assert_close(current[0], mixture[0])
        torch.testing.assert_close(current[0], prefix[0])
        torch.testing.assert_close(mixture[1], torch.full_like(mixture[1], 1 / 3))
        torch.testing.assert_close(mixture[1], uniform[1])
        differing_gates += int(not torch.allclose(mixture[0], uniform[0]))
    assert differing_gates > 0
    for rows in result["window_records"]:
        for row in rows:
            assert row["belief_gate_selection_flip"] == float(
                row["mixture_only_uniform_selected_candidate"] != row["uniform_selected_candidate"]
            )


def test_cost_selection_keeps_least_risk_fallback():
    costs = {
        "total": torch.tensor([0.0, -2.0, 1.0]),
        "safe_mask": torch.tensor([False, False, False]),
        "violation_probability": torch.tensor([0.4, 0.7, 0.2]),
    }
    assert loop._select_response_costs(costs, "total") == (2, True)
    costs["safe_mask"][0] = True
    assert loop._select_response_costs(costs, "total") == (0, False)


def test_same_gate_has_uniform_reset_and_persistent_belief_controls():
    policies = loop._configured_policies({
        "additional_policies": [
            "uniform-gated-joint", "reset-gated-joint", "persistent-gated-joint",
        ]
    })
    selected = {policy.name: policy for policy in policies}
    assert selected["uniform-gated-joint"].belief_type == "uniform"
    assert selected["reset-gated-joint"].belief_type == "reset"
    assert selected["persistent-gated-joint"].belief_type == "persistent"
    assert all(selected[name].response_gate for name in (
        "uniform-gated-joint", "reset-gated-joint", "persistent-gated-joint",
    ))


def test_adaptive_likelihood_preserves_evidence_under_large_common_density_offset():
    config = AdaptivePersistentBeliefConfig(max_hazard=0.0, likelihood_scale=0.2)
    prior = torch.tensor([0.2, 0.3, 0.5])
    centered = torch.tensor([0.0, -20.0, -40.0])
    ordinary = AdaptivePersistentBelief(3, prior=prior, config=config)
    shifted = AdaptivePersistentBelief(3, prior=prior, config=config)
    expected = categorical_bayes_update(prior.log(), config.likelihood_scale * centered)
    actual, _ = ordinary.update(centered)
    offset, diagnostic = shifted.update(centered - 1000.0)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(offset, expected)
    assert float(diagnostic["surprise"]) > 1000.0
