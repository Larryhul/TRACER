import math

import numpy as np
import pytest
import torch

from tracer.experiments import eval_persistent_closed_loop as loop
from tracer.experiments.planner_sensitivity import (
    belief_sensitivity_diagnostics,
    candidate_score_decomposition,
    mode_cost_selection_envelope,
)
from tracer.response.closed_loop import ClosedLoopSpec, PlanningCostConfig
from tracer.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
)


def _decomposition(total, scores, mask, selected, human):
    return candidate_score_decomposition(
        {"weighted_task": np.array([-1.0, -1.0, -1.0]), "weighted_human": human},
        np.asarray(total),
        np.asarray(scores),
        np.asarray(mask),
        selected,
        fallback=False,
    )


def test_belief_sensitivity_separates_cost_shift_from_feasibility_change():
    belief = _decomposition(
        total=[2.0, 0.8, 1.2], scores=[np.inf, 0.8, 1.2],
        mask=[False, True, True], selected=1, human=[0.0, 0.2, 0.4],
    )
    uniform = _decomposition(
        total=[2.0, 1.1, 0.7], scores=[np.inf, 1.1, 0.7],
        mask=[False, True, True], selected=2, human=[0.0, 0.5, 0.1],
    )
    diagnostic = belief_sensitivity_diagnostics(belief, uniform, 1, 2)

    assert diagnostic["belief_selection_changed"] == 1.0
    assert diagnostic["belief_admissibility_changed_fraction"] == 0.0
    assert diagnostic["belief_common_admissible_fraction"] == 2 / 3
    assert diagnostic["belief_weighted_task_l1"] == 0.0
    assert diagnostic["belief_weighted_human_l1"] > 0.0
    assert diagnostic["belief_selected_decision_margin"] == pytest.approx(0.4)
    assert diagnostic["uniform_selected_decision_margin"] == pytest.approx(0.4)


def test_vertex_envelope_detects_reachable_ranking_change():
    costs = np.array([[[0., .1], [0., 1.]]])
    scores = np.array([0., .2])
    result = mode_cost_selection_envelope(costs, scores, 0, [True, True])
    assert result["belief_vertex_selection_possible"] == 1.
    assert result["belief_vertex_minimum_margin"] == pytest.approx(-.25)


def test_common_mode_shift_cannot_change_action_order():
    costs = np.array([[[0., 1.], [10., 11.]]])
    result = mode_cost_selection_envelope(costs, [0., .2], 0, [True, True])
    assert result["belief_vertex_selection_possible"] == 0.
    assert result["belief_vertex_minimum_margin"] == pytest.approx(.2)


def test_vertex_envelope_does_not_include_masked_alternatives():
    costs = np.array([[[0., -100.], [0., 100.]]])
    result = mode_cost_selection_envelope(costs, [0., np.inf], 0, [True, False])
    assert result["belief_vertex_selection_possible"] == 0.


def test_fallback_decomposition_matches_synthetic_selection_score():
    costs = {
        "total": torch.tensor([0.0, -4.0]),
        "unconstrained_total": torch.tensor([0.0, -4.0]),
        "task": torch.tensor([0.1, 0.2]),
        "safety": torch.tensor([0.3, 0.4]),
        "human": torch.tensor([0.5, 0.6]),
        "smoothness": torch.tensor([0.0, 0.0]),
        "violation_probability": torch.tensor([0.7, 0.2]),
        "safe_mask": torch.tensor([False, False]),
    }
    selected, fallback = loop._select_response_costs(costs, "total")
    decomposition = loop._synthetic_score_decomposition(
        costs, PlanningCostConfig(), "total", selected
    )

    assert selected == 1
    assert fallback
    assert decomposition["fallback"] == 1.0
    np.testing.assert_allclose(decomposition["decision_score"], [0.7, 0.2])


def test_synthetic_loop_records_planner_sensitivity_metrics():
    torch.manual_seed(17)
    spec = ClosedLoopSpec(
        windows=2, history=4, horizon=4, num_robots=2, num_candidates=6,
        execute_steps=1, dt=0.4, seed=17,
    )
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=4, num_robots=2,
        hidden_dim=16, num_heads=2, dropout=0.0, dt=0.4,
    )).eval()
    result = loop._evaluate_policy(
        model, loop.Policy("persistent", "persistent", True), spec, episodes=1,
        device="cpu", planning_cost=PlanningCostConfig(task_steps=1, response_steps=1),
        likelihood_scale=0.2,
    )

    row = result["window_records"][0][0]
    assert "belief_total_score_l1" in row
    assert "belief_selected_decision_margin" in row
    assert "belief_weighted_human_l1" in row
    assert math.isfinite(row["belief_total_score_l1"])
    assert result["belief_total_score_l1"]["count"] == 1
