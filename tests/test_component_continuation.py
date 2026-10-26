import copy

import pytest
import torch

from sage.experiments.audit_component_continuation import compare_continuation, continuation_seed_sequences, first_eligible_disagreements
from sage.experiments.eval_component_factorial import FACTORIAL_POLICIES
from sage.experiments.eval_persistent_closed_loop import _evaluate_policy
from sage.response.closed_loop import ClosedLoopSpec, PlanningCostConfig
from sage.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig


def test_first_disagreement_selection_has_no_outcome_based_filtering():
    first = [[{"state_hash": str(window), "candidate_hash": "same", "selected_candidate": 0} for window in range(5)]]
    second = copy.deepcopy(first)
    second[0][2]["selected_candidate"] = 1
    points = first_eligible_disagreements(first, second, 2)
    assert points == [{"episode": 0, "window": 2, "first_action": 0, "second_action": 1}]
    spec = ClosedLoopSpec(seed=123, split="ood")
    assert continuation_seed_sequences(spec, points, 2) == [[123 + 7_000_003 + 3 * 1009], [123 + 7_000_003 + 4 * 1009]]
    second[0][0]["state_hash"] = "different"
    with pytest.raises(ValueError, match="identical"):
        first_eligible_disagreements(first, second, 2)


@pytest.mark.parametrize("retain_previous_plan", [False, True])
def test_intervention_continuation_preserves_counterfactual_prefix_and_random_streams(retain_previous_plan):
    torch.manual_seed(51)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(history=4, horizon=4, num_robots=2, hidden_dim=16, dropout=0.)).eval()
    shared = dict(model=model, spec=ClosedLoopSpec(history=4, horizon=4, windows=4, num_robots=2, num_candidates=6, execute_steps=1, seed=51), episodes=2, device="cpu", planning_cost=PlanningCostConfig(task_steps=1), likelihood_scale=.05, prefix_objective=True, response_gate_min=1., response_gate_max=1.)
    shared["retain_previous_plan"] = retain_previous_plan
    trace = []
    first = _evaluate_policy(**shared, policy=FACTORIAL_POLICIES[0], captured_reference_trace=trace, evaluation_mode="same_state")
    second = copy.deepcopy(first)
    for rows in second["window_records"]:
        row = rows[0]
        row["selected_candidate"] = (row["selected_candidate"] + 1) % 6
        row["selected_true_cost"] = -row["ground_truth_candidate_utility"][row["selected_candidate"]]
    result = compare_continuation(trace, first, second, shared, 2)
    assert result["summary"]["interventions"] == 2
    assert len(result["continuation_audits"]) == 4
    assert all(row["paired_environment_rng"] for row in result["continuation_audits"])
    assert all(row["retained_plan_present"] == retain_previous_plan for row in result["continuation_audits"])
    same = compare_continuation(trace, first, first, shared, 2)
    assert same["summary"]["combined_cost_difference"]["all_scene_mean"] == 0.
    with pytest.raises(ValueError, match="align"):
        _evaluate_policy(**shared, policy=FACTORIAL_POLICIES[0], candidate_seed_sequences=[[1]])
