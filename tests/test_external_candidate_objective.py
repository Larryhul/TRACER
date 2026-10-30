import numpy as np
import pytest
import torch

from tracer.experiments.train_rolling_external_belief import grouped_candidate_costs
from tracer.response.candidate_ranking import pairwise_ranking_loss
from tracer.response.closed_loop import PlanningCostConfig, expected_candidate_costs


def _group():
    candidates = np.zeros((3, 1, 2, 4), dtype=np.float32)
    candidates[:2, ..., 2] = 1.
    data = {"case": np.ones(3), "human": np.zeros(3), "window": np.zeros(3),
            "branch": np.array([2, 5, 9]), "reference_branch": np.full(3, 9),
            "candidates": candidates}
    means = torch.zeros(3, 2, 2, 2)
    means[0, 0, :, 0] = 1.
    means[0, 1, :, 0] = -1.
    means[1, :, :, 0] = .5
    return data, means.requires_grad_(), means.detach()[:, 0].clone()


def test_candidate_objective_matches_planner_expectation_not_cost_of_mean():
    data, means, target = _group()
    predicted, oracle, valid = grouped_candidate_costs(
        data, np.arange(3), {"mean": means}, target, "cpu", .25
    )
    planned = expected_candidate_costs(
        means, torch.full((2,), .5), torch.zeros(3, 1, 4),
        torch.from_numpy(data["candidates"]), means[2:].expand_as(means), .25,
        PlanningCostConfig(risk_weight=0.),
    )
    torch.testing.assert_close(predicted[0, :2], planned["human"][:2])
    torch.testing.assert_close(predicted[0, :2], torch.tensor([1., .25]))
    torch.testing.assert_close(oracle[0, :2], target[:2].square().sum(-1).mean(-1))
    assert valid.tolist() == [[True, True, False]]
    loss = pairwise_ranking_loss(predicted, oracle, valid_mask=valid, temperature=.05)
    loss.backward()
    assert torch.isfinite(means.grad).all()


def test_conditional_ranking_supervises_only_the_observed_mode():
    data, means, target = _group()
    predicted, _, valid = grouped_candidate_costs(
        data, np.arange(3), {"mean": means}, target, "cpu", .25,
        mode_weights=torch.tensor([[1., 0.]]).expand(3, -1),
    )
    predicted[valid].sum().backward()
    assert means.grad[:, 1].abs().sum() == 0.
    assert means.grad[:, 0].abs().sum() > 0.


def test_same_state_weights_cannot_depend_on_candidate():
    data, means, target = _group()
    with pytest.raises(ValueError, match="same decision-time"):
        grouped_candidate_costs(data, np.arange(3), {"mean": means}, target, "cpu", .25,
                                mode_weights=torch.tensor([[1., 0.], [0., 1.], [1., 0.]]))


def test_branch_number_does_not_prove_stationary_reference():
    data, means, target = _group()
    data["candidates"][2, ..., 2] = .1
    with pytest.raises(ValueError, match="executed stationary"):
        grouped_candidate_costs(data, np.arange(3), {"mean": means}, target, "cpu", .25)


def test_ranking_temperature_rescales_prediction_gap_without_changing_labels():
    predicted = torch.tensor([[.02, .0]], requires_grad=True)
    oracle = torch.tensor([[.1, .0]])
    raw = pairwise_ranking_loss(predicted, oracle)
    scaled = pairwise_ranking_loss(predicted, oracle, temperature=.05)
    assert scaled < raw
    scaled.backward()
    assert predicted.grad[0, 0] < 0 and predicted.grad[0, 1] > 0


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf")])
def test_ranking_temperature_must_be_positive_and_finite(temperature):
    with pytest.raises(ValueError, match="temperature"):
        pairwise_ranking_loss(torch.zeros(1, 2), torch.ones(1, 2), temperature=temperature)
