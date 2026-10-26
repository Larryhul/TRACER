import math

import pytest
import torch

from sage.response.candidate_ranking import candidate_ranking_metrics, pairwise_ranking_loss


def test_perfect_ranking_has_low_loss_and_perfect_metrics():
    predicted = torch.tensor([[0.1, 0.4, 0.9]], requires_grad=True)
    oracle = torch.tensor([[1.0, 2.0, 3.0]])
    loss = pairwise_ranking_loss(predicted, oracle)
    loss.backward()
    metrics = candidate_ranking_metrics(predicted.detach(), oracle)
    assert loss.item() < 0.6
    assert torch.isfinite(predicted.grad).all()
    assert metrics["rank_spearman"] == pytest.approx(1.0)
    assert metrics["rank_kendall_tau_b"] == pytest.approx(1.0)
    assert metrics["pairwise_accuracy"] == pytest.approx(1.0)
    assert metrics["selected_candidate_regret"] == pytest.approx(0.0)
    assert metrics["top1_agreement"] == pytest.approx(1.0)


def test_reversed_ranking_is_penalized_and_has_regret():
    oracle = torch.tensor([[0.0, 1.0, 2.0]])
    correct = pairwise_ranking_loss(torch.tensor([[0.0, 1.0, 2.0]]), oracle)
    reversed_loss = pairwise_ranking_loss(torch.tensor([[2.0, 1.0, 0.0]]), oracle)
    metrics = candidate_ranking_metrics(torch.tensor([[2.0, 1.0, 0.0]]), oracle)
    assert reversed_loss > correct
    assert metrics["rank_spearman"] == pytest.approx(-1.0)
    assert metrics["rank_kendall_tau_b"] == pytest.approx(-1.0)
    assert metrics["pairwise_accuracy"] == pytest.approx(0.0)
    assert metrics["selected_candidate_regret"] == pytest.approx(2.0)
    assert metrics["top1_agreement"] == pytest.approx(0.0)


def test_oracle_ties_are_excluded_and_top1_accepts_tied_oracle_best():
    predicted = torch.tensor([[0.2, 0.0, 0.3]])
    oracle = torch.tensor([[1.0, 1.0 + 5e-7, 3.0]])
    loss = pairwise_ranking_loss(predicted, oracle, tie_tolerance=1e-6)
    metrics = candidate_ranking_metrics(predicted, oracle, tie_tolerance=1e-6)
    assert torch.isfinite(loss)
    assert metrics["pairwise_count"] == 2
    assert metrics["selected_candidate_regret"] == pytest.approx(0.0)
    assert metrics["top1_agreement"] == pytest.approx(1.0)


def test_invalid_candidates_are_excluded_from_loss_selection_and_correlation():
    predicted = torch.tensor([[1.0, -100.0, 3.0, float("nan")]])
    oracle = torch.tensor([[2.0, -50.0, 0.0, 5.0]])
    mask = torch.tensor([[True, False, True, True]])
    loss = pairwise_ranking_loss(predicted, oracle, valid_mask=mask)
    metrics = candidate_ranking_metrics(predicted, oracle, valid_mask=mask)
    assert torch.isfinite(loss)
    assert metrics["decision_count"] == 1
    assert metrics["selected_candidate_regret"] == pytest.approx(2.0)
    assert metrics["top1_agreement"] == pytest.approx(0.0)
    assert metrics["pairwise_count"] == 1


def test_single_or_empty_candidate_groups_are_safe_and_do_not_create_pairs():
    predicted = torch.tensor([[4.0], [float("nan")]], requires_grad=True)
    oracle = torch.tensor([[3.0], [1.0]])
    loss = pairwise_ranking_loss(predicted, oracle)
    loss.backward()
    metrics = candidate_ranking_metrics(predicted.detach(), oracle)
    assert loss.item() == pytest.approx(0.0)
    assert predicted.grad.tolist() == [[0.0], [0.0]]
    assert metrics["decision_count"] == 1
    assert metrics["pairwise_count"] == 0
    assert math.isnan(metrics["pairwise_accuracy"])
    assert math.isnan(metrics["rank_spearman"])
    assert metrics["selected_candidate_regret"] == pytest.approx(0.0)


def test_candidate_and_direct_pair_weights_and_gap_weighting_are_supported():
    predicted = torch.tensor([[1.0, 0.0, 2.0]], requires_grad=True)
    oracle = torch.tensor([[0.0, 1.0, 3.0]])
    candidate_weighted = pairwise_ranking_loss(
        predicted, oracle, pair_weights=torch.tensor([[0.0, 1.0, 1.0]])
    )
    direct_weights = torch.zeros(1, 3, 3)
    direct_weights[0, 0, 1] = 1.0
    direct_weighted = pairwise_ranking_loss(predicted, oracle, pair_weights=direct_weights)
    gap_weighted = pairwise_ranking_loss(predicted, oracle, gap_weight_power=1.0)
    assert candidate_weighted < direct_weighted
    assert torch.isfinite(gap_weighted)
    (candidate_weighted + direct_weighted + gap_weighted).backward()
    assert torch.isfinite(predicted.grad).all()


@pytest.mark.parametrize("kwargs", [
    {"tie_tolerance": -1.0},
    {"margin": -1.0},
    {"gap_weight_power": -1.0},
    {"reduction": "median"},
])
def test_loss_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        pairwise_ranking_loss(torch.zeros(1, 2), torch.zeros(1, 2), **kwargs)


def test_cost_shape_and_mask_contracts_are_checked():
    with pytest.raises(ValueError, match="shape"):
        pairwise_ranking_loss(torch.zeros(2), torch.zeros(2))
    with pytest.raises(ValueError, match="identical"):
        candidate_ranking_metrics(torch.zeros(1, 2), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="boolean"):
        candidate_ranking_metrics(torch.zeros(1, 2), torch.zeros(1, 2), valid_mask=torch.ones(1, 2))


def test_near_tied_prediction_receives_half_credit_not_one_and_a_half():
    metrics = candidate_ranking_metrics(torch.tensor([[0., 5e-7]]), torch.tensor([[0., 1.]]))
    assert metrics["pairwise_accuracy"] == pytest.approx(.5)


def test_weighted_mean_is_invariant_to_uniform_weight_scaling():
    predicted = torch.tensor([[0., 1.]])
    oracle = torch.tensor([[0., 1.]])
    original = pairwise_ranking_loss(predicted, oracle)
    scaled = pairwise_ranking_loss(predicted, oracle, pair_weights=torch.full((1, 2, 2), .01))
    torch.testing.assert_close(original, scaled)
