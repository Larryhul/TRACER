"""Reusable losses and diagnostics for counterfactual candidate ranking.

All costs follow the planner convention: lower is better.  The metrics only
evaluate supplied candidate scores against supplied branch outcomes; they do
not alter a planner decision or count an intervention as an improvement.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as functional


def _validate_costs(
    predicted_costs: torch.Tensor,
    oracle_costs: torch.Tensor,
    valid_mask: torch.Tensor | None,
    tie_tolerance: float,
) -> torch.Tensor:
    if predicted_costs.ndim != 2 or oracle_costs.ndim != 2:
        raise ValueError("predicted_costs and oracle_costs must have shape (batch, candidates)")
    if predicted_costs.shape != oracle_costs.shape:
        raise ValueError("predicted_costs and oracle_costs must have identical shapes")
    if not torch.is_floating_point(predicted_costs) or not torch.is_floating_point(oracle_costs):
        raise ValueError("candidate costs must be floating point tensors")
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be nonnegative")
    finite = torch.isfinite(predicted_costs) & torch.isfinite(oracle_costs)
    if valid_mask is None:
        return finite
    if valid_mask.shape != predicted_costs.shape:
        raise ValueError("valid_mask must have shape (batch, candidates)")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean")
    return finite & valid_mask.to(device=predicted_costs.device)


def _pair_weights(
    pair_weights: torch.Tensor | None,
    shape: torch.Size,
    valid: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    batch, candidates = shape
    if pair_weights is None:
        weights = torch.ones((batch, candidates, candidates), dtype=dtype, device=device)
    else:
        weights = pair_weights.to(device=device, dtype=dtype)
        if weights.shape == (batch, candidates):
            weights = weights.unsqueeze(-1) * weights.unsqueeze(-2)
        elif weights.shape != (batch, candidates, candidates):
            raise ValueError(
                "pair_weights must have shape (batch, candidates) or (batch, candidates, candidates)"
            )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("pair_weights must be finite and nonnegative")
    return weights * (valid.unsqueeze(-1) & valid.unsqueeze(-2)).to(dtype)


def pairwise_ranking_loss(
    predicted_costs: torch.Tensor,
    oracle_costs: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    tie_tolerance: float = 1e-6,
    pair_weights: torch.Tensor | None = None,
    gap_weight_power: float = 0.0,
    margin: float = 0.0,
    temperature: float = 1.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Return a differentiable logistic loss that orders lower oracle costs first.

    Approximate oracle ties are excluded.  ``pair_weights`` can weight each
    candidate (outer-product weighting) or each directed candidate pair.
    ``gap_weight_power`` optionally emphasizes oracle pairs with larger gaps.
    """

    valid = _validate_costs(predicted_costs, oracle_costs, valid_mask, tie_tolerance)
    if gap_weight_power < 0.0 or margin < 0.0:
        raise ValueError("gap_weight_power and margin must be nonnegative")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    safe_predicted = torch.where(valid, predicted_costs, torch.zeros_like(predicted_costs))
    safe_oracle = torch.where(valid, oracle_costs, torch.zeros_like(oracle_costs))
    oracle_delta = safe_oracle.unsqueeze(-1) - safe_oracle.unsqueeze(-2)
    predicted_delta = safe_predicted.unsqueeze(-1) - safe_predicted.unsqueeze(-2)
    upper = torch.triu(torch.ones_like(oracle_delta, dtype=torch.bool), diagonal=1)
    active = upper & (oracle_delta.abs() > tie_tolerance)
    weights = _pair_weights(pair_weights, predicted_costs.shape, valid, predicted_costs.dtype, predicted_costs.device)
    weights = weights * active.to(weights.dtype)
    if gap_weight_power:
        weights = weights * oracle_delta.abs().pow(gap_weight_power)

    correct_signed_gap = oracle_delta.sign() * predicted_delta
    losses = functional.softplus(margin - correct_signed_gap / temperature) * weights
    if reduction == "none":
        return losses
    total = losses.sum()
    if reduction == "sum":
        return total
    denominator = weights.sum()
    return total / denominator.clamp_min(torch.finfo(weights.dtype).tiny)


def _average_ranks(values: torch.Tensor, tie_tolerance: float) -> torch.Tensor:
    """Compute average ranks for one finite one-dimensional cost vector."""

    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and bool((sorted_values[stop] - sorted_values[start]).abs() <= tie_tolerance):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(predicted: torch.Tensor, oracle: torch.Tensor, tie_tolerance: float) -> float:
    predicted_rank = _average_ranks(predicted, tie_tolerance)
    oracle_rank = _average_ranks(oracle, tie_tolerance)
    predicted_centered = predicted_rank - predicted_rank.mean()
    oracle_centered = oracle_rank - oracle_rank.mean()
    denominator = predicted_centered.norm() * oracle_centered.norm()
    if float(denominator) == 0.0:
        return float("nan")
    return float((predicted_centered @ oracle_centered / denominator).item())


def _kendall_tau_b(predicted: torch.Tensor, oracle: torch.Tensor, tie_tolerance: float) -> float:
    predicted_delta = predicted.unsqueeze(-1) - predicted.unsqueeze(-2)
    oracle_delta = oracle.unsqueeze(-1) - oracle.unsqueeze(-2)
    upper = torch.triu(torch.ones_like(predicted_delta, dtype=torch.bool), diagonal=1)
    predicted_tie = predicted_delta.abs() <= tie_tolerance
    oracle_tie = oracle_delta.abs() <= tie_tolerance
    concordant = int((upper & ~predicted_tie & ~oracle_tie & (predicted_delta * oracle_delta > 0)).sum())
    discordant = int((upper & ~predicted_tie & ~oracle_tie & (predicted_delta * oracle_delta < 0)).sum())
    predicted_only_ties = int((upper & predicted_tie & ~oracle_tie).sum())
    oracle_only_ties = int((upper & oracle_tie & ~predicted_tie).sum())
    denominator = ((concordant + discordant + predicted_only_ties) *
                   (concordant + discordant + oracle_only_ties)) ** 0.5
    return (concordant - discordant) / denominator if denominator else float("nan")


@torch.no_grad()
def candidate_ranking_metrics(
    predicted_costs: torch.Tensor,
    oracle_costs: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    tie_tolerance: float = 1e-6,
) -> dict[str, float | int]:
    """Evaluate ranking and decision quality for independent candidate groups.

    ``top1_agreement`` accepts any oracle candidate tied with the oracle best.
    ``selected_candidate_regret`` uses the deterministic first minimum of the
    supplied predicted costs.  Neither metric creates or rewards forced flips.
    """

    valid = _validate_costs(predicted_costs, oracle_costs, valid_mask, tie_tolerance)
    rank_values: list[float] = []
    kendall_values: list[float] = []
    pairwise_correct = 0.0
    pairwise_count = 0
    total_regret = 0.0
    top1_agreements = 0.0
    decision_count = 0

    for row in range(predicted_costs.shape[0]):
        indices = torch.nonzero(valid[row], as_tuple=False).squeeze(-1)
        if not len(indices):
            continue
        predicted = predicted_costs[row, indices]
        oracle = oracle_costs[row, indices]
        selected = int(torch.argmin(predicted))
        oracle_best = oracle.min()
        regret = (oracle[selected] - oracle_best).clamp_min(0.0)
        total_regret += 0.0 if float(regret) <= tie_tolerance else float(regret)
        top1_agreements += float(bool(oracle[selected] <= oracle_best + tie_tolerance))
        decision_count += 1
        if len(indices) < 2:
            continue
        spearman = _spearman(predicted, oracle, tie_tolerance)
        kendall = _kendall_tau_b(predicted, oracle, tie_tolerance)
        if spearman == spearman:
            rank_values.append(spearman)
        if kendall == kendall:
            kendall_values.append(kendall)
        predicted_delta = predicted.unsqueeze(-1) - predicted.unsqueeze(-2)
        oracle_delta = oracle.unsqueeze(-1) - oracle.unsqueeze(-2)
        pairs = torch.triu(torch.ones_like(predicted_delta, dtype=torch.bool), diagonal=1)
        pairs &= oracle_delta.abs() > tie_tolerance
        count = int(pairs.sum())
        if count:
            predicted_tie = predicted_delta.abs() <= tie_tolerance
            correct = (predicted_delta * oracle_delta > 0) & ~predicted_tie
            pairwise_correct += float(correct[pairs].sum()) + 0.5 * float(predicted_tie[pairs].sum())
            pairwise_count += count

    return {
        "rank_spearman": sum(rank_values) / len(rank_values) if rank_values else float("nan"),
        "rank_kendall_tau_b": sum(kendall_values) / len(kendall_values) if kendall_values else float("nan"),
        "pairwise_accuracy": pairwise_correct / pairwise_count if pairwise_count else float("nan"),
        "selected_candidate_regret": total_regret / decision_count if decision_count else float("nan"),
        "top1_agreement": top1_agreements / decision_count if decision_count else float("nan"),
        "rank_correlation_count": len(rank_values),
        "kendall_count": len(kendall_values),
        "pairwise_count": pairwise_count,
        "decision_count": decision_count,
    }
