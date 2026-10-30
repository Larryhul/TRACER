"""Auditable candidate-score and belief-sensitivity diagnostics.

The planners use hard feasibility masks and, in the external loop, fallback
tiers.  A raw difference between two belief-weighted costs is therefore not
enough to explain a decision.  These helpers keep the decomposition and the
ranking-boundary diagnostics identical across the synthetic and external
evaluators without touching the response models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


PLANNER_COMPONENTS = (
    "weighted_task",
    "weighted_safety",
    "weighted_human",
    "weighted_smoothness",
    "weighted_risk",
    "observable_penalty",
    "candidate_bias",
    "weighted_belief_adjustment",
)

BELIEF_SENSITIVITY_METRICS = (
    "belief_selection_changed",
    "belief_total_score_l1",
    "belief_total_score_max_abs_delta",
    "belief_selected_total_delta",
    "belief_uniform_selected_total_delta",
    "belief_selected_decision_margin",
    "uniform_selected_decision_margin",
    "belief_top2_decision_margin",
    "uniform_top2_decision_margin",
    "belief_common_admissible_fraction",
    "belief_admissibility_changed_fraction",
    "belief_fallback_changed",
    "belief_vertex_selection_possible",
    "belief_vertex_minimum_margin",
    *(f"belief_{name}_l1" for name in PLANNER_COMPONENTS),
)


def mode_cost_selection_envelope(mode_costs, uniform_scores, selected, admissible):
    costs = np.asarray(mode_costs, dtype=np.float64)
    scores = _array(uniform_scores)
    valid = np.asarray(admissible, dtype=bool) & np.isfinite(scores)
    if costs.ndim != 3 or costs.shape[-1] != len(scores) or not np.isfinite(costs).all():
        raise ValueError("mode costs must be finite (humans, modes, candidates) tensors")
    if not 0 <= selected < len(scores) or not valid[selected]:
        raise ValueError("uniform selected candidate must be admissible")
    gaps = costs - costs[..., selected:selected + 1]
    centered = gaps - gaps.mean(axis=1, keepdims=True)
    minimum_margins = scores - scores[selected] + centered.min(axis=1).mean(axis=0)
    alternatives = valid.copy()
    alternatives[selected] = False
    if not alternatives.any():
        return {"belief_vertex_selection_possible": 0., "belief_vertex_minimum_margin": float("nan")}
    minimum = float(minimum_margins[alternatives].min())
    return {"belief_vertex_selection_possible": float(minimum < -1e-6),
            "belief_vertex_minimum_margin": minimum}


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("planner diagnostic values must be one-dimensional")
    return array


def candidate_score_decomposition(
    components: Mapping[str, Any],
    unconstrained_total: Any,
    decision_scores: Any,
    admissible_mask: Any,
    selected: int,
    fallback: bool,
) -> dict[str, np.ndarray | float]:
    """Return candidate-aligned components plus selection-boundary metadata."""

    total = _array(unconstrained_total)
    scores = _array(decision_scores)
    admissible = _array(admissible_mask).astype(bool)
    if total.shape != scores.shape or total.shape != admissible.shape:
        raise ValueError("total, decision scores, and admissible mask must align")
    if not 0 <= int(selected) < len(total):
        raise ValueError("selected candidate must lie in the candidate set")

    result: dict[str, np.ndarray | float] = {
        "unconstrained_total": total,
        "decision_score": scores,
        "admissible_mask": admissible.astype(np.float64),
        "selected_decision_margin": _selected_margin(scores, int(selected)),
        "top2_decision_margin": _top2_margin(scores),
        "finite_decision_fraction": float(np.isfinite(scores).mean()),
        "admissible_fraction": float(admissible.mean()),
        "fallback": float(bool(fallback)),
    }
    for name in PLANNER_COMPONENTS:
        value = components.get(name)
        result[name] = np.zeros_like(total) if value is None else _array(value)
        if np.asarray(result[name]).shape != total.shape:
            raise ValueError(f"component {name} must align with candidates")
    return result


def belief_sensitivity_diagnostics(
    belief: Mapping[str, np.ndarray | float],
    uniform: Mapping[str, np.ndarray | float],
    belief_selected: int,
    uniform_selected: int,
) -> dict[str, float]:
    """Measure whether belief changes a candidate ranking and why.

    All component deltas are evaluated over the identical candidate pool.  The
    decision scores additionally expose masking/fallback effects, which can
    suppress a real response-cost change before it reaches the executor.
    """

    belief_total = _array(belief["unconstrained_total"])
    uniform_total = _array(uniform["unconstrained_total"])
    if belief_total.shape != uniform_total.shape:
        raise ValueError("belief and uniform candidate pools must align")
    if not 0 <= int(belief_selected) < len(belief_total):
        raise ValueError("belief selected candidate is invalid")
    if not 0 <= int(uniform_selected) < len(belief_total):
        raise ValueError("uniform selected candidate is invalid")

    result = {
        "belief_selection_changed": float(int(belief_selected) != int(uniform_selected)),
        "belief_total_score_l1": _mean_absolute_delta(belief_total, uniform_total),
        "belief_total_score_max_abs_delta": _max_absolute_delta(belief_total, uniform_total),
        "belief_selected_total_delta": _point_delta(
            uniform_total, belief_total, int(belief_selected)
        ),
        "belief_uniform_selected_total_delta": _point_delta(
            uniform_total, belief_total, int(uniform_selected)
        ),
        "belief_selected_decision_margin": float(belief["selected_decision_margin"]),
        "uniform_selected_decision_margin": float(uniform["selected_decision_margin"]),
        "belief_top2_decision_margin": float(belief["top2_decision_margin"]),
        "uniform_top2_decision_margin": float(uniform["top2_decision_margin"]),
        "belief_common_admissible_fraction": float(
            np.logical_and(
                _array(belief["admissible_mask"]).astype(bool),
                _array(uniform["admissible_mask"]).astype(bool),
            ).mean()
        ),
        "belief_admissibility_changed_fraction": float(
            np.logical_xor(
                _array(belief["admissible_mask"]).astype(bool),
                _array(uniform["admissible_mask"]).astype(bool),
            ).mean()
        ),
        "belief_fallback_changed": float(
            bool(belief["fallback"]) != bool(uniform["fallback"])
        ),
    }
    for name in PLANNER_COMPONENTS:
        result[f"belief_{name}_l1"] = _mean_absolute_delta(
            _array(belief[name]), _array(uniform[name])
        )
    return result


def _mean_absolute_delta(first: np.ndarray, second: np.ndarray) -> float:
    finite = np.isfinite(first) & np.isfinite(second)
    return float(np.abs(first[finite] - second[finite]).mean()) if finite.any() else float("nan")


def _max_absolute_delta(first: np.ndarray, second: np.ndarray) -> float:
    finite = np.isfinite(first) & np.isfinite(second)
    return float(np.abs(first[finite] - second[finite]).max()) if finite.any() else float("nan")


def _point_delta(first: np.ndarray, second: np.ndarray, index: int) -> float:
    if not np.isfinite(first[index]) or not np.isfinite(second[index]):
        return float("nan")
    return float(first[index] - second[index])


def _selected_margin(scores: np.ndarray, selected: int) -> float:
    selected_score = scores[selected]
    alternatives = np.delete(scores, selected)
    alternatives = alternatives[np.isfinite(alternatives)]
    if not np.isfinite(selected_score) or not len(alternatives):
        return float("nan")
    return float(alternatives.min() - selected_score)


def _top2_margin(scores: np.ndarray) -> float:
    finite = np.sort(scores[np.isfinite(scores)])
    return float(finite[1] - finite[0]) if len(finite) >= 2 else float("nan")
