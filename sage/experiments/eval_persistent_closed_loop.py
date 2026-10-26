from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from sage.belief.filter import (
    adaptive_categorical_update,
    apply_belief_forgetting,
    categorical_bayes_update,
    uniform_log_belief,
)
from sage.belief.adaptive_persistent import (
    AdaptivePersistentBelief,
    AdaptivePersistentBeliefConfig,
)
from sage.belief.correlated_emission import CorrelatedEmission
from sage.experiments.formal_manifest import candidate_pool_sha256
from sage.experiments.planner_sensitivity import (
    belief_sensitivity_diagnostics,
    candidate_score_decomposition,
)
from sage.experiments.closed_loop_protocol import (
    ReferenceWindow,
    audit_candidate_protocol,
    prepare_manifest,
)
from sage.response.closed_loop_contract import (
    observable_state_sha256,
    task_goals,
    validate_candidate_state,
)
from sage.response.closed_loop import (
    ClosedLoopSpec,
    PlanningCostConfig,
    SyntheticClosedLoopSimulator,
    expected_candidate_costs,
)
from sage.response.plan_retention import plan_context_sha256, retained_candidate_index, retain_plan_candidate, shifted_plan_candidate
from sage.response.continuation_value import FrozenContinuationValue, add_continuation_value
from sage.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    TrajectoryRecurrentBelief,
    TrajectoryRecurrentConfig,
    stationary_reference_candidates,
    trajectory_mode_log_likelihood,
)
from sage.response.residual_gate import (
    ResidualGateConfig,
    apply_joint_residual_gate,
    compute_belief_confidence,
    compute_joint_residual_gate,
)
from sage.response.sage_candidate_adapter import (
    SAGECandidateAdapter,
    SAGECandidateAdapterConfig,
)


@dataclass(frozen=True)
class Policy:
    name: str
    belief_type: str
    include_pair_interactions: bool
    selection_objective: str = "total"
    response_gate: bool = False


POLICIES = (
    Policy("task-only", "uniform", True, "task"),
    Policy("no-belief-joint", "uniform", True),
    Policy("reset-additive", "reset", False),
    Policy("reset-joint", "reset", True),
    Policy("persistent-additive", "persistent", False),
    Policy("persistent-joint", "persistent", True),
    Policy("oracle-joint", "oracle", True),
)


OPTIONAL_POLICIES = {
    "recurrent-joint": Policy("recurrent-joint", "recurrent", True),
    "adaptive-joint": Policy("adaptive-joint", "adaptive", True),
    "adaptive-additive": Policy("adaptive-additive", "adaptive", False),
    "adaptive-gated-joint": Policy(
        "adaptive-gated-joint", "adaptive", True, response_gate=True
    ),
    "uniform-gated-joint": Policy(
        "uniform-gated-joint", "uniform", True, response_gate=True
    ),
    "reset-gated-joint": Policy(
        "reset-gated-joint", "reset", True, response_gate=True
    ),
    "persistent-gated-joint": Policy(
        "persistent-gated-joint", "persistent", True, response_gate=True
    ),
    "adaptive-persistent-joint": Policy(
        "adaptive-persistent-joint", "adaptive-persistent", True
    ),
    "adaptive-persistent-gated-joint": Policy(
        "adaptive-persistent-gated-joint",
        "adaptive-persistent",
        True,
        response_gate=True,
    ),
}

CONTRACT_METRICS = (
    "candidate_position_error", "candidate_velocity_error", "candidate_integration_error",
    "execution_position_error", "executed_total_cost", "executed_task_progress",
    "executed_collision", "goal_distance", "goal_reached",
)
EPISODE_OUTCOME_METRICS = (
    "cumulative_executed_cost", "episode_collision", "terminal_goal_distance",
    "terminal_goal_success",
)
BELIEF_DIAGNOSTIC_METRICS = (
    "decision_mode_accuracy", "decision_mode_nll",
    "posterior_mode_accuracy", "posterior_mode_nll",
    "belief_action_flip", "belief_mixture_selection_flip",
    "belief_gate_selection_flip", "belief_gate_cost_delta",
    "decision_prefix_nll", "planning_prefix_nll", "planning_prefix_ade",
    "belief_selection_changed",
    "belief_total_score_l1", "belief_total_score_max_abs_delta",
    "belief_selected_total_delta", "belief_uniform_selected_total_delta",
    "belief_selected_decision_margin", "uniform_selected_decision_margin",
    "belief_top2_decision_margin", "uniform_top2_decision_margin",
    "belief_common_admissible_fraction", "belief_admissibility_changed_fraction",
    "belief_fallback_changed", "belief_weighted_task_l1",
    "belief_weighted_safety_l1", "belief_weighted_human_l1",
    "belief_weighted_smoothness_l1", "belief_weighted_risk_l1",
    "belief_observable_penalty_l1", "belief_candidate_bias_l1",
    "belief_weighted_belief_adjustment_l1",
)


def _load_model(
    checkpoint_path: Path,
    device: str,
    expected_dt: float | None = None,
) -> BeliefConditionedTeamResponseModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = PersistentTrajectoryModelConfig(**checkpoint["model_config"])
    if expected_dt is not None:
        expected_dt = float(expected_dt)
        if expected_dt <= 0.0:
            raise ValueError("expected_dt must be positive")
        # The model derives history velocities from position differences using
        # config.dt. External simulators may run at a different fixed rate, so
        # align this inference-time conversion with the actual environment.
        config = dataclass_replace(config, dt=expected_dt)
    model = BeliefConditionedTeamResponseModel(config).to(device)
    model.training_dt = float(checkpoint["model_config"].get("dt", 0.4))
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _load_recurrent_model(
    checkpoint_path: Path, device: str
) -> TrajectoryRecurrentBelief:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TrajectoryRecurrentConfig(**checkpoint["model_config"])
    model = TrajectoryRecurrentBelief(config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _configured_policies(raw: dict[str, Any]) -> tuple[Policy, ...]:
    policies = list(POLICIES)
    for name in raw.get("additional_policies", []):
        if name not in OPTIONAL_POLICIES:
            raise ValueError(f"unknown additional closed-loop policy: {name}")
        if name not in {policy.name for policy in policies}:
            policies.append(OPTIONAL_POLICIES[name])
    return tuple(policies)


def _mean_std(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    return {
        "mean": statistics.fmean(finite),
        "std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "count": len(finite),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = _rankdata(first)
    second_rank = _rankdata(second)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def _kendall_tau_b(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_finite = np.isfinite(first)
    second_finite = np.isfinite(second)
    if not first_finite.all():
        finite_values = first[first_finite]
        replacement = (
            float(finite_values.max()) + 1.0
            if finite_values.size
            else 0.0
        )
        first = np.where(first_finite, first, replacement)
    if not second_finite.all():
        finite_values = second[second_finite]
        replacement = (
            float(finite_values.max()) + 1.0
            if finite_values.size
            else 0.0
        )
        second = np.where(second_finite, second, replacement)
    concordant = 0
    discordant = 0
    ties_first = 0
    ties_second = 0
    for left in range(first.size):
        for right in range(left + 1, first.size):
            first_sign = int(np.sign(first[left] - first[right]))
            second_sign = int(np.sign(second[left] - second[right]))
            if first_sign == 0 and second_sign == 0:
                continue
            if first_sign == 0:
                ties_first += 1
            elif second_sign == 0:
                ties_second += 1
            elif first_sign == second_sign:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_first)
        * (concordant + discordant + ties_second)
    )
    if denominator <= 1e-12:
        return 0.0
    return float((concordant - discordant) / denominator)


def _realized_total_cost(
    realized: Any,
    spec: ClosedLoopSpec,
    planning_cost: PlanningCostConfig,
) -> float:
    normalized_task = -realized.task_progress / (
        spec.execute_steps * spec.dt * planning_cost.task_speed_scale
    )
    return float(
        planning_cost.task_weight * normalized_task
        + planning_cost.safety_weight * realized.safety_cost
        + planning_cost.human_weight * realized.human_cost
        + planning_cost.smoothness_weight * getattr(realized, "smoothness_cost", 0.0)
        + planning_cost.risk_weight * realized.collision
    )


def _candidate_and_reference_outputs(
    model: BeliefConditionedTeamResponseModel,
    history: torch.Tensor,
    current: torch.Tensor,
    candidates: torch.Tensor,
    robot_mask: torch.Tensor,
    include_pair_interactions: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    output = model(
        history,
        current[:, None],
        candidates[:, None],
        robot_mask=robot_mask,
        include_pair_interactions=include_pair_interactions,
    )
    reference_candidates = stationary_reference_candidates(
        current[:, None], model.config.horizon
    )
    reference_output = model(
        history,
        current[:, None],
        reference_candidates,
        robot_mask=robot_mask,
        include_pair_interactions=include_pair_interactions,
    )
    return output, reference_output


def _blend_gated_response_outputs(
    additive_output: dict[str, torch.Tensor],
    joint_output: dict[str, torch.Tensor],
    belief: torch.Tensor,
    belief_threshold: float,
    belief_scale: float,
    uncertainty_threshold: float,
    uncertainty_scale: float,
    min_gate: float = 0.0,
    max_gate: float = 1.0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend the joint residual only when belief and uncertainty support it."""

    additive_mean = additive_output["mean"]
    joint_mean = joint_output["mean"]
    additive_log_std = additive_output["log_std"]
    joint_log_std = joint_output["log_std"]
    if additive_mean.shape != joint_mean.shape or additive_log_std.shape != joint_log_std.shape:
        raise ValueError("additive and joint response outputs must have matching shapes")
    if belief.ndim != 1 or belief.numel() < 2:
        raise ValueError("belief must be a one-dimensional categorical distribution")
    if belief_scale <= 0.0 or uncertainty_scale <= 0.0:
        raise ValueError("response gate scales must be positive")
    if not torch.isfinite(belief).all() or (belief < 0.0).any():
        raise ValueError("belief must be finite and non-negative")

    gate_config = ResidualGateConfig(
        confidence_threshold=float(belief_threshold),
        confidence_scale=float(belief_scale),
        uncertainty_threshold=float(uncertainty_threshold),
        uncertainty_scale=float(uncertainty_scale),
        min_gate=float(min_gate),
        max_gate=float(max_gate),
    )
    candidate_count = joint_mean.shape[0]
    candidate_belief = belief.expand(candidate_count, -1)
    joint_std = joint_log_std.exp()
    gate = compute_joint_residual_gate(candidate_belief, joint_std, gate_config)
    belief_confidence = torch.sigmoid(
        (
            compute_belief_confidence(candidate_belief)
            - float(belief_threshold)
        )
        / float(belief_scale)
    )
    joint_std_mean = joint_std.mean(dim=tuple(range(1, joint_std.ndim)))
    uncertainty_confidence = torch.sigmoid(
        (float(uncertainty_threshold) - joint_std_mean)
        / float(uncertainty_scale)
    )
    blended_mean = apply_joint_residual_gate(additive_mean, joint_mean, gate)
    blended_log_std = torch.maximum(additive_log_std, joint_log_std)
    blended = dict(joint_output)
    blended["mean"] = blended_mean
    blended["log_std"] = blended_log_std
    return blended, gate, belief_confidence.expand_as(gate), uncertainty_confidence


def _select_response_costs(costs: dict[str, torch.Tensor], objective: str) -> tuple[int, bool]:
    if objective not in costs:
        raise ValueError(f"unknown selection objective: {objective}")
    safe_mask = costs.get("safe_mask")
    if safe_mask is not None and not bool(safe_mask.any()):
        return int(costs["violation_probability"].argmin()), True
    scores = costs[objective]
    if safe_mask is not None:
        scores = torch.where(safe_mask, scores, torch.full_like(scores, float("inf")))
    return int(scores.argmin()), False


def _prefix_objective_with_full_safety(
    prefix_costs: dict[str, torch.Tensor], full_costs: dict[str, torch.Tensor],
    risk_weight: float,
) -> dict[str, torch.Tensor]:
    result = dict(prefix_costs)
    result["safe_mask"] = full_costs["safe_mask"]
    result["violation_probability"] = full_costs["violation_probability"]
    total = prefix_costs["unconstrained_total"] + risk_weight * (
        full_costs["violation_probability"] - prefix_costs["violation_probability"]
    )
    result["unconstrained_total"] = total
    result["total"] = torch.where(result["safe_mask"], total, torch.full_like(total, float("inf")))
    return result


def _decision_scores(costs: dict[str, torch.Tensor], objective: str) -> tuple[torch.Tensor, bool]:
    """Expose the exact masked score vector consumed by `_select_response_costs`."""

    if objective not in costs:
        raise ValueError(f"unknown selection objective: {objective}")
    safe_mask = costs.get("safe_mask")
    if safe_mask is not None and not bool(safe_mask.any()):
        return costs["violation_probability"], True
    scores = costs[objective]
    if safe_mask is not None:
        scores = torch.where(safe_mask, scores, torch.full_like(scores, float("inf")))
    return scores, False


def _synthetic_score_decomposition(
    costs: dict[str, torch.Tensor],
    planning_cost: PlanningCostConfig,
    objective: str,
    selected: int,
) -> dict[str, np.ndarray | float]:
    decision_scores, fallback = _decision_scores(costs, objective)
    return candidate_score_decomposition(
        {
            "weighted_task": planning_cost.task_weight * costs["task"],
            "weighted_safety": planning_cost.safety_weight * costs["safety"],
            "weighted_human": planning_cost.human_weight * costs["human"],
            "weighted_smoothness": planning_cost.smoothness_weight * costs["smoothness"],
            "weighted_risk": planning_cost.risk_weight * costs["violation_probability"],
            "candidate_bias": costs.get("continuation_adjustment", torch.zeros_like(decision_scores)),
        },
        costs["unconstrained_total"],
        decision_scores,
        costs.get("safe_mask", torch.ones_like(decision_scores, dtype=torch.bool)),
        selected,
        fallback,
    )


@torch.no_grad()
def _recurrent_belief(
    recurrent_model: TrajectoryRecurrentBelief,
    currents: list[torch.Tensor],
    candidates: list[torch.Tensor],
    responses: list[torch.Tensor],
    histories: list[torch.Tensor],
    device: str,
) -> torch.Tensor:
    if not currents:
        return uniform_log_belief(recurrent_model.config.num_modes, device=device).exp()
    current = torch.stack(currents, dim=0)[None].to(device)
    candidate = torch.stack(candidates, dim=0)[None].to(device)
    response = torch.stack(responses, dim=0)[None].to(device)
    entity_history = torch.stack(histories, dim=0)[None].to(device)
    output = recurrent_model(
        current,
        candidate,
        response,
        entity_history=entity_history
        if recurrent_model.config.use_entity_history
        else None,
    )
    return output[0, -1].exp()


@torch.no_grad()
def _evaluate_policy(
    model: BeliefConditionedTeamResponseModel,
    policy: Policy,
    spec: ClosedLoopSpec,
    episodes: int,
    device: str,
    planning_cost: PlanningCostConfig,
    likelihood_scale: float,
    adaptive_base_hazard: float = 0.0,
    adaptive_max_hazard: float = 0.10,
    adaptive_surprise_threshold: float = 2.0,
    adaptive_surprise_scale: float = 0.10,
    adaptive_hard_reset: bool = False,
    adaptive_threshold_hysteresis: float = 0.0,
    adaptive_cooldown_steps: int = 0,
    response_gate_belief_threshold: float = 0.35,
    response_gate_belief_scale: float = 0.10,
    response_gate_uncertainty_threshold: float = 0.45,
    response_gate_uncertainty_scale: float = 0.08,
    response_gate_min: float = 0.0,
    response_gate_max: float = 1.0,
    candidate_adapter: SAGECandidateAdapter | None = None,
    recurrent_model: TrajectoryRecurrentBelief | None = None,
    simulator_factory: Callable[[ClosedLoopSpec, int], Any] = SyntheticClosedLoopSimulator,
    shared_candidate_pools: list[list[Any]] | None = None,
    captured_candidate_pools: list[list[Any]] | None = None,
    evaluation_mode: str = "closed_loop",
    reference_trace: list[list[ReferenceWindow]] | None = None,
    captured_reference_trace: list[list[ReferenceWindow]] | None = None,
    prefix_objective: bool = False,
    persistent_forgetting_rate: float = 0.0,
    emission_model: BeliefConditionedTeamResponseModel | None = None,
    candidate_seed_sequences: list[list[int]] | None = None,
    retain_previous_plan: bool = False,
    initial_previous_plans: list[torch.Tensor] | None = None,
    continuation_value_model: FrozenContinuationValue | None = None,
    continuation_value_weight: float = 1.0,
    emission_noise_model: CorrelatedEmission | None = None,
) -> dict[str, Any]:
    if shared_candidate_pools is not None or captured_candidate_pools is not None:
        raise ValueError("bare cross-policy candidate reuse is invalid; use same_state reference replay")
    if evaluation_mode not in {"closed_loop", "same_state"}:
        raise ValueError("evaluation_mode must be 'closed_loop' or 'same_state'")
    if reference_trace is not None and evaluation_mode != "same_state":
        raise ValueError("reference replay is only valid in same_state diagnostics")
    if evaluation_mode == "same_state" and reference_trace is None and captured_reference_trace is None:
        raise ValueError("same_state diagnostics require a reference trace")
    if episodes < 1 or not np.isfinite(likelihood_scale) or likelihood_scale <= 0.0:
        raise ValueError("episodes and likelihood_scale must be positive")
    if emission_noise_model is not None and (emission_noise_model.execute_steps != spec.execute_steps or emission_noise_model.dt != spec.dt):
        raise ValueError("correlated emission clock must match actual execution")
    if not np.isfinite(continuation_value_weight) or continuation_value_weight < 0:
        raise ValueError("continuation value weight must be finite and nonnegative")
    if continuation_value_model is not None:
        if not prefix_objective or retain_previous_plan or spec.task_type != "entity_approach":
            raise ValueError("continuation value requires a prefix objective and memoryless entity-approach candidates")
        if (continuation_value_model.total_windows, continuation_value_model.execute_steps, continuation_value_model.dt) != (spec.windows, spec.execute_steps, spec.dt):
            raise ValueError("continuation value horizon and execution timestep must match")
    if initial_previous_plans is not None and (not retain_previous_plan or len(initial_previous_plans) != episodes):
        raise ValueError("initial previous plans require retention and must align with episodes")
    if candidate_seed_sequences is not None:
        if len(candidate_seed_sequences) != spec.windows or any(len(row) != episodes for row in candidate_seed_sequences):
            raise ValueError("candidate seed sequences must align with windows and episodes")
        if any(not isinstance(value, int) or value < 0 for row in candidate_seed_sequences for value in row):
            raise ValueError("candidate seed sequences must contain nonnegative integers")
    if not np.isfinite(persistent_forgetting_rate) or not 0.0 <= persistent_forgetting_rate <= 1.0:
        raise ValueError("persistent forgetting rate must lie in [0, 1]")
    if not math.isclose(model.config.dt, spec.dt):
        raise ValueError("response checkpoint dt must match the simulator")
    if model.config.horizon != spec.horizon or model.config.num_robots != spec.num_robots:
        raise ValueError("response checkpoint horizon/robots must match the simulator")
    if emission_model is not None:
        if emission_model.config != model.config:
            raise ValueError("planning and emission model configurations must match")
        emission_state = emission_model.state_dict()
        for name, value in model.state_dict().items():
            if not name.startswith(("pair_encoder.", "pair_decoder.", "set_encoder.", "set_decoder.")):
                if not torch.equal(value, emission_state[name]):
                    raise ValueError("planning and emission branches must share frozen noninteraction parameters")
    prefix_cost_config = dataclass_replace(
        planning_cost, task_steps=spec.execute_steps, response_steps=spec.execute_steps
    )
    episode_metric_names = (
        "realized_total_cost",
        "task_progress",
        "safety_cost",
        "human_cost",
        "min_distance",
        "collision_rate",
        "mean_activity",
        "oracle_regret",
        "oracle_hit_rate",
        "candidate_rank_spearman",
        "candidate_rank_kendall",
        "task_feasible_candidate_count",
        "task_feasible_candidate_fraction",
        "candidate_true_cost_range",
        "candidate_response_spread",
        "belief_entropy",
        "belief_max_probability",
        "response_gate_weight",
        "response_gate_belief_confidence",
        "response_gate_uncertainty_confidence",
        "belief_selection_flip",
        "belief_selection_predicted_cost_delta",
        "belief_selection_cost_delta",
        "belief_selection_oracle_regret_delta",
        "belief_flip_predicted_cost_delta",
        "belief_flip_true_cost_delta",
        "belief_flip_oracle_regret_delta",
        "safety_filter_rejections",
        "safety_fallback",
        "adaptive_hazard",
        "adaptive_surprise",
        "adaptive_change_point",
    )
    metrics: dict[str, list[float]] = {
        "realized_total_cost": [],
        "task_progress": [],
        "safety_cost": [],
        "human_cost": [],
        "min_distance": [],
        "collision_rate": [],
        "mean_activity": [],
        "oracle_regret": [],
        "oracle_hit_rate": [],
        "candidate_rank_spearman": [],
        "candidate_rank_kendall": [],
        "task_feasible_candidate_count": [],
        "task_feasible_candidate_fraction": [],
        "candidate_true_cost_range": [],
        "candidate_response_spread": [],
        "belief_entropy": [],
        "belief_max_probability": [],
        "belief_selection_flip": [],
        "belief_selection_predicted_cost_delta": [],
        "belief_selection_cost_delta": [],
        "belief_selection_oracle_regret_delta": [],
        "belief_flip_predicted_cost_delta": [],
        "belief_flip_true_cost_delta": [],
        "belief_flip_oracle_regret_delta": [],
        "safety_filter_rejections": [],
        "safety_fallback": [],
        "adaptive_hazard": [],
        "adaptive_surprise": [],
        "adaptive_change_point": [],
        "response_gate_weight": [],
        "response_gate_belief_confidence": [],
        "response_gate_uncertainty_confidence": [],
        "final_belief_entropy": [],
        "final_mode_accuracy": [],
    }
    mode_correct_by_window = [[] for _ in range(spec.windows)]
    decision_mode_correct_by_window = [[] for _ in range(spec.windows)]
    simulators = [simulator_factory(spec, episode) for episode in range(episodes)]
    previous_executed_plans = [None] * episodes if initial_previous_plans is None else [plan.detach().clone() for plan in initial_previous_plans]
    log_beliefs = [
        uniform_log_belief(model.config.num_modes, device=device) for _ in range(episodes)
    ]
    adaptive_persistent_states = None
    if policy.belief_type == "adaptive-persistent":
        adaptive_persistent_states = [
            AdaptivePersistentBelief(
                model.config.num_modes,
                config=AdaptivePersistentBeliefConfig(
                    base_hazard=adaptive_base_hazard,
                    max_hazard=adaptive_max_hazard,
                    surprise_threshold=adaptive_surprise_threshold,
                    surprise_scale=adaptive_surprise_scale,
                    likelihood_scale=likelihood_scale,
                    hard_reset=adaptive_hard_reset,
                    threshold_hysteresis=adaptive_threshold_hysteresis,
                    cooldown_steps=adaptive_cooldown_steps,
                ),
                device=device,
            )
            for _ in range(episodes)
        ]
    selected_by_episode: list[list[int]] = [[] for _ in range(episodes)]
    emission_noise_states = [None for _ in range(episodes)]
    recurrent_currents: list[list[torch.Tensor]] = [[] for _ in range(episodes)]
    recurrent_candidates: list[list[torch.Tensor]] = [[] for _ in range(episodes)]
    recurrent_responses: list[list[torch.Tensor]] = [[] for _ in range(episodes)]
    recurrent_histories: list[list[torch.Tensor]] = [[] for _ in range(episodes)]
    episode_records: list[dict[str, float | int]] = []
    window_records_by_episode: list[list[dict[str, Any]]] = [
        [] for _ in range(episodes)
    ]
    episode_metric_names = episode_metric_names + CONTRACT_METRICS + BELIEF_DIAGNOSTIC_METRICS
    metrics.update({name: [] for name in BELIEF_DIAGNOSTIC_METRICS})
    metrics.update({name: [] for name in CONTRACT_METRICS})
    metrics.update({name: [] for name in EPISODE_OUTCOME_METRICS})
    episode_values = [
        {name: [] for name in episode_metric_names} for _ in range(episodes)
    ]

    runtime_start = time.perf_counter()

    # All episodes are at the same receding-horizon window here, which lets the
    # SAGE adapter evaluate their independent candidate pools as one GPU batch.
    for window in range(spec.windows):
        split_offset = 0 if spec.split == "id" else 7_000_003
        sample_seeds = [
            spec.seed + split_offset + episode * 104729 + window * 1009
            for episode in range(episodes)
        ]
        if candidate_seed_sequences is not None:
            sample_seeds = candidate_seed_sequences[window]
        if reference_trace is not None:
            if len(reference_trace) != spec.windows or len(reference_trace[window]) != episodes:
                raise ValueError("reference trace must cover every episode/window")
            simulators = [copy.deepcopy(record.simulator) for record in reference_trace[window]]
            pools = [record.pool for record in reference_trace[window]]
        elif candidate_adapter is None:
            pools = [simulator.candidate_pool() for simulator in simulators]
        else:
            scenes = [
                (
                    simulator.history,
                    simulator.robot_positions,
                    simulator.robot_velocities,
                    spec.dt,
                    task_goals(simulator),
                )
                for simulator in simulators
            ]
            pools = candidate_adapter.candidate_pools(scenes, sample_seeds)
        retained_indices = [-1] * episodes
        candidate_context_hashes = [plan_context_sha256(plan if retain_previous_plan else None, spec.execute_steps) for plan in previous_executed_plans]
        if retain_previous_plan:
            for episode, pool in enumerate(pools):
                previous_plan = previous_executed_plans[episode]
                if previous_plan is None:
                    continue
                if reference_trace is None:
                    speed_scale = candidate_adapter.config.goal_speed if candidate_adapter is not None else planning_cost.task_speed_scale
                    pools[episode], retained_indices[episode] = retain_plan_candidate(pool, previous_plan, spec.execute_steps, spec.dt, speed_scale)
                else:
                    candidate = shifted_plan_candidate(pool, previous_plan, spec.execute_steps, spec.dt)
                    retained_indices[episode] = retained_candidate_index(pool, candidate)
                    if retained_indices[episode] < 0:
                        raise ValueError("shared factual trace does not contain the retained plan")
        if captured_reference_trace is not None:
            captured_reference_trace.append([])
        contract_errors = [
            validate_candidate_state(simulator, pool)
            for simulator, pool in zip(simulators, pools)
        ]

        candidate_count = int(pools[0].robot_candidates.shape[0])
        if any(int(pool.robot_candidates.shape[0]) != candidate_count for pool in pools):
            raise ValueError("all episode candidate pools must have the same size")
        histories = torch.cat(
            [
                torch.from_numpy(simulator.history)[None, None].expand(
                    candidate_count, 1, -1, -1
                )
                for simulator in simulators
            ],
            dim=0,
        ).to(device)
        currents = torch.cat(
            [pool.robot_current.to(device) for pool in pools], dim=0
        )
        all_candidates = torch.cat(
            [pool.robot_candidates.to(device) for pool in pools], dim=0
        )
        robot_mask = torch.ones(
            all_candidates.shape[0],
            1,
            all_candidates.shape[1],
            device=device,
            dtype=torch.bool,
        )
        batch_output, batch_reference_output = _candidate_and_reference_outputs(
            model,
            histories,
            currents,
            all_candidates,
            robot_mask,
            policy.include_pair_interactions,
        )
        batch_emission_output = batch_output
        if emission_model is not None:
            batch_emission_output = emission_model(
                histories, currents[:, None], all_candidates[:, None],
                robot_mask=robot_mask, include_pair_interactions=True,
            )
        batch_additive_output = batch_additive_reference_output = None
        if policy.response_gate:
            batch_additive_output, batch_additive_reference_output = (
                _candidate_and_reference_outputs(
                    model,
                    histories,
                    currents,
                    all_candidates,
                    robot_mask,
                    False,
                )
            )

        for episode, (simulator, pool) in enumerate(zip(simulators, pools)):
            if policy.belief_type == "oracle":
                belief = F.one_hot(
                    torch.tensor(simulator.mode, device=device),
                    num_classes=model.config.num_modes,
                ).float()
                log_belief = belief.clamp_min(1e-9).log()
                log_beliefs[episode] = log_belief
            elif policy.belief_type == "recurrent":
                if recurrent_model is None:
                    raise ValueError("recurrent-joint requires a recurrent checkpoint")
                belief = _recurrent_belief(
                    recurrent_model,
                    recurrent_currents[episode],
                    recurrent_candidates[episode],
                    recurrent_responses[episode],
                    recurrent_histories[episode],
                    device,
                )
                log_belief = belief.clamp_min(1e-9).log()
                log_beliefs[episode] = log_belief
            elif policy.belief_type == "adaptive-persistent":
                assert adaptive_persistent_states is not None
                stored_belief = adaptive_persistent_states[episode].log_belief
                log_belief = (
                    stored_belief
                    if stored_belief is not None
                    else uniform_log_belief(model.config.num_modes, device=device)
                )
                belief = log_belief.exp()
                log_beliefs[episode] = log_belief
            else:
                log_belief = log_beliefs[episode]
                belief = log_belief.exp()
            decision_mode = simulator.mode
            decision_belief = belief.clone()
            decision_log_belief = log_belief.clone()
            decision_correct = float(int(belief.argmax()) == decision_mode)
            decision_mode_correct_by_window[window].append(decision_correct)
            candidate_hash = candidate_pool_sha256(pool)
            state_hash = observable_state_sha256(simulator)
            environment_rng_hash = hashlib.sha256(
                json.dumps(simulator.rng.bit_generator.state, sort_keys=True).encode()
            ).hexdigest()
            batch_slice = slice(
                episode * candidate_count,
                (episode + 1) * candidate_count,
            )
            current = currents[batch_slice]
            candidates = all_candidates[batch_slice]
            output = {
                name: value[batch_slice]
                for name, value in batch_output.items()
                if torch.is_tensor(value)
            }
            reference_output = {
                name: value[batch_slice]
                for name, value in batch_reference_output.items()
                if torch.is_tensor(value)
            }
            emission_means = batch_emission_output["mean"][batch_slice, 0]
            emission_log_std = batch_emission_output["log_std"][batch_slice, 0]
            uniform_output = output
            uniform_reference_output = reference_output
            uniform_belief = torch.full_like(belief, 1.0 / belief.numel())
            gate = torch.zeros(candidate_count, device=device, dtype=candidates.dtype)
            gate_belief_confidence = torch.zeros_like(gate)
            gate_uncertainty_confidence = torch.zeros(candidate_count, device=device, dtype=candidates.dtype)
            if policy.response_gate:
                assert batch_additive_output is not None
                assert batch_additive_reference_output is not None
                additive_output = {
                    name: value[batch_slice]
                    for name, value in batch_additive_output.items()
                    if torch.is_tensor(value)
                }
                additive_reference_output = {
                    name: value[batch_slice]
                    for name, value in batch_additive_reference_output.items()
                    if torch.is_tensor(value)
                }
                uniform_output, _, _, _ = _blend_gated_response_outputs(
                    additive_output, output, uniform_belief,
                    response_gate_belief_threshold, response_gate_belief_scale,
                    response_gate_uncertainty_threshold, response_gate_uncertainty_scale,
                    response_gate_min, response_gate_max,
                )
                uniform_reference_output, _, _, _ = _blend_gated_response_outputs(
                    additive_reference_output, reference_output, uniform_belief,
                    response_gate_belief_threshold, response_gate_belief_scale,
                    response_gate_uncertainty_threshold, response_gate_uncertainty_scale,
                    response_gate_min, response_gate_max,
                )
                output, gate, gate_belief_confidence, gate_uncertainty_confidence = _blend_gated_response_outputs(
                    additive_output,
                    output,
                    belief,
                    response_gate_belief_threshold,
                    response_gate_belief_scale,
                    response_gate_uncertainty_threshold,
                    response_gate_uncertainty_scale,
                    response_gate_min,
                    response_gate_max,
                )
                reference_output, reference_gate, _, reference_uncertainty_confidence = _blend_gated_response_outputs(
                    additive_reference_output,
                    reference_output,
                    belief,
                    response_gate_belief_threshold,
                    response_gate_belief_scale,
                    response_gate_uncertainty_threshold,
                    response_gate_uncertainty_scale,
                    response_gate_min,
                    response_gate_max,
                )
                gate_uncertainty_confidence = (
                    0.5 * (gate_uncertainty_confidence + reference_uncertainty_confidence)
                )
            mode_means = output["mean"][:, 0]
            mode_log_std = output["log_std"][:, 0]
            reference_mode_means = reference_output["mean"][:, 0]
            response_features = mode_means.mean(dim=1).reshape(candidate_count, -1)
            candidate_response_spread = float(
                torch.pdist(response_features).mean().item()
                if candidate_count > 1
                else 0.0
            )
            belief_entropy = float(
                -(belief * belief.clamp_min(1e-9).log()).sum().item()
            )
            belief_max_probability = float(belief.max().item())
            task_direction = torch.from_numpy(
                task_goals(simulator) - simulator.robot_positions
            ).to(device=device, dtype=candidates.dtype)
            goal_offsets = task_direction if spec.task_type == "fixed_goal" else None
            costs = expected_candidate_costs(
                mode_means,
                belief,
                current,
                candidates,
                reference_mode_means,
                spec.dt,
                planning_cost,
                mode_log_stds=mode_log_std,
                task_direction=task_direction,
                task_goal_offsets=goal_offsets,
            )
            mixture_only_costs = expected_candidate_costs(
                mode_means,
                uniform_belief,
                current,
                candidates,
                reference_mode_means,
                spec.dt,
                planning_cost,
                mode_log_stds=mode_log_std,
                task_direction=task_direction,
                task_goal_offsets=goal_offsets,
            )
            uniform_costs = mixture_only_costs
            if policy.response_gate:
                uniform_costs = expected_candidate_costs(
                    uniform_output["mean"][:, 0], uniform_belief, current,
                    candidates, uniform_reference_output["mean"][:, 0], spec.dt,
                    planning_cost, mode_log_stds=uniform_output["log_std"][:, 0],
                    task_direction=task_direction, task_goal_offsets=goal_offsets,
                )
            prefix_costs = expected_candidate_costs(
                mode_means, belief, current, candidates, reference_mode_means,
                spec.dt, prefix_cost_config, mode_log_stds=mode_log_std,
                task_direction=task_direction, task_goal_offsets=goal_offsets,
            )
            if prefix_objective:
                mixture_prefix_costs = expected_candidate_costs(
                    mode_means, uniform_belief, current, candidates, reference_mode_means,
                    spec.dt, prefix_cost_config, mode_log_stds=mode_log_std,
                    task_direction=task_direction, task_goal_offsets=goal_offsets,
                )
                uniform_prefix_costs = expected_candidate_costs(
                    uniform_output["mean"][:, 0], uniform_belief, current, candidates,
                    uniform_reference_output["mean"][:, 0], spec.dt, prefix_cost_config,
                    mode_log_stds=uniform_output["log_std"][:, 0],
                    task_direction=task_direction, task_goal_offsets=goal_offsets,
                )
                costs = _prefix_objective_with_full_safety(prefix_costs, costs, planning_cost.risk_weight)
                mixture_only_costs = _prefix_objective_with_full_safety(mixture_prefix_costs, mixture_only_costs, planning_cost.risk_weight)
                uniform_costs = _prefix_objective_with_full_safety(uniform_prefix_costs, uniform_costs, planning_cost.risk_weight)
            if continuation_value_model is not None:
                observable_history = torch.as_tensor(simulator.history, device=device, dtype=candidates.dtype)
                remaining_windows = spec.windows - window - 1
                values = continuation_value_model.candidate_values(observable_history, current, candidates, mode_means, belief, remaining_windows)
                mixture_values = continuation_value_model.candidate_values(observable_history, current, candidates, mode_means, uniform_belief, remaining_windows)
                uniform_values = continuation_value_model.candidate_values(observable_history, current, candidates, uniform_output["mean"][:, 0], uniform_belief, remaining_windows)
                costs = add_continuation_value(costs, values, continuation_value_weight)
                mixture_only_costs = add_continuation_value(mixture_only_costs, mixture_values, continuation_value_weight)
                uniform_costs = add_continuation_value(uniform_costs, uniform_values, continuation_value_weight)
            safe_mask = costs.get("safe_mask")
            selected, fallback = _select_response_costs(costs, policy.selection_objective)
            uniform_selected, _ = _select_response_costs(uniform_costs, policy.selection_objective)
            mixture_selected, _ = _select_response_costs(mixture_only_costs, policy.selection_objective)
            belief_planner = _synthetic_score_decomposition(
                costs, planning_cost, policy.selection_objective, selected
            )
            uniform_planner = _synthetic_score_decomposition(
                uniform_costs, planning_cost, policy.selection_objective, uniform_selected
            )
            planner_sensitivity = belief_sensitivity_diagnostics(
                belief_planner, uniform_planner, selected, uniform_selected
            )
            if bool(fallback) != bool(belief_planner["fallback"]):
                raise AssertionError("synthetic score decomposition disagrees with selection fallback")
            counterfactuals = simulator.counterfactuals(pool, planning_cost)
            true_costs = np.asarray(
                [
                    _realized_total_cost(outcome, spec, planning_cost)
                    for outcome in counterfactuals
                ],
                dtype=np.float64,
            )
            predicted_costs = prefix_costs["unconstrained_total"].detach().cpu().numpy().astype(np.float64)
            oracle_selected = int(true_costs.argmin())
            oracle_cost = float(true_costs[oracle_selected])
            selected_true_cost = float(true_costs[selected])
            uniform_true_cost = float(true_costs[uniform_selected])
            belief_action_flip = float(not torch.allclose(
                candidates[selected, :, :spec.execute_steps, 2:],
                candidates[uniform_selected, :, :spec.execute_steps, 2:],
                rtol=0.0, atol=1e-6,
            ))
            belief_diagnostics = {
                "decision_mode_accuracy": decision_correct,
                "decision_mode_nll": float(-decision_log_belief[decision_mode]),
                "belief_action_flip": belief_action_flip,
                "belief_mixture_selection_flip": float(selected != mixture_selected),
                "belief_gate_selection_flip": float(mixture_selected != uniform_selected),
                "belief_gate_cost_delta": uniform_true_cost - float(true_costs[mixture_selected]),
                **planner_sensitivity,
            }
            oracle_regret = max(selected_true_cost - oracle_cost, 0.0)
            selected_predicted_cost = float(predicted_costs[selected])
            uniform_predicted_cost = float(predicted_costs[uniform_selected])
            belief_selection_flip = float(selected != uniform_selected)
            belief_selection_predicted_cost_delta = (
                uniform_predicted_cost - selected_predicted_cost
                if math.isfinite(uniform_predicted_cost)
                and math.isfinite(selected_predicted_cost)
                else float("nan")
            )
            belief_selection_cost_delta = uniform_true_cost - selected_true_cost
            belief_selection_oracle_regret_delta = max(
                uniform_true_cost - oracle_cost, 0.0
            ) - oracle_regret
            belief_flip_predicted_cost_delta = (
                belief_selection_predicted_cost_delta
                if belief_selection_flip
                else float("nan")
            )
            belief_flip_true_cost_delta = (
                belief_selection_cost_delta
                if belief_selection_flip
                else float("nan")
            )
            belief_flip_oracle_regret_delta = (
                belief_selection_oracle_regret_delta
                if belief_selection_flip
                else float("nan")
            )
            task_feasible_count = sum(
                outcome.task_progress > 0.0 for outcome in counterfactuals
            )
            rank_spearman = _spearman(predicted_costs, true_costs)
            rank_kendall = _kendall_tau_b(predicted_costs, true_costs)
            selected_by_episode[episode].append(selected)
            history_before = simulator.history.copy()
            executed = (
                selected if reference_trace is None
                else reference_trace[window][episode].executed_candidate
            )
            if captured_reference_trace is not None:
                captured_reference_trace[window].append(
                    ReferenceWindow(copy.deepcopy(simulator), pool, executed)
                )
            selected_current = current[executed].detach().cpu()
            selected_candidate = candidates[executed].detach().cpu()
            previous_executed_plans[episode] = selected_candidate.clone()
            expected_position = (
                simulator.entity_position[None]
                + selected_candidate[:, spec.execute_steps - 1, :2].numpy()
            )
            executed_outcome = simulator.step(pool, executed, planning_cost)
            torch.testing.assert_close(
                executed_outcome.response,
                counterfactuals[executed].response,
                rtol=0.0,
                atol=0.0,
            )
            execution_error = float(np.max(np.abs(simulator.robot_positions - expected_position)))
            if not np.allclose(simulator.robot_positions, expected_position, rtol=1e-5, atol=2e-5):
                raise ValueError("executed robot positions disagree with the scored candidate prefix")
            realized = counterfactuals[selected]
            goal_distances = np.linalg.norm(task_goals(simulator) - simulator.robot_positions, axis=-1)
            contract_metrics = {
                **contract_errors[episode],
                "execution_position_error": execution_error,
                "executed_total_cost": _realized_total_cost(executed_outcome, spec, planning_cost),
                "executed_task_progress": executed_outcome.task_progress,
                "executed_collision": executed_outcome.collision,
                "goal_distance": float(goal_distances.mean()) if spec.task_type == "fixed_goal" else float("nan"),
                "goal_reached": float(np.all(goal_distances <= spec.goal_tolerance)) if spec.task_type == "fixed_goal" else float("nan"),
            }
            for name, value in contract_metrics.items():
                episode_values[episode][name].append(value)
            realized_total = _realized_total_cost(realized, spec, planning_cost)
            episode_values[episode]["realized_total_cost"].append(realized_total)
            episode_values[episode]["task_progress"].append(realized.task_progress)
            episode_values[episode]["safety_cost"].append(realized.safety_cost)
            episode_values[episode]["human_cost"].append(realized.human_cost)
            episode_values[episode]["min_distance"].append(realized.min_distance)
            episode_values[episode]["collision_rate"].append(realized.collision)
            episode_values[episode]["mean_activity"].append(realized.mean_activity)
            episode_values[episode]["oracle_regret"].append(oracle_regret)
            episode_values[episode]["oracle_hit_rate"].append(float(selected == oracle_selected))
            episode_values[episode]["candidate_rank_spearman"].append(rank_spearman)
            episode_values[episode]["candidate_rank_kendall"].append(rank_kendall)
            episode_values[episode]["task_feasible_candidate_count"].append(
                float(task_feasible_count)
            )
            episode_values[episode]["task_feasible_candidate_fraction"].append(
                task_feasible_count / len(counterfactuals)
            )
            episode_values[episode]["candidate_true_cost_range"].append(
                float(true_costs.max() - true_costs.min())
            )
            episode_values[episode]["candidate_response_spread"].append(
                candidate_response_spread
            )
            episode_values[episode]["belief_entropy"].append(belief_entropy)
            episode_values[episode]["belief_max_probability"].append(
                belief_max_probability
            )
            episode_values[episode]["response_gate_weight"].append(
                float(gate.mean().item())
            )
            episode_values[episode]["response_gate_belief_confidence"].append(
                float(gate_belief_confidence.mean().item())
            )
            episode_values[episode]["response_gate_uncertainty_confidence"].append(
                float(gate_uncertainty_confidence.mean().item())
            )
            episode_values[episode]["belief_selection_flip"].append(
                belief_selection_flip
            )
            episode_values[episode]["belief_selection_predicted_cost_delta"].append(
                belief_selection_predicted_cost_delta
            )
            episode_values[episode]["belief_selection_cost_delta"].append(
                belief_selection_cost_delta
            )
            episode_values[episode]["belief_selection_oracle_regret_delta"].append(
                belief_selection_oracle_regret_delta
            )
            episode_values[episode]["belief_flip_predicted_cost_delta"].append(
                belief_flip_predicted_cost_delta
            )
            episode_values[episode]["belief_flip_true_cost_delta"].append(
                belief_flip_true_cost_delta
            )
            episode_values[episode]["belief_flip_oracle_regret_delta"].append(
                belief_flip_oracle_regret_delta
            )
            episode_values[episode].setdefault("safety_filter_rejections", []).append(
                float((~safe_mask).sum()) if safe_mask is not None else 0.0
            )
            episode_values[episode].setdefault("safety_fallback", []).append(
                float(fallback)
            )
            window_records_by_episode[episode].append(
                {
                    "window": window,
                    "evaluation_mode": evaluation_mode,
                    "prefix_objective_with_full_safety": bool(prefix_objective),
                    "persistent_forgetting_rate": persistent_forgetting_rate if policy.belief_type == "persistent" else 0.0,
                    "retained_plan_enabled": bool(retain_previous_plan),
                    "continuation_value_enabled": continuation_value_model is not None,
                    "continuation_value_weight": continuation_value_weight if continuation_value_model is not None else 0.,
                    "continuation_value_adjustment": costs.get("continuation_adjustment", torch.zeros_like(costs["unconstrained_total"])).detach().cpu().tolist(),
                    "candidate_context_hash": candidate_context_hashes[episode],
                    "retained_plan_candidate": retained_indices[episode],
                    "retained_plan_selected": float(selected == retained_indices[episode]),
                    "separate_frozen_emission_residual": emission_model is not None,
                    "task_type": spec.task_type,
                    "state_hash": state_hash,
                    "environment_rng_hash": environment_rng_hash,
                    "executed_candidate": executed,
                    "executed_response_hash": hashlib.sha256(
                        executed_outcome.response[:spec.execute_steps].detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                    "planning_response_steps": spec.execute_steps if prefix_objective else (planning_cost.response_steps or spec.horizon),
                    "diagnostic_response_steps": spec.execute_steps,
                    "planning_candidate_utility": (-costs["unconstrained_total"]).detach().cpu().tolist(),
                    "planning_admissible_mask": costs["safe_mask"].detach().cpu().tolist(),
                    "belief_selection_planning_cost_delta": float(
                        costs["unconstrained_total"][uniform_selected] - costs["unconstrained_total"][selected]
                    ),
                    **contract_metrics,
                    "selected_candidate": selected,
                    "uniform_selected_candidate": uniform_selected,
                    "mixture_only_uniform_selected_candidate": mixture_selected,
                    "decision_mode_probabilities": decision_belief.cpu().tolist(),
                    "belief_evidence_source": "ungated_mode_conditional_response",
                    "belief_evidence_steps": spec.execute_steps,
                    **belief_diagnostics,
                    "belief_entropy": belief_entropy,
                    "belief_max_probability": belief_max_probability,
                    "response_gate_weight": float(gate.mean().item()),
                    "response_gate_belief_confidence": float(gate_belief_confidence.mean().item()),
                    "response_gate_uncertainty_confidence": float(gate_uncertainty_confidence.mean().item()),
                    "belief_selection_flip": belief_selection_flip,
                    "belief_selection_predicted_cost_delta": belief_selection_predicted_cost_delta,
                    "belief_selection_cost_delta": belief_selection_cost_delta,
                    "belief_selection_oracle_regret_delta": belief_selection_oracle_regret_delta,
                    "belief_flip_predicted_cost_delta": belief_flip_predicted_cost_delta,
                    "belief_flip_true_cost_delta": belief_flip_true_cost_delta,
                    "belief_flip_oracle_regret_delta": belief_flip_oracle_regret_delta,
                    "oracle_candidate": oracle_selected,
                    "selected_true_cost": selected_true_cost,
                    "oracle_true_cost": oracle_cost,
                    "candidate_response_spread": candidate_response_spread,
                    "oracle_regret": oracle_regret,
                    "oracle_hit": float(selected == oracle_selected),
                    "candidate_rank_spearman": rank_spearman,
                    "candidate_rank_kendall": rank_kendall,
                    "task_feasible_candidate_count": task_feasible_count,
                    "safety_filter_rejections": int((~safe_mask).sum()) if safe_mask is not None else 0,
                    "safety_fallback": int(fallback),
                    "selected_violation_probability": float(
                        costs["violation_probability"][selected]
                    ),
                    "candidate_hash": candidate_hash,
                    "candidate_seed": sample_seeds[episode],
                    "candidate_generation_key": (
                        f"{spec.seed}:{spec.split}:{episode}:{window}"
                    ),
                    "predicted_candidate_utility": (-predicted_costs).tolist(),
                    "ground_truth_candidate_utility": (-true_costs).tolist(),
                }
            )

            steps = spec.execute_steps
            observation = executed_outcome.response[:steps].to(device)
            log_likelihood = trajectory_mode_log_likelihood(
                emission_means[executed, :, :steps][None, None],
                emission_log_std[executed, :, :steps][None, None],
                observation[None, None],
            )[0, 0]
            diagonal_log_likelihood = log_likelihood
            if emission_noise_model is not None:
                log_likelihood, emission_noise_states[episode] = emission_noise_model.step(
                    emission_means[executed, :, :steps], emission_log_std[executed, :, :steps],
                    observation, emission_noise_states[episode],
                )
            window_records_by_episode[episode][-1].update({
                "correlated_emission_enabled": emission_noise_model is not None,
                "diagonal_executed_mode_log_likelihood": diagonal_log_likelihood.detach().cpu().tolist(),
                "diagonal_decision_prefix_nll": float(-torch.logsumexp(decision_log_belief + diagonal_log_likelihood, dim=-1) / (2 * steps)),
            })
            belief_diagnostics["decision_prefix_nll"] = float(
                -torch.logsumexp(decision_log_belief + log_likelihood, dim=-1) / (2 * steps)
            )
            planning_log_likelihood = trajectory_mode_log_likelihood(
                mode_means[executed, :, :steps][None, None],
                mode_log_std[executed, :, :steps][None, None],
                observation[None, None],
            )[0, 0]
            planning_velocity = (
                decision_belief[:, None, None] * mode_means[executed, :, :steps]
            ).sum(dim=0)
            prefix_displacement_error = (planning_velocity - observation).cumsum(dim=0) * spec.dt
            belief_diagnostics["planning_prefix_nll"] = float(
                -torch.logsumexp(decision_log_belief + planning_log_likelihood, dim=-1) / (2 * steps)
            )
            belief_diagnostics["planning_prefix_ade"] = float(
                torch.linalg.vector_norm(prefix_displacement_error, dim=-1).mean()
            )
            window_records_by_episode[episode][-1]["executed_mode_log_likelihood"] = (
                log_likelihood.detach().cpu().tolist()
            )
            if policy.belief_type in {
                "persistent",
                "reset",
                "adaptive",
                "adaptive-persistent",
            }:
                if policy.belief_type == "adaptive":
                    log_belief, adaptive_hazard, adaptive_surprise, adaptive_changed = (
                        adaptive_categorical_update(
                            log_belief,
                            log_likelihood,
                            base_hazard=adaptive_base_hazard,
                            max_hazard=adaptive_max_hazard,
                            surprise_threshold=adaptive_surprise_threshold,
                            surprise_scale=adaptive_surprise_scale,
                            likelihood_scale=likelihood_scale,
                        )
                    )
                    episode_values[episode].setdefault("adaptive_hazard", []).append(
                        float(adaptive_hazard)
                    )
                    episode_values[episode].setdefault("adaptive_surprise", []).append(
                        float(adaptive_surprise)
                    )
                    episode_values[episode].setdefault("adaptive_change_point", []).append(
                        float(adaptive_changed)
                    )
                elif policy.belief_type == "adaptive-persistent":
                    assert adaptive_persistent_states is not None
                    log_belief, adaptive_diagnostics = adaptive_persistent_states[
                        episode
                    ].update(log_likelihood)
                    episode_values[episode].setdefault("adaptive_hazard", []).append(
                        float(adaptive_diagnostics["hazard"])
                    )
                    episode_values[episode].setdefault("adaptive_surprise", []).append(
                        float(adaptive_diagnostics["surprise"])
                    )
                    episode_values[episode].setdefault("adaptive_change_point", []).append(
                        float(adaptive_diagnostics["changed"])
                    )
                else:
                    update_prior = (
                        log_belief
                        if policy.belief_type == "persistent"
                        else uniform_log_belief(model.config.num_modes, device=device)
                    )
                    if policy.belief_type == "persistent" and persistent_forgetting_rate:
                        update_prior = apply_belief_forgetting(update_prior, persistent_forgetting_rate)
                    log_belief = categorical_bayes_update(
                        update_prior,
                        likelihood_scale * log_likelihood,
                    )
                log_beliefs[episode] = log_belief
            if policy.belief_type == "recurrent":
                recurrent_currents[episode].append(selected_current)
                recurrent_candidates[episode].append(selected_candidate)
                recurrent_responses[episode].append(executed_outcome.response[:spec.execute_steps].detach().cpu())
                recurrent_histories[episode].append(
                    torch.from_numpy(history_before).float()
                )
                posterior_belief = _recurrent_belief(
                    recurrent_model, recurrent_currents[episode],
                    recurrent_candidates[episode], recurrent_responses[episode],
                    recurrent_histories[episode], device,
                )
                log_belief = posterior_belief.clamp_min(1e-9).log()
                log_beliefs[episode] = log_belief
            correct = float(int(log_belief.argmax()) == simulator.mode)
            mode_correct_by_window[window].append(correct)
            belief_diagnostics.update({
                "posterior_mode_accuracy": correct,
                "posterior_mode_nll": float(-log_belief[simulator.mode]),
            })
            for name, value in belief_diagnostics.items():
                episode_values[episode][name].append(value)
            window_records_by_episode[episode][-1].update({
                **belief_diagnostics,
                "posterior_mode_probabilities": log_belief.exp().cpu().tolist(),
            })

    for episode, (simulator, values) in enumerate(zip(simulators, episode_values)):
        for name in ("adaptive_hazard", "adaptive_surprise", "adaptive_change_point"):
            if not values[name]:
                values[name] = [0.0] * spec.windows
        final_belief = log_beliefs[episode].exp()
        entropy = float(-(final_belief * final_belief.clamp_min(1e-9).log()).sum())
        episode_record: dict[str, float | int] = {"episode": episode}
        for name, metric_values in values.items():
            finite_values = [
                float(value)
                for value in metric_values
                if math.isfinite(float(value))
            ]
            episode_value = (
                statistics.fmean(finite_values) if finite_values else float("nan")
            )
            metrics[name].append(episode_value)
            episode_record[name] = episode_value
        metrics["final_belief_entropy"].append(entropy)
        final_accuracy = float(int(final_belief.argmax()) == simulator.mode)
        metrics["final_mode_accuracy"].append(final_accuracy)
        episode_record["final_belief_entropy"] = entropy
        episode_record["final_mode_accuracy"] = final_accuracy
        outcomes = {
            "cumulative_executed_cost": float(sum(values["executed_total_cost"])),
            "episode_collision": float(any(values["executed_collision"])),
            "terminal_goal_distance": values["goal_distance"][-1],
            "terminal_goal_success": values["goal_reached"][-1],
        }
        for name, value in outcomes.items():
            metrics[name].append(value)
            episode_record[name] = value
        episode_records.append(episode_record)

    runtime_ms_per_episode = (
        1000.0 * (time.perf_counter() - runtime_start) / max(episodes, 1)
    )
    runtime_ms_per_window = runtime_ms_per_episode / max(spec.windows, 1)
    metrics["runtime_ms_per_episode"] = [runtime_ms_per_episode] * episodes
    metrics["runtime_ms_per_window"] = [runtime_ms_per_window] * episodes
    for record in episode_records:
        record["runtime_ms_per_episode"] = runtime_ms_per_episode
        record["runtime_ms_per_window"] = runtime_ms_per_window
    result = {name: _mean_std(values) for name, values in metrics.items()}
    result["mode_accuracy_by_window"] = [
        statistics.fmean(values) for values in mode_correct_by_window
    ]
    result["posterior_mode_accuracy_by_window"] = result["mode_accuracy_by_window"]
    result["decision_mode_accuracy_by_window"] = [
        statistics.fmean(values) for values in decision_mode_correct_by_window
    ]
    result["mode_accuracy_by_window_semantics"] = "posterior_after_executed_prefix"
    result["selected_candidates"] = selected_by_episode
    result["window_records"] = window_records_by_episode
    result["episode_records"] = episode_records
    return result


def _aggregate_runs(
    runs: dict[str, dict[str, dict]], policies: tuple[Policy, ...] | None = None
) -> dict[str, dict]:
    policies = policies or POLICIES
    policy_names = tuple(policy.name for policy in policies)
    summary: dict[str, dict] = {}
    for split in next(iter(runs.values())):
        summary[split] = {}
        for policy_name in policy_names:
            summary[split][policy_name] = {}
            metric_names = runs[next(iter(runs))][split][policy_name]
            for metric, value in metric_names.items():
                if isinstance(value, dict) and "mean" in value:
                    seed_means = [runs[seed][split][policy_name][metric]["mean"] for seed in runs]
                    summary[split][policy_name][metric] = _mean_std(seed_means)
        persistent = summary[split]["persistent-joint"]
        reset = summary[split]["reset-joint"]
        reset_additive = summary[split]["reset-additive"]
        additive = summary[split]["persistent-additive"]
        oracle = summary[split]["oracle-joint"]
        task_only = summary[split]["task-only"]
        no_belief = summary[split]["no-belief-joint"]
        summary[split]["contrasts"] = {
            "reset_minus_persistent_total_cost": (
                reset["realized_total_cost"]["mean"]
                - persistent["realized_total_cost"]["mean"]
            ),
            "reset_minus_persistent_human_cost": (
                reset["human_cost"]["mean"] - persistent["human_cost"]["mean"]
            ),
            "additive_minus_joint_total_cost": (
                additive["realized_total_cost"]["mean"]
                - persistent["realized_total_cost"]["mean"]
            ),
            "reset_additive_minus_reset_joint_total_cost": (
                reset_additive["realized_total_cost"]["mean"]
                - reset["realized_total_cost"]["mean"]
            ),
            "interaction_of_belief_and_response_realized_total_cost": (
                reset_additive["realized_total_cost"]["mean"]
                - reset["realized_total_cost"]["mean"]
                - additive["realized_total_cost"]["mean"]
                + persistent["realized_total_cost"]["mean"]
            ),
            "persistent_minus_oracle_total_cost": (
                persistent["realized_total_cost"]["mean"]
                - oracle["realized_total_cost"]["mean"]
            ),
            "reset_minus_persistent_oracle_regret": (
                reset["oracle_regret"]["mean"]
                - persistent["oracle_regret"]["mean"]
            ),
            "no_belief_minus_persistent_oracle_regret": (
                no_belief["oracle_regret"]["mean"]
                - persistent["oracle_regret"]["mean"]
            ),
            "task_only_minus_persistent_oracle_regret": (
                task_only["oracle_regret"]["mean"]
                - persistent["oracle_regret"]["mean"]
            ),
            "task_only_minus_persistent_total_cost": (
                task_only["realized_total_cost"]["mean"]
                - persistent["realized_total_cost"]["mean"]
            ),
            "persistent_selection_flip_rate": persistent.get(
                "belief_selection_flip", {"mean": 0.0}
            )["mean"],
            "persistent_selection_cost_delta": persistent.get(
                "belief_selection_cost_delta", {"mean": 0.0}
            )["mean"],
            "persistent_selection_oracle_regret_delta": persistent.get(
                "belief_selection_oracle_regret_delta", {"mean": 0.0}
            )["mean"],
        }
        recurrent = summary[split].get("recurrent-joint")
        if recurrent is not None:
            summary[split]["contrasts"].update(
                {
                    "recurrent_minus_persistent_total_cost": (
                        recurrent["realized_total_cost"]["mean"]
                        - persistent["realized_total_cost"]["mean"]
                    ),
                    "recurrent_minus_persistent_task_progress": (
                        recurrent["task_progress"]["mean"]
                        - persistent["task_progress"]["mean"]
                    ),
                    "recurrent_minus_persistent_human_cost": (
                        recurrent["human_cost"]["mean"]
                        - persistent["human_cost"]["mean"]
                    ),
                    "recurrent_minus_persistent_oracle_regret": (
                        recurrent["oracle_regret"]["mean"]
                        - persistent["oracle_regret"]["mean"]
                    ),
                    "recurrent_runtime_ms_per_window": recurrent[
                        "runtime_ms_per_window"
                    ]["mean"],
                }
            )
    return summary


def _write_csv(runs: dict[str, dict], output_path: Path) -> None:
    fields = [
        "seed",
        "split",
        "policy",
        "realized_total_cost",
        "task_progress",
        "safety_cost",
        "human_cost",
        "min_distance",
        "collision_rate",
        "mean_activity",
        "oracle_regret",
        "oracle_hit_rate",
        "candidate_rank_spearman",
        "candidate_rank_kendall",
        "task_feasible_candidate_count",
        "task_feasible_candidate_fraction",
        "candidate_true_cost_range",
        "candidate_response_spread",
        "belief_entropy",
        "belief_max_probability",
        "response_gate_weight",
        "response_gate_belief_confidence",
        "response_gate_uncertainty_confidence",
        "belief_selection_flip",
        "belief_selection_predicted_cost_delta",
        "belief_selection_cost_delta",
        "belief_selection_oracle_regret_delta",
        "belief_flip_predicted_cost_delta",
        "belief_flip_true_cost_delta",
        "belief_flip_oracle_regret_delta",
        "safety_filter_rejections",
        "safety_fallback",
        "adaptive_hazard",
        "adaptive_surprise",
        "adaptive_change_point",
        "final_mode_accuracy",
        "runtime_ms_per_episode",
        "runtime_ms_per_window",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fields.extend(CONTRACT_METRICS + EPISODE_OUTCOME_METRICS + BELIEF_DIAGNOSTIC_METRICS)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed, splits in sorted(runs.items()):
            for split, policies in splits.items():
                for policy, metrics in policies.items():
                    writer.writerow(
                        {
                            "seed": seed,
                            "split": split,
                            "policy": policy,
                            **{field: metrics[field]["mean"] for field in fields[3:]},
                        }
                    )


def _write_episode_csv(runs: dict[str, dict], output_path: Path) -> None:
    fields = [
        "seed",
        "split",
        "policy",
        "episode",
        "realized_total_cost",
        "task_progress",
        "safety_cost",
        "human_cost",
        "min_distance",
        "collision_rate",
        "mean_activity",
        "oracle_regret",
        "oracle_hit_rate",
        "candidate_rank_spearman",
        "candidate_rank_kendall",
        "task_feasible_candidate_count",
        "task_feasible_candidate_fraction",
        "candidate_true_cost_range",
        "candidate_response_spread",
        "belief_entropy",
        "belief_max_probability",
        "response_gate_weight",
        "response_gate_belief_confidence",
        "response_gate_uncertainty_confidence",
        "belief_selection_flip",
        "belief_selection_predicted_cost_delta",
        "belief_selection_cost_delta",
        "belief_selection_oracle_regret_delta",
        "belief_flip_predicted_cost_delta",
        "belief_flip_true_cost_delta",
        "belief_flip_oracle_regret_delta",
        "safety_filter_rejections",
        "safety_fallback",
        "adaptive_hazard",
        "adaptive_surprise",
        "adaptive_change_point",
        "final_belief_entropy",
        "final_mode_accuracy",
        "runtime_ms_per_episode",
        "runtime_ms_per_window",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fields.extend(CONTRACT_METRICS + EPISODE_OUTCOME_METRICS + BELIEF_DIAGNOSTIC_METRICS)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed, splits in sorted(runs.items()):
            for split, policies in splits.items():
                for policy, result in policies.items():
                    for record in result["episode_records"]:
                        writer.writerow(
                            {
                                "seed": seed,
                                "split": split,
                                "policy": policy,
                                **record,
                            }
                        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the minimal belief-aware loop.")
    parser.add_argument("--config", default="configs/persistent_closed_loop_state_consistent.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume complete seed/split blocks from an existing runs.json",
    )
    parser.add_argument(
        "--likelihood-scale",
        type=float,
        default=None,
        help="validation-only override for the belief update temperature",
    )
    parser.add_argument(
        "--response-gate-max",
        type=float,
        default=None,
        help="validation-only override for the Joint response gate maximum mix",
    )
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    evaluation = raw.get("evaluation", {})
    evaluation_mode = evaluation.get("mode", "closed_loop")
    if evaluation_mode not in {"closed_loop", "same_state"}:
        raise ValueError("evaluation.mode must be closed_loop or same_state")
    phase = evaluation.get("phase", "diagnostic")
    if phase not in {"diagnostic", "validation", "test"}:
        raise ValueError("evaluation.phase must be diagnostic, validation, or test")
    if args.likelihood_scale is not None and phase != "validation":
        raise ValueError("--likelihood-scale is only allowed in evaluation.phase=validation")
    if args.response_gate_max is not None and phase != "validation":
        raise ValueError("--response-gate-max is only allowed in evaluation.phase=validation")
    simulation_cfg = raw["simulation"]
    planning_cfg = raw["planning"]
    belief_cfg = raw["belief"]
    calibration_pattern = belief_cfg.get("calibration_path")
    if calibration_pattern and args.likelihood_scale is not None:
        raise ValueError("a frozen belief calibration cannot be combined with a scale override")
    response_gate_cfg = raw.get("response_gate", {})
    if args.response_gate_max is not None:
        if not math.isfinite(args.response_gate_max) or not 0.0 <= args.response_gate_max <= 1.0:
            raise ValueError("response gate maximum must be finite and in [0, 1]")
        response_gate_cfg = dict(response_gate_cfg)
        response_gate_cfg["max_gate"] = float(args.response_gate_max)
    likelihood_scale = float(
        belief_cfg["likelihood_scale"]
        if args.likelihood_scale is None
        else args.likelihood_scale
    )
    if likelihood_scale <= 0.0 or not math.isfinite(likelihood_scale):
        raise ValueError("likelihood scale must be positive and finite")
    candidate_cfg = raw.get("candidate_source", {"type": "synthetic"})
    if candidate_cfg.get("shared_candidate_pools", False):
        raise ValueError("legacy shared_candidate_pools is state-inconsistent; migrate to evaluation.mode")
    suite_cfg = raw["suite"]
    torch.set_num_threads(int(suite_cfg.get("num_threads_per_seed", 4)))
    policies = _configured_policies(raw)
    seeds = args.seeds or [int(seed) for seed in suite_cfg["seeds"]]
    episodes = int(simulation_cfg["episodes"] if args.episodes is None else args.episodes)
    if episodes < 1:
        raise ValueError("episodes must be positive")
    output_dir = Path(args.output_dir or suite_cfg["output_dir"])
    checkpoints = [Path(suite_cfg["checkpoint_dir"]) / f"seed{seed}" / "response.pt" for seed in seeds]
    if any(policy.belief_type == "recurrent" for policy in policies):
        checkpoints += [Path(suite_cfg["checkpoint_dir"]) / f"seed{seed}" / "recurrent.pt" for seed in seeds]
    if candidate_cfg.get("type") == "sage":
        checkpoints.append(Path(candidate_cfg["checkpoint"]))
    belief_calibrations = {}
    if calibration_pattern:
        from sage.experiments.calibrate_persistent_belief import load_belief_calibration

        for seed in seeds:
            calibration_path = Path(str(calibration_pattern).format(seed=seed))
            checkpoint_path = Path(suite_cfg["checkpoint_dir"]) / f"seed{seed}" / "response.pt"
            belief_calibrations[str(seed)] = load_belief_calibration(
                calibration_path, checkpoint_path,
                execute_steps=int(simulation_cfg["execute_steps"]),
                dt=float(simulation_cfg["dt"]), num_robots=int(simulation_cfg["num_robots"]),
            )
            checkpoints.append(calibration_path)
    source_paths = [Path(__file__).resolve()]
    for directory in ("sage/response", "sage/belief", "sage/models", "sage/guidance"):
        source_paths.extend(sorted(Path(directory).rglob("*.py")))
    source_paths.extend([Path("sage/experiments/closed_loop_protocol.py"), Path("sage/experiments/formal_manifest.py")])
    manifest = prepare_manifest(
        output_dir,
        {"raw": raw, "seeds": seeds, "episodes": episodes, "likelihood_scale": likelihood_scale,
         "belief_calibrations": belief_calibrations,
         "response_gate_max": args.response_gate_max,
         "evaluation_mode": evaluation_mode, "phase": phase, "device": args.device, "torch_version": str(torch.__version__)},
        checkpoints, source_paths, args.resume,
    )
    planning_cfg = dict(planning_cfg)
    human_cost_scale = float(planning_cfg.pop("human_cost_scale", 1.0))
    if not math.isfinite(human_cost_scale) or human_cost_scale <= 0.0:
        raise ValueError("planning.human_cost_scale must be finite and positive")
    planning_cfg["human_weight"] = float(planning_cfg.get("human_weight", 0.6)) * human_cost_scale
    planning_cost = PlanningCostConfig(**planning_cfg)
    candidate_type = str(candidate_cfg.get("type", "synthetic")).lower()
    if candidate_type == "synthetic":
        candidate_adapter = None
    elif candidate_type == "sage":
        adapter_config = SAGECandidateAdapterConfig(
            num_candidates=int(simulation_cfg["num_candidates"]),
            guided=bool(candidate_cfg.get("guided", True)),
            step_stride=int(candidate_cfg.get("step_stride", 10)),
            goal_speed=float(candidate_cfg.get("goal_speed", 1.2)),
            oversample_factor=int(candidate_cfg.get("oversample_factor", 1)),
            task_prefilter=bool(candidate_cfg.get("task_prefilter", False)),
            task_prefix_steps=int(
                candidate_cfg.get(
                    "task_prefix_steps", simulation_cfg.get("execute_steps", 3)
                )
            ),
            goal_direction=str(candidate_cfg.get("goal_direction", "velocity")),
            include_task_anchor=bool(candidate_cfg.get("include_task_anchor", False)),
            task_anchor_speed_scales=tuple(
                float(scale)
                for scale in candidate_cfg.get("task_anchor_speed_scales", [1.0])
            ),
            cache_candidate_pools=bool(
                candidate_cfg.get("cache_candidate_pools", False)
            ),
            batch_size=int(candidate_cfg.get("batch_size", 4)),
            guidance_step_clip_norm=(
                None
                if candidate_cfg.get("guidance_step_clip_norm") is None
                else float(candidate_cfg["guidance_step_clip_norm"])
            ),
            diversity_weight=float(candidate_cfg.get("diversity_weight", 0.0)),
            goal_distance_prefilter=bool(candidate_cfg.get("goal_distance_prefilter", False)),
            stop_task_anchors_at_goal=bool(candidate_cfg.get("stop_task_anchors_at_goal", False)),
        )
        candidate_adapter = SAGECandidateAdapter.from_checkpoint(
            candidate_cfg["checkpoint"], args.device, adapter_config
        )
        if candidate_adapter.horizon != int(simulation_cfg["horizon"]):
            raise ValueError("SAGE candidate horizon must match the closed-loop horizon")
        print(
            f"candidate_source=sage checkpoint={candidate_cfg['checkpoint']} "
            f"scope={candidate_cfg.get('checkpoint_scope', 'unspecified')} "
            f"cache_candidate_pools={adapter_config.cache_candidate_pools}",
            flush=True,
        )
    else:
        raise ValueError("candidate_source.type must be 'synthetic' or 'sage'")
    runs: dict[str, dict] = {}
    resume_path = output_dir / "runs.json"
    if args.resume and resume_path.exists():
        loaded = json.loads(resume_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("resume runs.json must contain an object")
        runs = loaded
        print(f"resume_loaded={resume_path}", flush=True)

    for seed in seeds:
        checkpoint = Path(suite_cfg["checkpoint_dir"]) / f"seed{seed}" / "response.pt"
        model = _load_model(checkpoint, args.device)
        seed_likelihood_scale = float(
            belief_calibrations[str(seed)]["selected"]["likelihood_scale"]
            if str(seed) in belief_calibrations else likelihood_scale
        )
        recurrent_model = None
        if any(policy.belief_type == "recurrent" for policy in policies):
            recurrent_checkpoint = (
                Path(suite_cfg["checkpoint_dir"]) / f"seed{seed}" / "recurrent.pt"
            )
            recurrent_model = _load_recurrent_model(recurrent_checkpoint, args.device)
        data_seed = int(simulation_cfg.get("data_seed", seed))
        runs.setdefault(str(seed), {})
        for split in simulation_cfg["splits"]:
            spec = ClosedLoopSpec(
                **{
                    key: value
                    for key, value in simulation_cfg.items()
                    if key not in {"episodes", "splits", "data_seed"}
                },
                seed=data_seed,
                split=str(split),
            )
            expected_policy_names = {policy.name for policy in policies}
            existing_split = runs[str(seed)].get(str(split), {})
            split_complete = (
                expected_policy_names == set(existing_split)
                and all(
                    len(existing_split[name].get("episode_records", [])) == episodes
                    for name in expected_policy_names
                )
            )
            if split_complete:
                print(
                    f"resume_skip model_seed={seed} data_seed={data_seed} split={split}",
                    flush=True,
                )
                continue
            runs[str(seed)][str(split)] = {}
            reference_trace: list[list[ReferenceWindow]] | None = [] if evaluation_mode == "same_state" else None
            for policy_index, policy in enumerate(policies):
                print(
                    f"closed_loop_run model_seed={seed} data_seed={data_seed} "
                    f"split={split} policy={policy.name}",
                    flush=True,
                )
                runs[str(seed)][str(split)][policy.name] = _evaluate_policy(
                    model,
                    policy,
                    spec,
                    episodes,
                    args.device,
                    planning_cost,
                    seed_likelihood_scale,
                    float(belief_cfg.get("adaptive_base_hazard", 0.0)),
                    float(belief_cfg.get("adaptive_max_hazard", 0.10)),
                    float(belief_cfg.get("adaptive_surprise_threshold", 2.0)),
                    float(belief_cfg.get("adaptive_surprise_scale", 0.10)),
                    bool(belief_cfg.get("adaptive_hard_reset", False)),
                    float(belief_cfg.get("adaptive_threshold_hysteresis", 0.0)),
                    int(belief_cfg.get("adaptive_cooldown_steps", 0)),
                    float(response_gate_cfg.get("belief_threshold", 0.35)),
                    float(response_gate_cfg.get("belief_scale", 0.10)),
                    float(response_gate_cfg.get("uncertainty_threshold", 0.45)),
                    float(response_gate_cfg.get("uncertainty_scale", 0.08)),
                    float(response_gate_cfg.get("min_gate", 0.0)),
                    float(response_gate_cfg.get("max_gate", 1.0)),
                    candidate_adapter,
                    recurrent_model,
                    evaluation_mode=evaluation_mode,
                    reference_trace=reference_trace if policy_index > 0 else None,
                    captured_reference_trace=reference_trace if policy_index == 0 else None,
                    retain_previous_plan=bool(raw.get("candidate_source", {}).get("retain_previous_plan", False)),
                )
                (output_dir / "runs.json").write_text(
                    json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )

    candidate_protocol = audit_candidate_protocol(
        runs, [policy.name for policy in policies], evaluation_mode
    )
    candidate_protocol.update({
        "fingerprint": manifest["fingerprint"],
        "candidate_source": candidate_type,
        "likelihood_scale": likelihood_scale,
        "likelihood_scale_by_seed": {
            str(seed): float(belief_calibrations[str(seed)]["selected"]["likelihood_scale"])
            if str(seed) in belief_calibrations else likelihood_scale for seed in seeds
        },
        "belief_timing": "decision_before_current_prefix_posterior_after_prefix",
        "belief_gate_audit": "uniform_recomputes_mixture_and_gate",
        "task_type": simulation_cfg.get("task_type", "entity_approach"),
        "phase": phase,
        "diagnostic_horizon": simulation_cfg["execute_steps"],
        "selected_cost_semantics": "executed_closed_loop" if evaluation_mode == "closed_loop" else "same_state_counterfactual",
        "candidate_goal_direction": (
            None if candidate_adapter is None else candidate_adapter.config.goal_direction
        ),
        "candidate_diversity_weight": (
            None
            if candidate_adapter is None
            else candidate_adapter.config.diversity_weight
        ),
        "policies": [policy.name for policy in policies],
    })
    (output_dir / "candidate_protocol.json").write_text(
        json.dumps(candidate_protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = _aggregate_runs(runs, policies)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(runs, output_dir / "runs.csv")
    _write_episode_csv(runs, output_dir / "episode_metrics.csv")
    for split in simulation_cfg["splits"]:
        print(
            f"closed_loop_{split}_contrasts="
            f"{json.dumps(summary[str(split)]['contrasts'], sort_keys=True)}",
            flush=True,
        )
    print(f"saved_summary={output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
