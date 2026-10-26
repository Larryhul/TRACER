"""Run the TRACER gap-closure robustness and system-baseline suite.

The suite is deliberately separate from the SAGE portability experiments. It
keeps the trained TRACER response model and the finite-candidate planning
contract, while exercising a resettable multi-entity interaction world under
non-stationarity, observation/communication faults, static obstacles, and
model-free system baselines. External SocialGym2/CrowdNav/robot adapters are
audited in the output, but are not silently replaced by this local backend.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sage.belief.filter import (
    apply_belief_forgetting,
    categorical_bayes_update,
    detect_change_point,
    hazard_belief_update,
    uniform_log_belief,
)
from sage.belief.related_work import (
    UKFBeliefConfig,
    UKFParameterBelief,
    categorical_pomdp_predict,
    categorical_pomdp_update,
)
from sage.experiments.eval_persistent_closed_loop import _load_model
from sage.experiments.run_independent_multientity_suite import (
    _build_spec,
    _multi_entity_predicted_costs,
    _multi_entity_true_costs,
)
from sage.response.closed_loop import CandidatePool, PlanningCostConfig
from sage.response.independent_social_force import (
    MultiEntityIndependentSocialForceSimulator,
)
from sage.response.persistent_models import (
    stationary_reference_candidates,
    trajectory_mode_log_likelihood,
)


@dataclass(frozen=True)
class FaultScenario:
    name: str
    partial_observation: bool = False
    missed_detection_rate: float = 0.0
    id_switch: bool = False
    communication_delay: int = 0
    asynchronous_execution: bool = False
    nonstationary: bool = False
    static_obstacles: bool = False


SCENARIOS = (
    FaultScenario("clean"),
    FaultScenario("nonstationary-hazard", nonstationary=True),
    FaultScenario("partial-observation", partial_observation=True),
    FaultScenario("missed-detection", missed_detection_rate=0.35),
    FaultScenario("id-switch", id_switch=True),
    FaultScenario("communication-delay-2", communication_delay=2),
    FaultScenario("asynchronous-execution", asynchronous_execution=True),
    FaultScenario(
        "combined-realism",
        partial_observation=True,
        missed_detection_rate=0.25,
        id_switch=True,
        communication_delay=2,
        asynchronous_execution=True,
        nonstationary=True,
        static_obstacles=True,
    ),
    FaultScenario("hard-safety-obstacles", static_obstacles=True),
)

POLICIES = (
    "tracer-persistent",
    "tracer-hazard",
    "tracer-change-point",
    "tracer-forgetting",
    "tracer-reset",
    "mpogames-style",
    "mpogames-pomdp-style",
    "lucidgames-ukf-style",
    "atom-ukf-style",
    "orca-style",
    "mpc-style",
)

RELATED_UKF_POLICIES = {"lucidgames-ukf-style", "atom-ukf-style"}
RELATED_BELIEF_POLICIES = RELATED_UKF_POLICIES | {"mpogames-pomdp-style"}


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
        "branch": run("branch", "--show-current"),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ci(values: list[float], seed: int, samples: int = 4000) -> dict[str, Any]:
    if not values:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "count": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "count": int(len(array)),
    }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _set_modes(simulator: MultiEntityIndependentSocialForceSimulator, modes: np.ndarray) -> None:
    """Change the active force parameters without changing scene geometry."""

    simulator.modes = np.asarray(modes, dtype=np.int64).copy()
    simulator.parameter_coordinates = simulator.modes.astype(np.float32)
    table = {
        "relaxation_time": simulator.parameters.relaxation_time,
        "repulsion_strength": simulator.parameters.repulsion_strength,
        "repulsion_decay": simulator.parameters.repulsion_decay,
        "personal_radius": simulator.parameters.personal_radius,
        "group_pressure": simulator.parameters.group_pressure,
    }
    simulator.parameter_values = {
        name: np.asarray(values, dtype=np.float32)[simulator.modes]
        for name, values in table.items()
    }


def _obstacles(
    simulator: MultiEntityIndependentSocialForceSimulator,
    seed: int,
) -> np.ndarray:
    """Return deterministic circular obstacles in world coordinates."""

    rng = np.random.default_rng(seed)
    center = simulator.robot_positions.mean(axis=0)
    entity_center = simulator.entity_positions.mean(axis=0)
    direction = entity_center - center
    norm = max(float(np.linalg.norm(direction)), 1e-6)
    direction = direction / norm
    lateral = np.asarray([-direction[1], direction[0]])
    points = np.stack(
        [center + 0.45 * direction + 0.55 * lateral,
         center + 0.95 * direction - 0.45 * lateral],
        axis=0,
    )
    points += rng.normal(0.0, 0.08, size=points.shape)
    return np.concatenate([points, np.full((2, 1), 0.18)], axis=1).astype(np.float32)


def _static_safe_mask(
    simulator: MultiEntityIndependentSocialForceSimulator,
    pool: CandidatePool,
    obstacles: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check every candidate against obstacles and robot-robot collisions."""

    candidate = pool.robot_candidates.numpy()[:, 0, :, :, :2]
    world = candidate + simulator.entity_positions[0][None, None, None, :]
    obstacle_violation = np.zeros(len(candidate), dtype=bool)
    for center_x, center_y, radius in obstacles:
        distance = np.linalg.norm(world - np.asarray([center_x, center_y]), axis=-1)
        obstacle_violation |= (distance < float(radius) + float(margin)).any(axis=(1, 2))
    robot_violation = np.zeros(len(candidate), dtype=bool)
    for first in range(world.shape[1]):
        for second in range(first + 1, world.shape[1]):
            distance = np.linalg.norm(world[:, first] - world[:, second], axis=-1)
            robot_violation |= (distance < float(margin)).any(axis=1)
    violation = obstacle_violation | robot_violation
    return ~violation, obstacle_violation, robot_violation


def _async_pool(pool: CandidatePool, execute_steps: int) -> CandidatePool:
    candidates = pool.robot_candidates.clone()
    if candidates.shape[1] < 2:
        return pool
    # Robot slot 1 executes one control period late; all other slots are unchanged.
    original = candidates[:, 1].clone()
    current = pool.robot_current[:, 1]
    candidates[:, 1, 0, :2] = current[:, :2]
    candidates[:, 1, 0, 2:] = 0.0
    if candidates.shape[2] > 1:
        candidates[:, 1, 1:] = original[:, :-1]
    return CandidatePool(pool.robot_current, candidates, pool.activities)


def _observed_histories(
    simulator: MultiEntityIndependentSocialForceSimulator,
    scenario: FaultScenario,
    window: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histories = simulator.histories.copy()
    entity_map = np.arange(simulator.num_entities)
    if scenario.id_switch and window >= max(1, simulator.spec.windows // 2):
        entity_map[:2] = entity_map[[1, 0]]
        histories = histories[entity_map]
    if scenario.partial_observation:
        # An occluded history is represented by the last visible location.
        histories[:, :-2:2] = histories[:, -2:-1]
    actual_active = simulator.active_mask()
    observed_active = actual_active.copy()
    if scenario.missed_detection_rate:
        active_indices = np.flatnonzero(observed_active)
        dropped = rng.random(len(active_indices)) < scenario.missed_detection_rate
        observed_active[active_indices[dropped]] = False
        if not observed_active.any() and active_indices.size:
            observed_active[active_indices[0]] = True
    return histories, entity_map, observed_active


def _model_outputs(
    model: torch.nn.Module,
    simulator: MultiEntityIndependentSocialForceSimulator,
    pool: CandidatePool,
    histories: np.ndarray,
    include_pair: bool,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    entities = simulator.num_entities
    candidates = int(pool.robot_candidates.shape[0])
    history = torch.from_numpy(histories).to(device=device, dtype=torch.float32)
    history = history[:, None, None].expand(entities, candidates, 1, *history.shape[1:])
    history = history.reshape(entities * candidates, 1, history.shape[-2], 2)
    current = pool.robot_current.permute(1, 0, 2, 3).reshape(
        entities * candidates, simulator.spec.num_robots, 4
    ).to(device)
    actions = pool.robot_candidates.permute(1, 0, 2, 3, 4).reshape(
        entities * candidates,
        simulator.spec.num_robots,
        simulator.spec.horizon,
        4,
    ).to(device)
    mask = torch.ones(
        entities * candidates, 1, simulator.spec.num_robots,
        device=device,
        dtype=torch.bool,
    )
    output = model(
        history,
        current[:, None],
        actions[:, None],
        include_pair_interactions=include_pair,
        robot_mask=mask,
    )
    reference = stationary_reference_candidates(current[:, None], simulator.spec.horizon)
    reference_output = model(
        history,
        current[:, None],
        reference,
        include_pair_interactions=include_pair,
        robot_mask=mask,
    )
    mode_count = model.config.num_modes
    means = output["mean"][:, 0].reshape(
        entities, candidates, mode_count, simulator.spec.horizon, 2
    )
    log_stds = output["log_std"][:, 0].reshape_as(means)
    reference_means = reference_output["mean"][:, 0].reshape_as(means)
    return (
        {"mean": means, "log_std": log_stds, "reference_mean": reference_means},
        {"current": current.reshape(entities, candidates, simulator.spec.num_robots, 4),
         "actions": actions.reshape(entities, candidates, simulator.spec.num_robots, simulator.spec.horizon, 4)},
    )


def _orca_scores(
    simulator: MultiEntityIndependentSocialForceSimulator,
    pool: CandidatePool,
    active: np.ndarray,
    hard_safe: np.ndarray,
) -> np.ndarray:
    """A deterministic velocity-obstacle style team baseline."""

    candidate = pool.robot_candidates.numpy()[:, 0, :, :, :2]
    world = candidate + simulator.entity_positions[0][None, None, None, :]
    active_positions = simulator.entity_positions[active]
    distance = np.linalg.norm(world[:, :, :, None] - active_positions[None, None, None, :, :], axis=-1)
    clearance = distance.min(axis=(1, 2, 3)) if active_positions.size else np.full(len(candidate), 10.0)
    progress = np.linalg.norm(candidate[:, :, -1], axis=-1).mean(axis=1)
    scores = -0.8 * clearance - 0.15 * progress
    return np.where(hard_safe, scores, np.inf)


def _mpc_scores(
    simulator: MultiEntityIndependentSocialForceSimulator,
    pool: CandidatePool,
    active: np.ndarray,
    hard_safe: np.ndarray,
    dt: float,
) -> np.ndarray:
    """A finite-horizon constant-velocity MPC baseline on the same candidates."""

    candidate = pool.robot_candidates.numpy()[:, 0, :, :, :2]
    current = simulator.robot_positions
    world = candidate + simulator.entity_positions[0][None, None, None, :]
    if len(simulator.histories[active]):
        velocity = (simulator.histories[active, -1] - simulator.histories[active, -2]) / dt
        target = simulator.entity_positions[active, None] + velocity[:, None] * dt * np.arange(
            1, simulator.spec.horizon + 1, dtype=np.float32
        )[None, :, None]
        distance = np.linalg.norm(world[:, :, None, :, :] - target[None, None, :, :, :], axis=-1)
        min_distance = distance.min(axis=(1, 2, 3))
    else:
        min_distance = np.full(len(candidate), 10.0)
    displacement = candidate[:, :, simulator.spec.execute_steps - 1] - current[None]
    toward = simulator.entity_positions[active].mean(axis=0) - current.mean(axis=0) if active.any() else np.zeros(2)
    task = np.sum(displacement * toward[None, None], axis=-1).mean(axis=1)
    scores = -0.35 * task - 0.8 * min_distance
    return np.where(hard_safe, scores, np.inf)


def _mpogames_scores(
    costs: dict[str, torch.Tensor],
    means: torch.Tensor,
    active: np.ndarray,
    hard_safe: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Score a local maximum-entropy multimodal game-style baseline.

    The official MPOGames method is not reimplemented here. This controlled
    baseline keeps the same candidate/action interface, updates a categorical
    response belief from the executed prefix, and adds a mode-dispersion hedge
    rather than selecting from a single mean response.
    """

    active_indices = np.flatnonzero(active)
    if active_indices.size:
        active_means = means[active_indices]
        mode_center = active_means.mean(dim=2, keepdim=True)
        dispersion = (active_means - mode_center).square().mean(dim=(0, 2, 3, 4))
    else:
        dispersion = torch.zeros_like(costs["total"])
    scores = costs["total"] + temperature * dispersion
    safe = costs["safe_mask"].detach().cpu().numpy().astype(bool) & hard_safe
    values = scores.detach().cpu().numpy().astype(np.float64)
    return np.where(safe, values, np.inf)


def _update_belief(
    belief: torch.Tensor,
    likelihood: torch.Tensor,
    policy: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, bool]:
    if policy == "tracer-reset":
        prior = uniform_log_belief(belief.numel(), device=belief.device)
        return categorical_bayes_update(prior, args.likelihood_scale * likelihood), False
    if policy == "tracer-hazard":
        return hazard_belief_update(
            belief, args.likelihood_scale * likelihood, args.hazard_rate
        ), False
    if policy == "tracer-forgetting":
        posterior = categorical_bayes_update(
            belief, args.likelihood_scale * likelihood
        )
        return apply_belief_forgetting(posterior, args.forgetting_rate), False
    if policy == "tracer-change-point":
        changed = bool(detect_change_point(
            belief, args.likelihood_scale * likelihood, args.change_point_threshold
        ))
        prior = uniform_log_belief(belief.numel(), device=belief.device) if changed else belief
        return categorical_bayes_update(prior, args.likelihood_scale * likelihood), changed
    if policy == "mpogames-pomdp-style":
        _, posterior = categorical_pomdp_update(
            belief.exp(), args.likelihood_scale * likelihood, args.mpogames_transition
        )
        return posterior.clamp_min(1e-9).log(), False
    return categorical_bayes_update(belief, args.likelihood_scale * likelihood), False


@torch.no_grad()
def _run_policy(
    model: torch.nn.Module | None,
    policy: str,
    scenario: FaultScenario,
    args: argparse.Namespace,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    planning = PlanningCostConfig(
        task_weight=1.0,
        safety_weight=2.0,
        human_weight=0.6,
        comfort_scale=1.0,
        collision_margin=args.collision_margin,
        task_steps=args.execute_steps,
        violation_probability_alpha=args.risk_alpha,
        risk_samples=args.risk_samples,
    )
    scoring = PlanningCostConfig(
        task_weight=planning.task_weight,
        safety_weight=planning.safety_weight,
        human_weight=getattr(args, "belief_cost_weight", planning.human_weight),
        comfort_scale=planning.comfort_scale,
        collision_margin=planning.collision_margin,
        task_steps=planning.task_steps,
        violation_probability_alpha=planning.violation_probability_alpha,
        risk_samples=planning.risk_samples,
    )
    episodes: list[dict[str, Any]] = []
    for episode in range(args.episodes):
        simulator = MultiEntityIndependentSocialForceSimulator(
            _build_spec(args, split, args.num_robots),
            episode,
            args.num_entities,
            args.active_entities,
        )
        rng = np.random.default_rng(args.data_seed + seed * 1009 + episode * 9176)
        initial_modes = simulator.modes.copy()
        beliefs = torch.stack([
            uniform_log_belief(model.config.num_modes, device=args.device)
            if model is not None else torch.zeros(3, device=args.device)
            for _ in range(args.num_entities)
        ])
        ukf_states: list[UKFParameterBelief | None] = [None] * args.num_entities
        if policy in RELATED_UKF_POLICIES:
            ukf_config = UKFBeliefConfig(
                num_modes=model.config.num_modes,
                prior_variance=1.0,
                process_variance=(
                    0.015 if policy == "lucidgames-ukf-style" else 0.08
                ),
                observation_variance=float(args.response_noise) ** 2,
                adaptive_process=policy == "atom-ukf-style",
                max_process_variance=0.35,
            )
            ukf_states = [
                UKFParameterBelief(ukf_config, device=args.device)
                for _ in range(args.num_entities)
            ]
        pending: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        values: dict[str, list[float]] = {
            "realized_total_cost": [], "task_progress": [], "safety_cost": [],
            "human_cost": [], "min_distance": [], "collision_rate": [],
            "oracle_regret": [], "oracle_hit_rate": [], "hard_safe_fraction": [],
            "hard_safety_rejections": [], "hard_safety_fallback": [],
            "obstacle_collision": [], "belief_selection_flip": [],
            "robot_collision": [], "hard_safety_violation": [],
            "hard_safety_violation_without_fallback": [],
            "change_point_detected": [], "visible_entity_fraction": [],
        }
        for window in range(args.windows):
            if scenario.nonstationary and window == args.switch_window:
                _set_modes(simulator, (simulator.modes + 1) % 3)
            pool = simulator.candidate_pool()
            if scenario.asynchronous_execution:
                pool = _async_pool(pool, args.execute_steps)
            actual_active = simulator.active_mask()
            histories, entity_map, observed_active = _observed_histories(
                simulator, scenario, window, rng
            )
            obstacle_array = (
                _obstacles(simulator, args.data_seed + episode)
                if scenario.static_obstacles else np.empty((0, 3), dtype=np.float32)
            )
            hard_safe, obstacle_violation, robot_violation = _static_safe_mask(
                simulator, pool, obstacle_array, args.hard_safety_margin
            ) if scenario.static_obstacles else (
                np.ones(args.num_candidates, dtype=bool),
                np.zeros(args.num_candidates, dtype=bool),
                np.zeros(args.num_candidates, dtype=bool),
            )
            if model is not None:
                output, tensors = _model_outputs(
                    model, simulator, pool, histories,
                    include_pair=True, device=args.device,
                )
                active_for_planning = observed_active
                current = tensors["current"]
                actions = tensors["actions"]
                planning_beliefs = beliefs.exp()
                if policy == "mpogames-pomdp-style":
                    planning_beliefs = torch.stack([
                        categorical_pomdp_predict(
                            beliefs[entity].exp(), args.mpogames_transition
                        )
                        for entity in range(args.num_entities)
                    ])
                elif policy in RELATED_UKF_POLICIES:
                    planning_beliefs = torch.stack([
                        state.probabilities()
                        for state in ukf_states
                        if state is not None
                    ])
                costs = _multi_entity_predicted_costs(
                    output["mean"], output["log_std"], output["reference_mean"],
                    current, actions, planning_beliefs, active_for_planning, args, scoring,
                )
                model_safe = costs["safe_mask"].detach().cpu().numpy().astype(bool)
                safe = model_safe & hard_safe
                scores = (
                    _mpogames_scores(
                        costs,
                        output["mean"].reshape(
                            simulator.num_entities,
                            args.num_candidates,
                            model.config.num_modes,
                            simulator.spec.horizon,
                            2,
                        ),
                        active_for_planning,
                        hard_safe,
                        args.mpogames_temperature,
                    )
                    if policy == "mpogames-style"
                    else costs["total"].detach().cpu().numpy().astype(np.float64)
                )
                scores[~safe] = np.inf
                fallback = not np.isfinite(scores).any()
                if fallback:
                    selected = int(np.lexsort((costs["violation_probability"].detach().cpu().numpy(), ~hard_safe))[0])
                else:
                    selected = int(np.argmin(scores))
                uniform_costs = _multi_entity_predicted_costs(
                    output["mean"], output["log_std"], output["reference_mean"],
                    current, actions,
                    torch.full_like(beliefs.exp(), 1.0 / model.config.num_modes),
                    active_for_planning, args, scoring,
                )
                uniform_scores = uniform_costs["total"].detach().cpu().numpy().astype(np.float64)
                uniform_safe = uniform_costs["safe_mask"].detach().cpu().numpy().astype(bool) & hard_safe
                uniform_scores[~uniform_safe] = np.inf
                uniform_selected = int(np.argmin(uniform_scores)) if np.isfinite(uniform_scores).any() else int(np.argmin(uniform_costs["violation_probability"].detach().cpu().numpy()))
            else:
                if policy == "orca-style":
                    scores = _orca_scores(simulator, pool, observed_active, hard_safe)
                else:
                    scores = _mpc_scores(simulator, pool, observed_active, hard_safe, args.dt)
                fallback = not np.isfinite(scores).any()
                selected = int(np.argmin(scores)) if not fallback else int(np.argmin(obstacle_violation.astype(float)))
                uniform_selected = selected
                output = None
                tensors = None
            outcomes = simulator.counterfactuals(pool, planning)
            true_costs = _multi_entity_true_costs(outcomes, actual_active, simulator.spec, planning)
            oracle = int(np.argmin(true_costs))
            realized = simulator.step(pool, selected, planning)
            active_indices = np.flatnonzero(actual_active)
            if active_indices.size:
                task = float(realized.task_progress[active_indices].mean())
                safety = float(realized.safety_cost[active_indices].mean())
                human = float(realized.human_cost[active_indices].mean())
                minimum = float(realized.min_distance[active_indices].min())
                collision = float(realized.collision[active_indices].max())
            else:
                task = safety = human = minimum = collision = 0.0
            total = -task / (args.execute_steps * args.dt * 1.7) + 2.0 * safety + 0.6 * human
            values["realized_total_cost"].append(total)
            values["task_progress"].append(task)
            values["safety_cost"].append(safety)
            values["human_cost"].append(human)
            values["min_distance"].append(minimum)
            values["collision_rate"].append(collision)
            values["oracle_regret"].append(max(float(true_costs[selected] - true_costs[oracle]), 0.0))
            values["oracle_hit_rate"].append(float(selected == oracle))
            values["hard_safe_fraction"].append(float(hard_safe.mean()))
            values["hard_safety_rejections"].append(float((~hard_safe).sum()))
            values["hard_safety_fallback"].append(float(fallback))
            values["obstacle_collision"].append(float(obstacle_violation[selected]))
            values["robot_collision"].append(float(robot_violation[selected]))
            selected_hard_violation = float(not hard_safe[selected])
            values["hard_safety_violation"].append(selected_hard_violation)
            values["hard_safety_violation_without_fallback"].append(
                selected_hard_violation if not fallback else 0.0
            )
            values["belief_selection_flip"].append(float(selected != uniform_selected))
            values["visible_entity_fraction"].append(float(observed_active[actual_active].mean()) if actual_active.any() else 1.0)
            if model is not None:
                changed_count = 0
                if policy.startswith("tracer-") or policy in {
                    "mpogames-style",
                    "mpogames-pomdp-style",
                    *RELATED_UKF_POLICIES,
                }:
                    for entity in np.flatnonzero(observed_active):
                        response_prefix = realized.response[
                            entity, :args.execute_steps
                        ].to(args.device)
                        mode_means = output["mean"][
                            entity, selected, :, :args.execute_steps
                        ].detach()
                        likelihood = trajectory_mode_log_likelihood(
                            mode_means[None],
                            output["log_std"][entity, selected][None, :, :args.execute_steps],
                            response_prefix[None],
                        )[0]
                        destination = int(entity_map[entity]) if scenario.id_switch else int(entity)
                        ready = window + scenario.communication_delay
                        pending.append(
                            (
                                ready,
                                destination,
                                likelihood.detach(),
                                mode_means,
                                response_prefix.detach(),
                            )
                        )
                while pending and pending[0][0] <= window:
                    _, entity, likelihood, mode_means, response_prefix = pending.pop(0)
                    if policy in RELATED_UKF_POLICIES:
                        state = ukf_states[entity]
                        if state is None:
                            raise RuntimeError("missing UKF state for related baseline")
                        state.update(mode_means, response_prefix)
                        changed = False
                    else:
                        beliefs[entity], changed = _update_belief(
                            beliefs[entity], likelihood, policy, args
                        )
                    changed_count += int(changed)
                values["change_point_detected"].append(float(changed_count))
            else:
                values["change_point_detected"].append(0.0)
        record: dict[str, Any] = {
            "seed": seed, "split": split, "scenario": scenario.name,
            "policy": policy, "episode": episode,
            "initial_modes": initial_modes.tolist(), "final_modes": simulator.modes.tolist(),
        }
        record.update({name: _mean(items) for name, items in values.items()})
        if model is not None:
            posterior = (
                torch.stack([state.probabilities() for state in ukf_states if state is not None])
                if policy in RELATED_UKF_POLICIES
                else beliefs.exp()
            )
            record["final_mode_accuracy"] = float((posterior.argmax(dim=-1).cpu().numpy() == simulator.modes).mean())
            record["final_belief_entropy"] = float((-(posterior * posterior.clamp_min(1e-9).log()).sum(dim=-1)).mean())
        else:
            record["final_mode_accuracy"] = float("nan")
            record["final_belief_entropy"] = float("nan")
        episodes.append(record)
    return episodes


def _available_external_environments() -> dict[str, Any]:
    modules = {
        "socialgym2": "socialgym",
        "crowdnav": "crowd_nav",
        "rvo2_orca": "rvo2",
        "casadi_mpc": "casadi",
    }
    status = {name: {"installed": importlib.util.find_spec(module) is not None, "module": module} for name, module in modules.items()}
    status["real_robot"] = {
        "installed": False,
        "module": None,
        "reason": "No ROS/robot hardware endpoint is configured in this workspace.",
    }
    return status


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    git = _git_state()
    if git["dirty"]:
        raise RuntimeError("gap-closure experiments require a clean Git worktree")
    checkpoint_root = Path(args.checkpoint_dir)
    records: list[dict[str, Any]] = []
    scenarios = [item for item in SCENARIOS if item.name in args.scenarios]
    for seed in args.seeds:
        model = _load_model(checkpoint_root / f"seed{seed}" / "response.pt", args.device)
        for scenario in scenarios:
            for policy in POLICIES:
                # Related-belief policies share the frozen TRACER response
                # model, while ORCA/MPC remain model-free system controls.
                policy_model = model if policy.startswith("tracer-") or policy in {
                    "mpogames-style",
                    "mpogames-pomdp-style",
                    *RELATED_UKF_POLICIES,
                } else None
                records.extend(_run_policy(policy_model, policy, scenario, args, args.split, seed))
                print(f"gap_closure seed={seed} scenario={scenario.name} policy={policy} episodes={args.episodes}", flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with (output_dir / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary: dict[str, Any] = {"scenarios": {}, "paired_contrasts": {}}
    for scenario in scenarios:
        summary["scenarios"][scenario.name] = {}
        for policy in POLICIES:
            selected = [row for row in records if row["scenario"] == scenario.name and row["policy"] == policy]
            summary["scenarios"][scenario.name][policy] = {
                metric: _ci([float(row[metric]) for row in selected], args.bootstrap_seed + index)
                for index, metric in enumerate(("realized_total_cost", "task_progress", "collision_rate", "oracle_regret", "hard_safe_fraction", "hard_safety_fallback", "obstacle_collision", "robot_collision", "hard_safety_violation", "hard_safety_violation_without_fallback", "final_mode_accuracy", "final_belief_entropy", "belief_selection_flip", "change_point_detected", "visible_entity_fraction"))
            }
        for first, second in (
            ("tracer-persistent", "tracer-reset"),
            ("tracer-persistent", "mpogames-style"),
            ("tracer-persistent", "mpogames-pomdp-style"),
            ("tracer-persistent", "lucidgames-ukf-style"),
            ("tracer-persistent", "atom-ukf-style"),
            ("tracer-persistent", "orca-style"),
            ("tracer-persistent", "mpc-style"),
        ):
            left = {(row["seed"], row["episode"]): row for row in records if row["scenario"] == scenario.name and row["policy"] == first}
            right = {(row["seed"], row["episode"]): row for row in records if row["scenario"] == scenario.name and row["policy"] == second}
            delta = [float(right[key]["realized_total_cost"] - left[key]["realized_total_cost"]) for key in left.keys() & right.keys()]
            summary["paired_contrasts"][f"{scenario.name}:{second}-persistent"] = _ci(delta, args.bootstrap_seed + len(summary["paired_contrasts"]))
    metadata = {
        "protocol": "tracer-gap-closure-v1",
        "backend": MultiEntityIndependentSocialForceSimulator.backend_name,
        "claim_boundary": "TRACER closed-loop robustness and method-level related-belief evidence; no real-human causal claim and no official complete literature reimplementation claim",
        "git": git,
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": list(args.seeds),
        "episodes_per_policy_scenario": args.episodes,
        "scenarios": [scenario.__dict__ for scenario in scenarios],
        "policies": list(POLICIES),
        "parameters": vars(args),
        "external_environment_audit": _available_external_environments(),
        "runtime_seconds": time.perf_counter() - start,
    }
    payload = {"metadata": _jsonable(metadata), "summary": _jsonable(summary)}
    (output_dir / "metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# TRACER Gap-Closure Suite", "", "## Scope", "",
        f"Backend: `{metadata['backend']}`; commit: `{git['commit']}`; clean: `{not git['dirty']}`.",
        f"CUDA: `{metadata['cuda_available']}` (`{metadata['cuda_device_name']}`); seeds `{args.seeds}`; episodes per cell `{args.episodes}`.",
        "All cells use the TRACER response model, ID-bound belief, candidate risk filtering, and realized resettable counterfactual outcomes when the policy is a TRACER policy.",
        "",
        "## Main Results", "",
    ]
    for scenario in scenarios:
        persistent = summary["scenarios"][scenario.name]["tracer-persistent"]["realized_total_cost"]
        reset = summary["scenarios"][scenario.name]["tracer-reset"]["realized_total_cost"]
        lines.append(
            f"- `{scenario.name}` Persistent cost `{persistent['mean']:.4f}`; Reset `{reset['mean']:.4f}`; "
            f"Reset-Persistent `{summary['paired_contrasts'][scenario.name + ':tracer-reset-persistent']['mean']:.4f}` "
            f"CI `[{summary['paired_contrasts'][scenario.name + ':tracer-reset-persistent']['ci95_low']:.4f}, {summary['paired_contrasts'][scenario.name + ':tracer-reset-persistent']['ci95_high']:.4f}]`; "
            f"hard-safe fraction `{summary['scenarios'][scenario.name]['tracer-persistent']['hard_safe_fraction']['mean']:.3f}`.",
        )
    lines += ["", "## Baselines", ""]
    for scenario in scenarios:
        for policy in (
            "mpogames-style",
            "mpogames-pomdp-style",
            "lucidgames-ukf-style",
            "atom-ukf-style",
            "orca-style",
            "mpc-style",
        ):
            metric = summary["scenarios"][scenario.name][policy]["realized_total_cost"]
            lines.append(f"- `{scenario.name}` `{policy}` cost `{metric['mean']:.4f}` CI `[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]`.")
    lines += ["", "## External Environment Gate", ""]
    for name, status in metadata["external_environment_audit"].items():
        lines.append(f"- `{name}` installed `{status['installed']}`.")
    lines += [
        "",
        "The external gate is intentionally explicit: without the named third-party package or a configured robot endpoint, this run does not claim SocialGym2/CrowdNav/real-robot execution. The MPOGames-POMDP, LUCIDGames-UKF, and AToM-UKF rows are method-level belief components under the common TRACER interface; MPOGames-style, ORCA-style, and MPC-style remain local system baselines, not official complete literature reimplementations.",
        "",
    ]
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "status": "COMPLETED",
        "protocol": metadata["protocol"],
        "git": git,
        "claim_boundary": metadata["claim_boundary"],
        "artifacts": {name: _sha256(output_dir / name) for name in ("metadata.json", "summary.json", "episode_metrics.csv", "report.md")},
        "external_environment_audit": metadata["external_environment_audit"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "completion.json").write_text(json.dumps({"status": "COMPLETED", "finished_unix": time.time()}, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/tracer_gap_closure_20260902")
    parser.add_argument("--checkpoint-dir", default="experiments/independent_multientity_suite_20260902_final/checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2031, 2039, 2053, 2063])
    parser.add_argument("--scenarios", nargs="+", default=[scenario.name for scenario in SCENARIOS])
    parser.add_argument("--split", choices=["id", "ood"], default="ood")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=618731)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--num-entities", type=int, default=6)
    parser.add_argument("--active-entities", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.3)
    parser.add_argument("--response-noise", type=float, default=0.08)
    parser.add_argument("--entity-max-speed", type=float, default=2.2)
    parser.add_argument("--scene-family", default="crossing")
    parser.add_argument("--likelihood-scale", type=float, default=0.05)
    parser.add_argument("--hazard-rate", type=float, default=0.18)
    parser.add_argument("--forgetting-rate", type=float, default=0.18)
    parser.add_argument("--change-point-threshold", type=float, default=5.0)
    parser.add_argument("--switch-window", type=int, default=4)
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument("--hard-safety-margin", type=float, default=0.08)
    parser.add_argument("--risk-alpha", type=float, default=0.1)
    parser.add_argument("--risk-samples", type=int, default=32)
    parser.add_argument("--mpogames-temperature", type=float, default=0.15)
    parser.add_argument("--mpogames-transition", type=float, default=0.10)
    parser.add_argument("--bootstrap-seed", type=int, default=618731)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
