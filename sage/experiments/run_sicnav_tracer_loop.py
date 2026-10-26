"""Run the TRACER response-belief-planning loop in official CrowdSimPlus.

This runner is intentionally separate from ``run_sicnav_official_benchmark``.
The latter measures the official SICNav policies.  Here, the official
CrowdSimPlus environment supplies the human dynamics, reset protocol, static
obstacles, and execution; TRACER supplies the response model, per-human
beliefs, finite candidate planner, and executed robot actions.
"""

from __future__ import annotations

import copy
import argparse
import configparser
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sage.belief.filter import (
    adaptive_categorical_update,
    apply_belief_forgetting,
    categorical_bayes_update,
    detect_change_point,
    hazard_belief_update,
    uniform_log_belief,
)
from sage.belief.adaptive_persistent import (
    AdaptivePersistentBelief,
    AdaptivePersistentBeliefConfig,
)
from sage.experiments.eval_persistent_closed_loop import _load_model
from sage.experiments.planner_sensitivity import (
    BELIEF_SENSITIVITY_METRICS,
    belief_sensitivity_diagnostics,
    candidate_score_decomposition,
    mode_cost_selection_envelope,
)
from sage.response.closed_loop import PlanningCostConfig, expected_candidate_costs
from sage.response.external_checkpoint_adapter import ExternalResponseCheckpointAdapter
from sage.experiments.navigation_contract import initial_navigation_state, navigation_outcome, validate_response_clock
from sage.experiments.live_response_audit import collect_live_responses, save_response_audit
from sage.response.unicycle_candidates import UnicycleLimits, braking_after_prefix, diversify_unicycle_prefixes, project_unicycle_candidates
from sage.response.static_route import static_route
from sage.response.planning_support import aggregate_entity_costs, belief_cost_adjustment, swept_surface_clearance, goal_return_trajectories
from sage.response.persistent_models import stationary_reference_candidates, trajectory_mode_log_likelihood
from sage.response.sage_candidate_adapter import (
    SAGECandidateAdapter,
    SAGECandidateAdapterConfig,
)


POLICIES = (
    "tracer-persistent",
    "tracer-hazard",
    "tracer-forgetting",
    "tracer-change-point",
    "tracer-reset",
    "tracer-adaptive",
    "tracer-adaptive-persistent",
    "tracer-uniform",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    status = run("status", "--porcelain").splitlines()
    # Permit a clean audit worktree to mount the immutable external checkout.
    unexpected_status = [line for line in status if line != "?? raw_data/external"]
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(unexpected_status),
    }


def _import_official(external_root: Path):
    source = str(external_root.resolve())
    rvo2_source = external_root.parent / "python_rvo2_official"
    for item in (source, str(rvo2_source.resolve())):
        if item not in sys.path:
            sys.path.insert(0, item)
    import gym  # noqa: F401
    from crowd_sim_plus.envs.crowd_sim_plus import CrowdSimPlus
    from crowd_sim_plus.envs.policy.linear import Linear
    from crowd_sim_plus.envs.policy.orca import ORCA
    from crowd_sim_plus.envs.utils.action import ActionXY
    from crowd_sim_plus.envs.utils.robot_plus import Robot
    from crowd_sim_plus.envs.utils.state_plus import JointState

    return CrowdSimPlus, Linear, ORCA, ActionXY, Robot, JointState


def _bootstrap(values: list[float], seed: int, samples: int = 4000) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "count": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "count": int(len(array)),
    }


def _finite_mean(values: list[float]) -> float:
    """Average finite audit values without emitting warnings for all-tie windows."""

    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _info_value(info: dict[str, Any], key: str) -> float:
    value = info.get(key)
    return float(value.val) if value is not None else 0.0


def _new_episode(
    external_root: Path,
    case: int,
    env_config_path: Path,
    human_num: int | None = None,
    motion_profile: str = "holonomic",
):
    CrowdSimPlus, Linear, ORCA, ActionXY, Robot, JointState = _import_official(external_root)
    env_config = configparser.RawConfigParser()
    if not env_config.read(env_config_path):
        raise FileNotFoundError(env_config_path)
    # TRACER owns the robot action; Linear is only a holonomic kinematics
    # provider used by CrowdSimPlus for action validation and reset warm-up.
    env_config.set("robot", "policy", "linear")
    env_config.set("robot", "visible", "true")
    env = CrowdSimPlus()
    env.configure(env_config)
    policy = Linear()
    policy.configure(env_config)
    robot = Robot(env_config, "robot")
    robot.set_policy(policy)
    env.set_robot(robot)
    reset_kwargs: dict[str, Any] = {"return_stat": True}
    if human_num is not None:
        if human_num < 1:
            raise ValueError("human_num must be positive")
        reset_kwargs["testing_human_num"] = int(human_num)
    observation, static_obs = env.reset("test", case, **reset_kwargs)
    if motion_profile == "sicnav-unicycle":
        robot.kinematics = "unicycle"
        policy.kinematics = "unicycle"
        direction = np.asarray(robot.get_goal_position()) - np.asarray(robot.get_position())
        robot.theta = float(np.arctan2(direction[1], direction[0]))
    orca = ORCA()
    orca.configure(env_config)
    orca.time_step = env.time_step
    return env, robot, observation, static_obs, ActionXY, orca, JointState


def _velocity_action(robot, action_type, velocity, target_heading=None):
    if getattr(robot, "kinematics", "holonomic") != "unicycle":
        return action_type(float(velocity[0]), float(velocity[1]))
    from crowd_sim_plus.envs.utils.action import ActionRot
    speed = float(np.linalg.norm(velocity))
    heading = float(target_heading) if target_heading is not None else (float(np.arctan2(velocity[1], velocity[0])) if speed > 1e-8 else float(robot.theta))
    if target_heading is not None:
        speed = float(np.asarray(velocity) @ np.array([np.cos(heading), np.sin(heading)]))
    rotation = (heading - robot.theta + np.pi) % (2 * np.pi) - np.pi
    return ActionRot(speed, float(rotation))


def _action_velocity(robot, action):
    if hasattr(action, "vx"):
        return np.array([action.vx, action.vy], dtype=np.float32)
    heading = robot.theta + action.r
    return np.array([action.v * np.cos(heading), action.v * np.sin(heading)], dtype=np.float32)


def _candidate_pool(
    robot: Any,
    humans: list[Any],
    num_candidates: int,
    horizon: int,
    dt: float,
    rng: np.random.Generator,
    proposal_velocity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create common global candidate trajectories and human-relative views."""

    robot_position = np.asarray(robot.get_position(), dtype=np.float32)
    robot_velocity = np.asarray(robot.get_velocity(), dtype=np.float32)
    goal = np.asarray(robot.get_goal_position(), dtype=np.float32)
    direction = goal - robot_position
    norm = max(float(np.linalg.norm(direction)), 1e-6)
    direction /= norm
    tangent = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    activities = rng.uniform(0.15, 1.0, size=(num_candidates, 1)).astype(np.float32)
    activities[: min(3, num_candidates), 0] = np.asarray([0.15, 1.0, 0.55])[: min(3, num_candidates)]
    global_candidates = np.zeros((num_candidates, horizon, 4), dtype=np.float32)
    for index, activity in enumerate(activities[:, 0]):
        lateral = (1.0 - activity) * rng.uniform(-0.82, 0.82)
        heading = direction + tangent * lateral
        heading /= max(float(np.linalg.norm(heading)), 1e-6)
        speed = min(float(robot.v_pref), 0.30 + 0.70 * float(activity))
        turn = rng.uniform(-0.055, 0.055)
        position = robot_position.copy()
        for step in range(horizon):
            angle = turn * (step + 1)
            velocity = np.asarray(
                [
                    math.cos(angle) * heading[0] - math.sin(angle) * heading[1],
                    math.sin(angle) * heading[0] + math.cos(angle) * heading[1],
                ],
                dtype=np.float32,
            ) * speed
            position = position + velocity * dt
            global_candidates[index, step, :2] = position
            global_candidates[index, step, 2:] = velocity
    # Keep the first candidate as a low-speed feasible hold/progress branch.
    global_candidates[0, :, 2:] = 0.15 * direction
    global_candidates[0, :, :2] = robot_position + np.arange(1, horizon + 1, dtype=np.float32)[:, None] * dt * global_candidates[0, :, 2:]
    if proposal_velocity is not None and num_candidates > 1:
        proposal = np.asarray(proposal_velocity, dtype=np.float32).reshape(2)
        if np.isfinite(proposal).all():
            global_candidates[1, :, 2:] = proposal
            global_candidates[1, :, :2] = robot_position + np.arange(1, horizon + 1, dtype=np.float32)[:, None] * dt * proposal
    humans_position = np.asarray([human.get_position() for human in humans], dtype=np.float32)
    humans_velocity = np.asarray([human.get_velocity() for human in humans], dtype=np.float32)
    current = np.concatenate(
        [
            np.broadcast_to(robot_position, (len(humans), 2)),
            np.broadcast_to(robot_velocity, (len(humans), 2)),
        ],
        axis=-1,
    ).astype(np.float32)
    current[:, :2] -= humans_position
    candidates = np.broadcast_to(
        global_candidates[None], (len(humans), num_candidates, horizon, 4)
    ).copy()
    candidates[..., :2] -= humans_position[:, None, None, :]
    # Only positions are translated into the human-relative frame; velocities
    # remain shared in the world frame and are the action features expected by
    # TRACER.
    candidates[..., 2:] = global_candidates[None, ..., 2:]
    del humans_velocity
    return current, candidates, global_candidates


def _validate_executable_trajectory_contract(
    global_candidates: np.ndarray,
    robot_position: np.ndarray,
    dt: float,
    execute_steps: int,
    tolerance: float = 2e-3,
) -> None:
    """Validate the world-frame continuous trajectory sent to CrowdSimPlus.

    A candidate is represented as ``(x, y, vx, vy)`` at each future simulator
    tick.  The first position is integrated from the current robot position;
    every subsequent position is integrated from the previous position.  This
    prevents a generator that returns a prediction-only trajectory, a
    relative-coordinate path, or a different clock from silently entering the
    executor.
    """

    candidates = np.asarray(global_candidates, dtype=np.float64)
    position = np.asarray(robot_position, dtype=np.float64).reshape(-1)
    if candidates.ndim != 3 or candidates.shape[-1] != 4:
        raise ValueError("executable candidates must have shape (N, horizon, 4)")
    if position.size != 2:
        raise ValueError("robot_position must contain exactly two coordinates")
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("trajectory dt must be positive and finite")
    if not 1 <= int(execute_steps) <= candidates.shape[1]:
        raise ValueError("execute_steps must lie within the trajectory horizon")
    if not np.isfinite(candidates).all() or not np.isfinite(position).all():
        raise ValueError("executable candidates must contain finite values")
    positions = candidates[..., :2]
    velocities = candidates[..., 2:]
    expected_delta = velocities * float(dt)
    actual_delta = np.empty_like(positions)
    actual_delta[:, 0] = positions[:, 0] - position[None]
    if positions.shape[1] > 1:
        actual_delta[:, 1:] = positions[:, 1:] - positions[:, :-1]
    error = np.max(np.abs(actual_delta - expected_delta), axis=(1, 2))
    if bool(np.any(error > float(tolerance))):
        bad = int(np.argmax(error))
        raise ValueError(
            f"candidate {bad} violates executable trajectory contract: "
            f"max position integration error {float(error[bad]):.6f} > {float(tolerance):.6f}"
        )


def _project_sage_speed_limit(
    global_candidates: np.ndarray,
    robot: Any,
    dt: float,
    speed_scale: float,
) -> np.ndarray:
    """Project SAGE velocities to the official robot limit and re-integrate.

    SAGE is a trajectory generator rather than the CrowdSimPlus action
    controller. Its samples can exceed the external robot's preferred speed;
    clipping only the first action would break the position/velocity contract,
    so the complete path is re-integrated after projection.
    """

    candidates = np.asarray(global_candidates, dtype=np.float32).copy()
    if speed_scale <= 0.0:
        return candidates
    max_speed = float(getattr(robot, "v_pref", 0.0)) * float(speed_scale)
    if max_speed <= 0.0:
        return candidates
    velocity = candidates[..., 2:]
    norm = np.linalg.norm(velocity, axis=-1, keepdims=True)
    velocity *= np.minimum(1.0, max_speed / np.maximum(norm, 1e-6))
    position = np.asarray(robot.get_position(), dtype=np.float32)
    candidates[:, 0, :2] = position[None] + velocity[:, 0] * float(dt)
    if candidates.shape[1] > 1:
        candidates[:, 1:, :2] = candidates[:, :1, :2] + np.cumsum(
            velocity[:, 1:] * float(dt), axis=1
        )
    return candidates


def _sage_candidate_pool(
    adapter: SAGECandidateAdapter,
    history_positions: list[list[np.ndarray]],
    robot: Any,
    humans: list[Any],
    num_candidates: int,
    dt: float,
    sample_seed: int,
    history_mode: str = "per-human",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Generate executable world-frame candidates from SAGE.

    SAGE's adapter accepts the complete entity graph in the default
    ``per-human`` mode, so the generated robot path is conditioned on every
    observed human simultaneously. ``nearest`` is retained for reproducing
    the original single-focal-track protocol.
    """

    if not humans:
        raise ValueError("SAGE candidate generation requires at least one human")
    robot_position = np.asarray(robot.get_position(), dtype=np.float32)
    robot_velocity = np.asarray(robot.get_velocity(), dtype=np.float32)
    robot_goal = np.asarray(robot.get_goal_position(), dtype=np.float32)
    human_positions = np.asarray(
        [human.get_position() for human in humans], dtype=np.float32
    )
    if history_mode not in {"per-human", "nearest"}:
        raise ValueError("SAGE history_mode must be 'per-human' or 'nearest'")
    histories = [np.asarray(history, dtype=np.float32) for history in history_positions]
    if any(history.ndim != 2 or history.shape[-1] != 2 for history in histories):
        raise ValueError("SAGE human histories must have shape (history, 2)")
    if len({history.shape[0] for history in histories}) != 1:
        raise ValueError("all SAGE human histories must have the same length")
    if history_mode == "nearest":
        distances = np.linalg.norm(human_positions - robot_position[None], axis=-1)
        focal = int(distances.argmin())
        scene_history = histories[focal]
    else:
        # The adapter's multi-entity contract keeps all human tracks in one
        # graph, so SAGE can model their shared context before FRM scoring.
        scene_history = np.stack(histories, axis=0)
    pool = adapter.candidate_pool(
        scene_history,
        robot_position[None],
        robot_velocity[None],
        dt,
        sample_seed,
        robot_goals=robot_goal[None],
    )
    global_candidates = pool.robot_candidates.numpy()[:, 0]
    entity_predictions = (
        pool.entity_candidates.numpy()
        if pool.entity_candidates is not None
        else None
    )
    if history_mode == "nearest":
        global_candidates = global_candidates.copy()
        global_candidates[..., :2] += scene_history[-1][None, None, :]
    if global_candidates.shape[0] < num_candidates:
        raise ValueError(
            f"SAGE returned {global_candidates.shape[0]} candidates, need {num_candidates}"
        )
    global_candidates = global_candidates[:num_candidates]
    current = np.concatenate(
        [
            np.broadcast_to(robot_position, (len(humans), 2)),
            np.broadcast_to(robot_velocity, (len(humans), 2)),
        ],
        axis=-1,
    ).astype(np.float32)
    current[:, :2] -= human_positions
    candidates = np.broadcast_to(
        global_candidates[None],
        (len(humans), num_candidates, global_candidates.shape[1], 4),
    ).copy()
    candidates[..., :2] -= human_positions[:, None, None, :]
    if entity_predictions is not None:
        entity_predictions = entity_predictions[:num_candidates]
    return current, candidates, global_candidates, entity_predictions


def _sage_hybrid_candidate_pool(
    adapter: SAGECandidateAdapter,
    history_positions: list[list[np.ndarray]],
    robot: Any,
    humans: list[Any],
    num_candidates: int,
    horizon: int,
    dt: float,
    sample_seed: int,
    rng: np.random.Generator,
    history_mode: str = "per-human",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Keep SAGE diversity while reserving deterministic executable anchors."""

    sage_count = max(1, num_candidates - min(4, num_candidates - 1))
    current, candidates, global_candidates, entity_predictions = _sage_candidate_pool(
        adapter,
        history_positions,
        robot,
        humans,
        sage_count,
        dt,
        sample_seed,
        history_mode,
    )
    anchor_count = num_candidates - sage_count
    if anchor_count == 0:
        return current, candidates, global_candidates, entity_predictions
    anchor_current, anchor_candidates, anchor_global = _candidate_pool(
        robot,
        humans,
        anchor_count,
        global_candidates.shape[1],
        dt,
        rng,
    )
    return (
        current,
        np.concatenate([candidates, anchor_candidates], axis=1),
        np.concatenate([global_candidates, anchor_global], axis=0),
        None,
    )


def _relative_candidates(
    global_candidates: np.ndarray, humans: list[Any]
) -> np.ndarray:
    human_positions = np.asarray(
        [human.get_position() for human in humans], dtype=np.float32
    )
    candidates = np.broadcast_to(
        global_candidates[None],
        (len(humans), global_candidates.shape[0], global_candidates.shape[1], 4),
    ).copy()
    candidates[..., :2] -= human_positions[:, None, None, :]
    return candidates


def _augment_with_safe_recovery_candidates(
    current: np.ndarray,
    candidates: np.ndarray,
    global_candidates: np.ndarray,
    env: Any,
    robot: Any,
    humans: list[Any],
    static_obstacles: list[Any],
    action_type: Any,
    dt: float,
    kinematic_margin: float,
    static_margin: float,
    recovery_margin: float,
    max_recovery_candidates: int,
    direction_count: int,
    activation: str = "empty-support",
    activation_distance: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Append continuous recovery controls only when native support is empty.

    Recovery controls use observable state and the same one-step executor,
    human-body, and static-obstacle constraints as the planner. Native SAGE
    branches are preserved. The augmented set is subsequently scored by FRM;
    this helper never overrides the selected action at execution time.
    """

    executor_safe = _executor_safe_mask(env, robot, action_type, global_candidates)
    kinematic_safe = _kinematic_safe_mask(
        global_candidates,
        robot,
        humans,
        dt,
        kinematic_margin,
        prefix_steps=1,
    )
    static_safe = _static_obstacle_safe_mask(
        global_candidates,
        robot,
        static_obstacles,
        static_margin,
        prefix_steps=1,
    )
    if activation not in {"empty-support", "nearby", "always"}:
        raise ValueError(
            "recovery activation must be 'empty-support', 'nearby', or 'always'"
        )
    native_support = bool((executor_safe & kinematic_safe & static_safe).any())
    if humans:
        position = np.asarray(robot.get_position(), dtype=np.float32)
        nearest_distance = min(
            float(
                np.linalg.norm(
                    position - np.asarray(human.get_position(), dtype=np.float32)
                )
            )
            for human in humans
        )
    else:
        nearest_distance = float("inf")
    if native_support and (
        activation == "empty-support"
        or (activation == "nearby" and nearest_distance > float(activation_distance))
    ):
        return current, candidates, global_candidates, 0
    if max_recovery_candidates < 1 or direction_count < 4:
        return current, candidates, global_candidates, 0

    position = np.asarray(robot.get_position(), dtype=np.float32)
    goal = np.asarray(robot.get_goal_position(), dtype=np.float32)
    goal_direction = goal - position
    goal_angle = math.atan2(float(goal_direction[1]), float(goal_direction[0]))
    angles = goal_angle + np.linspace(
        -math.pi, math.pi, int(direction_count), endpoint=False, dtype=np.float32
    )
    speed_limit = max(float(robot.v_pref), 1e-6)
    speeds = speed_limit * np.asarray([0.0, 0.10, 0.25, 0.45, 0.70, 1.0], dtype=np.float32)
    velocities = [np.zeros(2, dtype=np.float32)]
    for speed in speeds[1:]:
        velocities.extend(
            np.stack([np.cos(angles), np.sin(angles)], axis=-1) * float(speed)
        )
    velocity_array = np.asarray(velocities, dtype=np.float32)
    horizon = int(global_candidates.shape[1])
    recovery = np.zeros((velocity_array.shape[0], horizon, 4), dtype=np.float32)
    recovery[..., 2:] = velocity_array[:, None]
    recovery[..., :2] = position[None, None] + (
        np.arange(1, horizon + 1, dtype=np.float32)[None, :, None]
        * float(dt)
        * velocity_array[:, None]
    )
    recovery_executor = _executor_safe_mask(env, robot, action_type, recovery)
    recovery_kinematic = _kinematic_safe_mask(
        recovery,
        robot,
        humans,
        dt,
        max(float(kinematic_margin), float(recovery_margin)),
        prefix_steps=1,
    )
    recovery_static = _static_obstacle_safe_mask(
        recovery,
        robot,
        static_obstacles,
        max(float(static_margin), 0.0),
        prefix_steps=1,
    )
    feasible = np.flatnonzero(recovery_executor & recovery_kinematic & recovery_static)
    best_effort = feasible.size == 0
    if feasible.size == 0:
        # In an already unsafe state no sampled control may satisfy the full
        # body margin. Retain executor-valid controls and let the fallback
        # choose the one with maximum predicted surface clearance. This is a
        # bounded observable-state recovery, not a simulator lookahead.
        feasible = np.flatnonzero(recovery_executor & recovery_static)
        if feasible.size == 0:
            return current, candidates, global_candidates, 0

    direction = goal_direction / max(float(np.linalg.norm(goal_direction)), 1e-6)
    task_progress = velocity_array[feasible] @ direction
    clearance = _kinematic_surface_clearance(
        recovery[feasible], robot, humans, dt, prefix_steps=1
    )
    # Strictly prioritize clearance for best-effort escape controls. For fully
    # feasible recovery support, retain task progress as the primary signal.
    utility = (
        clearance + 1e-3 * task_progress
        if best_effort
        else task_progress + 0.20 * clearance
    )
    order = feasible[np.argsort(-utility, kind="mergesort")]

    # Greedy velocity separation prevents the appended support from collapsing
    # to adjacent headings with indistinguishable actions.
    selected: list[int] = []
    separation = 0.12 * speed_limit
    for index in order:
        velocity = velocity_array[index]
        if selected and min(
            float(np.linalg.norm(velocity - velocity_array[other]))
            for other in selected
        ) < separation:
            continue
        selected.append(int(index))
        if len(selected) >= int(max_recovery_candidates):
            break
    if not selected:
        selected = [int(order[0])]
    appended = recovery[np.asarray(selected, dtype=np.int64)]
    combined_global = np.concatenate([global_candidates, appended], axis=0)
    return (
        current,
        _relative_candidates(combined_global, humans),
        combined_global,
        int(len(selected)),
    )


def _repair_sage_candidates(
    current: np.ndarray,
    candidates: np.ndarray,
    global_candidates: np.ndarray,
    env: Any,
    robot: Any,
    humans: list[Any],
    static_obstacles: list[Any],
    action_type: Any,
    dt: float,
    kinematic_margin: float,
    static_margin: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Replace only SAGE branches that the official executor cannot accept."""

    exact = _executor_safe_mask(env, robot, action_type, global_candidates)
    kinematic = _kinematic_safe_mask(
        global_candidates, robot, humans, dt, kinematic_margin, prefix_steps=1
    )
    static = _static_obstacle_safe_mask(
        global_candidates, robot, static_obstacles, static_margin, prefix_steps=1
    )
    invalid = ~(exact & kinematic & static)
    if not bool(invalid.any()):
        return current, candidates, global_candidates, 0
    _, _, anchors = _candidate_pool(
        robot, humans, int(invalid.sum()), global_candidates.shape[1], dt, rng
    )
    repaired = global_candidates.copy()
    repaired[invalid] = anchors
    return current, _relative_candidates(repaired, humans), repaired, int(invalid.sum())


def _histories(
    history_positions: list[list[np.ndarray]],
    humans: list[Any],
    history: int,
    dt: float,
) -> np.ndarray:
    values = []
    for index, human in enumerate(humans):
        observed = history_positions[index]
        position = np.asarray(human.get_position(), dtype=np.float32)
        velocity = np.asarray(human.get_velocity(), dtype=np.float32)
        if not observed:
            observed.append(position)
        if len(observed) > 1:
            velocity = (np.asarray(observed[1]) - np.asarray(observed[0])) / dt
        while len(observed) < history:
            observed.insert(0, np.asarray(observed[0], dtype=np.float32) - velocity * dt)
        values.append(np.asarray(observed[-history:], dtype=np.float32))
    return np.asarray(values, dtype=np.float32)


def _kinematic_safe_mask(
    global_candidates: np.ndarray,
    robot: Any,
    humans: list[Any],
    dt: float,
    margin: float,
    prefix_steps: int | None = None,
) -> np.ndarray:
    """Reject candidate prefixes that enter a constant-velocity body tube.

    The learned risk filter models response uncertainty, but its fallback can
    still select a candidate after every learned candidate is rejected.  This
    observed-state guard is deliberately separate from that model filter: it
    only uses current positions/velocities and body radii, and is disabled by
    a negative margin so legacy runs remain bit-for-bit comparable.
    """

    candidates = np.asarray(global_candidates, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[-1] != 4:
        raise ValueError("global_candidates must have shape (candidates, horizon, 4)")
    if margin < 0.0 or not humans:
        return np.ones(candidates.shape[0], dtype=bool)
    clearance = _kinematic_surface_clearance(
        candidates, robot, humans, dt, prefix_steps
    )
    return clearance >= float(margin) - 1e-6


def _kinematic_surface_clearance(
    global_candidates: np.ndarray,
    robot: Any,
    humans: list[Any],
    dt: float,
    prefix_steps: int | None = None,
) -> np.ndarray:
    """Minimum constant-velocity body-surface clearance for each branch."""

    candidates = np.asarray(global_candidates, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[-1] != 4:
        raise ValueError("global_candidates must have shape (candidates, horizon, 4)")
    if not humans:
        return np.full(candidates.shape[0], np.inf, dtype=np.float32)
    if prefix_steps is not None:
        candidates = candidates[:, : max(1, min(int(prefix_steps), candidates.shape[1]))]
    human_positions = np.asarray([human.get_position() for human in humans], dtype=np.float32)
    human_velocities = np.asarray([human.get_velocity() for human in humans], dtype=np.float32)
    steps = np.arange(1, candidates.shape[1] + 1, dtype=np.float32) * float(dt)
    human_predictions = human_positions[:, None, :] + human_velocities[:, None, :] * steps[None, :, None]
    distances = np.linalg.norm(
        candidates[None, :, :, :2] - human_predictions[:, None, :, :], axis=-1
    )
    body_radii = np.asarray(
        [float(robot.radius) + float(human.radius) for human in humans],
        dtype=np.float32,
    )
    surface_clearance = distances - body_radii[:, None, None]
    return surface_clearance.min(axis=(0, 2))


def _sage_dynamic_safe_mask(
    global_candidates: np.ndarray,
    entity_predictions: np.ndarray | None,
    robot: Any,
    humans: list[Any],
    margin: float,
    prefix_steps: int | None = None,
) -> np.ndarray:
    """Check robot branches against SAGE's candidate-conditioned entity futures."""

    candidates = np.asarray(global_candidates, dtype=np.float32)
    if entity_predictions is None or margin < 0.0 or not humans:
        return np.ones(candidates.shape[0], dtype=bool)
    predictions = np.asarray(entity_predictions, dtype=np.float32)
    if predictions.ndim != 4 or predictions.shape[0] != candidates.shape[0]:
        raise ValueError(
            "entity_predictions must have shape (candidates, entities, horizon, 2)"
        )
    if predictions.shape[1] != len(humans) or predictions.shape[2] != candidates.shape[1]:
        # Legacy nearest-track mode cannot support an all-human dynamic check.
        return np.ones(candidates.shape[0], dtype=bool)
    steps = predictions.shape[2]
    if prefix_steps is not None:
        steps = max(1, min(int(prefix_steps), steps))
    distances = np.linalg.norm(
        candidates[:, None, :steps, :2] - predictions[:, :, :steps, :], axis=-1
    )
    thresholds = np.asarray(
        [float(robot.radius) + float(human.radius) + float(margin) for human in humans],
        dtype=np.float32,
    )
    return np.all(distances >= thresholds[None, :, None], axis=(1, 2))


def _kinematic_clearance_cost(
    global_candidates: np.ndarray,
    robot: Any,
    humans: list[Any],
    dt: float,
    prefix_steps: int,
) -> np.ndarray:
    """Softly prefer observed-state clearance before trusting FRM ranking."""

    candidates = np.asarray(global_candidates, dtype=np.float32)
    if not humans:
        return np.zeros(candidates.shape[0], dtype=np.float32)
    steps = max(1, min(int(prefix_steps), candidates.shape[1]))
    human_positions = np.asarray([human.get_position() for human in humans], dtype=np.float32)
    human_velocities = np.asarray([human.get_velocity() for human in humans], dtype=np.float32)
    times = np.arange(1, steps + 1, dtype=np.float32) * float(dt)
    human_predictions = human_positions[:, None, :] + human_velocities[:, None, :] * times[None, :, None]
    distances = np.linalg.norm(
        candidates[None, :, :steps, :2] - human_predictions[:, None, :, :], axis=-1
    )
    minimum = distances.min(axis=(0, 2))
    threshold = max(float(robot.radius), 1e-6)
    return np.maximum(0.0, threshold + 0.35 - minimum).astype(np.float32) ** 2


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, direction) / denominator)
    fraction = min(1.0, max(0.0, fraction))
    return float(np.linalg.norm(point - (start + fraction * direction)))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    epsilon = 1e-9
    orientations = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    if (
        ((orientations[0] > epsilon and orientations[1] < -epsilon) or
         (orientations[0] < -epsilon and orientations[1] > epsilon))
        and ((orientations[2] > epsilon and orientations[3] < -epsilon) or
             (orientations[2] < -epsilon and orientations[3] > epsilon))
    ):
        return True

    def on_segment(start: np.ndarray, point: np.ndarray, end: np.ndarray) -> bool:
        return (
            min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
            and min(start[1], end[1]) - epsilon <= point[1] <= max(start[1], end[1]) + epsilon
        )

    return any(
        abs(value) <= epsilon and on_segment(start, point, end)
        for value, start, point, end in (
            (orientations[0], first_start, second_start, first_end),
            (orientations[1], first_start, second_end, first_end),
            (orientations[2], second_start, first_start, second_end),
            (orientations[3], second_start, first_end, second_end),
        )
    )


def _segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    if _segments_intersect(first_start, first_end, second_start, second_end):
        return 0.0
    return min(
        _point_segment_distance(first_start, second_start, second_end),
        _point_segment_distance(first_end, second_start, second_end),
        _point_segment_distance(second_start, first_start, first_end),
        _point_segment_distance(second_end, first_start, first_end),
    )


def _static_obstacle_safe_mask(
    global_candidates: np.ndarray,
    robot: Any,
    static_obstacles: list[Any],
    margin: float,
    prefix_steps: int | None = None,
) -> np.ndarray:
    """Reject candidate prefixes whose swept disk intersects a line obstacle."""

    candidates = np.asarray(global_candidates, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[-1] != 4:
        raise ValueError("global_candidates must have shape (candidates, horizon, 4)")
    if margin < 0.0 or not static_obstacles:
        return np.ones(candidates.shape[0], dtype=bool)
    if prefix_steps is not None:
        candidates = candidates[:, : max(1, min(int(prefix_steps), candidates.shape[1]))]
    robot_position = np.asarray(robot.get_position(), dtype=np.float64)
    positions = candidates[..., :2].astype(np.float64)
    starts = np.empty_like(positions)
    starts[:, 0] = robot_position
    if positions.shape[1] > 1:
        starts[:, 1:] = positions[:, :-1]
    threshold = float(robot.radius) + float(margin)
    safe = np.ones(candidates.shape[0], dtype=bool)
    for obstacle in static_obstacles:
        line = np.asarray(obstacle, dtype=np.float64)
        if line.shape != (2, 2):
            raise ValueError("static obstacles must contain two 2D endpoints")
        for candidate_index in range(candidates.shape[0]):
            for step in range(candidates.shape[1]):
                if _segment_distance(starts[candidate_index, step], positions[candidate_index, step], line[0], line[1]) < threshold:
                    safe[candidate_index] = False
                    break
    return safe


def _update_belief(
    belief: torch.Tensor,
    likelihood: torch.Tensor,
    policy: str,
    args: argparse.Namespace,
    adaptive_state: AdaptivePersistentBelief | None = None,
) -> tuple[torch.Tensor, bool]:
    scaled = args.likelihood_scale * likelihood
    if policy == "tracer-uniform":
        return uniform_log_belief(belief.numel(), device=belief.device), False
    if policy == "tracer-adaptive":
        posterior, _, _, changed = adaptive_categorical_update(
            belief, likelihood,
            base_hazard=getattr(args, "adaptive_base_hazard", 0.0),
            max_hazard=args.hazard_rate,
            surprise_threshold=args.change_point_threshold,
            surprise_scale=getattr(args, "adaptive_surprise_scale", 1.0),
            likelihood_scale=args.likelihood_scale,
        )
        return posterior, bool(changed)
    if policy == "tracer-adaptive-persistent":
        if adaptive_state is None:
            raise ValueError("tracer-adaptive-persistent requires an adaptive belief state")
        posterior, diagnostics = adaptive_state.update(likelihood)
        return posterior, bool(diagnostics["changed"])
    if policy == "tracer-reset":
        return categorical_bayes_update(uniform_log_belief(belief.numel(), device=belief.device), scaled), False
    if policy == "tracer-hazard":
        return hazard_belief_update(belief, scaled, args.hazard_rate), False
    if policy == "tracer-forgetting":
        return apply_belief_forgetting(categorical_bayes_update(belief, scaled), args.forgetting_rate), False
    if policy == "tracer-change-point":
        changed = bool(detect_change_point(belief, likelihood, args.change_point_threshold))
        prior = uniform_log_belief(belief.numel(), device=belief.device) if changed else belief
        return categorical_bayes_update(prior, scaled), changed
    return categorical_bayes_update(belief, scaled), False


def _belief_observation_due(window: int, execute_steps: int, stride: int | None) -> bool:
    if execute_steps < 1:
        raise ValueError("execute_steps must be positive")
    if stride is None:
        stride = execute_steps
    if stride < 1 or stride % execute_steps:
        raise ValueError("belief observation stride must be a positive multiple of execute_steps")
    return window * execute_steps % stride == 0


def _validate_external_belief_cases(checkpoint: dict, case_start: int, case_count: int) -> None:
    splits = checkpoint.get("provenance", {}).get("case_splits", {})
    if not splits.get("train") or not splits.get("val"):
        raise ValueError("external belief checkpoint must disclose training and calibration cases")
    used = set(splits["train"]) | set(splits["val"])
    if used & set(range(case_start, case_start + case_count)):
        raise ValueError("navigation cases overlap response training or calibration cases")


def _executor_safe_mask(
    env: Any,
    robot: Any,
    action_type: Any,
    global_candidates: np.ndarray,
    candidate_headings: np.ndarray | None = None,
) -> np.ndarray:
    """Match the official action constraint before FRM candidate scoring."""

    candidates = np.asarray(global_candidates, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[-1] != 4 or not np.isfinite(candidates).all():
        raise ValueError("global_candidates must have finite shape (candidates, horizon, 4)")
    headings = None
    if candidate_headings is not None:
        headings = np.asarray(candidate_headings, dtype=np.float64)
        if headings.shape != candidates.shape[:2] or not np.isfinite(headings).all():
            raise ValueError("candidate_headings must have finite shape (candidates, horizon)")
    mask = np.ones(candidates.shape[0], dtype=bool)
    for index, candidate in enumerate(candidates):
        velocity = candidate[0, 2:]
        target_heading = None if headings is None else headings[index, 0]
        raw = _velocity_action(robot, action_type, velocity, target_heading=target_heading)
        constrained = env.constrain_agent_action_exact(robot, raw)
        constrained_velocity = _action_velocity(robot, constrained)
        mask[index] = bool(np.allclose(velocity, constrained_velocity, atol=1e-6, rtol=0.0))
    return mask


def _planning_cost_config(args: argparse.Namespace) -> PlanningCostConfig:
    return PlanningCostConfig(
        task_weight=1.0,
        task_speed_scale=getattr(args, "task_speed_scale", 1.7),
        safety_weight=args.safety_weight,
        human_weight=args.human_cost_weight * getattr(args, "human_cost_scale", 1.0),
        comfort_scale=1.0,
        collision_margin=args.collision_margin,
        task_steps=args.execute_steps,
        violation_probability_alpha=(args.risk_alpha if args.hard_risk_filter else None),
        risk_samples=args.risk_samples,
        risk_weight=args.risk_weight,
        smoothness_weight=args.smoothness_weight,
        crowd_aggregation=getattr(args, "crowd_risk_aggregation", "mean"),
        temporal_risk_sampling=getattr(args, "temporal_risk_sampling", "independent"),
    )


def _goal_response_mask(candidates: np.ndarray, goal: np.ndarray, radius: float) -> np.ndarray:
    reached = np.linalg.norm(candidates[..., :2] - np.asarray(goal), axis=-1) < radius
    reached_before = np.concatenate([np.zeros((len(candidates), 1), dtype=bool),
                                     np.maximum.accumulate(reached, axis=1)[:, :-1]], axis=1)
    return ~reached_before


def _fallback_tiebreak(primary: torch.Tensor, planning_cost: torch.Tensor) -> int:
    best = primary.min()
    tied = torch.isfinite(primary) & torch.isclose(primary, best, atol=1e-7, rtol=0.)
    if not tied.any():
        raise ValueError("fallback tie-breaking requires at least one finite primary safety score")
    return int(torch.argmin(torch.where(tied, planning_cost, torch.full_like(planning_cost, float("inf")))).item())


def _observable_candidate_penalty(global_candidates, robot, humans, args, static_obstacles=None) -> np.ndarray:
    penalty = np.zeros(len(global_candidates), dtype=np.float64)
    terminal_weight = getattr(args, "terminal_goal_weight", 0.0)
    route_weight = getattr(args, "static_route_weight", 0.0)
    if route_weight > 0.0:
        if static_obstacles is None:
            raise ValueError("static-route scoring requires the observed obstacle map")
        route = static_route(static_obstacles, robot.get_goal_position(), robot.radius)
        improvement = route.distance_at(robot.get_position()) - route.distance_at(global_candidates[:, -1, :2])
        scale = max(global_candidates.shape[1] * args.env_dt * robot.v_pref, 1e-6)
        penalty -= route_weight * improvement / scale
        penalty += route_weight * (~route.free_positions(global_candidates[..., :2])).mean(axis=1)
    if terminal_weight > 0.0:
        position = np.asarray(robot.get_position())
        goal = np.asarray(robot.get_goal_position())
        progress = np.linalg.norm(goal - position) - np.linalg.norm(global_candidates[:, -1, :2] - goal[None], axis=-1)
        scale = max(global_candidates.shape[1] * args.env_dt * robot.v_pref, 1e-6)
        penalty -= terminal_weight * progress / scale
    clearance_weight = getattr(args, "kinematic_clearance_weight", 0.0)
    if clearance_weight > 0.0:
        penalty += clearance_weight * _kinematic_clearance_cost(
            global_candidates, robot, humans, args.env_dt, args.execute_steps,
        )
    predictive_weight = getattr(args, "predictive_clearance_weight", 0.0)
    if predictive_weight > 0.0 and humans:
        clearance = swept_surface_clearance(
            global_candidates, np.asarray(robot.get_position()),
            np.asarray([human.get_position() for human in humans]),
            np.asarray([human.get_velocity() for human in humans]),
            np.asarray([robot.radius + human.radius for human in humans]),
            args.env_dt, getattr(args, "predictive_clearance_steps", 4),
        )
        penalty += predictive_weight * np.exp(-np.clip(clearance, -0.6, 5.0) / 0.3).max(axis=1)
    return penalty


def _differentiable_response_predictions(
    model: torch.nn.Module,
    histories: np.ndarray,
    current: np.ndarray,
    candidates: np.ndarray,
    device: str,
    static_obstacles: list[Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    humans, candidate_count = candidates.shape[:2]
    horizon = min(int(candidates.shape[2]), int(model.config.horizon))
    candidates = candidates[:, :, :horizon]
    history = torch.from_numpy(histories).to(device=device, dtype=torch.float32)
    history = history[:, None].expand(humans, candidate_count, -1, -1).reshape(
        humans * candidate_count, 1, histories.shape[-2], 2
    )
    neighbor_history = torch.from_numpy(histories).to(
        device=device, dtype=torch.float32
    )
    neighbor_history = neighbor_history[None, None].expand(
        humans, candidate_count, humans, histories.shape[-2], 2
    ).reshape(humans * candidate_count, 1, humans, histories.shape[-2], 2)
    neighbor_mask = ~torch.eye(humans, device=device, dtype=torch.bool)
    neighbor_mask = neighbor_mask[:, None].expand(
        humans, candidate_count, humans
    ).reshape(humans * candidate_count, 1, humans)
    robot_current = torch.from_numpy(current).to(device=device, dtype=torch.float32)
    robot_current = robot_current[:, None].expand(humans, candidate_count, 4).reshape(
        humans * candidate_count, 1, 1, 4
    )
    robot_candidates = torch.from_numpy(candidates).to(device=device, dtype=torch.float32)
    robot_candidates = robot_candidates.reshape(humans * candidate_count, 1, 1, horizon, 4)
    mask = torch.ones(humans * candidate_count, 1, 1, device=device, dtype=torch.bool)
    context, reference_context = {}, {}
    if getattr(model.config, "static_geometry_features", False):
        if static_obstacles is None:
            raise ValueError("map-conditioned response model requires the observed static obstacle map")
        segments = torch.as_tensor(np.asarray(static_obstacles).reshape(-1, 4), device=device, dtype=history.dtype)
        origins = torch.as_tensor(histories[:, -1], device=device, dtype=history.dtype).repeat(1, 2)
        relative_segments = segments[None] - origins[:, None]
        reference_context["static_segments"] = relative_segments[:, None]
        context["static_segments"] = relative_segments[:, None].expand(-1, candidate_count, -1, -1).reshape(
            humans * candidate_count, 1, len(segments), 4)
    output = model(
        history,
        robot_current,
        robot_candidates,
        include_pair_interactions=False,
        robot_mask=mask,
        neighbor_history=neighbor_history,
        neighbor_mask=neighbor_mask,
        **context,
    )
    reference_current = robot_current.reshape(humans, candidate_count, 1, 1, 4)[:, 0]
    reference = stationary_reference_candidates(reference_current, horizon)
    reference_output = model(
        history.reshape(humans, candidate_count, 1, histories.shape[-2], 2)[:, 0],
        reference_current,
        reference,
        include_pair_interactions=False,
        robot_mask=mask.reshape(humans, candidate_count, 1, 1)[:, 0],
        neighbor_history=neighbor_history.reshape(humans, candidate_count, 1, humans, histories.shape[-2], 2)[:, 0],
        neighbor_mask=neighbor_mask.reshape(humans, candidate_count, 1, humans)[:, 0],
        **reference_context,
    )
    means = output["mean"][:, 0].reshape(humans, candidate_count, -1, horizon, 2)
    log_stds = output["log_std"][:, 0].reshape_as(means)
    references = reference_output["mean"][:, 0, None].expand_as(means)
    current = robot_current.reshape(humans, candidate_count, 1, 4)
    robot_candidates = robot_candidates.reshape(humans, candidate_count, 1, horizon, 4)
    return means, log_stds, references, current, robot_candidates


_response_predictions = torch.no_grad()(_differentiable_response_predictions)


@torch.no_grad()
def _score_candidates(
    model: torch.nn.Module,
    histories: np.ndarray,
    current: np.ndarray,
    candidates: np.ndarray,
    global_candidates: np.ndarray,
    robot: Any,
    humans_list: list[Any],
    static_obstacles: list[Any],
    beliefs: torch.Tensor,
    args: argparse.Namespace,
    fallback_candidate: int | None = None,
    executor_safe: np.ndarray | None = None,
    sage_entity_predictions: np.ndarray | None = None,
    candidate_bias: np.ndarray | None = None,
    diagnostics: dict[str, np.ndarray] | None = None,
    braking_safe: np.ndarray | None = None,
    cached_predictions: tuple | None = None,
    prediction_cache: dict | None = None,
    fallback_progress_active: bool = False,
) -> tuple:
    humans = candidates.shape[0]
    device = args.device
    prediction = cached_predictions if cached_predictions is not None else _response_predictions(
        model, histories, current, candidates, device, static_obstacles)
    means, log_stds, references, current, robot_candidates = prediction
    if prediction_cache is not None:
        prediction_cache["values"] = prediction
    planning = _planning_cost_config(args)
    goal_direction = np.asarray(robot.get_goal_position(), dtype=np.float32) - np.asarray(
        robot.get_position(), dtype=np.float32
    )
    goal_norm = max(float(np.linalg.norm(goal_direction)), 1e-6)
    goal_direction = goal_direction / goal_norm
    response_valid_steps = None
    if getattr(args, "goal_terminal_response", False):
        response_valid_steps = torch.from_numpy(_goal_response_mask(
            global_candidates[:, :means.shape[-2]], np.asarray(robot.get_goal_position()), float(robot.radius),
        )).to(device)
    costs = []
    for human in range(humans):
        costs.append(
            expected_candidate_costs(
                means[human],
                beliefs[human].exp(),
                current[human],
                robot_candidates[human],
                references[human],
                args.model_dt,
                planning,
                log_stds[human],
                task_direction=torch.from_numpy(goal_direction).to(device=device),
                response_valid_steps=response_valid_steps,
            )
        )
    total, violation = aggregate_entity_costs(
        costs, planning.task_weight, planning.safety_weight, planning.human_weight,
        planning.smoothness_weight, planning.risk_weight, planning.crowd_aggregation,
    )
    uniform_belief = torch.full(
        (beliefs.shape[-1],),
        1.0 / float(beliefs.shape[-1]),
        device=device,
        dtype=beliefs.dtype,
    )
    uniform_costs = []
    for human in range(humans):
        uniform_costs.append(
            expected_candidate_costs(
                means[human],
                uniform_belief,
                current[human],
                robot_candidates[human],
                references[human],
                args.model_dt,
                planning,
                log_stds[human],
                task_direction=torch.from_numpy(goal_direction).to(device=device),
                response_valid_steps=response_valid_steps,
            )
        )
    uniform_total, _ = aggregate_entity_costs(
        uniform_costs,
        planning.task_weight,
        planning.safety_weight,
        planning.human_weight,
        planning.smoothness_weight,
        planning.risk_weight,
        planning.crowd_aggregation,
    )
    belief_cost_leverage = float(getattr(args, "belief_cost_leverage", 1.0))
    adjustment = belief_cost_adjustment(total, uniform_total, belief_cost_leverage)
    total = total + adjustment
    core_total = total.clone()
    if diagnostics is not None:
        diagnostics["mode_mean_spread"] = means.std(dim=2, unbiased=False).mean(dim=(0, 2, 3)).cpu().numpy()
        diagnostics["action_response_delta_mse"] = (means - references).square().mean(dim=(0, 2, 3, 4)).cpu().numpy()
        for name in ("task", "human", "smoothness"):
            diagnostics[name] = torch.stack([item[name] for item in costs]).mean(dim=0).cpu().numpy()
        safety_values = torch.stack([item["safety"] for item in costs])
        diagnostics["safety"] = (safety_values.mean(dim=0) if planning.crowd_aggregation == "mean" else safety_values.amax(dim=0)).cpu().numpy()
        diagnostics["risk"] = violation.cpu().numpy()
        diagnostics["belief_cost_delta"] = (total - uniform_total).detach().cpu().numpy()
        diagnostics["belief_cost_leverage"] = np.full(len(total), belief_cost_leverage, dtype=np.float32)
        diagnostics["observable_penalty"] = _observable_candidate_penalty(global_candidates, robot, humans_list, args, static_obstacles)
    if candidate_bias is not None:
        bias = torch.as_tensor(candidate_bias, device=device, dtype=total.dtype)
        if bias.shape != total.shape:
            raise ValueError("candidate_bias must align with candidate count")
        total = total + bias
    else:
        bias = torch.zeros_like(total)
    learned_safe = torch.stack([item["safe_mask"] for item in costs], dim=0).all(dim=0)
    if args.hard_risk_filter:
        learned_safe = learned_safe & violation.le(args.risk_alpha)
    kinematic_safe = torch.as_tensor(
        _kinematic_safe_mask(
            global_candidates,
            robot,
            humans_list,
            args.env_dt,
            args.kinematic_safety_margin,
            getattr(args, "geometric_safety_prefix_steps", 0) or None,
        ),
        device=device,
        dtype=torch.bool,
    )
    static_safe = torch.as_tensor(
        _static_obstacle_safe_mask(
            global_candidates,
            robot,
            static_obstacles,
            args.static_safety_margin,
            getattr(args, "geometric_safety_prefix_steps", 0) or None,
        ),
        device=device,
        dtype=torch.bool,
    )
    dynamic_safe = torch.as_tensor(
        _sage_dynamic_safe_mask(
            global_candidates,
            sage_entity_predictions,
            robot,
            humans_list,
            args.kinematic_safety_margin,
            getattr(args, "geometric_safety_prefix_steps", 0) or None,
        ),
        device=device,
        dtype=torch.bool,
    )
    geometric_safe = kinematic_safe & static_safe
    if getattr(args, "swept_human_guard", False):
        swept_clearance = swept_surface_clearance(
            global_candidates, np.asarray(robot.get_position()),
            np.asarray([human.get_position() for human in humans_list]),
            np.asarray([human.get_velocity() for human in humans_list]),
            np.asarray([robot.radius + human.radius for human in humans_list]),
            args.env_dt, args.execute_steps,
        )
        geometric_safe = geometric_safe & torch.as_tensor(
            np.all(swept_clearance >= max(0.0, args.kinematic_safety_margin), axis=1),
            device=device,
        )
    observable_penalty = torch.as_tensor(
        _observable_candidate_penalty(global_candidates, robot, humans_list, args, static_obstacles),
        device=device, dtype=total.dtype,
    )
    total = total + observable_penalty
    executor_safe_tensor = torch.ones_like(geometric_safe)
    if executor_safe is not None:
        executor_safe_tensor = torch.as_tensor(
            executor_safe, device=device, dtype=torch.bool
        )
        if executor_safe_tensor.shape != geometric_safe.shape:
            raise ValueError("executor_safe must align with candidate count")
    admissible = torch.ones_like(executor_safe_tensor)
    if braking_safe is not None:
        admissible = torch.as_tensor(braking_safe, device=device, dtype=torch.bool)
        if admissible.shape != executor_safe_tensor.shape:
            raise ValueError("braking_safe must align with candidate count")
        admissible = admissible & executor_safe_tensor
        if not admissible.any():
            raise ValueError("static braking invariant lost: no executable candidate preserves a safe stop")
        if diagnostics is not None:
            diagnostics["braking_safe"] = admissible.cpu().numpy().astype(np.float32)
    executor_safe_tensor = executor_safe_tensor & admissible
    executor_geometric_safe = geometric_safe & dynamic_safe & executor_safe_tensor
    safe = learned_safe & executor_geometric_safe
    scores = torch.where(safe, total, torch.full_like(total, float("inf")))
    surface_clearance = torch.as_tensor(
        _kinematic_surface_clearance(
            global_candidates,
            robot,
            humans_list,
            args.env_dt,
            getattr(args, "geometric_safety_prefix_steps", 0) or None,
        ),
        device=device,
        dtype=total.dtype,
    )
    fallback = not bool(torch.isfinite(scores).any())
    decision_scores = scores
    fallback_tier = "none"
    if (
        fallback
        and args.fallback_profile == "orca-proposal"
        and fallback_candidate is not None
        and 0 <= fallback_candidate < len(geometric_safe)
        and bool(executor_geometric_safe[fallback_candidate])
    ):
        selected = int(fallback_candidate)
        decision_scores = torch.full_like(total, float("inf"))
        decision_scores[selected] = violation[selected]
        fallback_tier = "orca-proposal"
    elif fallback:
        # A fully safe branch is preferred, but when learned risk rejects the
        # set (or a long-horizon guard is too strict), do not immediately pick
        # an arbitrary high-risk action.  Keep the executor-accepted branch
        # and then the least geometrically risky branch as explicit fallback
        # tiers.  This makes the fallback behavior measurable and prevents a
        # static-wall projection from being reintroduced by argmin(violation).
        fallback_mask = executor_geometric_safe
        fallback_tier = "executor-geometric"
        if not bool(fallback_mask.any()):
            fallback_mask = executor_safe_tensor
            fallback_tier = "executor"
        if not bool(fallback_mask.any()):
            fallback_mask = kinematic_safe & admissible
            fallback_tier = "kinematic"
        if not bool(fallback_mask.any()):
            fallback_mask = static_safe & admissible
            fallback_tier = "static"
        if bool(fallback_mask.any()):
            fallback_value = violation
            if fallback_tier == "executor":
                # Every executor-accepted branch violates the observed body
                # tube. Choose maximum predicted surface clearance first;
                # learned violation is only a deterministic tie breaker.
                fallback_value = -surface_clearance + 1e-3 * violation
                if fallback_progress_active and getattr(args, "fallback_progress_weight", 0.0) > 0.0:
                    goal = torch.as_tensor(
                        np.asarray(robot.get_goal_position()), device=device, dtype=total.dtype
                    )
                    position = torch.as_tensor(
                        np.asarray(robot.get_position()), device=device, dtype=total.dtype
                    )
                    progress = (
                        torch.linalg.vector_norm(goal - position)
                        - torch.linalg.vector_norm(
                            torch.as_tensor(global_candidates[:, -1, :2], device=device, dtype=total.dtype)
                            - goal,
                            dim=-1,
                        )
                    )
                    scale = max(
                        global_candidates.shape[1] * args.env_dt * float(robot.v_pref), 1e-6
                    )
                    fallback_value = (
                        fallback_value
                        - float(args.fallback_progress_weight) * progress / scale
                    )
            fallback_scores = torch.where(
                fallback_mask,
                fallback_value,
                torch.full_like(violation, float("inf")),
            )
            selected = int(torch.argmin(fallback_scores).item())
            if getattr(args, "fallback_task_tiebreak", False):
                selected = _fallback_tiebreak(fallback_scores, total)
            decision_scores = fallback_scores
        else:
            decision_scores = torch.where(admissible, violation, torch.full_like(violation, float("inf")))
            selected = int(torch.argmin(decision_scores).item())
            fallback_tier = "violation-only"
    else:
        selected = int(torch.argmin(violation if fallback else scores).item())
    if diagnostics is not None:
        task = torch.stack([item["task"] for item in costs]).mean(dim=0)
        human = torch.stack([item["human"] for item in costs]).mean(dim=0)
        smoothness = torch.stack([item["smoothness"] for item in costs]).mean(dim=0)
        safety_values = torch.stack([item["safety"] for item in costs])
        safety = (
            safety_values.mean(dim=0)
            if planning.crowd_aggregation == "mean"
            else safety_values.amax(dim=0)
        )
        diagnostics.update({
            "weighted_task": (planning.task_weight * task).detach().cpu().numpy(),
            "weighted_safety": (planning.safety_weight * safety).detach().cpu().numpy(),
            "weighted_human": (planning.human_weight * human).detach().cpu().numpy(),
            "weighted_smoothness": (planning.smoothness_weight * smoothness).detach().cpu().numpy(),
            "weighted_risk": (planning.risk_weight * violation).detach().cpu().numpy(),
            "weighted_belief_adjustment": adjustment.detach().cpu().numpy(),
            "observable_penalty": observable_penalty.detach().cpu().numpy(),
            "candidate_bias": bias.detach().cpu().numpy(),
            "core_total": core_total.detach().cpu().numpy(),
            "unconstrained_total": total.detach().cpu().numpy(),
        })
    return (
        selected,
        means,
        log_stds,
        references,
        torch.tensor(float(fallback), device=device),
        total.detach().cpu(),
        decision_scores.detach().cpu(),
        safe.detach().cpu(),
        learned_safe.detach().cpu(),
        geometric_safe.detach().cpu(),
        executor_safe_tensor.detach().cpu(),
        kinematic_safe.detach().cpu(),
        static_safe.detach().cpu(),
        dynamic_safe.detach().cpu(),
        fallback_tier,
    )


def _rank_correlation(predicted: np.ndarray, realized: np.ndarray) -> tuple[float, float]:
    """Return tie-aware enough Spearman/Kendall diagnostics without scipy."""

    predicted = np.asarray(predicted, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64)
    if predicted.shape != realized.shape or predicted.size < 2:
        return float("nan"), float("nan")
    valid = np.isfinite(predicted) & np.isfinite(realized)
    predicted = predicted[valid]
    realized = realized[valid]
    if predicted.size < 2:
        return float("nan"), float("nan")
    def rankdata(values: np.ndarray) -> np.ndarray:
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

    pred_order = rankdata(predicted)
    real_order = rankdata(realized)
    pred_centered = pred_order - pred_order.mean()
    real_centered = real_order - real_order.mean()
    denominator = float(np.linalg.norm(pred_centered) * np.linalg.norm(real_centered))
    spearman = float(np.dot(pred_centered, real_centered) / denominator) if denominator > 0 else 0.0
    concordant = discordant = 0
    for first in range(predicted.size):
        for second in range(first + 1, predicted.size):
            product = (predicted[first] - predicted[second]) * (realized[first] - realized[second])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    pairs = concordant + discordant
    kendall = float((concordant - discordant) / pairs) if pairs else 0.0
    return spearman, kendall


def _ranking_top_k_hit(predicted: np.ndarray, realized: np.ndarray, k: int) -> float:
    """Whether the predicted top-k set contains the realized best branch."""

    predicted = np.asarray(predicted, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64)
    if predicted.shape != realized.shape or predicted.size == 0:
        return float("nan")
    finite = np.isfinite(predicted) & np.isfinite(realized)
    if not bool(finite.any()):
        return float("nan")
    indices = np.flatnonzero(finite)
    width = max(1, min(int(k), len(indices)))
    predicted_order = indices[np.argsort(predicted[indices], kind="mergesort")[:width]]
    oracle = int(indices[np.argmin(realized[indices])])
    return float(oracle in set(int(index) for index in predicted_order))


def _ranking_oracle_index(predicted: np.ndarray, realized: np.ndarray) -> int | None:
    """Return the realized best candidate inside the planner's decision support."""

    predicted = np.asarray(predicted, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64)
    if predicted.shape != realized.shape or predicted.size == 0:
        return None
    valid = np.isfinite(predicted) & np.isfinite(realized)
    if not bool(valid.any()):
        return None
    indices = np.flatnonzero(valid)
    return int(indices[np.argmin(realized[indices])])


def _counterfactual_candidate_costs(
    external_root: Path,
    case: int,
    env_config_path: Path,
    human_num: int | None,
    action_history: list[np.ndarray],
    global_candidates: np.ndarray,
    audit_horizon: int,
    dt: float,
    planning: PlanningCostConfig,
    candidate_penalty: np.ndarray | None = None,
    motion_profile: str = "holonomic",
) -> np.ndarray:
    """Evaluate candidates under the same horizon and terms as FRM planning.

    The controller still executes only ``planning.task_steps`` before
    replanning.  For a decision-level ranking audit, however, the external
    simulator must reveal the complete response horizon that FRM used to rank
    the branches.  A stationary-robot rollout supplies the realized reference
    response used by the human-impact term.
    """

    if motion_profile == "sicnav-unicycle":
        raise ValueError(
            "unicycle ranking audits require heading-aware action histories; "
            "use the matched live response audit instead"
        )

    candidates = np.asarray(global_candidates, dtype=np.float32)
    horizon = min(int(audit_horizon), int(candidates.shape[1]))
    if horizon < 1:
        raise ValueError("audit_horizon must be positive")
    task_steps = int(planning.task_steps) if planning.task_steps is not None else horizon
    if not 1 <= task_steps <= horizon:
        raise ValueError("ranking audit task_steps must lie within audit_horizon")

    def rollout(velocities: np.ndarray) -> dict[str, np.ndarray]:
        env, robot, _, _, ActionXY, _, _ = _new_episode(
            external_root, case, env_config_path, human_num, motion_profile
        )
        done = False
        for branch in action_history:
            for velocity in branch:
                _, _, done, info = env.step(_velocity_action(robot, ActionXY, velocity))
                if done:
                    break
            if done:
                break
        if done:
            raise RuntimeError("counterfactual replay terminated before the audited window")
        robot_start = np.asarray(robot.get_position(), dtype=np.float64)
        robot_velocity = np.asarray(robot.get_velocity(), dtype=np.float64)
        goal = np.asarray(robot.get_goal_position(), dtype=np.float64)
        robot_positions: list[np.ndarray] = []
        executed_velocities: list[np.ndarray] = []
        human_positions: list[np.ndarray] = []
        human_velocities: list[np.ndarray] = []
        for velocity in np.asarray(velocities, dtype=np.float32)[:horizon]:
            raw = _velocity_action(robot, ActionXY, velocity)
            action = env.constrain_agent_action_exact(robot, raw)
            constrained = _action_velocity(robot, action).astype(np.float64)
            _, _, done, _ = env.step(action)
            robot_positions.append(np.asarray(robot.get_position(), dtype=np.float64))
            executed_velocities.append(constrained)
            human_positions.append(
                np.asarray([human.get_position() for human in env.humans], dtype=np.float64)
            )
            human_velocities.append(
                np.asarray([human.get_velocity() for human in env.humans], dtype=np.float64)
            )
            if done:
                break
        while len(robot_positions) < horizon:
            robot_positions.append(robot_positions[-1].copy())
            executed_velocities.append(np.zeros(2, dtype=np.float64))
            human_positions.append(human_positions[-1].copy())
            human_velocities.append(np.zeros_like(human_velocities[-1]))
        return {
            "robot_start": robot_start,
            "robot_velocity": robot_velocity,
            "goal": goal,
            "robot_positions": np.stack(robot_positions),
            "robot_velocities": np.stack(executed_velocities),
            "human_positions": np.stack(human_positions),
            "human_velocities": np.stack(human_velocities),
        }

    reference = rollout(np.zeros((horizon, 2), dtype=np.float32))
    values: list[float] = []
    for candidate in candidates:
        realized = rollout(candidate[:horizon, 2:])
        distances = np.linalg.norm(
            realized["robot_positions"][:, None, :] - realized["human_positions"],
            axis=-1,
        )
        per_human_minimum = distances.min(axis=0)
        per_human_safety = np.exp(-per_human_minimum / float(planning.comfort_scale))
        safety = float(per_human_safety.mean() if planning.crowd_aggregation == "mean" else per_human_safety.max())
        human = float(
            np.square(
                realized["human_velocities"] - reference["human_velocities"]
            ).sum(axis=-1).mean()
        )
        direction = realized["goal"] - realized["robot_start"]
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        displacement = (
            realized["robot_positions"][task_steps - 1] - realized["robot_start"]
        )
        progress = float(np.dot(displacement, direction))
        task = -progress / max(task_steps * float(dt) * planning.task_speed_scale, 1e-6)
        per_human_violation = per_human_minimum < float(planning.collision_margin)
        violation = float(per_human_violation.mean() if planning.crowd_aggregation == "mean" else per_human_violation.any())
        smoothness = float(
            np.linalg.norm(
                realized["robot_velocities"][0] - realized["robot_velocity"]
            )
        )
        values.append(
            float(
                planning.task_weight * task
                + planning.safety_weight * safety
                + planning.human_weight * human
                + planning.risk_weight * violation
                + planning.smoothness_weight * smoothness
            )
        )
    costs = np.asarray(values, dtype=np.float64)
    if candidate_penalty is not None:
        penalty = np.asarray(candidate_penalty, dtype=np.float64)
        if penalty.shape != costs.shape or not np.isfinite(penalty).all():
            raise ValueError("candidate_penalty must be finite and align with candidates")
        costs = costs + penalty
    return costs


def _run_episode(
    model: torch.nn.Module,
    sage_adapter: SAGECandidateAdapter | None,
    policy: str,
    external_root: Path,
    case: int,
    env_config_path: Path,
    args: argparse.Namespace,
    seed: int,
    ranking_audit_rows: list[dict[str, Any]] | None = None,
    window_audit_rows: list[dict[str, Any]] | None = None,
    decision_audit: Any | None = None,
    controller_state: Any | None = None,
    initial_action: Any | None = None,
    executable_value_model: dict | None = None,
) -> dict[str, Any]:
    from sage.experiments.external_value_policy import value_policy_digest, value_policy_stages
    from sage.experiments.navigation_exposure import NavigationExposure

    executable_value_sha256 = value_policy_digest(executable_value_model)
    if initial_action is not None and controller_state is None:
        raise ValueError("offline first-action intervention requires a captured controller state")
    if controller_state is None:
        environment = _new_episode(
            external_root, case, env_config_path, args.human_num, getattr(args, "motion_profile", "holonomic")
        )
    else:
        controller_state.validate(case, seed, policy, executable_value_sha256)
        environment = copy.deepcopy(controller_state.environment)
    env, robot, _, static_obs, ActionXY, orca, JointState = environment
    motion_limits = (
        UnicycleLimits.from_sicnav_config(Path(args.motion_policy_config))
        if getattr(args, "motion_profile", "holonomic") == "sicnav-unicycle" else None
    )
    humans = list(env.humans)
    initial_contract = initial_navigation_state(env, robot, static_obs)
    mode_count = int(model.config.num_modes)
    beliefs = torch.stack([uniform_log_belief(mode_count, device=args.device) for _ in humans])
    adaptive_states: list[AdaptivePersistentBelief] | None = None
    if policy == "tracer-adaptive-persistent":
        adaptive_states = [
            AdaptivePersistentBelief(
                mode_count,
                config=AdaptivePersistentBeliefConfig(
                    base_hazard=float(getattr(args, "adaptive_base_hazard", 0.0)),
                    max_hazard=float(args.adaptive_max_hazard if getattr(args, "adaptive_max_hazard", None) is not None else args.hazard_rate),
                    surprise_threshold=float(args.adaptive_surprise_threshold if getattr(args, "adaptive_surprise_threshold", None) is not None else args.change_point_threshold),
                    surprise_scale=float(getattr(args, "adaptive_surprise_scale", 1.0)),
                    likelihood_scale=float(args.likelihood_scale),
                    hard_reset=bool(getattr(args, "adaptive_hard_reset", False)),
                    threshold_hysteresis=float(getattr(args, "adaptive_threshold_hysteresis", 0.0)),
                    cooldown_steps=int(getattr(args, "adaptive_cooldown_steps", 0)),
                ),
                device=args.device,
            )
            for _ in humans
        ]
    history_positions = [[] for _ in humans]
    rng = np.random.default_rng(args.candidate_seed + seed * 1009 + case * 9176)
    initial = np.asarray(robot.get_position(), dtype=np.float64)
    goal = np.asarray(robot.get_goal_position(), dtype=np.float64)
    initial_distance = max(float(np.linalg.norm(goal - initial)), 1e-6)
    min_distance = float("inf")
    collisions = 0
    wall_collisions = 0
    exposure = NavigationExposure()
    fallback_count = 0
    fallback_streak = 0
    fallback_tier_counts = {
        tier: 0
        for tier in (
            "orca-proposal",
            "executor-geometric",
            "executor",
            "kinematic",
            "static",
            "violation-only",
        )
    }
    change_points = 0
    belief_update_windows = 0
    belief_selection_changes = 0
    belief_action_changes = 0
    belief_score_l1: list[float] = []
    belief_selected_regret: list[float] = []
    planner_sensitivity_values: dict[str, list[float]] = {}
    belief_prediction_nll_gains: list[float] = []
    belief_prediction_mse_gains: list[float] = []
    action_history: list[np.ndarray] = []
    response_audits = []
    executor_projection_events = 0
    ranking_spearman: list[float] = []
    ranking_kendall: list[float] = []
    ranking_oracle_hit: list[float] = []
    ranking_oracle_top3: list[float] = []
    ranking_regret: list[float] = []
    learned_safe_fractions: list[float] = []
    geometric_safe_fractions: list[float] = []
    executor_safe_fractions: list[float] = []
    executor_geometric_safe_fractions: list[float] = []
    kinematic_safe_fractions: list[float] = []
    static_safe_fractions: list[float] = []
    dynamic_safe_fractions: list[float] = []
    learned_rejection_fractions: list[float] = []
    task_progress_fractions: list[float] = []
    windows = 0
    steps = 0
    started = time.perf_counter()
    final_info: dict[str, Any] = {}
    previous_plan = None
    value_action_changes = 0
    value_action_change_details = []
    value_stage_action_changes = [0] * len(value_policy_stages(executable_value_model))
    value_stage_action_change_details = [[] for _ in value_stage_action_changes]
    if controller_state is not None:
        history_positions = copy.deepcopy(controller_state.histories)
        beliefs = controller_state.beliefs.detach().clone().to(args.device)
        adaptive_states = copy.deepcopy(controller_state.adaptive_states)
        rng.bit_generator.state = copy.deepcopy(controller_state.candidate_rng)
        previous_plan = copy.deepcopy(controller_state.previous_plan)
        fallback_streak = controller_state.fallback_streak
        steps, windows = controller_state.steps, controller_state.windows
        value_action_changes = controller_state.value_action_changes
        value_action_change_details = copy.deepcopy(controller_state.value_action_change_details or [])
        restored_stage_changes = controller_state.value_stage_action_changes
        if restored_stage_changes is None:
            if len(value_stage_action_changes) != 1:
                raise ValueError("continuation is missing chained value-policy intervention state")
            restored_stage_changes = [value_action_changes]
        if len(restored_stage_changes) != len(value_stage_action_changes):
            raise ValueError("continuation value-policy stages differ from the captured controller")
        value_stage_action_changes = copy.deepcopy(restored_stage_changes)
        restored_stage_details = controller_state.value_stage_action_change_details
        if restored_stage_details is None:
            if len(value_stage_action_change_details) != 1:
                raise ValueError("continuation is missing chained value-policy action details")
            restored_stage_details = [value_action_change_details]
        if len(restored_stage_details) != len(value_stage_action_change_details):
            raise ValueError("continuation value-policy action details differ from the captured controller")
        value_stage_action_change_details = copy.deepcopy(restored_stage_details)
        if controller_state.outcome_state is not None:
            outcomes = controller_state.outcome_state
            initial_contract = copy.deepcopy(outcomes["initial_contract"])
            initial_distance = outcomes["initial_distance"]
            collisions, wall_collisions = outcomes["collisions"], outcomes["wall_collisions"]
            min_distance = outcomes["min_distance"]
            exposure = NavigationExposure(**outcomes["exposure"])
    while steps < args.max_steps and windows < args.windows:
        if decision_audit is not None and hasattr(decision_audit, "capture_controller_state"):
            decision_audit.capture_controller_state(
                environment=environment, histories=history_positions, beliefs=beliefs,
                adaptive_states=adaptive_states, rng=rng, previous_plan=previous_plan,
                fallback_streak=fallback_streak, steps=steps, windows=windows,
                case=case, seed=seed, policy=policy,
                executable_value_sha256=executable_value_sha256,
                value_action_changes=value_action_changes,
                value_action_change_details=value_action_change_details,
                value_stage_action_changes=value_stage_action_changes,
                value_stage_action_change_details=value_stage_action_change_details,
                outcome_state={"initial_contract": initial_contract, "initial_distance": initial_distance,
                               "collisions": collisions, "wall_collisions": wall_collisions,
                               "min_distance": min_distance, "exposure": exposure.snapshot()},
            )
        for index, human in enumerate(humans):
            position = np.asarray(human.get_position(), dtype=np.float32)
            if not history_positions[index] or not np.array_equal(history_positions[index][-1], position):
                history_positions[index].append(position)
        current_history = _histories(history_positions, humans, model.config.history, args.env_dt)
        proposal_velocity = None
        sage_entity_predictions: np.ndarray | None = None
        if args.candidate_profile == "orca-augmented":
            state = JointState(
                robot.get_full_state(),
                [human.get_observable_state() for human in humans],
                static_obs,
            )
            proposal = orca.predict(state)
            proposal_velocity = np.asarray([proposal.vx, proposal.vy], dtype=np.float32)
        recovery_start: int | None = None
        route_start: int | None = None
        incumbent_start: int | None = None
        recovery_added = 0
        if args.candidate_profile in {"sage", "sage-hybrid", "sage-adaptive", "sage-recovery"}:
            if sage_adapter is None:
                raise RuntimeError("candidate-profile=sage requires a SAGE adapter")
            if args.candidate_profile == "sage-hybrid":
                current, candidates, global_candidates, sage_entity_predictions = _sage_hybrid_candidate_pool(
                    sage_adapter,
                    history_positions,
                    robot,
                    humans,
                    args.num_candidates,
                    model.config.horizon,
                    args.env_dt,
                    args.candidate_seed + seed * 1009 + case * 9176 + windows,
                    rng,
                    args.sage_history_mode,
                )
            else:
                current, candidates, global_candidates, sage_entity_predictions = _sage_candidate_pool(
                    sage_adapter,
                    history_positions,
                    robot,
                    humans,
                    args.num_candidates,
                    args.env_dt,
                    args.candidate_seed + seed * 1009 + case * 9176 + windows,
                    args.sage_history_mode,
                )
            if args.candidate_profile == "sage-adaptive":
                current, candidates, global_candidates, repaired_count = _repair_sage_candidates(
                    current,
                    candidates,
                    global_candidates,
                    env,
                    robot,
                    humans,
                    static_obs,
                    ActionXY,
                    args.env_dt,
                    args.kinematic_safety_margin,
                    args.static_safety_margin,
                    rng,
                )
                # Repaired branches are procedural anchors, so the original
                # joint SAGE entity futures no longer align candidate-wise.
                sage_entity_predictions = None
        else:
            current, candidates, global_candidates = _candidate_pool(
                robot,
                humans,
                args.num_candidates,
                model.config.horizon,
                args.env_dt,
                rng,
                proposal_velocity,
            )
        if args.candidate_profile in {"sage", "sage-hybrid", "sage-adaptive", "sage-recovery"}:
            global_candidates = _project_sage_speed_limit(
                global_candidates,
                robot,
                args.env_dt,
                args.sage_speed_limit_scale,
            )
            candidates = _relative_candidates(global_candidates, humans)
            if sage_entity_predictions is not None and args.sage_speed_limit_scale > 0.0:
                # The robot path changed under speed projection, so the joint
                # SAGE entity predictions are no longer paired with it.
                sage_entity_predictions = None
        if args.candidate_profile == "sage-recovery":
            recovery_start = int(global_candidates.shape[0])
            current, candidates, global_candidates, recovery_added = (
                _augment_with_safe_recovery_candidates(
                    current,
                    candidates,
                    global_candidates,
                    env,
                    robot,
                    humans,
                    static_obs,
                    ActionXY,
                    args.env_dt,
                    args.kinematic_safety_margin,
                    args.static_safety_margin,
                    args.recovery_safety_margin,
                    args.recovery_candidates,
                    args.recovery_directions,
                    args.recovery_activation,
                    args.recovery_activation_distance,
                )
            )
            if recovery_added:
                # Recovery paths do not have candidate-aligned SAGE entity
                # futures, so the optional joint generator check is disabled.
                sage_entity_predictions = None
        if getattr(args, "goal_return_candidates", False) and recovery_start is not None and recovery_added:
            returned = goal_return_trajectories(
                global_candidates[recovery_start:], np.asarray(robot.get_position()),
                np.asarray(robot.get_goal_position()), args.env_dt, robot.v_pref,
            )
            global_candidates = np.concatenate([global_candidates, returned], axis=0)
            candidates = _relative_candidates(global_candidates, humans)
            sage_entity_predictions = None
        if getattr(args, "static_route_candidates", False):
            route = static_route(static_obs, robot.get_goal_position(), robot.radius)
            route_start = len(global_candidates)
            routed = route.candidates(robot.get_position(), global_candidates.shape[1], args.env_dt,
                                      [.4 * robot.v_pref, .7 * robot.v_pref, .95 * robot.v_pref])
            global_candidates = np.concatenate([global_candidates, routed], axis=0)
            candidates = _relative_candidates(global_candidates, humans)
            sage_entity_predictions = None
        if getattr(args, "include_incumbent", False) and previous_plan is not None:
            remaining = previous_plan[args.execute_steps:, 2:]
            fill = np.repeat(previous_plan[-1:, 2:], args.execute_steps, axis=0)
            velocities = np.concatenate([remaining, fill], axis=0)
            positions = np.asarray(robot.get_position())[None] + np.cumsum(velocities * args.env_dt, axis=0)
            incumbent = np.concatenate([positions, velocities], axis=-1).astype(np.float32)
            incumbent_start = len(global_candidates)
            global_candidates = np.concatenate([global_candidates, incumbent[None]])
            candidates = _relative_candidates(global_candidates, humans)
            sage_entity_predictions = None
        motion_projected_candidates = 0
        braking_start = None
        if getattr(args, "static_braking_guard", False):
            braking_start = len(global_candidates)
            global_candidates = np.concatenate([global_candidates, np.zeros_like(global_candidates[:1])])
        candidate_headings = None
        aligned_route_start = None
        if motion_limits is not None:
            projected, candidate_headings = project_unicycle_candidates(
                global_candidates, np.asarray(robot.get_position()), np.asarray(robot.get_velocity()),
                robot.theta, args.env_dt, motion_limits, return_headings=True,
            )
            motion_projected_candidates = int(np.any(np.abs(projected - global_candidates) > 1e-6, axis=(1, 2)).sum())
            global_candidates = projected
            if route_start is not None and getattr(args, "feedback_route_candidates", False):
                routed, routed_headings = route.unicycle_candidates(
                    robot.get_position(), robot.get_velocity(), robot.theta, global_candidates.shape[1],
                    args.env_dt, [.4 * robot.v_pref, .7 * robot.v_pref, .95 * robot.v_pref], motion_limits,
                )
                global_candidates[route_start:route_start + 3] = routed
                candidate_headings[route_start:route_start + 3] = routed_headings
                if getattr(args, "align_route_before_motion", False):
                    aligned_route_start = len(global_candidates)
                    aligned, aligned_headings = route.unicycle_candidates(
                        robot.get_position(), robot.get_velocity(), robot.theta, global_candidates.shape[1],
                        args.env_dt, [.4 * robot.v_pref, .7 * robot.v_pref, .95 * robot.v_pref], motion_limits,
                        align_before_motion=True,
                    )
                    global_candidates = np.concatenate([global_candidates, aligned])
                    candidate_headings = np.concatenate([candidate_headings, aligned_headings])
            candidates = _relative_candidates(global_candidates, humans)
            sage_entity_predictions = None
        braking_safe = None
        prefix_variant_start = None
        if getattr(args, "prefix_diversity", False):
            if motion_limits is None:
                raise ValueError("prefix diversity requires unicycle motion limits")
            variants, variant_headings = diversify_unicycle_prefixes(
                global_candidates, robot.get_position(), robot.get_velocity(), robot.theta,
                args.env_dt, motion_limits, args.execute_steps,
            )
            prefix_variant_start = len(global_candidates)
            global_candidates = np.concatenate([global_candidates, variants])
            candidate_headings = np.concatenate([candidate_headings, variant_headings])
            candidates = _relative_candidates(global_candidates, humans)
            sage_entity_predictions = None
        if getattr(args, "static_braking_guard", False):
            stopping_paths = braking_after_prefix(global_candidates, args.env_dt, motion_limits.deceleration, args.execute_steps)
            braking_safe = _static_obstacle_safe_mask(stopping_paths, robot, static_obs, max(args.static_safety_margin, .001))
        _validate_executable_trajectory_contract(
            global_candidates,
            np.asarray(robot.get_position(), dtype=np.float32),
            args.env_dt,
            args.execute_steps,
        )
        candidate_bias = np.zeros(global_candidates.shape[0], dtype=np.float32)
        score_diagnostics: dict[str, np.ndarray] = {}
        prediction_cache = {}
        if recovery_start is not None and recovery_added:
            candidate_bias[recovery_start:] = float(args.recovery_cost_penalty)
        selected, means, log_stds, references, fallback, predicted_costs, decision_scores, safe_mask, learned_safe_mask, geometric_safe_mask, executor_safe_mask, kinematic_safe_mask, static_safe_mask, dynamic_safe_mask, fallback_tier = _score_candidates(
            model,
            current_history,
            current,
            candidates,
            global_candidates,
            robot,
            humans,
            static_obs,
            beliefs,
            args,
            fallback_candidate=1 if args.candidate_profile == "orca-augmented" else None,
            executor_safe=_executor_safe_mask(
                env, robot, ActionXY, global_candidates, candidate_headings=candidate_headings
            ),
            sage_entity_predictions=(
                sage_entity_predictions if args.sage_dynamic_safety_filter else None
            ),
            candidate_bias=candidate_bias,
            diagnostics=score_diagnostics,
            braking_safe=braking_safe,
            prediction_cache=prediction_cache,
            fallback_progress_active=(
                fallback_streak >= getattr(args, "fallback_progress_after", 0)
                and getattr(args, "fallback_progress_weight", 0.0) > 0.0
            ),
        )
        fallback_progress_active = bool(
            fallback_streak >= getattr(args, "fallback_progress_after", 0)
            and getattr(args, "fallback_progress_weight", 0.0) > 0.0
        )
        if executable_value_model is not None and not bool(fallback):
            from sage.experiments.external_value_policy import select_scheduled_value_policy
            from sage.response.executable_value import executable_value_features
            features, feature_names = executable_value_features(
                candidates=global_candidates, headings=candidate_headings,
                robot_position=np.asarray(robot.get_position()), robot_velocity=np.asarray(robot.get_velocity()),
                robot_goal=np.asarray(robot.get_goal_position()), robot_heading=robot.theta,
                robot_radius=robot.radius, human_positions=np.asarray([human.get_position() for human in humans]),
                human_velocities=np.asarray([human.get_velocity() for human in humans]),
                human_radii=np.asarray([human.radius for human in humans]), beliefs=beliefs.exp().cpu().numpy(),
                means=prediction_cache["values"][0].cpu().numpy(),
                log_stds=prediction_cache["values"][1].cpu().numpy(), diagnostics=score_diagnostics,
                dt=args.model_dt, execute_steps=args.execute_steps,
            )
            replacement, value_stage_action_changes, stage_trace = select_scheduled_value_policy(
                executable_value_model, features, feature_names, global_candidates, candidate_headings,
                decision_scores.numpy(), selected, args.execute_steps, score_diagnostics,
                windows, value_stage_action_changes, admissible_mask=safe_mask.numpy().astype(bool),
            )
            if not bool(safe_mask[replacement]):
                raise RuntimeError("executable value policy selected outside the planner safety support")
            refinement = stage_trace[-1]
            for stage in stage_trace:
                if stage["changed"]:
                    value_stage_action_change_details[stage["stage"]].append(
                        (int(windows), stage["before"], stage["after"])
                    )
            if refinement["changed"]:
                value_action_change_details.append(
                    (int(windows), refinement["before"], refinement["after"])
                )
            value_action_changes += int(refinement["changed"])
            selected = replacement
        if initial_action is not None and windows == controller_state.windows:
            selected = initial_action.validate(global_candidates, candidate_headings, decision_scores.numpy())
        belief_audit = {}
        if (
            getattr(args, "belief_selection_audit", False)
            or getattr(args, "planner_sensitivity_audit", False)
        ):
            uniform = torch.full_like(beliefs, -math.log(mode_count))
            uniform_score_diagnostics: dict[str, np.ndarray] = {}
            counterfactual = _score_candidates(
                model, current_history, current, candidates, global_candidates, robot, humans, static_obs,
                uniform, args, fallback_candidate=1 if args.candidate_profile == "orca-augmented" else None,
                executor_safe=_executor_safe_mask(
                    env, robot, ActionXY, global_candidates, candidate_headings=candidate_headings
                ),
                sage_entity_predictions=sage_entity_predictions if args.sage_dynamic_safety_filter else None,
                candidate_bias=candidate_bias, braking_safe=braking_safe,
                cached_predictions=prediction_cache["values"],
                diagnostics=uniform_score_diagnostics,
                fallback_progress_active=fallback_progress_active,
            )
            uniform_selected = counterfactual[0]
            action_changed = bool(
                not np.allclose(
                    global_candidates[uniform_selected, :args.execute_steps],
                    global_candidates[selected, :args.execute_steps],
                    atol=1e-6,
                )
                or (
                    candidate_headings is not None
                    and not np.allclose(
                        candidate_headings[uniform_selected, :args.execute_steps],
                        candidate_headings[selected, :args.execute_steps],
                        atol=1e-6,
                    )
                )
            )
            current_scores = decision_scores.numpy()
            uniform_scores = counterfactual[6].numpy()
            finite_scores = np.isfinite(current_scores) & np.isfinite(uniform_scores)
            score_l1 = (
                float(np.mean(np.abs(current_scores[finite_scores] - uniform_scores[finite_scores])))
                if np.any(finite_scores)
                else float("nan")
            )
            selected_regret = (
                float(uniform_scores[selected] - uniform_scores[uniform_selected])
                if np.isfinite(uniform_scores[selected]) and np.isfinite(uniform_scores[uniform_selected])
                else float("nan")
            )
            selected_advantage = (
                float(current_scores[uniform_selected] - current_scores[selected])
                if np.isfinite(current_scores[uniform_selected]) and np.isfinite(current_scores[selected])
                else float("nan")
            )
            selection_changed = int(uniform_selected != selected)
            belief_selection_changes += selection_changed
            belief_action_changes += int(action_changed)
            if math.isfinite(score_l1):
                belief_score_l1.append(score_l1)
            if math.isfinite(selected_regret):
                belief_selected_regret.append(selected_regret)
            belief_planner = candidate_score_decomposition(
                score_diagnostics,
                predicted_costs,
                decision_scores,
                safe_mask,
                selected,
                bool(fallback.item()),
            )
            uniform_planner = candidate_score_decomposition(
                uniform_score_diagnostics,
                counterfactual[5],
                counterfactual[6],
                counterfactual[7],
                uniform_selected,
                bool(counterfactual[4].item()),
            )
            planner_sensitivity = belief_sensitivity_diagnostics(
                belief_planner, uniform_planner, selected, uniform_selected
            )
            if getattr(args, "belief_vertex_audit", False) and not bool(counterfactual[4].item()):
                planning = _planning_cost_config(args)
                if planning.crowd_aggregation != "mean" or args.hard_risk_filter:
                    raise ValueError("belief vertex audit requires mean aggregation without belief-dependent hard risk masks")
                goal_direction = np.asarray(robot.get_goal_position()) - robot.get_position()
                goal_direction = goal_direction / max(np.linalg.norm(goal_direction), 1e-9)
                valid_steps = None
                if getattr(args, "goal_terminal_response", False):
                    valid_steps = torch.from_numpy(_goal_response_mask(
                        global_candidates[:, :means.shape[-2]], np.asarray(robot.get_goal_position()), float(robot.radius)
                    )).to(args.device)
                per_human_mode_costs = []
                for human_index in range(len(humans)):
                    candidate_current = torch.as_tensor(current[human_index], device=args.device).float().expand(len(global_candidates), 1, 4)
                    candidate_paths = torch.as_tensor(candidates[human_index], device=args.device).float()[:, None]
                    mode_costs = []
                    for mode_index in range(mode_count):
                        mode_costs.append(expected_candidate_costs(
                            means[human_index, :, mode_index:mode_index + 1], torch.ones(1, device=args.device),
                            candidate_current, candidate_paths, references[human_index, :, mode_index:mode_index + 1],
                            args.model_dt, planning, log_stds[human_index, :, mode_index:mode_index + 1],
                            task_direction=torch.as_tensor(goal_direction, device=args.device), response_valid_steps=valid_steps,
                        )["unconstrained_total"])
                    per_human_mode_costs.append(torch.stack(mode_costs))
                planner_sensitivity.update(mode_cost_selection_envelope(
                    float(getattr(args, "belief_cost_leverage", 1.0)) * torch.stack(per_human_mode_costs).cpu().numpy(), uniform_scores,
                    uniform_selected, counterfactual[7].numpy(),
                ))
            for name, value in planner_sensitivity.items():
                if math.isfinite(value):
                    planner_sensitivity_values.setdefault(name, []).append(value)
            belief_audit = {
                "same_state_uniform_selected": int(uniform_selected),
                "same_state_belief_selection_changed": selection_changed,
                "same_state_belief_action_changed": int(action_changed),
                "same_state_belief_score_l1": score_l1,
                "same_state_belief_selected_regret": selected_regret,
                "same_state_belief_selected_advantage": selected_advantage,
                "same_state_belief_uniform_score_penalty": selected_regret,
                **planner_sensitivity,
            }
        if decision_audit is not None:
            decision_audit.observe(
                env=env, model=model, histories=current_history, current=current,
                candidates=candidates, global_candidates=global_candidates,
                headings=candidate_headings, static_obstacles=static_obs,
                beliefs=beliefs, args=args, selected=selected,
                decision_scores=decision_scores, diagnostics=score_diagnostics,
                cached_predictions=prediction_cache["values"], candidate_bias=candidate_bias,
                braking_safe=braking_safe, fallback_progress_active=fallback_progress_active,
                sage_entity_predictions=sage_entity_predictions if args.sage_dynamic_safety_filter else None,
                case=case, window=windows, seed=seed, policy=policy,
            )
        if getattr(args, "candidate_debug_windows", 0) > windows:
            debug_path = Path(args.output_dir) / "candidate_debug" / f"seed{seed}_case{case}_{policy}_window{windows}.npz"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            if debug_path.exists():
                raise ValueError("candidate debug snapshot already exists")
            np.savez_compressed(debug_path, candidates=global_candidates,
                                headings=candidate_headings if candidate_headings is not None else np.empty((0,)),
                                robot_position=np.asarray(robot.get_position()), robot_velocity=np.asarray(robot.get_velocity()),
                                robot_heading=np.asarray(robot.theta), goal=np.asarray(robot.get_goal_position()),
                                human_positions=np.asarray([human.get_position() for human in humans]),
                                human_velocities=np.asarray([human.get_velocity() for human in humans]),
                                human_radii=np.asarray([human.radius for human in humans]), robot_radius=np.asarray(robot.radius),
                                static_obstacles=np.asarray(static_obs), selected=np.asarray(selected),
                                decision_scores=decision_scores.numpy(), predicted_costs=predicted_costs.numpy(),
                                response_means=means.detach().cpu().numpy(), response_log_stds=log_stds.detach().cpu().numpy(),
                                observable_beliefs=beliefs.exp().cpu().numpy(),
                                **{f"diagnostic_{name}": np.asarray(values) for name, values in score_diagnostics.items()},
                                geometric_safe=geometric_safe_mask.numpy(), kinematic_safe=kinematic_safe_mask.numpy(),
                                static_safe=static_safe_mask.numpy(), executor_safe=executor_safe_mask.numpy(),
                                braking_safe=braking_safe if braking_safe is not None else np.ones(len(global_candidates), dtype=bool))
        if windows < args.ranking_audit_windows:
            realized_costs = _counterfactual_candidate_costs(
                external_root,
                case,
                env_config_path,
                args.human_num,
                action_history,
                global_candidates,
                model.config.horizon,
                args.env_dt,
                _planning_cost_config(args),
                candidate_penalty=candidate_bias + _observable_candidate_penalty(
                    global_candidates, robot, humans, args, static_obs,
                ),
                motion_profile=getattr(args, "motion_profile", "holonomic"),
            )
            spearman, kendall = _rank_correlation(
                decision_scores.numpy(), realized_costs
            )
            oracle = _ranking_oracle_index(decision_scores.numpy(), realized_costs)
            ranking_spearman.append(spearman)
            ranking_kendall.append(kendall)
            ranking_oracle_hit.append(float(selected == oracle) if oracle is not None else float("nan"))
            ranking_oracle_top3.append(
                _ranking_top_k_hit(decision_scores.numpy(), realized_costs, 3)
            )
            ranking_regret.append(
                float(realized_costs[selected] - realized_costs[oracle])
                if oracle is not None
                else float("nan")
            )
            if ranking_audit_rows is not None:
                decision_array = decision_scores.numpy()
                for candidate_index, (decision, predicted, realized) in enumerate(
                    zip(decision_array, predicted_costs.numpy(), realized_costs)
                ):
                    ranking_audit_rows.append(
                        {
                            "seed": int(seed),
                            "case": int(case),
                            "policy": policy,
                            "window": int(windows),
                            "candidate": int(candidate_index),
                            "predicted_cost": float(decision),
                            "predicted_total_cost": float(predicted),
                            "realized_cost": float(realized),
                            "selected": int(candidate_index == selected),
                            "oracle": int(oracle is not None and candidate_index == oracle),
                            "learned_safe": int(learned_safe_mask[candidate_index]),
                            "geometric_safe": int(geometric_safe_mask[candidate_index]),
                            "executor_safe": int(executor_safe_mask[candidate_index]),
                            "kinematic_safe": int(kinematic_safe_mask[candidate_index]),
                            "static_safe": int(static_safe_mask[candidate_index]),
                            "dynamic_safe": int(dynamic_safe_mask[candidate_index]),
                            **{f"predicted_{name}": float(values[candidate_index]) for name, values in score_diagnostics.items()},
                        }
                    )
        fallback_count += int(fallback.item())
        fallback_streak = fallback_streak + 1 if bool(fallback.item()) else 0
        audit_stride = int(getattr(args, "response_audit_stride", 0))
        if audit_stride and windows % audit_stride == 0:
            audit_horizon = int(model.config.horizon)
            response_audits.append(collect_live_responses(
                env, global_candidates[:, :audit_horizon],
                None if candidate_headings is None else candidate_headings[:, :audit_horizon],
                current_history, selected, args.response_audit_candidates,
                args.candidate_seed + seed * 1009 + case * 9176 + windows,
                case, windows, ActionXY, _velocity_action,
            ))
        if fallback_tier != "none":
            fallback_tier_counts[fallback_tier] += 1
        learned_safe_fractions.append(float(learned_safe_mask.float().mean().item()))
        geometric_safe_fractions.append(float(geometric_safe_mask.float().mean().item()))
        executor_safe_fractions.append(float(executor_safe_mask.float().mean().item()))
        executor_geometric_safe_fractions.append(
            float((executor_safe_mask & geometric_safe_mask).float().mean().item())
        )
        kinematic_safe_fractions.append(float(kinematic_safe_mask.float().mean().item()))
        static_safe_fractions.append(float(static_safe_mask.float().mean().item()))
        dynamic_safe_fractions.append(float(dynamic_safe_mask.float().mean().item()))
        learned_rejection_fractions.append(
            float((geometric_safe_mask & ~learned_safe_mask).float().mean().item())
        )
        goal = np.asarray(robot.get_goal_position(), dtype=np.float32)
        robot_position = np.asarray(robot.get_position(), dtype=np.float32)
        direction = goal - robot_position
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        prefix = max(1, min(int(args.execute_steps), global_candidates.shape[1]))
        displacement = global_candidates[:, prefix - 1, :2] - robot_position[None]
        task_progress_fractions.append(float(((displacement * direction[None]).sum(axis=-1) > 0.0).mean()))
        selected_progress = float(np.dot(displacement[selected], direction))
        pre_min_distance = min(
            float(
                np.linalg.norm(
                    robot_position - np.asarray(human.get_position(), dtype=np.float32)
                )
            )
            for human in humans
        )
        realized_velocities = []
        previous_plan = global_candidates[selected].copy()
        executed_action_path: list[np.ndarray] = []
        window_collisions = 0
        window_wall_collisions = 0
        window_projection_events = 0
        window_min_distance = float("inf")
        for step in range(args.execute_steps):
            velocity = global_candidates[selected, min(step, global_candidates.shape[1] - 1), 2:]
            target_heading = None if candidate_headings is None else candidate_headings[selected, step]
            raw_action = _velocity_action(robot, ActionXY, velocity, target_heading)
            if motion_limits is not None:
                speed_change = abs(raw_action.v) - np.linalg.norm(robot.get_velocity())
                if (abs(raw_action.r) > motion_limits.max_yaw_rate * args.env_dt + 1e-5
                    or abs(raw_action.v) > motion_limits.max_speed + 1e-5
                    or speed_change > motion_limits.acceleration * args.env_dt + 1e-5
                    or speed_change < -motion_limits.deceleration * args.env_dt - 1e-5):
                    raise ValueError("selected unicycle action violates configured motion limits")
            constrained_action = env.constrain_agent_action_exact(robot, raw_action)
            position_before_action = np.asarray(robot.get_position())
            human_velocities_before_action = np.asarray([human.get_velocity() for human in humans]).reshape(-1, 2)
            action = constrained_action
            _, _, done, info = env.step(action)
            exposure.observe(
                np.linalg.norm(goal - position_before_action), np.linalg.norm(goal - robot.get_position()),
                human_velocities_before_action,
                np.asarray([human.get_velocity() for human in humans]).reshape(-1, 2), args.env_dt,
            )
            constrained_velocity = np.asarray(robot.get_velocity(), dtype=np.float32)
            if not np.allclose(robot.get_position(), position_before_action + constrained_velocity * args.env_dt, atol=2e-5):
                raise ValueError("executed robot position differs from its observed velocity integral")
            projected = int(
                not np.allclose(velocity, constrained_velocity, atol=1e-6, rtol=0.0)
            )
            if getattr(args, "static_braking_guard", False) and projected:
                raise ValueError("braking-safe selected control was changed by the static executor")
            executor_projection_events += projected
            window_projection_events += projected
            executed_action_path.append(constrained_velocity)
            final_info = info
            steps += 1
            collision_event = int(_info_value(info, "Collision") != 0.0)
            wall_collision_event = int(_info_value(info, "WallCollision") != 0.0)
            collisions += collision_event
            wall_collisions += wall_collision_event
            window_collisions += collision_event
            window_wall_collisions += wall_collision_event
            robot_position = np.asarray(robot.get_position(), dtype=np.float64)
            for human in humans:
                distance = float(
                    np.linalg.norm(robot_position - np.asarray(human.get_position()))
                )
                min_distance = min(min_distance, distance)
                window_min_distance = min(window_min_distance, distance)
            realized_velocities.append(
                [np.asarray(human.get_velocity(), dtype=np.float32) for human in humans]
            )
            for index, human in enumerate(humans):
                history_positions[index].append(np.asarray(human.get_position(), dtype=np.float32))
            if done or steps >= args.max_steps:
                break
        action_history.append(
            np.asarray(executed_action_path, dtype=np.float32)
        )
        if decision_audit is not None:
            decision_audit.verify_execution(env, len(executed_action_path))
        if window_audit_rows is not None:
            if prefix_variant_start is not None and selected >= prefix_variant_start:
                selected_origin = "prefix-variant"
            elif aligned_route_start is not None and aligned_route_start <= selected < aligned_route_start + 3:
                selected_origin = "aligned-route"
            elif braking_start is not None and selected == braking_start:
                selected_origin = "braking"
            elif incumbent_start is not None and selected == incumbent_start:
                selected_origin = "incumbent"
            elif route_start is not None and route_start <= selected < route_start + 3:
                selected_origin = "static-route"
            elif recovery_start is not None and selected >= recovery_start:
                selected_origin = "recovery"
            elif args.candidate_profile == "sage-hybrid":
                sage_count = max(
                    1,
                    args.num_candidates - min(4, args.num_candidates - 1),
                )
                selected_origin = "sage" if selected < sage_count else "anchor"
            elif args.candidate_profile in {"sage", "sage-adaptive", "sage-recovery"}:
                selected_origin = "sage"
            elif args.candidate_profile == "orca-augmented" and selected == 1:
                selected_origin = "orca"
            else:
                selected_origin = "procedural"
            window_audit_rows.append(
                {
                    "seed": int(seed),
                    "case": int(case),
                    "policy": policy,
                    "window": int(windows),
                    "motion_projected_candidates": motion_projected_candidates,
                    **belief_audit,
                    "total_candidate_count": len(global_candidates),
                    "static_route_candidates_added": 3 if route_start is not None else 0,
                    "incumbent_candidates_added": int(incumbent_start is not None),
                    "braking_candidates_added": int(braking_start is not None),
                    "aligned_route_candidates_added": 3 if aligned_route_start is not None else 0,
                    "braking_safe_fraction": float(np.mean(braking_safe)) if braking_safe is not None else float("nan"),
                    "robot_x": float(robot.get_position()[0]),
                    "robot_y": float(robot.get_position()[1]),
                    "robot_heading": float(getattr(robot, "theta", 0.)),
                    "candidate_profile": args.candidate_profile,
                    "selected_candidate": int(selected),
                    "selected_origin": selected_origin,
                    "recovery_candidates_added": int(recovery_added),
                    "selected_candidate_bias": float(candidate_bias[selected]),
                    "fallback": int(bool(fallback.item())),
                    "fallback_tier": fallback_tier,
                    "fallback_progress_active": int(fallback_progress_active),
                    "predicted_total_cost": float(predicted_costs[selected]),
                    "decision_score": float(decision_scores[selected]),
                    "selected_progress": selected_progress,
                    "selected_speed": float(
                        np.linalg.norm(global_candidates[selected, 0, 2:])
                    ),
                    "selected_learned_safe": int(learned_safe_mask[selected]),
                    "selected_geometric_safe": int(geometric_safe_mask[selected]),
                    "selected_executor_safe": int(executor_safe_mask[selected]),
                    "selected_kinematic_safe": int(kinematic_safe_mask[selected]),
                    "selected_static_safe": int(static_safe_mask[selected]),
                    "learned_safe_fraction": float(learned_safe_mask.float().mean()),
                    "geometric_safe_fraction": float(geometric_safe_mask.float().mean()),
                    "executor_safe_fraction": float(executor_safe_mask.float().mean()),
                    "pre_min_distance": pre_min_distance,
                    "post_min_distance": window_min_distance,
                    "executor_projection": int(window_projection_events > 0),
                    "collision": int(window_collisions > 0),
                    "wall_collision": int(window_wall_collisions > 0),
                    **{f"selected_predicted_{name}": float(values[selected]) for name, values in score_diagnostics.items()},
                }
            )
        realized = torch.from_numpy(np.asarray(realized_velocities, dtype=np.float32)).to(args.device)
        if audit_stride and windows % audit_stride == 0:
            selected_rows = response_audits[-1]["selected_branch"]
            target_prefix = response_audits[-1]["target"][selected_rows, :realized.shape[0]]
            if not np.allclose(target_prefix, realized.cpu().numpy().transpose(1, 0, 2), atol=1e-6):
                raise ValueError("cloned selected-prefix response differs from factual execution")
        executed_means = means[:, selected]
        executed_log_stds = log_stds[:, selected]
        if window_projection_events:
            executed_candidates = candidates[:, selected:selected + 1].copy()
            prefix = len(executed_action_path)
            executed_candidates[:, :, :prefix, 2:] = np.asarray(executed_action_path)[None, None]
            executed_candidates[..., :2] = current[:, None, None, :2] + np.cumsum(
                executed_candidates[..., 2:] * args.env_dt, axis=2,
            )
            corrected_means, corrected_log_stds, _, _, _ = _response_predictions(
                model, current_history, current, executed_candidates, args.device, static_obs,
            )
            executed_means = corrected_means[:, 0]
            executed_log_stds = corrected_log_stds[:, 0]
        factual_target = realized.permute(1, 0, 2)
        factual_mean = executed_means[:, :, :realized.shape[0]]
        factual_log_std = executed_log_stds[:, :, :realized.shape[0]]
        factual_likelihood = trajectory_mode_log_likelihood(factual_mean, factual_log_std, factual_target)
        mixture_mean = (beliefs.exp()[..., None, None] * factual_mean).sum(dim=1)
        factual_nll = float(-torch.logsumexp(beliefs + factual_likelihood, dim=-1).mean() / (2 * realized.shape[0]))
        uniform_nll = float(-torch.logsumexp(factual_likelihood - math.log(mode_count), dim=-1).mean() / (2 * realized.shape[0]))
        factual_mse = float((mixture_mean - factual_target).square().mean())
        uniform_mse = float((factual_mean.mean(dim=1) - factual_target).square().mean())
        belief_prediction_nll_gains.append(uniform_nll - factual_nll)
        belief_prediction_mse_gains.append(uniform_mse - factual_mse)
        if window_audit_rows is not None:
            window_audit_rows[-1].update({
                "executed_prefix_mixture_nll": factual_nll,
                "executed_prefix_velocity_mse": factual_mse,
                "same_state_uniform_prefix_nll": uniform_nll,
                "same_state_uniform_prefix_velocity_mse": uniform_mse,
                "belief_prediction_nll_gain": uniform_nll - factual_nll,
                "belief_prediction_mse_gain": uniform_mse - factual_mse,
                "belief_updated_this_window": int(_belief_observation_due(windows, args.execute_steps, getattr(args, "belief_update_stride", args.execute_steps))),
            })
            if getattr(args, "belief_evidence_audit", False):
                window_audit_rows[-1].update({
                    "factual_mode_log_likelihood": factual_likelihood.cpu().tolist(),
                    "decision_log_beliefs": beliefs.cpu().tolist(),
                    "factual_response_steps": int(realized.shape[0]),
                })
        if _belief_observation_due(
            windows,
            args.execute_steps,
            getattr(args, "belief_update_stride", args.execute_steps),
        ):
            for human_index in range(len(humans)):
                prediction = executed_means[human_index, :, : realized.shape[0]]
                log_std = executed_log_stds[human_index, :, : realized.shape[0]]
                likelihood = trajectory_mode_log_likelihood(
                    prediction[None], log_std[None], realized[:, human_index][None]
                )[0]
                beliefs[human_index], changed = _update_belief(
                    beliefs[human_index],
                    likelihood,
                    policy,
                    args,
                    adaptive_state=(
                        None
                        if adaptive_states is None
                        else adaptive_states[human_index]
                    ),
                )
                change_points += int(changed)
            belief_update_windows += 1
        if window_audit_rows is not None and getattr(args, "belief_evidence_audit", False):
            window_audit_rows[-1]["posterior_log_beliefs"] = beliefs.cpu().tolist()
        windows += 1
        if final_info and (_info_value(final_info, "ReachGoal") != 0.0 or _info_value(final_info, "Timeout") != 0.0):
            break
    reached = float(_info_value(final_info, "ReachGoal") != 0.0)
    timeout = float(_info_value(final_info, "Timeout") != 0.0)
    position = np.asarray(robot.get_position(), dtype=np.float64)
    progress = (initial_distance - float(np.linalg.norm(goal - position))) / initial_distance
    posterior = beliefs.exp()
    save_response_audit(args.output_dir, seed, case, policy, response_audits)
    return {
        "seed": seed,
        "case": int(case),
        "human_num": int(len(humans)),
        "candidate_profile": args.candidate_profile,
        "policy": policy,
        "steps": int(steps),
        "windows": int(windows),
        **navigation_outcome(reached, timeout, steps, min(args.max_steps, args.windows * args.execute_steps), collisions, wall_collisions),
        **initial_contract,
        "collision": float(collisions > 0),
        "collision_events": int(collisions),
        "wall_collision": float(wall_collisions > 0),
        "wall_collision_events": int(wall_collisions),
        "task_progress": float(progress),
        "total_time": float(env.global_time),
        "terminal_goal_distance": float(np.linalg.norm(goal - position)),
        **exposure.snapshot(),
        "min_distance": float(min_distance) if math.isfinite(min_distance) else float("nan"),
        "fallback_rate": float(fallback_count / max(windows, 1)),
        "fallback_orca_proposal_rate": float(
            fallback_tier_counts["orca-proposal"] / max(windows, 1)
        ),
        "fallback_executor_geometric_rate": float(
            fallback_tier_counts["executor-geometric"] / max(windows, 1)
        ),
        "fallback_executor_rate": float(
            fallback_tier_counts["executor"] / max(windows, 1)
        ),
        "fallback_kinematic_rate": float(
            fallback_tier_counts["kinematic"] / max(windows, 1)
        ),
        "fallback_static_rate": float(
            fallback_tier_counts["static"] / max(windows, 1)
        ),
        "fallback_violation_only_rate": float(
            fallback_tier_counts["violation-only"] / max(windows, 1)
        ),
        "ranking_audit_windows": int(len(ranking_spearman)),
        "ranking_spearman": _finite_mean(ranking_spearman),
        "ranking_kendall": _finite_mean(ranking_kendall),
        "ranking_oracle_hit_rate": _finite_mean(ranking_oracle_hit),
        "ranking_oracle_top3_hit_rate": _finite_mean(ranking_oracle_top3),
        "ranking_oracle_regret": _finite_mean(ranking_regret),
        "learned_safe_fraction": float(np.mean(learned_safe_fractions)) if learned_safe_fractions else float("nan"),
        "geometric_safe_fraction": float(np.mean(geometric_safe_fractions)) if geometric_safe_fractions else float("nan"),
        "executor_safe_fraction": float(np.mean(executor_safe_fractions)) if executor_safe_fractions else float("nan"),
        "executor_geometric_safe_fraction": float(np.mean(executor_geometric_safe_fractions)) if executor_geometric_safe_fractions else float("nan"),
        "executor_projection_events": int(executor_projection_events),
        "kinematic_safe_fraction": float(np.mean(kinematic_safe_fractions)) if kinematic_safe_fractions else float("nan"),
        "static_safe_fraction": float(np.mean(static_safe_fractions)) if static_safe_fractions else float("nan"),
        "dynamic_safe_fraction": float(np.mean(dynamic_safe_fractions)) if dynamic_safe_fractions else float("nan"),
        "learned_rejection_fraction": float(np.mean(learned_rejection_fractions)) if learned_rejection_fractions else float("nan"),
        "task_positive_progress_fraction": float(np.mean(task_progress_fractions)) if task_progress_fractions else float("nan"),
        "change_point_events": int(change_points),
        "belief_update_windows": int(belief_update_windows),
        "belief_selection_audit_windows": int(
            len(belief_score_l1)
            if (
                getattr(args, "belief_selection_audit", False)
                or getattr(args, "planner_sensitivity_audit", False)
            )
            else 0
        ),
        "planner_sensitivity_audit_windows": int(
            len(planner_sensitivity_values.get("belief_total_score_l1", []))
        ),
        "belief_selection_change_rate": float(
            belief_selection_changes / max(windows, 1)
        ),
        "belief_action_change_rate": float(
            belief_action_changes / max(windows, 1)
        ),
        "belief_score_l1": _finite_mean(belief_score_l1),
        "belief_selected_regret": _finite_mean(belief_selected_regret),
        "belief_prediction_nll_gain": _finite_mean(belief_prediction_nll_gains),
        "belief_prediction_mse_gain": _finite_mean(belief_prediction_mse_gains),
        **{
            name: _finite_mean(planner_sensitivity_values.get(name, []))
            for name in BELIEF_SENSITIVITY_METRICS
        },
        "final_belief_entropy": float((-(posterior * posterior.clamp_min(1e-9).log()).sum(dim=-1)).mean().item()),
        "wall_time_seconds": float(time.perf_counter() - started),
        "static_obstacle_count": int(len(static_obs)),
        **({"executable_value_action_changes": value_action_changes,
            "executable_value_action_change_rate": value_action_changes / max(windows, 1),
            "executable_value_action_change_details": json.dumps(value_action_change_details),
            "executable_value_stage_action_changes": json.dumps(value_stage_action_changes),
            "executable_value_stage_action_change_details": json.dumps(value_stage_action_change_details)}
           if executable_value_model is not None else {}),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    git = _git_state(root)
    if git["dirty"] and not getattr(args, "allow_dirty", False):
        raise RuntimeError("official CrowdSimPlus TRACER experiment requires a clean main-repository worktree")
    if not math.isclose(float(args.model_dt), float(args.env_dt), rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(
            "model_dt and env_dt must match: candidate integration, FRM history velocities, and environment execution share one clock"
        )
    if getattr(args, "motion_profile", "holonomic") == "sicnav-unicycle" and args.ranking_audit_windows:
        raise ValueError("unicycle ranking audit requires a matched braking reference; use --ranking-audit-windows 0 for navigation validation")
    if getattr(args, "goal_terminal_response", False) and args.ranking_audit_windows:
        raise ValueError("terminal response scoring requires a terminal-matched ranking audit; use --ranking-audit-windows 0")
    if getattr(args, "static_braking_guard", False) and getattr(args, "motion_profile", "holonomic") != "sicnav-unicycle":
        raise ValueError("static braking guard requires the SICNav unicycle motion profile")
    if getattr(args, "align_route_before_motion", False) and not (
        getattr(args, "motion_profile", "holonomic") == "sicnav-unicycle"
        and getattr(args, "static_route_candidates", False) and getattr(args, "feedback_route_candidates", False)
    ):
        raise ValueError("aligned route support requires feedback static-route candidates and SICNav unicycle dynamics")
    if getattr(args, "response_audit_stride", 0) < 0 or getattr(args, "response_audit_candidates", 6) < 1:
        raise ValueError("response audit stride must be nonnegative and candidate budget positive")
    if getattr(args, "fallback_progress_weight", 0.0) < 0.0 or getattr(args, "fallback_progress_after", 0) < 0:
        raise ValueError("fallback progress weight and activation threshold must be nonnegative")
    if getattr(args, "fallback_task_tiebreak", False) and args.ranking_audit_windows:
        raise ValueError("lexicographic fallback requires a tie-aware ranking audit; use --ranking-audit-windows 0")
    if (
        not math.isfinite(float(getattr(args, "human_cost_scale", 1.0)))
        or float(getattr(args, "human_cost_scale", 1.0)) <= 0.0
    ):
        raise ValueError("human-cost-scale must be finite and positive")
    if (
        not math.isfinite(float(getattr(args, "belief_cost_leverage", 1.0)))
        or float(getattr(args, "belief_cost_leverage", 1.0)) <= 0.0
    ):
        raise ValueError("belief-cost-leverage must be finite and positive")
    calibration_pattern = getattr(args, "belief_calibration_path", None)
    if calibration_pattern and args.response_backend != "crowdsimplus-persistent":
        raise ValueError("external belief calibration requires crowdsimplus-persistent")
    if calibration_pattern and getattr(args, "allow_belief_cadence_override", False):
        raise ValueError("calibrated cadence cannot be combined with diagnostic override")
    if getattr(args, "belief_update_stride", None) is not None:
        if not getattr(args, "allow_belief_cadence_override", False) and not calibration_pattern:
            raise ValueError(
                "--belief-update-stride requires --allow-belief-cadence-override"
            )
        if args.belief_update_stride < 1 or args.belief_update_stride % args.execute_steps:
            raise ValueError(
                "belief-update-stride must be a positive multiple of execute-steps"
            )
    guard_steps = int(getattr(args, "geometric_safety_prefix_steps", 0))
    if 0 < guard_steps < args.execute_steps:
        raise ValueError("geometric safety guard must cover every executed step")
    external_root = (root / args.external_root).resolve()
    env_config_path = (root / args.env_config).resolve()
    checkpoint_root = (root / args.checkpoint_dir).resolve()
    output = (root / args.output_dir).resolve()
    if (output / "episode_metrics.csv").exists():
        raise ValueError("TRACER navigation output already contains results; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {}
    for source in sorted(source_root.rglob("*.py")):
        relative = source.relative_to(source_root)
        snapshot = output / "source_snapshot" / "sage" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        snapshot.write_bytes(content)
        source_hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    (output / "source_hashes.json").write_text(
        json.dumps(source_hashes, indent=2) + "\n", encoding="utf-8"
    )
    sage_adapter = None
    if args.candidate_profile in {"sage", "sage-hybrid", "sage-adaptive", "sage-recovery"}:
        sage_adapter = SAGECandidateAdapter.from_checkpoint(
            (root / args.sage_checkpoint).resolve(),
            args.device,
            SAGECandidateAdapterConfig(
                num_candidates=args.num_candidates,
                guided=True,
                step_stride=args.sage_step_stride,
                goal_speed=args.sage_goal_speed,
                oversample_factor=args.sage_oversample_factor,
                task_prefilter=True,
                task_prefix_steps=args.execute_steps,
                goal_direction="task_goal",
                include_task_anchor=True,
                task_anchor_speed_scales=(0.45,),
                batch_size=args.num_candidates,
                guidance_step_clip_norm=0.1,
                diversity_weight=getattr(args, "sage_diversity_weight", 0.0),
            ),
        )
    records: list[dict[str, Any]] = []
    ranking_audit_rows: list[dict[str, Any]] = []
    window_audit_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    response_checkpoints = {}
    for seed in args.seeds:
        seed_args = argparse.Namespace(**vars(args))
        checkpoint_path = (root / args.external_response_checkpoint_dir / f"joint_seed{seed}.pt"
                           if args.response_backend == "crowdsimplus-paired" else
                           checkpoint_root / f"seed{seed}" / "response.pt")
        checkpoint_sha256 = _sha256(checkpoint_path)
        response_checkpoints[str(seed)] = {"path": str(checkpoint_path), "sha256": checkpoint_sha256,
                                           "likelihood_scale": seed_args.likelihood_scale}
        if args.response_backend == "crowdsimplus-paired":
            model = ExternalResponseCheckpointAdapter.from_checkpoint(
                checkpoint_path,
                args.device,
                expected_dt=args.env_dt,
            )
        else:
            model = _load_model(
                checkpoint_path,
                args.device,
                expected_dt=args.env_dt,
            )
        if args.response_backend == "crowdsimplus-persistent":
            checkpoint_path = checkpoint_root / f"seed{seed}" / "response.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            calibration = checkpoint.get("belief_calibration", {})
            calibration_path = None
            if calibration_pattern:
                from sage.experiments.calibrate_persistent_belief import load_belief_calibration

                calibration_path = Path(str(calibration_pattern).format(seed=seed))
                calibration = load_belief_calibration(
                    calibration_path, checkpoint_path, execute_steps=args.execute_steps,
                    dt=args.env_dt, num_robots=model.config.num_robots,
                )
                _validate_external_belief_cases(
                    {"provenance": {"case_splits": calibration["case_splits"]}},
                    args.case_start, args.cases,
                )
                if calibration.get("num_modes") != model.config.num_modes:
                    raise ValueError("external calibration mode count mismatch")
            if calibration.get("phase") != "validation" or not checkpoint.get("mode_semantics"):
                raise ValueError("external persistent checkpoints require validation-only calibration and mode semantics")
            if model.config.response_delay_steps != 1 or model.config.evidence_conditioning:
                raise ValueError("external persistent checkpoints require a causal first step without untracked evidence inputs")
            if args.execute_steps != 1:
                raise ValueError("external belief calibration covers one executed response step")
            _validate_external_belief_cases(checkpoint, args.case_start, args.cases)
            calibrated_stride = int(calibration["observation_stride"])
            requested_stride = getattr(args, "belief_update_stride", None)
            if calibration_path is not None:
                if requested_stride is not None and requested_stride != calibrated_stride:
                    raise ValueError("requested cadence differs from validation calibration")
                seed_args.belief_update_stride = calibrated_stride
                seed_args.belief_cadence_source = "validation_artifact"
            elif requested_stride is not None:
                if not getattr(args, "allow_belief_cadence_override", False):
                    raise ValueError(
                        "--belief-update-stride requires --allow-belief-cadence-override"
                    )
                if requested_stride < 1 or requested_stride % args.execute_steps:
                    raise ValueError(
                        "belief-update-stride must be a positive multiple of execute-steps"
                    )
                seed_args.belief_update_stride = int(requested_stride)
                seed_args.belief_cadence_source = "explicit_override"
            else:
                seed_args.belief_update_stride = calibrated_stride
                seed_args.belief_cadence_source = "checkpoint_calibration"
            _belief_observation_due(0, args.execute_steps, seed_args.belief_update_stride)
            seed_args.likelihood_scale = float(calibration["selected"]["likelihood_scale"])
            seed_args.adaptive_max_hazard = float(
                args.adaptive_max_hazard
                if args.adaptive_max_hazard is not None
                else args.hazard_rate
            )
            seed_args.adaptive_surprise_threshold = float(
                args.adaptive_surprise_threshold
                if args.adaptive_surprise_threshold is not None
                else args.change_point_threshold
            )
            if not math.isfinite(seed_args.likelihood_scale) or seed_args.likelihood_scale <= 0:
                raise ValueError("checkpoint likelihood scale must be finite and positive")
            model.mode_semantics = checkpoint["mode_semantics"]
            response_checkpoints[str(seed)] = {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path),
                                               "likelihood_scale": seed_args.likelihood_scale,
                                               "belief_update_stride": seed_args.belief_update_stride,
                                               "calibration_phase": "validation",
                                               "belief_cadence_source": seed_args.belief_cadence_source,
                                               "calibration_path": str(calibration_path) if calibration_path else None,
                                               "calibration_sha256": _sha256(calibration_path) if calibration_path else None,
                                               "cadence_matches_calibration": seed_args.belief_update_stride == calibrated_stride}
        if _sha256(checkpoint_path) != checkpoint_sha256:
            raise ValueError("response checkpoint changed while it was being loaded")
        validate_response_clock(model, args.env_dt, getattr(args, "allow_temporal_transfer", False))
        for case in range(args.case_start, args.case_start + args.cases):
            for policy in args.policies:
                print(f"sicnav_tracer seed={seed} case={case} policy={policy}", flush=True)
                records.append(
                    _run_episode(
                        model,
                        sage_adapter,
                        policy,
                        external_root,
                        case,
                        env_config_path,
                        seed_args,
                        seed,
                        ranking_audit_rows,
                        window_audit_rows,
                    )
                )
    fields = list(records[0])
    with (output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    ranking_audit_path = output / "ranking_audit.csv"
    if ranking_audit_rows:
        ranking_fields = list(ranking_audit_rows[0])
        with ranking_audit_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ranking_fields)
            writer.writeheader()
            writer.writerows(ranking_audit_rows)
    window_audit_path = output / "window_metrics.csv"
    if window_audit_rows:
        window_fields = list(window_audit_rows[0])
        with window_audit_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=window_fields)
            writer.writeheader()
            writer.writerows(window_audit_rows)
    metrics = [
        "success", "collision_free_success", "timeout", "collision", "wall_collision", "task_progress",
        "total_time", "min_distance", "fallback_rate",
        "fallback_orca_proposal_rate", "fallback_executor_geometric_rate",
        "fallback_executor_rate", "fallback_kinematic_rate",
        "fallback_static_rate", "fallback_violation_only_rate",
        "change_point_events",
        "belief_update_windows", "belief_selection_audit_windows",
        "planner_sensitivity_audit_windows",
        "belief_selection_change_rate", "belief_action_change_rate",
        "belief_score_l1", "belief_selected_regret",
        "belief_total_score_l1", "belief_total_score_max_abs_delta",
        "belief_selected_total_delta", "belief_uniform_selected_total_delta",
        "belief_selected_decision_margin", "uniform_selected_decision_margin",
        "belief_top2_decision_margin", "uniform_top2_decision_margin",
        "belief_common_admissible_fraction", "belief_admissibility_changed_fraction",
        "belief_fallback_changed", "belief_weighted_task_l1",
        "belief_vertex_selection_possible", "belief_vertex_minimum_margin",
        "belief_weighted_safety_l1", "belief_weighted_human_l1",
        "belief_weighted_smoothness_l1", "belief_weighted_risk_l1",
        "belief_observable_penalty_l1", "belief_candidate_bias_l1",
        "belief_prediction_nll_gain", "belief_prediction_mse_gain",
        "final_belief_entropy", "ranking_spearman", "ranking_kendall",
        "ranking_oracle_hit_rate", "ranking_oracle_top3_hit_rate", "ranking_oracle_regret", "learned_safe_fraction",
        "geometric_safe_fraction",
        "executor_safe_fraction", "executor_geometric_safe_fraction", "kinematic_safe_fraction", "static_safe_fraction", "dynamic_safe_fraction",
        "learned_rejection_fraction", "task_positive_progress_fraction", "executor_projection_events",
    ]
    summary = {"policies": {}}
    for index, policy in enumerate(args.policies):
        selected = [row for row in records if row["policy"] == policy]
        summary["policies"][policy] = {}
        for metric_index, metric in enumerate(metrics):
            values = [float(row[metric]) for row in selected if math.isfinite(float(row[metric]))]
            if metric.startswith("ranking_") and not values:
                continue
            summary["policies"][policy][metric] = _bootstrap(
                values, args.bootstrap_seed + index * 31 + metric_index
            )
    persistent = {(row["seed"], row["case"]): row for row in records if row["policy"] == "tracer-persistent"}
    summary["paired_contrasts"] = {}
    for policy in args.policies:
        if policy == "tracer-persistent":
            continue
        compared = {(row["seed"], row["case"]): row for row in records if row["policy"] == policy}
        delta = [float(compared[key]["task_progress"] - persistent[key]["task_progress"]) for key in persistent.keys() & compared.keys()]
        summary["paired_contrasts"][f"{policy}-persistent:task_progress"] = _bootstrap(delta, args.bootstrap_seed + 500 + len(summary["paired_contrasts"]))
    external_commit = subprocess.check_output(["git", "-C", str(external_root), "rev-parse", "HEAD"], text=True).strip()
    for checkpoint in response_checkpoints.values():
        if _sha256(Path(checkpoint["path"])) != checkpoint["sha256"]:
            raise ValueError("response checkpoint changed during navigation evaluation")
        if checkpoint.get("calibration_path") and _sha256(Path(checkpoint["calibration_path"])) != checkpoint["calibration_sha256"]:
            raise ValueError("belief calibration changed during navigation evaluation")
    try:
        external_path = str(external_root.relative_to(root))
    except ValueError:
        # A clean audit worktree may intentionally consume a shared, immutable
        # external checkout located beside the worktree.
        external_path = str(external_root)
    metadata = {
        "protocol": "tracer-official-crowdsimplus-full-loop-v1",
        "claim_boundary": "TRACER response-belief-planning loop executed in official CrowdSimPlus with official ORCA+ human dynamics; behavior-response/navigation evidence only, not real-human causality",
        "git": git,
        "development_run": bool(git["dirty"]),
        "source_hashes": source_hashes,
        "external_repository": {"path": external_path, "commit": external_commit},
        "device": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parameters": vars(args),
        "trajectory_contract": {
            "motion_profile": getattr(args, "motion_profile", "holonomic"),
            "motion_policy_sha256": _sha256(Path(args.motion_policy_config)) if getattr(args, "motion_profile", "holonomic") == "sicnav-unicycle" else None,
            "unicycle_support": "signed speed with no moving sign reversal; explicit heading for zero-speed turns",
            "static_braking_guard": bool(getattr(args, "static_braking_guard", False)),
            "candidate_fields": "world_x, world_y, velocity_x, velocity_y",
            "position_integration": "p_0=robot_position+dt*v_0; p_t=p_(t-1)+dt*v_t",
            "executed_prefix": int(args.execute_steps),
            "dt": float(args.env_dt),
            "task_progress_direction": "normalize(robot_goal-robot_position), never entity-relative robot_current",
            "sage_multi_entity_mode": "complete entity graph -> world-frame robot control trajectory",
        },
        "response_contract": {
            "response_delay_steps": int(getattr(model, "response_delay_steps", getattr(model.config, "response_delay_steps", 0))),
            "backend": args.response_backend,
            "checkpoints": response_checkpoints,
            "mode_semantics": (
                "single CrowdSimPlus response distribution duplicated into two neutral belief slots; no external behavior-mode claim"
                if args.response_backend == "crowdsimplus-paired"
                else getattr(model, "mode_semantics", "persistent synthetic behavior modes")
            ),
            "prediction_horizon": int(model.config.horizon),
            "dt": float(model.config.dt),
            "training_dt": float(getattr(model, "training_dt", model.config.dt)),
        },
        "ranking_audit_contract": {
            "realized_horizon": int(model.config.horizon),
            "task_steps": int(args.execute_steps),
            "stationary_reference_rollout": True,
            "observable_penalties_matched": True,
            "oracle_support": "finite planner decision scores after safety/fallback gating",
        },
        "records": len(records),
        "ranking_audit_rows": len(ranking_audit_rows),
        "window_audit_rows": len(window_audit_rows),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# TRACER on official CrowdSimPlus", "",
        f"Main commit `{git['commit']}`; clean `{not git['dirty']}`; official source `{external_commit}`.",
        "The official environment supplies resettable scene generation, ORCA+ human response, static line obstacles, and action execution. TRACER supplies per-human belief updates and finite candidate planning.",
        "",
        "## Results", "",
    ]
    for policy in args.policies:
        item = summary["policies"][policy]
        lines.append(
            f"- `{policy}`: success `{item['success']['mean']:.3f}`, "
            f"collision-free success `{item['collision_free_success']['mean']:.3f}`, "
            f"collision `{item['collision']['mean']:.3f}`, progress `{item['task_progress']['mean']:.3f}`, "
            f"min distance `{item['min_distance']['mean']:.3f}`, fallback `{item['fallback_rate']['mean']:.3f}`."
        )
    report = "\n".join(lines) + "\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    artifact_names = ["metadata.json", "summary.json", "episode_metrics.csv", "report.md", "source_hashes.json"]
    if ranking_audit_rows:
        artifact_names.append("ranking_audit.csv")
    if window_audit_rows:
        artifact_names.append("window_metrics.csv")
    artifacts = {name: _sha256(output / name) for name in artifact_names}
    manifest = {
        "status": "COMPLETED",
        "protocol": metadata["protocol"],
        "git": git,
        "external_repository": metadata["external_repository"],
        "artifacts": artifacts,
        "claim_boundary": metadata["claim_boundary"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"metadata": metadata, "summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Allow an explicitly labelled development run with a source snapshot; never a clean confirmatory run.",
    )
    parser.add_argument("--external-root", default="raw_data/external/source_repos/sicnav_official_filtered")
    parser.add_argument("--env-config", default="raw_data/external/source_repos/sicnav_official_filtered/sicnav/configs/env.config")
    parser.add_argument("--checkpoint-dir", default="experiments/independent_multientity_suite_20260902_final/checkpoints")
    parser.add_argument(
        "--response-backend",
        choices=("synthetic-persistent", "crowdsimplus-paired", "crowdsimplus-persistent"),
        default="synthetic-persistent",
        help="FRM checkpoint family used for candidate-conditioned human response prediction.",
    )
    parser.add_argument(
        "--external-response-checkpoint-dir",
        default="experiments/external_mechanism/crowdsimplus_paired_h1_20260904/checkpoints",
        help="Directory containing joint_seed{seed}.pt for the CrowdSimPlus-paired backend.",
    )
    parser.add_argument(
        "--sage-checkpoint",
        default="experiments/pilot/h3_sage_generator_p0_seed2027/synthetic_sage_epoch20.pt",
        help="SAGE generator checkpoint used by candidate-profile=sage.",
    )
    parser.add_argument("--output-dir", default="experiments/tracer_sicnav_crowdsimplus_20260903")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2031, 2039, 2053, 2063])
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument(
        "--case-start",
        type=int,
        default=0,
        help="First official test-case index; use disjoint ranges for validation and test runs.",
    )
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument(
        "--candidate-profile",
        choices=("sage", "sage-hybrid", "sage-adaptive", "sage-recovery", "goal-random", "orca-augmented"),
        default="goal-random",
        help="Candidate source. 'sage-recovery' preserves native SAGE and appends continuous safe controls only when its executable support is empty.",
    )
    parser.add_argument(
        "--sage-history-mode",
        choices=("per-human", "nearest"),
        default="per-human",
        help="SAGE conditioning for CrowdSimPlus; per-human covers every observed track, nearest reproduces the legacy focal-track protocol.",
    )
    parser.add_argument(
        "--fallback-profile",
        choices=("least-risk", "orca-proposal"),
        default="least-risk",
        help="Fallback after all model candidates are rejected; ORCA proposal is only valid with the augmented profile.",
    )
    parser.add_argument(
        "--human-num",
        type=int,
        default=None,
        help="Override the official test reset with this many human entities; None uses env.config.",
    )
    # CrowdSimPlus uses a 15 s time limit and 0.25 s steps.  One executed
    # step per receding-horizon window keeps the action sent to the simulator
    # identical to the branch that was safety-checked and ranked.
    parser.add_argument("--windows", type=int, default=60)
    parser.add_argument("--execute-steps", type=int, default=1)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--env-dt", type=float, default=0.25)
    parser.add_argument("--motion-profile", choices=("holonomic", "sicnav-unicycle"), default="holonomic")
    parser.add_argument("--allow-temporal-transfer", action="store_true", help="explicit diagnostic only; does not retrain a decoder at the environment clock")
    parser.add_argument("--motion-policy-config", default="raw_data/external/source_repos/sicnav_official_filtered/sicnav/configs/policy.config")
    parser.add_argument(
        "--model-dt",
        type=float,
        default=0.25,
        help="Time step used by candidate integration and FRM scoring; keep equal to env-dt.",
    )
    parser.add_argument("--human-cost-weight", type=float, default=0.6)
    parser.add_argument(
        "--human-cost-scale",
        type=float,
        default=1.0,
        help="Optional multiplicative scale for response-induced human cost; keep 1.0 for the frozen protocol and calibrate any alternative on validation.",
    )
    parser.add_argument(
        "--belief-cost-leverage",
        type=float,
        default=1.0,
        help="Scale only the belief-dependent deviation from uniform response costs; 1.0 preserves the frozen protocol.",
    )
    parser.add_argument("--task-speed-scale", type=float, default=1.7)
    parser.add_argument("--safety-weight", type=float, default=2.0)
    parser.add_argument("--likelihood-scale", type=float, default=0.05)
    parser.add_argument("--belief-calibration-path",
                        help="Validation-only checkpoint-bound calibration JSON; supports {seed} and supplies evidence cadence and scale.")
    parser.add_argument("--adaptive-base-hazard", type=float, default=0.0)
    parser.add_argument("--adaptive-max-hazard", type=float, default=None)
    parser.add_argument("--adaptive-surprise-threshold", type=float, default=None)
    parser.add_argument("--adaptive-surprise-scale", type=float, default=1.0)
    parser.add_argument("--adaptive-hard-reset", action="store_true")
    parser.add_argument("--adaptive-threshold-hysteresis", type=float, default=0.0)
    parser.add_argument("--adaptive-cooldown-steps", type=int, default=0)
    parser.add_argument(
        "--belief-update-stride",
        type=int,
        default=None,
        help="Override a checkpoint's calibrated evidence cadence; requires --allow-belief-cadence-override.",
    )
    parser.add_argument(
        "--allow-belief-cadence-override",
        action="store_true",
        help="Allow an explicitly requested belief cadence for causal closed-loop diagnostics.",
    )
    parser.add_argument("--hazard-rate", type=float, default=0.18)
    parser.add_argument("--forgetting-rate", type=float, default=0.18)
    parser.add_argument("--change-point-threshold", type=float, default=5.0)
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument("--risk-alpha", type=float, default=0.1)
    parser.add_argument("--risk-samples", type=int, default=32)
    parser.add_argument("--crowd-risk-aggregation", choices=("mean", "max", "union-bound"), default="mean")
    parser.add_argument("--temporal-risk-sampling", choices=("independent", "shared"), default="independent")
    parser.add_argument("--swept-human-guard", dest="swept_human_guard", action="store_true")
    parser.add_argument("--no-swept-human-guard", dest="swept_human_guard", action="store_false")
    parser.set_defaults(swept_human_guard=False)
    parser.add_argument("--predictive-clearance-weight", type=float, default=0.0)
    parser.add_argument("--predictive-clearance-steps", type=int, default=4)
    parser.add_argument("--sage-diversity-weight", type=float, default=0.0)
    parser.add_argument("--goal-return-candidates", action="store_true")
    parser.add_argument("--terminal-goal-weight", type=float, default=0.0)
    parser.add_argument("--static-route-candidates", action="store_true")
    parser.add_argument("--align-route-before-motion", action="store_true",
                        help="Add three braking/turning route candidates that align the body before moving along the observed static route.")
    parser.add_argument("--fallback-task-tiebreak", action="store_true",
                        help="Break tied fallback safety scores using the existing planning cost without accepting a worse safety score.")
    parser.add_argument("--fallback-progress-weight", type=float, default=0.0,
                        help="When fallback is forced, reward normalized goal progress within the executor-safe braking set.")
    parser.add_argument("--fallback-progress-after", type=int, default=0,
                        help="Activate fallback progress after this many consecutive fallback windows; zero activates immediately.")
    parser.add_argument("--response-audit-stride", type=int, default=0,
                        help="Collect cloned response branches every N windows after selection; zero disables the audit.")
    parser.add_argument("--response-audit-candidates", type=int, default=6)
    parser.add_argument("--belief-selection-audit", action="store_true",
                        help="Rescore the identical state and feasible candidate set with uniform belief; never execute the audit action.")
    parser.add_argument(
        "--planner-sensitivity-audit",
        action="store_true",
        help="Record belief-vs-uniform score decomposition, ranking margins, masks, and fallback changes on the identical candidate pool.",
    )
    parser.add_argument("--belief-evidence-audit", action="store_true",
                        help="Record per-human factual prefix likelihoods and pre/post-update beliefs without hidden mode labels.")
    parser.add_argument("--candidate-debug-windows", type=int, default=0,
                        help="Save observable pre-selection state, feasible candidates and scores for the first N windows without changing control.")
    parser.add_argument("--static-braking-guard", action="store_true",
                        help="Add a feasible braking candidate and require every selected prefix to preserve a static-safe stopping path.")
    parser.add_argument("--goal-terminal-response", action="store_true",
                        help="Stop predicted human-impact and risk costs after the candidate first reaches the navigation goal.")
    parser.add_argument("--static-route-weight", type=float, default=0.0)
    parser.add_argument("--feedback-route-candidates", action="store_true")
    parser.add_argument("--prefix-diversity", action="store_true",
                        help="Append acceleration-feasible first-prefix speed variants while retaining all original candidates and safety guards.")
    parser.add_argument("--include-incumbent", action="store_true")
    parser.add_argument(
        "--kinematic-clearance-weight",
        type=float,
        default=0.0,
        help="Soft cost from observed constant-velocity human clearance; hard geometric checks remain separate.",
    )
    parser.add_argument(
        "--hard-risk-filter",
        action="store_true",
        help="Hard-reject candidates above risk-alpha; default is a soft risk penalty plus geometric hard safety.",
    )
    parser.add_argument(
        "--risk-weight",
        type=float,
        default=0.5,
        help="Additional ranking penalty for predicted violation probability among safe candidates.",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=0.0,
        help="Penalty on the first-action velocity jump relative to the current robot velocity.",
    )
    parser.add_argument(
        "--kinematic-safety-margin",
        type=float,
        default=0.0,
        help="Extra observed-state body-tube margin; zero enables the executed-prefix guard.",
    )
    parser.add_argument(
        "--static-safety-margin",
        type=float,
        default=0.0,
        help="Extra static-obstacle swept-path margin; zero enables the executed-prefix guard.",
    )
    parser.add_argument(
        "--geometric-safety-prefix-steps",
        type=int,
        default=1,
        help="Limit geometric guards to this executed prefix; zero checks the full candidate horizon.",
    )
    parser.add_argument("--candidate-seed", type=int, default=839731)
    parser.add_argument(
        "--ranking-audit-windows",
        type=int,
        default=1,
        help="Replay every candidate for this many initial windows; set to zero only for a speed-only run.",
    )
    parser.add_argument("--sage-step-stride", type=int, default=10)
    parser.add_argument("--sage-goal-speed", type=float, default=1.0)
    parser.add_argument("--sage-oversample-factor", type=int, default=2)
    parser.add_argument(
        "--recovery-candidates",
        type=int,
        default=8,
        help="Maximum continuous recovery candidates appended when native SAGE has no executable geometric-safe branch.",
    )
    parser.add_argument(
        "--recovery-directions",
        type=int,
        default=32,
        help="Angular resolution of the observable-state recovery velocity set.",
    )
    parser.add_argument(
        "--recovery-safety-margin",
        type=float,
        default=0.02,
        help="Additional one-step human-body margin required from recovery candidates; 0.02 m balances clearance and deadlock risk.",
    )
    parser.add_argument(
        "--recovery-activation",
        choices=("empty-support", "nearby", "always"),
        default="empty-support",
        help="Append recovery controls only for empty native support or at every planning window.",
    )
    parser.add_argument(
        "--recovery-activation-distance",
        type=float,
        default=0.8,
        help="Nearest-human distance that activates recovery controls in nearby mode.",
    )
    parser.add_argument(
        "--recovery-cost-penalty",
        type=float,
        default=0.0,
        help="Switching penalty applied only to appended recovery candidates before ranking.",
    )
    parser.add_argument(
        "--sage-speed-limit-scale",
        type=float,
        default=0.0,
        help="Optional multiplier on robot v_pref for SAGE control projection; non-positive preserves native controls.",
    )
    parser.add_argument(
        "--sage-dynamic-safety-filter",
        action="store_true",
        help="Hard-filter SAGE branches against generated entity futures; disabled by default until external response calibration is validated.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=849731)
    parser.add_argument("--belief-vertex-audit", action="store_true")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
