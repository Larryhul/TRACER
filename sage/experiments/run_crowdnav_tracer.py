"""Run the TRACER response-belief-planning loop in official CrowdNav.

The upstream CrowdNav simulator and ORCA human policy are kept unchanged. The
adapter uses SAGE to generate a finite single-robot candidate pool, evaluates
each candidate with the frozen TRACER response model, updates one categorical
belief per CrowdNav human track from the executed three-step prefix, and
replans. Each policy is reset on the same CrowdNav ``test_case`` so the
comparison is paired at the simulator seed level.

This is a third-party single-robot portability experiment. It is not the
three-robot H3 protocol and it does not identify real-human response modes.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import math
import statistics
import subprocess
import time
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
from sage.experiments.eval_persistent_closed_loop import _load_model
from sage.response.persistent_models import trajectory_mode_log_likelihood
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
    "official-orca",
    "official-linear",
)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap(values: list[float], seed: int, samples: int = 4000) -> dict[str, Any]:
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


def _crowdnav_env(config_path: Path, policy_name: str):
    # Imports are intentionally local so the main TRACER test suite does not
    # require the legacy third-party environment.
    from crowd_sim.envs import CrowdSim
    from crowd_sim.envs.policy.linear import Linear
    from crowd_sim.envs.policy.orca import ORCA
    from crowd_sim.envs.utils.robot import Robot

    config = configparser.RawConfigParser()
    if not config.read(config_path):
        raise FileNotFoundError(config_path)
    # The default upstream config hides the robot from humans. For a response
    # experiment the robot must be an observable agent; all other parameters
    # remain from the official config.
    config.set("robot", "visible", "true")
    env = CrowdSim()
    env.configure(config)
    robot = Robot(config, "robot")
    robot.set_policy(ORCA() if policy_name == "official-orca" else Linear())
    env.set_robot(robot)
    return env, robot


def _human_positions(env: Any) -> np.ndarray:
    return np.asarray([human.get_position() for human in env.humans], dtype=np.float32)


def _human_velocities(env: Any) -> np.ndarray:
    return np.asarray([(human.vx, human.vy) for human in env.humans], dtype=np.float32)


def _initial_histories(
    positions: np.ndarray,
    velocities: np.ndarray,
    history: int,
    dt: float,
) -> np.ndarray:
    offsets = np.arange(1 - history, 1, dtype=np.float32) * float(dt)
    return positions[:, None, :] + velocities[:, None, :] * offsets[None, :, None]


def _candidate_world_paths(
    adapter: SAGECandidateAdapter,
    histories: np.ndarray,
    robot_position: np.ndarray,
    robot_velocity: np.ndarray,
    robot_goal: np.ndarray,
    dt: float,
    sample_seed: int,
    history_mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    if histories.ndim != 3 or histories.shape[-1] != 2:
        raise ValueError("histories must have shape (humans, history, 2)")
    if history_mode == "per-human":
        candidate_blocks = []
        for human_index, human_history in enumerate(histories):
            pool = adapter.candidate_pool(
                human_history,
                robot_position[None],
                robot_velocity[None],
                dt,
                sample_seed + human_index * 1009,
                robot_goals=robot_goal[None],
            )
            relative = pool.robot_candidates.numpy()[:, 0]
            world = relative.copy()
            world[..., :2] += human_history[-1][None, None, :]
            candidate_blocks.append(world)
        merged = np.concatenate(candidate_blocks, axis=0)
        target = robot_goal - robot_position
        target /= max(float(np.linalg.norm(target)), 1e-6)
        progress = ((merged[:, -1, :2] - robot_position[None]) * target[None]).sum(axis=-1)
        count = int(adapter.config.num_candidates)
        if merged.shape[0] > count:
            order = np.argsort(-progress, kind="mergesort")
            # Retain both high-progress and lower-speed interaction branches.
            selected = order[np.linspace(0, len(order) - 1, count).astype(int)]
            merged = merged[selected]
        world_position = merged[..., :2]
        world_velocity = merged[..., 2:]
        return world_position.astype(np.float32), world_velocity.astype(np.float32)
    if history_mode == "nearest":
        focal = int(np.linalg.norm(histories[:, -1] - robot_position[None], axis=-1).argmin())
        aggregate_history = histories[focal]
    elif history_mode == "mean":
        aggregate_history = histories.mean(axis=0)
    else:
        raise ValueError("history_mode must be 'per-human', 'nearest' or 'mean'")
    pool = adapter.candidate_pool(
        aggregate_history,
        robot_position[None],
        robot_velocity[None],
        dt,
        sample_seed,
        robot_goals=robot_goal[None],
    )
    relative = pool.robot_candidates.numpy()[:, 0]
    world_position = relative[:, :, :2] + aggregate_history[-1][None, None, :]
    world_velocity = relative[:, :, 2:]
    return world_position.astype(np.float32), world_velocity.astype(np.float32)


def _model_predictions(
    model: torch.nn.Module,
    histories: np.ndarray,
    human_positions: np.ndarray,
    robot_position: np.ndarray,
    robot_velocity: np.ndarray,
    world_position: np.ndarray,
    world_velocity: np.ndarray,
    dt: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict each candidate for every official CrowdNav human track."""

    humans, history = histories.shape[:2]
    candidates, horizon = world_position.shape[:2]
    current = np.zeros((humans, candidates, 3, 4), dtype=np.float32)
    actions = np.zeros((humans, candidates, 3, horizon, 4), dtype=np.float32)
    current[:, :, 0, :2] = robot_position[None, None] - human_positions[:, None]
    current[:, :, 0, 2:] = robot_velocity[None, None]
    actions[:, :, 0, : , :2] = world_position[None] - human_positions[:, None, None]
    actions[:, :, 0, :, 2:] = world_velocity[None]
    history_tensor = torch.from_numpy(histories[:, None]).to(device=device)
    history_tensor = history_tensor[:, None].expand(humans, candidates, 1, history, 2)
    history_tensor = history_tensor.reshape(humans * candidates, 1, history, 2)
    current_tensor = torch.from_numpy(current).to(device=device).reshape(humans * candidates, 1, 3, 4)
    action_tensor = torch.from_numpy(actions).to(device=device).reshape(
        humans * candidates, 1, 3, horizon, 4
    )
    robot_mask = torch.tensor([True, False, False], device=device)[None, None].expand(
        humans * candidates, 1, 3
    )
    output = model(
        history_tensor,
        current_tensor,
        action_tensor,
        include_pair_interactions=True,
        robot_mask=robot_mask,
    )
    means = output["mean"][:, 0].reshape(humans, candidates, model.config.num_modes, horizon, 2)
    log_stds = output["log_std"][:, 0].reshape_as(means)
    return means.detach().cpu().numpy(), log_stds.detach().cpu().numpy()


def _select_candidate(
    beliefs: np.ndarray,
    means: np.ndarray,
    world_position: np.ndarray,
    robot_position: np.ndarray,
    robot_goal: np.ndarray,
    human_positions: np.ndarray,
    dt: float,
    collision_margin: float,
    official_safe: np.ndarray | None = None,
    task_steps: int = 1,
    robot_velocity: np.ndarray | None = None,
    world_velocity: np.ndarray | None = None,
    log_stds: np.ndarray | None = None,
    response_uncertainty_weight: float = 0.0,
    action_smoothness_weight: float = 0.0,
) -> tuple[int, np.ndarray, dict[str, float]]:
    probabilities = np.exp(beliefs)
    if means.shape[:2] != (len(human_positions), len(world_position)):
        raise ValueError("means must have shape (humans, candidates, modes, horizon, 2)")
    predicted_velocity = np.einsum("hm,hcmtd->hctd", probabilities, means)
    prefix_steps = max(1, min(int(task_steps), predicted_velocity.shape[2], world_position.shape[1]))
    predicted_positions = human_positions[:, None, None, :] + np.cumsum(
        predicted_velocity[:, :, :prefix_steps] * float(dt), axis=2
    )
    distances = np.linalg.norm(
        world_position[None, :, :prefix_steps, :] - predicted_positions,
        axis=-1,
    )
    min_distance = distances.min(axis=(0, 2))
    safety = np.maximum(0.0, 1.0 - min_distance) ** 2
    response_cost = np.linalg.norm(
        predicted_velocity[:, :, :prefix_steps], axis=-1
    ).mean(axis=(0, 2))
    goal_direction = robot_goal - robot_position
    goal_norm = max(float(np.linalg.norm(goal_direction)), 1e-6)
    goal_direction /= goal_norm
    displacement = world_position[:, prefix_steps - 1] - robot_position[None]
    task_progress = (displacement * goal_direction[None]).sum(axis=-1)
    scores = -task_progress / max(goal_norm, 1.0) + 2.0 * safety + 0.06 * response_cost
    uncertainty_cost = np.zeros_like(scores)
    if log_stds is not None and response_uncertainty_weight > 0.0:
        uncertainty_cost = np.exp(
            np.clip(log_stds[:, :, :, :prefix_steps], -10.0, 5.0)
        ).mean(axis=(0, 2, 3, 4))
        scores += float(response_uncertainty_weight) * uncertainty_cost
    smoothness_cost = np.zeros_like(scores)
    if (
        robot_velocity is not None
        and world_velocity is not None
        and action_smoothness_weight > 0.0
    ):
        smoothness_cost = np.linalg.norm(
            world_velocity[:, 0] - np.asarray(robot_velocity)[None], axis=-1
        )
        scores += float(action_smoothness_weight) * smoothness_cost
    hard_safe = min_distance >= float(collision_margin)
    if official_safe is not None:
        if official_safe.shape != hard_safe.shape:
            raise ValueError("official_safe must provide one flag per candidate")
        hard_safe &= official_safe
    fallback = not bool(hard_safe.any())
    selection_scores = np.where(hard_safe, scores, np.inf)
    if fallback:
        fallback_support = np.ones_like(hard_safe)
        if official_safe is not None and bool(official_safe.any()):
            fallback_support = official_safe
        selected = int(np.argmax(np.where(fallback_support, min_distance, -np.inf)))
    else:
        selected = int(np.argmin(selection_scores))
    return selected, predicted_velocity, {
        "task_progress": float(task_progress[selected]),
        "safety_cost": float(safety[selected]),
        "human_cost": float(response_cost[selected]),
        "uncertainty_cost": float(uncertainty_cost[selected]),
        "smoothness_cost": float(smoothness_cost[selected]),
        "predicted_min_distance": float(min_distance[selected]),
        "hard_safe_fraction": float(hard_safe.mean()),
        "safety_fallback": float(fallback),
        "official_safe_fraction": float(official_safe.mean()) if official_safe is not None else float("nan"),
    }


def _official_prefix_safe(
    config_path: Path,
    case: int,
    action_history: list[np.ndarray],
    candidate_velocity: np.ndarray,
    execute_steps: int,
) -> np.ndarray:
    """Replay each branch from the same official test case and action history."""

    from crowd_sim.envs.utils.action import ActionXY

    safe = []
    for path in candidate_velocity:
        branch, _ = _crowdnav_env(config_path, "tracer-prefix-check")
        branch.reset("test", case)
        collided = False
        for replay_path in [*action_history, path]:
            for velocity in replay_path[:execute_steps]:
                _, _, done, info = branch.step(
                    ActionXY(float(velocity[0]), float(velocity[1]))
                )
                if type(info).__name__ == "Collision":
                    collided = True
                    break
                if done:
                    break
            if collided or done:
                break
        safe.append(not collided)
    return np.asarray(safe, dtype=bool)


def _reset_official_policy_state(env: Any) -> None:
    """Remove lookahead-only ORCA caches before executing the selected branch."""

    for human in env.humans:
        if hasattr(human.policy, "sim"):
            human.policy.sim = None


def _update_beliefs(
    beliefs: np.ndarray,
    means: np.ndarray,
    log_stds: np.ndarray,
    selected: int,
    observed_velocity: np.ndarray,
    execute_steps: int,
    likelihood_scale: float,
    policy: str,
    hazard_rate: float,
    forgetting_rate: float,
    change_point_threshold: float,
) -> tuple[np.ndarray, int]:
    if not policy.startswith("tracer-"):
        return beliefs, 0
    updated = beliefs.copy()
    change_points = 0
    for human in range(len(updated)):
        likelihood = trajectory_mode_log_likelihood(
            torch.from_numpy(means[human, selected, :, :execute_steps]),
            torch.from_numpy(log_stds[human, selected, :, :execute_steps]),
            torch.from_numpy(observed_velocity[human, :execute_steps]),
        )
        current = torch.from_numpy(updated[human])
        scaled_likelihood = likelihood_scale * likelihood
        if policy == "tracer-reset":
            current = uniform_log_belief(current.numel())
        elif policy == "tracer-hazard":
            updated[human] = hazard_belief_update(
                current, scaled_likelihood, hazard_rate
            ).numpy()
            continue
        elif policy == "tracer-forgetting":
            posterior = categorical_bayes_update(current, scaled_likelihood)
            updated[human] = apply_belief_forgetting(
                posterior, forgetting_rate
            ).numpy()
            continue
        elif policy == "tracer-change-point":
            changed = bool(
                detect_change_point(
                    current, likelihood, change_point_threshold
                )
            )
            if changed:
                current = uniform_log_belief(current.numel())
                change_points += 1
        posterior = categorical_bayes_update(current, scaled_likelihood)
        updated[human] = posterior.numpy()
    return updated, change_points


@torch.no_grad()
def _run_episode(
    policy: str,
    model: torch.nn.Module | None,
    adapter: SAGECandidateAdapter | None,
    config_path: Path,
    case: int,
    args: argparse.Namespace,
    sample_seed: int,
) -> dict[str, Any]:
    env, robot = _crowdnav_env(config_path, policy)
    observation = env.reset("test", case)
    del observation
    human_positions = _human_positions(env)
    human_velocities = _human_velocities(env)
    histories = _initial_histories(human_positions, human_velocities, args.history, args.dt)
    beliefs = np.full((len(env.humans), model.config.num_modes if model else 3), -math.log(model.config.num_modes if model else 3), dtype=np.float32)
    initial_robot = np.asarray(robot.get_position(), dtype=np.float32)
    goal = np.asarray(robot.get_goal_position(), dtype=np.float32)
    previous_robot = initial_robot.copy()
    min_distance = float("inf")
    collisions = 0
    task_progress = 0.0
    human_cost = 0.0
    safety_cost = 0.0
    hard_safe_fractions = []
    safety_fallbacks = []
    official_safe_fractions = []
    selected_actions = []
    action_history: list[np.ndarray] = []
    belief_entropies = []
    change_point_events = []
    done = False
    final_info = ""

    for window in range(args.windows):
        before_positions = _human_positions(env)
        robot_position = np.asarray(robot.get_position(), dtype=np.float32)
        robot_velocity = np.asarray(robot.get_velocity(), dtype=np.float32)
        if model is not None:
            world_position, world_velocity = _candidate_world_paths(
                adapter,
                histories,
                robot_position,
                robot_velocity,
                goal,
                args.dt,
                sample_seed + window,
                args.sage_history_mode,
            )
            means, log_stds = _model_predictions(
                model,
                histories,
                before_positions,
                robot_position,
                robot_velocity,
                world_position,
                world_velocity,
                args.dt,
                args.device,
            )
            official_safe = _official_prefix_safe(
                config_path,
                case,
                action_history,
                world_velocity,
                args.execute_steps,
            )
            decision_beliefs = beliefs if policy != "tracer-reset" else np.full_like(
                beliefs, -math.log(beliefs.shape[-1])
            )
            selected, _, diagnostics = _select_candidate(
                decision_beliefs,
                means,
                world_position,
                robot_position,
                goal,
                before_positions,
                args.dt,
                args.collision_margin,
                official_safe,
                task_steps=(
                    args.selection_steps
                    if args.selection_steps > 0
                    else args.execute_steps
                ),
                robot_velocity=robot_velocity,
                world_velocity=world_velocity,
                log_stds=log_stds,
                response_uncertainty_weight=args.response_uncertainty_weight,
                action_smoothness_weight=args.action_smoothness_weight,
            )
            hard_safe_fractions.append(diagnostics["hard_safe_fraction"])
            safety_fallbacks.append(diagnostics["safety_fallback"])
            official_safe_fractions.append(diagnostics["official_safe_fraction"])
            selected_actions.append(selected)
            posterior_probability = np.exp(beliefs)
            belief_entropies.append(float((-(posterior_probability * beliefs).sum(axis=-1)).mean()))
            candidate_steps = world_velocity[selected]
        else:
            current_observation = [human.get_observable_state() for human in env.humans]
            action = robot.act(current_observation)
            candidate_steps = np.repeat(
                np.asarray([[action.vx, action.vy]], dtype=np.float32),
                args.execute_steps,
                axis=0,
            )
            diagnostics = {
                "task_progress": 0.0,
                "safety_cost": 0.0,
                "human_cost": 0.0,
                "predicted_min_distance": float("nan"),
                "hard_safe_fraction": float("nan"),
                "safety_fallback": float("nan"),
                "official_safe_fraction": float("nan"),
            }

        executed_positions = []
        executed_robot_positions = []
        for step in range(args.execute_steps):
            if done:
                break
            from crowd_sim.envs.utils.action import ActionXY

            action = ActionXY(float(candidate_steps[min(step, len(candidate_steps) - 1), 0]), float(candidate_steps[min(step, len(candidate_steps) - 1), 1]))
            _, _, done, info = env.step(action)
            final_info = type(info).__name__
            executed_positions.append(_human_positions(env))
            executed_robot_positions.append(np.asarray(robot.get_position(), dtype=np.float32))
            if final_info == "Collision":
                collisions += 1
            if done:
                break
        if executed_positions:
            after_positions = np.asarray(executed_positions, dtype=np.float32)
            observed_velocity = np.diff(
                np.concatenate([before_positions[None], after_positions], axis=0), axis=0
            ) / float(args.dt)
            if len(observed_velocity) < args.execute_steps:
                observed_velocity = np.concatenate(
                    [observed_velocity, np.repeat(observed_velocity[-1:], args.execute_steps - len(observed_velocity), axis=0)],
                    axis=0,
                )
            observed_velocity = observed_velocity.transpose(1, 0, 2)
            if model is not None:
                beliefs, change_points = _update_beliefs(
                    beliefs,
                    means,
                    log_stds,
                    selected,
                    observed_velocity,
                    args.execute_steps,
                    args.likelihood_scale,
                    policy,
                    args.hazard_rate,
                    args.forgetting_rate,
                    args.change_point_threshold,
                )
                change_point_events.append(float(change_points))
            histories = np.concatenate([histories[:, len(executed_positions):], after_positions.transpose(1, 0, 2)], axis=1)
            action_history.append(candidate_steps[: len(executed_positions)].copy())
            current_robot = executed_robot_positions[-1]
            task_progress += float(np.linalg.norm(previous_robot - goal) - np.linalg.norm(current_robot - goal))
            previous_robot = current_robot
            human_cost += float(np.linalg.norm(observed_velocity, axis=-1).mean() * len(executed_positions))
            observed_distances = np.linalg.norm(
                after_positions - np.asarray(executed_robot_positions)[:, None, :], axis=-1
            )
            min_distance = min(min_distance, float(observed_distances.min()))
            safety_cost += float(np.maximum(0.0, 1.0 - observed_distances.min(axis=1)).mean())
        if done:
            break

    success = float(final_info == "ReachGoal")
    timeout = float(final_info == "Timeout")
    total_cost = -task_progress / max(float(np.linalg.norm(goal - initial_robot)), 1.0) + 2.0 * safety_cost + 0.06 * human_cost
    return {
        "policy": policy,
        "case": case,
        "sample_seed": sample_seed,
        "windows_completed": window + 1,
        "success": success,
        "collision": float(collisions > 0),
        "timeout": timeout,
        "final_info": final_info,
        "total_time": float(env.global_time),
        "task_progress": task_progress,
        "safety_cost": safety_cost,
        "human_cost": human_cost,
        "min_distance": min_distance if math.isfinite(min_distance) else float("nan"),
        "realized_total_cost": total_cost,
        "action_change_rate": float(np.mean(np.diff(selected_actions) != 0)) if len(selected_actions) > 1 else 0.0,
        "belief_entropy": float(np.mean(belief_entropies)) if belief_entropies else float("nan"),
        "selected_action_count": len(set(selected_actions)),
        "hard_safe_fraction": float(np.mean(hard_safe_fractions)) if hard_safe_fractions else float("nan"),
        "safety_fallback": float(np.mean(safety_fallbacks)) if safety_fallbacks else float("nan"),
        "official_safe_fraction": float(np.mean(official_safe_fractions)) if official_safe_fractions else float("nan"),
        "change_point_detected": float(np.mean(change_point_events)) if change_point_events else 0.0,
        "predicted_first_window_task": float(diagnostics["task_progress"]),
        "predicted_first_window_safety": float(diagnostics["safety_cost"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    git = _git_state()
    if git["dirty"]:
        raise RuntimeError("CrowdNav third-party experiment requires a clean Git worktree")
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model_checkpoint_dir = Path(args.response_checkpoint_dir)
    adapter = SAGECandidateAdapter.from_checkpoint(
        args.sage_checkpoint,
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
        ),
    )
    records: list[dict[str, Any]] = []
    for seed in args.seeds:
        model = _load_model(
            model_checkpoint_dir / f"seed{seed}" / "response.pt",
            args.device,
            expected_dt=args.dt,
        )
        for case in range(args.case_start, args.case_start + args.cases):
            for policy in args.policies:
                policy_model = model if policy.startswith("tracer-") else None
                policy_adapter = adapter if policy_model is not None else None
                record = _run_episode(
                    policy,
                    policy_model,
                    policy_adapter,
                    Path(args.crowdnav_config),
                    case,
                    args,
                    args.data_seed + seed * 1009 + case * 9176,
                )
                record["model_seed"] = seed
                records.append(record)
                print(f"crowdnav_tracer seed={seed} case={case} policy={policy}", flush=True)
    fields = list(records[0])
    csv_path = output / "episode_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary: dict[str, Any] = {"policies": {}, "paired_contrasts": {}}
    metric_names = (
        "realized_total_cost", "success", "collision", "timeout", "total_time",
        "task_progress", "safety_cost", "human_cost", "min_distance",
        "action_change_rate", "belief_entropy", "selected_action_count",
        "hard_safe_fraction", "safety_fallback",
        "official_safe_fraction", "change_point_detected",
    )
    for index, policy in enumerate(args.policies):
        rows = [row for row in records if row["policy"] == policy]
        summary["policies"][policy] = {
            metric: _bootstrap([float(row[metric]) for row in rows if math.isfinite(float(row[metric]))], args.bootstrap_seed + index)
            for metric in metric_names
        }
    for baseline in args.policies:
        if baseline == "tracer-persistent":
            continue
        left = {(row["model_seed"], row["case"]): row for row in records if row["policy"] == "tracer-persistent"}
        right = {(row["model_seed"], row["case"]): row for row in records if row["policy"] == baseline}
        for metric in ("realized_total_cost", "success", "collision"):
            delta = [float(right[key][metric]) - float(left[key][metric]) for key in left.keys() & right.keys()]
            summary["paired_contrasts"][f"{baseline}-persistent:{metric}"] = _bootstrap(delta, args.bootstrap_seed + len(summary["paired_contrasts"]))
    crowdnav_config_path = Path(args.crowdnav_config).resolve()
    crowdnav_root = crowdnav_config_path.parents[2]
    rvo2_root = crowdnav_root.parent / "python_rvo2_official"

    def _external_repo_entry(path: Path) -> dict[str, str]:
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path)
        return {"path": label, "commit": subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()}

    metadata = {
        "protocol": "tracer-crowdnav-third-party-single-robot-v1",
        "claim_boundary": "official CrowdNav single-robot closed-loop portability; conditional response association only; no real-human causality and no multi-robot H3 claim",
        "git": git,
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": list(args.seeds),
        "cases_per_seed": args.cases,
        "paired_case_protocol": "Every policy calls official CrowdSim.reset(phase=test, test_case=case); CrowdNav maps the test_case to a deterministic NumPy seed.",
        "upstream_repositories": {
            "crowdnav": _external_repo_entry(crowdnav_root),
            "python_rvo2": _external_repo_entry(rvo2_root),
        },
        "external_semantics": "Official CrowdNav ORCA humans observe the robot; TRACER sees all stable human track slots and uses the observed executed prefix for belief updates.",
        "parameters": vars(args),
        "runtime_seconds": time.perf_counter() - start,
    }
    payload = {"metadata": metadata, "summary": summary}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    lines = [
        "# TRACER CrowdNav Third-Party Experiment", "",
        f"Official CrowdNav commit `{metadata['upstream_repositories']['crowdnav']['commit']}`; Python-RVO2 commit `{metadata['upstream_repositories']['python_rvo2']['commit']}`.",
        f"Clean main-repository commit `{git['commit']}`; CUDA `{metadata['cuda_available']}` (`{metadata['cuda_device_name']}`).",
        f"Paired protocol: `{args.seeds}` model seeds x `{args.cases}` official test cases x `{len(args.policies)}` selected policies; each policy resets the same test case.",
        "",
        "## Results", "",
    ]
    for policy in args.policies:
        metrics = summary["policies"][policy]
        lines.append(
            f"- `{policy}` cost `{metrics['realized_total_cost']['mean']:.4f}` CI "
            f"`[{metrics['realized_total_cost']['ci95_low']:.4f}, {metrics['realized_total_cost']['ci95_high']:.4f}]`; "
            f"success `{metrics['success']['mean']:.3f}`; collision `{metrics['collision']['mean']:.3f}`; "
            f"min distance `{metrics['min_distance']['mean']:.3f}`; hard-safe fraction "
            f"`{metrics['hard_safe_fraction']['mean'] if metrics['hard_safe_fraction']['count'] else float('nan'):.3f}`; "
            f"fallback `{metrics['safety_fallback']['mean'] if metrics['safety_fallback']['count'] else float('nan'):.3f}`."
        )
    lines += [
        "",
        "TRACER uses the frozen response model and SAGE candidate generator, then updates one belief per official human slot from the observed executed prefix. Persistent, hazard, forgetting, change-point, and reset are all evaluated with the same candidate/replay protocol. ORCA and Linear are official CrowdNav-compatible robot controls; they do not use TRACER belief.",
        "",
        "This experiment is single-robot portability evidence on an official simulator. It does not provide alternate-action human outcomes beyond the executed branch, does not replace the three-robot Formal-v2 H3 protocol, and does not support real-human causal or persistent-preference claims.",
        "",
    ]
    report_path = output / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "status": "COMPLETED",
        "protocol": metadata["protocol"],
        "git": git,
        "artifacts": {name: _sha256(output / name) for name in ("metadata.json", "summary.json", "episode_metrics.csv", "report.md")},
        "claim_boundary": metadata["claim_boundary"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "completion.json").write_text(json.dumps({"status": "COMPLETED", "finished_unix": time.time()}, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/tracer_crowdnav_20260902")
    parser.add_argument("--crowdnav-config", default="raw_data/external/source_repos/crowdnav_official/crowd_nav/configs/env.config")
    parser.add_argument("--response-checkpoint-dir", default="experiments/independent_multientity_suite_20260902_final/checkpoints")
    parser.add_argument("--sage-checkpoint", default="experiments/pilot/h3_sage_generator_p0_seed2027/synthetic_sage_epoch5.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2031, 2039, 2053, 2063])
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument(
        "--case-start",
        type=int,
        default=0,
        help="First official test-case index; use disjoint ranges for validation and test runs.",
    )
    parser.add_argument("--data-seed", type=int, default=728731)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument(
        "--windows",
        type=int,
        default=100,
        help="Receding-horizon windows; 100 covers CrowdNav's 25 s limit at dt=0.25 and one-step execution.",
    )
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=1,
        help="Number of actions executed before replanning; one keeps the ranked branch identical to the executed action.",
    )
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--likelihood-scale", type=float, default=0.05)
    parser.add_argument("--hazard-rate", type=float, default=0.05)
    parser.add_argument("--forgetting-rate", type=float, default=0.05)
    parser.add_argument("--change-point-threshold", type=float, default=1.0)
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument("--sage-step-stride", type=int, default=10)
    parser.add_argument("--sage-goal-speed", type=float, default=1.0)
    parser.add_argument("--sage-oversample-factor", type=int, default=2)
    parser.add_argument(
        "--sage-history-mode",
        choices=("per-human", "nearest", "mean"),
        default="per-human",
        help="SAGE conditioning mode; per-human is the default and removes virtual mean entities, mean preserves the legacy protocol.",
    )
    parser.add_argument(
        "--selection-steps",
        type=int,
        default=0,
        help="Prefix used for candidate scoring; zero uses execute-steps, while horizon reproduces endpoint scoring.",
    )
    parser.add_argument(
        "--response-uncertainty-weight",
        type=float,
        default=0.0,
        help="Penalty on SAGE/FRM response uncertainty in candidate ranking.",
    )
    parser.add_argument(
        "--action-smoothness-weight",
        type=float,
        default=0.0,
        help="Penalty on the first executed velocity jump.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=828731)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
        help="Policies to execute; default runs the complete comparison.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
