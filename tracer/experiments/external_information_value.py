from __future__ import annotations

import copy
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tracer.experiments.run_sicnav_tracer_loop import (
    _executor_safe_mask,
    _goal_response_mask,
    _info_value,
    _planning_cost_config,
    _rank_correlation,
    _score_candidates,
    _velocity_action,
)
from tracer.response.closed_loop import expected_candidate_costs
from tracer.response.planning_support import aggregate_entity_costs
from tracer.response.unicycle_candidates import UnicycleLimits


def simulator_snapshot(env):
    agents = [env.robot, *env.humans]
    return np.asarray([
        [*agent.get_position(), *agent.get_velocity(), float(getattr(agent, "theta", 0.0)),
         *agent.get_goal_position(), float(agent.v_pref)]
        for agent in agents
    ], dtype=np.float64)


def true_speed_log_beliefs(humans, thresholds, num_modes, device):
    boundaries = np.asarray(thresholds, dtype=np.float64)
    if boundaries.shape != (num_modes - 1,) or not np.isfinite(boundaries).all() or np.any(np.diff(boundaries) <= 0):
        raise ValueError("true-mode audit requires the checkpoint's ordered training speed thresholds")
    speeds = np.asarray([human.v_pref for human in humans], dtype=np.float64)
    if not np.isfinite(speeds).all():
        raise ValueError("offline preferred speeds must be finite")
    labels = np.searchsorted(boundaries, speeds, side="right")
    result = torch.full((len(humans), num_modes), -torch.inf, device=device)
    result[torch.arange(len(humans), device=device), torch.as_tensor(labels, device=device)] = 0.0
    return result, labels


def replay_candidates(env, candidates, headings, action_type, *, limits=None, action_converter=_velocity_action):
    paths = np.asarray(candidates, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[-1] != 4 or min(paths.shape[:2]) < 1 or not np.isfinite(paths).all():
        raise ValueError("audit candidates must be finite candidate×horizon×4 trajectories")
    if headings is not None and (np.shape(headings) != paths.shape[:2] or not np.isfinite(headings).all()):
        raise ValueError("audit headings must align with the candidate horizon")
    if getattr(env.robot, "kinematics", "holonomic") == "unicycle" and headings is None:
        raise ValueError("unicycle oracle replay requires explicit candidate headings")
    if not np.isfinite(env.time_step) or env.time_step <= 0 or not env.humans:
        raise ValueError("oracle replay requires a positive timestep and at least one human")
    before = simulator_snapshot(env)
    before_time = float(env.global_time)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    horizon = paths.shape[1]

    def rollout(trajectory, branch_headings, check_limits):
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        clone = copy.deepcopy(env)
        snapshots, active, collision, wall_collision, projections = [], [], [], [], []
        motion_valid = True
        done = False
        try:
            for step, desired in enumerate(trajectory):
                if done:
                    padded = snapshots[-1].copy()
                    padded[:, 2:4] = 0.0
                    snapshots.append(padded)
                    active.append(False)
                    collision.append(False)
                    wall_collision.append(False)
                    projections.append(False)
                    continue
                target_heading = None if branch_headings is None else branch_headings[step]
                raw = action_converter(clone.robot, action_type, desired[2:], target_heading)
                if check_limits and limits is not None:
                    speed_change = abs(raw.v) - np.linalg.norm(clone.robot.get_velocity())
                    motion_valid &= bool(
                        abs(raw.r) <= limits.max_yaw_rate * clone.time_step + 1e-5
                        and abs(raw.v) <= limits.max_speed + 1e-5
                        and speed_change <= limits.acceleration * clone.time_step + 1e-5
                        and speed_change >= -limits.deceleration * clone.time_step - 1e-5
                    )
                position = np.asarray(clone.robot.get_position())
                previous_speed = np.linalg.norm(clone.robot.get_velocity())
                previous_heading = float(getattr(clone.robot, "theta", 0.0))
                action = clone.constrain_agent_action_exact(clone.robot, raw)
                _, _, done, info = clone.step(action)
                snapshot = simulator_snapshot(clone)
                if check_limits and limits is not None:
                    actual_speed = np.linalg.norm(snapshot[0, 2:4])
                    actual_rotation = (snapshot[0, 4] - previous_heading + np.pi) % (2 * np.pi) - np.pi
                    motion_valid &= bool(
                        abs(actual_rotation) <= limits.max_yaw_rate * clone.time_step + 1e-5
                        and actual_speed <= limits.max_speed + 1e-5
                        and actual_speed - previous_speed <= limits.acceleration * clone.time_step + 1e-5
                        and actual_speed - previous_speed >= -limits.deceleration * clone.time_step - 1e-5
                    )
                if not np.allclose(snapshot[0, :2], position + snapshot[0, 2:4] * clone.time_step, atol=2e-5, rtol=0):
                    raise ValueError("oracle replay violated the actual-velocity integration contract")
                snapshots.append(snapshot)
                active.append(True)
                collision.append(_info_value(info, "Collision") != 0.0)
                wall_collision.append(_info_value(info, "WallCollision") != 0.0)
                projections.append(not np.allclose(snapshot[0, 2:4], desired[2:], atol=1e-6, rtol=0))
        finally:
            clone.close()
        return {
            "snapshots": np.stack(snapshots), "active": np.asarray(active),
            "collision": np.asarray(collision), "wall_collision": np.asarray(wall_collision),
            "projected": np.asarray(projections), "motion_valid": np.asarray(motion_valid),
        }

    try:
        reference_path = np.zeros((horizon, 4), dtype=np.float64)
        reference_path[:, :2] = before[0, :2]
        reference = rollout(reference_path, np.full(horizon, before[0, 4]), False)
        branches = [rollout(path, None if headings is None else headings[index], True)
                    for index, path in enumerate(paths)]
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    if not np.array_equal(before, simulator_snapshot(env)) or before_time != float(env.global_time):
        raise ValueError("oracle replay mutated the live environment")
    result = {name: np.stack([branch[name] for branch in branches]) for name in branches[0]}
    result["reference_snapshots"] = reference["snapshots"]
    result["reference_active"] = reference["active"]
    result["initial_snapshot"] = before
    human_positions = result["snapshots"][:, :, 1:, :2]
    integrated = before[None, None, 1:, :2] + np.cumsum(result["snapshots"][:, :, 1:, 2:4] * env.time_step, axis=1)
    if not np.allclose(human_positions, integrated, atol=2e-5, rtol=0):
        raise ValueError("oracle human velocities do not integrate to observed positions")
    return result


def factual_candidate_costs(replay, dt, planning, penalty, goal_terminal_response=False):
    snapshots = replay["snapshots"]
    initial = replay["initial_snapshot"]
    paths = snapshots[:, :, 0, :4]
    human_velocities = snapshots[:, :, 1:, 2:4].transpose(2, 0, 1, 3)
    reference = replay["reference_snapshots"][:, 1:, 2:4].transpose(1, 0, 2)
    active = replay["active"].copy()
    if goal_terminal_response:
        active &= _goal_response_mask(paths, initial[0, 5:7], replay["robot_radius"])
    direction = torch.as_tensor(initial[0, 5:7] - initial[0, :2], dtype=torch.float32)
    costs = []
    for human_index, velocities in enumerate(human_velocities):
        origin = initial[human_index + 1, :2]
        relative = paths.copy()
        relative[..., :2] -= origin
        current = initial[0, :4].copy()
        current[:2] -= origin
        means = torch.as_tensor(velocities[:, None], dtype=torch.float32)
        references = torch.as_tensor(np.broadcast_to(reference[human_index], velocities.shape)[:, None].copy(), dtype=torch.float32)
        costs.append(expected_candidate_costs(
            means, torch.ones(1), torch.as_tensor(current, dtype=torch.float32).expand(len(paths), 1, 4),
            torch.as_tensor(relative[:, None], dtype=torch.float32), references, dt,
            replace(planning, violation_probability_alpha=None), torch.full_like(means, -100.0),
            task_direction=direction, response_valid_steps=torch.as_tensor(active),
        ))
    total, violation = aggregate_entity_costs(
        costs, planning.task_weight, planning.safety_weight, planning.human_weight,
        planning.smoothness_weight, planning.risk_weight, planning.crowd_aggregation,
    )
    penalty = np.asarray(penalty, dtype=np.float64)
    if penalty.shape != (len(paths),) or not np.isfinite(penalty).all():
        raise ValueError("oracle penalty must be finite and candidate-aligned")
    result = {"total": total.numpy().astype(np.float64) + penalty, "risk": violation.numpy()}
    for name in ("task", "safety", "human", "smoothness"):
        values = np.stack([cost[name].numpy() for cost in costs])
        result[name] = values.max(axis=0) if name == "safety" and planning.crowd_aggregation != "mean" else values.mean(axis=0)
    return result


def same_executed_prefix(snapshots, selected, steps):
    return np.all(np.isclose(snapshots[:, :steps, 0, :5], snapshots[selected, :steps, 0, :5],
                             atol=1e-6, rtol=0), axis=(1, 2))


def ranking_information(scores, factual, snapshots, eligible, selected, execute_steps):
    support = np.asarray(eligible, dtype=bool) & np.isfinite(factual)
    if not support.any():
        raise ValueError("information audit needs nonempty matched candidate support")
    oracle = int(np.flatnonzero(support)[np.argmin(factual[support])])
    result = {"oracle_selected": oracle, "eligible_candidates": int(support.sum()),
              "oracle_cost": float(factual[oracle])}
    uniform_prefix = same_executed_prefix(snapshots, selected["uniform"], execute_steps)
    result["oracle_action_change_vs_uniform"] = int(not uniform_prefix[oracle])
    for name, index in selected.items():
        result[f"{name}_selected"] = int(index)
        result[f"{name}_supported"] = bool(support[index])
        result[f"{name}_cost"] = float(factual[index])
        result[f"{name}_action_change_vs_uniform"] = int(not uniform_prefix[index])
        result[f"{name}_regret"] = float(factual[index] - factual[oracle]) if support[index] else None
        same_prefix = same_executed_prefix(snapshots, index, execute_steps) & support
        result[f"{name}_prefix_regret"] = float(factual[same_prefix].min() - factual[oracle]) if same_prefix.any() else None
        valid = support & np.isfinite(scores[name])
        correlation, _ = _rank_correlation(np.asarray(scores[name])[valid], factual[valid])
        result[f"{name}_spearman"] = float(correlation) if np.isfinite(correlation) else None
    return result


class DecisionInformationAudit:
    def __init__(self, output, thresholds, stride=1, comparison_model=None):
        if stride < 1:
            raise ValueError("audit stride must be positive")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.thresholds = thresholds
        self.stride = stride
        self.records = []
        self.pending = None
        self.comparison_model = comparison_model

    @torch.no_grad()
    def observe(self, *, env, model, histories, current, candidates, global_candidates, headings,
                static_obstacles, beliefs, args, selected, decision_scores, diagnostics,
                cached_predictions, candidate_bias, braking_safe, fallback_progress_active,
                sage_entity_predictions, case, window, seed, policy):
        if self.pending is not None:
            raise ValueError("previous oracle prefix has not been checked against live execution")
        if window % self.stride:
            return
        if args.hard_risk_filter or float(getattr(args, "belief_cost_leverage", 1.0)) != 1.0:
            raise ValueError("information audit requires fixed belief-independent safety masks and leverage one")
        from crowd_sim_plus.envs.utils.action import ActionXY

        hidden_beliefs, labels = true_speed_log_beliefs(env.humans, self.thresholds, beliefs.shape[-1], args.device)
        uniform = torch.full_like(beliefs, -np.log(beliefs.shape[-1]))
        comparison = {}
        interventions = [("uniform", uniform, model, cached_predictions),
                         ("true_mode", hidden_beliefs, model, cached_predictions)]
        if self.comparison_model is not None:
            interventions.append(("candidate", beliefs, self.comparison_model, None))
        for name, intervention, response_model, prediction_cache in interventions:
            comparison[name] = _score_candidates(
                response_model, histories, current, candidates, global_candidates, env.robot, env.humans,
                static_obstacles, intervention, args,
                fallback_candidate=1 if args.candidate_profile == "orca-augmented" else None,
                executor_safe=_executor_safe_mask(env, env.robot, ActionXY, global_candidates, candidate_headings=headings),
                sage_entity_predictions=sage_entity_predictions, candidate_bias=candidate_bias,
                braking_safe=braking_safe, cached_predictions=prediction_cache,
                fallback_progress_active=fallback_progress_active,
            )
        scores = {"persistent": decision_scores.numpy(), **{name: value[6].numpy() for name, value in comparison.items()}}
        selections = {"persistent": selected, **{name: value[0] for name, value in comparison.items()}}
        horizon = int(model.config.horizon)
        limits = UnicycleLimits.from_sicnav_config(Path(args.motion_policy_config)) if args.motion_profile == "sicnav-unicycle" else None
        replay = replay_candidates(env, global_candidates[:, :horizon], None if headings is None else headings[:, :horizon],
                                   ActionXY, limits=limits)
        replay["robot_radius"] = env.robot.radius
        self.pending = replay["snapshots"][selected, :args.execute_steps].copy()
        penalty = np.asarray(diagnostics["observable_penalty"]) + np.asarray(candidate_bias)
        factual = factual_candidate_costs(replay, args.env_dt, _planning_cost_config(args), penalty,
                                          getattr(args, "goal_terminal_response", False))
        support = np.logical_and.reduce([np.isfinite(value) for value in scores.values()]) & replay["motion_valid"]
        record = {"case": int(case), "window": int(window), "seed": int(seed), "policy": policy,
                  "candidate_count": len(global_candidates), "motion_valid_fraction": float(replay["motion_valid"].mean()),
                  "projected_fraction": float(replay["projected"].any(axis=1).mean()),
                  "true_modes_offline_only": labels.tolist(),
                  "decision_mode_accuracy": float((beliefs.argmax(-1).cpu().numpy() == labels).mean()),
                  "decision_true_mode_probability": float(beliefs.exp()[torch.arange(len(labels), device=beliefs.device),
                                                                          torch.as_tensor(labels, device=beliefs.device)].mean()),
                  "status": "matched" if support.any() else "no_common_support"}
        if any(bool(value[4]) for value in comparison.values()):
            record["status"] = "fallback_not_comparable"
        if record["status"] == "matched":
            record.update(ranking_information(scores, factual["total"], replay["snapshots"], support, selections, args.execute_steps))
            for name, index in {**selections, "oracle": record["oracle_selected"]}.items():
                record[f"{name}_horizon_collision"] = bool(replay["collision"][index].any())
                record[f"{name}_horizon_wall_collision"] = bool(replay["wall_collision"][index].any())
        path = self.output / f"seed{seed}_case{case}_window{window}.npz"
        if path.exists():
            raise ValueError("oracle snapshot already exists")
        np.savez_compressed(path, **replay, **{f"factual_{name}": value for name, value in factual.items()},
                            **{f"score_{name}": value for name, value in scores.items()}, eligible=support,
                            planned_candidates=global_candidates, planned_headings=headings)
        self.records.append(record)

    def verify_execution(self, env, steps):
        if self.pending is None:
            return
        if not 1 <= steps <= len(self.pending) or not np.allclose(simulator_snapshot(env), self.pending[steps - 1], atol=2e-5, rtol=0):
            raise ValueError("cloned selected prefix differs from the actual robot/human transition")
        self.records[-1]["live_prefix_verified"] = True
        with (self.output / "windows.jsonl").open("a") as stream:
            stream.write(json.dumps(self.records[-1], allow_nan=False) + "\n")
        self.pending = None
