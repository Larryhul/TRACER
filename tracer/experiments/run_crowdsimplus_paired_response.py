"""Validate action-conditioned response mechanisms with paired CrowdSimPlus resets.

Each official test case is reset independently for every robot-action branch.
CrowdSimPlus derives the reset RNG from ``test_case``; the runner verifies the
full initial state hash before accepting a paired group.  The human policy and
all scene attributes therefore remain fixed while only the robot action branch
changes.

This is an external single-robot response experiment.  It supports H1-style
action-conditioned response evidence, but it does not support multi-robot
interaction or real-human causal claims.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tracer.data.heterogeneous_scene import HeterogeneousSceneBatch
from tracer.experiments.run_sicnav_tracer_loop import _new_episode
from tracer.response.data import (
    build_entity_response_batch,
    shuffle_robot_candidates_within_groups,
)
from tracer.response.experiment import train_epoch
from tracer.response.metrics import response_metrics
from tracer.response.models import ResponseModelConfig, TeamResponsePredictor


BRANCHES = ("reference", "forward", "left", "right")
VARIANTS = ("history", "nearest", "additive", "joint")


@dataclass
class BranchRollout:
    case: int
    branch: str
    initial_hash: str
    initial_robot_history: np.ndarray
    initial_human_history: np.ndarray
    robot_positions: np.ndarray
    human_positions: np.ndarray
    robot_actions: np.ndarray
    human_velocities: np.ndarray
    human_params: np.ndarray
    robot_goal: np.ndarray
    static_obstacle_count: int


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _state_hash(robot: Any, humans: list[Any], static_obstacles: list[Any]) -> str:
    def full_state_values(agent: Any) -> list[float]:
        state = agent.get_full_state(original=True)
        return [
            float(getattr(state, name))
            for name in ("px", "py", "vx", "vy", "radius", "gx", "gy", "v_pref", "theta")
        ]

    values: list[float] = []
    values.extend(full_state_values(robot))
    for human in humans:
        values.extend(full_state_values(human))
    values.extend(np.asarray(static_obstacles, dtype=np.float64).reshape(-1))
    return _sha256_bytes(np.asarray(values, dtype=np.float64).tobytes())


def _state_history(
    env: Any,
    agent: Any,
    history: int,
    dt: float,
) -> np.ndarray:
    observed: list[np.ndarray] = []
    for state in getattr(env, "states", []):
        if agent is env.robot:
            full = state[0]
        else:
            index = env.humans.index(agent)
            full = state[1][index]
        observed.append(np.asarray(full.position, dtype=np.float32))
    current = np.asarray(agent.get_position(), dtype=np.float32)
    velocity = np.asarray(agent.get_velocity(), dtype=np.float32)
    while len(observed) < history:
        offset = history - len(observed)
        observed.insert(0, current - velocity * dt * offset)
    return np.asarray(observed[-history:], dtype=np.float32)


def _branch_action(robot: Any, branch: str) -> np.ndarray:
    if branch not in BRANCHES:
        raise ValueError(f"unknown action branch {branch!r}")
    position = np.asarray(robot.get_position(), dtype=np.float32)
    goal = np.asarray(robot.get_goal_position(), dtype=np.float32)
    direction = goal - position
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    tangent = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    if branch == "reference":
        return np.zeros(2, dtype=np.float32)
    if branch == "forward":
        return direction * min(float(robot.v_pref), 0.65)
    signed = 1.0 if branch == "left" else -1.0
    diagonal = 0.78 * direction + signed * 0.626 * tangent
    diagonal /= max(float(np.linalg.norm(diagonal)), 1e-6)
    return diagonal * min(float(robot.v_pref), 0.65)


def _run_branch(
    external_root: Path,
    env_config_path: Path,
    case: int,
    branch: str,
    history: int,
    horizon: int,
    human_num: int,
) -> BranchRollout:
    env, robot, _, static_obstacles, ActionXY, _, _ = _new_episode(
        external_root, case, env_config_path, human_num
    )
    humans = list(env.humans)
    initial_hash = _state_hash(robot, humans, static_obstacles)
    initial_robot_history = _state_history(env, robot, history, env.time_step)
    initial_human_history = np.stack(
        [_state_history(env, human, history, env.time_step) for human in humans]
    )
    robot_positions = np.zeros((horizon + 1, 2), dtype=np.float32)
    human_positions = np.zeros((horizon + 1, human_num, 2), dtype=np.float32)
    robot_actions = np.zeros((horizon, 2), dtype=np.float32)
    human_velocities = np.zeros((horizon, human_num, 2), dtype=np.float32)
    robot_positions[0] = np.asarray(robot.get_position(), dtype=np.float32)
    human_positions[0] = np.asarray([human.get_position() for human in humans], dtype=np.float32)
    action = _branch_action(robot, branch)
    for step in range(horizon):
        robot_actions[step] = action
        _, _, _, _ = env.step(ActionXY(float(action[0]), float(action[1])))
        robot_positions[step + 1] = np.asarray(robot.get_position(), dtype=np.float32)
        human_positions[step + 1] = np.asarray(
            [human.get_position() for human in humans], dtype=np.float32
        )
        human_velocities[step] = np.asarray(
            [human.get_velocity() for human in humans], dtype=np.float32
        )
    human_params = np.asarray(
        [
            [
                float(human.radius),
                float(human.v_pref),
                float(human.final_gx),
                float(human.final_gy),
            ]
            for human in humans
        ],
        dtype=np.float32,
    )
    return BranchRollout(
        case=case,
        branch=branch,
        initial_hash=initial_hash,
        initial_robot_history=initial_robot_history,
        initial_human_history=initial_human_history,
        robot_positions=robot_positions,
        human_positions=human_positions,
        robot_actions=robot_actions,
        human_velocities=human_velocities,
        human_params=human_params,
        robot_goal=np.asarray(robot.get_goal_position(), dtype=np.float32),
        static_obstacle_count=len(static_obstacles),
    )


def _collect(
    external_root: Path,
    env_config_path: Path,
    cases: list[int],
    history: int,
    horizon: int,
    human_num: int,
) -> tuple[list[BranchRollout], dict[str, Any]]:
    rollouts: list[BranchRollout] = []
    replay_groups: list[dict[str, Any]] = []
    for case in cases:
        group = [
            _run_branch(
                external_root,
                env_config_path,
                case,
                branch,
                history,
                horizon,
                human_num,
            )
            for branch in BRANCHES
        ]
        hashes = [rollout.initial_hash for rollout in group]
        same_initial = len(set(hashes)) == 1
        if not same_initial:
            raise RuntimeError(f"paired reset mismatch in case {case}: {hashes}")
        params = np.stack([rollout.human_params for rollout in group])
        params_stable = bool(np.allclose(params, params[0:1], atol=0.0, rtol=0.0))
        replay_groups.append(
            {
                "case": int(case),
                "initial_hash": hashes[0],
                "branches": list(BRANCHES),
                "same_initial_state": same_initial,
                "stable_human_attributes_across_branches": params_stable,
            }
        )
        rollouts.extend(group)
    return rollouts, {
        "cases": cases,
        "branches": list(BRANCHES),
        "groups": replay_groups,
        "paired_groups": len(replay_groups),
        "replay_equal_groups": sum(
            int(item["same_initial_state"]) for item in replay_groups
        ),
        "stable_attribute_groups": sum(
            int(item["stable_human_attributes_across_branches"])
            for item in replay_groups
        ),
    }


def _write_rollout_csv(rollouts: list[BranchRollout], path: Path) -> None:
    fields = [
        "case",
        "branch",
        "initial_hash",
        "step",
        "human_id",
        "robot_x",
        "robot_y",
        "robot_action_x",
        "robot_action_y",
        "human_x",
        "human_y",
        "human_vx",
        "human_vy",
        "human_radius",
        "human_v_pref",
        "human_goal_x",
        "human_goal_y",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rollout in rollouts:
            for step in range(len(rollout.robot_positions)):
                action = (
                    rollout.robot_actions[min(step, len(rollout.robot_actions) - 1)]
                    if rollout.robot_actions.size
                    else np.zeros(2)
                )
                for human_id in range(rollout.human_positions.shape[1]):
                    velocity = (
                        rollout.human_velocities[min(step, len(rollout.human_velocities) - 1), human_id]
                        if step > 0
                        else np.zeros(2)
                    )
                    params = rollout.human_params[human_id]
                    writer.writerow(
                        {
                            "case": rollout.case,
                            "branch": rollout.branch,
                            "initial_hash": rollout.initial_hash,
                            "step": step,
                            "human_id": human_id,
                            "robot_x": float(rollout.robot_positions[step, 0]),
                            "robot_y": float(rollout.robot_positions[step, 1]),
                            "robot_action_x": float(action[0]),
                            "robot_action_y": float(action[1]),
                            "human_x": float(rollout.human_positions[step, human_id, 0]),
                            "human_y": float(rollout.human_positions[step, human_id, 1]),
                            "human_vx": float(velocity[0]),
                            "human_vy": float(velocity[1]),
                            "human_radius": float(params[0]),
                            "human_v_pref": float(params[1]),
                            "human_goal_x": float(params[2]),
                            "human_goal_y": float(params[3]),
                        }
                    )


def _pad_history(history: np.ndarray, target: int) -> np.ndarray:
    if history.shape[0] >= target:
        return history[-target:]
    return np.concatenate(
        [np.repeat(history[:1], target - history.shape[0], axis=0), history], axis=0
    )


def _rollout_scene(rollout: BranchRollout, dt: float) -> HeterogeneousSceneBatch:
    entities = rollout.human_positions.shape[1]
    agents = entities + 1
    history = rollout.initial_robot_history.shape[0]
    horizon = rollout.robot_actions.shape[0]
    agent_history = np.zeros((agents, history, 2), dtype=np.float32)
    agent_history[0] = _pad_history(rollout.initial_robot_history, history)
    agent_history[1:] = rollout.initial_human_history
    future = np.zeros((agents, horizon, 2), dtype=np.float32)
    future[0] = np.diff(rollout.robot_positions, axis=0) / float(dt)
    # Human velocities are emitted by the official step; recompute from positions
    # for a single consistent target definition at the environment timestep.
    future[1:] = np.transpose(
        np.diff(rollout.human_positions, axis=0) / float(dt), (1, 0, 2)
    )
    current_pos = np.concatenate(
        [rollout.robot_positions[0:1], rollout.human_positions[0]], axis=0
    ).astype(np.float32)
    roles = np.concatenate([np.zeros(1, dtype=np.int64), np.ones(entities, dtype=np.int64)])
    agent_mask = np.ones(agents, dtype=bool)
    robot_mask = np.zeros(agents, dtype=bool)
    robot_mask[0] = True
    entity_mask = ~robot_mask
    loss_mask = agent_mask.copy()
    goals = np.zeros((agents, 2), dtype=np.float32)
    goals[0] = rollout.robot_goal
    goal_mask = np.zeros(agents, dtype=bool)
    goal_mask[0] = True
    return HeterogeneousSceneBatch(
        history=torch.from_numpy(agent_history[None]),
        future=torch.from_numpy(future[None]),
        current_pos=torch.from_numpy(current_pos[None]),
        roles=torch.from_numpy(roles[None]),
        agent_mask=torch.from_numpy(agent_mask[None]),
        robot_mask=torch.from_numpy(robot_mask[None]),
        entity_mask=torch.from_numpy(entity_mask[None]),
        loss_mask=torch.from_numpy(loss_mask[None]),
        goals=torch.from_numpy(goals[None]),
        goal_mask=torch.from_numpy(goal_mask[None]),
        dt=float(dt),
    )


class PairedSceneDataset(Dataset[HeterogeneousSceneBatch]):
    def __init__(self, rollouts: list[BranchRollout], dt: float) -> None:
        self.scenes = [_rollout_scene(rollout, dt) for rollout in rollouts]

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> HeterogeneousSceneBatch:
        return self.scenes[index]


def _collate_scenes(samples: list[HeterogeneousSceneBatch]) -> HeterogeneousSceneBatch:
    if not samples:
        raise ValueError("cannot collate an empty scene list")
    return HeterogeneousSceneBatch(
        history=torch.cat([sample.history for sample in samples], dim=0),
        future=torch.cat([sample.future for sample in samples], dim=0),
        current_pos=torch.cat([sample.current_pos for sample in samples], dim=0),
        roles=torch.cat([sample.roles for sample in samples], dim=0),
        agent_mask=torch.cat([sample.agent_mask for sample in samples], dim=0),
        robot_mask=torch.cat([sample.robot_mask for sample in samples], dim=0),
        entity_mask=torch.cat([sample.entity_mask for sample in samples], dim=0),
        loss_mask=torch.cat([sample.loss_mask for sample in samples], dim=0),
        goals=torch.cat([sample.goals for sample in samples], dim=0),
        goal_mask=torch.cat([sample.goal_mask for sample in samples], dim=0),
        dt=samples[0].dt,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    dataset: PairedSceneDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=_collate_scenes,
        num_workers=0,
    )


def _train_variant(
    variant: str,
    train_dataset: PairedSceneDataset,
    val_dataset: PairedSceneDataset,
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
) -> tuple[TeamResponsePredictor, dict[str, float]]:
    _set_seed(seed)
    config = ResponseModelConfig(
        history=args.history,
        horizon=args.horizon,
        num_robots=1,
        hidden_dim=args.hidden_dim,
        num_heads=4,
        dropout=args.dropout,
        architecture="tracer",
        graph_layers=2,
        dt=args.dt,
    )
    model = TeamResponsePredictor(variant, config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_loader = _loader(train_dataset, args.batch_size, True, seed + 17)
    val_loader = _loader(val_dataset, args.eval_batch_size, False, seed + 19)
    best_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            num_robots=1,
            interaction_radius=args.interaction_radius,
        )
        metrics = _evaluate_model(model, val_loader, args, include_shuffle=False)[variant]
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_metrics = {**metrics, "train_loss": train_loss, "epoch": float(epoch)}
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 4, 1) == 0:
            print(
                f"seed={seed} variant={variant} epoch={epoch} "
                f"train_nll={train_loss:.6f} val_nll={metrics['nll']:.6f} "
                f"val_ade={metrics['ade']:.6f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError(f"no checkpoint selected for {variant}")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": vars(config),
            "variant": variant,
            "seed": seed,
            "best_metrics": best_metrics,
        },
        output_dir / f"{variant}_seed{seed}.pt",
    )
    return model.eval(), best_metrics


def _load_variant(
    variant: str,
    checkpoint_path: Path,
    device: str,
) -> TeamResponsePredictor:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ResponseModelConfig(**checkpoint["model_config"])
    model = TeamResponsePredictor(variant, config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _mean_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in values for key in item})
    output: dict[str, float] = {}
    for key in keys:
        numbers = np.asarray([item[key] for item in values if math.isfinite(item[key])])
        output[key] = float(numbers.mean()) if numbers.size else float("nan")
        output[f"{key}_std"] = float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0
    return output


@torch.no_grad()
def _evaluate_model(
    model: TeamResponsePredictor,
    loader: DataLoader,
    args: argparse.Namespace,
    include_shuffle: bool,
) -> dict[str, dict[str, float]]:
    model.eval()
    names = [model.variant]
    if model.variant == "joint":
        names.extend(["joint-without-pair", "shuffled-joint"] if include_shuffle else ["joint-without-pair"])
    totals: dict[str, list[float]] = {
        name: [] for name in names
    }
    effect_errors: dict[str, list[float]] = {name: [] for name in names}
    for scene in loader:
        response = build_entity_response_batch(
            scene.to(args.device), num_robots=1, interaction_radius=args.interaction_radius
        ).to(args.device)
        predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        output = model(response)
        predictions[model.variant] = (output["mean"], output["log_std"])
        if model.variant == "joint":
            without_pair = model(response, include_pair_interactions=False)
            predictions["joint-without-pair"] = (
                without_pair["mean"],
                without_pair["log_std"],
            )
            if include_shuffle:
                shuffled_response = shuffle_robot_candidates_within_groups(
                    response,
                    group_size=scene.batch_size * 0 + len(BRANCHES) * (scene.num_agents - 1),
                    branch_size=scene.num_agents - 1,
                )
                shuffled = model(shuffled_response)
                predictions["shuffled-joint"] = (shuffled["mean"], shuffled["log_std"])
        target = response.target_future_vel
        for name, (mean, log_std) in predictions.items():
            metrics = response_metrics(mean, log_std, target, response.dt)
            totals[name].extend(
                float(metrics[key].mean().detach().cpu()) for key in ("nll", "ade", "fde")
            )
            batch_scenes = scene.batch_size
            entities = scene.num_agents - 1
            if batch_scenes % len(BRANCHES) != 0:
                raise ValueError("evaluation batch must contain complete branch groups")
            predicted_position = torch.cumsum(mean * response.dt, dim=1).reshape(
                batch_scenes, entities, args.horizon, 2
            )
            target_position = torch.cumsum(target * response.dt, dim=1).reshape(
                batch_scenes, entities, args.horizon, 2
            )
            predicted_position = predicted_position.reshape(
                -1, len(BRANCHES), entities, args.horizon, 2
            )
            target_position = target_position.reshape(
                -1, len(BRANCHES), entities, args.horizon, 2
            )
            predicted_effect = predicted_position[:, 1:] - predicted_position[:, :1]
            target_effect = target_position[:, 1:] - target_position[:, :1]
            error = torch.linalg.vector_norm(
                predicted_effect[:, :, :, -1] - target_effect[:, :, :, -1], dim=-1
            ).mean()
            effect_errors[name].append(float(error.detach().cpu()))
    results: dict[str, dict[str, float]] = {}
    for name in names:
        chunks = totals[name]
        if not chunks:
            continue
        results[name] = {
            "nll": float(np.mean(chunks[0::3])),
            "ade": float(np.mean(chunks[1::3])),
            "fde": float(np.mean(chunks[2::3])),
            "action_effect_fde_error": float(np.mean(effect_errors[name])),
        }
    return results


def _response_effect_summary(
    rollouts: list[BranchRollout],
    dt: float,
) -> dict[str, Any]:
    by_case = {case: {rollout.branch: rollout for rollout in rollouts if rollout.case == case} for case in sorted({r.case for r in rollouts})}
    values: dict[str, list[float]] = {branch: [] for branch in BRANCHES if branch != "reference"}
    action_norms: dict[str, list[float]] = {branch: [] for branch in BRANCHES if branch != "reference"}
    for branches in by_case.values():
        reference = branches["reference"]
        for branch in values:
            rollout = branches[branch]
            displacement_gap = rollout.human_positions[-1] - reference.human_positions[-1]
            values[branch].extend(np.linalg.norm(displacement_gap, axis=-1).tolist())
            action_norms[branch].append(float(np.linalg.norm(rollout.robot_actions[0])))
    return {
        "paired_action_response_gap_final": {
            branch: {
                "mean": float(np.mean(items)) if items else float("nan"),
                "std": float(np.std(items)) if items else float("nan"),
                "count": len(items),
            }
            for branch, items in values.items()
        },
        "branch_action_norm": {
            branch: float(np.mean(items)) for branch, items in action_norms.items()
        },
        "dt": dt,
        "interpretation": "difference in simulated human displacement under paired robot-action branches, not a causal real-human estimate",
    }


def _identity_persistence_audit(
    rollouts: list[BranchRollout],
    replay_summary: dict[str, Any],
) -> dict[str, Any]:
    """Separate stable simulator fields from an unobserved behavioral mode."""

    stable_groups = [
        item
        for item in replay_summary["groups"]
        if item["stable_human_attributes_across_branches"]
    ]
    return {
        "status": "UNSUPPORTED_FOR_H2_MODE_CLAIM",
        "stable_simulator_attributes": [
            "human.radius",
            "human.v_pref",
            "human.final_gx/final_gy",
            "human.policy.name",
        ],
        "stable_attribute_groups": len(stable_groups),
        "paired_groups": replay_summary["paired_groups"],
        "hidden_behavior_mode_ground_truth": False,
        "persistent_identity_observation": "human list indices remain stable within each simulator episode, but CrowdSimPlus exposes no latent discrete response mode or mode switch process",
        "reason": "ORCA+ is one fixed simulator policy; radius and preferred speed are physical attributes, not validated psychological response modes. H2 reset/recurrent/Bayesian/UKF mode-persistence metrics would therefore be an unsupported construct on this environment.",
        "interpretation": "stable attributes are recorded for reproducibility only and are not used to claim identity persistence",
        "rollout_count": len(rollouts),
    }


def _write_rollout_npz(rollouts: list[BranchRollout], path: Path) -> None:
    """Persist numeric branch arrays so the learned comparison is replayable."""

    np.savez_compressed(
        path,
        case=np.asarray([rollout.case for rollout in rollouts], dtype=np.int64),
        branch=np.asarray(
            [BRANCHES.index(rollout.branch) for rollout in rollouts], dtype=np.int64
        ),
        initial_hash=np.asarray([rollout.initial_hash for rollout in rollouts]),
        initial_robot_history=np.stack(
            [rollout.initial_robot_history for rollout in rollouts]
        ),
        initial_human_history=np.stack(
            [rollout.initial_human_history for rollout in rollouts]
        ),
        robot_positions=np.stack([rollout.robot_positions for rollout in rollouts]),
        human_positions=np.stack([rollout.human_positions for rollout in rollouts]),
        robot_actions=np.stack([rollout.robot_actions for rollout in rollouts]),
        human_params=np.stack([rollout.human_params for rollout in rollouts]),
        robot_goal=np.stack([rollout.robot_goal for rollout in rollouts]),
    )


def _write_report(
    output: Path,
    metadata: dict[str, Any],
    replay_summary: dict[str, Any],
    effect_summary: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# CrowdSimPlus paired-intervention response experiment",
        "",
        f"Commit `{metadata['git']['commit']}`; clean at execution `{not metadata['git']['dirty']}`; official source `{metadata['external_repository']['commit']}`.",
        "",
        "## Protocol",
        "",
        "Every official test case is reset once per branch with the same `test_case`. The runner hashes robot/human full states and static obstacles and rejects a case if branches do not start identically. Only the robot action branch changes; CrowdSimPlus then computes ORCA+ human actions and advances the same simulator timestep.",
        f"Paired groups: `{replay_summary['paired_groups']}`; exact replay groups: `{replay_summary['replay_equal_groups']}`; stable simulator attributes across branches: `{replay_summary['stable_attribute_groups']}`.",
        "",
        "## Direct response signal",
        "",
    ]
    for branch, item in effect_summary["paired_action_response_gap_final"].items():
        lines.append(
            f"- `{branch}` vs `reference`: final human displacement gap `{item['mean']:.6f}` +/- `{item['std']:.6f}` m over `{item['count']}` entity pairs."
        )
    lines.extend(["", "## Predictive comparison", ""])
    for method, item in summary.get("methods", {}).items():
        lines.append(
            f"- `{method}`: NLL `{item['nll']:.6f}` +/- `{item['nll_std']:.6f}`, ADE `{item['ade']:.6f}`, FDE `{item['fde']:.6f}`, action-effect FDE error `{item['action_effect_fde_error']:.6f}`."
        )
    lines.extend(
        [
            "",
        "## Claim boundary",
        "",
        "The separate identity audit marks H2 mode-persistence evaluation as unsupported: the simulator keeps physical attributes fixed but exposes no latent behavioral mode or mode-switch process.",
        "This is third-party, single-robot, paired action-conditioned response prediction in official CrowdSimPlus with ORCA+ simulated humans. It supports an external response-mechanism check and does not support multi-robot joint-interaction H3, real-human causality, or long-term identity persistence. The `joint` model is evaluated with one robot, so its pair residual is structurally inactive; a positive joint-interaction claim requires a multi-robot environment such as SocialGym2.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    external_root = (root / args.external_root).resolve()
    env_config = (root / args.env_config).resolve()
    git = _git_state(root)
    cases = list(range(args.case_start, args.case_start + args.cases))
    started = time.perf_counter()
    rollouts, replay_summary = _collect(
        external_root,
        env_config,
        cases,
        args.history,
        args.horizon,
        args.human_num,
    )
    collection_dir = output / "collection"
    collection_dir.mkdir(parents=True, exist_ok=True)
    _write_rollout_csv(rollouts, collection_dir / "paired_rollouts.csv")
    _write_rollout_npz(rollouts, collection_dir / "paired_rollouts.npz")
    effect_summary = _response_effect_summary(rollouts, args.dt)
    identity_audit = _identity_persistence_audit(rollouts, replay_summary)
    (collection_dir / "replay_summary.json").write_text(
        json.dumps(replay_summary, indent=2) + "\n", encoding="utf-8"
    )
    (collection_dir / "response_effect_summary.json").write_text(
        json.dumps(effect_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (collection_dir / "identity_persistence_audit.json").write_text(
        json.dumps(identity_audit, indent=2) + "\n", encoding="utf-8"
    )

    by_case = {case: [r for r in rollouts if r.case == case] for case in cases}
    train_cases = [case for case in cases if case < args.val_start]
    val_cases = [case for case in cases if args.val_start <= case < args.test_start]
    test_cases = [case for case in cases if case >= args.test_start]
    if not train_cases or not val_cases or not test_cases:
        raise ValueError("case range must contain non-empty train, validation, and test splits")
    train_rollouts = [rollout for case in train_cases for rollout in by_case[case]]
    val_rollouts = [rollout for case in val_cases for rollout in by_case[case]]
    test_rollouts = [rollout for case in test_cases for rollout in by_case[case]]
    _write_rollout_npz(train_rollouts, collection_dir / "paired_train.npz")
    _write_rollout_npz(val_rollouts, collection_dir / "paired_val.npz")
    _write_rollout_npz(test_rollouts, collection_dir / "paired_test.npz")
    train_dataset = PairedSceneDataset(train_rollouts, args.dt)
    val_dataset = PairedSceneDataset(val_rollouts, args.dt)
    test_dataset = PairedSceneDataset(test_rollouts, args.dt)
    print(
        f"paired CrowdsimPlus cases train={len(train_cases)} val={len(val_cases)} test={len(test_cases)} "
        f"scenes={len(rollouts)} device={args.device} cuda={torch.cuda.is_available()}",
        flush=True,
    )

    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    checkpoint_dir = output / "checkpoints"
    for seed in args.model_seeds:
        per_seed[str(seed)] = {}
        for variant in VARIANTS:
            checkpoint_path = checkpoint_dir / f"{variant}_seed{seed}.pt"
            if args.resume_existing and checkpoint_path.is_file():
                model = _load_variant(variant, checkpoint_path, args.device)
                print(
                    f"seed={seed} variant={variant} checkpoint=resume",
                    flush=True,
                )
            else:
                model, _ = _train_variant(
                    variant,
                    train_dataset,
                    val_dataset,
                    args,
                    seed,
                    checkpoint_dir,
                )
            test_loader = _loader(test_dataset, args.eval_batch_size, False, seed + 23)
            metrics = _evaluate_model(
                model, test_loader, args, include_shuffle=variant == "joint"
            )
            for method, values in metrics.items():
                per_seed[str(seed)][method] = values

    method_values: dict[str, list[dict[str, float]]] = {}
    raw_rows: list[dict[str, Any]] = []
    for seed, methods in per_seed.items():
        for method, values in methods.items():
            method_values.setdefault(method, []).append(values)
            raw_rows.append({"model_seed": seed, "method": method, **values})
    summary = {
        "methods": {method: _mean_metrics(values) for method, values in method_values.items()},
        "model_seeds": args.model_seeds,
        "splits": {
            "train_cases": train_cases,
            "val_cases": val_cases,
            "test_cases": test_cases,
        },
    }
    with (output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    try:
        external_path = str(external_root.relative_to(root))
    except ValueError:
        external_path = str(external_root)
    metadata = {
        "protocol": "crowdsimplus-paired-intervention-response-v1",
        "claim_boundary": "third-party single-robot action-conditioned simulated response prediction; no multi-robot joint interaction, real-human causality, or long-term identity persistence",
        "git": git,
        "external_repository": {
            "path": external_path,
            "commit": subprocess.check_output(
                ["git", "-C", str(external_root), "rev-parse", "HEAD"], text=True
            ).strip(),
        },
        "device": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parameters": vars(args),
        "records": len(rollouts),
        "runtime_seconds": time.perf_counter() - started,
        "replay_summary": replay_summary,
        "identity_persistence_audit": identity_audit,
        "supported_baselines": list(VARIANTS),
        "unsupported_external_baselines": [
            "ScePT/2019-style implementation not claimed without a matched official implementation",
            "multi-robot joint interaction unavailable in this CrowdSimPlus protocol",
        ],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "replay": replay_summary,
                "response_effect": effect_summary,
                "identity_persistence": identity_audit,
                **summary,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output, metadata, replay_summary, effect_summary, summary)
    artifacts = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("metadata.json", "summary.json", "test_metrics.csv", "report.md")
    }
    manifest = {
        "status": "COMPLETED",
        "protocol": metadata["protocol"],
        "records": len(rollouts),
        "artifacts": artifacts,
        "claim_boundary": metadata["claim_boundary"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "records": len(rollouts), "output": str(output)}, indent=2), flush=True)
    return {"metadata": metadata, "summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", default="raw_data/external/source_repos/sicnav_official_filtered")
    parser.add_argument("--env-config", default="raw_data/external/source_repos/sicnav_official_filtered/sicnav/configs/env.config")
    parser.add_argument("--output-dir", default="experiments/external_mechanism/crowdsimplus_paired_h1_20260904")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--val-start", type=int, default=40)
    parser.add_argument("--test-start", type=int, default=48)
    parser.add_argument("--human-num", type=int, default=3)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--interaction-radius", type=float, default=3.0)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[2027, 2031, 2039])
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse model checkpoints from an interrupted run after rechecking the paired collection.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
