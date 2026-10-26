"""Collect paired multi-robot response interventions in official SocialGym2.

SocialGym2 has a discrete GoAlone/Halt action interface and no persistent
human IDs in its ROS response.  This runner therefore uses one human, spatial
matching within each short branch, and reports only action-conditioned response
differences.  It verifies that all branches start from the same ROS reset state
before writing a paired group.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sage.data.heterogeneous_scene import HeterogeneousSceneBatch
from sage.experiments.run_socialgym2_tracer_loop import (
    _build_env,
    _capture_observations,
    _scene_from_responses,
)
from sage.response.data import (
    build_entity_response_batch,
    shuffle_robot_candidates_within_groups,
)
from sage.response.experiment import train_epoch
from sage.response.metrics import response_metrics
from sage.response.models import ResponseModelConfig, TeamResponsePredictor, build_response_predictor
from sage.response.external_baselines import external_method_metadata


BRANCHES = {
    "".join(map(str, action)): action
    for action in itertools.product((0, 1), repeat=3)
}
BRANCH_ORDER = tuple(BRANCHES)
ACTION_TUPLES = tuple(BRANCHES[branch] for branch in BRANCH_ORDER)
METHOD_VARIANTS = ("history", "nearest", "additive", "eiffert2019-style", "team-concat", "matched-additive", "joint")
METHOD_NAMES = {
    "history": "history-only",
    "nearest": "nearest-robot",
    "additive": "additive",
    "joint": "joint",
    "team-concat": "ScePT-style",
    "eiffert2019-style": "eiffert2019-style",
    "matched-additive": "matched-additive",
}


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _hash_scene(
    robot_positions: np.ndarray,
    robot_velocities: np.ndarray,
    goals: np.ndarray,
    observations: list[tuple[np.ndarray, np.ndarray]],
) -> str:
    values = [robot_positions, robot_velocities, goals]
    if observations:
        ordered = sorted(observations, key=lambda item: tuple(item[0].tolist()))
        values.extend([np.asarray(item, dtype=np.float32) for pair in ordered for item in pair])
    payload = np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1) for value in values])
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _scene(capture: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]], str]:
    responses = capture.pop("responses")
    robot_positions, robot_velocities, goals, observations = _scene_from_responses(
        responses, partial_observation=False
    )
    return (
        robot_positions,
        robot_velocities,
        goals,
        observations,
        _hash_scene(robot_positions, robot_velocities, goals, observations),
    )


def _human_position(observations: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if not observations:
        return np.full(2, np.nan, dtype=np.float32)
    return sorted(observations, key=lambda item: tuple(item[0].tolist()))[0][0].copy()


def _human_observation(
    observations: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if not observations:
        nan = np.full(2, np.nan, dtype=np.float32)
        return nan.copy(), nan
    position, velocity = sorted(observations, key=lambda item: tuple(item[0].tolist()))[0]
    return position.copy(), velocity.copy()


def _run_branch(
    env: Any,
    capture: dict[str, Any],
    seed: int,
    branch_name: str,
    steps: int,
) -> dict[str, Any]:
    env.reset(seed=seed, return_info=True)
    robot_positions, robot_velocities, goals, observations, initial_hash = _scene(capture)
    human_position, human_velocity = _human_observation(observations)
    rows: list[dict[str, Any]] = []
    robot_position_history = [robot_positions.copy()]
    robot_velocity_history = [robot_velocities.copy()]
    goal_history = [goals.copy()]
    human_position_history = [human_position.copy()]
    human_velocity_history = [human_velocity.copy()]

    def add_row(step: int) -> None:
        rows.append(
            {
                "seed": seed,
                "branch": branch_name,
                "step": step,
                "initial_hash": initial_hash,
                "human_x": float(human_position_history[-1][0]),
                "human_y": float(human_position_history[-1][1]),
                "human_vx": float(human_velocity_history[-1][0]),
                "human_vy": float(human_velocity_history[-1][1]),
                "observed": int(np.isfinite(human_position_history[-1]).all()),
                "robot_positions": robot_position_history[-1].tolist(),
                "robot_velocities": robot_velocity_history[-1].tolist(),
                "goals": goal_history[-1].tolist(),
                "action": list(BRANCHES[branch_name]),
            }
        )

    add_row(0)
    action = {
        agent: int(BRANCHES[branch_name][index])
        for index, agent in enumerate(env.agents)
    }
    for step in range(1, steps + 1):
        env.step(action)
        robot_positions, robot_velocities, goals, observations, _ = _scene(capture)
        human_position, human_velocity = _human_observation(observations)
        robot_position_history.append(robot_positions.copy())
        robot_velocity_history.append(robot_velocities.copy())
        goal_history.append(goals.copy())
        human_position_history.append(human_position.copy())
        human_velocity_history.append(human_velocity.copy())
        add_row(step)
    return {
        "seed": seed,
        "branch": branch_name,
        "initial_hash": initial_hash,
        "rows": rows,
        "robot_positions": np.asarray(robot_position_history, dtype=np.float32),
        "robot_velocities": np.asarray(robot_velocity_history, dtype=np.float32),
        "human_positions": np.asarray(human_position_history, dtype=np.float32),
        "human_velocities": np.asarray(human_velocity_history, dtype=np.float32),
        "goals": np.asarray(goal_history, dtype=np.float32),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "seed", "branch", "step", "initial_hash", "human_id", "human_x", "human_y",
        "human_vx", "human_vy",
        "observed", "robot_action_0", "robot_action_1", "robot_action_2",
        "robot_positions", "robot_velocities", "goals",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            action = row["action"]
            writer.writerow(
                {
                    **{key: row[key] for key in fields if key in row},
                    "human_id": 0,
                    "robot_action_0": action[0],
                    "robot_action_1": action[1],
                    "robot_action_2": action[2],
                    "robot_positions": json.dumps(row["robot_positions"], separators=(",", ":")),
                    "robot_velocities": json.dumps(row["robot_velocities"], separators=(",", ":")),
                    "goals": json.dumps(row["goals"], separators=(",", ":")),
                }
            )


def _load_collection_bundles(paths: list[Path]) -> list[dict[str, Any]]:
    """Load validated paired CSVs without starting the ROS simulator."""

    rows_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                seed = int(raw["seed"])
                branch = str(raw["branch"])
                step = int(raw["step"])
                key = (seed, branch, step)
                if key in rows_by_key:
                    raise ValueError(f"duplicate collection key across CSVs: {key}")
                rows_by_key[key] = {
                    "seed": seed,
                    "branch": branch,
                    "step": step,
                    "initial_hash": str(raw["initial_hash"]),
                    "human_x": float(raw["human_x"]),
                    "human_y": float(raw["human_y"]),
                    "human_vx": float(raw["human_vx"]),
                    "human_vy": float(raw["human_vy"]),
                    "observed": int(raw["observed"]),
                    "robot_positions": json.loads(raw["robot_positions"]),
                    "robot_velocities": json.loads(raw["robot_velocities"]),
                    "goals": json.loads(raw["goals"]),
                    "action": [
                        int(raw["robot_action_0"]),
                        int(raw["robot_action_1"]),
                        int(raw["robot_action_2"]),
                    ],
                }

    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for (seed, branch, _), row in rows_by_key.items():
        groups.setdefault((seed, branch), []).append(row)
    bundles_by_seed: dict[int, dict[str, Any]] = {}
    for (seed, branch), group_rows in sorted(groups.items()):
        group_rows.sort(key=lambda row: row["step"])
        hashes = {row["initial_hash"] for row in group_rows}
        if len(hashes) != 1:
            raise ValueError(f"initial hash changed within collection group {(seed, branch)}")
        group = {
            "seed": seed,
            "branch": branch,
            "initial_hash": group_rows[0]["initial_hash"],
            "rows": group_rows,
            "robot_positions": np.asarray(
                [row["robot_positions"] for row in group_rows], dtype=np.float32
            ),
            "robot_velocities": np.asarray(
                [row["robot_velocities"] for row in group_rows], dtype=np.float32
            ),
            "human_positions": np.asarray(
                [[row["human_x"], row["human_y"]] for row in group_rows],
                dtype=np.float32,
            ),
            "human_velocities": np.asarray(
                [[row["human_vx"], row["human_vy"]] for row in group_rows],
                dtype=np.float32,
            ),
            "goals": np.asarray(
                [row["goals"] for row in group_rows], dtype=np.float32
            ),
        }
        bundles_by_seed.setdefault(seed, {"seed": seed, "branches": {}})
        bundles_by_seed[seed]["branches"][branch] = group
    bundles = [bundles_by_seed[seed] for seed in sorted(bundles_by_seed)]
    for bundle in bundles:
        missing = set(BRANCH_ORDER) - set(bundle["branches"])
        if missing:
            raise ValueError(f"collection seed {bundle['seed']} is missing branches {sorted(missing)}")
        hashes = {group["initial_hash"] for group in bundle["branches"].values()}
        if len(hashes) != 1:
            raise ValueError(f"collection seed {bundle['seed']} has mismatched initial hashes")
        bundle["initial_hash"] = next(iter(hashes))
    return bundles


def _write_collection_only_artifacts(
    root: Path,
    source: Path,
    output: Path,
    args: argparse.Namespace,
    paired_bundles: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    completeness: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    """Write a raw collection manifest for one isolated simulator process."""

    git = _git_state(root)
    split = {"train_seeds": [], "val_seeds": [], "test_seeds": []}
    summary["split"] = split
    summary["response_model_metrics"] = {}
    collection = output / "collection"
    (collection / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for bundle in paired_bundles:
        seed_manifest = {
            "status": "COMPLETED",
            "protocol": "socialgym2-paired-multi-robot-response-v1",
            "collection_only": True,
            "seed": bundle["seed"],
            "initial_hash": bundle["initial_hash"],
            "branches": {
                branch: {
                    "rows": len(group["rows"]),
                    "initial_hash": group["initial_hash"],
                }
                for branch, group in bundle["branches"].items()
            },
        }
        (manifest_dir / f"seed{bundle['seed']}.json").write_text(
            json.dumps(seed_manifest, indent=2) + "\n", encoding="utf-8"
        )
    metadata = {
        "protocol": "socialgym2-paired-multi-robot-response-v1",
        "collection_only": True,
        "claim_boundary": "official SocialGym2 multi-robot paired action-conditioned response difference; no persistent human identity, real-human causality, or continuous H3 equivalence",
        "git": git,
        "source": str(source),
        "scenario": args.scenario,
        "seeds": args.seeds,
        "branches": BRANCHES,
        "num_agents": 3,
        "num_humans": 1,
        "steps": args.steps,
        "records": len(rows),
        "collection_integrity": completeness,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# SocialGym2 paired multi-robot response collection",
        "",
        f"Seeds: `{', '.join(map(str, args.seeds))}`; exact same-reset groups: `{summary['exact_reset_groups']}`; rows: `{len(rows)}`.",
        "This directory contains raw paired interventions collected in an isolated simulator process. Model fitting is performed only after seed-level CSVs are merged.",
        "",
        "## Boundary",
        "",
        "SocialGym2 has no persistent human IDs in this observation path, so this is a paired multi-robot response-mechanism check only. It does not validate identity persistence, real-human causality, or continuous-action H3.",
        "",
    ]
    (output / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(
            handle,
            fieldnames=[
                "model_seed", "train_variant", "prediction_variant",
                "method", "metric", "value",
            ],
        ).writeheader()
    artifacts = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("metadata.json", "report.md")
    }
    artifacts["collection/paired_rollouts.csv"] = hashlib.sha256(
        (collection / "paired_rollouts.csv").read_bytes()
    ).hexdigest()
    artifacts["collection/summary.json"] = hashlib.sha256(
        (collection / "summary.json").read_bytes()
    ).hexdigest()
    artifacts["metrics.csv"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    manifest = {
        "status": "COMPLETED",
        "protocol": metadata["protocol"],
        "collection_only": True,
        "records": len(rows),
        "seeds": args.seeds,
        "exact_reset_groups": summary["exact_reset_groups"],
        "split": split,
        "collection_integrity": completeness,
        "artifacts": artifacts,
        "claim_boundary": metadata["claim_boundary"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {"metadata": metadata, "summary": summary, "manifest": manifest}


def _summarize(groups: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    reference = {group["seed"]: group for group in groups if group["branch"] == "111"}
    gaps: dict[str, list[float]] = {branch: [] for branch in BRANCHES if branch != "111"}
    observed_counts: dict[str, int] = {branch: 0 for branch in BRANCHES}
    for group in groups:
        observed_counts[group["branch"]] += sum(row["observed"] for row in group["rows"])
        if group["branch"] == "111":
            continue
        baseline = reference[group["seed"]]["rows"][-1]
        current = group["rows"][-1]
        if baseline["observed"] and current["observed"]:
            gaps[group["branch"]].append(
                float(
                    np.linalg.norm(
                        np.asarray([current["human_x"], current["human_y"]])
                        - np.asarray([baseline["human_x"], baseline["human_y"]])
                    )
                )
            )
    return {
        "paired_groups": len(reference),
        "branches": list(BRANCHES),
        "steps": steps,
        "observed_rows": observed_counts,
        "final_human_displacement_gap_vs_all_halt": {
            branch: {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values)) if values else None,
                "count": len(values),
            }
            for branch, values in gaps.items()
        },
        "interpretation": "paired response difference in official SocialGym2 with one human and three controllable robots; no real-human causal claim",
    }


def _scene_from_group(group: dict[str, Any], history: int, dt: float) -> HeterogeneousSceneBatch:
    """Convert one eight-branch group to the common response-model contract."""

    robot_positions = np.asarray(group["robot_positions"], dtype=np.float32)
    robot_velocities = np.asarray(group["robot_velocities"], dtype=np.float32)
    human_positions = np.asarray(group["human_positions"], dtype=np.float32)
    goals = np.asarray(group["goals"], dtype=np.float32)
    robots = robot_positions.shape[1]
    agents = robots + 1
    robot_history = np.zeros((robots, history, 2), dtype=np.float32)
    for robot in range(robots):
        current = robot_positions[0, robot]
        velocity = robot_velocities[0, robot]
        offsets = np.arange(history - 1, -1, -1, dtype=np.float32)[:, None]
        robot_history[robot] = current[None] - offsets * velocity[None] * float(dt)
    human_current = human_positions[0]
    human_velocity = np.asarray(group["human_velocities"], dtype=np.float32)[0]
    offsets = np.arange(history - 1, -1, -1, dtype=np.float32)[:, None]
    human_history = human_current[None] - offsets * human_velocity[None] * float(dt)
    agent_history = np.concatenate([robot_history, human_history[None]], axis=0)

    future = np.zeros((agents, robot_positions.shape[0] - 1, 2), dtype=np.float32)
    future[:robots] = np.diff(robot_positions, axis=0).transpose(1, 0, 2) / float(dt)
    future[robots] = np.diff(human_positions, axis=0) / float(dt)
    current_pos = np.concatenate([robot_positions[0], human_positions[0:1]], axis=0)
    role_ids = np.concatenate([
        np.zeros(robots, dtype=np.int64),
        np.ones(1, dtype=np.int64),
    ])
    robot_mask = np.concatenate([np.ones(robots, dtype=bool), np.zeros(1, dtype=bool)])
    entity_mask = ~robot_mask
    goal_values = np.zeros((agents, 2), dtype=np.float32)
    goal_values[:robots] = goals[0]
    goal_mask = robot_mask.copy()
    robot_actions = np.full(agents, -1, dtype=np.int64)
    robot_actions[:robots] = np.asarray(group["rows"][0]["action"], dtype=np.int64)
    active_mask = np.ones(agents, dtype=bool)
    return HeterogeneousSceneBatch(
        history=torch.from_numpy(agent_history[None]),
        future=torch.from_numpy(future[None]),
        current_pos=torch.from_numpy(current_pos[None]),
        roles=torch.from_numpy(role_ids[None]),
        agent_mask=torch.from_numpy(active_mask[None]),
        robot_mask=torch.from_numpy(robot_mask[None]),
        entity_mask=torch.from_numpy(entity_mask[None]),
        loss_mask=torch.from_numpy(active_mask[None]),
        goals=torch.from_numpy(goal_values[None]),
        goal_mask=torch.from_numpy(goal_mask[None]),
        robot_actions=torch.from_numpy(robot_actions[None]),
        dt=float(dt),
    )


class _SocialGymPairedDataset(Dataset[HeterogeneousSceneBatch]):
    def __init__(self, bundles: list[dict[str, Any]], history: int, dt: float) -> None:
        self.bundles = bundles
        self.scenes = [
            _scene_from_group(bundle["branches"][branch], history, dt)
            for bundle in bundles
            for branch in BRANCH_ORDER
        ]

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> HeterogeneousSceneBatch:
        return self.scenes[index]


def _collate_scenes(samples: list[HeterogeneousSceneBatch]) -> HeterogeneousSceneBatch:
    if not samples:
        raise ValueError("cannot collate an empty SocialGym2 paired batch")
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
        robot_actions=torch.cat([sample.robot_actions for sample in samples], dim=0),
        dt=samples[0].dt,
    )


def _loader(
    dataset: _SocialGymPairedDataset,
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


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ece(probability: torch.Tensor, label: torch.Tensor, bins: int = 10) -> torch.Tensor:
    confidence = torch.maximum(probability, 1.0 - probability)
    prediction = probability >= 0.5
    correct = prediction.eq(label.bool()).to(probability.dtype)
    result = torch.zeros((), device=probability.device, dtype=probability.dtype)
    for index in range(bins):
        lower = float(index) / bins
        upper = float(index + 1) / bins
        mask = (confidence >= lower) & (
            confidence < upper if index + 1 < bins else confidence <= upper
        )
        if mask.any():
            result = result + mask.to(probability.dtype).mean() * (
                confidence[mask].mean() - correct[mask].mean()
            ).abs()
    return result


def _factorial_interaction(
    values: torch.Tensor,
    actions: tuple[tuple[int, int, int], ...],
) -> torch.Tensor:
    """Return averaged pair second differences over the remaining robot."""

    index = {action: position for position, action in enumerate(actions)}
    residuals = []
    for first in range(3):
        for second in range(first + 1, 3):
            remaining = next(item for item in range(3) if item not in (first, second))
            for fixed in (0, 1):
                base = [1, 1, 1]
                base[remaining] = fixed
                a00 = base.copy()
                a10 = base.copy()
                a01 = base.copy()
                a11 = base.copy()
                a10[first] = 0
                a01[second] = 0
                a11[first] = 0
                a11[second] = 0
                residuals.append(
                    values[:, index[tuple(a11)]]
                    - values[:, index[tuple(a10)]]
                    - values[:, index[tuple(a01)]]
                    + values[:, index[tuple(a00)]]
                )
    return torch.stack(residuals, dim=1)


@torch.no_grad()
def _evaluate_model(
    model: TeamResponsePredictor,
    loader: DataLoader,
    device: str,
    include_shuffle: bool,
    actions: tuple[tuple[int, int, int], ...],
) -> dict[str, float]:
    model.eval()
    names = [model.variant]
    if model.variant == "joint":
        names.extend(
            ["joint-without-pair", "shuffled-joint"]
            if include_shuffle
            else ["joint-without-pair"]
        )
    metric_names = (
        "nll", "ade", "fde", "brier", "ece", "coverage_1sigma",
        "coverage_2sigma", "action_effect_fde_error", "action_effect_error",
        "joint_interaction_residual_error", "pair_rmse",
    )
    values: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in metric_names} for name in names
    }
    for scene in loader:
        response = build_entity_response_batch(
            scene.to(device),
            num_robots=3,
            interaction_radius=3.0,
            robot_actions=scene.robot_actions,
        ).to(device)
        outputs: dict[str, dict[str, torch.Tensor]] = {model.variant: model(response)}
        if model.variant == "joint":
            outputs["joint-without-pair"] = model(response, include_pair_interactions=False)
            if include_shuffle:
                shuffled_response = shuffle_robot_candidates_within_groups(
                    response,
                    group_size=len(actions),
                    branch_size=1,
                )
                outputs["shuffled-joint"] = model(shuffled_response)
        target = response.target_future_vel
        groups = scene.batch_size
        entities = 1
        if groups != len(actions):
            raise ValueError("SocialGym2 evaluation batches must contain one complete action group")
        target_position = torch.cumsum(target * response.dt, dim=1).reshape(
            1, len(actions), entities, -1, 2
        )
        for name, output in outputs.items():
            mean = output["mean"]
            log_std = output["log_std"]
            metric = response_metrics(mean, log_std, target, response.dt)
            sigma = log_std.exp().clamp_min(1e-8)
            probability = 0.5 * (1.0 + torch.erf(mean[..., 0] / (sigma[..., 0] * np.sqrt(2.0))))
            label = target[..., 0] > 0.0
            z = (target - mean) / sigma
            predicted_position = torch.cumsum(mean * response.dt, dim=1).reshape(
                1, len(actions), entities, -1, 2
            )
            effect = predicted_position[:, :, :, -1] - predicted_position[:, -1:, :, -1]
            target_effect = target_position[:, :, :, -1] - target_position[:, -1:, :, -1]
            effect_error = torch.linalg.vector_norm(effect - target_effect, dim=-1).mean()
            predicted_residual = _factorial_interaction(predicted_position[:, :, :, -1], actions)
            target_residual = _factorial_interaction(target_position[:, :, :, -1], actions)
            interaction_error = torch.linalg.vector_norm(
                predicted_residual - target_residual, dim=-1
            ).mean()
            for key, value in (
                ("nll", metric["nll"].mean()),
                ("ade", metric["ade"].mean()),
                ("fde", metric["fde"].mean()),
                ("brier", (probability - label.to(probability.dtype)).square().mean()),
                ("ece", _ece(probability, label)),
                ("coverage_1sigma", (z.abs() <= 1.0).to(z.dtype).mean()),
                ("coverage_2sigma", (z.abs() <= 2.0).to(z.dtype).mean()),
                ("action_effect_fde_error", effect_error),
                ("action_effect_error", effect_error),
                ("joint_interaction_residual_error", interaction_error),
                ("pair_rmse", interaction_error),
            ):
                values[name][key].append(float(value.detach().cpu()))
    result: dict[str, float] = {}
    for name in names:
        for metric_name in metric_names:
            metric_values = [
                value for value in values[name][metric_name]
                if math.isfinite(value)
            ]
            result[f"{name}:{metric_name}"] = (
                float(np.mean(metric_values)) if metric_values else float("nan")
            )
            result[f"{name}:{metric_name}_std"] = (
                float(np.std(metric_values, ddof=1))
                if len(metric_values) > 1
                else (0.0 if metric_values else float("nan"))
            )
    return result


def _train_variant(
    variant: str,
    train_dataset: _SocialGymPairedDataset,
    val_dataset: _SocialGymPairedDataset,
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
) -> TeamResponsePredictor:
    _set_seed(seed)
    config = ResponseModelConfig(
        history=args.history,
        horizon=args.steps,
        num_robots=3,
        hidden_dim=args.hidden_dim,
        num_heads=4,
        dropout=args.dropout,
        architecture="tracer",
        graph_layers=2,
        dt=args.dt,
    )
    model = build_response_predictor(variant, config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Keep all interventions from a base scene together for paired metrics.
    train_loader = _loader(train_dataset, len(BRANCH_ORDER), False, seed + 17)
    val_loader = _loader(val_dataset, len(BRANCH_ORDER), False, seed + 19)
    best_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, args.device,
            num_robots=3, interaction_radius=3.0,
        )
        val_metrics = _evaluate_model(model, val_loader, args.device, False, ACTION_TUPLES)
        val_nll = val_metrics[f"{variant}:nll"]
        if val_nll < best_nll:
            best_nll = val_nll
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 3, 1) == 0:
            print(
                f"socialgym2 model_seed={seed} variant={variant} epoch={epoch} "
                f"train_nll={train_loss:.6f} val_nll={val_nll:.6f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError(f"no validation checkpoint for {variant}")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "model_config": vars(config), "variant": variant, "seed": seed},
        output_dir / f"{variant}_seed{seed}.pt",
    )
    return model.eval()


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    source = (root / args.source).resolve()
    launch_config = (source / "config/gym_gen/launch.launch").resolve()
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.collection_only and args.collection_csv:
        raise ValueError("--collection-only and --collection-csv cannot be combined")
    if args.collection_csv:
        collection_paths = [
            (root / path).resolve() for path in args.collection_csv
        ]
        paired_bundles = _load_collection_bundles(collection_paths)
        if args.seeds != [bundle["seed"] for bundle in paired_bundles]:
            raise ValueError(
                "--seeds must match the sorted seed order represented by --collection-csv"
            )
    else:
        # The official simulator resolves its legacy robot config relative to
        # the SocialGym2 checkout. Keep that launch cwd separate from provenance root.
        os.chdir(source)
        paired_bundles = []
        # RosSocialEnv is a singleton. Keep one launch alive for every seed so a
        # later reset cannot reuse an already-shutdown singleton instance.
        env = _build_env(source, 3, 1, False, launch_config, args.scenario)
        capture = _capture_observations(env)
        try:
            for seed in args.seeds:
                seed_groups = {
                    branch: _run_branch(env, capture, seed, branch, args.steps)
                    for branch in BRANCH_ORDER
                }
                hashes = [group["initial_hash"] for group in seed_groups.values()]
                if len(set(hashes)) != 1:
                    raise RuntimeError(
                        f"same-seed SocialGym2 reset mismatch for seed {seed}: {hashes}"
                    )
                paired_bundles.append(
                    {"seed": seed, "initial_hash": hashes[0], "branches": seed_groups}
                )
                print(
                    f"socialgym2_paired_completed seed={seed} initial_hash={hashes[0]}",
                    flush=True,
                )
        finally:
            try:
                env.close()
            finally:
                env.launch.shutdown()

    groups = [
        group
        for bundle in paired_bundles
        for group in bundle["branches"].values()
    ]
    rows = [row for group in groups for row in group["rows"]]
    collection = output / "collection"
    collection.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, collection / "paired_rollouts.csv")
    summary = _summarize(groups, args.steps)
    summary["seeds"] = args.seeds
    summary["exact_reset_groups"] = len(args.seeds)
    expected_rows_per_branch = args.steps + 1
    completeness = {
        "expected_rows_per_branch": expected_rows_per_branch,
        "expected_rows_total": expected_rows_per_branch * len(BRANCH_ORDER) * len(args.seeds),
        "actual_rows_total": len(rows),
        "missing_branch_rows": [],
        "duplicate_keys": [],
    }
    seen_keys: set[tuple[int, str, int]] = set()
    for group in groups:
        key_rows = {(int(row["seed"]), str(row["branch"]), int(row["step"])) for row in group["rows"]}
        if len(key_rows) != len(group["rows"]):
            completeness["duplicate_keys"].append(
                {"seed": group["seed"], "branch": group["branch"]}
            )
        overlap = seen_keys.intersection(key_rows)
        if overlap:
            completeness["duplicate_keys"].append(
                {"keys": [list(key) for key in sorted(overlap)]}
            )
        seen_keys.update(key_rows)
        if len(group["rows"]) != expected_rows_per_branch:
            completeness["missing_branch_rows"].append(
                {
                    "seed": group["seed"],
                    "branch": group["branch"],
                    "actual": len(group["rows"]),
                }
            )
    expected_keys = {
        (seed, branch, step)
        for seed in args.seeds
        for branch in BRANCH_ORDER
        for step in range(expected_rows_per_branch)
    }
    completeness["missing_keys"] = [
        list(key) for key in sorted(expected_keys - seen_keys)
    ]
    completeness["unexpected_keys"] = [
        list(key) for key in sorted(seen_keys - expected_keys)
    ]
    completeness["complete"] = not any(
        completeness[key]
        for key in ("missing_branch_rows", "duplicate_keys", "missing_keys", "unexpected_keys")
    ) and len(rows) == completeness["expected_rows_total"]
    if not completeness["complete"]:
        raise RuntimeError(f"incomplete SocialGym2 paired collection: {completeness}")
    summary["collection_integrity"] = completeness
    if args.collection_only:
        result = _write_collection_only_artifacts(
            root,
            source,
            output,
            args,
            paired_bundles,
            rows,
            summary,
            completeness,
            started,
        )
        print(
            json.dumps(
                {"status": result["manifest"]["status"], "records": len(rows), "output": str(output)},
                indent=2,
            ),
            flush=True,
        )
        return result
    git = _git_state(root)
    val_count = min(args.val_count, len(paired_bundles) - 2)
    test_count = min(args.test_count, len(paired_bundles) - val_count - 1)
    train_count = len(paired_bundles) - val_count - test_count
    if min(train_count, val_count, test_count) < 1:
        raise ValueError(
            f"need at least one train, validation, and test seed; got total={len(paired_bundles)}, "
            f"train={train_count}, val={val_count}, test={test_count}"
        )
    train_bundles = paired_bundles[:train_count]
    val_bundles = paired_bundles[train_count:train_count + val_count]
    test_bundles = paired_bundles[train_count + val_count:]
    split = {
        "train_seeds": [bundle["seed"] for bundle in train_bundles],
        "val_seeds": [bundle["seed"] for bundle in val_bundles],
        "test_seeds": [bundle["seed"] for bundle in test_bundles],
    }
    train_dataset = _SocialGymPairedDataset(train_bundles, args.history, args.dt)
    val_dataset = _SocialGymPairedDataset(val_bundles, args.history, args.dt)
    test_dataset = _SocialGymPairedDataset(test_bundles, args.history, args.dt)
    checkpoints = output / "checkpoints"
    model_metrics: list[dict[str, Any]] = []
    model_metadata: dict[str, dict[str, Any]] = {}
    for model_seed in args.model_seeds:
        for variant in METHOD_VARIANTS:
            model = _train_variant(
                variant, train_dataset, val_dataset, args, model_seed, checkpoints
            )
            model_metadata[f"{variant}:seed{model_seed}"] = external_method_metadata(
                model, variant
            )
            metrics = _evaluate_model(
                model,
                _loader(test_dataset, len(BRANCH_ORDER), False, model_seed + 29),
                args.device,
                include_shuffle=True,
                actions=ACTION_TUPLES,
            )
            for key, value in metrics.items():
                prediction_variant, metric_name = key.split(":", 1)
                model_metrics.append(
                    {
                        "model_seed": model_seed,
                        "train_variant": variant,
                        "prediction_variant": prediction_variant,
                        "method": (
                            "independent"
                            if prediction_variant == "joint-without-pair"
                            else "shuffled-joint"
                            if prediction_variant == "shuffled-joint"
                            else METHOD_NAMES[prediction_variant]
                        ),
                        "metric": metric_name,
                        "value": value,
                    }
                )
    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_seed", "train_variant", "prediction_variant",
                "method", "metric", "value",
            ],
        )
        writer.writeheader()
        writer.writerows(model_metrics)
    aggregate_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for row in model_metrics:
        item = aggregate_metrics.setdefault(row["method"], {}).setdefault(
            row["metric"], {"values": []}
        )
        item["values"].append(float(row["value"]))
    for method, method_metrics in aggregate_metrics.items():
        for metric, item in method_metrics.items():
            numeric = [value for value in item["values"] if math.isfinite(value)]
            item["mean"] = float(np.mean(numeric)) if numeric else float("nan")
            item["std"] = (
                float(np.std(numeric, ddof=1))
                if len(numeric) > 1
                else 0.0
            )
            item["n"] = len(numeric)
            del item["values"]
    summary["split"] = split
    summary["response_model_metrics"] = aggregate_metrics
    (collection / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for bundle in paired_bundles:
        seed_manifest = {
            "status": "COMPLETED",
            "protocol": "socialgym2-paired-multi-robot-response-v1",
            "seed": bundle["seed"],
            "initial_hash": bundle["initial_hash"],
            "branches": {
                branch: {
                    "rows": len(group["rows"]),
                    "initial_hash": group["initial_hash"],
                }
                for branch, group in bundle["branches"].items()
            },
        }
        (manifest_dir / f"seed{bundle['seed']}.json").write_text(
            json.dumps(seed_manifest, indent=2) + "\n", encoding="utf-8"
        )
    metadata = {
        "protocol": "socialgym2-paired-multi-robot-response-v1",
        "claim_boundary": "official SocialGym2 multi-robot paired action-conditioned response difference; no persistent human identity, real-human causality, or continuous H3 equivalence",
        "git": git,
        "source": str(source),
        "scenario": args.scenario,
        "seeds": args.seeds,
        "branches": BRANCHES,
        "num_agents": 3,
        "num_humans": 1,
        "steps": args.steps,
        "records": len(rows),
        "model_seeds": args.model_seeds,
        "device": args.device,
        "action_context": "optional robot_actions mode code; required for SocialGym2 GoAlone/Halt because short-horizon geometry is not mode-identifying",
        "method_taxonomy": {
            "eiffert2019-style": "external action-conditioned response reimplementation; not official",
            "ScePT-style": "team-concat external scene-context proxy; not official ScePT reproduction",
        },
        "model_metadata": model_metadata,
        "collection_csv": args.collection_csv,
        "split": split,
        "collection_integrity": completeness,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_lines = [
        "# SocialGym2 paired multi-robot response",
        "",
        f"Seeds: `{', '.join(map(str, args.seeds))}`; exact same-reset groups: `{summary['exact_reset_groups']}`; rows: `{len(rows)}`.",
        "Each seed is reset with the same seed for eight branches. The initial ROS scene hash must match across all branches; only the three-robot GoAlone/Halt joint action changes.",
        "SocialGym2 high-level GoAlone/Halt codes are passed as explicit optional robot action context; geometric robot trajectories alone are not sufficient to identify this intervention.",
        "",
        "## Response difference vs all-halt",
        "",
    ]
    for branch, item in summary["final_human_displacement_gap_vs_all_halt"].items():
        report_lines.append(f"- `{branch}`: final human displacement gap `{item['mean']:.6f}` +/- `{item['std']:.6f}` m, n=`{item['count']}`.")
    report_lines.extend(["", "## Held-out response prediction", ""])
    metric_order = (
        "nll", "brier", "ece", "ade", "fde",
        "action_effect_fde_error", "joint_interaction_residual_error",
    )
    report_lines.append("| Method | NLL | Brier | ECE | ADE | FDE | Effect error | Joint residual |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for method in (
        "history-only", "nearest-robot", "additive", "eiffert2019-style", "ScePT-style", "matched-additive", "joint",
        "independent", "shuffled-joint",
    ):
        item = summary["response_model_metrics"].get(method, {})
        values = [item.get(metric, {}).get("mean", float("nan")) for metric in metric_order]
        formatted = ["n/a" if not math.isfinite(value) else f"{value:.4f}" for value in values]
        report_lines.append(
            f"| {method} | {formatted[0]} | {formatted[1]} | {formatted[2]} | {formatted[3]} | {formatted[4]} | {formatted[5]} | {formatted[6]} |"
        )
    report_lines.extend([
        "",
        f"Train seeds: `{', '.join(map(str, split['train_seeds']))}`; validation seeds: `{', '.join(map(str, split['val_seeds']))}`; held-out test seeds: `{', '.join(map(str, split['test_seeds']))}`.",
        "The `joint` versus `independent` comparison is the pair-interaction ablation. `shuffled-joint` breaks the branch-to-response pairing within each held-out base scene and is a negative control. `eiffert2019-style` is a method-level reimplementation, and `ScePT-style` is the existing team-concat proxy; neither is an official reproduction.",
    ])
    report_lines.extend([
        "",
        "## Boundary",
        "",
        "SocialGym2 has no persistent human IDs in this observation path, so this is a paired multi-robot response-mechanism check only. It does not validate identity persistence, real-human causality, or continuous-action H3.",
        "",
    ])
    (output / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    artifacts = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("metadata.json", "report.md")
    }
    artifacts["collection/paired_rollouts.csv"] = hashlib.sha256((collection / "paired_rollouts.csv").read_bytes()).hexdigest()
    artifacts["collection/summary.json"] = hashlib.sha256((collection / "summary.json").read_bytes()).hexdigest()
    artifacts["metrics.csv"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    manifest = {
        "status": "COMPLETED",
        "protocol": metadata["protocol"],
        "records": len(rows),
        "seeds": args.seeds,
        "exact_reset_groups": summary["exact_reset_groups"],
        "split": split,
        "collection_integrity": completeness,
        "artifacts": artifacts,
        "claim_boundary": metadata["claim_boundary"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "records": len(rows), "output": str(output)}, indent=2), flush=True)
    return {"metadata": metadata, "summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="raw_data/external/source_repos/socialgym2_official")
    parser.add_argument("--output-dir", default="experiments/external_mechanism/socialgym2_paired_h1_20260904")
    parser.add_argument("--scenario", choices=("envs/scenario/intersection", "envs/scenario/round_about"), default="envs/scenario/intersection")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2031, 2039, 2053, 2063, 2071, 2081, 2091])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[2027, 2031, 2039])
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=2)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument(
        "--collection-only",
        action="store_true",
        help="collect paired rollouts and manifests without fitting response models",
    )
    parser.add_argument(
        "--collection-csv",
        nargs="+",
        default=[],
        help="load one or more seed-level paired CSVs instead of starting SocialGym2",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
