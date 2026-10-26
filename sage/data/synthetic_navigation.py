from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from sage.data.heterogeneous_scene import AgentRole, HeterogeneousSceneBatch
from sage.data.sdd_adapter import ROLE_DEFAULT_GEOMETRY


ROLE_SPEED_RANGES = {
    int(AgentRole.PHA): (0.55, 1.35),
    int(AgentRole.HNAA): (1.30, 2.65),
    int(AgentRole.SSAA): (0.80, 2.10),
}

ROLE_COMFORT_RADIUS = {
    int(AgentRole.PHA): 2.0,
    int(AgentRole.HNAA): 2.6,
    int(AgentRole.SSAA): 1.4,
    int(AgentRole.UNKNOWN_ENTITY): 1.6,
}


@dataclass(frozen=True)
class SyntheticNavigationSpec:
    split: str = "train"
    num_samples: int = 4096
    history: int = 8
    horizon: int = 12
    num_robots: int = 3
    num_entities: int = 50
    dt: float = 0.4
    layout: str = "mixed"
    area_size: float = 12.0
    corridor_width: float = 3.0
    seed: int = 2027
    robot_speed: float = 1.25
    robot_max_speed: float = 2.4
    entity_noise: float = 0.03
    role_mix: Mapping[str, float] | None = None


def _role_mix(raw: Mapping[str, float] | None) -> tuple[np.ndarray, np.ndarray]:
    raw = raw or {"pha": 0.55, "hnaa": 0.30, "ssaa": 0.15}
    role_ids = []
    probs = []
    mapping = {
        "pha": int(AgentRole.PHA),
        "pedestrian": int(AgentRole.PHA),
        "hnaa": int(AgentRole.HNAA),
        "cyclist": int(AgentRole.HNAA),
        "ssaa": int(AgentRole.SSAA),
        "vehicle": int(AgentRole.SSAA),
    }
    for name, prob in raw.items():
        role_id = mapping.get(str(name).lower())
        if role_id is None:
            continue
        role_ids.append(role_id)
        probs.append(float(prob))
    if not role_ids or sum(probs) <= 0.0:
        role_ids = [int(AgentRole.PHA), int(AgentRole.HNAA), int(AgentRole.SSAA)]
        probs = [0.55, 0.30, 0.15]
    probs_np = np.asarray(probs, dtype=np.float64)
    probs_np = probs_np / probs_np.sum()
    return np.asarray(role_ids, dtype=np.int64), probs_np


def _unit(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, eps)


def _clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / np.maximum(norm, 1e-6))
    return vec * scale


class SyntheticNavigationDataset(Dataset):
    """Controlled heterogeneous navigation episodes for SAGE.

    The dataset is generated deterministically per index. Each episode contains
    controllable robots with local waypoints and independently moving entities
    with semantic roles PHA/HNAA/SSAA. Robot future trajectories are generated
    by a simple role-aware expert with robot-entity and robot-robot repulsion,
    giving the diffusion model a clean synthetic target for scalability tests.
    """

    def __init__(self, spec: SyntheticNavigationSpec) -> None:
        self.spec = spec
        self.num_samples = int(spec.num_samples)
        self.history = int(spec.history)
        self.horizon = int(spec.horizon)
        self.num_robots = int(spec.num_robots)
        self.num_entities = int(spec.num_entities)
        self.num_agents = self.num_robots + self.num_entities
        self.dt = float(spec.dt)
        self.layout = str(spec.layout).lower()
        self.role_ids, self.role_probs = _role_mix(spec.role_mix)
        split_offsets = {"train": 0, "val": 100_000, "test": 200_000}
        self.seed = int(spec.seed) + split_offsets.get(str(spec.split).lower(), 300_000)

    def __len__(self) -> int:
        return self.num_samples

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + int(idx) * 104729)

    def _layout_for_index(self, rng: np.random.Generator, idx: int) -> str:
        if self.layout != "mixed":
            return self.layout
        layouts = ("open", "crossing", "corridor", "bottleneck", "intersection")
        return layouts[int(idx) % len(layouts)]

    def _robot_tasks(self, rng: np.random.Generator, layout: str) -> tuple[np.ndarray, np.ndarray]:
        n = self.num_robots
        area = float(self.spec.area_size)
        half = area * 0.5
        starts = np.zeros((n, 2), dtype=np.float32)
        goals = np.zeros((n, 2), dtype=np.float32)
        lanes = np.linspace(-0.8, 0.8, max(n, 2), dtype=np.float32)[:n]
        rng.shuffle(lanes)

        if layout == "corridor":
            y = rng.uniform(-self.spec.corridor_width * 0.35, self.spec.corridor_width * 0.35, size=n)
            direction = rng.choice([-1.0, 1.0], size=n)
            starts[:, 0] = -direction * (half - 0.8)
            starts[:, 1] = y
            goals[:, 0] = direction * (half - 0.8)
            goals[:, 1] = y + rng.normal(0.0, 0.25, size=n)
        elif layout == "bottleneck":
            side = rng.choice([-1.0, 1.0], size=n)
            starts[:, 0] = -side * (half - 0.7)
            starts[:, 1] = lanes * self.spec.corridor_width + rng.normal(0.0, 0.15, size=n)
            goals[:, 0] = side * (half - 0.7)
            goals[:, 1] = -starts[:, 1] * 0.4 + rng.normal(0.0, 0.2, size=n)
        elif layout == "intersection":
            arms = rng.integers(0, 4, size=n)
            for i, arm in enumerate(arms):
                offset = rng.normal(0.0, 0.45)
                if arm == 0:
                    starts[i] = (-half + 0.6, offset)
                    goals[i] = (half - 0.6, -offset)
                elif arm == 1:
                    starts[i] = (half - 0.6, offset)
                    goals[i] = (-half + 0.6, -offset)
                elif arm == 2:
                    starts[i] = (offset, -half + 0.6)
                    goals[i] = (-offset, half - 0.6)
                else:
                    starts[i] = (offset, half - 0.6)
                    goals[i] = (-offset, -half + 0.6)
        elif layout == "crossing":
            angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + rng.uniform(-0.25, 0.25, size=n)
            starts = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * (half - 0.7)
            goals = -starts + rng.normal(0.0, 0.35, size=(n, 2))
        else:
            sides = rng.choice([-1.0, 1.0], size=n)
            starts[:, 0] = -sides * rng.uniform(half * 0.55, half * 0.90, size=n)
            starts[:, 1] = rng.uniform(-half * 0.60, half * 0.60, size=n)
            goals[:, 0] = sides * rng.uniform(half * 0.55, half * 0.90, size=n)
            goals[:, 1] = rng.uniform(-half * 0.60, half * 0.60, size=n)

        starts += rng.normal(0.0, 0.08, size=starts.shape)
        goals = np.clip(goals, -half + 0.2, half - 0.2)
        return starts.astype(np.float32), goals.astype(np.float32)

    def _entity_roles(self, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.role_ids, size=self.num_entities, p=self.role_probs).astype(np.int64)

    def _entity_motion(
        self,
        rng: np.random.Generator,
        roles: np.ndarray,
        robot_starts: np.ndarray,
        robot_goals: np.ndarray,
        layout: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        e = self.num_entities
        total = self.history + self.horizon
        current_index = self.history - 1
        area = float(self.spec.area_size)
        half = area * 0.5
        positions = np.zeros((e, total, 2), dtype=np.float32)

        for i in range(e):
            role = int(roles[i])
            speed_low, speed_high = ROLE_SPEED_RANGES.get(role, (0.5, 1.5))
            speed = rng.uniform(speed_low, speed_high)
            angle = rng.uniform(-np.pi, np.pi)
            if layout in {"corridor", "bottleneck"}:
                angle = rng.choice([0.0, np.pi]) + rng.normal(0.0, 0.25)
            elif layout in {"crossing", "intersection"}:
                angle = np.arctan2(-rng.normal(0.0, 2.5), -rng.normal(0.0, 2.5)) + rng.normal(0.0, 0.5)
            direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
            velocity = direction * speed

            near_path = rng.random() < 0.70
            if near_path and self.num_robots > 0:
                ridx = int(rng.integers(0, self.num_robots))
                alpha = rng.uniform(0.25, 0.75)
                center = (1.0 - alpha) * robot_starts[ridx] + alpha * robot_goals[ridx]
                offset = rng.normal(0.0, 1.2, size=2)
                if layout == "bottleneck":
                    center *= 0.25
                    offset *= 0.55
                current = center + offset
            else:
                current = rng.uniform(-half * 0.85, half * 0.85, size=2)

            turn = rng.normal(0.0, 0.035)
            for t in range(total):
                tau = t - current_index
                rot = np.asarray(
                    [
                        [np.cos(turn * tau), -np.sin(turn * tau)],
                        [np.sin(turn * tau), np.cos(turn * tau)],
                    ],
                    dtype=np.float32,
                )
                step_vel = rot @ velocity
                noise = rng.normal(0.0, self.spec.entity_noise, size=2)
                positions[i, t] = current + step_vel * (tau * self.dt) + noise
            positions[i, :, :] = np.clip(positions[i, :, :], -half, half)

        history = positions[:, : self.history]
        future_pos = positions[:, self.history :]
        current = history[:, -1:]
        future_vel = np.diff(np.concatenate([current, future_pos], axis=1), axis=1) / self.dt
        return history.astype(np.float32), future_vel.astype(np.float32)

    def _robot_history(self, starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
        direction = _unit(goals - starts)
        prev_vel = direction * min(float(self.spec.robot_speed), float(self.spec.robot_max_speed))
        history = np.zeros((self.num_robots, self.history, 2), dtype=np.float32)
        for h in range(self.history):
            tau = h - (self.history - 1)
            history[:, h] = starts + prev_vel * (tau * self.dt)
        return history

    def _robot_expert_future(
        self,
        starts: np.ndarray,
        goals: np.ndarray,
        entity_history: np.ndarray,
        entity_future_vel: np.ndarray,
        entity_roles: np.ndarray,
    ) -> np.ndarray:
        r = self.num_robots
        robot_pos = starts.astype(np.float32).copy()
        future_vel = np.zeros((r, self.horizon, 2), dtype=np.float32)
        entity_pos = entity_history[:, -1].astype(np.float32).copy()
        max_speed = float(self.spec.robot_max_speed)

        for t in range(self.horizon):
            entity_pos = entity_pos + entity_future_vel[:, t] * self.dt
            desired = _unit(goals - robot_pos) * float(self.spec.robot_speed)
            repulse = np.zeros_like(desired)

            for i in range(self.num_entities):
                rel = robot_pos - entity_pos[i][None, :]
                dist = np.linalg.norm(rel, axis=-1, keepdims=True)
                role = int(entity_roles[i])
                radius = ROLE_COMFORT_RADIUS.get(role, 1.6)
                active = dist < radius
                gain = 1.20 if role == int(AgentRole.PHA) else 1.45 if role == int(AgentRole.HNAA) else 0.80
                repulse += active * _unit(rel) * gain * ((radius - dist) / max(radius, 1e-6))

            if r > 1:
                rel_rr = robot_pos[:, None, :] - robot_pos[None, :, :]
                dist_rr = np.linalg.norm(rel_rr, axis=-1, keepdims=True)
                active_rr = (dist_rr < 1.3) & (dist_rr > 1e-5)
                rr_msg = active_rr * _unit(rel_rr) * (1.3 - dist_rr) / 1.3
                repulse += 1.10 * rr_msg.sum(axis=1)

            vel = _clip_norm(desired + repulse, max_speed)
            remaining = goals - robot_pos
            close = np.linalg.norm(remaining, axis=-1, keepdims=True) < (max_speed * self.dt)
            vel = np.where(close, remaining / self.dt, vel)
            future_vel[:, t] = vel
            robot_pos = robot_pos + vel * self.dt
        return future_vel.astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        rng = self._rng(idx)
        layout = self._layout_for_index(rng, idx)
        robot_starts, robot_goals = self._robot_tasks(rng, layout)
        entity_roles = self._entity_roles(rng)
        entity_history, entity_future = self._entity_motion(rng, entity_roles, robot_starts, robot_goals, layout)
        robot_history = self._robot_history(robot_starts, robot_goals)
        robot_future = self._robot_expert_future(robot_starts, robot_goals, entity_history, entity_future, entity_roles)

        history = np.concatenate([robot_history, entity_history], axis=0)
        future = np.concatenate([robot_future, entity_future], axis=0)
        roles = np.concatenate(
            [
                np.full(self.num_robots, int(AgentRole.ROBOT), dtype=np.int64),
                entity_roles,
            ],
            axis=0,
        )
        geometry = np.asarray(
            [ROLE_DEFAULT_GEOMETRY[int(role)] for role in roles],
            dtype=np.float32,
        )
        agent_mask = np.ones(self.num_agents, dtype=bool)
        robot_mask = np.zeros(self.num_agents, dtype=bool)
        robot_mask[: self.num_robots] = True
        entity_mask = agent_mask & ~robot_mask
        loss_mask = agent_mask.copy()
        goals = np.zeros((self.num_agents, 2), dtype=np.float32)
        goals[: self.num_robots] = robot_goals
        goals[self.num_robots :] = history[self.num_robots :, -1, :2]
        goal_mask = robot_mask.copy()

        return {
            "history": torch.tensor(history, dtype=torch.float32),
            "future": torch.tensor(future, dtype=torch.float32),
            "current_pos": torch.tensor(history[:, -1, :2], dtype=torch.float32),
            "roles": torch.tensor(roles, dtype=torch.long),
            "agent_mask": torch.tensor(agent_mask, dtype=torch.bool),
            "robot_mask": torch.tensor(robot_mask, dtype=torch.bool),
            "entity_mask": torch.tensor(entity_mask, dtype=torch.bool),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
            "geometry": torch.tensor(geometry, dtype=torch.float32),
            "goals": torch.tensor(goals, dtype=torch.float32),
            "goal_mask": torch.tensor(goal_mask, dtype=torch.bool),
            "dt": float(self.dt),
            "layout": layout,
        }


def collate_synthetic_scene_batch(samples: list[dict]) -> HeterogeneousSceneBatch:
    if not samples:
        raise ValueError("cannot collate an empty synthetic batch")
    return HeterogeneousSceneBatch(
        history=torch.stack([s["history"] for s in samples], dim=0),
        future=torch.stack([s["future"] for s in samples], dim=0),
        current_pos=torch.stack([s["current_pos"] for s in samples], dim=0),
        roles=torch.stack([s["roles"] for s in samples], dim=0),
        agent_mask=torch.stack([s["agent_mask"] for s in samples], dim=0),
        robot_mask=torch.stack([s["robot_mask"] for s in samples], dim=0),
        entity_mask=torch.stack([s["entity_mask"] for s in samples], dim=0),
        loss_mask=torch.stack([s["loss_mask"] for s in samples], dim=0),
        geometry=torch.stack([s["geometry"] for s in samples], dim=0),
        goals=torch.stack([s["goals"] for s in samples], dim=0),
        goal_mask=torch.stack([s["goal_mask"] for s in samples], dim=0),
        dt=float(samples[0].get("dt", 0.4)),
    )


def make_synthetic_dataset_from_config(raw_cfg: Mapping, split: str = "train", **overrides) -> SyntheticNavigationDataset:
    dataset_cfg = raw_cfg.get("dataset", {})
    synthetic_cfg = dataset_cfg.get("synthetic", {})
    split_key = f"{split}_samples"
    num_samples = int(overrides.pop("num_samples", synthetic_cfg.get(split_key, synthetic_cfg.get("samples", 4096))))
    spec = SyntheticNavigationSpec(
        split=split,
        num_samples=num_samples,
        history=int(overrides.pop("history", dataset_cfg.get("history", 8))),
        horizon=int(overrides.pop("horizon", dataset_cfg.get("horizon", 12))),
        num_robots=int(overrides.pop("num_robots", synthetic_cfg.get("num_robots", dataset_cfg.get("num_robot_proxies", 3)))),
        num_entities=int(overrides.pop("num_entities", synthetic_cfg.get("num_entities", dataset_cfg.get("max_entities", 50)))),
        dt=float(overrides.pop("dt", dataset_cfg.get("dt", 0.4))),
        layout=str(overrides.pop("layout", synthetic_cfg.get("layout", "mixed"))),
        area_size=float(overrides.pop("area_size", synthetic_cfg.get("area_size", 12.0))),
        corridor_width=float(overrides.pop("corridor_width", synthetic_cfg.get("corridor_width", 3.0))),
        seed=int(overrides.pop("seed", synthetic_cfg.get("seed", 2027))),
        robot_speed=float(overrides.pop("robot_speed", synthetic_cfg.get("robot_speed", 1.25))),
        robot_max_speed=float(overrides.pop("robot_max_speed", synthetic_cfg.get("robot_max_speed", 2.4))),
        entity_noise=float(overrides.pop("entity_noise", synthetic_cfg.get("entity_noise", 0.03))),
        role_mix=overrides.pop("role_mix", synthetic_cfg.get("role_mix", None)),
    )
    return SyntheticNavigationDataset(spec)
