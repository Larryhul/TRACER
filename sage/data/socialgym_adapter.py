from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from sage.data.heterogeneous_scene import AgentRole, HeterogeneousSceneBatch
from sage.data.sdd_adapter import ROLE_DEFAULT_GEOMETRY
from sage.data.synthetic_navigation import (
    ROLE_SPEED_RANGES,
    SyntheticNavigationDataset,
    SyntheticNavigationSpec,
    _unit,
)


SOCIALGYM_SCENES = ("doorway", "hallway", "intersection", "roundabout", "open")

ROLE_ALIASES = {
    "robot": int(AgentRole.ROBOT),
    "eaa": int(AgentRole.ROBOT),
    "agent": int(AgentRole.ROBOT),
    "human": int(AgentRole.PHA),
    "person": int(AgentRole.PHA),
    "pedestrian": int(AgentRole.PHA),
    "pha": int(AgentRole.PHA),
    "walker": int(AgentRole.PHA),
    "hnaa": int(AgentRole.HNAA),
    "bike": int(AgentRole.HNAA),
    "biker": int(AgentRole.HNAA),
    "bicycle": int(AgentRole.HNAA),
    "cyclist": int(AgentRole.HNAA),
    "skater": int(AgentRole.HNAA),
    "scooter": int(AgentRole.HNAA),
    "ssaa": int(AgentRole.SSAA),
    "vehicle": int(AgentRole.SSAA),
    "car": int(AgentRole.SSAA),
    "cart": int(AgentRole.SSAA),
    "unknown": int(AgentRole.UNKNOWN_ENTITY),
    "entity": int(AgentRole.UNKNOWN_ENTITY),
}


@dataclass(frozen=True)
class SocialGymStyleSpec:
    split: str = "train"
    num_samples: int = 4096
    history: int = 8
    horizon: int = 12
    num_robots: int = 6
    num_entities: int = 40
    dt: float = 0.4
    scene: str = "mixed"
    area_size: float = 14.0
    corridor_width: float = 3.0
    doorway_width: float = 1.8
    roundabout_radius: float = 2.4
    seed: int = 31415
    robot_speed: float = 1.20
    robot_max_speed: float = 2.4
    entity_noise: float = 0.035
    role_mix: Mapping[str, float] | None = None


class SocialGymStyleDataset(SyntheticNavigationDataset):
    """SocialGym-style 2D benchmark episodes for SAGE.

    SocialGym/SocialGym2 are online simulators rather than static trajectory
    datasets. This dataset mirrors their common benchmark semantics -- doorway,
    hallway, intersection, roundabout, and open crowd navigation -- while
    producing the same HeterogeneousSceneBatch schema used by SAGE. It lets the
    paper pipeline run without a ROS/Docker dependency, and exported SocialGym2
    rollouts can be loaded by SocialGymTrajectoryDataset below.
    """

    def __init__(self, spec: SocialGymStyleSpec) -> None:
        self.socialgym_spec = spec
        synthetic_spec = SyntheticNavigationSpec(
            split=spec.split,
            num_samples=spec.num_samples,
            history=spec.history,
            horizon=spec.horizon,
            num_robots=spec.num_robots,
            num_entities=spec.num_entities,
            dt=spec.dt,
            layout=spec.scene,
            area_size=spec.area_size,
            corridor_width=spec.corridor_width,
            seed=spec.seed,
            robot_speed=spec.robot_speed,
            robot_max_speed=spec.robot_max_speed,
            entity_noise=spec.entity_noise,
            role_mix=spec.role_mix or {"pha": 0.72, "hnaa": 0.22, "ssaa": 0.06},
        )
        super().__init__(synthetic_spec)
        self.scene = str(spec.scene).lower()
        self.doorway_width = float(spec.doorway_width)
        self.roundabout_radius = float(spec.roundabout_radius)

    def _layout_for_index(self, rng: np.random.Generator, idx: int) -> str:
        if self.scene != "mixed":
            return self.scene
        return SOCIALGYM_SCENES[int(idx) % len(SOCIALGYM_SCENES)]

    def _robot_tasks(self, rng: np.random.Generator, layout: str) -> tuple[np.ndarray, np.ndarray]:
        if layout in {"hallway", "corridor"}:
            return super()._robot_tasks(rng, "corridor")
        if layout in {"door", "doorway"}:
            return self._doorway_robot_tasks(rng)
        if layout == "roundabout":
            return self._roundabout_robot_tasks(rng)
        if layout == "intersection":
            return super()._robot_tasks(rng, "intersection")
        return super()._robot_tasks(rng, "open")

    def _doorway_robot_tasks(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        n = self.num_robots
        half = float(self.spec.area_size) * 0.5
        lanes = np.linspace(-0.45, 0.45, max(n, 2), dtype=np.float32)[:n]
        rng.shuffle(lanes)
        side = rng.choice([-1.0, 1.0], size=n)
        starts = np.zeros((n, 2), dtype=np.float32)
        goals = np.zeros((n, 2), dtype=np.float32)
        starts[:, 0] = -side * (half - 0.7)
        starts[:, 1] = lanes * self.doorway_width + rng.normal(0.0, 0.10, size=n)
        goals[:, 0] = side * (half - 0.7)
        goals[:, 1] = -0.35 * starts[:, 1] + rng.normal(0.0, 0.15, size=n)
        return starts.astype(np.float32), goals.astype(np.float32)

    def _roundabout_robot_tasks(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        n = self.num_robots
        half = float(self.spec.area_size) * 0.5
        radius = min(self.roundabout_radius + 2.4, half - 0.7)
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + rng.normal(0.0, 0.18, size=n)
        starts = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radius
        # Encourage lane-changing around a central shared space rather than pure antipodal crossings.
        delta = rng.choice([0.55 * np.pi, 0.75 * np.pi, 1.05 * np.pi], size=n)
        goals_angle = angles + delta
        goals = np.stack([np.cos(goals_angle), np.sin(goals_angle)], axis=-1) * radius
        starts += rng.normal(0.0, 0.08, size=starts.shape)
        goals += rng.normal(0.0, 0.12, size=goals.shape)
        return starts.astype(np.float32), goals.astype(np.float32)

    def _entity_motion(
        self,
        rng: np.random.Generator,
        roles: np.ndarray,
        robot_starts: np.ndarray,
        robot_goals: np.ndarray,
        layout: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if layout in {"hallway", "doorway", "roundabout"}:
            return self._socialgym_entity_motion(rng, roles, robot_starts, robot_goals, layout)
        return super()._entity_motion(rng, roles, robot_starts, robot_goals, layout)

    def _socialgym_entity_motion(
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
        half = float(self.spec.area_size) * 0.5
        positions = np.zeros((e, total, 2), dtype=np.float32)
        door_half = max(0.35, self.doorway_width * 0.5)

        for i in range(e):
            role = int(roles[i])
            speed_low, speed_high = ROLE_SPEED_RANGES.get(role, (0.45, 1.4))
            speed = rng.uniform(speed_low, speed_high)
            if layout == "hallway":
                sign = rng.choice([-1.0, 1.0])
                current = np.array(
                    [rng.uniform(-half * 0.8, half * 0.8), rng.normal(0.0, self.spec.corridor_width * 0.30)],
                    dtype=np.float32,
                )
                direction = np.array([sign, rng.normal(0.0, 0.08)], dtype=np.float32)
            elif layout == "doorway":
                sign = rng.choice([-1.0, 1.0])
                near_door = rng.random() < 0.72
                current = np.array(
                    [rng.normal(0.0, 1.2) if near_door else rng.uniform(-half * 0.85, half * 0.85),
                     rng.normal(0.0, door_half * 0.75) if near_door else rng.uniform(-half * 0.45, half * 0.45)],
                    dtype=np.float32,
                )
                direction = np.array([sign, -0.18 * current[1] + rng.normal(0.0, 0.10)], dtype=np.float32)
            else:  # roundabout
                radius = rng.uniform(max(1.2, self.roundabout_radius - 0.5), self.roundabout_radius + 1.3)
                angle = rng.uniform(-np.pi, np.pi)
                current = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * radius
                tangent = np.array([-np.sin(angle), np.cos(angle)], dtype=np.float32)
                direction = tangent * rng.choice([-1.0, 1.0]) + 0.18 * _unit(-current[None, :])[0]
            direction = _unit(direction[None, :])[0]
            velocity = direction * speed
            turn = rng.normal(0.0, 0.025 if layout != "roundabout" else 0.055)

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
            positions[i] = np.clip(positions[i], -half, half)

        history = positions[:, : self.history]
        future_pos = positions[:, self.history :]
        current = history[:, -1:]
        future_vel = np.diff(np.concatenate([current, future_pos], axis=1), axis=1) / self.dt
        return history.astype(np.float32), future_vel.astype(np.float32)

    def _robot_expert_future(
        self,
        starts: np.ndarray,
        goals: np.ndarray,
        entity_history: np.ndarray,
        entity_future_vel: np.ndarray,
        entity_roles: np.ndarray,
    ) -> np.ndarray:
        # Start from the generic role-aware expert and add mild doorway/roundabout structure.
        return super()._robot_expert_future(starts, goals, entity_history, entity_future_vel, entity_roles)


class SocialGymTrajectoryDataset(Dataset):
    """Load offline trajectories exported from SocialGym/SocialGym2 rollouts.

    Supported inputs:
    - .npz files with either pre-windowed SAGE arrays or long trajectories.
    - .pt/.pth files storing a dict with the same arrays.
    - .csv long format with columns episode, agent_id, t, x, y plus optional
      role, is_robot, goal_x, goal_y, length, width.
    - .jsonl episodes with an agents list, each containing trajectory and role.

    Future targets may be supplied as velocities. If positions are supplied, the
    dataset converts them to velocities over the configured dt.
    """

    def __init__(
        self,
        path: str | Path,
        split: str = "train",
        history: int = 8,
        horizon: int = 12,
        dt: float = 0.4,
        max_agents: int | None = None,
        max_entities: int | None = None,
        num_robots: int | None = None,
        stride: int = 1,
    ) -> None:
        self.path = Path(path)
        self.split = str(split)
        self.history = int(history)
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.max_agents = max_agents
        self.max_entities = max_entities
        self.num_robots = num_robots
        self.stride = max(1, int(stride))
        if not self.path.exists():
            raise FileNotFoundError(f"SocialGym trajectory path does not exist: {self.path}")
        source = self._resolve_source(self.path, self.split)
        self.samples = self._load_source(source)
        if not self.samples:
            raise ValueError(f"No SocialGym trajectory windows were loaded from {source}")

    def _resolve_source(self, path: Path, split: str) -> Path:
        if path.is_file():
            return path
        candidates = [
            path / f"{split}.npz",
            path / f"socialgym_{split}.npz",
            path / f"{split}.pt",
            path / f"socialgym_{split}.pt",
            path / f"{split}.jsonl",
            path / f"episodes_{split}.jsonl",
            path / f"{split}.csv",
            path / f"rollouts_{split}.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No SocialGym trajectory file for split={split!r} under {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float | str]:
        return self.samples[int(idx)]

    def _load_source(self, source: Path) -> list[dict[str, torch.Tensor | float | str]]:
        suffix = source.suffix.lower()
        if suffix == ".npz":
            return self._load_npz(source)
        if suffix in {".pt", ".pth"}:
            data = torch.load(source, map_location="cpu")
            return self._load_mapping(data, source)
        if suffix == ".csv":
            return self._load_csv(source)
        if suffix in {".jsonl", ".ndjson"}:
            return self._load_jsonl(source)
        raise ValueError(f"Unsupported SocialGym trajectory file type: {source.suffix}")

    def _load_npz(self, source: Path) -> list[dict[str, torch.Tensor | float | str]]:
        with np.load(source, allow_pickle=True) as data:
            mapping = {key: data[key] for key in data.files}
        return self._load_mapping(mapping, source)

    def _load_mapping(self, data: Mapping[str, Any], source: Path) -> list[dict[str, torch.Tensor | float | str]]:
        keys = set(data.keys())
        if {"history", "future"}.issubset(keys):
            return self._load_prewindowed(data, source)
        if {"positions", "roles"}.issubset(keys):
            return self._windows_from_dense_positions(data, source)
        if {"episodes"}.issubset(keys):
            return self._windows_from_episode_objects(data["episodes"], source)
        raise ValueError(
            f"Unsupported SocialGym mapping in {source}; expected history/future arrays, positions arrays, or episodes."
        )

    def _load_prewindowed(self, data: Mapping[str, Any], source: Path) -> list[dict[str, torch.Tensor | float | str]]:
        history = np.asarray(data["history"], dtype=np.float32)
        future = np.asarray(data["future"], dtype=np.float32)
        if future.shape[-1] != 2:
            raise ValueError("future must have last dimension 2")
        if history.ndim == 3:
            history = history[None]
        if future.ndim == 3:
            future = future[None]
        if future.shape[-2] == self.horizon and _bool_value(data.get("future_is_position", False)):
            current = history[:, :, -1:, :2]
            future = np.diff(np.concatenate([current, future], axis=2), axis=2) / self.dt
        n, a = history.shape[:2]
        role_arr = np.asarray(data.get("roles", np.full(a, int(AgentRole.PHA), dtype=np.int64)), dtype=object)
        robot_arr = np.asarray(data.get("robot_mask", _robot_mask_from_roles(role_arr)))
        if robot_arr.shape == () or robot_arr.size != (n * a if robot_arr.ndim == 2 else a):
            robot_arr = role_arr == int(AgentRole.ROBOT)
        if not np.asarray(robot_arr).any():
            robot_arr = np.zeros((a,), dtype=bool)
            limit = int(self.num_robots or 1)
            robot_arr[: max(1, min(limit, a))] = True
        goals_arr = None if "goals" not in data else np.asarray(data["goals"], dtype=np.float32)
        geom_arr = None if "geometry" not in data else np.asarray(data["geometry"], dtype=np.float32)
        scene_arr = np.asarray(data.get("scene_id", ["socialgym"])).reshape(-1)
        samples = []
        for i in range(n):
            roles_i = role_arr[i] if role_arr.ndim == 2 else role_arr
            robots_i = robot_arr[i] if np.asarray(robot_arr).ndim == 2 else robot_arr
            sample = self._finalize_window(
                history=history[i],
                future=future[i],
                roles=roles_i,
                robot_mask=robots_i,
                goals=None if goals_arr is None else (goals_arr[i] if goals_arr.ndim == 3 else goals_arr),
                geometry=None if geom_arr is None else (geom_arr[i] if geom_arr.ndim == 3 else geom_arr),
                scene_id=str(scene_arr[min(i, scene_arr.size - 1)]),
            )
            samples.append(sample)
        return samples

    def _windows_from_dense_positions(self, data: Mapping[str, Any], source: Path) -> list[dict[str, torch.Tensor | float | str]]:
        positions = np.asarray(data["positions"], dtype=np.float32)
        # Accept (E,A,T,2), (A,T,2), or (T,A,2).
        if positions.ndim == 3:
            if positions.shape[0] > positions.shape[1]:
                positions = np.transpose(positions, (1, 0, 2))[None]
            else:
                positions = positions[None]
        roles = np.asarray(data.get("roles", np.full(positions.shape[1], int(AgentRole.PHA))), dtype=object)
        robot_mask = np.asarray(data.get("robot_mask", _robot_mask_from_roles(roles)))
        goals = data.get("goals", None)
        geometry = data.get("geometry", None)
        samples: list[dict[str, torch.Tensor | float | str]] = []
        total = self.history + self.horizon
        for ep in range(positions.shape[0]):
            ep_pos = positions[ep]
            ep_roles = roles[ep] if roles.ndim == 2 else roles
            ep_robot = robot_mask[ep] if robot_mask.ndim == 2 else robot_mask
            ep_goals = None if goals is None else (np.asarray(goals)[ep] if np.asarray(goals).ndim == 3 else np.asarray(goals))
            ep_geom = None if geometry is None else (np.asarray(geometry)[ep] if np.asarray(geometry).ndim == 3 else np.asarray(geometry))
            for start in range(0, ep_pos.shape[1] - total + 1, self.stride):
                window = ep_pos[:, start : start + total]
                history = window[:, : self.history]
                future_pos = window[:, self.history :]
                current = history[:, -1:]
                future_vel = np.diff(np.concatenate([current, future_pos], axis=1), axis=1) / self.dt
                samples.append(self._finalize_window(history, future_vel, ep_roles, ep_robot, ep_goals, ep_geom, source.stem))
        return samples

    def _load_csv(self, source: Path) -> list[dict[str, torch.Tensor | float | str]]:
        rows_by_episode: dict[str, list[dict[str, str]]] = {}
        with source.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episode = row.get("episode") or row.get("episode_id") or row.get("scene") or "0"
                rows_by_episode.setdefault(str(episode), []).append(row)
        episodes = []
        for episode_id, rows in rows_by_episode.items():
            agents: dict[str, dict[str, Any]] = {}
            for row in rows:
                aid = str(row.get("agent_id") or row.get("track_id") or row.get("id"))
                if aid == "None":
                    continue
                agent = agents.setdefault(aid, {"id": aid, "states": []})
                agent["states"].append(row)
                for key in ("role", "type", "category", "is_robot", "goal_x", "goal_y", "length", "width"):
                    if row.get(key) not in {None, ""}:
                        agent[key] = row[key]
            episodes.append({"episode_id": episode_id, "agents": list(agents.values())})
        return self._windows_from_episode_objects(episodes, source)

    def _load_jsonl(self, source: Path) -> list[dict[str, torch.Tensor | float | str]]:
        episodes = []
        with source.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        return self._windows_from_episode_objects(episodes, source)

    def _windows_from_episode_objects(self, episodes: Iterable[Any], source: Path) -> list[dict[str, torch.Tensor | float | str]]:
        samples: list[dict[str, torch.Tensor | float | str]] = []
        total = self.history + self.horizon
        for ep_no, episode in enumerate(episodes):
            agents_obj = episode.get("agents", episode.get("tracks", [])) if isinstance(episode, Mapping) else []
            parsed = []
            for agent in agents_obj:
                traj = self._trajectory_from_agent(agent)
                if traj.shape[0] < total:
                    continue
                role_id = _role_id(agent.get("role", agent.get("type", agent.get("category", "human"))))
                is_robot = _bool_value(agent.get("is_robot", role_id == int(AgentRole.ROBOT)))
                if is_robot:
                    role_id = int(AgentRole.ROBOT)
                goal = agent.get("goal", None)
                if goal is None and "goal_x" in agent and "goal_y" in agent:
                    goal = [float(agent["goal_x"]), float(agent["goal_y"])]
                geometry = agent.get("geometry", None)
                if geometry is None and "length" in agent and "width" in agent:
                    geometry = [float(agent["length"]), float(agent["width"])]
                parsed.append((traj, role_id, is_robot, goal, geometry))
            if not parsed:
                continue
            min_len = min(p[0].shape[0] for p in parsed)
            for start in range(0, min_len - total + 1, self.stride):
                trajs = np.stack([p[0][start : start + total] for p in parsed], axis=0).astype(np.float32)
                roles = np.asarray([p[1] for p in parsed], dtype=np.int64)
                robot_mask = np.asarray([p[2] for p in parsed], dtype=bool)
                goals = np.asarray(
                    [p[3] if p[3] is not None else trajs[i, -1] for i, p in enumerate(parsed)],
                    dtype=np.float32,
                )
                geometry = np.asarray(
                    [p[4] if p[4] is not None else ROLE_DEFAULT_GEOMETRY[int(roles[i])] for i, p in enumerate(parsed)],
                    dtype=np.float32,
                )
                history = trajs[:, : self.history]
                future_pos = trajs[:, self.history :]
                current = history[:, -1:]
                future_vel = np.diff(np.concatenate([current, future_pos], axis=1), axis=1) / self.dt
                scene_id = str(episode.get("scene_id", episode.get("episode_id", f"{source.stem}_{ep_no}"))) if isinstance(episode, Mapping) else source.stem
                samples.append(self._finalize_window(history, future_vel, roles, robot_mask, goals, geometry, scene_id))
        return samples

    def _trajectory_from_agent(self, agent: Mapping[str, Any]) -> np.ndarray:
        if "trajectory" in agent:
            traj = np.asarray(agent["trajectory"], dtype=np.float32)
        elif "states" in agent:
            states = sorted(agent["states"], key=lambda r: float(r.get("t", r.get("time", r.get("step", 0)))))
            traj = np.asarray([[float(r["x"]), float(r["y"])] for r in states], dtype=np.float32)
        else:
            raise ValueError("agent must contain trajectory or states")
        if traj.ndim != 2 or traj.shape[-1] < 2:
            raise ValueError("trajectory must be (T,2+) positions")
        return traj[:, :2]

    def _finalize_window(
        self,
        history: np.ndarray,
        future: np.ndarray,
        roles: np.ndarray,
        robot_mask: np.ndarray,
        goals: np.ndarray | None,
        geometry: np.ndarray | None,
        scene_id: str,
    ) -> dict[str, torch.Tensor | float | str]:
        history = np.asarray(history, dtype=np.float32)[..., :2]
        future = np.asarray(future, dtype=np.float32)[..., :2]
        roles = np.asarray([_role_id(x) for x in np.asarray(roles).reshape(-1)], dtype=np.int64)
        robot_mask = np.asarray(robot_mask, dtype=bool).reshape(-1)
        if robot_mask.size != roles.size:
            robot_mask = roles == int(AgentRole.ROBOT)
        roles = np.where(robot_mask, int(AgentRole.ROBOT), roles)

        keep = np.ones(roles.shape[0], dtype=bool)
        if self.num_robots is not None:
            robot_indices = np.where(robot_mask)[0][: int(self.num_robots)]
            keep_robot = np.zeros_like(keep)
            keep_robot[robot_indices] = True
        else:
            keep_robot = robot_mask
        if self.max_entities is not None:
            entity_indices = np.where(~robot_mask)[0][: int(self.max_entities)]
            keep_entity = np.zeros_like(keep)
            keep_entity[entity_indices] = True
        else:
            keep_entity = ~robot_mask
        keep = keep_robot | keep_entity
        candidate_indices = np.where(keep)[0]
        if self.max_agents is not None:
            max_agents = int(self.max_agents)
            robot_candidates = candidate_indices[robot_mask[candidate_indices]]
            entity_candidates = candidate_indices[~robot_mask[candidate_indices]]
            selected = np.concatenate([robot_candidates, entity_candidates], axis=0)[:max_agents]
        else:
            selected = candidate_indices
        if selected.size == 0:
            raise ValueError("SocialGym window has no selected agents")

        history = history[selected]
        future = future[selected]
        roles = roles[selected]
        robot_mask = robot_mask[selected]
        if not robot_mask.any():
            robot_mask[0] = True
            roles[0] = int(AgentRole.ROBOT)
        agent_mask = np.ones(selected.size, dtype=bool)
        entity_mask = agent_mask & ~robot_mask
        if goals is None:
            goals_arr = history[:, -1] + future.sum(axis=1) * self.dt
        else:
            goals_arr = np.asarray(goals, dtype=np.float32).reshape(-1, 2)[selected]
        goals_arr = np.where(robot_mask[:, None], goals_arr, history[:, -1])
        if geometry is None:
            geom = np.asarray([ROLE_DEFAULT_GEOMETRY[int(role)] for role in roles], dtype=np.float32)
        else:
            geom = np.asarray(geometry, dtype=np.float32).reshape(-1, 2)[selected]

        return {
            "history": torch.tensor(history, dtype=torch.float32),
            "future": torch.tensor(future, dtype=torch.float32),
            "current_pos": torch.tensor(history[:, -1, :2], dtype=torch.float32),
            "roles": torch.tensor(roles, dtype=torch.long),
            "agent_mask": torch.tensor(agent_mask, dtype=torch.bool),
            "robot_mask": torch.tensor(robot_mask, dtype=torch.bool),
            "entity_mask": torch.tensor(entity_mask, dtype=torch.bool),
            "loss_mask": torch.tensor(agent_mask, dtype=torch.bool),
            "geometry": torch.tensor(geom, dtype=torch.float32),
            "goals": torch.tensor(goals_arr, dtype=torch.float32),
            "goal_mask": torch.tensor(robot_mask, dtype=torch.bool),
            "dt": float(self.dt),
            "scene_id": scene_id,
        }


def _role_id(value: Any) -> int:
    if isinstance(value, (np.integer, int)):
        ivalue = int(value)
        if ivalue in {int(r) for r in AgentRole}:
            return ivalue
    text = str(value).strip().lower()
    return ROLE_ALIASES.get(text, int(AgentRole.UNKNOWN_ENTITY))


def _bool_value(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        if value.size == 1:
            return _bool_value(value.reshape(-1)[0].item())
        return bool(value.any())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"", "0", "false", "f", "no", "n", "none", "null"}:
        return False
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    return bool(value)

def _robot_mask_from_roles(roles: Any) -> np.ndarray:
    role_arr = np.asarray(roles, dtype=object)
    flat = np.asarray([_role_id(x) == int(AgentRole.ROBOT) for x in role_arr.reshape(-1)], dtype=bool)
    return flat.reshape(role_arr.shape)


def _pad_tensor(tensor: torch.Tensor, target_agents: int, pad_value: float | int | bool = 0) -> torch.Tensor:
    if tensor.shape[0] == target_agents:
        return tensor
    pad_shape = (target_agents - tensor.shape[0], *tensor.shape[1:])
    if tensor.dtype == torch.bool:
        pad = torch.full(pad_shape, bool(pad_value), dtype=tensor.dtype)
    else:
        pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=0)


def collate_socialgym_scene_batch(samples: Sequence[dict]) -> HeterogeneousSceneBatch:
    if not samples:
        raise ValueError("cannot collate an empty SocialGym batch")
    max_agents = max(int(s["history"].shape[0]) for s in samples)
    batch = HeterogeneousSceneBatch(
        history=torch.stack([_pad_tensor(s["history"], max_agents, 0.0) for s in samples], dim=0),
        future=torch.stack([_pad_tensor(s["future"], max_agents, 0.0) for s in samples], dim=0),
        current_pos=torch.stack([_pad_tensor(s["current_pos"], max_agents, 0.0) for s in samples], dim=0),
        roles=torch.stack([_pad_tensor(s["roles"], max_agents, int(AgentRole.UNKNOWN_ENTITY)) for s in samples], dim=0),
        agent_mask=torch.stack([_pad_tensor(s["agent_mask"], max_agents, False) for s in samples], dim=0),
        robot_mask=torch.stack([_pad_tensor(s["robot_mask"], max_agents, False) for s in samples], dim=0),
        entity_mask=torch.stack([_pad_tensor(s["entity_mask"], max_agents, False) for s in samples], dim=0),
        loss_mask=torch.stack([_pad_tensor(s["loss_mask"], max_agents, False) for s in samples], dim=0),
        geometry=torch.stack([_pad_tensor(s["geometry"], max_agents, 0.0) for s in samples], dim=0),
        goals=torch.stack([_pad_tensor(s["goals"], max_agents, 0.0) for s in samples], dim=0),
        goal_mask=torch.stack([_pad_tensor(s["goal_mask"], max_agents, False) for s in samples], dim=0),
        dt=float(samples[0].get("dt", 0.4)),
    )
    batch.validate()
    return batch


def make_socialgym_dataset_from_config(raw_cfg: Mapping, split: str = "train", **overrides) -> Dataset:
    dataset_cfg = raw_cfg.get("dataset", {})
    socialgym_cfg = dataset_cfg.get("socialgym", {})
    source = str(overrides.pop("source", socialgym_cfg.get("source", "style"))).lower()
    history = int(overrides.pop("history", dataset_cfg.get("history", 8)))
    horizon = int(overrides.pop("horizon", dataset_cfg.get("horizon", 12)))
    dt = float(overrides.pop("dt", dataset_cfg.get("dt", 0.4)))
    if source in {"trajectory", "rollout", "file", "offline"}:
        data_path = overrides.pop("data_path", socialgym_cfg.get("data_path", ""))
        if not data_path:
            raise ValueError("dataset.socialgym.data_path is required when source='trajectory'")
        return SocialGymTrajectoryDataset(
            data_path,
            split=split,
            history=history,
            horizon=horizon,
            dt=dt,
            max_agents=overrides.pop("max_agents", socialgym_cfg.get("max_agents", None)),
            max_entities=overrides.pop("num_entities", overrides.pop("max_entities", socialgym_cfg.get("num_entities", dataset_cfg.get("max_entities", None)))),
            num_robots=overrides.pop("num_robots", socialgym_cfg.get("num_robots", dataset_cfg.get("num_robot_proxies", None))),
            stride=overrides.pop("window_stride", socialgym_cfg.get("window_stride", 1)),
        )

    split_key = f"{split}_samples"
    num_samples = int(overrides.pop("num_samples", socialgym_cfg.get(split_key, socialgym_cfg.get("samples", 4096))))
    spec = SocialGymStyleSpec(
        split=split,
        num_samples=num_samples,
        history=history,
        horizon=horizon,
        num_robots=int(overrides.pop("num_robots", socialgym_cfg.get("num_robots", dataset_cfg.get("num_robot_proxies", 6)))),
        num_entities=int(overrides.pop("num_entities", socialgym_cfg.get("num_entities", dataset_cfg.get("max_entities", 40)))),
        dt=dt,
        scene=str(overrides.pop("scene", overrides.pop("layout", socialgym_cfg.get("scene", "mixed")))),
        area_size=float(overrides.pop("area_size", socialgym_cfg.get("area_size", 14.0))),
        corridor_width=float(overrides.pop("corridor_width", socialgym_cfg.get("corridor_width", 3.0))),
        doorway_width=float(overrides.pop("doorway_width", socialgym_cfg.get("doorway_width", 1.8))),
        roundabout_radius=float(overrides.pop("roundabout_radius", socialgym_cfg.get("roundabout_radius", 2.4))),
        seed=int(overrides.pop("seed", socialgym_cfg.get("seed", 31415))),
        robot_speed=float(overrides.pop("robot_speed", socialgym_cfg.get("robot_speed", 1.20))),
        robot_max_speed=float(overrides.pop("robot_max_speed", socialgym_cfg.get("robot_max_speed", 2.4))),
        entity_noise=float(overrides.pop("entity_noise", socialgym_cfg.get("entity_noise", 0.035))),
        role_mix=overrides.pop("role_mix", socialgym_cfg.get("role_mix", None)),
    )
    return SocialGymStyleDataset(spec)
