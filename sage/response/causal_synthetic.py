from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from sage.data.heterogeneous_scene import AgentRole
from sage.data.sdd_adapter import ROLE_DEFAULT_GEOMETRY


VALID_EFFECT_MODES = {"none", "additive", "nonlinear"}
VALID_SCENE_FAMILIES = {"radial", "corridor", "crossing", "bottleneck"}
VALID_RESPONSE_FAMILIES = {"pincer", "speed", "lateral", "threshold"}


def _unit(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), eps)


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector * np.minimum(1.0, float(max_norm) / np.maximum(norm, 1e-6))


@dataclass(frozen=True)
class CausalResponseSpec:
    split: str = "train"
    num_base_scenes: int = 256
    interventions_per_scene: int = 4
    history: int = 8
    horizon: int = 12
    num_robots: int = 2
    num_entities: int = 12
    dt: float = 0.4
    seed: int = 2027
    effect_mode: str = "nonlinear"
    area_size: float = 10.0
    influence_radius: float = 4.0
    individual_gain: float = 1.0
    joint_gain: float = 3.0
    joint_slowdown: float = 1.0
    joint_lateral_gain: float = 1.25
    entity_noise: float = 0.015
    entity_max_speed: float = 3.0
    scene_family: str = "radial"
    response_family: str = "pincer"
    cache_samples: bool = False


class CausalTeamResponseDataset(Dataset):
    """Paired interventions with the causal order team action -> entity response."""

    def __init__(self, spec: CausalResponseSpec) -> None:
        mode = str(spec.effect_mode).lower()
        if mode not in VALID_EFFECT_MODES:
            raise ValueError(f"effect_mode must be one of {sorted(VALID_EFFECT_MODES)}, got {mode!r}")
        scene_family = str(spec.scene_family).lower()
        if scene_family not in VALID_SCENE_FAMILIES:
            raise ValueError(
                f"scene_family must be one of {sorted(VALID_SCENE_FAMILIES)}, got {scene_family!r}"
            )
        response_family = str(spec.response_family).lower()
        if response_family not in VALID_RESPONSE_FAMILIES:
            raise ValueError(
                "response_family must be one of "
                f"{sorted(VALID_RESPONSE_FAMILIES)}, got {response_family!r}"
            )
        if spec.num_robots < 2:
            raise ValueError("causal response experiment requires at least two robots")
        expected_interventions = 2 ** int(spec.num_robots)
        if spec.interventions_per_scene != expected_interventions:
            raise ValueError(
                "factorial response experiment requires exactly "
                f"2**num_robots={expected_interventions} interventions per scene"
            )
        self.spec = spec
        self.effect_mode = mode
        self.scene_family = scene_family
        self.response_family = response_family
        self.num_base_scenes = int(spec.num_base_scenes)
        self.interventions_per_scene = int(spec.interventions_per_scene)
        self.history = int(spec.history)
        self.horizon = int(spec.horizon)
        self.num_robots = int(spec.num_robots)
        self.num_entities = int(spec.num_entities)
        self.num_agents = self.num_robots + self.num_entities
        self.dt = float(spec.dt)
        self.cache_samples = bool(spec.cache_samples)
        self._sample_cache: dict[int, dict[str, torch.Tensor | str | int | float]] = {}
        self._scene_sample_cache: dict[
            int, tuple[dict[str, torch.Tensor | str | int | float], ...]
        ] = {}
        split_offsets = {"train": 0, "val": 1_000_000, "test": 2_000_000}
        self.split_offset = split_offsets.get(str(spec.split).lower(), 3_000_000)
        self.seed = int(spec.seed) + self.split_offset

    def __len__(self) -> int:
        return self.num_base_scenes * self.interventions_per_scene

    def _base_rng(self, base_scene_id: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + int(base_scene_id) * 104729)

    def _action_rng(self, base_scene_id: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + int(base_scene_id) * 104729 + 15485863)

    def _base_scene(self, base_scene_id: int) -> dict[str, np.ndarray]:
        rng = self._base_rng(base_scene_id)
        half = float(self.spec.area_size) * 0.5
        family = self.scene_family
        if family == "radial":
            robot_angles = np.linspace(0.0, 2.0 * np.pi, self.num_robots, endpoint=False)
            robot_angles += rng.uniform(-0.45, 0.45, size=self.num_robots)
            robot_radius = rng.uniform(2.2, 3.4, size=self.num_robots)
            robot_current = np.stack(
                [np.cos(robot_angles), np.sin(robot_angles)], axis=-1
            ) * robot_radius[:, None]
            entity_current = rng.uniform(-1.8, 1.8, size=(self.num_entities, 2))
        elif family == "corridor":
            corridor_width = max(1.2, self.spec.area_size * 0.24)
            robot_current = np.zeros((self.num_robots, 2), dtype=np.float32)
            side = np.where(np.arange(self.num_robots) % 2 == 0, -1.0, 1.0)
            robot_current[:, 0] = rng.uniform(-half + 1.0, half - 1.0, self.num_robots)
            robot_current[:, 1] = side * rng.uniform(corridor_width, half - 1.0, self.num_robots)
            entity_current = np.stack(
                [
                    rng.uniform(-half * 0.65, half * 0.65, self.num_entities),
                    rng.normal(0.0, corridor_width * 0.42, self.num_entities),
                ],
                axis=-1,
            )
        elif family == "crossing":
            horizontal_count = max(1, self.num_robots // 2)
            horizontal = np.stack(
                [
                    rng.uniform(-half + 1.0, half - 1.0, horizontal_count),
                    rng.normal(0.0, 0.35, horizontal_count),
                ],
                axis=-1,
            )
            vertical_count = self.num_robots - horizontal_count
            vertical = (
                np.stack(
                    [
                        rng.normal(0.0, 0.35, vertical_count),
                        rng.uniform(-half + 1.0, half - 1.0, vertical_count),
                    ],
                    axis=-1,
                )
                if vertical_count
                else np.empty((0, 2))
            )
            robot_current = np.concatenate([horizontal, vertical], axis=0)
            entity_current = rng.normal(0.0, 1.25, size=(self.num_entities, 2))
        else:  # bottleneck
            robot_current = np.zeros((self.num_robots, 2), dtype=np.float32)
            sides = np.where(np.arange(self.num_robots) % 2 == 0, -1.0, 1.0)
            robot_current[:, 0] = sides * rng.uniform(2.0, half - 0.8, self.num_robots)
            robot_current[:, 1] = rng.normal(0.0, 0.7, self.num_robots)
            entity_current = rng.normal(0.0, 0.8, size=(self.num_entities, 2))

        robot_current += rng.normal(0.0, 0.15, size=robot_current.shape)
        robot_current = np.clip(robot_current, -half + 0.5, half - 0.5).astype(np.float32)
        entity_current = np.asarray(entity_current, dtype=np.float32)
        entity_current = np.clip(entity_current, -half + 0.5, half - 0.5)
        entity_angles = rng.uniform(-np.pi, np.pi, size=self.num_entities)
        entity_speeds = rng.uniform(0.45, 1.15, size=self.num_entities)
        preferred_velocity = np.stack([np.cos(entity_angles), np.sin(entity_angles)], axis=-1)
        preferred_velocity = (preferred_velocity * entity_speeds[:, None]).astype(np.float32)

        entity_history = np.zeros((self.num_entities, self.history, 2), dtype=np.float32)
        history_noise = rng.normal(0.0, self.spec.entity_noise * 0.25, size=entity_history.shape)
        for history_idx in range(self.history):
            tau = history_idx - (self.history - 1)
            entity_history[:, history_idx] = (
                entity_current + preferred_velocity * (tau * self.dt) + history_noise[:, history_idx]
            )
        entity_history[:, -1] = entity_current

        robot_nominal_velocity = _unit(-robot_current) * rng.uniform(0.7, 1.1, size=(self.num_robots, 1))
        robot_history = np.zeros((self.num_robots, self.history, 2), dtype=np.float32)
        for history_idx in range(self.history):
            tau = history_idx - (self.history - 1)
            robot_history[:, history_idx] = robot_current + robot_nominal_velocity * (tau * self.dt)

        response_noise = rng.normal(
            0.0, self.spec.entity_noise, size=(self.num_entities, self.horizon, 2)
        ).astype(np.float32)
        return {
            "robot_current": robot_current,
            "robot_history": robot_history,
            "entity_current": entity_current,
            "entity_history": entity_history,
            "preferred_velocity": preferred_velocity,
            "response_noise": response_noise,
        }

    def _candidate_action(self, base_scene_id: int, intervention_id: int, robot_current: np.ndarray) -> np.ndarray:
        rng = self._action_rng(base_scene_id)
        base_direction = _unit(-robot_current)
        base_angle = np.arctan2(base_direction[:, 1], base_direction[:, 0])
        active = np.asarray(
            [(intervention_id >> robot_idx) & 1 for robot_idx in range(self.num_robots)],
            dtype=bool,
        )
        lateral_sign = rng.choice([-1.0, 1.0], size=self.num_robots)
        active_offsets = rng.normal(0.0, 0.12, size=self.num_robots)
        yielding_offsets = lateral_sign * rng.uniform(0.75, 1.10, size=self.num_robots)
        offsets = np.where(active, active_offsets, yielding_offsets)
        active_speeds = rng.uniform(1.35, 1.75, size=self.num_robots)
        yielding_speeds = rng.uniform(0.55, 0.85, size=self.num_robots)
        speeds = np.where(active, active_speeds, yielding_speeds)
        turns = rng.uniform(-0.10, 0.10, size=self.num_robots)
        candidate = np.zeros((self.num_robots, self.horizon, 2), dtype=np.float32)
        for step in range(self.horizon):
            angles = base_angle + offsets + turns * step
            candidate[:, step, 0] = np.cos(angles) * speeds
            candidate[:, step, 1] = np.sin(angles) * speeds
        return candidate

    def _candidate_actions(self, base_scene_id: int, robot_current: np.ndarray) -> np.ndarray:
        """Generate all factorial branches from one shared action random stream."""
        rng = self._action_rng(base_scene_id)
        base_direction = _unit(-robot_current)
        base_angle = np.arctan2(base_direction[:, 1], base_direction[:, 0])
        lateral_sign = rng.choice([-1.0, 1.0], size=self.num_robots)
        active_offsets = rng.normal(0.0, 0.12, size=self.num_robots)
        yielding_offsets = lateral_sign * rng.uniform(0.75, 1.10, size=self.num_robots)
        active_speeds = rng.uniform(1.35, 1.75, size=self.num_robots)
        yielding_speeds = rng.uniform(0.55, 0.85, size=self.num_robots)
        turns = rng.uniform(-0.10, 0.10, size=self.num_robots)
        active = np.asarray(
            [[(intervention_id >> robot_idx) & 1 for robot_idx in range(self.num_robots)]
             for intervention_id in range(self.interventions_per_scene)],
            dtype=bool,
        )
        offsets = np.where(active, active_offsets[None, :], yielding_offsets[None, :])
        speeds = np.where(active, active_speeds[None, :], yielding_speeds[None, :])
        steps = np.arange(self.horizon, dtype=np.float32)[None, None, :]
        angles = base_angle[None, :, None] + offsets[:, :, None] + turns[None, :, None] * steps
        candidate = np.stack([np.cos(angles) * speeds[:, :, None], np.sin(angles) * speeds[:, :, None]], axis=-1)
        return candidate.astype(np.float32)

    def _entity_response(
        self,
        base: dict[str, np.ndarray],
        robot_future: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        nominal_entity_pos = base["entity_current"].copy()
        robot_pos = base["robot_current"].copy()
        preferred = base["preferred_velocity"]
        future = np.zeros((self.num_entities, self.horizon, 2), dtype=np.float32)
        radius = float(self.spec.influence_radius)

        for step in range(self.horizon):
            robot_pos = robot_pos + robot_future[:, step] * self.dt
            nominal_entity_pos = nominal_entity_pos + preferred * self.dt
            away = nominal_entity_pos[:, None, :] - robot_pos[None, :, :]
            distance = np.linalg.norm(away, axis=-1)
            away_unit = _unit(away)
            approach_speed = np.sum(robot_future[None, :, step, :] * away_unit, axis=-1)
            # Continuous support avoids mechanism-degenerate scenes while still
            # making distant robots exponentially less influential.
            proximity = np.exp(-0.5 * np.square(distance / radius))
            approach_gate = 0.45 + 0.55 / (1.0 + np.exp(-2.0 * approach_speed))
            strength = proximity * approach_gate
            individual = (
                away_unit * strength[..., None] * float(self.spec.individual_gain)
            ).sum(axis=1)

            velocity = preferred.copy()
            if mode in {"additive", "nonlinear"}:
                velocity += individual
            if mode == "nonlinear":
                pair_synergy = np.zeros(self.num_entities, dtype=np.float32)
                for first in range(self.num_robots):
                    for second in range(first + 1, self.num_robots):
                        alignment = np.sum(away_unit[:, first] * away_unit[:, second], axis=-1)
                        opposition = np.clip((-alignment - 0.15) / 0.85, 0.0, 1.0)
                        pair_synergy += strength[:, first] * strength[:, second] * opposition
                if self.response_family == "threshold":
                    pair_synergy = 1.0 / (
                        1.0 + np.exp(-12.0 * (pair_synergy - 0.12))
                    ) * pair_synergy
                pair_synergy = np.clip(pair_synergy * float(self.spec.joint_gain), 0.0, 0.95)
                preferred_unit = _unit(preferred)
                perpendicular = np.stack([-preferred_unit[:, 1], preferred_unit[:, 0]], axis=-1)
                team_midpoint = robot_pos.mean(axis=0, keepdims=True)
                side = np.sign(
                    np.sum(perpendicular * (nominal_entity_pos - team_midpoint), axis=-1)
                )
                side[side == 0.0] = 1.0
                slowdown = float(self.spec.joint_slowdown)
                lateral_gain = float(self.spec.joint_lateral_gain)
                if self.response_family == "lateral":
                    slowdown = 0.0
                elif self.response_family == "speed":
                    lateral_gain = 0.0
                velocity = velocity * (1.0 - slowdown * pair_synergy[:, None])
                velocity += (
                    perpendicular
                    * side[:, None]
                    * (lateral_gain * pair_synergy[:, None])
                )

            velocity += base["response_noise"][:, step]
            if mode == "nonlinear":
                velocity = _clip_norm(velocity, float(self.spec.entity_max_speed))
            velocity = velocity.astype(np.float32)
            future[:, step] = velocity
        return future

    def _entity_responses(self, base: dict[str, np.ndarray], robot_future: np.ndarray) -> np.ndarray:
        """Vectorized counterpart of _entity_response for all four interventions."""
        nominal_entity_pos = np.broadcast_to(
            base["entity_current"], (self.interventions_per_scene, self.num_entities, 2)
        ).copy()
        robot_pos = np.broadcast_to(
            base["robot_current"], (self.interventions_per_scene, self.num_robots, 2)
        ).copy()
        preferred = np.broadcast_to(
            base["preferred_velocity"], (self.interventions_per_scene, self.num_entities, 2)
        )
        response_noise = np.broadcast_to(
            base["response_noise"],
            (self.interventions_per_scene, self.num_entities, self.horizon, 2),
        )
        future = np.zeros(
            (self.interventions_per_scene, self.num_entities, self.horizon, 2),
            dtype=np.float32,
        )
        radius = float(self.spec.influence_radius)

        for step in range(self.horizon):
            robot_pos = robot_pos + robot_future[:, :, step] * self.dt
            nominal_entity_pos = nominal_entity_pos + preferred * self.dt
            away = nominal_entity_pos[:, :, None, :] - robot_pos[:, None, :, :]
            distance = np.linalg.norm(away, axis=-1)
            away_unit = _unit(away)
            approach_speed = np.sum(robot_future[:, None, :, step, :] * away_unit, axis=-1)
            proximity = np.exp(-0.5 * np.square(distance / radius))
            approach_gate = 0.45 + 0.55 / (1.0 + np.exp(-2.0 * approach_speed))
            strength = proximity * approach_gate
            individual = (
                away_unit * strength[..., None] * float(self.spec.individual_gain)
            ).sum(axis=2)

            velocity = preferred.copy()
            if self.effect_mode in {"additive", "nonlinear"}:
                velocity += individual
            if self.effect_mode == "nonlinear":
                pair_synergy = np.zeros(
                    (self.interventions_per_scene, self.num_entities), dtype=np.float32
                )
                for first in range(self.num_robots):
                    for second in range(first + 1, self.num_robots):
                        alignment = np.sum(
                            away_unit[:, :, first] * away_unit[:, :, second], axis=-1
                        )
                        opposition = np.clip((-alignment - 0.15) / 0.85, 0.0, 1.0)
                        pair_synergy += strength[:, :, first] * strength[:, :, second] * opposition
                if self.response_family == "threshold":
                    pair_synergy = 1.0 / (
                        1.0 + np.exp(-12.0 * (pair_synergy - 0.12))
                    ) * pair_synergy
                pair_synergy = np.clip(pair_synergy * float(self.spec.joint_gain), 0.0, 0.95)
                preferred_unit = _unit(preferred)
                perpendicular = np.stack([-preferred_unit[:, :, 1], preferred_unit[:, :, 0]], axis=-1)
                team_midpoint = robot_pos.mean(axis=1, keepdims=True)
                side = np.sign(
                    np.sum(perpendicular * (nominal_entity_pos - team_midpoint), axis=-1)
                )
                side[side == 0.0] = 1.0
                slowdown = float(self.spec.joint_slowdown)
                lateral_gain = float(self.spec.joint_lateral_gain)
                if self.response_family == "lateral":
                    slowdown = 0.0
                elif self.response_family == "speed":
                    lateral_gain = 0.0
                velocity = velocity * (1.0 - slowdown * pair_synergy[:, :, None])
                velocity += perpendicular * side[:, :, None] * (
                    lateral_gain * pair_synergy[:, :, None]
                )

            velocity += response_noise[:, :, step]
            if self.effect_mode == "nonlinear":
                velocity = _clip_norm(velocity, float(self.spec.entity_max_speed))
            future[:, :, step] = velocity.astype(np.float32)
        return future

    def _make_sample(
        self,
        base_scene_id: int,
        intervention_id: int,
        base: dict[str, np.ndarray],
        robot_future: np.ndarray,
        entity_future: np.ndarray,
    ) -> dict[str, torch.Tensor | str | int | float]:
        history = np.concatenate([base["robot_history"], base["entity_history"]], axis=0)
        future = np.concatenate([robot_future, entity_future], axis=0)
        roles = np.concatenate(
            [
                np.full(self.num_robots, int(AgentRole.ROBOT), dtype=np.int64),
                np.full(self.num_entities, int(AgentRole.PHA), dtype=np.int64),
            ]
        )
        agent_mask = np.ones(self.num_agents, dtype=bool)
        robot_mask = np.zeros(self.num_agents, dtype=bool)
        robot_mask[: self.num_robots] = True
        entity_mask = agent_mask & ~robot_mask
        goals = np.zeros((self.num_agents, 2), dtype=np.float32)
        goals[: self.num_robots] = base["robot_current"] + robot_future.sum(axis=1) * self.dt
        goals[self.num_robots :] = base["entity_current"]
        geometry = np.asarray([ROLE_DEFAULT_GEOMETRY[int(role)] for role in roles], dtype=np.float32)
        return {
            "history": torch.tensor(history, dtype=torch.float32),
            "future": torch.tensor(future, dtype=torch.float32),
            "current_pos": torch.tensor(history[:, -1], dtype=torch.float32),
            "roles": torch.tensor(roles, dtype=torch.long),
            "agent_mask": torch.tensor(agent_mask, dtype=torch.bool),
            "robot_mask": torch.tensor(robot_mask, dtype=torch.bool),
            "entity_mask": torch.tensor(entity_mask, dtype=torch.bool),
            "loss_mask": torch.tensor(entity_mask, dtype=torch.bool),
            "geometry": torch.tensor(geometry, dtype=torch.float32),
            "goals": torch.tensor(goals, dtype=torch.float32),
            "goal_mask": torch.tensor(robot_mask, dtype=torch.bool),
            "dt": self.dt,
            "layout": f"paired_intervention:{self.scene_family}",
            "base_scene_id": self.split_offset + base_scene_id,
            "intervention_id": intervention_id,
            "effect_mode": self.effect_mode,
            "scene_family": self.scene_family,
            "response_family": self.response_family,
        }

    def _materialize_scene(self, base_scene_id: int) -> tuple[dict[str, torch.Tensor | str | int | float], ...]:
        base = self._base_scene(base_scene_id)
        robot_futures = self._candidate_actions(base_scene_id, base["robot_current"])
        entity_futures = self._entity_responses(base, robot_futures)
        samples = tuple(
            self._make_sample(
                base_scene_id,
                intervention_id,
                base,
                robot_futures[intervention_id],
                entity_futures[intervention_id],
            )
            for intervention_id in range(self.interventions_per_scene)
        )
        return samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int | float]:
        idx = int(idx)
        if self.cache_samples:
            base_scene_id, intervention_id = divmod(idx, self.interventions_per_scene)
            if base_scene_id not in self._scene_sample_cache:
                self._scene_sample_cache[base_scene_id] = self._materialize_scene(base_scene_id)
            return self._scene_sample_cache[base_scene_id][intervention_id]
        if self.cache_samples and idx in self._sample_cache:
            return self._sample_cache[idx]
        base_scene_id, intervention_id = divmod(idx, self.interventions_per_scene)
        base = self._base_scene(base_scene_id)
        robot_future = self._candidate_action(base_scene_id, intervention_id, base["robot_current"])
        entity_future = self._entity_response(base, robot_future, self.effect_mode)

        history = np.concatenate([base["robot_history"], base["entity_history"]], axis=0)
        future = np.concatenate([robot_future, entity_future], axis=0)
        roles = np.concatenate(
            [
                np.full(self.num_robots, int(AgentRole.ROBOT), dtype=np.int64),
                np.full(self.num_entities, int(AgentRole.PHA), dtype=np.int64),
            ]
        )
        agent_mask = np.ones(self.num_agents, dtype=bool)
        robot_mask = np.zeros(self.num_agents, dtype=bool)
        robot_mask[: self.num_robots] = True
        entity_mask = agent_mask & ~robot_mask
        goals = np.zeros((self.num_agents, 2), dtype=np.float32)
        goals[: self.num_robots] = base["robot_current"] + robot_future.sum(axis=1) * self.dt
        goals[self.num_robots :] = base["entity_current"]
        geometry = np.asarray([ROLE_DEFAULT_GEOMETRY[int(role)] for role in roles], dtype=np.float32)

        sample = {
            "history": torch.tensor(history, dtype=torch.float32),
            "future": torch.tensor(future, dtype=torch.float32),
            "current_pos": torch.tensor(history[:, -1], dtype=torch.float32),
            "roles": torch.tensor(roles, dtype=torch.long),
            "agent_mask": torch.tensor(agent_mask, dtype=torch.bool),
            "robot_mask": torch.tensor(robot_mask, dtype=torch.bool),
            "entity_mask": torch.tensor(entity_mask, dtype=torch.bool),
            "loss_mask": torch.tensor(entity_mask, dtype=torch.bool),
            "geometry": torch.tensor(geometry, dtype=torch.float32),
            "goals": torch.tensor(goals, dtype=torch.float32),
            "goal_mask": torch.tensor(robot_mask, dtype=torch.bool),
            "dt": self.dt,
            "layout": f"paired_intervention:{self.scene_family}",
            "base_scene_id": self.split_offset + base_scene_id,
            "intervention_id": intervention_id,
            "effect_mode": self.effect_mode,
            "scene_family": self.scene_family,
            "response_family": self.response_family,
        }
        if self.cache_samples:
            self._sample_cache[idx] = sample
        return sample


def make_causal_response_dataset_from_config(
    raw_cfg: Mapping,
    split: str = "train",
    effect_mode: str | None = None,
    seed: int | None = None,
    **overrides,
) -> CausalTeamResponseDataset:
    dataset_cfg = raw_cfg.get("dataset", {})
    causal_cfg = dataset_cfg.get("causal_response", {})
    split_key = f"{split}_base_scenes"
    spec = CausalResponseSpec(
        split=split,
        num_base_scenes=int(overrides.pop("num_base_scenes", causal_cfg.get(split_key, 256))),
        interventions_per_scene=int(overrides.pop("interventions_per_scene", causal_cfg.get("interventions_per_scene", 4))),
        history=int(overrides.pop("history", dataset_cfg.get("history", 8))),
        horizon=int(overrides.pop("horizon", dataset_cfg.get("horizon", 12))),
        num_robots=int(overrides.pop("num_robots", causal_cfg.get("num_robots", 2))),
        num_entities=int(overrides.pop("num_entities", causal_cfg.get("num_entities", 12))),
        dt=float(overrides.pop("dt", dataset_cfg.get("dt", 0.4))),
        seed=int(seed if seed is not None else overrides.pop("seed", causal_cfg.get("seed", 2027))),
        effect_mode=str(effect_mode or overrides.pop("effect_mode", causal_cfg.get("effect_mode", "nonlinear"))),
        area_size=float(overrides.pop("area_size", causal_cfg.get("area_size", 10.0))),
        influence_radius=float(overrides.pop("influence_radius", causal_cfg.get("influence_radius", 4.0))),
        individual_gain=float(overrides.pop("individual_gain", causal_cfg.get("individual_gain", 1.0))),
        joint_gain=float(overrides.pop("joint_gain", causal_cfg.get("joint_gain", 3.0))),
        joint_slowdown=float(overrides.pop("joint_slowdown", causal_cfg.get("joint_slowdown", 1.0))),
        joint_lateral_gain=float(
            overrides.pop("joint_lateral_gain", causal_cfg.get("joint_lateral_gain", 1.25))
        ),
        entity_noise=float(overrides.pop("entity_noise", causal_cfg.get("entity_noise", 0.015))),
        entity_max_speed=float(overrides.pop("entity_max_speed", causal_cfg.get("entity_max_speed", 3.0))),
        scene_family=str(overrides.pop("scene_family", causal_cfg.get("scene_family", "radial"))),
        response_family=str(
            overrides.pop("response_family", causal_cfg.get("response_family", "pincer"))
        ),
        cache_samples=bool(overrides.pop("cache_samples", causal_cfg.get("cache_samples", False))),
    )
    return CausalTeamResponseDataset(spec)
