from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


TRAJECTORY_MODE_NAMES = ("yielding", "neutral", "assertive")
MODE_RESPONSE_GAINS = np.asarray(
    [
        [1.15, 1.00, 1.00],
        [0.82, 0.58, 0.64],
        [0.52, 0.24, 0.36],
    ],
    dtype=np.float32,
)
VALID_SCENE_FAMILIES = {"radial", "corridor", "crossing", "bottleneck"}
VALID_RESPONSE_FAMILIES = {"pincer", "speed", "lateral", "threshold"}


def _unit(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), eps)


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector * np.minimum(1.0, float(max_norm) / np.maximum(norm, 1e-6))


@dataclass(frozen=True)
class PersistentTrajectorySpec:
    split: str = "train"
    num_episodes: int = 1024
    windows: int = 8
    history: int = 8
    horizon: int = 12
    num_robots: int = 2
    dt: float = 0.4
    seed: int = 2027
    action_policy: str = "iid"
    response_noise: float = 0.12
    influence_radius: float = 4.0
    execute_steps: int = 3
    entity_max_speed: float = 3.0
    scene_family: str = "radial"
    response_family: str = "pincer"
    cache_samples: bool = False
    include_interaction_residual: bool = False


class PersistentTrajectoryEpisodeDataset(Dataset):
    """Continuous trajectory episodes with stable entity identity and mode."""

    def __init__(self, spec: PersistentTrajectorySpec) -> None:
        if spec.num_robots < 2:
            raise ValueError("persistent-trajectory episodes require at least two robots")
        if spec.windows < 2 or spec.history < 2 or spec.horizon < 2:
            raise ValueError("windows, history, and horizon must be at least two")
        if not 1 <= spec.execute_steps <= spec.horizon:
            raise ValueError("execute_steps must lie in [1, horizon]")
        if spec.action_policy not in {"iid", "correlated"}:
            raise ValueError("action_policy must be 'iid' or 'correlated'")
        if str(spec.scene_family).lower() not in VALID_SCENE_FAMILIES:
            raise ValueError(
                f"scene_family must be one of {sorted(VALID_SCENE_FAMILIES)}"
            )
        if str(spec.response_family).lower() not in VALID_RESPONSE_FAMILIES:
            raise ValueError(
                f"response_family must be one of {sorted(VALID_RESPONSE_FAMILIES)}"
            )
        self.spec = spec
        self.num_episodes = int(spec.num_episodes)
        self.windows = int(spec.windows)
        self.history = int(spec.history)
        self.horizon = int(spec.horizon)
        self.num_robots = int(spec.num_robots)
        self.scene_family = str(spec.scene_family).lower()
        self.response_family = str(spec.response_family).lower()
        self.dt = float(spec.dt)
        self.cache_samples = bool(spec.cache_samples)
        self._sample_cache: dict[int, dict[str, torch.Tensor | int | str | float]] = {}
        split_offsets = {"train": 0, "val": 1_000_000, "test": 2_000_000, "ood": 3_000_000}
        self.split_offset = split_offsets.get(spec.split.lower(), 4_000_000)
        self.seed = int(spec.seed) + self.split_offset

    def __len__(self) -> int:
        return self.num_episodes

    def _robot_candidates(
        self,
        rng: np.random.Generator,
        entity_current: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_robots = self.num_robots
        if self.scene_family == "radial":
            base_angle = rng.uniform(-np.pi, np.pi)
            angular_layout = np.linspace(0.0, 2.0 * np.pi, num_robots, endpoint=False)
            angles = base_angle + angular_layout + rng.normal(0.0, 0.24, size=num_robots)
            radii = rng.uniform(2.0, 3.5, size=num_robots)
            relative_current = np.stack(
                [np.cos(angles), np.sin(angles)], axis=-1
            ) * radii[:, None]
        elif self.scene_family == "corridor":
            side = np.where(np.arange(num_robots) % 2 == 0, -1.0, 1.0)
            relative_current = np.stack(
                [rng.uniform(1.8, 3.6, size=num_robots) * side,
                 rng.normal(0.0, 0.45, size=num_robots)],
                axis=-1,
            )
        elif self.scene_family == "crossing":
            horizontal_count = max(1, num_robots // 2)
            horizontal = np.stack(
                [rng.uniform(2.0, 3.6, size=horizontal_count),
                 rng.normal(0.0, 0.35, size=horizontal_count)],
                axis=-1,
            )
            vertical_count = num_robots - horizontal_count
            vertical = (
                np.stack(
                    [rng.normal(0.0, 0.35, size=vertical_count),
                     rng.uniform(2.0, 3.6, size=vertical_count)],
                    axis=-1,
                )
                if vertical_count
                else np.empty((0, 2))
            )
            relative_current = np.concatenate([horizontal, vertical], axis=0)
            relative_current[:horizontal_count, 0] *= np.where(
                np.arange(horizontal_count) % 2 == 0, 1.0, -1.0
            )
            if vertical_count:
                relative_current[horizontal_count:, 1] *= np.where(
                    np.arange(vertical_count) % 2 == 0, 1.0, -1.0
                )
        else:  # bottleneck
            side = np.where(np.arange(num_robots) % 2 == 0, -1.0, 1.0)
            relative_current = np.stack(
                [side * rng.uniform(2.0, 3.6, size=num_robots),
                 rng.normal(0.0, 0.55, size=num_robots)],
                axis=-1,
            )

        if self.spec.action_policy == "iid":
            activity = rng.uniform(0.18, 1.0, size=num_robots)
        else:
            common = rng.uniform(0.18, 1.0)
            activity = np.clip(
                common + rng.normal(0.0, 0.035, size=num_robots), 0.18, 1.0
            )
        lateral_sign = rng.choice(np.asarray([-1.0, 1.0]), size=num_robots)
        toward_angle = np.arctan2(-relative_current[:, 1], -relative_current[:, 0])
        offsets = (1.0 - activity) * lateral_sign * rng.uniform(
            0.55, 0.95, size=num_robots
        )
        offsets += activity * rng.normal(0.0, 0.10, size=num_robots)
        speeds = 0.45 + 1.25 * activity
        if self.spec.action_policy == "correlated":
            common_turn = rng.uniform(-0.055, 0.055)
            turns = common_turn + rng.normal(0.0, 0.008, size=num_robots)
        else:
            turns = rng.uniform(-0.075, 0.075, size=num_robots)

        velocities = np.zeros((num_robots, self.horizon, 2), dtype=np.float32)
        for step in range(self.horizon):
            step_angles = toward_angle + offsets + turns * step
            velocities[:, step, 0] = np.cos(step_angles) * speeds
            velocities[:, step, 1] = np.sin(step_angles) * speeds
        future_relative = relative_current[:, None, :] + np.cumsum(velocities * self.dt, axis=1)
        candidates = np.concatenate([future_relative, velocities], axis=-1).astype(np.float32)
        current = np.concatenate([relative_current, velocities[:, 0]], axis=-1).astype(np.float32)
        return current, candidates, activity.astype(np.float32)

    def _response(
        self,
        rng: np.random.Generator,
        mode: int,
        entity_current: np.ndarray,
        preferred_velocity: np.ndarray,
        robot_current: np.ndarray,
        robot_candidates: np.ndarray,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        individual_gain, joint_gain, lateral_gain = MODE_RESPONSE_GAINS[mode]
        nominal_position = entity_current.copy()
        robot_position = entity_current[None, :] + robot_current[:, :2]
        response = np.zeros((self.horizon, 2), dtype=np.float32)
        radius = float(self.spec.influence_radius)
        if noise is None:
            noise = rng.normal(0.0, self.spec.response_noise, size=response.shape)
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape != response.shape or not np.isfinite(noise).all():
            raise ValueError("noise must have shape (horizon, 2) and contain finite values")

        for step in range(self.horizon):
            robot_velocity = robot_candidates[:, step, 2:]
            robot_position = robot_position + robot_velocity * self.dt
            nominal_position = nominal_position + preferred_velocity * self.dt
            away = nominal_position[None, :] - robot_position
            distance = np.linalg.norm(away, axis=-1)
            away_unit = _unit(away)
            approach_speed = np.sum(robot_velocity * away_unit, axis=-1)
            proximity = np.exp(-0.5 * np.square(distance / radius))
            approach_gate = 0.45 + 0.55 / (1.0 + np.exp(-2.0 * approach_speed))
            strength = proximity * approach_gate
            individual = (away_unit * strength[:, None]).sum(axis=0)

            preferred_unit = _unit(preferred_velocity[None, :])[0]
            perpendicular = np.asarray([-preferred_unit[1], preferred_unit[0]])
            pair_synergy = 0.0
            pair_lateral = np.zeros(2, dtype=np.float32)
            num_pairs = 0
            for first in range(self.num_robots):
                for second in range(first + 1, self.num_robots):
                    alignment = float(np.dot(away_unit[first], away_unit[second]))
                    opposition = np.clip((-alignment - 0.10) / 0.90, 0.0, 1.0)
                    raw_synergy = strength[first] * strength[second] * opposition
                    if self.response_family == "threshold":
                        raw_synergy *= 1.0 / (1.0 + np.exp(-12.0 * (raw_synergy - 0.12)))
                    synergy = np.clip(raw_synergy * joint_gain, 0.0, 0.88)
                    midpoint = 0.5 * (robot_position[first] + robot_position[second])
                    side = np.sign(
                        np.dot(perpendicular, nominal_position - midpoint)
                    ) or 1.0
                    pair_synergy += float(synergy)
                    if self.response_family != "speed":
                        pair_lateral += perpendicular * side * lateral_gain * synergy
                    num_pairs += 1
            pair_synergy = np.clip(pair_synergy / num_pairs, 0.0, 0.88)
            pair_lateral = pair_lateral / num_pairs

            velocity = preferred_velocity + individual_gain * individual
            slowdown = 0.0 if self.response_family == "lateral" else 0.70
            velocity = velocity * (1.0 - slowdown * pair_synergy)
            velocity += pair_lateral
            velocity += noise[step]
            response[step] = _clip_norm(velocity[None, :], self.spec.entity_max_speed)[0]
        return response

    def _interaction_residual(
        self,
        mode: int,
        entity_current: np.ndarray,
        preferred_velocity: np.ndarray,
        robot_current: np.ndarray,
        robot_candidates: np.ndarray,
    ) -> np.ndarray:
        """Return the clean team-response residual beyond additive effects."""

        reference = np.zeros_like(robot_candidates)
        reference[..., :2] = robot_current[:, None, :2]
        zero_noise = np.zeros((self.horizon, 2), dtype=np.float32)
        full = self._response(
            np.random.default_rng(0), mode, entity_current, preferred_velocity,
            robot_current, robot_candidates, noise=zero_noise,
        )
        reference_response = self._response(
            np.random.default_rng(0), mode, entity_current, preferred_velocity,
            robot_current, reference, noise=zero_noise,
        )
        additive = reference_response.copy()
        for robot_index in range(self.num_robots):
            single = reference.copy()
            single[robot_index] = robot_candidates[robot_index]
            single_response = self._response(
                np.random.default_rng(0), mode, entity_current, preferred_velocity,
                robot_current, single, noise=zero_noise,
            )
            additive += single_response - reference_response
        return full - additive

    def _generate_sample(self, index: int) -> dict[str, torch.Tensor | int | str]:
        rng = np.random.default_rng(self.seed + int(index) * 104729)
        mode = int(rng.integers(0, len(TRAJECTORY_MODE_NAMES)))
        entity_current = rng.uniform(-0.6, 0.6, size=2).astype(np.float32)
        preferred_angle = rng.uniform(-np.pi, np.pi)
        preferred_speed = rng.uniform(0.55, 1.05)
        preferred_velocity = (
            np.asarray([np.cos(preferred_angle), np.sin(preferred_angle)]) * preferred_speed
        ).astype(np.float32)
        history = np.stack(
            [
                entity_current + preferred_velocity * ((step - self.history + 1) * self.dt)
                for step in range(self.history)
            ],
            axis=0,
        ).astype(np.float32)

        histories = np.zeros((self.windows, self.history, 2), dtype=np.float32)
        robot_current_all = np.zeros((self.windows, self.num_robots, 4), dtype=np.float32)
        robot_candidates_all = np.zeros(
            (self.windows, self.num_robots, self.horizon, 4), dtype=np.float32
        )
        responses = np.zeros((self.windows, self.horizon, 2), dtype=np.float32)
        interaction_residuals = None
        if self.spec.include_interaction_residual:
            interaction_residuals = np.zeros(
                (self.windows, len(TRAJECTORY_MODE_NAMES), self.horizon, 2),
                dtype=np.float32,
            )
        activities = np.zeros((self.windows, self.num_robots), dtype=np.float32)
        high_interaction = np.zeros(self.windows, dtype=bool)

        for window in range(self.windows):
            histories[window] = history
            robot_current, candidates, activity = self._robot_candidates(rng, entity_current)
            response = self._response(
                rng,
                mode,
                entity_current,
                preferred_velocity,
                robot_current,
                candidates,
            )
            robot_current_all[window] = robot_current
            robot_candidates_all[window] = candidates
            responses[window] = response
            if interaction_residuals is not None:
                for response_mode in range(len(TRAJECTORY_MODE_NAMES)):
                    interaction_residuals[window, response_mode] = self._interaction_residual(
                        response_mode,
                        entity_current,
                        preferred_velocity,
                        robot_current,
                        candidates,
                    )
            activities[window] = activity
            high_interaction[window] = bool(
                (np.linalg.norm(robot_current[:, :2], axis=-1) < self.spec.influence_radius).all()
            )

            for step in range(self.spec.execute_steps):
                entity_current = entity_current + response[step] * self.dt
                history = np.concatenate([history[1:], entity_current[None, :]], axis=0)

        sample = {
            "entity_id": self.split_offset + int(index),
            "mode": torch.tensor(mode, dtype=torch.long),
            "entity_history": torch.from_numpy(histories),
            "robot_current": torch.from_numpy(robot_current_all),
            "robot_candidates": torch.from_numpy(robot_candidates_all),
            "target_future_vel": torch.from_numpy(responses),
            "activities": torch.from_numpy(activities),
            "high_interaction": torch.from_numpy(high_interaction),
            "action_policy": self.spec.action_policy,
            "scene_family": self.scene_family,
            "response_family": self.response_family,
            "dt": self.dt,
        }
        if interaction_residuals is not None:
            sample["interaction_residual"] = torch.from_numpy(interaction_residuals)
        return sample

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        if not 0 <= index < self.num_episodes:
            raise IndexError(index)
        if not self.cache_samples:
            return self._generate_sample(index)
        if index not in self._sample_cache:
            self._sample_cache[index] = self._generate_sample(index)
        return self._sample_cache[index]


@dataclass(frozen=True)
class MultiEntityPersistentTrajectorySpec(PersistentTrajectorySpec):
    num_entities: int = 10
    active_entities: int = 3
    shared_team_actions: bool = False


def candidate_independent_active_mask(
    robot_current: torch.Tensor,
    presence_mask: torch.Tensor,
    active_entities: int,
    priority_entities: tuple[int, ...] = (0,),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select active IDs from current-state risk without candidate futures."""

    if robot_current.ndim != 4 or robot_current.shape[-1] != 4:
        raise ValueError("robot_current must have shape (entities, windows, robots, 4)")
    entities, windows = robot_current.shape[:2]
    if tuple(presence_mask.shape) != (entities, windows):
        raise ValueError("presence_mask must align with entities and windows")
    if not 1 <= active_entities <= entities:
        raise ValueError("active_entities must lie in [1, num_entities]")
    score = torch.linalg.vector_norm(robot_current[..., :2], dim=-1).min(dim=-1).values
    score = score.masked_fill(~presence_mask.bool(), torch.inf)
    for entity in priority_entities:
        if 0 <= entity < entities:
            score[entity] = torch.where(
                presence_mask[entity].bool(), score[entity] - 1_000.0, score[entity]
            )
    if (presence_mask.sum(dim=0) < active_entities).any():
        raise ValueError("every window must contain enough present entities")
    selected = score.topk(active_entities, dim=0, largest=False).indices
    active = torch.zeros_like(presence_mask, dtype=torch.bool)
    active.scatter_(0, selected, True)
    return active, score


class MultiEntityPersistentTrajectoryEpisodeDataset(Dataset):
    """Group stable-ID trajectories into episodes with top-K exit and re-entry."""

    def __init__(self, spec: MultiEntityPersistentTrajectorySpec) -> None:
        if spec.num_entities < 4:
            raise ValueError("multi-entity episodes require at least four entity IDs")
        if not 1 <= spec.active_entities < spec.num_entities:
            raise ValueError("active_entities must be smaller than num_entities")
        if spec.windows < 5:
            raise ValueError("multi-entity re-entry requires at least five windows")
        self.spec = spec
        self.num_episodes = int(spec.num_episodes)
        self.num_entities = int(spec.num_entities)
        self.active_entities = int(spec.active_entities)
        self.windows = int(spec.windows)
        self.history = int(spec.history)
        self.horizon = int(spec.horizon)
        self.num_robots = int(spec.num_robots)
        self.scene_family = str(spec.scene_family).lower()
        self.response_family = str(spec.response_family).lower()
        self.dt = float(spec.dt)
        self.cache_samples = bool(spec.cache_samples)
        self._sample_cache: dict[int, dict[str, torch.Tensor | int | str | float]] = {}
        single_spec = PersistentTrajectorySpec(
            split=spec.split,
            num_episodes=spec.num_episodes * spec.num_entities,
            windows=spec.windows,
            history=spec.history,
            horizon=spec.horizon,
            num_robots=spec.num_robots,
            dt=spec.dt,
            seed=spec.seed,
            action_policy=spec.action_policy,
            response_noise=spec.response_noise,
            influence_radius=spec.influence_radius,
            execute_steps=spec.execute_steps,
            entity_max_speed=spec.entity_max_speed,
            scene_family=spec.scene_family,
            response_family=spec.response_family,
            cache_samples=spec.cache_samples,
            include_interaction_residual=False,
        )
        self._tracks = PersistentTrajectoryEpisodeDataset(single_spec)

    def __len__(self) -> int:
        return self.num_episodes

    def _responses(
        self,
        rng: np.random.Generator,
        modes: np.ndarray,
        entity_position: np.ndarray,
        preferred_velocity: np.ndarray,
        robot_position: np.ndarray,
        future_velocity: np.ndarray,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the shared-team response field for all entities at once."""

        entities = entity_position.shape[0]
        if noise is None:
            noise = np.stack(
                [
                    rng.normal(
                        0.0,
                        self.spec.response_noise,
                        size=(self.horizon, 2),
                    )
                    for _ in range(entities)
                ],
                axis=0,
            ).astype(np.float32)
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape != (entities, self.horizon, 2) or not np.isfinite(noise).all():
            raise ValueError("noise must have shape (entities, horizon, 2) and contain finite values")
        gains = MODE_RESPONSE_GAINS[modes]
        robot_path = robot_position[:, None, :] + np.cumsum(
            future_velocity * self.dt, axis=1
        )
        nominal = entity_position.copy()
        response = np.zeros((entities, self.horizon, 2), dtype=np.float32)
        preferred_unit = _unit(preferred_velocity)
        perpendicular = np.stack(
            [-preferred_unit[:, 1], preferred_unit[:, 0]], axis=-1
        )
        pair_indices = [
            (first, second)
            for first in range(self.num_robots)
            for second in range(first + 1, self.num_robots)
        ]
        for step in range(self.horizon):
            nominal = nominal + preferred_velocity * self.dt
            away = nominal[:, None, :] - robot_path[:, step, :][None, :, :]
            distance = np.linalg.norm(away, axis=-1)
            away_unit = _unit(away)
            robot_velocity = future_velocity[:, step, :]
            approach_speed = np.sum(
                robot_velocity[None, :, :] * away_unit, axis=-1
            )
            proximity = np.exp(
                -0.5 * np.square(distance / float(self.spec.influence_radius))
            )
            approach_gate = 0.45 + 0.55 / (1.0 + np.exp(-2.0 * approach_speed))
            strength = proximity * approach_gate
            individual = (away_unit * strength[..., None]).sum(axis=1)

            pair_synergy = np.zeros(entities, dtype=np.float32)
            pair_lateral = np.zeros((entities, 2), dtype=np.float32)
            for first, second in pair_indices:
                alignment = np.sum(
                    away_unit[:, first] * away_unit[:, second], axis=-1
                )
                opposition = np.clip((-alignment - 0.10) / 0.90, 0.0, 1.0)
                raw_synergy = strength[:, first] * strength[:, second] * opposition
                if self.spec.response_family == "threshold":
                    raw_synergy *= 1.0 / (1.0 + np.exp(-12.0 * (raw_synergy - 0.12)))
                synergy = np.clip(raw_synergy * gains[:, 1], 0.0, 0.88)
                midpoint = 0.5 * (
                    robot_path[:, step, :][first]
                    + robot_path[:, step, :][second]
                )
                side = np.sign(
                    np.sum(perpendicular * (nominal - midpoint), axis=-1)
                )
                side = np.where(side == 0.0, 1.0, side)
                pair_synergy += synergy
                if self.spec.response_family != "speed":
                    pair_lateral += (
                        perpendicular * side[:, None] * gains[:, 2, None] * synergy[:, None]
                    )
            pair_synergy = np.clip(
                pair_synergy / float(len(pair_indices)), 0.0, 0.88
            )
            pair_lateral /= float(len(pair_indices))

            velocity = preferred_velocity + gains[:, 0, None] * individual
            slowdown = 0.0 if self.spec.response_family == "lateral" else 0.70
            velocity *= (1.0 - slowdown * pair_synergy)[:, None]
            velocity += pair_lateral
            velocity += noise[:, step]
            response[:, step] = _clip_norm(
                velocity, self.spec.entity_max_speed
            )
        return response

    def _interaction_residuals(
        self,
        entity_position: np.ndarray,
        preferred_velocity: np.ndarray,
        robot_position: np.ndarray,
        future_velocity: np.ndarray,
    ) -> np.ndarray:
        """Return clean mode-conditioned team residuals for every entity."""

        entities = entity_position.shape[0]
        zero_noise = np.zeros((entities, self.horizon, 2), dtype=np.float32)
        reference_velocity = np.zeros_like(future_velocity)
        residuals = np.zeros(
            (entities, len(TRAJECTORY_MODE_NAMES), self.horizon, 2),
            dtype=np.float32,
        )
        for response_mode in range(len(TRAJECTORY_MODE_NAMES)):
            modes = np.full(entities, response_mode, dtype=np.int64)
            full = self._responses(
                np.random.default_rng(0), modes, entity_position, preferred_velocity,
                robot_position, future_velocity, noise=zero_noise,
            )
            reference = self._responses(
                np.random.default_rng(0), modes, entity_position, preferred_velocity,
                robot_position, reference_velocity, noise=zero_noise,
            )
            additive = reference.copy()
            for robot_index in range(self.num_robots):
                single_velocity = reference_velocity.copy()
                single_velocity[robot_index] = future_velocity[robot_index]
                single = self._responses(
                    np.random.default_rng(0), modes, entity_position, preferred_velocity,
                    robot_position, single_velocity, noise=zero_noise,
                )
                additive += single - reference
            residuals[:, response_mode] = full - additive
        return residuals

    def _shared_team_episode(
        self, index: int
    ) -> dict[str, torch.Tensor | int | str | float]:
        rng = np.random.default_rng(self._tracks.seed + int(index) * 104729)
        modes = rng.integers(
            0, len(TRAJECTORY_MODE_NAMES), size=self.num_entities
        )
        if self.scene_family == "radial":
            entity_position = rng.uniform(
                -1.4, 1.4, size=(self.num_entities, 2)
            )
        elif self.scene_family == "corridor":
            entity_position = np.stack(
                [rng.uniform(-1.4, 1.4, self.num_entities),
                 rng.normal(0.0, 0.45, self.num_entities)],
                axis=-1,
            )
        elif self.scene_family == "crossing":
            entity_position = rng.normal(0.0, 0.9, size=(self.num_entities, 2))
        else:  # bottleneck
            entity_position = rng.normal(0.0, 0.55, size=(self.num_entities, 2))
        entity_position = np.asarray(entity_position, dtype=np.float32)
        preferred_angle = rng.uniform(-np.pi, np.pi, size=self.num_entities)
        preferred_speed = rng.uniform(0.55, 1.05, size=self.num_entities)
        preferred_velocity = np.stack(
            [np.cos(preferred_angle), np.sin(preferred_angle)], axis=-1
        ) * preferred_speed[:, None]
        preferred_velocity = preferred_velocity.astype(np.float32)
        history = np.stack(
            [
                entity_position
                + preferred_velocity
                * ((step - self.history + 1) * self.dt)
                for step in range(self.history)
            ],
            axis=1,
        ).astype(np.float32)

        team_center = entity_position.mean(axis=0)
        if self.scene_family == "radial":
            base_angle = rng.uniform(-np.pi, np.pi)
            angular_layout = np.linspace(
                0.0, 2.0 * np.pi, self.num_robots, endpoint=False
            )
            angles = base_angle + angular_layout + rng.normal(
                0.0, 0.20, size=self.num_robots
            )
            radii = rng.uniform(2.4, 3.8, size=self.num_robots)
            relative_robot_position = np.stack(
                [np.cos(angles), np.sin(angles)], axis=-1
            ) * radii[:, None]
        elif self.scene_family == "corridor":
            side = np.where(np.arange(self.num_robots) % 2 == 0, -1.0, 1.0)
            relative_robot_position = np.stack(
                [side * rng.uniform(2.4, 3.8, self.num_robots),
                 rng.normal(0.0, 0.45, self.num_robots)],
                axis=-1,
            )
        elif self.scene_family == "crossing":
            horizontal_count = max(1, self.num_robots // 2)
            horizontal = np.stack(
                [rng.uniform(2.4, 3.8, horizontal_count),
                 rng.normal(0.0, 0.35, horizontal_count)], axis=-1
            )
            vertical_count = self.num_robots - horizontal_count
            vertical = (
                np.stack(
                    [rng.normal(0.0, 0.35, vertical_count),
                     rng.uniform(2.4, 3.8, vertical_count)], axis=-1
                )
                if vertical_count
                else np.empty((0, 2))
            )
            relative_robot_position = np.concatenate([horizontal, vertical], axis=0)
            relative_robot_position[:horizontal_count, 0] *= np.where(
                np.arange(horizontal_count) % 2 == 0, 1.0, -1.0
            )
            if vertical_count:
                relative_robot_position[horizontal_count:, 1] *= np.where(
                    np.arange(vertical_count) % 2 == 0, 1.0, -1.0
                )
        else:  # bottleneck
            side = np.where(np.arange(self.num_robots) % 2 == 0, -1.0, 1.0)
            relative_robot_position = np.stack(
                [side * rng.uniform(2.4, 3.8, self.num_robots),
                 rng.normal(0.0, 0.55, self.num_robots)], axis=-1
            )
        robot_position = team_center[None, :] + relative_robot_position
        robot_position = robot_position.astype(np.float32)
        robot_velocity = (
            _unit(team_center[None, :] - robot_position)
            * rng.uniform(0.35, 0.65, size=(self.num_robots, 1))
        ).astype(np.float32)

        presence = np.ones((self.num_entities, self.windows), dtype=bool)
        presence[0, 2:4] = False
        if self.num_entities > 1 and self.windows >= 7:
            presence[1, 4:6] = False
        histories = np.zeros(
            (self.num_entities, self.windows, self.history, 2), dtype=np.float32
        )
        robot_current_all = np.zeros(
            (self.num_entities, self.windows, self.num_robots, 4),
            dtype=np.float32,
        )
        robot_candidates_all = np.zeros(
            (
                self.num_entities,
                self.windows,
                self.num_robots,
                self.horizon,
                4,
            ),
            dtype=np.float32,
        )
        responses = np.zeros(
            (self.num_entities, self.windows, self.horizon, 2), dtype=np.float32
        )
        interaction_residuals = None
        if self.spec.include_interaction_residual:
            interaction_residuals = np.zeros(
                (
                    self.num_entities,
                    self.windows,
                    len(TRAJECTORY_MODE_NAMES),
                    self.horizon,
                    2,
                ),
                dtype=np.float32,
            )
        activities = np.zeros(
            (self.num_entities, self.windows, self.num_robots), dtype=np.float32
        )
        high_interaction = np.zeros(
            (self.num_entities, self.windows), dtype=bool
        )

        for window in range(self.windows):
            histories[:, window] = history
            present_center = entity_position[presence[:, window]].mean(axis=0)
            if self.spec.action_policy == "iid":
                activity = rng.uniform(0.18, 1.0, size=self.num_robots)
            else:
                common = rng.uniform(0.18, 1.0)
                activity = np.clip(
                    common + rng.normal(0.0, 0.035, size=self.num_robots),
                    0.18,
                    1.0,
                )
            relative_center = robot_position - present_center[None, :]
            toward_angle = np.arctan2(-relative_center[:, 1], -relative_center[:, 0])
            lateral_sign = rng.choice(
                np.asarray([-1.0, 1.0]), size=self.num_robots
            )
            offsets = (1.0 - activity) * lateral_sign * rng.uniform(
                0.45, 0.85, size=self.num_robots
            )
            offsets += activity * rng.normal(0.0, 0.10, size=self.num_robots)
            speeds = 0.45 + 1.25 * activity
            if self.spec.action_policy == "correlated":
                common_turn = rng.uniform(-0.055, 0.055)
                turns = common_turn + rng.normal(
                    0.0, 0.008, size=self.num_robots
                )
            else:
                turns = rng.uniform(-0.075, 0.075, size=self.num_robots)
            future_velocity = np.zeros(
                (self.num_robots, self.horizon, 2), dtype=np.float32
            )
            for step in range(self.horizon):
                step_angle = toward_angle + offsets + turns * step
                future_velocity[:, step, 0] = np.cos(step_angle) * speeds
                future_velocity[:, step, 1] = np.sin(step_angle) * speeds
            future_position = robot_position[:, None, :] + np.cumsum(
                future_velocity * self.dt, axis=1
            )

            current_all = np.concatenate(
                [
                    robot_position[None, :, :] - entity_position[:, None, :],
                    np.broadcast_to(robot_velocity, (self.num_entities, self.num_robots, 2)),
                ],
                axis=-1,
            ).astype(np.float32)
            candidate_all = np.concatenate(
                [
                    future_position[None, :, :, :]
                    - entity_position[:, None, None, :],
                    np.broadcast_to(
                        future_velocity,
                        (self.num_entities, self.num_robots, self.horizon, 2),
                    ),
                ],
                axis=-1,
            ).astype(np.float32)
            response_all = self._responses(
                rng,
                modes,
                entity_position,
                preferred_velocity,
                robot_position,
                future_velocity,
            )
            if interaction_residuals is not None:
                interaction_residuals[:, window] = self._interaction_residuals(
                    entity_position,
                    preferred_velocity,
                    robot_position,
                    future_velocity,
                )
            for entity in range(self.num_entities):
                current = current_all[entity]
                candidate = candidate_all[entity]
                response = response_all[entity]
                robot_current_all[entity, window] = current
                robot_candidates_all[entity, window] = candidate
                responses[entity, window] = response
                activities[entity, window] = activity
                high_interaction[entity, window] = bool(
                    (
                        np.linalg.norm(current[:, :2], axis=-1)
                        < self.spec.influence_radius
                    ).all()
                )

            for step in range(self.spec.execute_steps):
                robot_position = robot_position + future_velocity[:, step] * self.dt
                for entity in range(self.num_entities):
                    executed = (
                        responses[entity, window, step]
                        if presence[entity, window]
                        else preferred_velocity[entity]
                    )
                    entity_position[entity] = (
                        entity_position[entity] + executed * self.dt
                    )
                    history[entity] = np.concatenate(
                        [history[entity, 1:], entity_position[entity][None, :]],
                        axis=0,
                    )
            robot_velocity = future_velocity[:, self.spec.execute_steps - 1].copy()

        tensor_current = torch.from_numpy(robot_current_all)
        tensor_presence = torch.from_numpy(presence)
        active, selection_score = candidate_independent_active_mask(
            tensor_current,
            tensor_presence,
            self.active_entities,
        )
        entity_ids = torch.arange(
            self._tracks.split_offset + index * self.num_entities,
            self._tracks.split_offset + (index + 1) * self.num_entities,
            dtype=torch.long,
        )
        sample = {
            "episode_id": int(index),
            "entity_id": entity_ids,
            "mode": torch.from_numpy(modes.astype(np.int64)),
            "entity_history": torch.from_numpy(histories),
            "robot_current": tensor_current,
            "robot_candidates": torch.from_numpy(robot_candidates_all),
            "target_future_vel": torch.from_numpy(responses),
            "activities": torch.from_numpy(activities),
            "high_interaction": torch.from_numpy(high_interaction),
            "presence_mask": tensor_presence,
            "active_mask": active,
            "selection_score": selection_score,
            "action_policy": self.spec.action_policy,
            "scene_family": self.scene_family,
            "response_family": self.response_family,
            "dt": self.dt,
        }
        if interaction_residuals is not None:
            sample["interaction_residual"] = torch.from_numpy(interaction_residuals)
        return sample

    def _generate_sample(self, index: int) -> dict[str, torch.Tensor | int | str | float]:
        if not 0 <= index < self.num_episodes:
            raise IndexError(index)
        if self.spec.shared_team_actions:
            return self._shared_team_episode(index)
        first = int(index) * self.num_entities
        tracks = [self._tracks[first + entity] for entity in range(self.num_entities)]
        tensor_fields = (
            "mode",
            "entity_history",
            "robot_current",
            "robot_candidates",
            "target_future_vel",
            "activities",
            "high_interaction",
        )
        result: dict[str, torch.Tensor | int | str | float] = {
            field: torch.stack([track[field] for track in tracks])
            for field in tensor_fields
        }
        entity_ids = torch.tensor(
            [int(track["entity_id"]) for track in tracks], dtype=torch.long
        )
        presence = torch.ones(self.num_entities, self.windows, dtype=torch.bool)
        presence[0, 2:4] = False
        if self.num_entities > 1 and self.windows >= 7:
            presence[1, 4:6] = False
        active, selection_score = candidate_independent_active_mask(
            result["robot_current"],
            presence,
            self.active_entities,
        )
        result.update(
            {
                "episode_id": int(index),
                "entity_id": entity_ids,
                "presence_mask": presence,
                "active_mask": active,
                "selection_score": selection_score,
                "action_policy": self.spec.action_policy,
                "scene_family": self.scene_family,
                "response_family": self.response_family,
                "dt": self.dt,
            }
        )
        return result

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str | float]:
        if not 0 <= index < self.num_episodes:
            raise IndexError(index)
        if self.cache_samples:
            if index not in self._sample_cache:
                self._sample_cache[index] = self._generate_sample(index)
            return self._sample_cache[index]
        return self._generate_sample(index)
