"""Independent resettable Social-Force environment for TRACER evaluation.

The dynamics in this module do not call the response field used by
``SyntheticClosedLoopSimulator``.  A pedestrian is integrated with a standard
goal-relaxation force and exponential robot-repulsion forces.  The same saved
state and process-noise path can be replayed under every robot action branch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from sage.response.closed_loop import (
    CandidatePool,
    ClosedLoopSpec,
    PlanningCostConfig,
    RealizedWindow,
)


SOCIAL_FORCE_MODE_NAMES = ("yielding", "neutral", "assertive")


def _unit(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), eps)


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector * np.minimum(1.0, float(max_norm) / np.maximum(norm, 1e-6))


def _stationary_reference(robot_current: np.ndarray, horizon: int) -> np.ndarray:
    position = np.broadcast_to(
        robot_current[:, None, :2], (robot_current.shape[0], horizon, 2)
    ).copy()
    return np.concatenate([position, np.zeros_like(position)], axis=-1).astype(
        np.float32
    )


@dataclass(frozen=True)
class SocialForceParameters:
    relaxation_time: tuple[float, float, float] = (0.32, 0.48, 0.68)
    repulsion_strength: tuple[float, float, float] = (4.8, 3.1, 1.8)
    repulsion_decay: tuple[float, float, float] = (0.70, 0.58, 0.48)
    personal_radius: tuple[float, float, float] = (0.95, 0.75, 0.58)
    group_pressure: tuple[float, float, float] = (1.10, 0.72, 0.42)
    anisotropy: float = 0.35
    acceleration_limit: float = 4.0


class IndependentSocialForceSimulator:
    """Resettable interaction environment with independent force dynamics."""

    backend_name = "independent-social-force-v2-nonlinear"

    def __init__(
        self,
        spec: ClosedLoopSpec,
        episode_index: int,
        parameters: SocialForceParameters | None = None,
    ) -> None:
        if spec.num_robots < 2:
            raise ValueError("independent interaction requires at least two robots")
        if spec.num_candidates < spec.num_robots + 3:
            raise ValueError("num_candidates must be at least num_robots + 3")
        if not 1 <= spec.execute_steps <= spec.horizon:
            raise ValueError("execute_steps must lie in [1, horizon]")
        if spec.split not in {"id", "ood"}:
            raise ValueError("split must be 'id' or 'ood'")
        self.spec = spec
        self.parameters = parameters or SocialForceParameters()
        split_offset = 0 if spec.split == "id" else 7_000_003
        self.rng = np.random.default_rng(
            spec.seed + split_offset + int(episode_index) * 104729
        )
        self.mode = int(self.rng.integers(0, len(SOCIAL_FORCE_MODE_NAMES)))
        self.entity_position = self._initial_entity_position()
        preferred_angle = self.rng.uniform(-np.pi, np.pi)
        preferred_speed = self.rng.uniform(0.60, 1.10)
        self.preferred_velocity = (
            np.asarray([np.cos(preferred_angle), np.sin(preferred_angle)])
            * preferred_speed
        ).astype(np.float32)
        self.entity_velocity = self.preferred_velocity.copy()
        self.history = np.stack(
            [
                self.entity_position
                + self.preferred_velocity * ((step - spec.history + 1) * spec.dt)
                for step in range(spec.history)
            ]
        ).astype(np.float32)
        relative = self._initial_robot_layout()
        self.robot_positions = self.entity_position[None] + relative
        self.robot_velocities = (
            _unit(-relative)
            * self.rng.uniform(0.35, 0.65, size=(spec.num_robots, 1))
        ).astype(np.float32)
        self.window = 0

    def _initial_entity_position(self) -> np.ndarray:
        family = str(self.spec.scene_family).lower()
        if family == "corridor":
            value = [self.rng.uniform(-0.5, 0.5), self.rng.normal(0.0, 0.20)]
        elif family == "crossing":
            value = self.rng.normal(0.0, 0.40, size=2)
        elif family == "bottleneck":
            value = self.rng.normal(0.0, 0.28, size=2)
        else:
            value = self.rng.uniform(-0.55, 0.55, size=2)
        return np.asarray(value, dtype=np.float32)

    def _initial_robot_layout(self) -> np.ndarray:
        robots = self.spec.num_robots
        near = (2.0, 3.4) if self.spec.split == "id" else (1.7, 3.0)
        family = str(self.spec.scene_family).lower()
        if family == "corridor" or family == "bottleneck":
            side = np.where(np.arange(robots) % 2 == 0, -1.0, 1.0)
            lateral_std = 0.28 if family == "corridor" else 0.42
            relative = np.stack(
                [
                    side * self.rng.uniform(*near, size=robots),
                    self.rng.normal(0.0, lateral_std, size=robots),
                ],
                axis=-1,
            )
        elif family == "crossing":
            angles = np.resize(
                np.asarray([0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi]), robots
            ) + self.rng.normal(0.0, 0.13, size=robots)
            radii = self.rng.uniform(*near, size=robots)
            relative = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
            relative *= radii[:, None]
        else:
            base = self.rng.uniform(-np.pi, np.pi)
            angles = base + np.linspace(0.0, 2.0 * np.pi, robots, endpoint=False)
            angles += self.rng.normal(0.0, 0.14, size=robots)
            radii = self.rng.uniform(*near, size=robots)
            relative = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
            relative *= radii[:, None]
        return relative.astype(np.float32)

    def candidate_pool(self) -> CandidatePool:
        spec = self.spec
        relative = self.robot_positions - self.entity_position[None]
        activities = self.rng.uniform(
            0.15, 1.0, size=(spec.num_candidates, spec.num_robots)
        )
        anchors = [
            np.full(spec.num_robots, 0.15),
            np.full(spec.num_robots, 1.0),
            np.full(spec.num_robots, 0.55),
        ]
        for robot in range(spec.num_robots):
            isolated = np.full(spec.num_robots, 0.18)
            isolated[robot] = 1.0
            anchors.append(isolated)
        anchor_count = min(len(anchors), spec.num_candidates)
        activities[:anchor_count] = np.asarray(anchors[:anchor_count])

        candidates = np.zeros(
            (spec.num_candidates, spec.num_robots, spec.horizon, 4),
            dtype=np.float32,
        )
        toward = _unit(-relative)
        tangent = np.stack([-toward[:, 1], toward[:, 0]], axis=-1)
        for candidate_index, activity in enumerate(activities):
            lateral_sign = self.rng.choice([-1.0, 1.0], size=spec.num_robots)
            lateral = (1.0 - activity) * lateral_sign * self.rng.uniform(
                0.35, 0.82, size=spec.num_robots
            )
            direction = _unit(toward + tangent * lateral[:, None])
            speed = 0.35 + 1.25 * activity
            turn = self.rng.uniform(-0.055, 0.055, size=spec.num_robots)
            position = relative.copy()
            base_angle = np.arctan2(direction[:, 1], direction[:, 0])
            for step in range(spec.horizon):
                angle = base_angle + turn * step
                velocity = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
                velocity *= speed[:, None]
                position = position + velocity * spec.dt
                candidates[candidate_index, :, step, :2] = position
                candidates[candidate_index, :, step, 2:] = velocity
        current = np.concatenate([relative, self.robot_velocities], axis=-1)
        current = np.broadcast_to(
            current, (spec.num_candidates, *current.shape)
        ).copy()
        return CandidatePool(
            robot_current=torch.from_numpy(current.astype(np.float32)),
            robot_candidates=torch.from_numpy(candidates),
            activities=torch.from_numpy(activities.astype(np.float32)),
        )

    def _response_batch(
        self,
        robot_current: np.ndarray,
        robot_candidates: np.ndarray,
        noise: np.ndarray,
    ) -> np.ndarray:
        """Integrate Social-Force dynamics for every action branch."""

        current = np.asarray(robot_current, dtype=np.float32)
        candidates = np.asarray(robot_candidates, dtype=np.float32)
        count = candidates.shape[0]
        position = np.broadcast_to(self.entity_position, (count, 2)).copy()
        velocity = np.broadcast_to(self.entity_velocity, (count, 2)).copy()
        response = np.zeros((count, self.spec.horizon, 2), dtype=np.float32)
        process_noise = np.asarray(noise, dtype=np.float32)
        if process_noise.shape == (self.spec.horizon, 2):
            process_noise = process_noise[None]
        if process_noise.shape[0] == 1:
            process_noise = np.broadcast_to(process_noise, response.shape)
        if process_noise.shape != response.shape:
            raise ValueError("noise must be common or candidate-specific horizon noise")

        tau = self.parameters.relaxation_time[self.mode]
        strength = self.parameters.repulsion_strength[self.mode]
        decay = self.parameters.repulsion_decay[self.mode]
        radius = self.parameters.personal_radius[self.mode]
        group_pressure = self.parameters.group_pressure[self.mode]
        for step in range(self.spec.horizon):
            robot_position = self.entity_position[None, None] + candidates[:, :, step, :2]
            difference = position[:, None] - robot_position
            distance = np.linalg.norm(difference, axis=-1)
            direction = _unit(difference)
            robot_velocity = candidates[:, :, step, 2:]
            relative_velocity = velocity[:, None] - robot_velocity
            closing = np.maximum(-np.sum(relative_velocity * direction, axis=-1), 0.0)
            desired_direction = _unit(self.preferred_velocity[None])[0]
            facing = np.sum(-direction * desired_direction[None, None], axis=-1)
            anisotropy = self.parameters.anisotropy + (
                1.0 - self.parameters.anisotropy
            ) * 0.5 * (1.0 + facing)
            magnitude = strength * np.exp((radius - distance) / decay)
            magnitude *= anisotropy * (1.0 + 0.35 * closing)
            repulsion = (direction * magnitude[..., None]).sum(axis=1)
            # A fixed nonlinear crowd-pressure term models the empirically
            # common case where two salient robots make a person slow down
            # more than the sum of two isolated robot responses.  It is zero
            # for a single robot and depends on the joint action branch.
            group_load = np.maximum(
                np.sum(magnitude, axis=1) - np.max(magnitude, axis=1), 0.0
            )
            group_force = -desired_direction[None] * (
                group_pressure * group_load
            )[:, None]
            goal_force = (self.preferred_velocity[None] - velocity) / tau
            acceleration = goal_force + repulsion + group_force + process_noise[:, step]
            acceleration = _clip_norm(acceleration, self.parameters.acceleration_limit)
            velocity = _clip_norm(
                velocity + acceleration * self.spec.dt, self.spec.entity_max_speed
            )
            position = position + velocity * self.spec.dt
            response[:, step] = velocity
        return response

    def _realized_window(
        self,
        pool: CandidatePool,
        selected: int,
        response: np.ndarray,
        reference_response: np.ndarray,
        cost: PlanningCostConfig,
    ) -> RealizedWindow:
        current = pool.robot_current[selected].numpy()
        candidate = pool.robot_candidates[selected].numpy()
        steps = self.spec.execute_steps
        displacement = np.cumsum(response[:steps] * self.spec.dt, axis=0)
        distance = np.linalg.norm(
            candidate[:, :steps, :2] - displacement[None], axis=-1
        )
        min_distance = float(distance.min())
        toward = _unit(-current[:, :2])
        robot_displacement = candidate[:, steps - 1, :2] - current[:, :2]
        progress = float(np.sum(robot_displacement * toward, axis=-1).mean())
        human_cost = float(
            np.square(response[:steps] - reference_response[:steps])
            .sum(axis=-1)
            .mean()
        )
        return RealizedWindow(
            response=torch.from_numpy(response.copy()),
            task_progress=progress,
            safety_cost=float(np.exp(-min_distance / cost.comfort_scale)),
            human_cost=human_cost,
            min_distance=min_distance,
            collision=float(min_distance < cost.collision_margin),
            mean_activity=float(pool.activities[selected].mean()),
        )

    def counterfactuals(
        self, pool: CandidatePool, cost: PlanningCostConfig
    ) -> tuple[RealizedWindow, ...]:
        state = copy.deepcopy(self.rng.bit_generator.state)
        try:
            noise = self.rng.normal(
                0.0, self.spec.response_noise, size=(self.spec.horizon, 2)
            ).astype(np.float32)
            current = pool.robot_current.numpy()
            candidates = pool.robot_candidates.numpy()
            stationary = _stationary_reference(current[0], self.spec.horizon)
            stationary = np.broadcast_to(
                stationary, (len(candidates), *stationary.shape)
            ).copy()
            responses = self._response_batch(current, candidates, noise)
            references = self._response_batch(current, stationary, noise)
            return tuple(
                self._realized_window(pool, index, responses[index], references[index], cost)
                for index in range(len(candidates))
            )
        finally:
            self.rng.bit_generator.state = state

    def step(
        self, pool: CandidatePool, selected: int, cost: PlanningCostConfig
    ) -> RealizedWindow:
        if not 0 <= selected < len(pool.robot_candidates):
            raise IndexError("selected candidate is outside the pool")
        current = pool.robot_current[selected].numpy()
        candidate = pool.robot_candidates[selected].numpy()
        noise = self.rng.normal(
            0.0, self.spec.response_noise, size=(self.spec.horizon, 2)
        ).astype(np.float32)
        reference = _stationary_reference(current, self.spec.horizon)
        response = self._response_batch(current[None], candidate[None], noise)[0]
        reference_response = self._response_batch(
            current[None], reference[None], noise
        )[0]
        realized = self._realized_window(
            pool, selected, response, reference_response, cost
        )
        initial_entity = self.entity_position.copy()
        for step in range(self.spec.execute_steps):
            self.entity_position = self.entity_position + response[step] * self.spec.dt
            self.robot_positions = initial_entity[None] + candidate[:, step, :2]
            self.history = np.concatenate(
                [self.history[1:], self.entity_position[None]], axis=0
            )
        self.entity_velocity = response[self.spec.execute_steps - 1].copy()
        self.robot_velocities = candidate[:, self.spec.execute_steps - 1, 2:].copy()
        self.window += 1
        return realized


@dataclass(frozen=True)
class MultiEntitySocialForceRealizedWindow:
    """Per-entity outcomes from one shared team action branch."""

    response: torch.Tensor
    task_progress: torch.Tensor
    safety_cost: torch.Tensor
    human_cost: torch.Tensor
    min_distance: torch.Tensor
    collision: torch.Tensor
    mean_activity: torch.Tensor


class MultiEntityIndependentSocialForceSimulator:
    """Resettable multi-entity Social-Force world with shared robot actions."""

    backend_name = "independent-social-force-v2-multientity-nonlinear"

    def __init__(
        self,
        spec: ClosedLoopSpec,
        episode_index: int,
        num_entities: int,
        active_entities: int,
        continuous_parameters: bool = False,
        entity_interaction_strength: float = 0.0,
        entity_interaction_decay: float = 0.72,
        entity_interaction_radius: float = 0.85,
        parameters: SocialForceParameters | None = None,
    ) -> None:
        if spec.num_robots < 2:
            raise ValueError("independent interaction requires at least two robots")
        if spec.num_candidates < spec.num_robots + 3:
            raise ValueError("num_candidates must be at least num_robots + 3")
        if num_entities < 2:
            raise ValueError("multi-entity interaction requires at least two entities")
        if not 1 <= active_entities <= num_entities:
            raise ValueError("active_entities must lie in [1, num_entities]")
        if not 1 <= spec.execute_steps <= spec.horizon:
            raise ValueError("execute_steps must lie in [1, horizon]")
        if spec.split not in {"id", "ood"}:
            raise ValueError("split must be 'id' or 'ood'")
        self.spec = spec
        self.num_entities = int(num_entities)
        self.active_entities = int(active_entities)
        if entity_interaction_strength < 0.0:
            raise ValueError("entity_interaction_strength must be non-negative")
        if entity_interaction_decay <= 0.0 or entity_interaction_radius <= 0.0:
            raise ValueError("entity interaction scale parameters must be positive")
        self.entity_interaction_strength = float(entity_interaction_strength)
        self.entity_interaction_decay = float(entity_interaction_decay)
        self.entity_interaction_radius = float(entity_interaction_radius)
        self.parameters = parameters or SocialForceParameters()
        split_offset = 0 if spec.split == "id" else 7_000_003
        self.rng = np.random.default_rng(
            spec.seed + split_offset + int(episode_index) * 104729
        )
        self.modes = self.rng.integers(
            0, len(SOCIAL_FORCE_MODE_NAMES), size=self.num_entities
        ).astype(np.int64)
        self.parameter_coordinates = self.modes.astype(np.float32)
        if continuous_parameters:
            self.parameter_coordinates = self.rng.uniform(
                0.0, float(len(SOCIAL_FORCE_MODE_NAMES) - 1), size=self.num_entities
            ).astype(np.float32)
            self.modes = np.rint(self.parameter_coordinates).astype(np.int64)
        parameter_table = {
            "relaxation_time": self.parameters.relaxation_time,
            "repulsion_strength": self.parameters.repulsion_strength,
            "repulsion_decay": self.parameters.repulsion_decay,
            "personal_radius": self.parameters.personal_radius,
            "group_pressure": self.parameters.group_pressure,
        }
        self.parameter_values = {
            name: np.interp(self.parameter_coordinates, [0.0, 1.0, 2.0], values).astype(
                np.float32
            )
            for name, values in parameter_table.items()
        }
        self._parameter_table = parameter_table
        self.entity_positions = self.rng.normal(
            0.0, 0.45 if spec.split == "id" else 0.58, size=(self.num_entities, 2)
        ).astype(np.float32)
        angles = self.rng.uniform(-np.pi, np.pi, size=self.num_entities)
        speeds = self.rng.uniform(0.60, 1.10, size=self.num_entities)
        self.preferred_velocities = np.stack(
            [np.cos(angles) * speeds, np.sin(angles) * speeds], axis=-1
        ).astype(np.float32)
        self.entity_velocities = self.preferred_velocities.copy()
        self.histories = np.stack(
            [
                self.entity_positions
                + self.preferred_velocities * ((step - spec.history + 1) * spec.dt)
                for step in range(spec.history)
            ],
            axis=1,
        ).astype(np.float32)
        team_center = self.entity_positions.mean(axis=0)
        self.robot_positions = team_center[None] + self._initial_robot_layout()
        self.robot_velocities = (
            _unit(-self.robot_positions)
            * self.rng.uniform(0.35, 0.65, size=(spec.num_robots, 1))
        ).astype(np.float32)
        self.presence = np.ones(
            (self.num_entities, spec.windows), dtype=np.bool_
        )
        if self.num_entities >= 1 and spec.windows >= 7:
            self.presence[0, 2:4] = False
        if self.num_entities >= 2 and spec.windows >= 7:
            self.presence[1, 4:6] = False
        self.window = 0

    def set_parameter_coordinates(self, coordinates: np.ndarray) -> None:
        """Set continuous behavior coordinates for counterfactual supervision."""

        values = np.asarray(coordinates, dtype=np.float32)
        if values.shape != (self.num_entities,):
            raise ValueError("coordinates must have shape (num_entities,)")
        if np.any(values < 0.0) or np.any(values > float(len(SOCIAL_FORCE_MODE_NAMES) - 1)):
            raise ValueError("coordinates must lie within the parameter table range")
        self.parameter_coordinates = values.copy()
        self.modes = np.rint(values).astype(np.int64)
        self.parameter_values = {
            name: np.interp(values, [0.0, 1.0, 2.0], table).astype(np.float32)
            for name, table in self._parameter_table.items()
        }

    def _initial_robot_layout(self) -> np.ndarray:
        robots = self.spec.num_robots
        near = (2.0, 3.4) if self.spec.split == "id" else (1.7, 3.0)
        angles = np.resize(
            np.asarray([0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi]), robots
        ) + self.rng.normal(0.0, 0.13, size=robots)
        radii = self.rng.uniform(*near, size=robots)
        relative = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        relative *= radii[:, None]
        return relative.astype(np.float32)

    def active_mask(self) -> np.ndarray:
        """Select nearby present entities without consulting model predictions."""

        present = self.presence[:, self.window]
        distance = np.linalg.norm(
            self.robot_positions[None] - self.entity_positions[:, None], axis=-1
        ).mean(axis=-1)
        order = np.argsort(np.where(present, distance, np.inf), kind="stable")
        active = np.zeros(self.num_entities, dtype=np.bool_)
        active[order[: self.active_entities]] = present[order[: self.active_entities]]
        return active

    def candidate_pool(self) -> CandidatePool:
        spec = self.spec
        activities = self.rng.uniform(
            0.15, 1.0, size=(spec.num_candidates, spec.num_robots)
        )
        anchors = [
            np.full(spec.num_robots, 0.15),
            np.full(spec.num_robots, 1.0),
            np.full(spec.num_robots, 0.55),
        ]
        for robot in range(spec.num_robots):
            isolated = np.full(spec.num_robots, 0.18)
            isolated[robot] = 1.0
            anchors.append(isolated)
        anchor_count = min(len(anchors), spec.num_candidates)
        activities[:anchor_count] = np.asarray(anchors[:anchor_count])
        world_candidates = np.zeros(
            (spec.num_candidates, spec.num_robots, spec.horizon, 4),
            dtype=np.float32,
        )
        relative = self.robot_positions.copy()
        toward = _unit(-relative)
        tangent = np.stack([-toward[:, 1], toward[:, 0]], axis=-1)
        for candidate_index, activity in enumerate(activities):
            lateral_sign = self.rng.choice([-1.0, 1.0], size=spec.num_robots)
            lateral = (1.0 - activity) * lateral_sign * self.rng.uniform(
                0.35, 0.82, size=spec.num_robots
            )
            direction = _unit(toward + tangent * lateral[:, None])
            speed = 0.35 + 1.25 * activity
            turn = self.rng.uniform(-0.055, 0.055, size=spec.num_robots)
            position = relative.copy()
            base_angle = np.arctan2(direction[:, 1], direction[:, 0])
            for step in range(spec.horizon):
                angle = base_angle + turn * step
                velocity = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
                velocity *= speed[:, None]
                position = position + velocity * spec.dt
                world_candidates[candidate_index, :, step, :2] = position
                world_candidates[candidate_index, :, step, 2:] = velocity
        current_world = np.concatenate([self.robot_positions, self.robot_velocities], axis=-1)
        current = np.broadcast_to(
            current_world[None, None],
            (spec.num_candidates, self.num_entities, spec.num_robots, 4),
        ).copy()
        current[..., :2] -= self.entity_positions[None, :, None, :]
        candidates = np.broadcast_to(
            world_candidates[:, None],
            (spec.num_candidates, self.num_entities, spec.num_robots, spec.horizon, 4),
        ).copy()
        candidates[..., :2] -= self.entity_positions[None, :, None, None, :]
        return CandidatePool(
            robot_current=torch.from_numpy(current.astype(np.float32)),
            robot_candidates=torch.from_numpy(candidates.astype(np.float32)),
            activities=torch.from_numpy(activities.astype(np.float32)),
        )

    def _response_batch(
        self,
        robot_candidates: np.ndarray,
        noise: np.ndarray,
    ) -> np.ndarray:
        candidates = np.asarray(robot_candidates, dtype=np.float32)
        count = candidates.shape[0]
        position = np.broadcast_to(
            self.entity_positions[None], (count, self.num_entities, 2)
        ).copy()
        velocity = np.broadcast_to(
            self.entity_velocities[None], (count, self.num_entities, 2)
        ).copy()
        response = np.zeros(
            (count, self.num_entities, self.spec.horizon, 2), dtype=np.float32
        )
        process_noise = np.asarray(noise, dtype=np.float32)
        if process_noise.shape == (self.num_entities, self.spec.horizon, 2):
            process_noise = process_noise[None]
        if process_noise.shape[0] == 1:
            process_noise = np.broadcast_to(process_noise, response.shape)
        if process_noise.shape != response.shape:
            raise ValueError("noise must be common or candidate-specific entity noise")

        tau = self.parameter_values["relaxation_time"]
        strength = self.parameter_values["repulsion_strength"]
        decay = self.parameter_values["repulsion_decay"]
        radius = self.parameter_values["personal_radius"]
        group_pressure = self.parameter_values["group_pressure"]
        desired_direction = _unit(self.preferred_velocities)
        for step in range(self.spec.horizon):
            robot_position = self.entity_positions[None, :, None, :] + candidates[
                ..., step, :2
            ]
            difference = position[:, :, None, :] - robot_position
            distance = np.linalg.norm(difference, axis=-1)
            direction = _unit(difference)
            robot_velocity = candidates[..., step, 2:]
            relative_velocity = velocity[:, :, None, :] - robot_velocity
            closing = np.maximum(
                -np.sum(relative_velocity * direction, axis=-1), 0.0
            )
            facing = np.sum(
                -direction * desired_direction[None, :, None, :], axis=-1
            )
            anisotropy = self.parameters.anisotropy + (
                1.0 - self.parameters.anisotropy
            ) * 0.5 * (1.0 + facing)
            magnitude = strength[None, :, None] * np.exp(
                (radius[None, :, None] - distance) / decay[None, :, None]
            )
            magnitude *= anisotropy * (1.0 + 0.35 * closing)
            repulsion = (direction * magnitude[..., None]).sum(axis=2)
            if self.entity_interaction_strength > 0.0 and self.num_entities > 1:
                entity_difference = position[:, :, None, :] - position[:, None, :, :]
                entity_distance = np.linalg.norm(entity_difference, axis=-1)
                entity_direction = _unit(entity_difference)
                entity_mask = ~np.eye(self.num_entities, dtype=bool)[None]
                entity_magnitude = self.entity_interaction_strength * np.exp(
                    (self.entity_interaction_radius - entity_distance)
                    / self.entity_interaction_decay
                )
                entity_magnitude = np.where(entity_mask, entity_magnitude, 0.0)
                entity_repulsion = (
                    entity_direction * entity_magnitude[..., None]
                ).sum(axis=2)
                repulsion = repulsion + entity_repulsion
            group_load = np.maximum(
                np.sum(magnitude, axis=2) - np.max(magnitude, axis=2), 0.0
            )
            group_force = -desired_direction[None] * (
                group_pressure[None, :] * group_load
            )[..., None]
            goal_force = (
                self.preferred_velocities[None] - velocity
            ) / tau[None, :, None]
            acceleration = goal_force + repulsion + group_force + process_noise[:, :, step]
            acceleration = _clip_norm(acceleration, self.parameters.acceleration_limit)
            velocity = _clip_norm(
                velocity + acceleration * self.spec.dt, self.spec.entity_max_speed
            )
            position = position + velocity * self.spec.dt
            response[:, :, step] = velocity
        return response

    def _realized_entity(
        self,
        pool: CandidatePool,
        selected: int,
        entity: int,
        response: np.ndarray,
        reference_response: np.ndarray,
        cost: PlanningCostConfig,
    ) -> tuple[float, float, float, float, float, float]:
        current = pool.robot_current[selected, entity].numpy()
        candidate = pool.robot_candidates[selected, entity].numpy()
        steps = self.spec.execute_steps
        displacement = np.cumsum(response[:steps] * self.spec.dt, axis=0)
        distance = np.linalg.norm(
            candidate[:, :steps, :2] - displacement[None], axis=-1
        )
        min_distance = float(distance.min())
        toward = _unit(-current[:, :2])
        robot_displacement = candidate[:, steps - 1, :2] - current[:, :2]
        progress = float(np.sum(robot_displacement * toward, axis=-1).mean())
        human_cost = float(
            np.square(response[:steps] - reference_response[:steps])
            .sum(axis=-1)
            .mean()
        )
        return (
            progress,
            float(np.exp(-min_distance / cost.comfort_scale)),
            human_cost,
            min_distance,
            float(min_distance < cost.collision_margin),
            float(pool.activities[selected].mean()),
        )

    def counterfactuals(
        self, pool: CandidatePool, cost: PlanningCostConfig
    ) -> tuple[MultiEntitySocialForceRealizedWindow, ...]:
        state = copy.deepcopy(self.rng.bit_generator.state)
        try:
            noise = self.rng.normal(
                0.0,
                self.spec.response_noise,
                size=(self.num_entities, self.spec.horizon, 2),
            ).astype(np.float32)
            candidates = pool.robot_candidates.numpy()
            stationary = np.zeros_like(candidates)
            stationary[..., :2] = pool.robot_current.numpy()[..., None, :2]
            responses = self._response_batch(candidates, noise)
            references = self._response_batch(stationary, noise)
            outcomes = []
            for selected in range(len(candidates)):
                values = [
                    self._realized_entity(
                        pool,
                        selected,
                        entity,
                        responses[selected, entity],
                        references[selected, entity],
                        cost,
                    )
                    for entity in range(self.num_entities)
                ]
                outcomes.append(
                    MultiEntitySocialForceRealizedWindow(
                        response=torch.from_numpy(responses[selected].copy()),
                        task_progress=torch.tensor([value[0] for value in values]),
                        safety_cost=torch.tensor([value[1] for value in values]),
                        human_cost=torch.tensor([value[2] for value in values]),
                        min_distance=torch.tensor([value[3] for value in values]),
                        collision=torch.tensor([value[4] for value in values]),
                        mean_activity=torch.tensor([value[5] for value in values]),
                    )
                )
            return tuple(outcomes)
        finally:
            self.rng.bit_generator.state = state

    def step(
        self, pool: CandidatePool, selected: int, cost: PlanningCostConfig
    ) -> MultiEntitySocialForceRealizedWindow:
        if not 0 <= selected < len(pool.robot_candidates):
            raise IndexError("selected candidate is outside the pool")
        current = pool.robot_current[selected].numpy()
        candidate = pool.robot_candidates[selected].numpy()
        noise = self.rng.normal(
            0.0,
            self.spec.response_noise,
            size=(self.num_entities, self.spec.horizon, 2),
        ).astype(np.float32)
        stationary = np.zeros_like(candidate)
        stationary[..., :2] = current[..., None, :2]
        response = self._response_batch(candidate[None], noise)[0]
        reference = self._response_batch(stationary[None], noise)[0]
        values = [
            self._realized_entity(
                pool, selected, entity, response[entity], reference[entity], cost
            )
            for entity in range(self.num_entities)
        ]
        realized = MultiEntitySocialForceRealizedWindow(
            response=torch.from_numpy(response.copy()),
            task_progress=torch.tensor([value[0] for value in values]),
            safety_cost=torch.tensor([value[1] for value in values]),
            human_cost=torch.tensor([value[2] for value in values]),
            min_distance=torch.tensor([value[3] for value in values]),
            collision=torch.tensor([value[4] for value in values]),
            mean_activity=torch.tensor([value[5] for value in values]),
        )
        initial_positions = self.robot_positions.copy()
        robot_displacement = candidate[0, :, : self.spec.execute_steps, :2] - current[0, :, None, :2]
        for step in range(self.spec.execute_steps):
            self.robot_positions = initial_positions + robot_displacement[:, step]
            self.entity_positions += response[:, step] * self.spec.dt
            self.histories = np.concatenate(
                [self.histories[:, 1:], self.entity_positions[:, None]], axis=1
            )
        self.entity_velocities = response[:, self.spec.execute_steps - 1].copy()
        self.robot_velocities = candidate[0, :, self.spec.execute_steps - 1, 2:].copy()
        self.window += 1
        return realized


@dataclass(frozen=True)
class MultiEntitySocialForceDatasetSpec:
    split: str = "train"
    num_episodes: int = 1024
    windows: int = 8
    history: int = 8
    horizon: int = 12
    num_robots: int = 3
    num_entities: int = 6
    active_entities: int = 3
    num_candidates: int = 16
    execute_steps: int = 3
    dt: float = 0.3
    response_noise: float = 0.08
    entity_max_speed: float = 2.2
    entity_interaction_strength: float = 0.0
    entity_interaction_decay: float = 0.72
    entity_interaction_radius: float = 0.85
    scene_family: str = "crossing"
    parameter_mode: str = "discrete"
    seed: int = 2027
    cache_samples: bool = False


class MultiEntityIndependentSocialForceDataset(Dataset):
    """Multi-entity training corpus generated by the independent backend."""

    def __init__(self, spec: MultiEntitySocialForceDatasetSpec) -> None:
        self.spec = spec
        self._cache: dict[int, dict[str, object]] = {}
        self._split_offset = {
            "train": 1_000_000,
            "val": 2_000_000,
            "test": 3_000_000,
            "ood": 7_000_003,
        }.get(spec.split, 4_000_000)

    def __len__(self) -> int:
        return int(self.spec.num_episodes)

    def _generate(self, index: int) -> dict[str, object]:
        simulation_spec = ClosedLoopSpec(
            windows=self.spec.windows,
            history=self.spec.history,
            horizon=self.spec.horizon,
            num_robots=self.spec.num_robots,
            num_candidates=self.spec.num_candidates,
            execute_steps=self.spec.execute_steps,
            dt=self.spec.dt,
            response_noise=self.spec.response_noise,
            influence_radius=4.0,
            entity_max_speed=self.spec.entity_max_speed,
            scene_family=self.spec.scene_family,
            response_family="pincer",
            seed=self.spec.seed + self._split_offset,
            split="ood" if self.spec.split == "ood" else "id",
        )
        simulator = MultiEntityIndependentSocialForceSimulator(
            simulation_spec,
            index,
            self.spec.num_entities,
            self.spec.active_entities,
            continuous_parameters=self.spec.parameter_mode == "continuous",
            entity_interaction_strength=self.spec.entity_interaction_strength,
            entity_interaction_decay=self.spec.entity_interaction_decay,
            entity_interaction_radius=self.spec.entity_interaction_radius,
        )
        rng = np.random.default_rng(
            self.spec.seed + self._split_offset + index * 65537
        )
        histories = []
        currents = []
        candidates = []
        responses = []
        activities = []
        high_interaction = []
        active = []
        cost = PlanningCostConfig()
        for _ in range(self.spec.windows):
            pool = simulator.candidate_pool()
            selected = int(rng.integers(0, self.spec.num_candidates))
            histories.append(simulator.histories.copy())
            currents.append(pool.robot_current[selected].numpy().copy())
            candidates.append(pool.robot_candidates[selected].numpy().copy())
            activities.append(pool.activities[selected].numpy().copy())
            active.append(simulator.active_mask())
            high_interaction.append(
                np.linalg.norm(
                    pool.robot_current[selected].numpy()[..., :2], axis=-1
                )
                .mean(axis=-1)
                < 4.0
            )
            responses.append(simulator.step(pool, selected, cost).response.numpy())
        entity_ids = np.arange(
            self._split_offset + index * self.spec.num_entities,
            self._split_offset + (index + 1) * self.spec.num_entities,
            dtype=np.int64,
        )
        return {
            "entity_id": torch.from_numpy(entity_ids),
            "episode_id": int(index),
            "mode": torch.from_numpy(simulator.modes.copy()),
            "parameter_coordinate": torch.from_numpy(
                simulator.parameter_coordinates.copy()
            ),
            "entity_history": torch.from_numpy(np.stack(histories).transpose(1, 0, 2, 3).astype(np.float32)),
            "robot_current": torch.from_numpy(np.stack(currents).transpose(1, 0, 2, 3).astype(np.float32)),
            "robot_candidates": torch.from_numpy(np.stack(candidates).transpose(1, 0, 2, 3, 4).astype(np.float32)),
            "target_future_vel": torch.from_numpy(np.stack(responses).transpose(1, 0, 2, 3).astype(np.float32)),
            "activities": torch.from_numpy(np.stack(activities).astype(np.float32)),
            "high_interaction": torch.from_numpy(np.stack(high_interaction).transpose(1, 0)),
            "active_mask": torch.from_numpy(np.stack(active).transpose(1, 0)),
            "presence_mask": torch.from_numpy(simulator.presence.copy()),
            "selection_score": torch.from_numpy(
                np.linalg.norm(
                    np.stack(currents)[..., :2], axis=-1
                ).mean(axis=-1).transpose(1, 0).astype(np.float32)
            ),
            "backend": MultiEntityIndependentSocialForceSimulator.backend_name,
            "parameter_mode": self.spec.parameter_mode,
            "dt": self.spec.dt,
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if not self.spec.cache_samples:
            return self._generate(index)
        if index not in self._cache:
            self._cache[index] = self._generate(index)
        return self._cache[index]


@dataclass(frozen=True)
class IndependentSocialForceDatasetSpec:
    split: str = "train"
    num_episodes: int = 1024
    windows: int = 8
    history: int = 8
    horizon: int = 12
    num_robots: int = 3
    num_candidates: int = 16
    execute_steps: int = 3
    dt: float = 0.3
    response_noise: float = 0.08
    entity_max_speed: float = 2.2
    scene_family: str = "crossing"
    seed: int = 2027
    cache_samples: bool = False


class IndependentSocialForceDataset(Dataset):
    """On-policy episodes generated only by the independent force backend."""

    def __init__(self, spec: IndependentSocialForceDatasetSpec) -> None:
        self.spec = spec
        self._cache: dict[int, dict[str, object]] = {}
        self._split_offset = {
            "train": 1_000_000,
            "val": 2_000_000,
            "test": 3_000_000,
            "ood": 7_000_003,
        }.get(spec.split, 4_000_000)

    def __len__(self) -> int:
        return int(self.spec.num_episodes)

    def _generate(self, index: int) -> dict[str, object]:
        simulation_spec = ClosedLoopSpec(
            windows=self.spec.windows,
            history=self.spec.history,
            horizon=self.spec.horizon,
            num_robots=self.spec.num_robots,
            num_candidates=self.spec.num_candidates,
            execute_steps=self.spec.execute_steps,
            dt=self.spec.dt,
            response_noise=self.spec.response_noise,
            influence_radius=4.0,
            entity_max_speed=self.spec.entity_max_speed,
            scene_family=self.spec.scene_family,
            response_family="pincer",
            seed=self.spec.seed + self._split_offset,
            split="ood" if self.spec.split == "ood" else "id",
        )
        simulator = IndependentSocialForceSimulator(simulation_spec, index)
        rng = np.random.default_rng(
            self.spec.seed + self._split_offset + index * 65537
        )
        histories = []
        currents = []
        candidates = []
        responses = []
        activities = []
        interactions = []
        cost = PlanningCostConfig()
        for _ in range(self.spec.windows):
            pool = simulator.candidate_pool()
            selected = int(rng.integers(0, self.spec.num_candidates))
            histories.append(simulator.history.copy())
            currents.append(pool.robot_current[selected].numpy().copy())
            candidates.append(pool.robot_candidates[selected].numpy().copy())
            activities.append(pool.activities[selected].numpy().copy())
            interactions.append(
                bool(
                    (
                        np.linalg.norm(
                            pool.robot_current[selected, :, :2].numpy(), axis=-1
                        )
                        < 4.0
                    ).all()
                )
            )
            responses.append(simulator.step(pool, selected, cost).response.numpy())
        return {
            "entity_id": self._split_offset + index,
            "episode_id": self._split_offset + index,
            "mode": torch.tensor(simulator.mode, dtype=torch.long),
            "entity_history": torch.from_numpy(np.stack(histories).astype(np.float32)),
            "robot_current": torch.from_numpy(np.stack(currents).astype(np.float32)),
            "robot_candidates": torch.from_numpy(np.stack(candidates).astype(np.float32)),
            "target_future_vel": torch.from_numpy(np.stack(responses).astype(np.float32)),
            "activities": torch.from_numpy(np.stack(activities).astype(np.float32)),
            "high_interaction": torch.from_numpy(np.asarray(interactions)),
            "active_mask": torch.ones(self.spec.windows, dtype=torch.bool),
            "presence_mask": torch.ones(self.spec.windows, dtype=torch.bool),
            "backend": IndependentSocialForceSimulator.backend_name,
            "dt": self.spec.dt,
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if not self.spec.cache_samples:
            return self._generate(index)
        if index not in self._cache:
            self._cache[index] = self._generate(index)
        return self._cache[index]
