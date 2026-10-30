from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from tracer.response.closed_loop_contract import task_goals, validate_candidate_state
from tracer.response.persistent_trajectory import (
    MODE_RESPONSE_GAINS,
    VALID_RESPONSE_FAMILIES,
    VALID_SCENE_FAMILIES,
)


@lru_cache(maxsize=32)
def _independent_trajectory_noise(samples: int, horizon: int) -> torch.Tensor:
    uniforms = torch.quasirandom.SobolEngine(
        2 * horizon, scramble=True, seed=1729
    ).draw(samples)
    normal = 2.0 ** 0.5 * torch.erfinv(2.0 * uniforms.clamp(1e-5, 1.0 - 1e-5) - 1.0)
    return normal.reshape(samples, horizon, 2)


def _unit_np(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), eps)


def _clip_norm_np(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector * np.minimum(1.0, float(max_norm) / np.maximum(norm, 1e-6))


def _stationary_reference_np(robot_current: np.ndarray, horizon: int) -> np.ndarray:
    if robot_current.ndim != 2 or robot_current.shape[-1] != 4:
        raise ValueError("robot_current must have shape (robots, 4)")
    position = np.broadcast_to(
        robot_current[:, None, :2],
        (robot_current.shape[0], int(horizon), 2),
    ).copy()
    velocity = np.zeros_like(position)
    return np.concatenate([position, velocity], axis=-1).astype(np.float32)


@dataclass(frozen=True)
class ClosedLoopSpec:
    windows: int = 8
    history: int = 8
    horizon: int = 12
    num_robots: int = 2
    num_candidates: int = 16
    execute_steps: int = 3
    dt: float = 0.4
    response_noise: float = 0.18
    influence_radius: float = 4.0
    entity_max_speed: float = 3.0
    scene_family: str = "radial"
    response_family: str = "pincer"
    seed: int = 2027
    split: str = "id"
    task_type: str = "entity_approach"
    goal_tolerance: float = 0.3


@dataclass(frozen=True)
class CandidatePool:
    robot_current: torch.Tensor
    robot_candidates: torch.Tensor
    activities: torch.Tensor
    # Optional world-frame entity futures emitted by a joint candidate
    # generator. Legacy pools do not provide this field.
    entity_candidates: torch.Tensor | None = None


@dataclass(frozen=True)
class RealizedWindow:
    response: torch.Tensor
    task_progress: float
    safety_cost: float
    human_cost: float
    min_distance: float
    collision: float
    mean_activity: float
    smoothness_cost: float = 0.0


@dataclass(frozen=True)
class PlanningCostConfig:
    task_weight: float = 1.0
    task_speed_scale: float = 1.7
    safety_weight: float = 2.0
    human_weight: float = 0.6
    comfort_scale: float = 1.0
    collision_margin: float = 0.65
    task_steps: int | None = None
    response_steps: int | None = None
    violation_probability_alpha: float | None = None
    risk_samples: int = 32
    crowd_aggregation: str = "mean"
    temporal_risk_sampling: str = "independent"
    # Once probabilistic safety is enabled, keep lower-risk candidates ahead
    # of other candidates that merely pass the hard threshold.
    risk_weight: float = 0.0
    # Penalize abrupt changes between the current velocity and the candidate's
    # first action. This is a closed-loop execution term, not a SAGE loss.
    smoothness_weight: float = 0.0
    task_aggregation: str = "mean"
    task_min_blend: float = 0.5


def aggregate_robot_progress(
    progress_by_robot: torch.Tensor,
    aggregation: str = "mean",
    active_mask: torch.Tensor | None = None,
    min_blend: float = 0.5,
) -> torch.Tensor:
    """Aggregate per-robot progress while optionally emphasizing the slowest robot."""

    if progress_by_robot.ndim < 1 or progress_by_robot.shape[-1] < 1:
        raise ValueError("progress_by_robot must have a nonempty robot dimension")
    aggregation = str(aggregation).lower()
    if aggregation not in {"mean", "lower-tail"}:
        raise ValueError("task aggregation must be 'mean' or 'lower-tail'")
    if not np.isfinite(min_blend) or not 0.0 <= float(min_blend) <= 1.0:
        raise ValueError("task minimum blend must lie in [0, 1]")
    selected_progress = progress_by_robot
    if active_mask is not None:
        mask = torch.as_tensor(active_mask, device=progress_by_robot.device, dtype=torch.bool)
        if mask.ndim != 1 or mask.shape[0] != progress_by_robot.shape[-1]:
            raise ValueError("active_mask must align with the robot dimension")
        if bool(mask.any()):
            selected_progress = progress_by_robot[..., mask]
    mean_progress = selected_progress.mean(dim=-1)
    if aggregation == "mean" or selected_progress.shape[-1] == 1:
        return mean_progress
    minimum_progress = selected_progress.min(dim=-1).values
    return mean_progress + float(min_blend) * (minimum_progress - mean_progress)


class SyntheticClosedLoopSimulator:
    """Receding-horizon simulator with persistent identity and joint candidates."""

    def __init__(self, spec: ClosedLoopSpec, episode_index: int) -> None:
        if spec.num_robots < 2:
            raise ValueError("the minimal closed-loop experiment requires at least two robots")
        if spec.num_candidates < spec.num_robots + 2:
            raise ValueError("num_candidates must be at least num_robots + 2")
        if not 1 <= spec.execute_steps <= spec.horizon:
            raise ValueError("execute_steps must lie in [1, horizon]")
        if spec.split not in {"id", "ood"}:
            raise ValueError("split must be 'id' or 'ood'")
        if str(spec.scene_family).lower() not in VALID_SCENE_FAMILIES:
            raise ValueError(
                f"scene_family must be one of {sorted(VALID_SCENE_FAMILIES)}"
            )
        if str(spec.response_family).lower() not in VALID_RESPONSE_FAMILIES:
            raise ValueError(
                f"response_family must be one of {sorted(VALID_RESPONSE_FAMILIES)}"
            )
        self.spec = spec
        if spec.task_type not in {"entity_approach", "fixed_goal"}:
            raise ValueError("task_type must be 'entity_approach' or 'fixed_goal'")
        if not np.isfinite(spec.goal_tolerance) or spec.goal_tolerance <= 0.0:
            raise ValueError("goal_tolerance must be positive and finite")
        split_offset = 0 if spec.split == "id" else 7_000_003
        self.rng = np.random.default_rng(spec.seed + split_offset + episode_index * 104729)
        self.mode = int(self.rng.integers(0, len(MODE_RESPONSE_GAINS)))
        if spec.scene_family == "radial":
            entity_position = self.rng.uniform(-0.6, 0.6, size=2)
        elif spec.scene_family == "corridor":
            entity_position = np.asarray(
                [self.rng.uniform(-0.6, 0.6), self.rng.normal(0.0, 0.25)]
            )
        elif spec.scene_family == "crossing":
            entity_position = self.rng.normal(0.0, 0.45, size=2)
        else:  # bottleneck
            entity_position = self.rng.normal(0.0, 0.30, size=2)
        self.entity_position = np.asarray(entity_position, dtype=np.float32)
        preferred_angle = self.rng.uniform(-np.pi, np.pi)
        preferred_speed = self.rng.uniform(0.55, 1.05)
        self.preferred_velocity = (
            np.asarray([np.cos(preferred_angle), np.sin(preferred_angle)]) * preferred_speed
        ).astype(np.float32)
        self.history = np.stack(
            [
                self.entity_position
                + self.preferred_velocity * ((step - spec.history + 1) * spec.dt)
                for step in range(spec.history)
            ]
        ).astype(np.float32)
        angular_noise = 0.16 if spec.split == "id" else 0.32
        radius_range = (2.2, 3.5) if spec.split == "id" else (1.8, 3.1)
        if spec.scene_family == "radial":
            base_angle = self.rng.uniform(-np.pi, np.pi)
            angular_layout = np.linspace(0.0, 2.0 * np.pi, spec.num_robots, endpoint=False)
            angles = base_angle + angular_layout + self.rng.normal(
                0.0, angular_noise, spec.num_robots
            )
            radii = self.rng.uniform(*radius_range, size=spec.num_robots)
            relative = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radii[:, None]
        elif spec.scene_family == "corridor":
            side = np.where(np.arange(spec.num_robots) % 2 == 0, -1.0, 1.0)
            relative = np.stack(
                [side * self.rng.uniform(*radius_range, size=spec.num_robots),
                 self.rng.normal(0.0, 0.30, size=spec.num_robots)], axis=-1
            )
        elif spec.scene_family == "crossing":
            horizontal_count = max(1, spec.num_robots // 2)
            horizontal = np.stack(
                [self.rng.uniform(*radius_range, size=horizontal_count),
                 self.rng.normal(0.0, 0.25, size=horizontal_count)], axis=-1
            )
            vertical_count = spec.num_robots - horizontal_count
            vertical = (
                np.stack(
                    [self.rng.normal(0.0, 0.25, size=vertical_count),
                     self.rng.uniform(*radius_range, size=vertical_count)], axis=-1
                )
                if vertical_count
                else np.empty((0, 2))
            )
            relative = np.concatenate([horizontal, vertical], axis=0)
            relative[:horizontal_count, 0] *= np.where(
                np.arange(horizontal_count) % 2 == 0, 1.0, -1.0
            )
            if vertical_count:
                relative[horizontal_count:, 1] *= np.where(
                    np.arange(vertical_count) % 2 == 0, 1.0, -1.0
                )
        else:  # bottleneck
            side = np.where(np.arange(spec.num_robots) % 2 == 0, -1.0, 1.0)
            relative = np.stack(
                [side * self.rng.uniform(*radius_range, size=spec.num_robots),
                 self.rng.normal(0.0, 0.35, size=spec.num_robots)], axis=-1
            )
        self.robot_positions = self.entity_position[None, :] + relative.astype(np.float32)
        self.robot_goals = (2.0 * self.entity_position[None] - self.robot_positions).copy()
        toward = _unit_np(self.entity_position[None, :] - self.robot_positions)
        self.robot_velocities = (
            toward * self.rng.uniform(0.35, 0.65, size=(spec.num_robots, 1))
        ).astype(np.float32)
        self.window = 0
        self.candidate_rng = np.random.default_rng(
            np.random.SeedSequence([spec.seed, split_offset, episode_index, 8675309])
        )

    def candidate_pool(self) -> CandidatePool:
        spec = self.spec
        relative = self.robot_positions - self.entity_position[None, :]
        activities = self.candidate_rng.uniform(
            0.18, 1.0, size=(spec.num_candidates, spec.num_robots)
        )
        anchors = [
            np.full(spec.num_robots, 0.2),
            np.full(spec.num_robots, 0.9),
        ]
        for robot in range(spec.num_robots):
            single_high = np.full(spec.num_robots, 0.2)
            single_high[robot] = 0.9
            anchors.append(single_high)
        anchor_count = min(len(anchors), spec.num_candidates)
        activities[:anchor_count] = np.asarray(anchors[:anchor_count])
        if spec.split == "ood" and spec.num_candidates > anchor_count:
            remaining = spec.num_candidates - anchor_count
            common = self.candidate_rng.uniform(0.18, 1.0, size=(remaining, 1))
            activities[anchor_count:] = np.clip(
                common
                + self.candidate_rng.normal(0.0, 0.025, size=(remaining, spec.num_robots)),
                0.18,
                1.0,
            )

        candidates = np.zeros(
            (spec.num_candidates, spec.num_robots, spec.horizon, 4), dtype=np.float32
        )
        for candidate_index in range(spec.num_candidates):
            activity = activities[candidate_index]
            toward = _unit_np(task_goals(self) - self.robot_positions)
            lateral = np.stack([-toward[:, 1], toward[:, 0]], axis=-1)
            lateral_sign = self.candidate_rng.choice(
                np.asarray([-1.0, 1.0]), size=spec.num_robots
            )
            lateral_scale = (1.0 - activity) * self.candidate_rng.uniform(
                0.45, 0.85, size=spec.num_robots
            )
            direction = _unit_np(toward + lateral * (lateral_sign * lateral_scale)[:, None])
            speed = 0.40 + 1.30 * activity
            turn_scale = 0.045 if spec.split == "id" else 0.07
            turns = self.candidate_rng.uniform(
                -turn_scale, turn_scale, size=spec.num_robots
            )
            position = relative.copy()
            for step in range(spec.horizon):
                angle = np.arctan2(direction[:, 1], direction[:, 0]) + turns * step
                velocity = np.stack([np.cos(angle), np.sin(angle)], axis=-1) * speed[:, None]
                position = position + velocity * spec.dt
                candidates[candidate_index, :, step, :2] = position
                candidates[candidate_index, :, step, 2:] = velocity

        current = np.concatenate([relative, self.robot_velocities], axis=-1).astype(np.float32)
        current = np.broadcast_to(current, (spec.num_candidates, *current.shape)).copy()
        return CandidatePool(
            robot_current=torch.from_numpy(current),
            robot_candidates=torch.from_numpy(candidates),
            activities=torch.from_numpy(activities.astype(np.float32)),
        )

    def _response_batch(
        self,
        robot_current: np.ndarray,
        robot_candidates: np.ndarray,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the response field for a batch of candidate joint actions."""

        spec = self.spec
        individual_gain, joint_gain, lateral_gain = MODE_RESPONSE_GAINS[self.mode]
        robot_current = np.asarray(robot_current, dtype=np.float32)
        robot_candidates = np.asarray(robot_candidates, dtype=np.float32)
        if robot_current.ndim != 3 or robot_current.shape[-1] != 4:
            raise ValueError("robot_current must have shape (candidates, robots, 4)")
        if robot_candidates.ndim != 4 or robot_candidates.shape[-1] != 4:
            raise ValueError(
                "robot_candidates must have shape (candidates, robots, horizon, 4)"
            )
        if tuple(robot_current.shape[:2]) != tuple(robot_candidates.shape[:2]):
            raise ValueError("robot_current and robot_candidates must align")
        if robot_candidates.shape[2] != spec.horizon:
            raise ValueError("robot_candidates horizon does not match the simulator")
        candidate_count = robot_current.shape[0]
        nominal_position = np.broadcast_to(
            self.entity_position, (candidate_count, 2)
        ).copy()
        robot_position = self.entity_position[None, None, :] + robot_current[:, :, :2]
        response = np.zeros((candidate_count, spec.horizon, 2), dtype=np.float32)
        if noise is None:
            noise = self.rng.normal(
                0.0, spec.response_noise, size=(spec.horizon, 2)
            )
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape == (spec.horizon, 2):
            noise = noise[None]
        if noise.shape != (1, spec.horizon, 2) and noise.shape != response.shape:
            raise ValueError(
                "noise must have shape (horizon, 2) or (candidates, horizon, 2)"
            )
        preferred = self.preferred_velocity[None, :]
        preferred_unit = _unit_np(self.preferred_velocity[None, :])[0]
        perpendicular = np.asarray([-preferred_unit[1], preferred_unit[0]])
        for step in range(spec.horizon):
            robot_velocity = robot_candidates[:, :, step, 2:]
            robot_position = robot_position + robot_velocity * spec.dt
            nominal_position = nominal_position + preferred * spec.dt
            away = nominal_position[:, None, :] - robot_position
            distance = np.linalg.norm(away, axis=-1)
            away_unit = _unit_np(away)
            approach_speed = np.sum(robot_velocity * away_unit, axis=-1)
            proximity = np.exp(-0.5 * np.square(distance / spec.influence_radius))
            approach_gate = 0.45 + 0.55 / (1.0 + np.exp(-2.0 * approach_speed))
            strength = proximity * approach_gate
            individual = (away_unit * strength[..., None]).sum(axis=1)
            pair_synergy = np.zeros(candidate_count, dtype=np.float32)
            pair_lateral = np.zeros((candidate_count, 2), dtype=np.float32)
            num_pairs = 0
            for first in range(spec.num_robots):
                for second in range(first + 1, spec.num_robots):
                    alignment = np.sum(
                        away_unit[:, first] * away_unit[:, second], axis=-1
                    )
                    opposition = np.clip((-alignment - 0.10) / 0.90, 0.0, 1.0)
                    raw_synergy = strength[:, first] * strength[:, second] * opposition
                    if spec.response_family == "threshold":
                        raw_synergy *= 1.0 / (1.0 + np.exp(-12.0 * (raw_synergy - 0.12)))
                    synergy = np.clip(raw_synergy * joint_gain, 0.0, 0.88)
                    midpoint = 0.5 * (
                        robot_position[:, first] + robot_position[:, second]
                    )
                    side = np.sign(
                        np.sum(
                            perpendicular[None, :]
                            * (nominal_position - midpoint),
                            axis=-1,
                        )
                    )
                    side = np.where(side == 0.0, 1.0, side)
                    pair_synergy += synergy
                    if spec.response_family != "speed":
                        pair_lateral += (
                            perpendicular[None, :] * side[:, None] * lateral_gain * synergy[:, None]
                        )
                    num_pairs += 1
            pair_synergy = np.clip(pair_synergy / num_pairs, 0.0, 0.88)
            pair_lateral = pair_lateral / num_pairs
            velocity = preferred + individual_gain * individual
            slowdown = 0.0 if spec.response_family == "lateral" else 0.70
            velocity = velocity * (1.0 - slowdown * pair_synergy)[:, None]
            velocity += pair_lateral
            velocity += noise[:, step]
            response[:, step] = _clip_norm_np(velocity, spec.entity_max_speed)
        return response

    def _response(
        self,
        robot_current: np.ndarray,
        robot_candidates: np.ndarray,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        current = np.asarray(robot_current, dtype=np.float32)[None]
        candidates = np.asarray(robot_candidates, dtype=np.float32)[None]
        return self._response_batch(current, candidates, noise=noise)[0]

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
        entity_displacement = np.cumsum(response[:steps] * self.spec.dt, axis=0)
        relative_robot = candidate[:, :steps, :2]
        distance = np.linalg.norm(relative_robot - entity_displacement[None, :, :], axis=-1)
        min_distance = float(distance.min())
        goal_offsets = task_goals(self) - self.robot_positions
        toward = _unit_np(goal_offsets)
        robot_displacement = candidate[:, steps - 1, :2] - current[:, :2]
        progress = float(np.sum(robot_displacement * toward, axis=-1).mean())
        if self.spec.task_type == "fixed_goal":
            progress = float((
                np.linalg.norm(goal_offsets, axis=-1)
                - np.linalg.norm(goal_offsets - robot_displacement, axis=-1)
            ).mean())
        if reference_response.shape != response.shape:
            raise ValueError("reference_response must match response shape")
        human_cost = float(
            np.square(response[:steps] - reference_response[:steps]).sum(axis=-1).mean()
        )
        safety_cost = float(np.exp(-min_distance / cost.comfort_scale))
        return RealizedWindow(
            response=torch.from_numpy(response.copy()),
            task_progress=progress,
            safety_cost=safety_cost,
            human_cost=human_cost,
            min_distance=min_distance,
            collision=float(min_distance < cost.collision_margin),
            mean_activity=float(pool.activities[selected].mean()),
            smoothness_cost=float(np.linalg.norm(
                candidate[:, 0, 2:] - current[:, 2:], axis=-1
            ).mean()),
        )

    def counterfactuals(
        self,
        pool: CandidatePool,
        cost: PlanningCostConfig,
    ) -> tuple[RealizedWindow, ...]:
        """Evaluate every candidate with common noise without advancing the simulator."""
        validate_candidate_state(self, pool)
        candidate_count = int(pool.robot_candidates.shape[0])
        if candidate_count == 0:
            raise ValueError("candidate pool must not be empty")
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        try:
            noise = self.rng.normal(
                0.0,
                self.spec.response_noise,
                size=(self.spec.horizon, 2),
            ).astype(np.float32)
            current = pool.robot_current.numpy()
            candidates = pool.robot_candidates.numpy()
            reference_candidates = _stationary_reference_np(
                current[0], self.spec.horizon
            )[None].repeat(candidate_count, axis=0)
            responses = self._response_batch(current, candidates, noise=noise)
            reference_responses = self._response_batch(
                current, reference_candidates, noise=noise
            )
            realized = [
                self._realized_window(
                    pool,
                    selected,
                    responses[selected],
                    reference_responses[selected],
                    cost,
                )
                for selected in range(candidate_count)
            ]
            return tuple(realized)
        finally:
            self.rng.bit_generator.state = rng_state

    def step(self, pool: CandidatePool, selected: int, cost: PlanningCostConfig) -> RealizedWindow:
        validate_candidate_state(self, pool)
        if not 0 <= selected < pool.robot_candidates.shape[0]:
            raise IndexError("selected candidate is outside the pool")
        current = pool.robot_current[selected].numpy()
        candidate = pool.robot_candidates[selected].numpy()
        noise = self.rng.normal(
            0.0,
            self.spec.response_noise,
            size=(self.spec.horizon, 2),
        ).astype(np.float32)
        reference_candidate = _stationary_reference_np(current, self.spec.horizon)
        response = self._response_batch(
            current[None], candidate[None], noise=noise
        )[0]
        reference_response = self._response_batch(
            current[None], reference_candidate[None], noise=noise
        )[0]
        realized = self._realized_window(
            pool, selected, response, reference_response, cost
        )
        steps = self.spec.execute_steps

        for step in range(steps):
            self.entity_position = self.entity_position + response[step] * self.spec.dt
            self.robot_positions = self.robot_positions + candidate[:, step, 2:] * self.spec.dt
            self.history = np.concatenate([self.history[1:], self.entity_position[None, :]], axis=0)
        self.robot_velocities = candidate[:, steps - 1, 2:].copy()
        self.window += 1
        return realized


def expected_candidate_costs(
    mode_means: torch.Tensor,
    belief: torch.Tensor,
    robot_current: torch.Tensor,
    robot_candidates: torch.Tensor,
    reference_mode_means: torch.Tensor,
    dt: float,
    config: PlanningCostConfig,
    mode_log_stds: torch.Tensor | None = None,
    task_direction: torch.Tensor | None = None,
    task_goal_offsets: torch.Tensor | None = None,
    response_valid_steps: torch.Tensor | None = None,
    risk_gradient_temperature: float | None = None,
    task_active_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate a finite joint candidate set under the current mode belief.

    Mode-dependent safety and human-response terms are evaluated before mixing.
    This preserves multimodal risk instead of collapsing the response mixture to
    a single mean trajectory. Human cost is measured relative to the model's
    reference-joint-action response, avoiding simulator-only hidden state.
    ``risk_gradient_temperature`` optionally provides a straight-through
    training gradient for sampled risk; forward scores and safety masks stay
    identical to the default deployed threshold calculation.
    """

    if mode_means.ndim != 4:
        raise ValueError("mode_means must have shape (candidates, modes, horizon, 2)")
    if risk_gradient_temperature is not None and (
        not np.isfinite(risk_gradient_temperature) or risk_gradient_temperature <= 0.0
    ):
        raise ValueError("risk_gradient_temperature must be finite and positive")
    if tuple(reference_mode_means.shape) != tuple(mode_means.shape):
        raise ValueError("reference_mode_means must match mode_means")
    if mode_log_stds is not None and tuple(mode_log_stds.shape) != tuple(mode_means.shape):
        raise ValueError("mode_log_stds must match mode_means")
    if tuple(belief.shape) != (mode_means.shape[1],):
        raise ValueError("belief must have shape (modes,)")
    if tuple(robot_current.shape[:2]) != tuple(robot_candidates.shape[:2]):
        raise ValueError("robot_current and robot_candidates must align")
    task_steps = (
        int(config.task_steps)
        if config.task_steps is not None
        else int(mode_means.shape[-2])
    )
    if not 1 <= task_steps <= int(mode_means.shape[-2]):
        raise ValueError("task_steps must lie within the candidate horizon")
    response_steps = (
        int(config.response_steps)
        if config.response_steps is not None
        else int(mode_means.shape[-2])
    )
    if not 1 <= response_steps <= int(mode_means.shape[-2]):
        raise ValueError("response_steps must lie within the candidate horizon")
    if config.risk_weight < 0.0 or config.smoothness_weight < 0.0:
        raise ValueError("risk_weight and smoothness_weight must be non-negative")
    if not np.isfinite(config.task_speed_scale) or config.task_speed_scale <= 0.0:
        raise ValueError("task_speed_scale must be finite and positive")
    if str(config.task_aggregation).lower() not in {"mean", "lower-tail"}:
        raise ValueError("task_aggregation must be 'mean' or 'lower-tail'")
    if not np.isfinite(config.task_min_blend) or not 0.0 <= float(config.task_min_blend) <= 1.0:
        raise ValueError("task_min_blend must lie in [0, 1]")

    mode_weights = belief.to(device=mode_means.device, dtype=mode_means.dtype)
    mode_weights = mode_weights / mode_weights.sum().clamp_min(1e-9)
    response_means = mode_means[:, :, :response_steps]
    response_valid = torch.ones(
        mode_means.shape[0], response_steps, dtype=torch.bool, device=mode_means.device,
    )
    if response_valid_steps is not None:
        if tuple(response_valid_steps.shape) != (mode_means.shape[0], mode_means.shape[-2]):
            raise ValueError("response_valid_steps must have shape (candidates, horizon)")
        response_valid = response_valid_steps[:, :response_steps].to(device=mode_means.device, dtype=torch.bool)
        if not response_valid[:, 0].all() or (response_valid[:, 1:] & ~response_valid[:, :-1]).any():
            raise ValueError("response_valid_steps must define a nonempty contiguous prefix")
    entity_displacement = torch.cumsum(response_means * dt, dim=2)
    robot_relative = robot_candidates[..., :response_steps, :2]
    distances = torch.linalg.vector_norm(
        robot_relative[:, None] - entity_displacement[:, :, None], dim=-1
    )
    distances = distances.masked_fill(~response_valid[:, None, None], float("inf"))
    mode_min_distance = distances.flatten(start_dim=2).min(dim=2).values
    min_distance = mode_min_distance.min(dim=1).values
    if task_direction is None:
        # Legacy synthetic protocol: the entity is the task target.
        toward = torch.nn.functional.normalize(-robot_current[:, :, :2], dim=-1)
    else:
        toward = torch.as_tensor(
            task_direction, device=robot_candidates.device, dtype=robot_candidates.dtype
        )
        if toward.shape not in {(2,), tuple(robot_candidates.shape[1:2]) + (2,)}:
            raise ValueError(
                "task_direction must have shape (2,) or (robots, 2)"
            )
        toward = torch.nn.functional.normalize(toward, dim=-1)
        if toward.ndim == 1:
            toward = toward.view(1, 1, 2)
        else:
            toward = toward.view(1, robot_candidates.shape[1], 2)
    robot_displacement = (
        robot_candidates[:, :, task_steps - 1, :2] - robot_current[:, :, :2]
    )
    progress_by_robot = (robot_displacement * toward).sum(dim=-1)
    progress = aggregate_robot_progress(
        progress_by_robot,
        config.task_aggregation,
        task_active_mask,
        config.task_min_blend,
    )
    if task_goal_offsets is not None:
        offsets = task_goal_offsets.to(robot_candidates)
        if offsets.shape != robot_current.shape[1:2] + (2,):
            raise ValueError("task_goal_offsets must have shape (robots, 2)")
        progress_by_robot = (
            torch.linalg.vector_norm(offsets, dim=-1)
            - torch.linalg.vector_norm(offsets[None] - robot_displacement, dim=-1)
        )
        progress = aggregate_robot_progress(
            progress_by_robot,
            config.task_aggregation,
            task_active_mask,
            config.task_min_blend,
        )
    task = -progress / max(task_steps * dt * config.task_speed_scale, 1e-6)
    mode_safety = torch.exp(-mode_min_distance / config.comfort_scale)
    safety = (mode_safety * mode_weights[None]).sum(dim=1)
    human_per_step = torch.square(
        response_means - reference_mode_means[:, :, :response_steps]
    ).sum(dim=-1)
    if response_valid_steps is None:
        mode_human = human_per_step.mean(dim=-1)
    else:
        mode_human = human_per_step.masked_fill(~response_valid[:, None], 0).sum(-1) / response_valid.sum(-1)[:, None]
    human = (mode_human * mode_weights[None]).sum(dim=1)
    smoothness = torch.linalg.vector_norm(
        robot_candidates[:, :, 0, 2:] - robot_current[:, :, 2:], dim=-1
    ).mean(dim=-1)
    total = (
        config.task_weight * task
        + config.safety_weight * safety
        + config.human_weight * human
        + config.smoothness_weight * smoothness
    )
    safe_mask = torch.ones_like(total, dtype=torch.bool)
    violation_probability = torch.zeros_like(total)
    if config.violation_probability_alpha is not None or config.risk_weight > 0.0:
        if config.violation_probability_alpha is not None and not 0.0 < float(config.violation_probability_alpha) < 1.0:
            raise ValueError("violation_probability_alpha must lie in (0, 1)")
        if mode_log_stds is None:
            raise ValueError("mode_log_stds are required for probabilistic safety or risk ranking")
        if int(config.risk_samples) < 2:
            raise ValueError("risk_samples must be at least two")
        sample_ids = torch.arange(
            int(config.risk_samples), device=mode_means.device, dtype=mode_means.dtype
        )
        u1 = ((sample_ids + 0.5) / float(config.risk_samples)).clamp(1e-5, 1.0 - 1e-5)
        u2 = ((sample_ids * 0.61803398875 + 0.5 / float(config.risk_samples)) % 1.0).clamp(
            1e-5, 1.0 - 1e-5
        )
        normal = torch.stack(
            [
                torch.sqrt(torch.tensor(2.0, device=mode_means.device, dtype=mode_means.dtype))
                * torch.erfinv(2.0 * u1 - 1.0),
                torch.sqrt(torch.tensor(2.0, device=mode_means.device, dtype=mode_means.dtype))
                * torch.erfinv(2.0 * u2 - 1.0),
            ],
            dim=-1,
        )
        if config.temporal_risk_sampling == "independent":
            horizon = response_steps
            normal = _independent_trajectory_noise(int(config.risk_samples), horizon).to(mode_means)
        elif config.temporal_risk_sampling == "shared":
            normal = normal[:, None]
        else:
            raise ValueError("temporal_risk_sampling must be 'independent' or 'shared'")
        samples = response_means[:, :, None] + mode_log_stds[:, :, None, :response_steps].exp() * normal[None, None]
        sample_positions = torch.cumsum(samples * dt, dim=3)
        sample_distances = torch.linalg.vector_norm(
            robot_relative[:, None, None] - sample_positions[:, :, :, None], dim=-1
        )
        sample_distances = sample_distances.masked_fill(~response_valid[:, None, None, None], float("inf"))
        mode_violation_probability = sample_distances.lt(config.collision_margin).any(
            dim=(-1, -2)
        ).to(mode_means.dtype).mean(dim=-1)
        if risk_gradient_temperature is not None:
            sample_clearance = sample_distances.flatten(start_dim=-2).amin(dim=-1)
            smooth_probability = torch.sigmoid(
                (config.collision_margin - sample_clearance) / risk_gradient_temperature
            ).mean(dim=-1)
            mode_violation_probability = mode_violation_probability + (
                smooth_probability - smooth_probability.detach()
            )
        violation_probability = (mode_violation_probability * mode_weights[None]).sum(dim=1)
        total = total + config.risk_weight * violation_probability
        if config.violation_probability_alpha is not None:
            safe_mask = violation_probability.le(float(config.violation_probability_alpha))
    unconstrained_total = total
    total = torch.where(safe_mask, total, torch.full_like(total, float("inf")))
    return {
        "total": total,
        "unconstrained_total": unconstrained_total,
        "task": task,
        "safety": safety,
        "human": human,
        "smoothness": smoothness,
        "min_distance": min_distance,
        "violation_probability": violation_probability,
        "safe_mask": safe_mask,
    }
