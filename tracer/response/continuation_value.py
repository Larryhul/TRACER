"""Shared finite-horizon value evaluated at predicted next observable states."""

from dataclasses import dataclass
import math

import torch


def observable_value_features(history, robot_current, remaining_windows, total_windows, dt):
    if history.shape[-1] != 2 or history.shape[-2] < 3 or robot_current.shape[-1] != 4:
        raise ValueError("value features require position history and robot position/velocity")
    if history.shape[:-2] != robot_current.shape[:-2] or robot_current.shape[-2] < 2:
        raise ValueError("value feature batch dimensions and robot count must align")
    if not math.isfinite(dt) or dt <= 0 or not 0 <= remaining_windows <= total_windows or total_windows < 1:
        raise ValueError("finite timestep and valid remaining horizon are required")
    human_velocities = torch.diff(history, dim=-2) / dt
    human_velocity = human_velocities[..., -1, :]
    human_acceleration = (human_velocities[..., -1, :] - human_velocities[..., -2, :]) / dt
    position, velocity = robot_current[..., :2], robot_current[..., 2:]
    distance = torch.linalg.vector_norm(position, dim=-1)
    direction = position / distance.clamp_min(1e-5)[..., None]
    relative_velocity = velocity - human_velocity[..., None, :]
    measures = (
        distance, distance.square(), torch.exp(-distance),
        torch.linalg.vector_norm(velocity, dim=-1),
        torch.linalg.vector_norm(relative_velocity, dim=-1),
        (relative_velocity * direction).sum(-1),
        (human_velocity[..., None, :] * direction).sum(-1),
    )
    summaries = [operation(value, dim=-1) for value in measures for operation in (torch.mean, torch.amin, torch.amax)]
    indices = torch.triu_indices(position.shape[-2], position.shape[-2], offset=1, device=position.device)
    separations = torch.linalg.vector_norm(position[..., indices[0], :] - position[..., indices[1], :], dim=-1)
    summaries.extend([
        separations.mean(-1), separations.amin(-1),
        torch.linalg.vector_norm(human_velocity, dim=-1),
        torch.linalg.vector_norm(human_acceleration, dim=-1),
        torch.linalg.vector_norm(human_velocities, dim=-1).mean(-1),
        torch.full_like(distance[..., 0], remaining_windows / total_windows),
    ])
    features = torch.stack(summaries, dim=-1)
    if not torch.isfinite(features).all():
        raise ValueError("observable value features must be finite")
    return features


def predicted_next_value_features(history, robot_current, robot_candidates, mode_means, execute_steps, remaining_windows, total_windows, dt):
    candidates, modes = mode_means.shape[:2]
    if mode_means.ndim != 4 or not 1 <= execute_steps <= mode_means.shape[-2]:
        raise ValueError("mode responses and executed prefix must align")
    if history.ndim != 2 or robot_current.shape[:1] != (candidates,) or robot_candidates.shape[0] != candidates:
        raise ValueError("predicted value state must align with the candidate pool")
    human_positions = (mode_means[:, :, :execute_steps] * dt).cumsum(-2)
    old_history = (history - history[-1])[None, None].expand(candidates, modes, -1, -1)
    next_history = torch.cat([old_history, human_positions], dim=-2)[..., -history.shape[-2]:, :]
    robot_end = robot_candidates[:, :, execute_steps - 1]
    next_positions = robot_end[:, None, :, :2] - human_positions[:, :, -1, None, :]
    next_velocities = robot_end[:, None, :, 2:].expand(-1, modes, -1, -1)
    next_current = torch.cat([next_positions, next_velocities], dim=-1)
    return observable_value_features(next_history, next_current, remaining_windows, total_windows, dt)


@dataclass
class FrozenContinuationValue:
    center: torch.Tensor
    scale: torch.Tensor
    coefficients: torch.Tensor
    total_windows: int
    dt: float
    execute_steps: int

    def __post_init__(self):
        if self.center.ndim != 1 or self.scale.shape != self.center.shape or self.coefficients.ndim != 2 or self.coefficients.shape[1] != self.center.numel() + 1 or self.coefficients.shape[0] < 2:
            raise ValueError("continuation value parameter dimensions must align")
        if not all(torch.isfinite(value).all() for value in (self.center, self.scale, self.coefficients)) or (self.scale <= 0).any():
            raise ValueError("continuation value parameters must be finite with positive feature scales")
        if self.total_windows < 1 or self.execute_steps < 1 or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("continuation value time contract must be positive")

    def mode_values(self, features, remaining_windows):
        if not 0 <= remaining_windows <= self.total_windows or features.shape[-2] != self.coefficients.shape[0]:
            raise ValueError("value modes and remaining horizon must match the fitted head")
        normalized = (features - self.center.to(features)) / self.scale.to(features)
        design = torch.cat([torch.ones_like(normalized[..., :1]), normalized], dim=-1)
        values = (design * self.coefficients.to(features)).sum(-1) * remaining_windows
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite continuation value")
        return values

    def candidate_values(self, history, current, candidates, means, belief, remaining_windows):
        if belief.shape != (means.shape[1],) or not torch.isfinite(belief).all() or (belief < 0).any() or not torch.isclose(belief.sum(), belief.new_tensor(1.)):
            raise ValueError("continuation requires a normalized current decision belief")
        features = predicted_next_value_features(history, current, candidates, means, self.execute_steps, remaining_windows, self.total_windows, self.dt)
        return (self.mode_values(features, remaining_windows) * belief[None]).sum(-1)


def add_continuation_value(costs, values, weight):
    if not math.isfinite(weight) or weight < 0 or values.shape != costs["unconstrained_total"].shape or not torch.isfinite(values).all():
        raise ValueError("finite continuation values and nonnegative weight required")
    result = dict(costs)
    adjustment = weight * (values - values.mean())
    result["continuation_adjustment"] = adjustment
    result["unconstrained_total"] = costs["unconstrained_total"] + adjustment
    result["total"] = torch.where(costs["safe_mask"], result["unconstrained_total"], torch.full_like(values, float("inf")))
    return result
