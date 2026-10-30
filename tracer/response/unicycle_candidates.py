from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class UnicycleLimits:
    max_speed: float = 0.95
    max_yaw_rate: float = np.pi / 3
    acceleration: float = 0.5
    deceleration: float = 1.5

    @classmethod
    def from_sicnav_config(cls, path: Path):
        config = configparser.RawConfigParser()
        if not config.read(path):
            raise FileNotFoundError(path)
        return cls(config.getfloat("mpc_env", "max_speed"),
                   np.deg2rad(config.getfloat("mpc_env", "max_rot_degrees")),
                   config.getfloat("mpc_env", "max_l_acc"),
                   -config.getfloat("mpc_env", "max_l_dcc"))


def project_unicycle_candidates(candidates, position, velocity, heading, dt, limits, return_headings=False):
    if dt <= 0 or any(not np.isfinite(value) or value <= 0 for value in vars(limits).values()):
        raise ValueError("unicycle clock and motion limits must be finite and positive")
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[-1] != 4 or not np.isfinite(candidates).all():
        raise ValueError("unicycle candidates must have finite shape (candidates, horizon, 4)")
    count, horizon = candidates.shape[:2]
    result = candidates.copy()
    positions = np.broadcast_to(position, (count, 2)).astype(np.float64).copy()
    headings = np.full(count, heading, dtype=np.float64)
    speeds = np.full(count, np.linalg.norm(velocity), dtype=np.float64)
    heading_history = np.zeros((count, horizon), dtype=np.float64)
    signed_initial = float(np.asarray(velocity) @ np.array([np.cos(heading), np.sin(heading)]))
    directions = np.full(count, 1.0 if signed_initial >= 0 else -1.0)
    for step in range(horizon):
        desired = candidates[:, step, 2:]
        desired_speed = np.linalg.norm(desired, axis=-1)
        desired_heading = np.where(desired_speed > 1e-8, np.arctan2(desired[:, 1], desired[:, 0]), headings)
        choose_reverse = np.cos(desired_heading - headings) < 0.0
        directions = np.where(speeds <= 1e-8, np.where(choose_reverse, -1.0, 1.0), directions)
        desired_heading = np.where(directions < 0.0, desired_heading + np.pi, desired_heading)
        desired_heading = np.where(desired_speed > 1e-8, desired_heading, headings)
        heading_error = (desired_heading - headings + np.pi) % (2 * np.pi) - np.pi
        yaw = np.clip(heading_error, -limits.max_yaw_rate * dt, limits.max_yaw_rate * dt)
        headings += yaw
        target_speed = np.minimum(desired_speed, limits.max_speed)
        target_speed *= np.maximum(0.0, np.cos(heading_error - yaw))
        speeds = np.clip(target_speed, np.maximum(speeds - limits.deceleration * dt, 0.0),
                         np.minimum(speeds + limits.acceleration * dt, limits.max_speed))
        velocities = (directions * speeds)[:, None] * np.stack([np.cos(headings), np.sin(headings)], axis=-1)
        positions += velocities * dt
        result[:, step, :2] = positions
        result[:, step, 2:] = velocities
        heading_history[:, step] = headings
    return (result, heading_history) if return_headings else result


def braking_after_prefix(candidates, dt, deceleration, execute_steps=1):
    """Append a straight, maximum-deceleration stop to each executed prefix."""
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 3 or candidates.shape[-1] != 4 or not np.isfinite(candidates).all():
        raise ValueError("braking candidates must have finite shape (candidates, horizon, 4)")
    if not len(candidates) or not 1 <= execute_steps <= candidates.shape[1]:
        raise ValueError("braking requires nonempty candidates and a valid executed prefix")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(deceleration) or deceleration <= 0:
        raise ValueError("braking clock and deceleration must be finite and positive")
    speeds = np.linalg.norm(candidates[:, execute_steps - 1, 2:], axis=-1)
    directions = candidates[:, execute_steps - 1, 2:] / np.maximum(speeds[:, None], 1e-12)
    stop_steps = max(1, int(np.ceil(float(speeds.max()) / (deceleration * dt))))
    future_speeds = np.maximum(speeds[:, None] - np.arange(1, stop_steps + 1) * deceleration * dt, 0.)
    velocities = future_speeds[..., None] * directions[:, None]
    positions = candidates[:, execute_steps - 1:execute_steps, :2] + np.cumsum(velocities * dt, axis=1)
    return np.concatenate([candidates[:, :execute_steps], np.concatenate([positions, velocities], axis=-1)], axis=1)


def diversify_unicycle_prefixes(candidates, position, velocity, heading, dt, limits, execute_steps=1):
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[-1] != 4 or not len(candidates):
        raise ValueError("prefix diversity requires nonempty candidate trajectories")
    if not 1 <= execute_steps <= candidates.shape[1]:
        raise ValueError("prefix diversity needs a valid executed prefix")
    initial_speed = float(np.linalg.norm(velocity))
    lower_speed = max(0.0, initial_speed - limits.deceleration * dt)
    variants = []
    for fraction in (0.25, 0.5, 0.75):
        proposed = candidates.copy()
        speeds = np.linalg.norm(proposed[:, :execute_steps, 2:], axis=-1)
        lower = np.maximum(0.0, initial_speed - limits.deceleration * dt * np.arange(1, execute_steps + 1))
        target_speeds = lower[None] + fraction * (speeds - lower[None])
        proposed[:, :execute_steps, 2:] *= (target_speeds / np.maximum(speeds, 1e-12))[..., None]
        variants.append(proposed)
    projected, headings = project_unicycle_candidates(
        np.concatenate(variants), position, velocity, heading, dt, limits, return_headings=True,
    )
    if np.any(np.linalg.norm(projected[:, 0, 2:], axis=-1) < lower_speed - 1e-6):
        raise AssertionError("prefix diversity violated the deceleration limit")
    seen = {tuple(np.round(candidate.reshape(-1), 6)) for candidate in candidates}
    selected = []
    for index, candidate in enumerate(projected):
        key = tuple(np.round(candidate.reshape(-1), 6))
        if key not in seen:
            selected.append(index)
            seen.add(key)
    return projected[selected], headings[selected]
