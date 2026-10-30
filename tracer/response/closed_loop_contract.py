from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def task_goals(simulator: Any) -> np.ndarray:
    if getattr(simulator.spec, "task_type", "entity_approach") == "fixed_goal":
        return simulator.robot_goals.copy()
    return np.broadcast_to(
        simulator.entity_position, simulator.robot_positions.shape
    ).copy()


def observable_state_sha256(simulator: Any) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("history", simulator.history),
        ("robot_positions", simulator.robot_positions),
        ("robot_velocities", simulator.robot_velocities),
        ("task_goals", task_goals(simulator)),
    ):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_candidate_state(simulator: Any, pool: Any) -> dict[str, float]:
    current = pool.robot_current.detach().cpu().numpy()
    candidates = pool.robot_candidates.detach().cpu().numpy()
    activities = pool.activities.detach().cpu().numpy()
    robots = len(simulator.robot_positions)
    if candidates.ndim != 4 or candidates.shape[1:] != (
        robots, simulator.spec.horizon, 4
    ):
        raise ValueError("candidate trajectories must match the simulator robots/horizon")
    if len(candidates) == 0 or current.shape != (len(candidates), robots, 4):
        raise ValueError("candidate current states must align with trajectories")
    if activities.shape != (len(candidates), robots):
        raise ValueError("candidate activities must align with trajectories")
    if not all(np.isfinite(value).all() for value in (current, candidates, activities)):
        raise ValueError("candidate pool must contain only finite values")
    actual_position = simulator.robot_positions - simulator.entity_position[None]
    actual_velocity = simulator.robot_velocities
    integrated = current[:, :, None, :2] + np.cumsum(
        candidates[..., 2:] * simulator.spec.dt, axis=2
    )
    checks = {
        "candidate_position_error": (current[..., :2], actual_position[None]),
        "candidate_velocity_error": (current[..., 2:], actual_velocity[None]),
        "candidate_integration_error": (candidates[..., :2], integrated),
    }
    errors = {}
    for name, (supplied, expected) in checks.items():
        errors[name] = float(np.abs(supplied - expected).max())
        if not np.allclose(supplied, expected, rtol=1e-5, atol=2e-5):
            raise ValueError(f"candidate/state contract violated: {name}={errors[name]:.6g}")
    return errors
