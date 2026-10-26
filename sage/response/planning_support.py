from __future__ import annotations

import numpy as np
import torch


def belief_cost_adjustment(
    posterior_cost: torch.Tensor,
    uniform_cost: torch.Tensor,
    leverage: float = 1.0,
) -> torch.Tensor:
    if not np.isfinite(leverage) or leverage <= 0.0:
        raise ValueError("belief-cost-leverage must be finite and positive")
    if posterior_cost.shape != uniform_cost.shape:
        raise ValueError("belief cost tensors must have matching shapes")
    if not torch.isfinite(posterior_cost).all() or not torch.isfinite(uniform_cost).all():
        raise ValueError("belief adjustment requires finite unmasked costs")
    if leverage == 1.0:
        return torch.zeros_like(posterior_cost)
    return (float(leverage) - 1.0) * (posterior_cost - uniform_cost)


def swept_surface_clearance(
    candidates: np.ndarray,
    robot_position: np.ndarray,
    human_positions: np.ndarray,
    human_velocities: np.ndarray,
    body_radii: np.ndarray,
    dt: float,
    steps: int | None = None,
) -> np.ndarray:
    """Return minimum linear-segment surface clearance per candidate and human."""

    paths = np.asarray(candidates, dtype=np.float64)
    positions = np.asarray(human_positions, dtype=np.float64).reshape(-1, 2)
    velocities = np.asarray(human_velocities, dtype=np.float64).reshape(-1, 2)
    radii = np.asarray(body_radii, dtype=np.float64).reshape(-1)
    if paths.ndim != 3 or paths.shape[-1] != 4 or paths.shape[1] < 1:
        raise ValueError("candidates must have shape (candidates, horizon, 4)")
    if positions.shape != velocities.shape or len(radii) != len(positions):
        raise ValueError("human positions, velocities and radii must align")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not len(positions):
        return np.empty((len(paths), 0), dtype=np.float64)
    horizon = paths.shape[1] if steps is None else min(max(1, int(steps)), paths.shape[1])
    times = np.arange(horizon + 1, dtype=np.float64) * dt
    human_path = positions[:, None] + velocities[:, None] * times[None, :, None]
    robot_start = np.broadcast_to(np.asarray(robot_position), (len(paths), 1, 2))
    robot_path = np.concatenate([robot_start, paths[:, :horizon, :2]], axis=1)
    relative = robot_path[:, None] - human_path[None]
    start = relative[:, :, :-1]
    delta = relative[:, :, 1:] - start
    fraction = np.clip(
        -(start * delta).sum(axis=-1) / np.maximum((delta * delta).sum(axis=-1), 1e-12),
        0.0, 1.0,
    )
    closest = start + fraction[..., None] * delta
    return np.linalg.norm(closest, axis=-1).min(axis=-1) - radii[None]


def diverse_candidate_indices(
    trajectories: torch.Tensor,
    utility: torch.Tensor,
    count: int,
    diversity_weight: float = 0.4,
) -> torch.Tensor:
    """Keep the best task branch and greedily retain distinct control paths."""

    if trajectories.ndim < 3 or len(trajectories) != utility.numel():
        raise ValueError("trajectory and utility candidate dimensions must align")
    if not 1 <= count <= len(trajectories):
        raise ValueError("count must lie within the candidate pool")
    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError("diversity_weight must lie in [0, 1]")
    if not torch.isfinite(trajectories).all() or not torch.isfinite(utility).all():
        raise ValueError("candidate trajectories and utilities must be finite")
    position_features = trajectories[..., :2]
    position_features = position_features - position_features[..., :1, :]
    velocity_features = trajectories[..., 2:]
    features = torch.cat([position_features, velocity_features], dim=-1).flatten(
        start_dim=1
    )
    distances = torch.cdist(features, features) / max(features.shape[-1] ** 0.5, 1.0)
    distances = distances / distances.max().clamp_min(1e-6)
    quality = (utility - utility.min()) / (utility.max() - utility.min()).clamp_min(1e-6)
    selected = [int(utility.argmax())]
    available = torch.ones(len(trajectories), device=trajectories.device, dtype=torch.bool)
    nearest = distances[selected[0]].clone()
    for _ in range(count - 1):
        available[selected[-1]] = False
        score = (1.0 - diversity_weight) * quality + diversity_weight * nearest
        index = int(score.masked_fill(~available, float("-inf")).argmax())
        selected.append(index)
        nearest = torch.minimum(nearest, distances[index])
    return torch.tensor(selected, device=trajectories.device, dtype=torch.long)


def goal_return_trajectories(
    candidates: np.ndarray,
    robot_position: np.ndarray,
    robot_goal: np.ndarray,
    dt: float,
    speed_limit: float,
    turn_steps: int = 4,
) -> np.ndarray:
    """Keep the checked first control, then smoothly return toward the real goal."""

    paths = np.asarray(candidates, dtype=np.float32).copy()
    if paths.ndim != 3 or paths.shape[-1] != 4:
        raise ValueError("candidates must have shape (candidates, horizon, 4)")
    if dt <= 0.0 or speed_limit <= 0.0 or turn_steps < 1:
        raise ValueError("dt, speed_limit and turn_steps must be positive")
    position = np.broadcast_to(np.asarray(robot_position, dtype=np.float32), (len(paths), 2)).copy()
    initial_velocity = paths[:, 0, 2:].copy()
    goal = np.asarray(robot_goal, dtype=np.float32)
    for step in range(paths.shape[1]):
        displacement = goal[None] - position
        distance = np.linalg.norm(displacement, axis=-1, keepdims=True)
        goal_velocity = displacement / np.maximum(distance, 1e-6) * np.minimum(speed_limit, distance / dt)
        blend = min(float(step) / turn_steps, 1.0)
        velocity = (1.0 - blend) * initial_velocity + blend * goal_velocity
        position = position + dt * velocity
        paths[:, step, :2] = position
        paths[:, step, 2:] = velocity
    return paths


def aggregate_entity_costs(
    costs: list[dict[str, torch.Tensor]],
    task_weight: float,
    safety_weight: float,
    human_weight: float,
    smoothness_weight: float,
    risk_weight: float,
    aggregation: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate crowd risk without diluting a dangerous entity among safe ones."""

    if not costs or aggregation not in {"mean", "max", "union-bound"}:
        raise ValueError("nonempty costs and a supported risk aggregation are required")
    components = {
        name: torch.stack([cost[name] for cost in costs])
        for name in ("task", "safety", "human", "smoothness", "violation_probability")
    }
    if aggregation == "mean":
        risk = components["violation_probability"].mean(dim=0)
        safety = components["safety"].mean(dim=0)
    else:
        risk = components["violation_probability"].amax(dim=0)
        if aggregation == "union-bound":
            risk = components["violation_probability"].sum(dim=0).clamp_max(1.0)
        safety = components["safety"].amax(dim=0)
    total = (
        task_weight * components["task"].mean(dim=0)
        + safety_weight * safety
        + human_weight * components["human"].mean(dim=0)
        + smoothness_weight * components["smoothness"].mean(dim=0)
        + risk_weight * risk
    )
    return total, risk
