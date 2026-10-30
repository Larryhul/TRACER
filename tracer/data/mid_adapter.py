from __future__ import annotations

from typing import Optional

import torch

from .heterogeneous_scene import AgentRole, HeterogeneousSceneBatch


ROLE_STRATEGIES = ("speed_threshold", "speed", "speed_quantile", "all_pha", "all_unknown")


def _as_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.bool() if mask.dtype is not torch.bool else mask


def _speed_threshold_roles(
    speed: torch.Tensor,
    slow_speed: float,
    fast_speed: float,
) -> torch.Tensor:
    roles = torch.full_like(speed, int(AgentRole.UNKNOWN_ENTITY), dtype=torch.long)
    roles = torch.where(speed < slow_speed, torch.full_like(roles, int(AgentRole.SSAA)), roles)
    roles = torch.where(
        (speed >= slow_speed) & (speed < fast_speed),
        torch.full_like(roles, int(AgentRole.PHA)),
        roles,
    )
    roles = torch.where(speed >= fast_speed, torch.full_like(roles, int(AgentRole.HNAA)), roles)
    return roles


def _speed_quantile_roles(
    speed: torch.Tensor,
    entity_mask: torch.Tensor,
    slow_speed: float,
    fast_speed: float,
    slow_quantile: float,
    fast_quantile: float,
) -> torch.Tensor:
    active = speed[entity_mask]
    if active.numel() < 3:
        return _speed_threshold_roles(speed, slow_speed=slow_speed, fast_speed=fast_speed)

    sorted_speed = torch.sort(active).values
    n = sorted_speed.numel()
    low_idx = min(n - 1, max(0, int((n - 1) * slow_quantile)))
    high_idx = min(n - 1, max(0, int((n - 1) * fast_quantile)))
    low = sorted_speed[low_idx]
    high = sorted_speed[high_idx]

    roles = torch.full_like(speed, int(AgentRole.PHA), dtype=torch.long)
    roles = torch.where(speed <= low, torch.full_like(roles, int(AgentRole.SSAA)), roles)
    roles = torch.where(speed >= high, torch.full_like(roles, int(AgentRole.HNAA)), roles)
    return roles


def infer_entity_roles(
    future_vel: torch.Tensor,
    entity_mask: torch.Tensor,
    strategy: str = "speed_threshold",
    slow_speed: float = 0.15,
    fast_speed: float = 1.8,
    slow_quantile: float = 0.33,
    fast_quantile: float = 0.67,
) -> torch.Tensor:
    """Assign pseudo heterogeneous roles for datasets without semantic labels.

    The default ``speed_threshold`` strategy matches the original behavior used
    by the ETH/UCY adapter. Alternative strategies are intended for controlled
    pseudo-heterogeneity ablations and must be enabled explicitly.
    """

    entity_mask = _as_bool_mask(entity_mask)
    speed = torch.linalg.norm(future_vel, dim=-1).mean(dim=-1)
    strategy = "speed_threshold" if strategy == "speed" else strategy
    if strategy not in ROLE_STRATEGIES:
        raise ValueError(f"Unknown role strategy {strategy!r}; choose from {ROLE_STRATEGIES}")

    if strategy == "speed_threshold":
        roles = _speed_threshold_roles(speed, slow_speed=slow_speed, fast_speed=fast_speed)
    elif strategy == "speed_quantile":
        roles = _speed_quantile_roles(
            speed,
            entity_mask,
            slow_speed=slow_speed,
            fast_speed=fast_speed,
            slow_quantile=slow_quantile,
            fast_quantile=fast_quantile,
        )
    elif strategy == "all_pha":
        roles = torch.full_like(speed, int(AgentRole.PHA), dtype=torch.long)
    elif strategy == "all_unknown":
        roles = torch.full_like(speed, int(AgentRole.UNKNOWN_ENTITY), dtype=torch.long)
    else:
        raise AssertionError(f"unhandled role strategy: {strategy}")

    roles = torch.where(entity_mask, roles, torch.full_like(roles, int(AgentRole.ROBOT)))
    return roles


def infer_entity_roles_from_speed(future_vel: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
    """Backward-compatible wrapper for the original speed-threshold pseudo roles."""

    return infer_entity_roles(future_vel, entity_mask, strategy="speed_threshold")


def build_scene_batch_from_tensors(
    robot_history: torch.Tensor,
    robot_future: torch.Tensor,
    neighbor_history: Optional[torch.Tensor] = None,
    neighbor_future: Optional[torch.Tensor] = None,
    neighbor_mask: Optional[torch.Tensor] = None,
    robot_goal: Optional[torch.Tensor] = None,
    dt: float = 0.4,
    role_strategy: str = "speed_threshold",
    slow_speed: float = 0.15,
    fast_speed: float = 1.8,
    slow_quantile: float = 0.33,
    fast_quantile: float = 0.67,
) -> HeterogeneousSceneBatch:
    """Create a SAGE batch from tensor inputs.

    robot_history:   (B, H, F)
    robot_future:    (B, T, 2), future robot velocities/plans
    neighbor_history:(B, M, H, F), optional entity histories
    neighbor_future: (B, M, T, 2), optional entity future velocities/predictions
    """

    if robot_history.ndim != 3:
        raise ValueError("robot_history must be (B,H,F)")
    if robot_future.ndim != 3 or robot_future.shape[-1] != 2:
        raise ValueError("robot_future must be (B,T,2)")

    b, h, f = robot_history.shape
    t = robot_future.shape[1]
    device = robot_history.device
    dtype = robot_history.dtype

    if neighbor_history is None:
        neighbor_history = torch.zeros(b, 0, h, f, device=device, dtype=dtype)
    if neighbor_future is None:
        neighbor_future = torch.zeros(b, neighbor_history.shape[1], t, 2, device=device, dtype=robot_future.dtype)
    if neighbor_mask is None:
        neighbor_mask = torch.ones(b, neighbor_history.shape[1], device=device, dtype=torch.bool)

    history = torch.cat([robot_history[:, None], neighbor_history], dim=1)
    future = torch.cat([robot_future[:, None], neighbor_future], dim=1)
    current_pos = history[:, :, -1, :2]
    goals = torch.zeros_like(current_pos)
    if robot_goal is None:
        robot_goal = robot_history[:, -1, :2] + robot_future.sum(dim=1) * dt
    elif robot_goal.ndim == 3:
        robot_goal = robot_goal[:, 0]
    goals[:, 0] = robot_goal.to(device=device, dtype=current_pos.dtype)

    agent_mask = torch.cat(
        [
            torch.ones(b, 1, device=device, dtype=torch.bool),
            neighbor_mask.to(device=device).bool(),
        ],
        dim=1,
    )
    robot_mask = torch.zeros_like(agent_mask)
    robot_mask[:, 0] = True
    entity_mask = agent_mask & ~robot_mask
    goal_mask = robot_mask & agent_mask

    roles = infer_entity_roles(
        future,
        entity_mask,
        strategy=role_strategy,
        slow_speed=slow_speed,
        fast_speed=fast_speed,
        slow_quantile=slow_quantile,
        fast_quantile=fast_quantile,
    )
    roles[:, 0] = int(AgentRole.ROBOT)

    batch = HeterogeneousSceneBatch(
        history=history,
        future=future,
        current_pos=current_pos,
        goals=goals,
        goal_mask=goal_mask,
        roles=roles,
        agent_mask=agent_mask,
        robot_mask=robot_mask,
        entity_mask=entity_mask,
        loss_mask=agent_mask,
        dt=dt,
    )
    batch.validate()
    return batch


def build_scene_batch_from_mid_batch(mid_batch, dt: float = 0.4) -> HeterogeneousSceneBatch:
    """Best-effort adapter for the original MID/Trajectron batch tuple.

    The original MID data loader is target-node centered. SAGE treats that
    target as the controllable robot and its available neighbors as entities.
    If neighbor futures are absent, the adapter fills them with zeros so the
    training code remains executable; real heterogeneous data should provide
    neighbor future velocities.
    """

    (
        _first_history_index,
        x_t,
        y_t,
        _x_st_t,
        _y_st_t,
        neighbors_data_st,
        _neighbors_edge_value,
        _robot_traj_st_t,
        _map,
    ) = mid_batch

    robot_history = x_t[..., :2] if x_t.shape[-1] > 2 else x_t
    robot_future = y_t[..., :2]
    b, h, f = robot_history.shape
    t = robot_future.shape[1]
    device = robot_history.device
    dtype = robot_history.dtype

    neighbor_histories = []
    if isinstance(neighbors_data_st, dict):
        iterable = neighbors_data_st.values()
    else:
        iterable = neighbors_data_st if neighbors_data_st is not None else []

    for item in iterable:
        if torch.is_tensor(item) and item.ndim >= 3:
            tensor = item
            if tensor.ndim == 3:
                tensor = tensor[:, None, :, :]
            neighbor_histories.append(tensor[..., :f])

    if neighbor_histories:
        neighbor_history = torch.cat(neighbor_histories, dim=1).to(device=device, dtype=dtype)
        if neighbor_history.shape[2] != h:
            neighbor_history = neighbor_history[:, :, -h:, :]
        neighbor_mask = torch.ones(b, neighbor_history.shape[1], device=device, dtype=torch.bool)
    else:
        neighbor_history = torch.zeros(b, 0, h, f, device=device, dtype=dtype)
        neighbor_mask = torch.zeros(b, 0, device=device, dtype=torch.bool)

    neighbor_future = torch.zeros(b, neighbor_history.shape[1], t, 2, device=device, dtype=robot_future.dtype)
    return build_scene_batch_from_tensors(
        robot_history=robot_history,
        robot_future=robot_future,
        neighbor_history=neighbor_history,
        neighbor_future=neighbor_future,
        neighbor_mask=neighbor_mask,
        dt=dt,
    )
