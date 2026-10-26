from __future__ import annotations

from typing import Dict

import torch

from sage.data.heterogeneous_scene import AgentRole, HeterogeneousSceneBatch, integrate_velocities


ROLE_METRIC_NAMES = {
    int(AgentRole.PHA): "pha",
    int(AgentRole.HNAA): "hnaa",
    int(AgentRole.SSAA): "ssaa",
    int(AgentRole.UNKNOWN_ENTITY): "unknown_entity",
}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(values.dtype)
    return (values * m).sum() / m.sum().clamp_min(1.0)


def ade(pred_pos: torch.Tensor, gt_pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = torch.linalg.norm(pred_pos - gt_pos, dim=-1)
    m = mask[:, :, None].to(err.dtype)
    return (err * m).sum() / (m.sum().clamp_min(1.0) * err.shape[-1])


def fde(pred_pos: torch.Tensor, gt_pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = torch.linalg.norm(pred_pos[:, :, -1] - gt_pos[:, :, -1], dim=-1)
    m = mask.to(err.dtype)
    return (err * m).sum() / m.sum().clamp_min(1.0)


def goal_reaching_metrics(
    pred_pos: torch.Tensor,
    batch: HeterogeneousSceneBatch,
    goal_tolerance: float = 0.5,
) -> Dict[str, float]:
    goal_mask = batch.effective_goal_mask()
    if not goal_mask.any():
        return {
            "initial_goal_distance": 0.0,
            "goal_fde": 0.0,
            "goal_progress": 0.0,
            "normalized_goal_progress": 0.0,
            "goal_success_rate": 0.0,
        }
    goals = batch.effective_goals().to(pred_pos.dtype)
    initial = torch.linalg.norm(batch.current_pos.to(pred_pos.dtype) - goals, dim=-1)
    err = torch.linalg.norm(pred_pos[:, :, -1] - goals, dim=-1)
    progress = initial - err
    return {
        "initial_goal_distance": float(_masked_mean(initial, goal_mask).item()),
        "goal_fde": float(_masked_mean(err, goal_mask).item()),
        "goal_progress": float(_masked_mean(progress, goal_mask).item()),
        "normalized_goal_progress": float(
            _masked_mean(progress / initial.clamp_min(1e-6), goal_mask).item()
        ),
        "goal_success_rate": float(_masked_mean((err <= goal_tolerance).to(err.dtype), goal_mask).item()),
    }


def _role_masks(batch: HeterogeneousSceneBatch) -> dict[str, torch.Tensor]:
    masks = {}
    for role_id, name in ROLE_METRIC_NAMES.items():
        masks[name] = batch.entity_mask & (batch.roles == role_id)
    return masks


def role_counts(batch: HeterogeneousSceneBatch) -> Dict[str, float]:
    return {f"role_{name}_count": float(mask.sum().item()) for name, mask in _role_masks(batch).items()}


def role_navigation_metrics(pred_pos: torch.Tensor, gt_pos: torch.Tensor, batch: HeterogeneousSceneBatch) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name, mask in _role_masks(batch).items():
        count = int(mask.sum().item())
        metrics[f"role_{name}_count"] = float(count) / max(batch.batch_size, 1)
        if count == 0:
            continue
        metrics[f"role_{name}_ade"] = float(ade(pred_pos, gt_pos, mask).item())
        metrics[f"role_{name}_fde"] = float(fde(pred_pos, gt_pos, mask).item())
    return metrics


def navigation_metrics(
    pred_vel: torch.Tensor,
    batch: HeterogeneousSceneBatch,
    include_role_metrics: bool = False,
    goal_tolerance: float = 0.5,
) -> Dict[str, float]:
    gt_pos = integrate_velocities(batch.future, batch.current_pos, batch.dt)
    pred_pos = integrate_velocities(pred_vel, batch.current_pos, batch.dt)
    metrics = {
        "robot_ade": float(ade(pred_pos, gt_pos, batch.robot_mask).item()),
        "robot_fde": float(fde(pred_pos, gt_pos, batch.robot_mask).item()),
        "entity_ade": float(ade(pred_pos, gt_pos, batch.entity_mask).item()),
        "entity_fde": float(fde(pred_pos, gt_pos, batch.entity_mask).item()),
    }
    metrics.update(goal_reaching_metrics(pred_pos, batch, goal_tolerance=goal_tolerance))
    if include_role_metrics:
        metrics.update(role_navigation_metrics(pred_pos, gt_pos, batch))
    return metrics


def best_of_navigation_metrics(
    pred_vel_samples: torch.Tensor,
    batch: HeterogeneousSceneBatch,
    include_role_metrics: bool = False,
    goal_tolerance: float = 0.5,
) -> Dict[str, float]:
    """Best-of-k ADE/FDE for SAGE samples.

    pred_vel_samples: (K, B, A, T, 2)
    """

    if pred_vel_samples.ndim != 5:
        raise ValueError(f"pred_vel_samples must be (K,B,A,T,2), got {tuple(pred_vel_samples.shape)}")

    gt_pos = integrate_velocities(batch.future, batch.current_pos, batch.dt)
    pred_pos = batch.current_pos[None, :, :, None, :] + torch.cumsum(pred_vel_samples * batch.dt, dim=3)

    ade_err = torch.linalg.norm(pred_pos - gt_pos[None], dim=-1).mean(dim=-1)
    fde_err = torch.linalg.norm(pred_pos[:, :, :, -1] - gt_pos[None, :, :, -1], dim=-1)
    best_ade = ade_err.min(dim=0).values
    best_fde = fde_err.min(dim=0).values

    metrics = {
        "robot_ade": float(_masked_mean(best_ade, batch.robot_mask).item()),
        "robot_fde": float(_masked_mean(best_fde, batch.robot_mask).item()),
        "entity_ade": float(_masked_mean(best_ade, batch.entity_mask).item()),
        "entity_fde": float(_masked_mean(best_fde, batch.entity_mask).item()),
    }

    goal_mask = batch.effective_goal_mask()
    if goal_mask.any():
        initial_goal = torch.linalg.norm(
            batch.current_pos - batch.effective_goals().to(batch.current_pos.dtype),
            dim=-1,
        )
        goal_err = torch.linalg.norm(
            pred_pos[:, :, :, -1] - batch.effective_goals()[None].to(pred_pos.dtype),
            dim=-1,
        )
        best_goal = goal_err.min(dim=0).values
        progress = initial_goal - best_goal
        metrics["initial_goal_distance"] = float(
            _masked_mean(initial_goal, goal_mask).item()
        )
        metrics["goal_fde"] = float(_masked_mean(best_goal, goal_mask).item())
        metrics["goal_progress"] = float(_masked_mean(progress, goal_mask).item())
        metrics["normalized_goal_progress"] = float(
            _masked_mean(progress / initial_goal.clamp_min(1e-6), goal_mask).item()
        )
        metrics["goal_success_rate"] = float(
            _masked_mean((best_goal <= goal_tolerance).to(best_goal.dtype), goal_mask).item()
        )
    else:
        metrics["initial_goal_distance"] = 0.0
        metrics["goal_fde"] = 0.0
        metrics["goal_progress"] = 0.0
        metrics["normalized_goal_progress"] = 0.0
        metrics["goal_success_rate"] = 0.0

    if include_role_metrics:
        for name, mask in _role_masks(batch).items():
            count = int(mask.sum().item())
            metrics[f"role_{name}_count"] = float(count) / max(batch.batch_size, 1)
            if count == 0:
                continue
            metrics[f"role_{name}_ade"] = float(_masked_mean(best_ade, mask).item())
            metrics[f"role_{name}_fde"] = float(_masked_mean(best_fde, mask).item())
    return metrics



def _collision_rate(masked_dist: torch.Tensor, pair_time_mask: torch.Tensor, margin: float | torch.Tensor) -> float:
    if masked_dist.numel() == 0:
        return 0.0
    if torch.is_tensor(margin):
        masked_margin = margin.expand_as(pair_time_mask)[pair_time_mask]
        return float((masked_dist < masked_margin).float().mean().item())
    return float((masked_dist < float(margin)).float().mean().item())


def multi_robot_metrics(
    pred_vel: torch.Tensor,
    batch: HeterogeneousSceneBatch,
    collision_margin: float | torch.Tensor,
    goal_tolerance: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Metrics for samples with more than one controllable robot proxy."""

    pos = integrate_velocities(pred_vel, batch.current_pos, batch.dt)
    robot_mask = batch.robot_mask & batch.agent_mask
    robot_counts = robot_mask.sum(dim=1)
    multi_scene = robot_counts >= 2
    metrics: Dict[str, float] = {
        "num_robot_proxies_active": float(robot_counts.float().mean().item()),
        "multi_robot_scene_rate": float(multi_scene.float().mean().item()),
    }

    goal_mask = batch.effective_goal_mask()
    if goal_mask.any():
        goal_err = torch.linalg.norm(pos[:, :, -1] - batch.effective_goals().to(pos.dtype), dim=-1)
        success = (goal_err <= goal_tolerance).to(goal_err.dtype)
        per_scene_goal_count = goal_mask.sum(dim=1)
        per_scene_success_count = (success * goal_mask.to(success.dtype)).sum(dim=1)
        valid_goal_scene = per_scene_goal_count > 0
        valid_multi_goal_scene = valid_goal_scene & multi_scene
        all_success = (per_scene_success_count == per_scene_goal_count.to(success.dtype)) & valid_goal_scene
        metrics["all_goal_success_rate"] = float(
            all_success.to(goal_err.dtype)[valid_goal_scene].mean().item() if valid_goal_scene.any() else 0.0
        )
        metrics["multi_robot_all_goal_success_rate"] = float(
            all_success.to(goal_err.dtype)[valid_multi_goal_scene].mean().item() if valid_multi_goal_scene.any() else 0.0
        )
    else:
        metrics["all_goal_success_rate"] = 0.0
        metrics["multi_robot_all_goal_success_rate"] = 0.0

    rel = pos[:, :, None, :, :] - pos[:, None, :, :, :]
    dist = torch.sqrt((rel**2).sum(dim=-1) + eps)
    upper = torch.triu(
        torch.ones(batch.num_agents, batch.num_agents, dtype=torch.bool, device=batch.robot_mask.device),
        diagonal=1,
    ).view(1, batch.num_agents, batch.num_agents)
    rr_pair = (
        batch.robot_mask[:, :, None]
        & batch.robot_mask[:, None, :]
        & batch.agent_mask[:, :, None]
        & batch.agent_mask[:, None, :]
        & upper
    )
    re_pair = (
        batch.robot_mask[:, :, None]
        & batch.entity_mask[:, None, :]
        & batch.agent_mask[:, :, None]
        & batch.agent_mask[:, None, :]
    )

    rr_mask_t = rr_pair[:, :, :, None].expand_as(dist)
    re_mask_t = re_pair[:, :, :, None].expand_as(dist)
    rr_dist = dist[rr_mask_t]
    re_dist = dist[re_mask_t]
    metrics["robot_robot_pair_count"] = float(rr_pair.sum().item()) / max(batch.batch_size, 1)
    if rr_dist.numel():
        metrics["robot_robot_min_distance"] = float(rr_dist.min().item())
        metrics["robot_robot_collision_rate"] = _collision_rate(rr_dist, rr_mask_t, collision_margin)
    else:
        metrics["robot_robot_min_distance"] = float("inf")
        metrics["robot_robot_collision_rate"] = 0.0
    if re_dist.numel():
        metrics["robot_entity_min_distance"] = float(re_dist.min().item())
        metrics["robot_entity_collision_rate"] = _collision_rate(re_dist, re_mask_t, collision_margin)
    else:
        metrics["robot_entity_min_distance"] = float("inf")
        metrics["robot_entity_collision_rate"] = 0.0
    return metrics


def role_safety_metrics(
    pred_vel: torch.Tensor,
    batch: HeterogeneousSceneBatch,
    collision_margin: float,
    eps: float = 1e-6,
) -> Dict[str, float]:
    pos = integrate_velocities(pred_vel, batch.current_pos, batch.dt)
    rel = pos[:, :, None, :, :] - pos[:, None, :, :, :]
    dist = torch.sqrt((rel**2).sum(dim=-1) + eps)
    metrics: Dict[str, float] = {}
    for role_id, name in ROLE_METRIC_NAMES.items():
        entity_role_mask = batch.entity_mask & (batch.roles == role_id)
        pair_mask = batch.robot_mask[:, :, None] & entity_role_mask[:, None, :]
        masked = dist[pair_mask[:, :, :, None].expand_as(dist)]
        if masked.numel() == 0:
            continue
        metrics[f"role_{name}_min_distance"] = float(masked.min().item())
        metrics[f"role_{name}_collision_rate"] = float((masked < collision_margin).float().mean().item())
    return metrics


def _margin_value(margin: object, name: str, fallback: float) -> float:
    if isinstance(margin, dict):
        alt_name = "sigma_lon" if name == "sigma_long" else name
        return float(margin.get(name, margin.get(alt_name, fallback)))
    return float(getattr(margin, name, fallback))


def _comfort_tensors(
    roles: torch.Tensor,
    comfort: Dict[int, object],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    default = comfort.get(int(AgentRole.UNKNOWN_ENTITY)) or next(iter(comfort.values()), None)
    default_long = _margin_value(default, "sigma_long", 1.5) if default is not None else 1.5
    default_lat = _margin_value(default, "sigma_lat", 0.8) if default is not None else 0.8
    sigma_long = torch.full(roles.shape, default_long, dtype=torch.float32, device=roles.device)
    sigma_lat = torch.full(roles.shape, default_lat, dtype=torch.float32, device=roles.device)
    for role_id, margin in comfort.items():
        role_mask = roles == int(role_id)
        long_value = _margin_value(margin, "sigma_long", default_long)
        lat_value = _margin_value(margin, "sigma_lat", default_lat)
        sigma_long = torch.where(role_mask, torch.full_like(sigma_long, long_value), sigma_long)
        sigma_lat = torch.where(role_mask, torch.full_like(sigma_lat, lat_value), sigma_lat)
    return sigma_long.clamp_min(eps), sigma_lat.clamp_min(eps)


def social_compliance_metrics(
    pred_vel: torch.Tensor,
    batch: HeterogeneousSceneBatch,
    comfort: Dict[int, object],
    max_speed: float,
    personal_space_margin: float = 1.2,
    front_only: bool = True,
    eps: float = 1e-6,
    include_role_metrics: bool = False,
) -> Dict[str, float]:
    """Proxy social-compliance metrics used by guidance ablations.

    PIR is an isotropic personal-space intrusion rate. SVR uses the same
    role-conditioned anisotropic comfort ellipse as the guidance term.
    """

    pos = integrate_velocities(pred_vel, batch.current_pos, batch.dt)
    robot_pos = pos[:, :, None, :, :]
    entity_pos = pos[:, None, :, :, :]
    rel = entity_pos - robot_pos
    dist = torch.sqrt((rel**2).sum(dim=-1) + eps)

    pair_mask = (
        batch.robot_mask[:, :, None]
        & batch.entity_mask[:, None, :]
        & batch.agent_mask[:, :, None]
        & batch.agent_mask[:, None, :]
    )
    pair_mask_t = pair_mask[:, :, :, None].expand_as(dist)
    masked_dist = dist[pair_mask_t]

    metrics: Dict[str, float] = {}
    if masked_dist.numel() == 0:
        metrics["pir"] = 0.0
        metrics["svr"] = 0.0
    else:
        metrics["pir"] = float((masked_dist < personal_space_margin).float().mean().item())

        robot_vel = pred_vel[:, :, None, :, :]
        heading = torch.atan2(robot_vel[..., 1], robot_vel[..., 0] + eps)
        cos_t = torch.cos(heading)
        sin_t = torch.sin(heading)
        dx = rel[..., 0]
        dy = rel[..., 1]
        long = cos_t * dx + sin_t * dy
        lat = -sin_t * dx + cos_t * dy

        sigma_long, sigma_lat = _comfort_tensors(batch.roles, comfort, eps=eps)
        sl = sigma_long[:, None, :, None]
        st = sigma_lat[:, None, :, None]
        ellipse = (long / sl) ** 2 + (lat / st) ** 2
        violation = ellipse < 1.0
        if front_only:
            violation = violation & (long >= 0)
        metrics["svr"] = float(violation[pair_mask_t].float().mean().item())

        if include_role_metrics:
            for role_id, name in ROLE_METRIC_NAMES.items():
                entity_role_mask = batch.entity_mask & (batch.roles == role_id)
                role_pair = batch.robot_mask[:, :, None] & entity_role_mask[:, None, :]
                role_pair_t = role_pair[:, :, :, None].expand_as(dist)
                if role_pair_t.any():
                    role_dist = dist[role_pair_t]
                    metrics[f"role_{name}_pir"] = float(
                        (role_dist < personal_space_margin).float().mean().item()
                    )
                    metrics[f"role_{name}_svr"] = float(violation[role_pair_t].float().mean().item())

    speed = torch.linalg.norm(pred_vel, dim=-1)
    robot_time_mask = (batch.robot_mask & batch.agent_mask)[:, :, None].expand_as(speed)
    robot_speed = speed[robot_time_mask]
    if robot_speed.numel() == 0:
        metrics["speed_violation_rate"] = 0.0
        metrics["mean_robot_speed"] = 0.0
        metrics["max_robot_speed_observed"] = 0.0
    else:
        metrics["speed_violation_rate"] = float((robot_speed > max_speed).float().mean().item())
        metrics["mean_robot_speed"] = float(robot_speed.mean().item())
        metrics["max_robot_speed_observed"] = float(robot_speed.max().item())
    return metrics
