from __future__ import annotations

from dataclasses import dataclass

import torch

from tracer.data.heterogeneous_scene import AgentRole, HeterogeneousSceneBatch


@dataclass
class SocialForceConfig:
    """Deterministic social-force baseline for controlled navigation scenes."""

    desired_speed: float = 1.2
    max_speed: float = 2.5
    relaxation_time: float = 0.5
    repulsion_gain: float = 1.2
    robot_robot_gain: float = 1.6
    repulsion_sigma: float = 0.8
    interaction_radius: float = 4.0
    max_entity_speed: float = 3.0
    robot_radius: float = 0.9
    pha_radius: float = 1.2
    hnaa_radius: float = 1.7
    ssaa_radius: float = 1.2
    unknown_radius: float = 1.2


def _clip_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0:
        return vec
    norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
    scale = (float(max_norm) / norm.clamp_min(1e-6)).clamp(max=1.0)
    return vec * scale


def _last_observed_velocity(batch: HeterogeneousSceneBatch) -> torch.Tensor:
    xy = batch.history[..., :2]
    if xy.shape[2] < 2:
        return torch.zeros_like(batch.current_pos)
    vel = (xy[:, :, -1] - xy[:, :, -2]) / float(batch.dt)
    return torch.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)


def _role_radii(roles: torch.Tensor, cfg: SocialForceConfig) -> torch.Tensor:
    radii = torch.full(roles.shape, float(cfg.unknown_radius), dtype=torch.float32, device=roles.device)
    radii = torch.where(roles == int(AgentRole.ROBOT), torch.full_like(radii, float(cfg.robot_radius)), radii)
    radii = torch.where(roles == int(AgentRole.PHA), torch.full_like(radii, float(cfg.pha_radius)), radii)
    radii = torch.where(roles == int(AgentRole.HNAA), torch.full_like(radii, float(cfg.hnaa_radius)), radii)
    radii = torch.where(roles == int(AgentRole.SSAA), torch.full_like(radii, float(cfg.ssaa_radius)), radii)
    return radii


def social_force_baseline(batch: HeterogeneousSceneBatch, cfg: SocialForceConfig | None = None) -> torch.Tensor:
    """Generate future velocities with a classical social-force controller.

    Entities follow constant-velocity extrapolation from the observed history.
    Robots move toward their local goal/waypoint while being repelled by entities
    and other robots. The method is deterministic and uses no future ground truth
    except the goal field already exposed to all proxy-navigation methods.
    """

    cfg = cfg or SocialForceConfig()
    device = batch.current_pos.device
    dtype = batch.current_pos.dtype
    b, a, horizon, _ = batch.future.shape
    dt = float(batch.dt)

    pos = batch.current_pos.clone()
    last_vel = _last_observed_velocity(batch).to(device=device, dtype=dtype)
    entity_vel = _clip_norm(last_vel, cfg.max_entity_speed)
    robot_vel = _clip_norm(last_vel, cfg.max_speed)
    goals = batch.effective_goals().to(device=device, dtype=dtype)
    goal_mask = batch.effective_goal_mask()
    agent_mask = batch.agent_mask.bool()
    robot_mask = batch.robot_mask.bool() & agent_mask
    entity_mask = batch.entity_mask.bool() & agent_mask
    role_radii = _role_radii(batch.roles, cfg).to(dtype=dtype)

    eye = torch.eye(a, dtype=torch.bool, device=device).view(1, a, a)
    outputs = []
    for step in range(horizon):
        remaining = max((horizon - step) * dt, dt)
        to_goal = goals - pos
        direct_goal_vel = _clip_norm(to_goal / remaining, cfg.max_speed)
        history_goal_vel = _clip_norm(last_vel, cfg.max_speed)
        desired_vel = torch.where(goal_mask[..., None], direct_goal_vel, history_goal_vel)

        rel = pos[:, :, None, :] - pos[:, None, :, :]
        dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
        direction = rel / dist[..., None]
        receiver_robot = robot_mask[:, :, None]
        source_active = agent_mask[:, None, :]
        pair_mask = receiver_robot & source_active & (~eye) & (dist < float(cfg.interaction_radius))
        margin = role_radii[:, None, :] + float(cfg.robot_radius)
        gain = torch.full_like(dist, float(cfg.repulsion_gain))
        gain = torch.where(batch.robot_mask[:, None, :].bool(), torch.full_like(gain, float(cfg.robot_robot_gain)), gain)
        sigma = max(float(cfg.repulsion_sigma), 1e-6)
        mag = gain * torch.exp((margin - dist) / sigma)
        force = (direction * mag[..., None] * pair_mask[..., None].to(dtype)).sum(dim=2)

        accel = (desired_vel - robot_vel) / max(float(cfg.relaxation_time), 1e-6) + force
        next_robot_vel = _clip_norm(robot_vel + accel * dt, cfg.max_speed)
        full_vel = torch.where(robot_mask[..., None], next_robot_vel, entity_vel)
        full_vel = torch.where(entity_mask[..., None], entity_vel, full_vel)
        full_vel = full_vel * agent_mask[..., None].to(dtype)
        outputs.append(full_vel)
        pos = pos + full_vel * dt
        robot_vel = torch.where(robot_mask[..., None], next_robot_vel, robot_vel)

    return torch.stack(outputs, dim=2)
