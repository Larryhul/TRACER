from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch

from sage.data.heterogeneous_scene import AgentRole, HeterogeneousSceneBatch, integrate_velocities


@dataclass
class ComfortMargin:
    sigma_long: float
    sigma_lat: float


@dataclass
class SafetySocialGuidanceConfig:
    dt: float = 0.4
    collision_margin: float = 0.45
    collision_sigma: float = 0.8
    max_speed: float = 2.5
    max_lateral_speed: float = 0.15
    max_angular_speed: float = 1.2
    w_collision: float = 2.0
    w_robot_robot_collision: float = 1.0
    w_social: float = 1.0
    w_kinematic: float = 0.1
    w_goal: float = 0.0
    goal_tolerance: float = 0.5
    guidance_grad_clip_norm: float = 1.0
    guidance_step_clip_norm: float = 0.05
    guidance_velocity_clip_factor: float = 1.25
    project_guided_velocity: bool = True
    front_only: bool = True
    use_geometry_margin: bool = False
    eps: float = 1e-6
    comfort: Dict[int, ComfortMargin] = field(
        default_factory=lambda: {
            int(AgentRole.PHA): ComfortMargin(2.0, 1.0),
            int(AgentRole.HNAA): ComfortMargin(2.6, 1.2),
            int(AgentRole.SSAA): ComfortMargin(1.2, 0.7),
            int(AgentRole.UNKNOWN_ENTITY): ComfortMargin(1.5, 0.8),
        }
    )


class SafetySocialGuidance:
    """Phase 3: safety-social-goal energy applied to robot trajectory variables."""

    def __init__(self, config: SafetySocialGuidanceConfig) -> None:
        self.config = config

    def _comfort_tensors(self, roles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default = self.config.comfort[int(AgentRole.UNKNOWN_ENTITY)]
        sigma_long = torch.full_like(roles, default.sigma_long, dtype=torch.float32)
        sigma_lat = torch.full_like(roles, default.sigma_lat, dtype=torch.float32)
        for role_id, margin in self.config.comfort.items():
            role_mask = roles == int(role_id)
            sigma_long = torch.where(role_mask, torch.full_like(sigma_long, margin.sigma_long), sigma_long)
            sigma_lat = torch.where(role_mask, torch.full_like(sigma_lat, margin.sigma_lat), sigma_lat)
        return sigma_long.to(roles.device), sigma_lat.to(roles.device)

    def _collision_margins(self, batch: HeterogeneousSceneBatch, dtype: torch.dtype) -> torch.Tensor | float:
        if not self.config.use_geometry_margin or batch.geometry is None:
            return self.config.collision_margin
        geometry = batch.geometry.to(dtype=dtype).clamp_min(0.0)
        radii = 0.5 * torch.sqrt((geometry**2).sum(dim=-1) + self.config.eps)
        pair_radii = radii[:, :, None] + radii[:, None, :]
        return pair_radii[:, :, :, None] + self.config.collision_margin

    def _collision_energy(self, dist: torch.Tensor, margin: torch.Tensor | float) -> torch.Tensor:
        smooth = torch.exp(-(dist**2) / (2.0 * self.config.collision_sigma**2))
        barrier = torch.relu(margin - dist) ** 2
        return smooth + barrier

    def _kinematic_terms(self, x0_vel: torch.Tensor, batch: HeterogeneousSceneBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        speed = torch.linalg.norm(x0_vel, dim=-1)
        heading = torch.atan2(x0_vel[..., 1], x0_vel[..., 0] + self.config.eps)
        if batch.history.shape[2] >= 2:
            prev_vel = (batch.history[:, :, -1, :2] - batch.history[:, :, -2, :2]) / float(batch.dt)
        else:
            prev_vel = x0_vel[:, :, 0]
        prev_heading = torch.atan2(prev_vel[..., 1], prev_vel[..., 0] + self.config.eps)
        heading_ref = torch.cat([prev_heading[:, :, None], heading[:, :, :-1]], dim=2)
        lateral = -torch.sin(heading_ref) * x0_vel[..., 0] + torch.cos(heading_ref) * x0_vel[..., 1]
        if x0_vel.shape[2] > 1:
            delta = torch.atan2(torch.sin(heading[:, :, 1:] - heading[:, :, :-1]), torch.cos(heading[:, :, 1:] - heading[:, :, :-1]))
            angular = delta / float(batch.dt)
            angular = torch.cat([torch.zeros_like(angular[:, :, :1]), angular], dim=2)
        else:
            angular = torch.zeros_like(speed)
        return speed, lateral, angular

    def energy_components(self, x0_vel: torch.Tensor, batch: HeterogeneousSceneBatch) -> Dict[str, torch.Tensor]:
        pos = integrate_velocities(x0_vel, batch.current_pos, batch.dt)
        rel = pos[:, None, :, :, :] - pos[:, :, None, :, :]
        dist = torch.sqrt((rel**2).sum(dim=-1) + self.config.eps)

        robot_entity_mask = (
            batch.robot_mask[:, :, None]
            & batch.entity_mask[:, None, :]
            & batch.agent_mask[:, :, None]
            & batch.agent_mask[:, None, :]
        )
        robot_robot_mask = (
            batch.robot_mask[:, :, None]
            & batch.robot_mask[:, None, :]
            & batch.agent_mask[:, :, None]
            & batch.agent_mask[:, None, :]
        )
        if batch.num_agents > 0:
            upper = torch.triu(
                torch.ones(batch.num_agents, batch.num_agents, dtype=torch.bool, device=batch.robot_mask.device),
                diagonal=1,
            ).view(1, batch.num_agents, batch.num_agents)
            robot_robot_mask = robot_robot_mask & upper
        pair_mask = robot_entity_mask | robot_robot_mask
        pair_mask_f = pair_mask[:, :, :, None].to(x0_vel.dtype)

        robot_vel = x0_vel[:, :, None, :, :]
        heading = torch.atan2(robot_vel[..., 1], robot_vel[..., 0] + self.config.eps)
        cos_t = torch.cos(heading)
        sin_t = torch.sin(heading)
        dx = rel[..., 0]
        dy = rel[..., 1]
        long = cos_t * dx + sin_t * dy
        lat = -sin_t * dx + cos_t * dy

        sigma_long, sigma_lat = self._comfort_tensors(batch.roles)
        sl = sigma_long[:, None, :, None].to(x0_vel.device).clamp_min(self.config.eps)
        st = sigma_lat[:, None, :, None].to(x0_vel.device).clamp_min(self.config.eps)
        social = torch.exp(-((long / sl) ** 2 + (lat / st) ** 2))
        if self.config.front_only:
            social = social * (long >= 0).to(x0_vel.dtype)
        social = social * robot_entity_mask[:, :, :, None].to(x0_vel.dtype)

        collision_margin = self._collision_margins(batch, x0_vel.dtype)
        collision = self._collision_energy(dist, collision_margin) * pair_mask_f

        speed, lateral, angular = self._kinematic_terms(x0_vel, batch)
        speed_violation = torch.relu(speed - self.config.max_speed) ** 2
        lateral_violation = torch.relu(torch.abs(lateral) - self.config.max_lateral_speed) ** 2
        angular_violation = torch.relu(torch.abs(angular) - self.config.max_angular_speed) ** 2
        robot_mask_t = batch.robot_mask[:, :, None].to(x0_vel.dtype)
        kin = (speed_violation + lateral_violation + angular_violation) * robot_mask_t

        goal_mask = batch.effective_goal_mask().to(x0_vel.dtype)
        goal_error = ((pos[:, :, -1] - batch.effective_goals().to(pos.dtype)) ** 2).sum(dim=-1) * goal_mask

        return {
            "collision": collision.sum(dim=(1, 2, 3)),
            "social": social.sum(dim=(1, 2, 3)),
            "kinematic": kin.sum(dim=(1, 2)),
            "goal": goal_error.sum(dim=1),
            "robot_entity_collision": (collision * robot_entity_mask[:, :, :, None].to(x0_vel.dtype)).sum(dim=(1, 2, 3)),
            "robot_robot_collision": (collision * robot_robot_mask[:, :, :, None].to(x0_vel.dtype)).sum(dim=(1, 2, 3)),
            "speed_violation": (speed_violation * robot_mask_t).sum(dim=(1, 2)),
            "lateral_violation": (lateral_violation * robot_mask_t).sum(dim=(1, 2)),
            "angular_violation": (angular_violation * robot_mask_t).sum(dim=(1, 2)),
        }

    def energy(self, x0_vel: torch.Tensor, batch: HeterogeneousSceneBatch) -> torch.Tensor:
        terms = self.energy_components(x0_vel, batch)
        collision = (
            terms["robot_entity_collision"]
            + self.config.w_robot_robot_collision * terms["robot_robot_collision"]
        )
        return (
            self.config.w_collision * collision
            + self.config.w_social * terms["social"]
            + self.config.w_kinematic * terms["kinematic"]
            + self.config.w_goal * terms["goal"]
        )

    def _clip_robot_trajectory_norm(self, tensor: torch.Tensor, batch: HeterogeneousSceneBatch, max_norm: float) -> torch.Tensor:
        if max_norm <= 0.0:
            return tensor
        robot_mask = batch.robot_mask[:, :, None, None].to(tensor.dtype)
        flat = tensor.flatten(start_dim=2)
        norm = torch.linalg.vector_norm(flat, dim=-1).view(tensor.shape[0], tensor.shape[1], 1, 1)
        scale = (float(max_norm) / norm.clamp_min(self.config.eps)).clamp(max=1.0)
        return tensor * torch.where(robot_mask.bool(), scale, torch.ones_like(scale))

    def project_robot_velocities(self, future_vel: torch.Tensor, batch: HeterogeneousSceneBatch) -> torch.Tensor:
        if not self.config.project_guided_velocity:
            return future_vel
        limit = float(self.config.max_speed) * float(self.config.guidance_velocity_clip_factor)
        if limit <= 0.0:
            return future_vel
        vel = torch.nan_to_num(future_vel, nan=0.0, posinf=limit, neginf=-limit)
        speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
        scale = (limit / speed.clamp_min(self.config.eps)).clamp(max=1.0)
        projected = vel * scale
        robot_mask = batch.robot_mask[:, :, None, None].to(torch.bool)
        return torch.where(robot_mask, projected, vel)

    def robot_only_gradient(self, x0_vel: torch.Tensor, batch: HeterogeneousSceneBatch) -> torch.Tensor:
        with torch.enable_grad():
            var = x0_vel.detach().requires_grad_(True)
            energy = self.energy(var, batch)
            grad = torch.autograd.grad(energy.sum(), var, create_graph=False)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad = grad * batch.robot_mask[:, :, None, None].to(grad.dtype)
        return self._clip_robot_trajectory_norm(grad, batch, self.config.guidance_grad_clip_norm)

    def make_guidance_fn(self, batch: HeterogeneousSceneBatch):
        def _guide(x0_vel: torch.Tensor) -> torch.Tensor:
            return self.robot_only_gradient(x0_vel, batch)

        _guide.guidance_step_clip_norm = float(self.config.guidance_step_clip_norm)
        return _guide

    def make_projection_fn(self, batch: HeterogeneousSceneBatch):
        def _project(x0_vel: torch.Tensor) -> torch.Tensor:
            return self.project_robot_velocities(x0_vel, batch)

        return _project

    def diagnostics(self, x0_vel: torch.Tensor, batch: HeterogeneousSceneBatch) -> Dict[str, float]:
        with torch.no_grad():
            pos = integrate_velocities(x0_vel, batch.current_pos, batch.dt)
            rel = pos[:, None, :, :, :] - pos[:, :, None, :, :]
            dist = torch.sqrt((rel**2).sum(dim=-1) + self.config.eps)
            robot_entity_mask = batch.robot_mask[:, :, None] & batch.entity_mask[:, None, :]
            robot_robot_mask = batch.robot_mask[:, :, None] & batch.robot_mask[:, None, :]
            if batch.num_agents > 0:
                upper = torch.triu(
                    torch.ones(batch.num_agents, batch.num_agents, dtype=torch.bool, device=batch.robot_mask.device),
                    diagonal=1,
                ).view(1, batch.num_agents, batch.num_agents)
                robot_robot_mask = robot_robot_mask & upper
            pair_mask = robot_entity_mask | robot_robot_mask
            masked = dist[pair_mask[:, :, :, None].expand_as(dist)]
            re_masked = dist[robot_entity_mask[:, :, :, None].expand_as(dist)]
            rr_masked = dist[robot_robot_mask[:, :, :, None].expand_as(dist)]
            min_dist = float(masked.min().item()) if masked.numel() else float("inf")
            robot_entity_min = float(re_masked.min().item()) if re_masked.numel() else float("inf")
            robot_robot_min = float(rr_masked.min().item()) if rr_masked.numel() else float("inf")
            collision_margin = self._collision_margins(batch, x0_vel.dtype)
            if torch.is_tensor(collision_margin):
                margin_values = collision_margin.expand_as(dist)[pair_mask[:, :, :, None].expand_as(dist)]
                collision_rate = float((masked < margin_values).float().mean().item()) if masked.numel() else 0.0
            else:
                collision_rate = float((masked < collision_margin).float().mean().item()) if masked.numel() else 0.0
            terms = self.energy_components(x0_vel, batch)
            energy = float(self.energy(x0_vel, batch).mean().item())
            goal_mask = batch.effective_goal_mask()
            if goal_mask.any():
                goal_error = ((pos[:, :, -1] - batch.effective_goals()) ** 2).sum(dim=-1)
                goal_energy = float((goal_error * goal_mask.to(goal_error.dtype)).sum().item() / goal_mask.sum().clamp_min(1).item())
            else:
                goal_energy = 0.0
            robot_count = batch.robot_mask.sum().clamp_min(1).item()
        return {
            "energy": energy,
            "collision_energy": float(terms["collision"].mean().item()),
            "social_energy": float(terms["social"].mean().item()),
            "kinematic_energy": float(terms["kinematic"].mean().item()),
            "goal_energy": goal_energy,
            "min_distance": min_dist,
            "robot_entity_min_distance": robot_entity_min,
            "robot_robot_min_distance": robot_robot_min,
            "collision_rate": collision_rate,
            "robot_robot_collision_energy": float(terms["robot_robot_collision"].mean().item()),
            "lateral_violation_energy": float(terms["lateral_violation"].sum().item() / robot_count),
            "angular_violation_energy": float(terms["angular_violation"].sum().item() / robot_count),
        }
