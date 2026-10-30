from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict

import torch


class AgentRole(IntEnum):
    """Semantic roles used by SAGE.

    The role names follow the paper vocabulary:
    robots are controllable agents, entities are non-robot scene participants,
    and neighbors are entities inside a robot-specific perception region.
    """

    ROBOT = 0
    PHA = 1
    HNAA = 2
    SSAA = 3
    UNKNOWN_ENTITY = 4


ROLE_TO_ID: Dict[str, int] = {
    "robot": int(AgentRole.ROBOT),
    "pha": int(AgentRole.PHA),
    "pedestrian": int(AgentRole.PHA),
    "hnaa": int(AgentRole.HNAA),
    "vehicle": int(AgentRole.HNAA),
    "ssaa": int(AgentRole.SSAA),
    "static": int(AgentRole.SSAA),
    "unknown_entity": int(AgentRole.UNKNOWN_ENTITY),
    "entity": int(AgentRole.UNKNOWN_ENTITY),
}

ID_TO_ROLE: Dict[int, str] = {int(role): role.name.lower() for role in AgentRole}


@dataclass
class HeterogeneousSceneBatch:
    """Batched heterogeneous navigation scene.

    history:     (B, A, H, F) observed states. The first two channels must be xy.
    future:      (B, A, T, 2) future velocities used as diffusion targets.
    current_pos: (B, A, 2) current xy position at the planning instant.
    roles:       (B, A) role ids from AgentRole.
    agent_mask:  (B, A) active-agent mask.
    robot_mask:  (B, A) true for controllable robots.
    entity_mask: (B, A) true for non-robot entities.
    loss_mask:   (B, A) agents with valid future supervision.
    geometry:    optional (B, A, 2) physical length/width metadata.
    goals:       optional (B, A, 2) per-robot local goal/waypoint positions.
    goal_mask:   optional (B, A) robots with valid goal supervision.
    robot_actions: optional (B, A) high-level action/mode codes for controllable robots.
    """

    history: torch.Tensor
    future: torch.Tensor
    current_pos: torch.Tensor
    roles: torch.Tensor
    agent_mask: torch.Tensor
    robot_mask: torch.Tensor
    entity_mask: torch.Tensor
    loss_mask: torch.Tensor | None = None
    geometry: torch.Tensor | None = None
    goals: torch.Tensor | None = None
    goal_mask: torch.Tensor | None = None
    robot_actions: torch.Tensor | None = None
    dt: float = 0.4

    def to(self, device: torch.device | str) -> "HeterogeneousSceneBatch":
        return HeterogeneousSceneBatch(
            history=self.history.to(device),
            future=self.future.to(device),
            current_pos=self.current_pos.to(device),
            roles=self.roles.to(device),
            agent_mask=self.agent_mask.to(device),
            robot_mask=self.robot_mask.to(device),
            entity_mask=self.entity_mask.to(device),
            loss_mask=self.loss_mask.to(device) if self.loss_mask is not None else None,
            geometry=self.geometry.to(device) if self.geometry is not None else None,
            goals=self.goals.to(device) if self.goals is not None else None,
            goal_mask=self.goal_mask.to(device) if self.goal_mask is not None else None,
            robot_actions=self.robot_actions.to(device) if self.robot_actions is not None else None,
            dt=self.dt,
        )

    @property
    def batch_size(self) -> int:
        return int(self.history.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.history.shape[1])

    @property
    def horizon(self) -> int:
        return int(self.future.shape[2])

    @property
    def num_robots(self) -> int:
        return int((self.robot_mask & self.agent_mask).sum().item())

    @property
    def num_entities(self) -> int:
        return int((self.entity_mask & self.agent_mask).sum().item())

    def validate(self) -> None:
        if self.history.ndim != 4:
            raise ValueError(f"history must be (B,A,H,F), got {tuple(self.history.shape)}")
        if self.future.ndim != 4 or self.future.shape[-1] != 2:
            raise ValueError(f"future must be (B,A,T,2), got {tuple(self.future.shape)}")
        b, a = self.history.shape[:2]
        expected = (b, a)
        if tuple(self.current_pos.shape) != (b, a, 2):
            raise ValueError(f"current_pos must be (B,A,2), got {tuple(self.current_pos.shape)}")
        for name, tensor in (
            ("roles", self.roles),
            ("agent_mask", self.agent_mask),
            ("robot_mask", self.robot_mask),
            ("entity_mask", self.entity_mask),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must be {expected}, got {tuple(tensor.shape)}")
        if self.loss_mask is not None and tuple(self.loss_mask.shape) != expected:
            raise ValueError(f"loss_mask must be {expected}, got {tuple(self.loss_mask.shape)}")
        if self.geometry is not None and tuple(self.geometry.shape) != (b, a, 2):
            raise ValueError(f"geometry must be {(b, a, 2)}, got {tuple(self.geometry.shape)}")
        if self.goals is not None and tuple(self.goals.shape) != (b, a, 2):
            raise ValueError(f"goals must be {(b, a, 2)}, got {tuple(self.goals.shape)}")
        if self.goal_mask is not None and tuple(self.goal_mask.shape) != expected:
            raise ValueError(f"goal_mask must be {expected}, got {tuple(self.goal_mask.shape)}")
        if self.robot_actions is not None and tuple(self.robot_actions.shape) != expected:
            raise ValueError(f"robot_actions must be {expected}, got {tuple(self.robot_actions.shape)}")

    def supervision_mask(self) -> torch.Tensor:
        return self.agent_mask if self.loss_mask is None else self.loss_mask

    def effective_goal_mask(self) -> torch.Tensor:
        mask = self.goal_mask if self.goal_mask is not None else self.robot_mask
        return mask.bool() & self.robot_mask.bool() & self.agent_mask.bool()

    def effective_goals(self) -> torch.Tensor:
        if self.goals is not None:
            return self.goals
        return endpoint_from_velocities(self.future, self.current_pos, self.dt)

    def goal_displacement(self) -> torch.Tensor:
        disp = self.effective_goals() - self.current_pos
        return disp * self.effective_goal_mask()[..., None].to(disp.dtype)


def integrate_velocities(future_vel: torch.Tensor, current_pos: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrate future velocities into future positions."""

    if future_vel.ndim != 4:
        raise ValueError(f"future_vel must be (B,A,T,2), got {tuple(future_vel.shape)}")
    return current_pos[:, :, None, :] + torch.cumsum(future_vel * dt, dim=2)


def endpoint_from_velocities(future_vel: torch.Tensor, current_pos: torch.Tensor, dt: float) -> torch.Tensor:
    """Return the final endpoint implied by a future velocity sequence."""

    if future_vel.ndim != 4:
        raise ValueError(f"future_vel must be (B,A,T,2), got {tuple(future_vel.shape)}")
    return current_pos + future_vel.sum(dim=2) * dt
