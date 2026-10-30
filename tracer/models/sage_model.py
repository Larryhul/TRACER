from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
from torch import nn

from tracer.data.heterogeneous_scene import HeterogeneousSceneBatch, integrate_velocities
from tracer.guidance.safety_social import SafetySocialGuidance, SafetySocialGuidanceConfig
from tracer.models.hgt_encoder import HGTEncoderConfig, HeterogeneousGraphEncoder
from tracer.models.joint_diffusion import DiffusionConfig, JointDiffusion


@dataclass
class SAGEConfig:
    encoder: HGTEncoderConfig = field(default_factory=HGTEncoderConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    guidance: SafetySocialGuidanceConfig = field(default_factory=SafetySocialGuidanceConfig)
    guidance_scale: float = 0.03
    generation_mode: str = "joint"


class SAGE(nn.Module):
    """SAGE model with joint and decoupled generation variants."""

    def __init__(self, config: SAGEConfig) -> None:
        super().__init__()
        self.config = config
        self.generation_mode = str(config.generation_mode).lower()
        if self.generation_mode not in {"joint", "decoupled"}:
            raise ValueError(f"unknown generation_mode={config.generation_mode!r}")
        self.encoder = HeterogeneousGraphEncoder(config.encoder)
        if self.generation_mode == "decoupled":
            self.robot_diffusion = JointDiffusion(config.diffusion)
            self.entity_diffusion = JointDiffusion(config.diffusion)
            self.diffusion = None
        else:
            self.diffusion = JointDiffusion(config.diffusion)
        self.guidance = SafetySocialGuidance(config.guidance)

    def _shape(self, batch: HeterogeneousSceneBatch) -> tuple[int, int, int, int]:
        return (
            batch.batch_size,
            batch.num_agents,
            self.config.diffusion.horizon,
            self.config.diffusion.point_dim,
        )

    def _context(self, batch: HeterogeneousSceneBatch) -> torch.Tensor:
        node_context, graph_context = self.encoder(batch)
        denoiser_type = str(getattr(self.config.diffusion, "denoiser_type", "transformer")).lower()
        supports_context_tokens = denoiser_type in {"transformer", "attention"}
        if getattr(self.config.diffusion, "use_graph_context_token", False) and supports_context_tokens:
            return torch.cat([node_context, graph_context[:, None, :]], dim=1)
        return node_context

    def forward(self, batch: HeterogeneousSceneBatch) -> Dict[str, torch.Tensor]:
        node_context = self._context(batch)
        if self.generation_mode == "decoupled":
            supervision = batch.supervision_mask()
            robot_mask = supervision & batch.robot_mask
            entity_mask = supervision & batch.entity_mask
            robot_loss = self.robot_diffusion.loss(batch.future, node_context, robot_mask)
            entity_loss = self.entity_diffusion.loss(batch.future, node_context, entity_mask)
            loss = robot_loss + entity_loss
            return {"loss": loss, "robot_loss": robot_loss, "entity_loss": entity_loss}

        loss = self.diffusion.loss(batch.future, node_context, batch.supervision_mask())
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        batch: HeterogeneousSceneBatch,
        guided: bool = True,
        step_stride: int = 1,
        noise_schedules: tuple[list[torch.Tensor], list[torch.Tensor] | None]
        | list[torch.Tensor]
        | None = None,
    ) -> Dict[str, torch.Tensor]:
        node_context = self._context(batch)
        shape = self._shape(batch)
        if self.generation_mode == "decoupled":
            if noise_schedules is None:
                entity_noise = None
                robot_noise = None
            else:
                if not isinstance(noise_schedules, tuple):
                    raise ValueError("decoupled generation requires two noise schedules")
                entity_noise, robot_noise = noise_schedules
            entity_vel = self.entity_diffusion.sample(
                shape=shape,
                context=node_context,
                agent_mask=batch.entity_mask & batch.agent_mask,
                guidance_fn=None,
                guidance_scale=0.0,
                step_stride=step_stride,
                noise_schedule=entity_noise,
            )
            if guided:
                entity_vel_fixed = entity_vel.detach()

                def guidance_fn(robot_x0_vel: torch.Tensor) -> torch.Tensor:
                    full_x0_vel = torch.where(
                        batch.entity_mask[:, :, None, None],
                        entity_vel_fixed,
                        robot_x0_vel,
                    )
                    return self.guidance.robot_only_gradient(full_x0_vel, batch)

                guidance_fn.guidance_step_clip_norm = float(self.guidance.config.guidance_step_clip_norm)
                projection_fn = self.guidance.make_projection_fn(batch)
            else:
                guidance_fn = None
                projection_fn = None
            robot_vel = self.robot_diffusion.sample(
                shape=shape,
                context=node_context,
                agent_mask=batch.robot_mask & batch.agent_mask,
                guidance_fn=guidance_fn,
                guidance_scale=self.config.guidance_scale if guided else 0.0,
                step_stride=step_stride,
                projection_fn=projection_fn,
                noise_schedule=robot_noise,
            )
            future_vel = torch.where(batch.robot_mask[:, :, None, None], robot_vel, entity_vel)
        else:
            if isinstance(noise_schedules, tuple):
                raise ValueError("joint generation accepts one noise schedule")
            guidance_fn = self.guidance.make_guidance_fn(batch) if guided else None
            projection_fn = self.guidance.make_projection_fn(batch) if guided else None
            future_vel = self.diffusion.sample(
                shape=shape,
                context=node_context,
                agent_mask=batch.agent_mask,
                guidance_fn=guidance_fn,
                guidance_scale=self.config.guidance_scale if guided else 0.0,
                step_stride=step_stride,
                projection_fn=projection_fn,
                noise_schedule=noise_schedules,
            )
        future_pos = integrate_velocities(future_vel, batch.current_pos, batch.dt)
        return {
            "future_vel": future_vel,
            "future_pos": future_pos,
            "diagnostics": self.guidance.diagnostics(future_vel, batch),
        }
