from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn


@dataclass
class DiffusionConfig:
    horizon: int = 12
    point_dim: int = 2
    context_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ff_mult: int = 4
    num_steps: int = 100
    beta_1: float = 1e-4
    beta_T: float = 5e-2
    dropout: float = 0.1
    denoiser_type: str = "transformer"
    use_agent_attention: bool = True
    guidance_update: str = "mean"
    use_graph_context_token: bool = True


class SinusoidalTimeEmbedding(nn.Module):
    """Embed diffusion noise levels into the denoiser hidden space."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        if beta.ndim == 0:
            beta = beta[None]
        beta = beta.view(-1, 1)
        half_dim = self.hidden_dim // 2
        freqs = torch.exp(
            torch.linspace(
                0.0,
                -torch.log(torch.tensor(10000.0, device=beta.device, dtype=beta.dtype)),
                half_dim,
                device=beta.device,
                dtype=beta.dtype,
            )
        )
        angles = beta * freqs[None] * 1000.0
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.hidden_dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.proj(emb)


class TemporalCrossAttentionBlock(nn.Module):
    """Temporal, agent-agent, and scene cross-attention denoising block."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_mult: int, dropout: float, use_agent_attention: bool = True) -> None:
        super().__init__()
        self.use_agent_attention = bool(use_agent_attention)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.agent_attn = None
        self.norm_agent = None
        if self.use_agent_attention:
            self.agent_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm_agent = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_temporal = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_mult * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, batch_size: int, num_agents: int) -> torch.Tensor:
        temporal, _ = self.temporal_attn(x, x, x, need_weights=False)
        x = self.norm_temporal(x + self.dropout(temporal))
        if self.agent_attn is not None and self.norm_agent is not None and num_agents > 1:
            _, horizon, hidden = x.shape
            agent_tokens = x.view(batch_size, num_agents, horizon, hidden).transpose(1, 2).reshape(batch_size * horizon, num_agents, hidden)
            agent_msg, _ = self.agent_attn(agent_tokens, agent_tokens, agent_tokens, need_weights=False)
            agent_tokens = self.norm_agent(agent_tokens + self.dropout(agent_msg))
            x = agent_tokens.view(batch_size, horizon, num_agents, hidden).transpose(1, 2).reshape(batch_size * num_agents, horizon, hidden)
        cross, _ = self.cross_attn(x, context, context, need_weights=False)
        x = self.norm_cross(x + self.dropout(cross))
        return self.norm_ff(x + self.dropout(self.ffn(x)))



class LegacyMLPDenoiser(nn.Module):
    """MLP denoiser used by earlier SAGE checkpoints.

    The legacy model predicts each agent-time velocity independently from the
    noisy point, three scalar diffusion-time features, and that agent's context.
    """

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.point_dim + config.context_dim + 3, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.point_dim),
        )

    def forward(self, x: torch.Tensor, beta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, a, t, _ = x.shape
        if context.ndim == 2:
            context = context[:, None, :].expand(b, a, context.shape[-1])
        if context.ndim != 3:
            raise ValueError(f"context must be (B,C) or (B,A,C), got {tuple(context.shape)}")
        if context.shape[1] != a:
            if context.shape[1] == 1:
                context = context.expand(b, a, context.shape[-1])
            else:
                raise ValueError(f"context agent dimension {context.shape[1]} does not match x agent dimension {a}")
        beta_feat = beta.view(b, 1, 1, 1).expand(b, a, t, 1)
        time_feat = torch.cat([beta_feat, torch.sin(beta_feat), torch.cos(beta_feat)], dim=-1)
        ctx = context[:, :, None, :].expand(b, a, t, context.shape[-1])
        return self.net(torch.cat([x, time_feat, ctx], dim=-1))


class JointDenoiser(nn.Module):
    """Temporal Transformer denoiser with cross-attention scene conditioning."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_proj = nn.Linear(config.point_dim, config.hidden_dim)
        self.context_proj = nn.Linear(config.context_dim, config.hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(config.hidden_dim)
        self.agent_embedding = nn.Embedding(256, config.hidden_dim)
        self.temporal_embedding = nn.Embedding(config.horizon, config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                TemporalCrossAttentionBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    ff_mult=config.ff_mult,
                    dropout=config.dropout,
                    use_agent_attention=config.use_agent_attention,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(config.hidden_dim)
        self.out = nn.Linear(config.hidden_dim, config.point_dim)

    def _context_sequence(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim == 2:
            context = context[:, None, :]
        if context.ndim != 3:
            raise ValueError(f"context must be (B,C) or (B,L,C), got {tuple(context.shape)}")
        return self.context_proj(context)

    def forward(self, x: torch.Tensor, beta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, a, t, _ = x.shape
        if t > self.config.horizon:
            raise ValueError(f"trajectory horizon {t} exceeds configured horizon {self.config.horizon}")

        tokens = self.input_proj(x)
        time = self.time_embedding(beta).view(b, 1, 1, self.config.hidden_dim)
        agent_ids = torch.arange(a, device=x.device).clamp(max=self.agent_embedding.num_embeddings - 1)
        agent = self.agent_embedding(agent_ids).view(1, a, 1, self.config.hidden_dim)
        steps = torch.arange(t, device=x.device)
        temporal = self.temporal_embedding(steps).view(1, 1, t, self.config.hidden_dim)
        tokens = tokens + time + agent + temporal

        tokens = tokens.view(b * a, t, self.config.hidden_dim)
        scene = self._context_sequence(context)
        scene = scene[:, None, :, :].expand(b, a, scene.shape[1], scene.shape[2])
        scene = scene.reshape(b * a, scene.shape[2], scene.shape[3])
        for layer in self.layers:
            tokens = layer(tokens, scene, batch_size=b, num_agents=a)
        return self.out(self.out_norm(tokens)).view(b, a, t, self.config.point_dim)


class JointDiffusion(nn.Module):
    """Phase 2: conditional diffusion over robot and entity future velocities."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config
        denoiser_type = str(config.denoiser_type).lower()
        if denoiser_type in {"mlp", "legacy_mlp"}:
            self.denoiser = LegacyMLPDenoiser(config)
        elif denoiser_type in {"transformer", "attention"}:
            self.denoiser = JointDenoiser(config)
        else:
            raise ValueError(f"unknown denoiser_type={config.denoiser_type!r}")
        betas = torch.linspace(config.beta_1, config.beta_T, config.num_steps)
        betas = torch.cat([torch.zeros(1), betas])
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas", torch.sqrt(betas))

    def loss(self, x0: torch.Tensor, context: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        steps = torch.randint(1, self.config.num_steps + 1, (b,), device=x0.device)
        alpha_bar = self.alpha_bars[steps].view(b, 1, 1, 1)
        noise = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
        pred = self.denoiser(xt, self.betas[steps], context)
        mask = agent_mask[:, :, None, None].to(x0.dtype)
        denom = mask.sum().clamp_min(1.0) * x0.shape[2] * x0.shape[3]
        return (((pred - noise) ** 2) * mask).sum() / denom

    @staticmethod
    def _clip_trajectory_norm(tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
        if max_norm <= 0.0:
            return tensor
        flat = tensor.flatten(start_dim=2)
        norm = torch.linalg.vector_norm(flat, dim=-1).view(tensor.shape[0], tensor.shape[1], 1, 1)
        scale = (float(max_norm) / norm.clamp_min(1e-12)).clamp(max=1.0)
        return tensor * scale

    def _guidance_delta(
        self,
        grad: torch.Tensor | None,
        guidance_scale: float,
        guidance_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor | None:
        if grad is None:
            return None
        delta = float(guidance_scale) * torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        step_clip = float(getattr(guidance_fn, "guidance_step_clip_norm", 0.0) or 0.0)
        if step_clip > 0.0:
            delta = self._clip_trajectory_norm(delta, step_clip)
        return delta

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int, int],
        context: torch.Tensor,
        agent_mask: torch.Tensor,
        guidance_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        guidance_scale: float = 0.0,
        step_stride: int = 1,
        projection_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        noise_schedule: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if noise_schedule is not None and not noise_schedule:
            raise ValueError("noise_schedule must contain the initial noise tensor")
        noise_index = 0
        x = (
            noise_schedule[noise_index]
            if noise_schedule is not None
            else torch.randn(shape, device=context.device)
        )
        noise_index += 1
        mask = agent_mask[:, :, None, None].to(x.dtype)
        stride = max(1, int(step_stride))
        for step in range(self.config.num_steps, 0, -stride):
            step_next = max(step - stride, 0)
            beta = self.betas[[step] * shape[0]]
            alpha_bar = self.alpha_bars[step]
            alpha_bar_next = self.alpha_bars[step_next]
            eps = self.denoiser(x, beta, context)

            x0 = (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
            grad = None
            if guidance_fn is not None and guidance_scale > 0.0:
                grad = guidance_fn(x0)
            delta = self._guidance_delta(grad, guidance_scale, guidance_fn)

            update = str(self.config.guidance_update).lower()
            if update == "x0":
                if delta is not None:
                    x0 = x0 - delta
                if projection_fn is not None:
                    x0 = projection_fn(x0)
                if step_next == 0:
                    x = x0
                else:
                    if noise_schedule is None:
                        noise = torch.randn_like(x)
                    else:
                        if noise_index >= len(noise_schedule):
                            raise ValueError("noise_schedule is shorter than the diffusion path")
                        noise = noise_schedule[noise_index]
                        noise_index += 1
                    x = torch.sqrt(alpha_bar_next) * x0 + torch.sqrt(1.0 - alpha_bar_next) * noise
            elif update == "mean":
                sqrt_alpha_next = torch.sqrt(alpha_bar_next)
                if step_next == 0:
                    mean = x0
                    noise_scale = 0.0
                    mean_scale = 1.0
                else:
                    mean = sqrt_alpha_next * x0
                    noise_scale = torch.sqrt(1.0 - alpha_bar_next)
                    mean_scale = sqrt_alpha_next
                if delta is not None:
                    mean = mean - mean_scale * delta
                if projection_fn is not None:
                    if step_next == 0:
                        mean = projection_fn(mean)
                    else:
                        projected_x0 = projection_fn(mean / sqrt_alpha_next.clamp_min(1e-12))
                        mean = sqrt_alpha_next * projected_x0
                if step_next == 0:
                    x = mean
                else:
                    if noise_schedule is None:
                        noise = torch.randn_like(x)
                    else:
                        if noise_index >= len(noise_schedule):
                            raise ValueError("noise_schedule is shorter than the diffusion path")
                        noise = noise_schedule[noise_index]
                        noise_index += 1
                    x = mean + noise_scale * noise
            else:
                raise ValueError(f"unknown guidance_update={self.config.guidance_update!r}")
            x = x * mask
        if noise_schedule is not None and noise_index != len(noise_schedule):
            raise ValueError("noise_schedule contains unused diffusion noise")
        return x
