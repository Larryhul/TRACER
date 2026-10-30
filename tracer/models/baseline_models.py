from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from tracer.data.heterogeneous_scene import HeterogeneousSceneBatch, integrate_velocities
from tracer.models.hgt_encoder import HeterogeneousGraphEncoder
from tracer.models.joint_diffusion import JointDiffusion
from tracer.models.sage_model import SAGEConfig


@dataclass
class LearnedBaselineConfig:
    """Configuration shared by protocol-aligned learned baselines."""

    baseline: str = "trajectron_plus_plus"
    latent_dim: int = 32
    kl_weight: float = 0.01
    decoder_hidden_dim: int = 256
    min_logvar: float = -8.0
    max_logvar: float = 4.0


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask[:, :, None, None].to(pred.dtype)
    denom = mask_f.sum().clamp_min(1.0) * pred.shape[2] * pred.shape[3]
    return (((pred - target) ** 2) * mask_f).sum() / denom


def _kl_normal(
    mean_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mean_p: torch.Tensor,
    logvar_p: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mean_q - mean_p) ** 2) / var_p.clamp_min(1e-8) - 1.0)
    mask_f = mask.to(kl.dtype)
    return (kl.sum(dim=-1) * mask_f).sum() / mask_f.sum().clamp_min(1.0)


class TrajectronPlusPlusBaseline(nn.Module):
    """A compact Trajectron++-style CVAE baseline under the SAGE protocol.

    The implementation keeps the paper-comparison protocol identical to SAGE:
    it consumes the same heterogeneous scene batch and predicts stochastic future
    velocities for all supervised agents. The encoder is graph-conditioned, and
    the decoder samples a per-agent latent variable from a learned prior.
    """

    def __init__(self, sage_config: SAGEConfig, baseline_config: LearnedBaselineConfig) -> None:
        super().__init__()
        self.sage_config = sage_config
        self.baseline_config = baseline_config
        self.encoder = HeterogeneousGraphEncoder(sage_config.encoder)
        context_dim = int(sage_config.encoder.hidden_dim)
        latent_dim = int(baseline_config.latent_dim)
        horizon = int(sage_config.diffusion.horizon)
        point_dim = int(sage_config.diffusion.point_dim)
        decoder_hidden = int(baseline_config.decoder_hidden_dim)

        self.future_encoder = nn.GRU(
            input_size=point_dim,
            hidden_size=context_dim,
            num_layers=1,
            batch_first=True,
        )
        self.prior = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, 2 * latent_dim),
        )
        self.posterior = nn.Sequential(
            nn.LayerNorm(2 * context_dim),
            nn.Linear(2 * context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(context_dim + latent_dim),
            nn.Linear(context_dim + latent_dim, decoder_hidden),
            nn.GELU(),
            nn.Dropout(sage_config.encoder.dropout),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, horizon * point_dim),
        )

    def _split_stats(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = torch.chunk(stats, 2, dim=-1)
        logvar = logvar.clamp(
            min=float(self.baseline_config.min_logvar),
            max=float(self.baseline_config.max_logvar),
        )
        return mean, logvar

    def _sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def _decode(self, context: torch.Tensor, z: torch.Tensor, batch: HeterogeneousSceneBatch) -> torch.Tensor:
        b, a, _ = context.shape
        horizon = int(self.sage_config.diffusion.horizon)
        point_dim = int(self.sage_config.diffusion.point_dim)
        pred = self.decoder(torch.cat([context, z], dim=-1)).view(b, a, horizon, point_dim)
        return pred * batch.agent_mask[:, :, None, None].to(pred.dtype)

    def forward(self, batch: HeterogeneousSceneBatch) -> Dict[str, torch.Tensor]:
        context, _ = self.encoder(batch)
        b, a, t, d = batch.future.shape
        future = batch.future.reshape(b * a, t, d)
        _, hidden = self.future_encoder(future)
        future_context = hidden[-1].view(b, a, -1)

        prior_mean, prior_logvar = self._split_stats(self.prior(context))
        post_mean, post_logvar = self._split_stats(self.posterior(torch.cat([context, future_context], dim=-1)))
        z = self._sample(post_mean, post_logvar)
        pred = self._decode(context, z, batch)

        mask = batch.supervision_mask()
        recon = _masked_mse(pred, batch.future, mask)
        kl = _kl_normal(post_mean, post_logvar, prior_mean, prior_logvar, mask)
        loss = recon + float(self.baseline_config.kl_weight) * kl
        return {"loss": loss, "recon_loss": recon, "kl_loss": kl}

    @torch.no_grad()
    def generate(self, batch: HeterogeneousSceneBatch, samples: int = 1, step_stride: int = 1) -> Dict[str, torch.Tensor]:
        del step_stride
        context, _ = self.encoder(batch)
        prior_mean, prior_logvar = self._split_stats(self.prior(context))
        outputs = []
        for _ in range(max(1, int(samples))):
            z = self._sample(prior_mean, prior_logvar)
            outputs.append(self._decode(context, z, batch))
        future_vel = torch.stack(outputs, dim=0)
        return {
            "future_vel": future_vel,
            "future_pos": integrate_velocities(future_vel.flatten(0, 1), batch.current_pos.repeat(len(outputs), 1, 1), batch.dt),
        }


class MIDBaseline(nn.Module):
    """MID-style diffusion baseline under the same SAGE data protocol."""

    def __init__(self, sage_config: SAGEConfig, baseline_config: LearnedBaselineConfig) -> None:
        super().__init__()
        self.sage_config = sage_config
        self.baseline_config = baseline_config
        self.encoder = HeterogeneousGraphEncoder(sage_config.encoder)
        self.diffusion = JointDiffusion(sage_config.diffusion)

    def _context(self, batch: HeterogeneousSceneBatch) -> torch.Tensor:
        node_context, graph_context = self.encoder(batch)
        denoiser_type = str(getattr(self.sage_config.diffusion, "denoiser_type", "transformer")).lower()
        supports_context_tokens = denoiser_type in {"transformer", "attention"}
        if getattr(self.sage_config.diffusion, "use_graph_context_token", False) and supports_context_tokens:
            return torch.cat([node_context, graph_context[:, None, :]], dim=1)
        return node_context

    def forward(self, batch: HeterogeneousSceneBatch) -> Dict[str, torch.Tensor]:
        context = self._context(batch)
        loss = self.diffusion.loss(batch.future, context, batch.supervision_mask())
        return {"loss": loss}

    @torch.no_grad()
    def generate(self, batch: HeterogeneousSceneBatch, samples: int = 1, step_stride: int = 1) -> Dict[str, torch.Tensor]:
        context = self._context(batch)
        shape = (
            batch.batch_size,
            batch.num_agents,
            int(self.sage_config.diffusion.horizon),
            int(self.sage_config.diffusion.point_dim),
        )
        outputs = []
        for _ in range(max(1, int(samples))):
            outputs.append(
                self.diffusion.sample(
                    shape=shape,
                    context=context,
                    agent_mask=batch.agent_mask,
                    guidance_fn=None,
                    guidance_scale=0.0,
                    step_stride=step_stride,
                    projection_fn=None,
                )
            )
        future_vel = torch.stack(outputs, dim=0)
        return {
            "future_vel": future_vel,
            "future_pos": integrate_velocities(future_vel.flatten(0, 1), batch.current_pos.repeat(len(outputs), 1, 1), batch.dt),
        }


def create_learned_baseline(
    name: str,
    sage_config: SAGEConfig,
    baseline_config: LearnedBaselineConfig | None = None,
) -> nn.Module:
    baseline_config = baseline_config or LearnedBaselineConfig(baseline=name)
    baseline_config.baseline = str(name)
    key = str(name).lower().replace("-", "_")
    if key in {"trajectron_plus_plus", "trajectron++", "trajectronpp", "tpp"}:
        return TrajectronPlusPlusBaseline(sage_config, baseline_config)
    if key in {"mid", "motion_indeterminacy_diffusion"}:
        return MIDBaseline(sage_config, baseline_config)
    raise ValueError(f"unknown learned baseline: {name!r}")
