from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModeLikelihoodConfig:
    action_dim: int = 4
    response_dim: int = 2
    num_modes: int = 3
    hidden_dim: int = 64
    min_log_std: float = -4.0
    max_log_std: float = 1.0


class ModeConditionedResponseLikelihood(nn.Module):
    """Learned Gaussian p(response | team-action features, entity mode)."""

    def __init__(self, config: ModeLikelihoodConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.action_dim + config.num_modes
        self.mean_network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.response_dim),
        )
        self.log_std = nn.Parameter(torch.full((config.response_dim,), -1.5))

    def _means(self, actions: torch.Tensor, mode_ids: torch.Tensor) -> torch.Tensor:
        if actions.shape[:-1] != mode_ids.shape or actions.shape[-1] != self.config.action_dim:
            raise ValueError("actions and mode_ids must have aligned leading dimensions")
        mode_one_hot = F.one_hot(mode_ids, num_classes=self.config.num_modes).to(actions.dtype)
        return self.mean_network(torch.cat([actions, mode_one_hot], dim=-1))

    def forward(self, actions: torch.Tensor, mode_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = self._means(actions, mode_ids)
        log_std = self.log_std.clamp(
            self.config.min_log_std, self.config.max_log_std
        ).expand_as(mean)
        return {"mean": mean, "log_std": log_std}

    def all_mode_log_likelihood(
        self,
        actions: torch.Tensor,
        responses: torch.Tensor,
    ) -> torch.Tensor:
        if responses.shape != (*actions.shape[:-1], self.config.response_dim):
            raise ValueError("responses must align with actions")
        mode_shape = (*actions.shape[:-1], self.config.num_modes)
        mode_ids = torch.arange(self.config.num_modes, device=actions.device)
        mode_ids = mode_ids.view(*([1] * (actions.ndim - 1)), self.config.num_modes).expand(mode_shape)
        expanded_actions = actions[..., None, :].expand(*mode_shape, self.config.action_dim)
        mean = self._means(expanded_actions, mode_ids)
        log_std = self.log_std.clamp(self.config.min_log_std, self.config.max_log_std)
        residual = responses[..., None, :] - mean
        return (
            -0.5 * residual.square() * torch.exp(-2.0 * log_std)
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
