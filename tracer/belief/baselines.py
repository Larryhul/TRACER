from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RecurrentBeliefConfig:
    action_dim: int = 4
    response_dim: int = 2
    num_modes: int = 3
    hidden_dim: int = 32
    dropout: float = 0.0


class RecurrentBeliefBaseline(nn.Module):
    """Implicit-history baseline that predicts the entity mode after each window."""

    def __init__(self, config: RecurrentBeliefConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.GRU(
            input_size=config.action_dim + config.response_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_modes),
        )

    def forward(self, actions: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != self.config.action_dim:
            raise ValueError("actions must have shape (batch, windows, action_dim)")
        if responses.shape != (*actions.shape[:2], self.config.response_dim):
            raise ValueError("responses must align with actions and response_dim")
        states, _ = self.encoder(torch.cat([actions, responses], dim=-1))
        return torch.log_softmax(self.decoder(states), dim=-1)
