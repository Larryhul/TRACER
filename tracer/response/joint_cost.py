"""Permutation-invariant joint candidate cost model.

The response model remains useful for trajectory explanation and uncertainty,
but candidate selection can be trained against the same realized joint cost
that the closed-loop evaluator measures.  This module keeps that planner head
independent from the frozen response checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class JointCandidateCostConfig:
    history: int = 8
    horizon: int = 12
    num_robots: int = 3
    hidden_dim: int = 96
    dropout: float = 0.05
    dt: float = 0.3
    coordinate_min: float = 0.0
    coordinate_max: float = 2.0


class JointCandidateCostModel(nn.Module):
    """Predict one cost per shared candidate after entity-set aggregation."""

    def __init__(self, config: JointCandidateCostConfig) -> None:
        super().__init__()
        if config.history < 1 or config.horizon < 1:
            raise ValueError("history and horizon must be positive")
        if config.num_robots < 1:
            raise ValueError("num_robots must be positive")
        if config.coordinate_max <= config.coordinate_min:
            raise ValueError("coordinate_max must exceed coordinate_min")
        hidden = int(config.hidden_dim)
        self.config = config
        self.history_encoder = nn.GRU(
            input_size=4, hidden_size=hidden, batch_first=True
        )
        self.robot_encoder = nn.Sequential(
            nn.Linear(8, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.current_encoder = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.parameter_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.parameter_encoder = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.evidence_encoder = nn.GRU(
            input_size=6, hidden_size=hidden, batch_first=True
        )
        self.entity_encoder = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
        )
        self.cost_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _history_features(history: torch.Tensor, dt: float) -> torch.Tensor:
        current = history[..., -1:, :]
        relative = history - current
        velocity = torch.zeros_like(relative)
        velocity[..., 1:, :] = (history[..., 1:, :] - history[..., :-1, :]) / dt
        return torch.cat([relative, velocity], dim=-1)

    def forward(
        self,
        history: torch.Tensor,
        robot_current: torch.Tensor,
        robot_candidates: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        evidence_candidates: torch.Tensor | None = None,
        evidence_responses: torch.Tensor | None = None,
        evidence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return costs with shapes ``(batch, candidates)``.

        ``history`` is ``(B, E, H, 2)``, current state is
        ``(B, C, E, R, 4)``, and candidates are ``(B, C, E, R, T, 4)``.
        The entity axis is pooled with both mean and max statistics, making the
        head invariant to entity ordering while retaining interaction context.
        """

        if history.ndim != 4 or history.shape[-1] != 2:
            raise ValueError("history must have shape (batch, entities, history, 2)")
        if robot_current.ndim != 5 or robot_current.shape[-1] != 4:
            raise ValueError(
                "robot_current must have shape (batch, candidates, entities, robots, 4)"
            )
        if robot_candidates.ndim != 6 or robot_candidates.shape[-1] != 4:
            raise ValueError(
                "robot_candidates must have shape (batch, candidates, entities, robots, horizon, 4)"
            )
        batch, entities = history.shape[:2]
        candidate_count = robot_current.shape[1]
        if tuple(robot_current.shape[:3]) != (
            batch,
            candidate_count,
            entities,
        ):
            raise ValueError("robot_current must align with history and candidates")
        if tuple(robot_candidates.shape[:4]) != tuple(robot_current.shape[:4]):
            raise ValueError("robot_candidates must align with robot_current")
        if robot_candidates.shape[-2] != self.config.horizon:
            raise ValueError("robot_candidates horizon does not match config")
        if robot_current.shape[-2] != self.config.num_robots:
            raise ValueError("robot_current robot count does not match config")
        if active_mask is None:
            active_mask = torch.ones(
                batch, entities, device=history.device, dtype=torch.bool
            )
        elif tuple(active_mask.shape) != (batch, entities):
            raise ValueError("active_mask must have shape (batch, entities)")
        else:
            active_mask = active_mask.to(device=history.device, dtype=torch.bool)

        history_input = self._history_features(history, self.config.dt)
        _, history_state = self.history_encoder(
            history_input.reshape(batch * entities, history.shape[2], 4)
        )
        history_token = history_state[-1].reshape(batch, entities, -1)
        history_token = history_token + self._evidence_token(
            evidence_candidates,
            evidence_responses,
            evidence_mask,
            batch,
            entities,
            history.device,
            history.dtype,
        )
        parameter_prediction = self.parameter_prediction_from_history(history_token)
        normalized_parameter = (
            2.0 * (parameter_prediction - self.config.coordinate_min)
            / (self.config.coordinate_max - self.config.coordinate_min)
            - 1.0
        ).clamp(-1.0, 1.0)
        parameter_token = self.parameter_encoder(normalized_parameter.unsqueeze(-1))

        current = robot_current.unsqueeze(-2).expand(
            -1, -1, -1, -1, self.config.horizon, -1
        )
        robot_tokens = self.robot_encoder(torch.cat([current, robot_candidates], dim=-1))
        candidate_entity = robot_tokens.mean(dim=-2).mean(dim=-2)
        # The third block carries current robot geometry through a compact
        # encoder rather than duplicating raw coordinates in the cost head.
        current_entity = self.current_encoder(robot_current).mean(dim=-2)
        entity_tokens = torch.cat(
            [
                history_token[:, None].expand(
                    -1, candidate_count, entities, -1
                ),
                parameter_token[:, None].expand(
                    -1, candidate_count, -1, -1
                ),
                candidate_entity,
                current_entity,
            ],
            dim=-1,
        )
        entity_tokens = self.entity_encoder(entity_tokens)
        mask = active_mask[:, None, :, None].to(entity_tokens.dtype)
        count = mask.sum(dim=2).clamp_min(1.0)
        mean_token = (entity_tokens * mask).sum(dim=2) / count
        max_token = entity_tokens.masked_fill(~active_mask[:, None, :, None], -1e4).amax(dim=2)
        return self.cost_head(torch.cat([mean_token, max_token], dim=-1)).squeeze(-1)

    def parameter_prediction_from_history(self, history_token: torch.Tensor) -> torch.Tensor:
        """Return a bounded continuous behavior-coordinate estimate."""

        raw = self.parameter_head(history_token).squeeze(-1)
        return self.config.coordinate_min + torch.sigmoid(raw) * (
            self.config.coordinate_max - self.config.coordinate_min
        )

    def predict_parameters(
        self,
        history: torch.Tensor,
        evidence_candidates: torch.Tensor | None = None,
        evidence_responses: torch.Tensor | None = None,
        evidence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate one continuous coordinate for each entity history."""

        if history.ndim != 4 or history.shape[-1] != 2:
            raise ValueError("history must have shape (batch, entities, history, 2)")
        batch, entities = history.shape[:2]
        features = self._history_features(history, self.config.dt)
        _, state = self.history_encoder(
            features.reshape(batch * entities, history.shape[2], 4)
        )
        history_token = state[-1].reshape(batch, entities, -1)
        history_token = history_token + self._evidence_token(
            evidence_candidates,
            evidence_responses,
            evidence_mask,
            batch,
            entities,
            history.device,
            history.dtype,
        )
        return self.parameter_prediction_from_history(history_token)

    def _evidence_token(
        self,
        evidence_candidates: torch.Tensor | None,
        evidence_responses: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        batch: int,
        entities: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if evidence_candidates is None and evidence_responses is None:
            return torch.zeros(batch, entities, self.config.hidden_dim, device=device, dtype=dtype)
        if evidence_candidates is None or evidence_responses is None or evidence_mask is None:
            raise ValueError("evidence candidates, responses, and mask must be provided together")
        if evidence_candidates.ndim != 6 or evidence_candidates.shape[-1] != 4:
            raise ValueError("evidence_candidates must have shape (batch, entities, steps, robots, horizon, 4)")
        if evidence_responses.ndim != 5 or evidence_responses.shape[-1] != 2:
            raise ValueError("evidence_responses must have shape (batch, entities, steps, horizon, 2)")
        if tuple(evidence_candidates.shape[:3]) != tuple(evidence_responses.shape[:3]):
            raise ValueError("evidence candidates and responses must align")
        if tuple(evidence_candidates.shape[:2]) != (batch, entities):
            raise ValueError("evidence must align with history")
        steps = evidence_candidates.shape[2]
        if tuple(evidence_mask.shape) != (batch, steps):
            raise ValueError("evidence_mask must have shape (batch, steps)")
        action = evidence_candidates.mean(dim=(-3, -2))
        response = evidence_responses.mean(dim=-2)
        tokens = torch.cat([action, response], dim=-1).to(device=device, dtype=dtype)
        flat = tokens.reshape(batch * entities, steps, 6)
        _, state = self.evidence_encoder(flat)
        lengths = evidence_mask.to(device=device, dtype=torch.long).sum(dim=-1).clamp_min(1)
        indices = (lengths - 1)[:, None].expand(batch, entities).reshape(-1)
        encoded = state[-1].reshape(batch, entities, -1)
        selected = encoded.reshape(batch * entities, -1).gather(
            1, indices[:, None].expand(-1, encoded.shape[-1])
        )
        selected = selected.reshape(batch, entities, -1)
        has_evidence = evidence_mask.to(device=device).any(dim=-1)[:, None, None]
        return torch.where(has_evidence, selected, torch.zeros_like(selected))
