"""Shared TRACER feature encoders.

The module deliberately keeps the graph implementation dependency-free.  This
allows the synthetic and observational response experiments to use the same
architecture on machines where PyG is not installed, while retaining typed
message passing and the permutation-invariant robot-set interface described in
the paper.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tracer.models.hgt_encoder import HGTLayer


def stationary_reference_candidates(
    robot_current: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Return a geometry-preserving zero-velocity reference trajectory."""

    if robot_current.ndim < 2 or robot_current.shape[-1] != 4:
        raise ValueError("robot_current must have shape (..., robots, 4)")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    position = robot_current[..., None, :2].expand(
        *robot_current.shape[:-1], int(horizon), 2
    )
    return torch.cat([position, torch.zeros_like(position)], dim=-1)


class TemporalCandidateEncoder(nn.Module):
    """Encode one candidate trajectory per robot with temporal attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        pooling: str = "mean",
        dynamic_interaction_features: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if pooling not in {"mean", "attention"}:
            raise ValueError("pooling must be 'mean' or 'attention'")
        self.pooling = pooling
        self.dynamic_interaction_features = bool(dynamic_interaction_features)
        if pooling == "attention":
            self.pool_score = nn.Linear(hidden_dim, 1)
            self.phase_projection = nn.Linear(hidden_dim * 3, hidden_dim)
        input_dim = 16 if self.dynamic_interaction_features else 8
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _temporal_position(length: int, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype)[:, None]
        div = torch.exp(
            torch.arange(0, hidden_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(length, hidden_dim, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].shape[-1]])
        return encoding

    def forward(
        self,
        robot_current: torch.Tensor,
        robot_candidates: torch.Tensor,
        target_current: torch.Tensor | None = None,
        target_velocity: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if robot_current.ndim != 3 or robot_current.shape[-1] != 4:
            raise ValueError("robot_current must have shape (batch, robots, 4)")
        if robot_candidates.ndim != 4 or robot_candidates.shape[-1] != 4:
            raise ValueError(
                "robot_candidates must have shape (batch, robots, horizon, 4)"
            )
        if tuple(robot_current.shape[:2]) != tuple(robot_candidates.shape[:2]):
            raise ValueError("robot_current and robot_candidates must align")
        batch, robots, horizon, _ = robot_candidates.shape
        current = robot_current[:, :, None, :].expand(-1, -1, horizon, -1)
        features = [current, robot_candidates]
        if self.dynamic_interaction_features:
            if target_current is None or target_velocity is None:
                raise ValueError(
                    "dynamic interaction features require target_current and target_velocity"
                )
            if tuple(target_current.shape) != (batch, 2):
                raise ValueError("target_current must have shape (batch, 2)")
            if tuple(target_velocity.shape) != (batch, 2):
                raise ValueError("target_velocity must have shape (batch, 2)")
            relative_position = robot_candidates[..., :2] - target_current[:, None, None, :]
            relative_velocity = robot_candidates[..., 2:] - target_velocity[:, None, None, :]
            distance = torch.linalg.vector_norm(
                relative_position, dim=-1, keepdim=True
            )
            closing = -(
                relative_position * relative_velocity
            ).sum(dim=-1, keepdim=True) / distance.clamp_min(1e-6)
            time_to_closest = -(
                relative_position * relative_velocity
            ).sum(dim=-1, keepdim=True) / relative_velocity.square().sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            time_to_closest = time_to_closest.clamp(0.0, 10.0)
            closest_distance = torch.linalg.vector_norm(
                relative_position + time_to_closest * relative_velocity,
                dim=-1,
                keepdim=True,
            )
            features.append(
                torch.cat(
                    [
                        relative_position,
                        relative_velocity,
                        distance,
                        closing,
                        time_to_closest,
                        closest_distance,
                    ],
                    dim=-1,
                )
            )
        tokens = self.input(torch.cat(features, dim=-1))
        position = self._temporal_position(
            horizon, tokens.shape[-1], tokens.device, tokens.dtype
        )
        tokens = tokens.reshape(batch * robots, horizon, -1) + position[None]
        sequence = self.transformer(tokens)
        if self.pooling == "attention":
            weights = self.pool_score(sequence).softmax(dim=1)
            pooled = (sequence * weights).sum(dim=1)
            encoded = self.phase_projection(torch.cat([sequence[:, 0], pooled, sequence[:, -1]], dim=-1))
        else:
            encoded = sequence.mean(dim=1)
        encoded_output = self.output(encoded).reshape(batch, robots, -1)
        if return_sequence:
            sequence_output = self.output(sequence).reshape(batch, robots, horizon, -1)
            return encoded_output, sequence_output
        return encoded_output


class TRACERFeatureEncoder(nn.Module):
    """Encode target context and candidate/reference robot trajectories."""

    num_node_types = 3

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        graph_layers: int,
        dropout: float,
        dt: float,
        num_robots: int,
        geometry_features: bool = False,
        temporal_pooling: str = "mean",
        dynamic_interaction_features: bool = False,
        return_temporal_sequence: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.dt = float(dt)
        self.num_robots = int(num_robots)
        self.geometry_features = bool(geometry_features)
        self.dynamic_interaction_features = bool(dynamic_interaction_features)
        self.return_temporal_sequence = bool(return_temporal_sequence)
        if self.geometry_features:
            self.geometry_encoder = nn.Sequential(
                nn.Linear(8, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
            )
        self.history_encoder = nn.LSTM(
            input_size=4,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.robot_state_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_type_embedding = nn.Embedding(self.num_node_types, hidden_dim)
        self.graph_layers = nn.ModuleList(
            [
                HGTLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_relations=self.num_node_types * self.num_node_types,
                    dropout=dropout,
                )
                for _ in range(graph_layers)
            ]
        )
        self.scene_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.candidate_encoder = TemporalCandidateEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout,
            pooling=temporal_pooling,
            dynamic_interaction_features=self.dynamic_interaction_features,
        )

    def _history_features(self, history: torch.Tensor) -> torch.Tensor:
        current = history[..., -1:, :]
        relative = history - current
        velocity = torch.zeros_like(relative)
        velocity[..., 1:, :] = (history[..., 1:, :] - history[..., :-1, :]) / self.dt
        return torch.cat([relative, velocity], dim=-1)

    def _history_token(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != 2:
            raise ValueError("history must have shape (batch, history, 2)")
        _, state = self.history_encoder(self._history_features(history))
        return state[0][-1]

    def _relations(self, node_types: torch.Tensor) -> torch.Tensor:
        # HGT relation ids use the directed source/target node type pair.
        source = node_types[:, None, :]
        target = node_types[:, :, None]
        return source * self.num_node_types + target

    def _graph_context(
        self,
        entity_history: torch.Tensor,
        robot_current: torch.Tensor,
        robot_mask: torch.Tensor,
        neighbor_history: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, robots = robot_current.shape[:2]
        entity_token = self._history_token(entity_history)[:, None, :]
        nodes = [entity_token]
        node_masks = [torch.ones(batch, 1, device=robot_current.device, dtype=torch.bool)]
        node_types = [torch.zeros(batch, 1, device=robot_current.device, dtype=torch.long)]

        if neighbor_history is not None:
            if neighbor_history.ndim != 4 or neighbor_history.shape[-1] != 2:
                raise ValueError(
                    "neighbor_history must have shape (batch, neighbors, history, 2)"
                )
            if neighbor_history.shape[0] != batch:
                raise ValueError("neighbor_history must align with entity_history")
            neighbors = neighbor_history.shape[1]
            neighbor_tokens = self._history_token(
                neighbor_history.reshape(batch * neighbors, *neighbor_history.shape[2:])
            ).reshape(batch, neighbors, -1)
            if neighbor_mask is None:
                neighbor_mask = torch.ones(
                    batch, neighbors, device=robot_current.device, dtype=torch.bool
                )
            if tuple(neighbor_mask.shape) != (batch, neighbors):
                raise ValueError("neighbor_mask must align with neighbor_history")
            nodes.append(neighbor_tokens)
            node_masks.append(neighbor_mask.to(device=robot_current.device, dtype=torch.bool))
            node_types.append(
                torch.ones(batch, neighbors, device=robot_current.device, dtype=torch.long)
            )

        nodes.append(self.robot_state_encoder(robot_current))
        node_masks.append(robot_mask)
        node_types.append(
            torch.full(
                (batch, robots),
                2,
                device=robot_current.device,
                dtype=torch.long,
            )
        )
        node_features = torch.cat(nodes, dim=1)
        active = torch.cat(node_masks, dim=1).bool()
        types = torch.cat(node_types, dim=1)
        node_features = node_features + self.node_type_embedding(types)
        if self.geometry_features:
            target_velocity = (entity_history[:, -1] - entity_history[:, -2]) / self.dt
            positions = [torch.zeros(batch, 1, 2, device=robot_current.device, dtype=robot_current.dtype)]
            velocities = [target_velocity[:, None]]
            if neighbor_history is not None:
                positions.append(neighbor_history[:, :, -1] - entity_history[:, None, -1])
                velocities.append((neighbor_history[:, :, -1] - neighbor_history[:, :, -2]) / self.dt)
            positions.append(robot_current[..., :2])
            velocities.append(robot_current[..., 2:])
            relative_position = torch.cat(positions, dim=1)
            relative_velocity = torch.cat(velocities, dim=1) - target_velocity[:, None]
            distance = torch.linalg.vector_norm(relative_position, dim=-1, keepdim=True)
            closing = -(relative_position * relative_velocity).sum(dim=-1, keepdim=True) / distance.clamp_min(1e-6)
            time_to_closest = (-(relative_position * relative_velocity).sum(dim=-1, keepdim=True) / relative_velocity.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(0.0, 10.0)
            closest_distance = torch.linalg.vector_norm(relative_position + time_to_closest * relative_velocity, dim=-1, keepdim=True)
            geometry = torch.cat([relative_position, relative_velocity, distance, closing, time_to_closest, closest_distance], dim=-1)
            node_features = node_features + self.geometry_encoder(geometry)
        relations = self._relations(types)
        valid_edges = active[:, :, None] & active[:, None, :]
        for layer in self.graph_layers:
            node_features = layer(node_features, relations, active, edge_mask=valid_edges)
        return self.scene_projection(node_features[:, 0])

    def forward(
        self,
        entity_history: torch.Tensor,
        robot_current: torch.Tensor,
        robot_candidates: torch.Tensor,
        reference_candidates: torch.Tensor,
        robot_mask: torch.Tensor,
        neighbor_history: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        robot_actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        scene = self._graph_context(
            entity_history,
            robot_current,
            robot_mask,
            neighbor_history,
            neighbor_mask,
        )
        target_velocity = (
            entity_history[:, -1] - entity_history[:, -2]
        ) / self.dt
        candidate_result = self.candidate_encoder(
            robot_current,
            robot_candidates,
            entity_history[:, -1],
            target_velocity,
            return_sequence=self.return_temporal_sequence,
        )
        reference_result = self.candidate_encoder(
            robot_current,
            reference_candidates,
            entity_history[:, -1],
            target_velocity,
            return_sequence=self.return_temporal_sequence,
        )
        if self.return_temporal_sequence:
            candidate, candidate_sequence = candidate_result
            reference, reference_sequence = reference_result
        else:
            candidate = candidate_result
            reference = reference_result
        if robot_actions is not None:
            if tuple(robot_actions.shape) != tuple(robot_mask.shape):
                raise ValueError("robot_actions must align with robot_mask")
            # Reuse the existing entity type embedding as a trainable mode
            # direction. This keeps trajectory-only checkpoint state dicts
            # backward compatible while exposing high-level modes explicitly.
            action_mask = robot_actions.to(dtype=candidate.dtype).unsqueeze(-1)
            candidate = candidate + action_mask * self.node_type_embedding.weight[1]
            if self.return_temporal_sequence:
                candidate_sequence = candidate_sequence + (
                    action_mask.unsqueeze(-2) * self.node_type_embedding.weight[1]
                )
        active_count = robot_mask.sum(dim=1).long().clamp(0, self.num_robots)
        output = {
            "scene": scene,
            "candidate": candidate,
            "reference": reference,
            "robot_mask": robot_mask,
            "active_count": active_count,
        }
        if self.return_temporal_sequence:
            output["candidate_sequence"] = candidate_sequence
            output["reference_sequence"] = reference_sequence
        return output
