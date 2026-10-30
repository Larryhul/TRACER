from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from tracer.data.heterogeneous_scene import HeterogeneousSceneBatch


RELATION_TO_ID: Dict[str, int] = {
    "R2R": 0,
    "R2N": 1,
    "N2R": 2,
    "N2N": 3,
}


@dataclass
class HGTEncoderConfig:
    state_dim: int = 2
    robot_state_dim: int = 4
    geometry_dim: int = 2
    hidden_dim: int = 128
    context_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    num_roles: int = 5
    use_role_embedding: bool = True
    use_typed_relations: bool = True
    use_goal_conditioning: bool = False
    goal_dim: int = 2
    use_robot_current_state: bool = True
    use_geometry_embedding: bool = True
    use_entity_entity_edges: bool = False
    use_robot_robot_edges: bool = True
    relation_attention_type: str = "hgt"
    history_encoder_type: str = "lstm"
    feature_fusion: str = "concat"
    relation_granularity: str = "semantic"


class CompactHGTLayer(nn.Module):
    """Legacy compact typed graph transformer layer for old checkpoints."""

    def __init__(self, hidden_dim: int, num_heads: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.relation_bias = nn.Embedding(num_relations, num_heads)
        self.relation_gate = nn.Embedding(num_relations, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        relation_ids: torch.Tensor,
        agent_mask: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, a, d = h.shape
        q = self.q(h).view(b, a, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(b, a, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(b, a, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        rel_bias = self.relation_bias(relation_ids).permute(0, 3, 1, 2)
        logits = logits + rel_bias

        valid = edge_mask if edge_mask is not None else (agent_mask[:, :, None] & agent_mask[:, None, :])
        valid_h = valid[:, None, :, :]
        logits = logits.masked_fill(~valid_h, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        attn = torch.where(valid_h, attn, torch.zeros_like(attn))

        msg = torch.matmul(self.dropout(attn), v).transpose(1, 2).contiguous().view(b, a, d)
        gate = torch.sigmoid(self.relation_gate(relation_ids).mean(dim=2))
        h = self.norm1(h + self.dropout(self.out(msg * gate)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h * agent_mask[..., None].to(h.dtype)


class HGTLayer(nn.Module):
    """Relation-specific HGT layer with directed meta-relation projections."""

    def __init__(self, hidden_dim: int, num_heads: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.head_dim = hidden_dim // num_heads
        self.q = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)])
        self.k = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)])
        self.v = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)])
        self.relation_bias = nn.Embedding(num_relations, num_heads)
        self.relation_gate = nn.Embedding(num_relations, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _project(self, layer: nn.Linear, h: torch.Tensor) -> torch.Tensor:
        b, a, _ = h.shape
        return layer(h).view(b, a, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        h: torch.Tensor,
        relation_ids: torch.Tensor,
        agent_mask: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, a, d = h.shape
        valid = edge_mask if edge_mask is not None else (agent_mask[:, :, None] & agent_mask[:, None, :])
        valid_h = valid[:, None, :, :]
        min_value = torch.finfo(h.dtype).min
        logits = torch.full((b, self.num_heads, a, a), min_value, device=h.device, dtype=h.dtype)
        values: list[torch.Tensor] = []

        for rel_id in range(self.num_relations):
            rel_mask = relation_ids == rel_id
            q = self._project(self.q[rel_id], h)
            k = self._project(self.k[rel_id], h)
            v = self._project(self.v[rel_id], h)
            values.append(v)
            rel_logits = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            rel_logits = rel_logits + self.relation_bias.weight[rel_id].view(1, self.num_heads, 1, 1)
            logits = torch.where(rel_mask[:, None, :, :], rel_logits, logits)

        logits = logits.masked_fill(~valid_h, min_value)
        attn = torch.softmax(logits, dim=-1)
        attn = torch.where(valid_h, attn, torch.zeros_like(attn))
        attn = self.dropout(attn)

        msg = torch.zeros((b, self.num_heads, a, self.head_dim), device=h.device, dtype=h.dtype)
        gate_sum = torch.zeros((b, a, d), device=h.device, dtype=h.dtype)
        gate_count = torch.zeros((b, a, 1), device=h.device, dtype=h.dtype)
        for rel_id, v in enumerate(values):
            rel_mask = (relation_ids == rel_id) & valid
            rel_attn = attn * rel_mask[:, None, :, :].to(attn.dtype)
            msg = msg + torch.matmul(rel_attn, v)
            target_has_rel = rel_mask.any(dim=2)
            gate = self.relation_gate.weight[rel_id].view(1, 1, d)
            gate_sum = gate_sum + target_has_rel[..., None].to(h.dtype) * gate
            gate_count = gate_count + target_has_rel[..., None].to(h.dtype)

        msg = msg.transpose(1, 2).contiguous().view(b, a, d)
        gate = torch.sigmoid(gate_sum / gate_count.clamp_min(1.0))
        h = self.norm1(h + self.dropout(self.out(msg * gate)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h * agent_mask[..., None].to(h.dtype)


class HeterogeneousGraphEncoder(nn.Module):
    """Phase 1: heterogeneous scene encoding."""

    def __init__(self, config: HGTEncoderConfig) -> None:
        super().__init__()
        self.config = config
        encoder_type = str(config.history_encoder_type).lower()
        if encoder_type not in {"gru", "lstm"}:
            raise ValueError(f"unknown history_encoder_type={config.history_encoder_type!r}")
        encoder_cls = nn.LSTM if encoder_type == "lstm" else nn.GRU
        self.history_encoder = encoder_cls(
            input_size=config.state_dim,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.robot_state_encoder = None
        if config.use_robot_current_state:
            self.robot_state_encoder = nn.Sequential(
                nn.Linear(config.robot_state_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
        self.role_embedding = nn.Embedding(config.num_roles, config.hidden_dim)
        self.control_embedding = nn.Embedding(2, config.hidden_dim)
        self.geometry_embedding = None
        if config.use_geometry_embedding:
            self.geometry_embedding = nn.Sequential(
                nn.Linear(config.geometry_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
        self.goal_embedding = None
        if config.use_goal_conditioning:
            self.goal_embedding = nn.Sequential(
                nn.Linear(config.goal_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )

        self.feature_fusion = str(config.feature_fusion).lower()
        if self.feature_fusion not in {"sum", "concat"}:
            raise ValueError(f"unknown feature_fusion={config.feature_fusion!r}")
        self.feature_fusion_proj = None
        if self.feature_fusion == "concat":
            component_count = 2
            component_count += int(config.use_role_embedding)
            component_count += int(config.use_geometry_embedding)
            component_count += int(config.use_goal_conditioning)
            self.feature_fusion_proj = nn.Sequential(
                nn.LayerNorm(component_count * config.hidden_dim),
                nn.Linear(component_count * config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )

        layer_type = str(config.relation_attention_type).lower()
        layer_cls = CompactHGTLayer if layer_type in {"compact", "legacy"} else HGTLayer
        self.num_relations = self._num_relations()
        self.layers = nn.ModuleList(
            [
                layer_cls(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    num_relations=self.num_relations,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.context_dim),
            nn.GELU(),
            nn.Linear(config.context_dim, config.context_dim),
        )

    def _num_relations(self) -> int:
        if not self.config.use_typed_relations:
            return 1
        granularity = str(self.config.relation_granularity).lower()
        if granularity in {"semantic", "role", "role_pair", "role-pair"}:
            return int(self.config.num_roles) * int(self.config.num_roles)
        if granularity in {"control", "direction", "edge"}:
            return len(RELATION_TO_ID)
        raise ValueError(f"unknown relation_granularity={self.config.relation_granularity!r}")

    def _relation_ids(self, batch: HeterogeneousSceneBatch) -> torch.Tensor:
        rel = torch.zeros(
            (batch.batch_size, batch.num_agents, batch.num_agents),
            device=batch.roles.device,
            dtype=torch.long,
        )
        if not self.config.use_typed_relations:
            return rel

        granularity = str(self.config.relation_granularity).lower()
        if granularity in {"semantic", "role", "role_pair", "role-pair"}:
            roles = batch.roles.clamp(min=0, max=self.config.num_roles - 1)
            target_role = roles[:, :, None]
            source_role = roles[:, None, :]
            return source_role * int(self.config.num_roles) + target_role

        rel = torch.full_like(rel, RELATION_TO_ID["N2N"])
        target_robot = batch.robot_mask[:, :, None]
        source_robot = batch.robot_mask[:, None, :]
        rel = torch.where(source_robot & target_robot, torch.full_like(rel, RELATION_TO_ID["R2R"]), rel)
        rel = torch.where(source_robot & ~target_robot, torch.full_like(rel, RELATION_TO_ID["R2N"]), rel)
        rel = torch.where(~source_robot & target_robot, torch.full_like(rel, RELATION_TO_ID["N2R"]), rel)
        return rel

    def _edge_mask(self, batch: HeterogeneousSceneBatch) -> torch.Tensor:
        valid = batch.agent_mask[:, :, None] & batch.agent_mask[:, None, :]
        if self.config.use_entity_entity_edges:
            edge_mask = valid
        else:
            target_robot = batch.robot_mask[:, :, None]
            source_robot = batch.robot_mask[:, None, :]
            edge_mask = valid & (target_robot | source_robot)

        if not self.config.use_robot_robot_edges and batch.num_agents > 0:
            target_robot = batch.robot_mask[:, :, None]
            source_robot = batch.robot_mask[:, None, :]
            rr_edge = target_robot & source_robot
            eye = torch.eye(
                batch.num_agents,
                dtype=torch.bool,
                device=batch.robot_mask.device,
            ).view(1, batch.num_agents, batch.num_agents)
            edge_mask = edge_mask & (~rr_edge | eye)
        return edge_mask

    def _robot_state_features(self, batch: HeterogeneousSceneBatch) -> torch.Tensor:
        xy = batch.current_pos
        if batch.history.shape[2] >= 2:
            velocity = (batch.history[:, :, -1, :2] - batch.history[:, :, -2, :2]) / float(batch.dt)
        else:
            velocity = torch.zeros_like(xy)
        state = torch.cat([xy, velocity], dim=-1)
        if state.shape[-1] > self.config.robot_state_dim:
            state = state[..., : self.config.robot_state_dim]
        elif state.shape[-1] < self.config.robot_state_dim:
            pad = torch.zeros(
                *state.shape[:-1],
                self.config.robot_state_dim - state.shape[-1],
                device=state.device,
                dtype=state.dtype,
            )
            state = torch.cat([state, pad], dim=-1)
        return state

    def _geometry_features(self, batch: HeterogeneousSceneBatch) -> torch.Tensor:
        if batch.geometry is None:
            geometry = torch.zeros(
                batch.batch_size,
                batch.num_agents,
                self.config.geometry_dim,
                device=batch.history.device,
                dtype=batch.history.dtype,
            )
        else:
            geometry = batch.geometry.to(device=batch.history.device, dtype=batch.history.dtype)
            if geometry.shape[-1] > self.config.geometry_dim:
                geometry = geometry[..., : self.config.geometry_dim]
            elif geometry.shape[-1] < self.config.geometry_dim:
                pad = torch.zeros(
                    *geometry.shape[:-1],
                    self.config.geometry_dim - geometry.shape[-1],
                    device=geometry.device,
                    dtype=geometry.dtype,
                )
                geometry = torch.cat([geometry, pad], dim=-1)
        return geometry

    def _goal_features(self, batch: HeterogeneousSceneBatch) -> torch.Tensor:
        goal_delta = batch.goal_displacement()
        if goal_delta.shape[-1] > self.config.goal_dim:
            goal_delta = goal_delta[..., : self.config.goal_dim]
        elif goal_delta.shape[-1] < self.config.goal_dim:
            pad = torch.zeros(
                *goal_delta.shape[:-1],
                self.config.goal_dim - goal_delta.shape[-1],
                device=goal_delta.device,
                dtype=goal_delta.dtype,
            )
            goal_delta = torch.cat([goal_delta, pad], dim=-1)
        return goal_delta

    def _encode_history(self, hist: torch.Tensor) -> torch.Tensor:
        _, hidden = self.history_encoder(hist)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden[-1]

    def _fuse_features(self, components: list[torch.Tensor]) -> torch.Tensor:
        if self.feature_fusion == "sum":
            node = components[0]
            for component in components[1:]:
                node = node + component
            return node
        if self.feature_fusion_proj is None:
            raise RuntimeError("feature_fusion_proj is required for concat fusion")
        return self.feature_fusion_proj(torch.cat(components, dim=-1))

    def forward(self, batch: HeterogeneousSceneBatch) -> tuple[torch.Tensor, torch.Tensor]:
        batch.validate()
        b, a, h, f = batch.history.shape
        if f < self.config.state_dim:
            raise ValueError(f"history has {f} channels, but state_dim={self.config.state_dim}")

        hist = batch.history[..., : self.config.state_dim].reshape(b * a, h, self.config.state_dim)
        history_node = self._encode_history(hist).view(b, a, self.config.hidden_dim)
        motion_node = history_node
        if self.robot_state_encoder is not None:
            robot_node = self.robot_state_encoder(self._robot_state_features(batch))
            motion_node = torch.where(batch.robot_mask[..., None], robot_node, history_node)

        components = [motion_node]
        if self.config.use_role_embedding:
            components.append(self.role_embedding(batch.roles.clamp(min=0, max=self.config.num_roles - 1)))
        components.append(self.control_embedding(batch.robot_mask.long()))
        if self.geometry_embedding is not None:
            if batch.geometry is None and self.feature_fusion == "concat":
                components.append(torch.zeros_like(motion_node))
            elif batch.geometry is not None:
                components.append(self.geometry_embedding(self._geometry_features(batch)))
        if self.goal_embedding is not None:
            components.append(self.goal_embedding(self._goal_features(batch)))
        node = self._fuse_features(components)
        node = node * batch.agent_mask[..., None].to(node.dtype)

        rel = self._relation_ids(batch)
        edge_mask = self._edge_mask(batch)
        for layer in self.layers:
            node = layer(node, rel, batch.agent_mask, edge_mask=edge_mask)

        denom = batch.agent_mask.sum(dim=1, keepdim=True).clamp_min(1).to(node.dtype)
        graph = (node * batch.agent_mask[..., None].to(node.dtype)).sum(dim=1) / denom
        context = self.context_proj(graph)
        return node, F.normalize(context, dim=-1)
