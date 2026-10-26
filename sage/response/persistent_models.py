from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from sage.response.tracer import TRACERFeatureEncoder
from sage.response.static_context import segment_context_features


@dataclass(frozen=True)
class PersistentTrajectoryModelConfig:
    history: int = 8
    horizon: int = 12
    num_robots: int = 2
    num_modes: int = 3
    hidden_dim: int = 96
    dropout: float = 0.05
    min_log_std: float = -4.0
    max_log_std: float = 1.5
    dt: float = 0.4
    # ``legacy`` keeps old checkpoints loadable; all new formal runs set
    # ``architecture: tracer`` in their frozen config.
    architecture: str = "legacy"
    num_heads: int = 4
    graph_layers: int = 2
    prefix_loss_weight: float = 1.0
    conditioning: str = "discrete"
    coordinate_min: float = 0.0
    coordinate_max: float = 2.0
    evidence_conditioning: bool = False
    geometry_features: bool = False
    dynamic_interaction_features: bool = False
    temporal_pooling: str = "mean"
    candidate_uncertainty: bool = False
    set_interactions: bool = False
    evidence_pooling: str = "mean"
    evidence_prefix_steps: int = 3
    response_delay_steps: int = 0
    static_geometry_features: bool = False
    joint_uncertainty_decoder: bool = False
    zero_init_interaction_residual: bool = False
    observed_velocity_anchor: bool = False
    factual_prefix_dynamics: bool = False
    mode_speed_values: tuple[float, ...] = ()
    robot_positions_target_relative: bool = False
    mode_response_gate: bool = False


def stationary_reference_candidates(
    robot_current: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Build a feasible hold-position reference action for each robot slot.

    The current relative position is preserved while future velocities are set
    to zero, so response deltas are centered on an action reference rather than
    on a zeroed geometry vector.
    """

    if robot_current.ndim < 2 or robot_current.shape[-1] != 4:
        raise ValueError("robot_current must have shape (..., robots, 4)")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    position = robot_current[..., None, :2].expand(
        *robot_current.shape[:-1], int(horizon), 2
    )
    velocity = torch.zeros_like(position)
    return torch.cat([position, velocity], dim=-1)


class BeliefConditionedTeamResponseModel(nn.Module):
    """Mode-conditioned joint response model used as a belief-weighted mixture."""

    def __init__(self, config: PersistentTrajectoryModelConfig) -> None:
        super().__init__()
        self.config = config
        conditioning = str(config.conditioning).lower()
        if conditioning not in {"discrete", "continuous"}:
            raise ValueError("conditioning must be 'discrete' or 'continuous'")
        if config.coordinate_max <= config.coordinate_min:
            raise ValueError("coordinate_max must exceed coordinate_min")
        architecture = str(config.architecture).lower()
        if architecture not in {"legacy", "tracer"}:
            raise ValueError("architecture must be 'legacy' or 'tracer'")
        if not 0 <= config.response_delay_steps <= config.horizon:
            raise ValueError("response_delay_steps must lie within the prediction horizon")
        if architecture != "tracer" and config.response_delay_steps:
            raise ValueError("response delay requires architecture='tracer'")
        if architecture != "tracer" and config.static_geometry_features:
            raise ValueError("static geometry requires architecture='tracer'")
        if config.joint_uncertainty_decoder and (architecture != "tracer" or config.candidate_uncertainty):
            raise ValueError("joint uncertainty requires tracer architecture without additive candidate uncertainty")
        if architecture == "legacy" and conditioning == "continuous":
            raise ValueError("continuous conditioning requires architecture='tracer'")
        if config.mode_response_gate and (architecture != "tracer" or conditioning != "discrete"):
            raise ValueError("mode response gating requires discrete tracer conditioning")
        if (config.observed_velocity_anchor or config.factual_prefix_dynamics or config.mode_speed_values) and architecture != "tracer":
            raise ValueError("factual response conditioning requires architecture='tracer'")
        if config.factual_prefix_dynamics and (config.response_delay_steps < 1 or config.history < 3):
            raise ValueError("factual prefix dynamics requires a positive response delay and three history samples")
        if config.mode_speed_values and (
            len(config.mode_speed_values) != config.num_modes
            or any(not math.isfinite(value) or value <= 0 for value in config.mode_speed_values)
            or any(second <= first for first, second in zip(config.mode_speed_values, config.mode_speed_values[1:]))
        ):
            raise ValueError("mode speed values must be finite, positive, ordered training-bin centers")
        if architecture == "tracer":
            self._init_tracer()
            return
        hidden = config.hidden_dim
        robot_input_dim = 4 + config.horizon * 4
        pair_hidden = hidden // 2
        self.history_encoder = nn.GRU(input_size=4, hidden_size=hidden, batch_first=True)
        self.mode_embedding = nn.Embedding(config.num_modes, hidden)
        self.base_decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.horizon * 2),
        )
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.main_decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, config.horizon * 2),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(robot_input_dim * 3, pair_hidden),
            nn.SiLU(),
            nn.Linear(pair_hidden, pair_hidden),
        )
        self.pair_decoder = nn.Sequential(
            nn.Linear(hidden + pair_hidden, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.horizon * 2),
        )
        self.log_std = nn.Parameter(torch.full((config.num_modes, config.horizon, 2), -1.5))

    def _init_tracer(self) -> None:
        config = self.config
        hidden = config.hidden_dim
        self.tracer_encoder = TRACERFeatureEncoder(
            hidden_dim=hidden,
            num_heads=config.num_heads,
            graph_layers=config.graph_layers,
            dropout=config.dropout,
            dt=config.dt,
            num_robots=config.num_robots,
            geometry_features=config.geometry_features,
            dynamic_interaction_features=config.dynamic_interaction_features,
            temporal_pooling=config.temporal_pooling,
        )
        if config.mode_response_gate:
            self.factual_mode_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, config.horizon))
            self.candidate_mode_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, config.horizon))
            nn.init.constant_(self.factual_mode_gate[-1].bias, -2.)
            nn.init.constant_(self.candidate_mode_gate[-1].bias, -2.)
        if str(config.conditioning).lower() == "continuous":
            self.coordinate_encoder = nn.Sequential(
                nn.Linear(1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.coordinate_prior_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2),
            )
        else:
            self.mode_embedding = nn.Embedding(config.num_modes, hidden)
        if config.mode_speed_values:
            self.speed_condition_encoder = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        if config.factual_prefix_dynamics:
            self.factual_prefix_mean = nn.Sequential(
                nn.Linear(hidden * 2 + 8, hidden), nn.GELU(), nn.Linear(hidden, config.response_delay_steps * 2),
            )
            self.factual_prefix_log_std = nn.Sequential(
                nn.Linear(hidden + 8, hidden), nn.GELU(), nn.Linear(hidden, config.response_delay_steps * 2),
            )
            self.preferred_speed_decoder = nn.Sequential(nn.Linear(hidden + 8, hidden), nn.GELU(), nn.Linear(hidden, 1))
        if config.evidence_conditioning:
            if config.evidence_pooling not in {"mean", "attention"}:
                raise ValueError("evidence_pooling must be 'mean' or 'attention'")
            if not 1 <= config.evidence_prefix_steps <= config.horizon:
                raise ValueError("evidence_prefix_steps must lie within the prediction horizon")
            if config.evidence_pooling == "attention":
                self.evidence_token = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, hidden))
                self.evidence_time_encoder = nn.GRU(hidden, hidden, batch_first=True)
                self.evidence_robot_score = nn.Linear(hidden, 1)
                self.evidence_window_projection = nn.Linear(hidden, 6)
            self.evidence_encoder = nn.GRU(
                input_size=6, hidden_size=hidden, batch_first=True
            )
            self.evidence_projection = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )
        self.active_count_embedding = nn.Embedding(config.num_robots + 1, hidden)
        self.base_decoder = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, config.horizon * 2),
        )
        self.single_decoder = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, config.horizon * 2),
        )
        pair_hidden = max(hidden // 2, config.num_heads)
        self.pair_encoder = nn.Sequential(
            nn.Linear(hidden * 3, pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, pair_hidden),
        )
        self.pair_decoder = nn.Sequential(
            nn.Linear(hidden * 2 + pair_hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, config.horizon * 2),
        )
        if config.zero_init_interaction_residual:
            nn.init.zeros_(self.pair_decoder[-1].weight)
            nn.init.zeros_(self.pair_decoder[-1].bias)
        self.log_std_decoder = nn.Sequential(
            nn.Linear(hidden * (4 if config.joint_uncertainty_decoder else 3), hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, config.horizon * 2),
        )
        if config.static_geometry_features:
            self.static_encoder = nn.Sequential(nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        if config.candidate_uncertainty:
            self.candidate_std_decoder = nn.Sequential(
                nn.Linear(hidden * 4, hidden), nn.GELU(),
                nn.Linear(hidden, config.horizon * 2),
            )
        if config.set_interactions:
            self.set_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    hidden, config.num_heads, hidden * 2, config.dropout,
                    activation="gelu", batch_first=True,
                ), num_layers=1,
            )
            self.set_decoder = nn.Sequential(
                nn.Linear(hidden * 3, hidden), nn.GELU(),
                nn.Linear(hidden, config.horizon * 2),
            )
            if config.zero_init_interaction_residual:
                nn.init.zeros_(self.set_decoder[-1].weight)
                nn.init.zeros_(self.set_decoder[-1].bias)

    def _prepare_tracer_inputs(
        self,
        robot_current: torch.Tensor,
        robot_candidates: torch.Tensor,
        robot_mask: torch.Tensor | None,
        reference_candidates: torch.Tensor | None,
        batch: int,
        windows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if robot_current.ndim != 4 or robot_current.shape[-1] != 4:
            raise ValueError("robot_current must have shape (batch, windows, robots, 4)")
        if robot_candidates.ndim != 5 or robot_candidates.shape[-1] != 4:
            raise ValueError(
                "robot_candidates must have shape (batch, windows, robots, horizon, 4)"
            )
        if tuple(robot_current.shape[:2]) != (batch, windows):
            raise ValueError("robot_current must align with entity_history batch/windows")
        if tuple(robot_candidates.shape[:3]) != tuple(robot_current.shape[:3]):
            raise ValueError("robot_candidates must align with robot_current")
        if int(robot_candidates.shape[-2]) != self.config.horizon:
            raise ValueError("robot_candidates horizon does not match the model config")
        supplied = int(robot_current.shape[2])
        if supplied > self.config.num_robots:
            raise ValueError("received more robot slots than config.num_robots")
        if robot_mask is None:
            robot_mask = torch.ones(
                batch, windows, supplied, device=robot_current.device, dtype=torch.bool
            )
        else:
            if tuple(robot_mask.shape) != (batch, windows, supplied):
                raise ValueError("robot_mask must have shape (batch, windows, robots)")
            robot_mask = robot_mask.to(device=robot_current.device, dtype=torch.bool)
        if reference_candidates is None:
            reference_candidates = stationary_reference_candidates(
                robot_current, self.config.horizon
            )
        else:
            if tuple(reference_candidates.shape) != tuple(robot_candidates.shape):
                raise ValueError("reference_candidates must match robot_candidates shape")
            reference_candidates = reference_candidates.to(
                device=robot_current.device, dtype=robot_current.dtype
            )
        pad = self.config.num_robots - supplied
        if pad:
            robot_current = torch.cat(
                [robot_current, robot_current.new_zeros(batch, windows, pad, 4)], dim=2
            )
            robot_candidates = torch.cat(
                [
                    robot_candidates,
                    robot_candidates.new_zeros(batch, windows, pad, self.config.horizon, 4),
                ],
                dim=2,
            )
            reference_candidates = torch.cat(
                [
                    reference_candidates,
                    reference_candidates.new_zeros(
                        batch, windows, pad, self.config.horizon, 4
                    ),
                ],
                dim=2,
            )
            robot_mask = torch.cat(
                [
                    robot_mask,
                    torch.zeros(
                        batch, windows, pad, device=robot_mask.device, dtype=torch.bool
                    ),
                ],
                dim=2,
            )
        return robot_current, robot_candidates, reference_candidates, robot_mask

    def _tracer_pair_features(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.cat([first + second, (first - second).abs(), first * second], dim=-1)

    def _forward_tracer(
        self,
        entity_history: torch.Tensor,
        robot_current: torch.Tensor,
        robot_candidates: torch.Tensor,
        include_pair_interactions: bool,
        robot_mask: torch.Tensor | None,
        reference_candidates: torch.Tensor | None,
        neighbor_history: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
        condition_coordinates: torch.Tensor | None,
        evidence_candidates: torch.Tensor | None,
        evidence_responses: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        static_segments: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batch, windows = entity_history.shape[:2]
        if self.config.robot_positions_target_relative:
            origin = entity_history[..., -1:, :]
            entity_history = entity_history - origin
            if neighbor_history is not None:
                neighbor_history = neighbor_history - origin.unsqueeze(-3)
        current, candidates, reference, mask = self._prepare_tracer_inputs(
            robot_current,
            robot_candidates,
            robot_mask,
            reference_candidates,
            int(batch),
            int(windows),
        )
        flat_history = entity_history.reshape(-1, self.config.history, 2)
        flat_current = current.reshape(-1, self.config.num_robots, 4)
        flat_candidates = candidates.reshape(
            -1, self.config.num_robots, self.config.horizon, 4
        )
        flat_reference = reference.reshape(
            -1, self.config.num_robots, self.config.horizon, 4
        )
        flat_mask = mask.reshape(-1, self.config.num_robots)
        flat_neighbors = None
        flat_neighbor_mask = None
        if neighbor_history is not None:
            if neighbor_history.ndim != 5 or tuple(neighbor_history.shape[:2]) != (batch, windows):
                raise ValueError(
                    "neighbor_history must have shape (batch, windows, neighbors, history, 2)"
                )
            flat_neighbors = neighbor_history.reshape(
                -1, neighbor_history.shape[2], self.config.history, 2
            )
            if neighbor_mask is not None:
                if tuple(neighbor_mask.shape[:2]) != (batch, windows):
                    raise ValueError("neighbor_mask must align with entity_history")
                flat_neighbor_mask = neighbor_mask.reshape(-1, neighbor_mask.shape[-1])
        encoded = self.tracer_encoder(
            flat_history,
            flat_current,
            flat_candidates,
            flat_reference,
            flat_mask,
            flat_neighbors,
            flat_neighbor_mask,
        )
        if self.config.static_geometry_features:
            if static_segments is None or static_segments.ndim != 4 or tuple(static_segments.shape[:2]) != (batch, windows):
                raise ValueError("map-conditioned responses require static_segments shaped (batch, windows, segments, 4)")
            features = segment_context_features(static_segments.to(flat_history)).reshape(batch * windows, -1, 8)
            if features.shape[1]:
                weights = (-features[..., -1]).softmax(dim=1)
                encoded["scene"] = encoded["scene"] + (self.static_encoder(features) * weights[..., None]).sum(dim=1)
        if self.config.evidence_conditioning:
            encoded["scene"] = encoded["scene"] + self._evidence_context(
                evidence_candidates,
                evidence_responses,
                evidence_mask,
                int(batch),
                int(windows),
                entity_history.device,
                entity_history.dtype,
            )
        conditioning = str(self.config.conditioning).lower()
        if conditioning == "continuous":
            if condition_coordinates is None:
                if not self.config.mode_speed_values:
                    raise ValueError("continuous conditioning requires condition_coordinates or mode speed values")
                condition_coordinates = entity_history.new_tensor(self.config.mode_speed_values).reshape(1, 1, -1).expand(batch, windows, -1)
            if condition_coordinates.ndim == 2:
                condition_coordinates = condition_coordinates.unsqueeze(-1)
            if condition_coordinates.ndim != 3 or tuple(condition_coordinates.shape[:2]) != (batch, windows):
                raise ValueError(
                    "condition_coordinates must have shape (batch, windows, conditions)"
                )
            condition_count = int(condition_coordinates.shape[-1])
            if condition_count < 1:
                raise ValueError("condition_coordinates must contain at least one value")
            coordinates = condition_coordinates.to(
                device=entity_history.device, dtype=entity_history.dtype
            ).clamp(self.config.coordinate_min, self.config.coordinate_max)
            normalized = (
                2.0 * (coordinates - self.config.coordinate_min)
                / (self.config.coordinate_max - self.config.coordinate_min)
                - 1.0
            )
            mode = self.coordinate_encoder(normalized.reshape(-1, 1)).reshape(
                flat_history.shape[0], condition_count, -1
            )
        else:
            condition_count = self.config.num_modes
            mode_ids = torch.arange(self.config.num_modes, device=entity_history.device)
            mode = self.mode_embedding(mode_ids)[None].expand(flat_history.shape[0], -1, -1)
            if self.config.mode_speed_values:
                speeds = flat_history.new_tensor(self.config.mode_speed_values)[:, None]
                mode = mode + self.speed_condition_encoder(speeds)[None]
        count = self.active_count_embedding(encoded["active_count"])
        coordinate_prior = None
        if conditioning == "continuous":
            prior = self.coordinate_prior_head(encoded["scene"])
            prior_mean = self.config.coordinate_min + (
                torch.sigmoid(prior[..., 0])
                * (self.config.coordinate_max - self.config.coordinate_min)
            )
            prior_log_std = (-2.0 + F.softplus(prior[..., 1])).clamp(-3.0, 0.5)
            coordinate_prior = torch.stack([prior_mean, prior_log_std], dim=-1).reshape(
                batch, windows, 2
            )
        common = torch.cat(
            [encoded["scene"][:, None].expand(-1, condition_count, -1), mode, count[:, None].expand(-1, condition_count, -1)],
            dim=-1,
        )
        mean = self.base_decoder(common).reshape(
            -1, condition_count, self.config.horizon, 2
        )
        observed_velocity = (flat_history[:, -1] - flat_history[:, -2]) / self.config.dt
        if self.config.observed_velocity_anchor:
            mean = mean + observed_velocity[:, None, None]
        preferred_speed_mean = None
        factual_prefix_std = None
        if self.config.factual_prefix_dynamics:
            weights = flat_mask.to(flat_current.dtype)[..., None]
            factual_robot_state = (flat_current * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            previous_velocity = (flat_history[:, -2] - flat_history[:, -3]) / self.config.dt
            factual_context = torch.cat([encoded["scene"], observed_velocity, previous_velocity, factual_robot_state], -1)
            factual_mode_context = torch.cat([factual_context[:, None].expand(-1, condition_count, -1), mode], -1)
            prefix_mean = self.factual_prefix_mean(factual_mode_context).reshape(-1, condition_count, self.config.response_delay_steps, 2)
            prefix_mean = prefix_mean + observed_velocity[:, None, None]
            mean = torch.cat([prefix_mean, mean[..., self.config.response_delay_steps:, :]], -2)
            factual_prefix_std = self.factual_prefix_log_std(factual_context).reshape(-1, 1, self.config.response_delay_steps, 2)
            preferred_speed_mean = F.softplus(self.preferred_speed_decoder(factual_context)).reshape(batch, windows)
        base_mean = mean
        candidate = encoded["candidate"]
        reference_tokens = encoded["reference"]
        action_context = torch.cat(
            [
                encoded["scene"][:, None, None].expand(-1, condition_count, self.config.num_robots, -1),
                mode[:, :, None].expand(-1, -1, self.config.num_robots, -1),
            ],
            dim=-1,
        )
        candidate_context = torch.cat([action_context, candidate[:, None].expand(-1, condition_count, -1, -1)], dim=-1)
        reference_context = torch.cat([action_context, reference_tokens[:, None].expand(-1, condition_count, -1, -1)], dim=-1)
        single = self.single_decoder(candidate_context).reshape(
            -1, condition_count, self.config.num_robots, self.config.horizon, 2
        )
        reference_single = self.single_decoder(reference_context).reshape_as(single)
        mean = mean + ((single - reference_single) * flat_mask[:, None, :, None, None].to(mean.dtype)).sum(dim=2)

        if include_pair_interactions:
            for first in range(self.config.num_robots):
                for second in range(first + 1, self.config.num_robots):
                    pair_context = self._tracer_pair_features(candidate[:, first], candidate[:, second])
                    pair_first_ref = self._tracer_pair_features(candidate[:, first], reference_tokens[:, second])
                    pair_second_ref = self._tracer_pair_features(reference_tokens[:, first], candidate[:, second])
                    pair_reference = self._tracer_pair_features(reference_tokens[:, first], reference_tokens[:, second])

                    def decode(features: torch.Tensor) -> torch.Tensor:
                        encoded_pair = self.pair_encoder(features)
                        pair_input = torch.cat(
                            [
                                encoded["scene"][:, None].expand(-1, condition_count, -1),
                                mode,
                                encoded_pair[:, None].expand(-1, condition_count, -1),
                            ],
                            dim=-1,
                        )
                        return self.pair_decoder(pair_input)

                    valid_pair = (flat_mask[:, first] & flat_mask[:, second])[:, None, None, None].to(mean.dtype)
                    mean = mean + (
                        decode(pair_context)
                        - decode(pair_first_ref)
                        - decode(pair_second_ref)
                        + decode(pair_reference)
                    ).reshape(-1, condition_count, self.config.horizon, 2) * valid_pair

        if include_pair_interactions and self.config.set_interactions:
            padding = ~flat_mask
            padding = padding.clone()
            padding[~flat_mask.any(dim=-1), 0] = False
            token_mask = flat_mask[..., None].to(candidate.dtype)
            count_denominator = token_mask.sum(dim=1).clamp_min(1.0)

            def set_response(tokens: torch.Tensor) -> torch.Tensor:
                context = self.set_encoder(tokens, src_key_padding_mask=padding)
                pooled = (context * token_mask).sum(dim=1) / count_denominator
                features = torch.cat([
                    encoded["scene"][:, None].expand(-1, condition_count, -1),
                    mode, pooled[:, None].expand(-1, condition_count, -1),
                ], dim=-1)
                return self.set_decoder(features).reshape(-1, condition_count, self.config.horizon, 2)

            higher_order = (flat_mask.sum(dim=-1) >= 3)[:, None, None, None]
            mean = mean + (set_response(candidate) - set_response(reference_tokens)) * higher_order

        delay = self.config.response_delay_steps
        if delay:
            mean = torch.cat([base_mean[..., :delay, :], mean[..., delay:, :]], dim=-2)
        if self.config.joint_uncertainty_decoder:
            weights = flat_mask[..., None].to(candidate.dtype)
            pooled_candidate = (candidate * weights).sum(1) / weights.sum(1).clamp_min(1.)
            pooled_reference = (reference_tokens * weights).sum(1) / weights.sum(1).clamp_min(1.)
            std_features = self.log_std_decoder(torch.cat([common, pooled_candidate[:, None].expand(-1, condition_count, -1)], -1))
            base_std_features = self.log_std_decoder(torch.cat([common, pooled_reference[:, None].expand(-1, condition_count, -1)], -1))
        else:
            std_features = self.log_std_decoder(common)
            base_std_features = std_features
        if self.config.candidate_uncertainty:
            weights = flat_mask[..., None].to(candidate.dtype)
            pooled_candidate = (candidate * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            std_features = std_features + self.candidate_std_decoder(torch.cat([
                common, pooled_candidate[:, None].expand(-1, condition_count, -1),
            ], dim=-1))
        log_std = std_features.reshape(
            -1, condition_count, self.config.horizon, 2
        )
        if delay:
            base_log_std = base_std_features.reshape_as(log_std)
            log_std = torch.cat([base_log_std[..., :delay, :], log_std[..., delay:, :]], dim=-2)
        if factual_prefix_std is not None:
            log_std = torch.cat([factual_prefix_std.expand(-1, condition_count, -1, -1), log_std[..., delay:, :]], dim=-2)
        if self.config.joint_uncertainty_decoder:
            log_std = log_std.clamp(self.config.min_log_std, self.config.max_log_std)
        else:
            log_std = (self.config.min_log_std + F.softplus(log_std)).clamp(
                self.config.min_log_std, self.config.max_log_std
            )
        mode_gate_logits = None
        if self.config.mode_response_gate:
            weights = flat_mask[..., None].to(candidate.dtype)
            pooled_candidate = (candidate * weights).sum(1) / weights.sum(1).clamp_min(1.)
            mode_gate_logits = self.candidate_mode_gate(torch.cat([encoded["scene"], pooled_candidate], -1))
            if delay:
                factual_gate_logits = self.factual_mode_gate(encoded["scene"])
                mode_gate_logits = torch.cat([factual_gate_logits[:, :delay], mode_gate_logits[:, delay:]], -1)
            gate = mode_gate_logits.sigmoid()[:, None, :, None]
            central_mean = mean.mean(1, keepdim=True)
            mean = central_mean + gate * (mean - central_mean)
            central_log_std = log_std.mean(1, keepdim=True)
            log_std = central_log_std + gate * (log_std - central_log_std)
        return {
            "mean": mean.reshape(batch, windows, condition_count, self.config.horizon, 2),
            "log_std": log_std.reshape(batch, windows, condition_count, self.config.horizon, 2),
            "active_robot_count": encoded["active_count"].reshape(batch, windows),
            "condition_coordinates": condition_coordinates
            if conditioning == "continuous" else None,
            "coordinate_prior": coordinate_prior,
            "preferred_speed_mean": preferred_speed_mean,
            "mode_gate_logits": mode_gate_logits.reshape(batch, windows, self.config.horizon)
            if mode_gate_logits is not None else None,
        }

    def _entity_context(self, history: torch.Tensor) -> torch.Tensor:
        current = history[:, -1:, :]
        relative = history - current
        velocity = torch.zeros_like(relative)
        velocity[:, 1:] = (history[:, 1:] - history[:, :-1]) / self.config.dt
        _, state = self.history_encoder(torch.cat([relative, velocity], dim=-1))
        return state[-1]

    def _evidence_context(
        self,
        evidence_candidates: torch.Tensor | None,
        evidence_responses: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        batch: int,
        windows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        hidden = self.config.hidden_dim
        if evidence_candidates is None and evidence_responses is None:
            return torch.zeros(batch * windows, hidden, device=device, dtype=dtype)
        if evidence_candidates is None or evidence_responses is None or evidence_mask is None:
            raise ValueError("evidence candidates, responses, and mask must be provided together")
        if evidence_candidates.ndim != 6 or evidence_candidates.shape[-1] != 4:
            raise ValueError("evidence_candidates must have shape (batch, windows, steps, robots, horizon, 4)")
        if evidence_responses.ndim != 5 or evidence_responses.shape[-1] != 2:
            raise ValueError("evidence_responses must have shape (batch, windows, steps, horizon, 2)")
        if tuple(evidence_candidates.shape[:2]) != (batch, windows):
            raise ValueError("evidence candidates must align with entity history")
        steps = evidence_candidates.shape[2]
        if tuple(evidence_responses.shape[:3]) != (batch, windows, steps):
            raise ValueError("evidence responses must align with candidates")
        if tuple(evidence_mask.shape) != (batch, windows, steps):
            raise ValueError("evidence_mask must have shape (batch, windows, steps)")
        if self.config.evidence_pooling == "attention":
            prefix = min(self.config.evidence_prefix_steps, evidence_candidates.shape[-2], evidence_responses.shape[-2])
            robots = evidence_candidates.shape[-3]
            action = evidence_candidates[..., :prefix, :].to(device=device, dtype=dtype)
            response = evidence_responses[..., None, :prefix, :].to(device=device, dtype=dtype).expand(*action.shape[:-1], 2)
            observed = torch.cat([action, response], dim=-1)
            valid_evidence = evidence_mask.to(device=device, dtype=torch.bool)[..., None, None, None]
            observed = torch.where(valid_evidence, observed, torch.zeros_like(observed))
            trajectory_tokens = self.evidence_token(observed).reshape(-1, prefix, hidden)
            _, temporal_state = self.evidence_time_encoder(trajectory_tokens)
            robot_tokens = temporal_state[-1].reshape(batch, windows, steps, robots, hidden)
            weights = self.evidence_robot_score(robot_tokens).softmax(dim=-2)
            tokens = self.evidence_window_projection((robot_tokens * weights).sum(dim=-2))
        else:
            prefix = min(self.config.evidence_prefix_steps, evidence_candidates.shape[-2], evidence_responses.shape[-2])
            action = evidence_candidates[..., :prefix, :].mean(dim=(-3, -2))
            response = evidence_responses[..., :prefix, :].mean(dim=-2)
            tokens = torch.cat([action, response], dim=-1).to(device=device, dtype=dtype)
        tokens = tokens.reshape(batch * windows, steps, 6)
        valid = evidence_mask.to(device=device, dtype=torch.bool).reshape(batch * windows, steps)
        state = torch.zeros(1, batch * windows, self.config.hidden_dim, device=device, dtype=dtype)
        for step in range(steps):
            observed = torch.where(valid[:, step, None], tokens[:, step], torch.zeros_like(tokens[:, step]))
            _, proposed = self.evidence_encoder(observed[:, None], state)
            state = torch.where(valid[None, :, step, None], proposed, state)
        selected = state[-1]
        selected = self.evidence_projection(selected)
        return torch.where(
            evidence_mask.to(device=device).any(dim=-1).reshape(-1, 1),
            selected,
            torch.zeros_like(selected),
        )

    def _pair_features(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.cat([first + second, (first - second).abs(), first * second], dim=-1)

    def _prepare_robot_inputs(
        self,
        robot_current: torch.Tensor,
        robot_candidates: torch.Tensor,
        robot_mask: torch.Tensor | None,
        reference_candidates: torch.Tensor | None,
        batch: int,
        windows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if robot_current.ndim != 4 or robot_current.shape[-1] != 4:
            raise ValueError("robot_current must have shape (batch, windows, robots, 4)")
        if robot_candidates.ndim != 5 or robot_candidates.shape[-1] != 4:
            raise ValueError(
                "robot_candidates must have shape (batch, windows, robots, horizon, 4)"
            )
        if tuple(robot_current.shape[:2]) != (batch, windows):
            raise ValueError("robot_current must align with entity_history batch/windows")
        if tuple(robot_candidates.shape[:3]) != tuple(robot_current.shape[:3]):
            raise ValueError("robot_candidates must align with robot_current")
        if int(robot_candidates.shape[-2]) != self.config.horizon:
            raise ValueError("robot_candidates horizon does not match the model config")

        supplied_robots = int(robot_current.shape[2])
        max_robots = int(self.config.num_robots)
        if supplied_robots > max_robots:
            raise ValueError(
                f"received {supplied_robots} robot slots, but config.num_robots={max_robots}"
            )

        if robot_mask is None:
            robot_mask = torch.ones(
                batch,
                windows,
                supplied_robots,
                device=robot_current.device,
                dtype=torch.bool,
            )
        else:
            if tuple(robot_mask.shape) != (batch, windows, supplied_robots):
                raise ValueError("robot_mask must have shape (batch, windows, robots)")
            robot_mask = robot_mask.to(device=robot_current.device, dtype=torch.bool)

        if reference_candidates is None:
            reference_candidates = stationary_reference_candidates(
                robot_current, self.config.horizon
            )
        else:
            if tuple(reference_candidates.shape) != tuple(robot_candidates.shape):
                raise ValueError("reference_candidates must match robot_candidates shape")
            reference_candidates = reference_candidates.to(
                device=robot_candidates.device, dtype=robot_candidates.dtype
            )

        if supplied_robots < max_robots:
            pad = max_robots - supplied_robots
            current_pad = robot_current.new_zeros(batch, windows, pad, 4)
            candidate_pad = robot_candidates.new_zeros(
                batch, windows, pad, self.config.horizon, 4
            )
            reference_pad = reference_candidates.new_zeros(
                batch, windows, pad, self.config.horizon, 4
            )
            mask_pad = torch.zeros(
                batch,
                windows,
                pad,
                device=robot_mask.device,
                dtype=torch.bool,
            )
            robot_current = torch.cat([robot_current, current_pad], dim=2)
            robot_candidates = torch.cat([robot_candidates, candidate_pad], dim=2)
            reference_candidates = torch.cat(
                [reference_candidates, reference_pad], dim=2
            )
            robot_mask = torch.cat([robot_mask, mask_pad], dim=2)

        robot_inputs = torch.cat(
            [
                robot_current.reshape(-1, max_robots, 4),
                robot_candidates.reshape(
                    -1, max_robots, self.config.horizon * 4
                ),
            ],
            dim=-1,
        )
        reference_inputs = torch.cat(
            [
                robot_current.reshape(-1, max_robots, 4),
                reference_candidates.reshape(
                    -1, max_robots, self.config.horizon * 4
                ),
            ],
            dim=-1,
        )
        flat_mask = robot_mask.reshape(-1, max_robots)
        return robot_inputs, reference_inputs, flat_mask

    def forward(
        self,
        entity_history: torch.Tensor,
        robot_current: torch.Tensor,
        robot_candidates: torch.Tensor,
        include_pair_interactions: bool = True,
        *,
        robot_mask: torch.Tensor | None = None,
        reference_candidates: torch.Tensor | None = None,
        neighbor_history: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        condition_coordinates: torch.Tensor | None = None,
        evidence_candidates: torch.Tensor | None = None,
        evidence_responses: torch.Tensor | None = None,
        evidence_mask: torch.Tensor | None = None,
        static_segments: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if entity_history.ndim != 4:
            raise ValueError("entity_history must have shape (batch, windows, history, 2)")
        if str(self.config.architecture).lower() == "tracer":
            return self._forward_tracer(
                entity_history,
                robot_current,
                robot_candidates,
                include_pair_interactions,
                robot_mask,
                reference_candidates,
                neighbor_history,
                neighbor_mask,
                condition_coordinates,
                evidence_candidates,
                evidence_responses,
                evidence_mask,
                static_segments,
            )
        batch, windows = entity_history.shape[:2]
        flat_history = entity_history.reshape(-1, self.config.history, 2)
        robot_inputs, reference_inputs, flat_mask = self._prepare_robot_inputs(
            robot_current,
            robot_candidates,
            robot_mask,
            reference_candidates,
            int(batch),
            int(windows),
        )
        entity = self._entity_context(flat_history)
        mode_ids = torch.arange(self.config.num_modes, device=entity.device)
        context = entity[:, None, :] + self.mode_embedding(mode_ids)[None, :, :]
        mean = self.base_decoder(context).reshape(
            -1, self.config.num_modes, self.config.horizon, 2
        )

        robot_tokens = self.robot_encoder(robot_inputs)
        reference_tokens = self.robot_encoder(reference_inputs)
        expanded_context = context[:, :, None, :].expand(
            -1, -1, self.config.num_robots, -1
        )
        expanded_tokens = robot_tokens[:, None, :, :].expand(
            -1, self.config.num_modes, -1, -1
        )
        expanded_reference = reference_tokens[:, None, :, :].expand_as(expanded_tokens)
        main = self.main_decoder(torch.cat([expanded_context, expanded_tokens], dim=-1))
        reference_main = self.main_decoder(
            torch.cat([expanded_context, expanded_reference], dim=-1)
        )
        main_delta = (main - reference_main).reshape(
            -1, self.config.num_modes, self.config.num_robots, self.config.horizon, 2
        )
        mean = mean + (
            main_delta * flat_mask[:, None, :, None, None].to(main_delta.dtype)
        ).sum(dim=2)

        if include_pair_interactions:
            for first in range(self.config.num_robots):
                for second in range(first + 1, self.config.num_robots):
                    first_input = robot_inputs[:, first]
                    second_input = robot_inputs[:, second]
                    first_reference = reference_inputs[:, first]
                    second_reference = reference_inputs[:, second]

                    def encode(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                        return self.pair_encoder(self._pair_features(a, b))

                    pair_terms = (
                        encode(first_input, second_input),
                        encode(first_input, second_reference),
                        encode(first_reference, second_input),
                        encode(first_reference, second_reference),
                    )

                    def decode(pair: torch.Tensor) -> torch.Tensor:
                        expanded = pair[:, None, :].expand(-1, self.config.num_modes, -1)
                        return self.pair_decoder(torch.cat([context, expanded], dim=-1))

                    valid_pair = (flat_mask[:, first] & flat_mask[:, second])[
                        :, None, None
                    ].to(mean.dtype)
                    interaction = (
                        decode(pair_terms[0])
                        - decode(pair_terms[1])
                        - decode(pair_terms[2])
                        + decode(pair_terms[3])
                    ) * valid_pair
                    mean = mean + interaction.reshape(
                        -1, self.config.num_modes, self.config.horizon, 2
                    )

        mean = mean.reshape(batch, windows, self.config.num_modes, self.config.horizon, 2)
        log_std = self.log_std.clamp(
            self.config.min_log_std, self.config.max_log_std
        )[None, None, :, :, :].expand_as(mean)
        return {"mean": mean, "log_std": log_std}


@dataclass(frozen=True)
class TrajectoryRecurrentConfig:
    history: int = 8
    horizon: int = 12
    num_robots: int = 2
    num_modes: int = 3
    hidden_dim: int = 64
    execute_steps: int = 3
    use_entity_history: bool = False
    dt: float = 0.4


class TrajectoryRecurrentBelief(nn.Module):
    """Implicit belief baseline with the same executed-window observations."""

    def __init__(self, config: TrajectoryRecurrentConfig) -> None:
        super().__init__()
        self.config = config
        action_dim = config.num_robots * (4 + config.horizon * 4)
        if not 1 <= config.execute_steps <= config.horizon:
            raise ValueError("execute_steps must lie in [1, horizon]")
        response_dim = config.execute_steps * 2
        history_dim = config.history * 4 if config.use_entity_history else 0
        self.window_encoder = nn.Sequential(
            nn.Linear(action_dim + response_dim + history_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.recurrent = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.decoder = nn.Linear(config.hidden_dim, config.num_modes)

    def forward(
        self,
        robot_current: torch.Tensor,
        robot_candidates: torch.Tensor,
        responses: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
        entity_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if responses.shape[-2] < self.config.execute_steps:
            raise ValueError("responses do not contain the configured executed prefix")
        executed = responses[..., : self.config.execute_steps, :]
        action = torch.cat([robot_current.flatten(start_dim=2), robot_candidates.flatten(start_dim=2)], dim=-1)
        inputs = [action, executed.flatten(start_dim=2)]
        if self.config.use_entity_history:
            if entity_history is None:
                raise ValueError("entity_history is required by this recurrent config")
            if tuple(entity_history.shape[:2]) != tuple(responses.shape[:2]):
                raise ValueError("entity_history must align with batch and windows")
            current = entity_history[..., -1:, :]
            relative = entity_history - current
            velocity = torch.zeros_like(relative)
            velocity[..., 1:, :] = (
                entity_history[..., 1:, :] - entity_history[..., :-1, :]
            ) / self.config.dt
            inputs.append(torch.cat([relative, velocity], dim=-1).flatten(start_dim=2))
        window = self.window_encoder(torch.cat(inputs, dim=-1))
        if observation_mask is None:
            states, _ = self.recurrent(window)
        else:
            if tuple(observation_mask.shape) != tuple(window.shape[:2]):
                raise ValueError("observation_mask must align with batch and windows")
            hidden = torch.zeros(
                1,
                window.shape[0],
                self.config.hidden_dim,
                device=window.device,
                dtype=window.dtype,
            )
            states_by_window = []
            for index in range(window.shape[1]):
                candidate_state, candidate_hidden = self.recurrent(
                    window[:, index : index + 1], hidden
                )
                observed = observation_mask[:, index].to(
                    device=window.device, dtype=torch.bool
                )[None, :, None]
                hidden = torch.where(observed, candidate_hidden, hidden)
                states_by_window.append(hidden[0])
            states = torch.stack(states_by_window, dim=1)
        return torch.log_softmax(self.decoder(states), dim=-1)


def trajectory_mode_log_likelihood(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if mean.shape != log_std.shape or mean.shape[:-3] != target.shape[:-2]:
        raise ValueError("mode predictions and trajectory targets are not aligned")
    residual = target[..., None, :, :] - mean
    values = (
        -0.5 * residual.square() * torch.exp(-2.0 * log_std)
        - log_std
        - 0.5 * math.log(2.0 * math.pi)
    )
    return values.sum(dim=(-1, -2))


def continuous_trajectory_nll(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
    observation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return Gaussian trajectory NLL for a coordinate-conditioned response.

    ``mean`` and ``log_std`` may include a singleton condition dimension
    ``(..., 1, horizon, 2)``; the target is broadcast across that dimension.
    """

    if mean.shape != log_std.shape or mean.shape[-2:] != target.shape[-2:]:
        raise ValueError("continuous response tensors must align with target")
    residual = target.unsqueeze(-3) - mean
    values = (
        0.5 * residual.square() * torch.exp(-2.0 * log_std)
        + log_std
        + 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=(-1, -2))
    values = values.squeeze(-1)
    if observation_mask is None:
        return values.mean()
    if tuple(observation_mask.shape) != tuple(values.shape):
        raise ValueError("observation_mask must align with continuous NLL")
    selected = values[observation_mask]
    if not selected.numel():
        raise ValueError("observation_mask must select at least one observation")
    return selected.mean()


def continuous_coordinate_nll(
    prior: torch.Tensor,
    target: torch.Tensor,
    observation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative log likelihood for a scalar continuous behavior coordinate."""

    if prior.shape[-1] != 2:
        raise ValueError("prior must contain mean and log_std in its last dimension")
    mean, log_std = prior[..., 0], prior[..., 1]
    residual = target - mean
    values = (
        0.5 * residual.square() * torch.exp(-2.0 * log_std)
        + log_std
        + 0.5 * math.log(2.0 * math.pi)
    )
    if observation_mask is None:
        return values.mean()
    if tuple(observation_mask.shape) != tuple(values.shape):
        raise ValueError("observation_mask must align with coordinate NLL")
    selected = values[observation_mask]
    if not selected.numel():
        raise ValueError("observation_mask must select at least one coordinate")
    return selected.mean()


def executed_prefix_mode_log_likelihood(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
    execute_steps: int,
) -> torch.Tensor:
    """Score only response steps that were actually executed and observed."""

    if not 1 <= execute_steps <= target.shape[-2]:
        raise ValueError("execute_steps must lie within the target horizon")
    return trajectory_mode_log_likelihood(
        mean[..., :execute_steps, :],
        log_std[..., :execute_steps, :],
        target[..., :execute_steps, :],
    )


def mode_conditioned_trajectory_loss(
    log_likelihood: torch.Tensor,
    modes: torch.Tensor,
    coordinates: int,
    discrimination_weight: float = 0.0,
    temperature: float = 1.0,
    observation_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Combine true-mode trajectory NLL with likelihood-based mode discrimination."""
    if log_likelihood.ndim != 3 or modes.shape != log_likelihood.shape[:1]:
        raise ValueError("log_likelihood must be (batch, windows, modes) with one mode per batch")
    if coordinates <= 0:
        raise ValueError("coordinates must be positive")
    if discrimination_weight < 0.0:
        raise ValueError("discrimination_weight must be non-negative")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    targets = modes[:, None].expand(-1, log_likelihood.shape[1])
    if observation_mask is None:
        observation_mask = torch.ones_like(targets, dtype=torch.bool)
    elif tuple(observation_mask.shape) != tuple(targets.shape):
        raise ValueError("observation_mask must align with batch and windows")
    else:
        observation_mask = observation_mask.to(
            device=log_likelihood.device, dtype=torch.bool
        )
    if not bool(observation_mask.any()):
        raise ValueError("observation_mask must select at least one observation")
    selected = log_likelihood.gather(dim=-1, index=targets[..., None]).squeeze(-1)
    trajectory_nll = -selected[observation_mask].mean() / coordinates
    mode_logits = log_likelihood / (coordinates * temperature)
    discrimination = F.cross_entropy(
        mode_logits[observation_mask],
        targets[observation_mask],
    )
    return {
        "loss": trajectory_nll + discrimination_weight * discrimination,
        "trajectory_nll": trajectory_nll,
        "mode_discrimination": discrimination,
    }


def belief_weighted_mean(mode_means: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
    if mode_means.shape[:-3] != belief.shape[:-1] or mode_means.shape[-3] != belief.shape[-1]:
        raise ValueError("belief must align with the mode dimension")
    return (mode_means * belief[..., :, None, None]).sum(dim=-3)
