from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .data import EntityResponseBatch
from .tracer import TRACERFeatureEncoder, stationary_reference_candidates


@dataclass(frozen=True)
class ResponseModelConfig:
    history: int = 8
    horizon: int = 12
    num_robots: int = 2
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    min_log_std: float = -4.0
    max_log_std: float = 2.0
    architecture: str = "legacy"
    graph_layers: int = 2
    dt: float = 0.4
    geometry_features: bool = False
    temporal_pooling: str = "mean"
    dynamic_interaction_features: bool = False
    temporal_pair_interactions: bool = False
    hybrid_temporal_pair_interactions: bool = False


class TeamResponsePredictor(nn.Module):
    """Gaussian entity-response predictor with controlled conditioning variants."""

    VALID_VARIANTS = {
        "history",
        "nearest",
        "additive",
        "matched-additive",
        "joint",
        "team-concat",
        "eiffert2019-style",
        "kedia-style",
    }

    def __init__(self, variant: str, config: ResponseModelConfig) -> None:
        super().__init__()
        variant = str(variant).lower()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown response variant {variant!r}; expected {sorted(self.VALID_VARIANTS)}")
        self.variant = variant
        self.config = config
        if variant == "eiffert2019-style":
            # The external class is defined below; lookup occurs when the
            # constructor is called after module import has completed.
            self._external_model = Eiffert2019StyleResponsePredictor(config)
            return
        if variant == "kedia-style":
            self._external_model = KediaStyleResponsePredictor(config)
            return
        architecture = str(config.architecture).lower()
        if architecture not in {"legacy", "tracer"}:
            raise ValueError("architecture must be 'legacy' or 'tracer'")
        if architecture == "tracer":
            self._init_tracer()
            return
        hidden = config.hidden_dim
        self.history_encoder = nn.GRU(input_size=4, hidden_size=hidden, batch_first=True)
        self.base_mean_decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.horizon * 2),
        )
        self.log_std_decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.horizon * 2),
        )

        if variant != "history":
            robot_input_dim = 4 + config.horizon * 4
            robot_layers: list[nn.Module] = [
                nn.Linear(robot_input_dim, hidden),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, hidden),
            ]
            robot_layers.extend([nn.SiLU(), nn.Linear(hidden, hidden)])
            self.robot_encoder = nn.Sequential(*robot_layers)

            delta_hidden = hidden * 2
            delta_layers: list[nn.Module] = [
                nn.Linear(hidden * 2, delta_hidden),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            ]
            delta_layers.extend(
                [
                    nn.Linear(delta_hidden, delta_hidden),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(delta_hidden, delta_hidden),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            delta_layers.append(nn.Linear(delta_hidden, config.horizon * 2))
            self.response_delta_decoder = nn.Sequential(*delta_layers)
        if variant == "joint":
            pair_hidden = max(hidden // 2, config.num_heads)
            self.pair_encoder = nn.Sequential(
                nn.Linear(robot_input_dim * 3, pair_hidden),
                nn.SiLU(),
                nn.Linear(pair_hidden, pair_hidden),
            )
            pair_output = nn.Linear(pair_hidden, config.horizon * 2)
            nn.init.zeros_(pair_output.weight)
            nn.init.zeros_(pair_output.bias)
            self.pair_delta_decoder = nn.Sequential(
                nn.Linear(hidden + pair_hidden, pair_hidden),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                pair_output,
            )
        elif variant == "team-concat":
            # Generic joint baseline: all robot candidate features are exposed
            # to one decoder, without the proposed additive/pair decomposition.
            team_input_dim = robot_input_dim * config.num_robots
            self.team_encoder = nn.Sequential(
                nn.Linear(team_input_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.team_delta_decoder = nn.Sequential(
                nn.Linear(hidden * 2, hidden * 2),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden * 2, config.horizon * 2),
            )
        elif variant == "matched-additive":
            capacity_hidden = self._matched_additive_width(robot_input_dim)
            self.capacity_encoder = nn.Sequential(
                nn.Linear(robot_input_dim, capacity_hidden),
                nn.SiLU(),
                nn.Linear(capacity_hidden, capacity_hidden),
            )
            capacity_output = nn.Linear(capacity_hidden, config.horizon * 2)
            nn.init.zeros_(capacity_output.weight)
            nn.init.zeros_(capacity_output.bias)
            self.capacity_delta_decoder = nn.Sequential(
                nn.Linear(hidden + capacity_hidden, capacity_hidden),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                capacity_output,
            )

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
            temporal_pooling=config.temporal_pooling,
            dynamic_interaction_features=config.dynamic_interaction_features,
            return_temporal_sequence=(
                config.temporal_pair_interactions
                or config.hybrid_temporal_pair_interactions
            ),
        )
        self.tracer_base_mean_decoder = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, config.horizon * 2),
        )
        self.tracer_log_std_decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, config.horizon * 2),
        )
        self.tracer_single_decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, config.horizon * 2),
        )
        if config.temporal_pair_interactions:
            pair_target = self._tracer_pair_parameter_count(
                hidden,
                max(hidden // 2, config.num_heads),
                config.horizon * 2,
            )
            pair_hidden = min(
                range(1, hidden + 1),
                key=lambda width: abs(
                    self._tracer_pair_parameter_count(
                        hidden, width, config.horizon * 2, temporal=True
                    )
                    - pair_target
                ),
            )
        elif config.hybrid_temporal_pair_interactions:
            pair_target = self._tracer_pair_parameter_count(
                hidden,
                max(hidden // 2, config.num_heads),
                config.horizon * 2,
            )
            pair_hidden = min(
                range(1, hidden + 1),
                key=lambda width: abs(
                    self._tracer_pair_parameter_count(
                        hidden,
                        width,
                        config.horizon * 2,
                        hybrid=True,
                    )
                    - pair_target
                ),
            )
        else:
            pair_hidden = max(hidden // 2, config.num_heads)
        self.tracer_pair_encoder = nn.Sequential(
            nn.Linear(hidden * 3, pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, pair_hidden),
        )
        pair_output = nn.Linear(
            hidden, 2 if config.temporal_pair_interactions else config.horizon * 2
        )
        nn.init.zeros_(pair_output.weight)
        nn.init.zeros_(pair_output.bias)
        self.tracer_pair_decoder = nn.Sequential(
            nn.Linear(hidden + pair_hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            pair_output,
        )
        if config.hybrid_temporal_pair_interactions:
            temporal_pair_output = nn.Linear(hidden + pair_hidden, 2)
            nn.init.zeros_(temporal_pair_output.weight)
            nn.init.zeros_(temporal_pair_output.bias)
            self.tracer_temporal_pair_decoder = nn.Sequential(temporal_pair_output)
        capacity_hidden = self._matched_tracer_capacity_width(
            hidden * 2,
            config.horizon * 2,
            self._tracer_pair_parameter_count(
                hidden,
                pair_hidden,
                config.horizon * 2,
                temporal=config.temporal_pair_interactions,
                hybrid=config.hybrid_temporal_pair_interactions,
            ),
        )
        capacity_output = nn.Linear(capacity_hidden, config.horizon * 2)
        nn.init.zeros_(capacity_output.weight)
        nn.init.zeros_(capacity_output.bias)
        self.tracer_capacity_decoder = nn.Sequential(
            nn.Linear(hidden * 2, capacity_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            capacity_output,
        )
        self.tracer_team_decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, config.horizon * 2),
        )

    @staticmethod
    def _symmetric_pair(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.cat([first + second, (first - second).abs(), first * second], dim=-1)

    def _forward_tracer(
        self,
        batch: EntityResponseBatch,
        include_pair_interactions: bool,
    ) -> dict[str, torch.Tensor]:
        reference_candidates = stationary_reference_candidates(
            batch.robot_current, self.config.horizon
        )
        robot_mask = batch.robot_mask.bool()
        encoded = self.tracer_encoder(
            batch.entity_history,
            batch.robot_current,
            batch.robot_candidates,
            reference_candidates,
            robot_mask,
            batch.neighbor_history,
            batch.neighbor_mask,
            batch.robot_actions,
        )
        scene = encoded["scene"]
        candidate = encoded["candidate"]
        reference = encoded["reference"]
        candidate_sequence = encoded.get("candidate_sequence")
        reference_sequence = encoded.get("reference_sequence")
        mean = self.tracer_base_mean_decoder(scene).reshape(
            -1, self.config.horizon, 2
        )
        active = robot_mask[..., None].to(candidate.dtype)
        pooled = (candidate * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
        reference_pooled = (reference * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
        if self.variant != "history":
            context = scene[:, None].expand(-1, candidate.shape[1], -1)
            single = self.tracer_single_decoder(torch.cat([context, candidate], dim=-1))
            single_reference = self.tracer_single_decoder(
                torch.cat([context, reference], dim=-1)
            )
            per_robot = (single - single_reference).reshape(
                -1, candidate.shape[1], self.config.horizon, 2
            )
            if self.variant == "nearest":
                mean = mean + per_robot[:, 0]
                mean = mean * robot_mask[:, :1, None].to(mean.dtype) + (
                    mean - per_robot[:, 0]
                ) * (~robot_mask[:, :1, None]).to(mean.dtype)
            else:
                mean = mean + (per_robot * active[..., None]).sum(dim=1)
                if self.variant == "matched-additive":
                    capacity = self.tracer_capacity_decoder(
                        torch.cat([context, candidate], dim=-1)
                    )
                    capacity_reference = self.tracer_capacity_decoder(
                        torch.cat([context, reference], dim=-1)
                    )
                    mean = mean + (
                        (capacity - capacity_reference).reshape(
                            -1, candidate.shape[1], self.config.horizon, 2
                        )
                        * active[..., None]
                    ).sum(dim=1)
                elif self.variant == "team-concat":
                    mean = mean - (per_robot * active[..., None]).sum(dim=1)
                    mean = mean + (
                        self.tracer_team_decoder(torch.cat([scene, pooled], dim=-1))
                        - self.tracer_team_decoder(
                            torch.cat([scene, reference_pooled], dim=-1)
                        )
                    ).reshape(-1, self.config.horizon, 2)
                elif self.variant == "joint" and include_pair_interactions:
                    for first in range(candidate.shape[1]):
                        for second in range(first + 1, candidate.shape[1]):
                            use_temporal_pair = (
                                self.config.temporal_pair_interactions
                                or self.config.hybrid_temporal_pair_interactions
                            )
                            if use_temporal_pair:
                                if candidate_sequence is None or reference_sequence is None:
                                    raise RuntimeError(
                                        "temporal pair interactions require temporal sequences"
                                    )
                                if self.config.hybrid_temporal_pair_interactions:
                                    pair = self.tracer_pair_encoder(
                                        self._symmetric_pair(candidate[:, first], candidate[:, second])
                                    )
                                    pair_first_ref = self.tracer_pair_encoder(
                                        self._symmetric_pair(candidate[:, first], reference[:, second])
                                    )
                                    pair_second_ref = self.tracer_pair_encoder(
                                        self._symmetric_pair(reference[:, first], candidate[:, second])
                                    )
                                    pair_reference = self.tracer_pair_encoder(
                                        self._symmetric_pair(reference[:, first], reference[:, second])
                                    )
                                    scene_for_pair = scene
                                else:
                                    pair = self.tracer_pair_encoder(
                                        self._symmetric_pair(
                                            candidate_sequence[:, first], candidate_sequence[:, second]
                                        )
                                    )
                                    pair_first_ref = self.tracer_pair_encoder(
                                        self._symmetric_pair(
                                            candidate_sequence[:, first], reference_sequence[:, second]
                                        )
                                    )
                                    pair_second_ref = self.tracer_pair_encoder(
                                        self._symmetric_pair(
                                            reference_sequence[:, first], candidate_sequence[:, second]
                                        )
                                    )
                                    pair_reference = self.tracer_pair_encoder(
                                        self._symmetric_pair(
                                            reference_sequence[:, first], reference_sequence[:, second]
                                        )
                                    )
                                    scene_for_pair = scene[:, None, :].expand(
                                        -1, self.config.horizon, -1
                                    )
                            else:
                                pair = self.tracer_pair_encoder(
                                    self._symmetric_pair(candidate[:, first], candidate[:, second])
                                )
                                pair_first_ref = self.tracer_pair_encoder(
                                    self._symmetric_pair(candidate[:, first], reference[:, second])
                                )
                                pair_second_ref = self.tracer_pair_encoder(
                                    self._symmetric_pair(reference[:, first], candidate[:, second])
                                )
                                pair_reference = self.tracer_pair_encoder(
                                    self._symmetric_pair(reference[:, first], reference[:, second])
                                )
                                scene_for_pair = scene

                            def decode(value: torch.Tensor) -> torch.Tensor:
                                return self.tracer_pair_decoder(
                                    torch.cat([scene_for_pair, value], dim=-1)
                                )

                            pair_delta = (
                                decode(pair)
                                - decode(pair_first_ref)
                                - decode(pair_second_ref)
                                + decode(pair_reference)
                            )
                            valid = (robot_mask[:, first] & robot_mask[:, second])[:, None, None]
                            if self.config.hybrid_temporal_pair_interactions:
                                mean = mean + pair_delta.reshape(
                                    -1, self.config.horizon, 2
                                ) * valid.to(mean.dtype)
                                temporal_pair = self.tracer_pair_encoder(
                                    self._symmetric_pair(
                                        candidate_sequence[:, first], candidate_sequence[:, second]
                                    )
                                )
                                temporal_first_ref = self.tracer_pair_encoder(
                                    self._symmetric_pair(
                                        candidate_sequence[:, first], reference_sequence[:, second]
                                    )
                                )
                                temporal_second_ref = self.tracer_pair_encoder(
                                    self._symmetric_pair(
                                        reference_sequence[:, first], candidate_sequence[:, second]
                                    )
                                )
                                temporal_reference = self.tracer_pair_encoder(
                                    self._symmetric_pair(
                                        reference_sequence[:, first], reference_sequence[:, second]
                                    )
                                )
                                temporal_scene = scene[:, None, :].expand(
                                    -1, self.config.horizon, -1
                                )

                                def temporal_decode(value: torch.Tensor) -> torch.Tensor:
                                    return self.tracer_temporal_pair_decoder(
                                        torch.cat([temporal_scene, value], dim=-1)
                                    )

                                temporal_delta = (
                                    temporal_decode(temporal_pair)
                                    - temporal_decode(temporal_first_ref)
                                    - temporal_decode(temporal_second_ref)
                                    + temporal_decode(temporal_reference)
                                )
                                mean = mean + temporal_delta * valid.to(mean.dtype)
                            elif self.config.temporal_pair_interactions:
                                mean = mean + pair_delta * valid.to(mean.dtype)
                            else:
                                mean = mean + pair_delta.reshape(
                                    -1, self.config.horizon, 2
                                ) * valid.to(mean.dtype)
        log_std = self.tracer_log_std_decoder(torch.cat([scene, pooled], dim=-1)).reshape(
            -1, self.config.horizon, 2
        )
        log_std = log_std.clamp(self.config.min_log_std, self.config.max_log_std)
        return {"mean": mean, "log_std": log_std}


    def _joint_specific_parameter_count(self, robot_input_dim: int) -> int:
        pair_hidden = max(self.config.hidden_dim // 2, self.config.num_heads)
        output_dim = self.config.horizon * 2
        return (
            (robot_input_dim * 3 + 1) * pair_hidden
            + (pair_hidden + 1) * pair_hidden
            + (self.config.hidden_dim + pair_hidden + 1) * pair_hidden
            + (pair_hidden + 1) * output_dim
        )

    @staticmethod
    def _tracer_pair_parameter_count(
        hidden: int,
        pair_hidden: int,
        output_dim: int,
        temporal: bool = False,
        hybrid: bool = False,
    ) -> int:
        decoder_output = 2 if temporal else output_dim
        total = (
            (hidden * 3 + 1) * pair_hidden
            + (pair_hidden + 1) * pair_hidden
            + (hidden + pair_hidden + 1) * hidden
            + (hidden + 1) * decoder_output
        )
        if hybrid:
            total += (hidden + pair_hidden + 1) * 2
        return total

    @staticmethod
    def _matched_tracer_capacity_width(
        input_dim: int,
        output_dim: int,
        target: int,
    ) -> int:
        def branch_parameters(width: int) -> int:
            return (input_dim + 1) * width + (width + 1) * output_dim

        candidates = range(1, max(input_dim, output_dim) * 4 + 1)
        return min(candidates, key=lambda width: abs(branch_parameters(width) - target))

    def _matched_additive_width(self, robot_input_dim: int) -> int:
        """Choose an additive branch width with <1% parameter mismatch to Joint."""

        target = self._joint_specific_parameter_count(robot_input_dim)
        output_dim = self.config.horizon * 2

        def branch_parameters(width: int) -> int:
            return (
                (robot_input_dim + 1) * width
                + (width + 1) * width
                + (self.config.hidden_dim + width + 1) * width
                + (width + 1) * output_dim
            )

        candidates = range(1, self.config.hidden_dim * 4 + 1)
        return min(candidates, key=lambda width: abs(branch_parameters(width) - target))

    def _encode_history(self, batch: EntityResponseBatch) -> torch.Tensor:
        history = batch.entity_history
        current = history[:, -1:, :]
        relative = history - current
        velocity = torch.zeros_like(relative)
        velocity[:, 1:] = (history[:, 1:] - history[:, :-1]) / batch.dt
        sequence = torch.cat([relative, velocity], dim=-1)
        _, state = self.history_encoder(sequence)
        return state[-1]

    def _robot_inputs(self, batch: EntityResponseBatch) -> torch.Tensor:
        return torch.cat(
            [batch.robot_current, batch.robot_candidates.flatten(start_dim=2)], dim=-1
        )

    def _encode_pair(self, first_input: torch.Tensor, second_input: torch.Tensor) -> torch.Tensor:
        pair_features = torch.cat(
            [
                first_input + second_input,
                (first_input - second_input).abs(),
                first_input * second_input,
            ],
            dim=-1,
        )
        return self.pair_encoder(pair_features)

    def _pair_response_delta(
        self,
        entity_context: torch.Tensor,
        robot_inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pair_deltas = []
        for first in range(robot_inputs.shape[1]):
            for second in range(first + 1, robot_inputs.shape[1]):
                first_input = robot_inputs[:, first]
                second_input = robot_inputs[:, second]
                zeros = torch.zeros_like(first_input)
                both = self._encode_pair(first_input, second_input)
                first_only = self._encode_pair(first_input, zeros)
                second_only = self._encode_pair(zeros, second_input)
                neither = self._encode_pair(zeros, zeros)

                def decode(pair_context: torch.Tensor) -> torch.Tensor:
                    return self.pair_delta_decoder(
                        torch.cat([entity_context, pair_context], dim=-1)
                    )

                interaction = (
                    decode(both)
                    - decode(first_only)
                    - decode(second_only)
                    + decode(neither)
                )
                valid = (mask[:, first] & mask[:, second])[:, None].to(interaction.dtype)
                pair_deltas.append(interaction * valid)
        if not pair_deltas:
            return robot_inputs.new_zeros(
                robot_inputs.shape[0], self.config.horizon, 2
            )
        return torch.stack(pair_deltas, dim=1).sum(dim=1).reshape(
            -1, self.config.horizon, 2
        )

    def forward(
        self,
        batch: EntityResponseBatch,
        *,
        include_pair_interactions: bool = True,
    ) -> dict[str, torch.Tensor]:
        if hasattr(self, "_external_model"):
            return self._external_model(batch)
        if str(self.config.architecture).lower() == "tracer":
            return self._forward_tracer(batch, include_pair_interactions)
        entity_context = self._encode_history(batch)
        mean = self.base_mean_decoder(entity_context).reshape(-1, self.config.horizon, 2)
        log_std = self.log_std_decoder(entity_context).reshape(-1, self.config.horizon, 2)
        if self.variant == "history":
            response_delta = torch.zeros_like(mean)
        else:
            robot_inputs = self._robot_inputs(batch)
            robot_tokens = self.robot_encoder(robot_inputs)
            reference_tokens = self.robot_encoder(torch.zeros_like(robot_inputs))
            mask = batch.robot_mask
            if self.variant == "nearest":
                delta_input = torch.cat([entity_context, robot_tokens[:, 0]], dim=-1)
                reference_input = torch.cat([entity_context, reference_tokens[:, 0]], dim=-1)
                response_delta = (
                    self.response_delta_decoder(delta_input)
                    - self.response_delta_decoder(reference_input)
                ).reshape(
                    -1, self.config.horizon, 2
                )
                response_delta = response_delta * mask[:, :1, None].to(response_delta.dtype)
            elif self.variant == "team-concat":
                masked_inputs = robot_inputs * batch.robot_mask[:, :, None].to(
                    robot_inputs.dtype
                )
                team_features = self.team_encoder(masked_inputs.flatten(start_dim=1))
                reference_features = self.team_encoder(
                    torch.zeros_like(masked_inputs).flatten(start_dim=1)
                )
                response_delta = (
                    self.team_delta_decoder(
                        torch.cat([entity_context, team_features], dim=-1)
                    )
                    - self.team_delta_decoder(
                        torch.cat([entity_context, reference_features], dim=-1)
                    )
                ).reshape(-1, self.config.horizon, 2)
            elif self.variant in {"additive", "matched-additive", "joint"}:
                entity_tokens = entity_context[:, None, :].expand(-1, robot_tokens.shape[1], -1)
                delta_input = torch.cat([entity_tokens, robot_tokens], dim=-1)
                reference_input = torch.cat([entity_tokens, reference_tokens], dim=-1)
                per_robot_delta = (
                    self.response_delta_decoder(delta_input)
                    - self.response_delta_decoder(reference_input)
                ).reshape(
                    -1, robot_tokens.shape[1], self.config.horizon, 2
                )
                response_delta = (
                    per_robot_delta * mask[:, :, None, None].to(per_robot_delta.dtype)
                ).sum(dim=1)
                if self.variant == "matched-additive":
                    capacity_tokens = self.capacity_encoder(robot_inputs)
                    capacity_reference = self.capacity_encoder(torch.zeros_like(robot_inputs))
                    capacity_input = torch.cat([entity_tokens, capacity_tokens], dim=-1)
                    capacity_reference_input = torch.cat(
                        [entity_tokens, capacity_reference], dim=-1
                    )
                    per_robot_capacity = (
                        self.capacity_delta_decoder(capacity_input)
                        - self.capacity_delta_decoder(capacity_reference_input)
                    ).reshape(
                        -1, robot_tokens.shape[1], self.config.horizon, 2
                    )
                    response_delta = response_delta + (
                        per_robot_capacity
                        * mask[:, :, None, None].to(per_robot_capacity.dtype)
                    ).sum(dim=1)
                if self.variant == "joint" and include_pair_interactions:
                    response_delta = response_delta + self._pair_response_delta(
                        entity_context, robot_inputs, mask
                    )

        mean = mean + response_delta
        log_std = log_std.clamp(self.config.min_log_std, self.config.max_log_std)
        return {"mean": mean, "log_std": log_std}


class Eiffert2019StyleResponsePredictor(nn.Module):
    """Method-level Eiffert & Sukkarieh (2019)-style response predictor.

    This clean-room reimplementation encodes entity history and each explicit
    robot future, masks padded robot slots, and decodes a Gaussian response.
    It is not an official reproduction and uses the shared response interface.
    """

    def __init__(self, config: ResponseModelConfig) -> None:
        super().__init__()
        self.variant = "eiffert2019-style"
        self.config = config
        hidden = int(config.hidden_dim)
        self.history_encoder = nn.GRU(input_size=4, hidden_size=hidden, batch_first=True)
        self.robot_current_encoder = nn.GRU(input_size=4, hidden_size=hidden, batch_first=True)
        self.robot_future_encoder = nn.GRU(input_size=4, hidden_size=hidden, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden), nn.SiLU(),
        )
        output_dim = int(config.horizon) * 2
        self.mean_decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(hidden, output_dim)
        )
        self.log_std_decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(hidden, output_dim)
        )

    def _history(self, batch: EntityResponseBatch) -> torch.Tensor:
        history = batch.entity_history
        current = history[:, -1:, :]
        relative = history - current
        velocity = torch.zeros_like(relative)
        velocity[:, 1:] = (history[:, 1:] - history[:, :-1]) / batch.dt
        _, state = self.history_encoder(torch.cat([relative, velocity], dim=-1))
        return state[-1]

    def _robot_context(self, batch: EntityResponseBatch) -> torch.Tensor:
        current = batch.robot_current
        candidates = batch.robot_candidates
        batch_size, robots, horizon, _ = candidates.shape
        _, current_state = self.robot_current_encoder(current.reshape(batch_size * robots, 1, 4))
        _, future_state = self.robot_future_encoder(
            candidates.reshape(batch_size * robots, horizon, 4)
        )
        token = torch.cat([current_state[-1], future_state[-1]], dim=-1).reshape(batch_size, robots, -1)
        mask = batch.robot_mask.bool()
        token = token * mask[..., None].to(token.dtype)
        return token.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(token.dtype)

    def forward(self, batch: EntityResponseBatch) -> dict[str, torch.Tensor]:
        fused = self.fusion(torch.cat([self._history(batch), self._robot_context(batch)], dim=-1))
        mean = self.mean_decoder(fused).reshape(-1, self.config.horizon, 2)
        log_std = self.log_std_decoder(fused).reshape(-1, self.config.horizon, 2)
        return {"mean": mean, "log_std": log_std.clamp(self.config.min_log_std, self.config.max_log_std)}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class KediaStyleResponsePredictor(nn.Module):
    """Method-level game-theoretic latent-hypothesis response adapter.

    The original game-theoretic intention-learning setting is not trained on
    this repository's Gaussian entity-response target. This adapter preserves
    the compatible part of that idea: an action-conditioned latent hypothesis
    belief is updated for multiple reasoning rounds, and the hypotheses are
    marginalised into a calibrated Gaussian response. It is not an official
    reproduction of the cited planner or its data/training objective.
    """

    NUM_HYPOTHESES = 4
    REASONING_STEPS = 2

    def __init__(self, config: ResponseModelConfig) -> None:
        super().__init__()
        self.variant = "kedia-style"
        self.config = config
        hidden = int(config.hidden_dim)
        self.history_encoder = nn.GRU(input_size=4, hidden_size=hidden, batch_first=True)
        self.robot_current_encoder = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.robot_future_encoder = nn.GRU(input_size=4, hidden_size=hidden, batch_first=True)
        self.robot_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2), nn.SiLU(),
            nn.Dropout(config.dropout), nn.Linear(hidden * 2, hidden)
        )
        self.hypothesis_embedding = nn.Parameter(
            torch.randn(self.NUM_HYPOTHESES, hidden) * 0.02
        )
        self.hypothesis_prior = nn.Linear(hidden, self.NUM_HYPOTHESES)
        self.reasoning_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.hypothesis_score = nn.Linear(hidden, 1)
        output_dim = int(config.horizon) * 2
        self.hypothesis_mean = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2), nn.SiLU(),
            nn.Dropout(config.dropout), nn.Linear(hidden * 2, output_dim)
        )
        self.hypothesis_log_std = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2), nn.SiLU(),
            nn.Dropout(config.dropout), nn.Linear(hidden * 2, output_dim)
        )

    def _history_context(self, batch: EntityResponseBatch) -> torch.Tensor:
        history = batch.entity_history
        current = history[:, -1:, :]
        relative = history - current
        velocity = torch.zeros_like(history)
        velocity[:, 1:] = (history[:, 1:] - history[:, :-1]) / batch.dt
        _, state = self.history_encoder(torch.cat([relative, velocity], dim=-1))
        return state[-1]

    def _team_context(self, batch: EntityResponseBatch) -> torch.Tensor:
        batch_size, robots, horizon, _ = batch.robot_candidates.shape
        current_state = self.robot_current_encoder(
            batch.robot_current.reshape(batch_size * robots, 4)
        )
        _, future_state = self.robot_future_encoder(
            batch.robot_candidates.reshape(batch_size * robots, horizon, 4)
        )
        current_token = current_state
        future_token = future_state[-1]
        token = self.robot_fusion(torch.cat([current_token, future_token], dim=-1))
        token = token.reshape(batch_size, robots, -1)
        mask = batch.robot_mask.to(token.dtype)
        return (token * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    def forward(self, batch: EntityResponseBatch) -> dict[str, torch.Tensor]:
        history_context = self._history_context(batch)
        team_context = self._team_context(batch)
        context = self.context_fusion(torch.cat([history_context, team_context], dim=-1))
        batch_size = context.shape[0]
        states = self.hypothesis_embedding[None].expand(batch_size, -1, -1)
        logits = self.hypothesis_prior(context)
        context_tokens = context[:, None, :].expand(-1, self.NUM_HYPOTHESES, -1)
        for _ in range(self.REASONING_STEPS):
            states = states + self.reasoning_update(
                torch.cat([context_tokens, states], dim=-1)
            )
            logits = logits + self.hypothesis_score(states).squeeze(-1)
        weights = torch.softmax(logits / float(self.REASONING_STEPS + 1), dim=-1)
        hypothesis_input = torch.cat([context_tokens, states], dim=-1)
        hypothesis_mean = self.hypothesis_mean(hypothesis_input).reshape(
            batch_size, self.NUM_HYPOTHESES, self.config.horizon, 2
        )
        hypothesis_log_std = self.hypothesis_log_std(hypothesis_input).reshape(
            batch_size, self.NUM_HYPOTHESES, self.config.horizon, 2
        )
        hypothesis_log_std = hypothesis_log_std.clamp(
            self.config.min_log_std, self.config.max_log_std
        )
        mean = (weights[..., None, None] * hypothesis_mean).sum(dim=1)
        second_moment = (
            weights[..., None, None]
            * (torch.exp(2.0 * hypothesis_log_std) + hypothesis_mean.square())
        ).sum(dim=1)
        variance = (second_moment - mean.square()).clamp_min(1e-8)
        log_std = 0.5 * variance.log()
        return {
            "mean": mean,
            "log_std": log_std.clamp(self.config.min_log_std, self.config.max_log_std),
            "hypothesis_weights": weights,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_response_predictor(variant: str, config: ResponseModelConfig) -> nn.Module:
    """Construct a shared-interface response predictor by method name."""
    normalized = str(variant).lower()
    if normalized in {"eiffert2019-style", "eiffert2019_style", "eiffert"}:
        return Eiffert2019StyleResponsePredictor(config)
    return TeamResponsePredictor(normalized, config)
