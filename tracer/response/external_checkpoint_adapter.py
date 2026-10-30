from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import warnings

import torch
from torch import nn

from tracer.response.data import EntityResponseBatch
from tracer.response.models import ResponseModelConfig, TeamResponsePredictor
from tracer.response.tracer import stationary_reference_candidates


def delayed_response_prediction(model, batch, delay_steps, include_pair_interactions=True):
    if not 0 <= delay_steps <= model.config.horizon:
        raise ValueError("response delay must lie within the prediction horizon")
    prediction = model(batch, include_pair_interactions=include_pair_interactions)
    if not delay_steps:
        return prediction
    reference_batch = replace(batch, robot_candidates=stationary_reference_candidates(batch.robot_current, model.config.horizon))
    reference = model(reference_batch, include_pair_interactions=include_pair_interactions)
    return {name: torch.cat([reference[name][:, :delay_steps], prediction[name][:, delay_steps:]], dim=1)
            for name in ("mean", "log_std")}


@dataclass(frozen=True)
class ExternalResponseAdapterConfig:
    history: int
    horizon: int
    num_robots: int
    num_modes: int
    dt: float


class ExternalResponseCheckpointAdapter(nn.Module):
    """Expose a single-mode external response model through the closed-loop API."""

    def __init__(self, model: TeamResponsePredictor, dt: float) -> None:
        super().__init__()
        if model.config.num_robots != 1:
            raise ValueError("external closed-loop adapter currently requires one robot")
        self.predictor = model.eval()
        self.response_delay_steps = 0
        self.config = ExternalResponseAdapterConfig(
            history=int(model.config.history),
            horizon=int(model.config.horizon),
            num_robots=int(model.config.num_robots),
            # The paired external predictor has no latent behavior label.
            # Duplicate its distribution so the existing categorical belief
            # interface remains neutral instead of inventing a learned mode.
            num_modes=2,
            dt=float(dt),
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, device: str, expected_dt: float
    ) -> "ExternalResponseCheckpointAdapter":
        checkpoint = torch.load(
            Path(checkpoint_path), map_location=device, weights_only=False
        )
        if "model_config" not in checkpoint or "model" not in checkpoint:
            raise ValueError("external response checkpoint is missing model metadata")
        model_config = dict(checkpoint["model_config"])
        training_dt = float(model_config.get("dt", expected_dt))
        if abs(training_dt - expected_dt) > 1e-7:
            warnings.warn("Inference dt differs from training dt; changing velocity conversion does not retrain the response horizon.", RuntimeWarning)
        model_config["dt"] = float(expected_dt)
        config = ResponseModelConfig(**model_config)
        variant = str(checkpoint.get("variant", "joint"))
        model = TeamResponsePredictor(variant, config).to(device)
        restored_capacity_width = None
        capacity_weight = checkpoint["model"].get("tracer_capacity_decoder.0.weight")
        if capacity_weight is not None and hasattr(model, "tracer_capacity_decoder"):
            stored_width, stored_input = capacity_weight.shape
            if stored_input != config.hidden_dim * 2:
                raise ValueError("external checkpoint capacity input width conflicts with model configuration")
            if stored_width != model.tracer_capacity_decoder[0].out_features:
                model.tracer_capacity_decoder = nn.Sequential(
                    nn.Linear(stored_input, stored_width), nn.GELU(), nn.Dropout(config.dropout),
                    nn.Linear(stored_width, config.horizon * 2),
                ).to(device)
                restored_capacity_width = int(stored_width)
                warnings.warn("Restoring legacy capacity-decoder width from checkpoint weights, without changing learned parameters.", RuntimeWarning)
        model.load_state_dict(checkpoint["model"])
        adapter = cls(model, expected_dt).to(device).eval()
        adapter.training_dt = training_dt
        adapter.response_delay_steps = int(checkpoint.get("response_delay_steps", 0))
        if not 0 <= adapter.response_delay_steps <= config.horizon:
            raise ValueError("checkpoint response delay is outside the response horizon")
        adapter.restored_legacy_capacity_width = restored_capacity_width
        return adapter

    def forward(
        self,
        entity_history: torch.Tensor,
        robot_current: torch.Tensor,
        robot_candidates: torch.Tensor,
        include_pair_interactions: bool = True,
        *,
        robot_mask: torch.Tensor | None = None,
        neighbor_history: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        if entity_history.ndim != 4 or entity_history.shape[1] != 1:
            raise ValueError("entity_history must have shape (batch, 1, history, 2)")
        if robot_current.ndim != 4 or robot_current.shape[1] != 1:
            raise ValueError("robot_current must have shape (batch, 1, robots, 4)")
        if robot_candidates.ndim != 5 or robot_candidates.shape[1] != 1:
            raise ValueError(
                "robot_candidates must have shape (batch, 1, robots, horizon, 4)"
            )
        batch = entity_history.shape[0]
        candidates = robot_candidates[:, 0, :, : self.config.horizon]
        if candidates.shape[-2] != self.config.horizon:
            raise ValueError("candidate horizon is shorter than the response checkpoint")
        current = robot_current[:, 0]
        if robot_mask is None:
            mask = torch.ones(
                batch,
                current.shape[1],
                device=current.device,
                dtype=torch.bool,
            )
        else:
            mask = robot_mask[:, 0].bool()
        neighbors = None if neighbor_history is None else neighbor_history[:, 0]
        neighbors_mask = None if neighbor_mask is None else neighbor_mask[:, 0].bool()
        response_batch = EntityResponseBatch(
            entity_history=entity_history[:, 0],
            entity_current=entity_history[:, 0, -1],
            robot_current=current,
            robot_candidates=candidates,
            robot_mask=mask,
            target_future_vel=torch.zeros(
                batch,
                self.config.horizon,
                2,
                device=current.device,
                dtype=current.dtype,
            ),
            high_interaction=torch.zeros(batch, device=current.device, dtype=torch.bool),
            scene_index=torch.arange(batch, device=current.device),
            entity_index=torch.zeros(batch, device=current.device, dtype=torch.long),
            dt=self.config.dt,
            neighbor_history=neighbors,
            neighbor_mask=neighbors_mask,
        )
        output = delayed_response_prediction(
            self.predictor, response_batch, self.response_delay_steps, include_pair_interactions
        )
        mean = output["mean"][:, None, None].expand(-1, 1, 2, -1, -1)
        log_std = output["log_std"][:, None, None].expand_as(mean)
        return {"mean": mean, "log_std": log_std}
