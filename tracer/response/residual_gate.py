"""Confidence-gated interpolation for additive and joint response predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class ResidualGateConfig:
    """Configuration for confidence and uncertainty based residual gating."""

    confidence_threshold: float = 0.35
    confidence_scale: float = 0.10
    uncertainty_threshold: float = 0.45
    uncertainty_scale: float = 0.08
    min_gate: float = 0.0
    max_gate: float = 1.0

    def __post_init__(self) -> None:
        finite_values = (
            self.confidence_threshold,
            self.confidence_scale,
            self.uncertainty_threshold,
            self.uncertainty_scale,
            self.min_gate,
            self.max_gate,
        )
        if not all(torch.isfinite(torch.tensor(value)) for value in finite_values):
            raise ValueError("residual gate configuration must be finite")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be positive")
        if self.uncertainty_threshold < 0.0:
            raise ValueError("uncertainty_threshold must be non-negative")
        if self.uncertainty_scale <= 0.0:
            raise ValueError("uncertainty_scale must be positive")
        if not 0.0 <= self.min_gate <= 1.0:
            raise ValueError("min_gate must be in [0, 1]")
        if not 0.0 <= self.max_gate <= 1.0:
            raise ValueError("max_gate must be in [0, 1]")
        if self.min_gate > self.max_gate:
            raise ValueError("min_gate must not exceed max_gate")


def _validate_float_tensor(value: Tensor, name: str, min_ndim: int = 0) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating-point dtype")
    if value.ndim < min_ndim:
        raise ValueError(f"{name} must have at least {min_ndim} dimensions")
    if not torch.all(torch.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


def _normalise_belief(belief: Tensor) -> Tensor:
    _validate_float_tensor(belief, "belief", min_ndim=1)
    if belief.shape[-1] == 0:
        raise ValueError("belief must contain at least one mode")
    if torch.any(belief < 0.0):
        raise ValueError("belief must be non-negative")
    mass = belief.sum(dim=-1, keepdim=True)
    if torch.any(mass <= torch.finfo(belief.dtype).tiny):
        raise ValueError("each belief distribution must have positive mass")
    return belief / mass


def compute_belief_confidence(belief: Tensor) -> Tensor:
    """Return normalized max-probability confidence for every belief entry.

    The final dimension of ``belief`` is interpreted as the mode dimension.
    Inputs may be normalized probabilities or non-negative belief weights; the
    latter are normalized internally. The returned tensor has shape
    ``belief.shape[:-1]`` and maps uniform belief to zero and one-hot belief to
    one.
    """

    probabilities = _normalise_belief(belief)
    num_modes = probabilities.shape[-1]
    if num_modes == 1:
        return torch.ones_like(probabilities[..., 0])
    uniform_probability = 1.0 / float(num_modes)
    confidence = (probabilities.max(dim=-1).values - uniform_probability) / (
        1.0 - uniform_probability
    )
    return confidence.clamp(0.0, 1.0)


def _reduce_response_uncertainty(
    response_uncertainty: Tensor, leading_shape: torch.Size
) -> Tensor:
    _validate_float_tensor(response_uncertainty, "response_uncertainty")
    if torch.any(response_uncertainty < 0.0):
        raise ValueError("response_uncertainty must be non-negative")

    if len(leading_shape) == 0:
        return response_uncertainty.mean()

    if response_uncertainty.ndim < len(leading_shape):
        try:
            return torch.broadcast_to(response_uncertainty, leading_shape)
        except RuntimeError as error:
            raise ValueError(
                "response_uncertainty is not broadcastable to belief leading dimensions"
            ) from error

    uncertainty_leading = response_uncertainty.shape[: len(leading_shape)]
    for actual, target in zip(uncertainty_leading, leading_shape):
        if actual not in (1, target):
            raise ValueError(
                "response_uncertainty leading dimensions must match or broadcast "
                "against belief leading dimensions"
            )

    expanded = response_uncertainty.expand(
        *leading_shape, *response_uncertainty.shape[len(leading_shape) :]
    )
    if expanded.ndim == len(leading_shape):
        return expanded
    reduction_dims = tuple(range(len(leading_shape), expanded.ndim))
    return expanded.mean(dim=reduction_dims)


def compute_joint_residual_gate(
    belief: Tensor,
    response_uncertainty: Tensor,
    config: Optional[ResidualGateConfig] = None,
) -> Tensor:
    """Compute a ``[0, 1]`` gate for a joint response residual.

    ``belief`` has shape ``(..., modes)``. ``response_uncertainty`` may be a
    scalar, have shape ``belief.shape[:-1]``, or have matching leading
    batch/time/entity dimensions followed by arbitrary response dimensions.
    Extra uncertainty dimensions are averaged before gating. Uncertainty is
    expected to be a non-negative standard-deviation-like scale, not a log
    standard deviation.
    """

    gate_config = config or ResidualGateConfig()
    confidence = compute_belief_confidence(belief)
    uncertainty = _reduce_response_uncertainty(
        response_uncertainty, belief.shape[:-1]
    )
    confidence_term = torch.sigmoid(
        (confidence - gate_config.confidence_threshold)
        / gate_config.confidence_scale
    )
    uncertainty_term = torch.sigmoid(
        (gate_config.uncertainty_threshold - uncertainty)
        / gate_config.uncertainty_scale
    )
    gate = confidence_term * uncertainty_term
    gate = gate_config.min_gate + (
        gate_config.max_gate - gate_config.min_gate
    ) * gate
    return gate.clamp(gate_config.min_gate, gate_config.max_gate)


def _expand_gate_for_prediction(gate: Tensor, prediction_shape: torch.Size) -> Tensor:
    if gate.ndim > len(prediction_shape):
        raise ValueError("gate cannot have more dimensions than the predictions")

    padded_shape = tuple(gate.shape) + (1,) * (len(prediction_shape) - gate.ndim)
    for actual, target in zip(padded_shape, prediction_shape):
        if actual not in (1, target):
            raise ValueError(
                "gate leading dimensions must match or broadcast against predictions"
            )
    return gate.reshape(padded_shape).expand(prediction_shape)


def apply_joint_residual_gate(
    additive_mean: Tensor, joint_mean: Tensor, gate: Tensor
) -> Tensor:
    """Safely interpolate from an additive mean to a joint mean.

    The two predictions must have the same shape. A gate with shape matching
    their leading batch/time/entity dimensions is automatically expanded over
    remaining response dimensions. Since the gate is clamped to ``[0, 1]``,
    every output element lies between the corresponding additive and joint
    elements.
    """

    _validate_float_tensor(additive_mean, "additive_mean", min_ndim=1)
    _validate_float_tensor(joint_mean, "joint_mean", min_ndim=1)
    _validate_float_tensor(gate, "gate")
    if additive_mean.shape != joint_mean.shape:
        raise ValueError("additive_mean and joint_mean must have identical shapes")
    if torch.any(gate < 0.0) or torch.any(gate > 1.0):
        raise ValueError("gate values must be in [0, 1]")
    if additive_mean.device != joint_mean.device or additive_mean.device != gate.device:
        raise ValueError("additive_mean, joint_mean, and gate must share a device")

    expanded_gate = _expand_gate_for_prediction(gate, additive_mean.shape)
    return additive_mean + expanded_gate * (joint_mean - additive_mean)


__all__ = [
    "ResidualGateConfig",
    "apply_joint_residual_gate",
    "compute_belief_confidence",
    "compute_joint_residual_gate",
]
