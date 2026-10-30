from __future__ import annotations

import torch


def segment_context_features(segments: torch.Tensor) -> torch.Tensor:
    """Encode entity-relative line segments without endpoint-order dependence."""
    if segments.shape[-1] != 4 or not torch.isfinite(segments).all():
        raise ValueError("static segments must contain finite endpoint coordinates")
    first, second = segments[..., :2], segments[..., 2:]
    direction = second - first
    squared_length = direction.square().sum(-1, keepdim=True)
    fraction = (-(first * direction).sum(-1, keepdim=True) / squared_length.clamp_min(1e-8)).clamp(0, 1)
    closest = first + fraction * direction
    orientation = torch.cat([direction[..., :1].square() - direction[..., 1:].square(),
                             2 * direction[..., :1] * direction[..., 1:]], -1) / squared_length.clamp_min(1e-8)
    return torch.cat([(first + second) / 2, closest, orientation,
                      squared_length.sqrt(), closest.norm(dim=-1, keepdim=True)], -1)
