from __future__ import annotations

import math

import torch


def gaussian_nll_per_sample(mean: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mean.shape != target.shape or log_std.shape != target.shape:
        raise ValueError("mean, log_std, and target must have identical shapes")
    inv_var = torch.exp(-2.0 * log_std)
    nll = 0.5 * (target - mean).square() * inv_var + log_std + 0.5 * math.log(2.0 * math.pi)
    return nll.mean(dim=(-1, -2))


def trajectory_errors(mean_vel: torch.Tensor, target_vel: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_pos = torch.cumsum(mean_vel * dt, dim=1)
    target_pos = torch.cumsum(target_vel * dt, dim=1)
    displacement = torch.linalg.vector_norm(predicted_pos - target_pos, dim=-1)
    return displacement.mean(dim=1), displacement[:, -1]


def response_metrics(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
    dt: float,
) -> dict[str, torch.Tensor]:
    nll = gaussian_nll_per_sample(mean, log_std, target)
    ade, fde = trajectory_errors(mean, target, dt)
    return {"nll": nll, "ade": ade, "fde": fde}
