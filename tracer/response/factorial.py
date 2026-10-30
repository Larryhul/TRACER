from __future__ import annotations

import math

import torch


FACTORIAL_INTERVENTIONS = 4


def pairwise_factorial_interactions(
    values: torch.Tensor,
    num_robots: int,
    dim: int = 0,
) -> torch.Tensor:
    """Return pairwise interaction contrasts from a complete ``2**R`` design.

    Intervention ``u`` is encoded by the bits of its leading index. For each
    robot pair, the contrast averages over all remaining robot bits, so the
    result isolates the pair interaction and also works when ``R > 2``.
    """

    if num_robots < 2:
        raise ValueError("pairwise interactions require at least two robots")
    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    dim = dim % values.ndim
    expected = 2 ** int(num_robots)
    if values.shape[dim] != expected:
        raise ValueError(
            f"values must have {expected} interventions for R={num_robots}, "
            f"got {tuple(values.shape)}"
        )

    moved = values.movedim(dim, 0)
    all_ids = torch.arange(expected, device=values.device)
    contrasts = []
    for first in range(num_robots):
        for second in range(first + 1, num_robots):
            other_bits = [bit for bit in range(num_robots) if bit not in {first, second}]
            terms = []
            for first_bit, second_bit in ((0, 0), (1, 0), (0, 1), (1, 1)):
                ids = all_ids.clone()
                ids = ids & ~(1 << first)
                ids = ids & ~(1 << second)
                ids = ids | (first_bit << first) | (second_bit << second)
                # The selected ids contain every setting of the complement bits.
                if other_bits:
                    keep = torch.zeros_like(ids, dtype=torch.bool)
                    for bit_mask in range(2 ** len(other_bits)):
                        selected = (first_bit << first) | (second_bit << second)
                        for offset, bit in enumerate(other_bits):
                            selected |= ((bit_mask >> offset) & 1) << bit
                        keep |= ids.eq(selected)
                    ids = ids[keep]
                else:
                    ids = ids[:1]
                terms.append(moved[ids].mean(dim=0))
            contrasts.append(terms[3] - terms[1] - terms[2] + terms[0])
    return torch.stack(contrasts, dim=0)


def factorial_second_difference(values: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Return the 2x2 interaction contrast Y11 - Y10 - Y01 + Y00.

    The leading dimension must follow the intervention bit order used by
    CausalTeamResponseDataset: 00, 10, 01, 11.
    """

    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    dim = dim % values.ndim
    if values.shape[dim] != FACTORIAL_INTERVENTIONS:
        raise ValueError(
            "values must have four interventions on the selected dimension "
            f"in 00,10,01,11 order, got {tuple(values.shape)}"
        )
    return (
        values.select(dim, 3)
        - values.select(dim, 1)
        - values.select(dim, 2)
        + values.select(dim, 0)
    )


def interaction_recovery_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Measure recovery of a known factorial interaction contrast."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must share a shape, got {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}"
        )
    if prediction.numel() == 0:
        raise ValueError("interaction metrics require at least one value")

    prediction = prediction.detach().double().reshape(-1)
    target = target.detach().double().reshape(-1)
    residual = prediction - target
    target_energy = target.square().mean()
    prediction_energy = prediction.square().mean()
    mse = residual.square().mean()

    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    pearson_denominator = torch.sqrt(
        centered_prediction.square().sum() * centered_target.square().sum()
    )
    cosine_denominator = torch.sqrt(prediction.square().sum() * target.square().sum())

    target_rms = torch.sqrt(target_energy)
    prediction_rms = torch.sqrt(prediction_energy)
    rmse = torch.sqrt(mse)
    target_energy_sum = target.square().sum()
    has_target_signal = float(target_rms) > math.sqrt(eps)

    return {
        "num_values": float(target.numel()),
        "target_rms": float(target_rms),
        "prediction_rms": float(prediction_rms),
        "rmse": float(rmse),
        "mae": float(residual.abs().mean()),
        "bias": float(residual.mean()),
        "pearson": (
            float((centered_prediction * centered_target).sum() / pearson_denominator)
            if float(pearson_denominator) > eps
            else float("nan")
        ),
        "cosine": (
            float((prediction * target).sum() / cosine_denominator)
            if float(cosine_denominator) > eps
            else float("nan")
        ),
        "zero_baseline_r2": (
            1.0 - float(residual.square().sum() / target_energy_sum)
            if has_target_signal
            else float("nan")
        ),
        "relative_rmse": (
            float(rmse / target_rms) if has_target_signal else float("nan")
        ),
        "energy_ratio": (
            float(prediction_rms / target_rms)
            if has_target_signal
            else float("nan")
        ),
    }
