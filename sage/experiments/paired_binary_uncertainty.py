"""Uncertainty summaries for paired binary policy outcomes."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def paired_binary_jeffreys(
    first: Sequence[float],
    second: Sequence[float],
    *,
    seed: int,
    samples: int = 100_000,
) -> dict[str, Any]:
    """Summarize ``first - second`` with a multinomial Jeffreys posterior."""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.ndim != 1 or second_array.ndim != 1 or len(first_array) != len(second_array):
        raise ValueError("paired binary outcomes must be equal-length vectors")
    if not len(first_array):
        raise ValueError("paired binary outcomes cannot be empty")
    if samples <= 0:
        raise ValueError("posterior sample count must be positive")
    if not np.isin(first_array, (0.0, 1.0)).all() or not np.isin(second_array, (0.0, 1.0)).all():
        raise ValueError("paired outcomes must be binary")

    counts = np.asarray([
        np.sum((first_array == 0) & (second_array == 0)),
        np.sum((first_array == 0) & (second_array == 1)),
        np.sum((first_array == 1) & (second_array == 0)),
        np.sum((first_array == 1) & (second_array == 1)),
    ], dtype=np.int64)
    rng = np.random.default_rng(seed)
    posterior = rng.dirichlet(counts.astype(np.float64) + 0.5, size=samples)
    differences = posterior[:, 2] - posterior[:, 1]
    lower, upper = np.quantile(differences, (0.025, 0.975))
    return {
        "estimand": "paired probability difference: first minus second",
        "estimate": float((counts[2] - counts[1]) / len(first_array)),
        "pairs": int(len(first_array)),
        "counts": {
            "both_zero": int(counts[0]),
            "first_zero_second_one": int(counts[1]),
            "first_one_second_zero": int(counts[2]),
            "both_one": int(counts[3]),
        },
        "jeffreys_posterior": {
            "prior": "Dirichlet(0.5, 0.5, 0.5, 0.5)",
            "samples": int(samples),
            "credible_interval_95": [float(lower), float(upper)],
            "probability_first_greater": float(np.mean(differences > 0)),
            "probability_first_less": float(np.mean(differences < 0)),
        },
        "claim_boundary": "Bayesian paired binary uncertainty; not a frequentist confidence interval or noninferiority test",
    }
