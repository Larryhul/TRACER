from __future__ import annotations

import pytest

from sage.experiments.paired_binary_uncertainty import paired_binary_jeffreys


def test_all_ties_retain_population_uncertainty() -> None:
    result = paired_binary_jeffreys([1.0] * 16, [1.0] * 16, seed=19, samples=20_000)
    lower, upper = result["jeffreys_posterior"]["credible_interval_95"]
    assert result["estimate"] == 0.0
    assert lower < 0 < upper
    assert result["counts"]["both_one"] == 16


def test_consistent_discordance_has_positive_difference() -> None:
    result = paired_binary_jeffreys([1.0] * 20, [0.0] * 20, seed=23, samples=20_000)
    lower, _ = result["jeffreys_posterior"]["credible_interval_95"]
    assert result["estimate"] == 1.0
    assert lower > 0.7
    assert result["jeffreys_posterior"]["probability_first_greater"] > 0.999


@pytest.mark.parametrize(
    "first,second,error",
    [([], [], "empty"), ([0], [0, 1], "equal-length"), ([2], [0], "binary")],
)
def test_invalid_inputs_are_rejected(first, second, error) -> None:
    with pytest.raises(ValueError, match=error):
        paired_binary_jeffreys(first, second, seed=1)
