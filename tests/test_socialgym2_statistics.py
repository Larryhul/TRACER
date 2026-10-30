from __future__ import annotations

import pytest

from tracer.experiments.aggregate_socialgym2_tracer import _holm_adjust, _two_sided_sign_test, _wilson_interval


def test_binary_all_success_interval_retains_uncertainty() -> None:
    interval = _wilson_interval([1.0] * 4)
    assert 0.5 < interval["ci95_low"] < 0.6
    assert interval["ci95_high"] == pytest.approx(1.0)


def test_binary_zero_collision_interval_retains_uncertainty() -> None:
    interval = _wilson_interval([0.0] * 4)
    assert interval["ci95_low"] == pytest.approx(0.0)
    assert 0.4 < interval["ci95_high"] < 0.5


def test_holm_controls_navigation_comparison_family() -> None:
    assert _holm_adjust({"first": 0.01, "second": 0.04, "third": 0.03}) == pytest.approx(
        {"first": 0.03, "second": 0.06, "third": 0.06}
    )


def test_sign_test_does_not_count_ties_as_evidence() -> None:
    assert _two_sided_sign_test([0.0] * 10)["p_two_sided"] == 1.0
    assert _two_sided_sign_test([1.0] * 6)["p_two_sided"] == pytest.approx(0.03125)
