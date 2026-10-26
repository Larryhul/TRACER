import pytest
import torch

from sage.experiments.audit_external_mode_separation import mode_effect_metrics


def test_mode_audit_distinguishes_truthful_effects_from_artificial_spread():
    target = torch.zeros(2, 3, 2, 2)
    target[0, :, :, 0] = torch.tensor([0., 1., 2.])[:, None]
    correct = mode_effect_metrics(target, target, torch.zeros_like(target))
    assert correct["informative"]["groups"] == 1
    assert correct["informative"]["effect_rmse"] == 0.
    assert correct["informative"]["mode_recovery_accuracy"] == 1.
    artificial = target.clone()
    artificial[1] = target[0]
    altered = mode_effect_metrics(artificial, target, torch.zeros_like(target))
    assert altered["invariant"]["true_spread"] == 0.
    assert altered["invariant"]["predicted_spread"] > 0.
    assert altered["invariant"]["effect_rmse"] > 0.


def test_mode_effect_error_ignores_common_baseline_translation():
    target = torch.randn(3, 3, 2, 2)
    measured = mode_effect_metrics(target + 10., target, torch.zeros_like(target))
    assert measured["all"]["effect_rmse"] == pytest.approx(0., abs=1e-6)
    assert measured["all"]["conditional_rmse"] == pytest.approx(10.)
