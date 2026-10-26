import pytest
import torch

from sage.experiments.refine_component_value_pooling import choose_pooled_value_head, pooled_value_coefficients
from sage.experiments.train_component_value import value_error
from sage.response.continuation_value import FrozenContinuationValue


def test_partial_pooling_uses_train_only_normalization_and_preserves_parent_option():
    torch.manual_seed(151)
    center, scale = torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
    coefficients = torch.tensor([[0., 1., 0., 0.], [0., 1., 4., 0.], [0., 1., -4., 0.]], dtype=torch.float64)
    parent = FrozenContinuationValue(center, scale, coefficients, 8, .4, 3)
    features = torch.randn(120, 3, dtype=torch.float64)
    data = {"features": features, "modes": torch.arange(120) % 3, "remaining": torch.ones(120, dtype=torch.float64) * 4, "returns": features[:, 0] * 4}
    validation = {name: value.clone() for name, value in data.items()}
    head, selected, trials = choose_pooled_value_head(data, validation, parent, .001, [0., .25, .5, .75, 1.])
    assert selected["mode_specific_strength"] == 0.
    assert selected["validation_mse"] < value_error(parent, validation)
    torch.testing.assert_close(head.center, center)
    torch.testing.assert_close(head.scale, scale)
    torch.testing.assert_close(head.coefficients[0], head.coefficients[1])
    exact, _, _ = choose_pooled_value_head(data, validation, parent, .001, [1.])
    torch.testing.assert_close(exact.coefficients, parent.coefficients, rtol=0, atol=0)
    baseline = pooled_value_coefficients(data, parent, .001)
    validation["features"] += 100.
    torch.testing.assert_close(pooled_value_coefficients(data, parent, .001), baseline)
    assert len(trials) == 5


def test_invalid_pooling_parameters_are_rejected():
    parent = FrozenContinuationValue(torch.zeros(2), torch.ones(2), torch.zeros(3, 3), 8, .4, 3)
    data = {"features": torch.zeros(6, 2), "modes": torch.arange(6) % 3, "remaining": torch.ones(6), "returns": torch.zeros(6)}
    with pytest.raises(ValueError, match="regularization"):
        pooled_value_coefficients(data, parent, float("nan"))
    with pytest.raises(ValueError, match="strengths"):
        choose_pooled_value_head(data, data, parent, 1., [1.1])
