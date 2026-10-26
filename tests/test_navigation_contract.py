from types import SimpleNamespace

import pytest

from sage.experiments.navigation_contract import navigation_outcome, validate_response_clock


def test_budget_exhaustion_is_timeout_without_environment_event():
    outcome = navigation_outcome(False, False, 60, 60, 0, 0)
    assert outcome["timeout"] == 1.0
    assert outcome["budget_exhausted"] == 1.0
    assert outcome["environment_timeout"] == 0.0


def test_arrival_after_collision_is_not_collision_free_success():
    outcome = navigation_outcome(True, False, 60, 60, 1, 0)
    assert outcome["timeout"] == 0.0
    assert outcome["success"] == 1.0
    assert outcome["collision_free_success"] == 0.0


def test_inference_time_rescaling_does_not_hide_decoder_clock_mismatch():
    model = SimpleNamespace(config=SimpleNamespace(dt=.25), training_dt=.4)
    with pytest.raises(ValueError, match="decoder training dt"):
        validate_response_clock(model, .25)
    assert validate_response_clock(model, .25, True)["temporal_transfer"]
    model.training_dt = .25
    assert not validate_response_clock(model, .25)["temporal_transfer"]
