import pytest

from tracer.experiments.replay_external_protocol import replay_parameters


def _metadata():
    return {"parameters": {"checkpoint_dir": "old", "output_dir": "old_output", "seeds": [2027, 2031],
                           "cases": 24, "case_start": 88, "policies": ["persistent", "uniform"],
                           "static_route_weight": 2., "motion_profile": "sicnav-unicycle",
                           "new_future_controller_setting": {"nested": 5}}}


def test_protocol_replay_preserves_even_unknown_controller_settings():
    metadata = _metadata()
    parameters = replay_parameters(metadata, "new", "new_output", seeds=[2027], cases=8)
    assert parameters["static_route_weight"] == 2.
    assert parameters["motion_profile"] == "sicnav-unicycle"
    parameters["new_future_controller_setting"]["nested"] = 8
    assert metadata["parameters"]["new_future_controller_setting"]["nested"] == 5
    assert metadata["parameters"]["checkpoint_dir"] == "old"


@pytest.mark.parametrize("changes", [{"seeds": [999]}, {"cases": 0}, {"case_start": 87},
                                     {"policies": ["unseen"]}, {"cases": 25}])
def test_protocol_replay_rejects_undisclosed_evaluation_support(changes):
    with pytest.raises(ValueError, match="subset"):
        replay_parameters(_metadata(), "new", "new_output", **changes)


def test_explicit_leverage_override_does_not_modify_reference():
    metadata = _metadata()
    parameters = replay_parameters(metadata, "new", "new_output", belief_cost_leverage=2.0)
    assert parameters["belief_cost_leverage"] == 2.0
    assert "belief_cost_leverage" not in metadata["parameters"]


@pytest.mark.parametrize("leverage", [-1.0, 0.0, float("nan"), float("inf")])
def test_protocol_replay_rejects_invalid_leverage(leverage):
    with pytest.raises(ValueError, match="finite and positive"):
        replay_parameters(_metadata(), "new", "new_output", belief_cost_leverage=leverage)
