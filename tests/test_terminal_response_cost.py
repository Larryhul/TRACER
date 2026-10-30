import numpy as np
import pytest
import torch
from types import SimpleNamespace

from tracer.experiments.run_sicnav_tracer_loop import _fallback_tiebreak, _goal_response_mask
from tracer.response.closed_loop import PlanningCostConfig, expected_candidate_costs


def test_goal_response_mask_includes_arrival_and_ignores_later_departure():
    candidates = np.zeros((2, 4, 4))
    candidates[0, :, 0] = [.5, 1., 2., 3.]
    candidates[1, :, 0] = [-1., -.5, 0., .5]
    mask = _goal_response_mask(candidates, np.array([1., 0.]), .2)
    np.testing.assert_array_equal(mask, [[True, True, False, False], [True, True, True, True]])


def test_fallback_task_tiebreak_only_changes_equal_primary_safety_scores():
    primary = torch.tensor([.2, .2, .3, float("inf")])
    planning = torch.tensor([1., .5, -100., -1000.])
    assert _fallback_tiebreak(primary, planning) == 1
    primary[1] += .001
    assert _fallback_tiebreak(primary, planning) == 0
    with pytest.raises(ValueError, match="finite primary"):
        _fallback_tiebreak(torch.full((4,), float("inf")), planning)


@pytest.mark.parametrize("fallback", [False, True])
def test_terminal_scorer_aligns_long_generator_with_short_response_horizon(monkeypatch, fallback):
    import tracer.experiments.run_sicnav_tracer_loop as runner
    class Robot:
        radius = .25
        v_pref = .95
        def get_position(self):
            return (0., 0.)
        def get_velocity(self):
            return (0., 0.)
        def get_goal_position(self):
            return (1., 0.)
    means = torch.zeros(1, 2, 1, 3, 2)
    robot_candidates = torch.zeros(1, 2, 1, 3, 4)
    current = torch.zeros(1, 2, 1, 4)
    monkeypatch.setattr(runner, "_response_predictions", lambda *args: (means, means, means, current, robot_candidates))
    args = SimpleNamespace(
        device="cpu", safety_weight=2., human_cost_weight=.6, collision_margin=.65,
        execute_steps=1, hard_risk_filter=False, risk_samples=32, risk_weight=.5,
        smoothness_weight=0., model_dt=.25, env_dt=.25, kinematic_safety_margin=0.,
        static_safety_margin=0., goal_terminal_response=True, fallback_profile="none")
    global_candidates = np.zeros((2, 6, 4), dtype=np.float32)
    if fallback:
        monkeypatch.setattr(runner, "_kinematic_safe_mask", lambda *args: np.zeros(2, dtype=bool))
    prediction_cache = {}
    result = runner._score_candidates(None, None, None, global_candidates[None],
                                     global_candidates, Robot(), [], [], torch.zeros(1, 1), args,
                                     braking_safe=np.array([False, True]), prediction_cache=prediction_cache)
    assert result[0] == 1
    assert result[-1] == ("executor" if fallback else "none")
    def forbid_predictor(*args):
        raise AssertionError("same-state belief audit must reuse the identical predictions")
    monkeypatch.setattr(runner, "_response_predictions", forbid_predictor)
    cached = runner._score_candidates(None, None, None, global_candidates[None],
                                      global_candidates, Robot(), [], [], torch.zeros(1, 1), args,
                                      braking_safe=np.array([False, True]), cached_predictions=prediction_cache["values"])
    assert cached[0] == result[0]
    torch.testing.assert_close(cached[6], result[6])
    with pytest.raises(ValueError, match="braking invariant lost"):
        runner._score_candidates(None, None, None, global_candidates[None], global_candidates,
                                 Robot(), [], [], torch.zeros(1, 1), args, braking_safe=np.array([False, False]),
                                 cached_predictions=prediction_cache["values"])


@pytest.mark.parametrize("sampling", ["shared", "independent"])
def test_cost_does_not_penalize_human_response_after_goal_termination(sampling):
    means = torch.zeros(2, 1, 3, 2)
    means[..., 0] = .5
    reference = torch.zeros_like(means)
    current = torch.zeros(2, 1, 4)
    current[..., 0] = 2
    candidates = torch.zeros(2, 1, 3, 4)
    candidates[..., 0] = 2
    config = PlanningCostConfig(task_steps=1, risk_weight=.5, collision_margin=.6, temporal_risk_sampling=sampling)
    def score(response, valid=None):
        return expected_candidate_costs(response, torch.ones(1), current, candidates, reference, 1., config,
                                        mode_log_stds=torch.full_like(response, -20), response_valid_steps=valid)
    full = score(means)
    all_valid = score(means, torch.ones(2, 3, dtype=torch.bool))
    for name in full:
        torch.testing.assert_close(full[name], all_valid[name])
    valid = torch.tensor([[True, False, False], [True, True, True]])
    masked = score(means, valid)
    assert masked["violation_probability"][0] == 0
    assert full["violation_probability"][0] == 1
    altered = means.clone()
    altered[0, :, 1:] = 1000
    for name, value in score(altered, valid).items():
        torch.testing.assert_close(masked[name], value)
    with pytest.raises(ValueError, match="contiguous prefix"):
        score(means, torch.tensor([[True, False, True], [True, True, True]]))
