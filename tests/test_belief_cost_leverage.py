from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tracer.response.planning_support import belief_cost_adjustment
from tracer.experiments.planner_sensitivity import PLANNER_COMPONENTS


@pytest.mark.parametrize("leverage", [0.5, 1.0, 2.0, 4.0])
def test_uniform_belief_is_unchanged(leverage):
    costs = torch.tensor([0.1, 0.2, 0.3])
    torch.testing.assert_close(belief_cost_adjustment(costs, costs, leverage), torch.zeros_like(costs))


def test_default_preserves_original_cost_exactly():
    costs = torch.tensor([1e8, -1e-9, 0.12345])
    reference = torch.tensor([-1e8, 1e-9, -0.12345])
    assert torch.equal(costs + belief_cost_adjustment(costs, reference), costs)


@pytest.mark.parametrize("leverage", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_leverage_is_rejected(leverage):
    with pytest.raises(ValueError, match="finite and positive"):
        belief_cost_adjustment(torch.zeros(2), torch.zeros(2), leverage)


def test_adjustment_never_operates_on_masked_infinities():
    with pytest.raises(ValueError, match="finite unmasked"):
        belief_cost_adjustment(torch.tensor([float("inf")]), torch.tensor([float("inf")]), 2.0)


@pytest.mark.parametrize("admissible", [[True, True], [True, False]])
def test_external_leverage_changes_only_scores_not_safety(monkeypatch, admissible):
    import tracer.experiments.run_sicnav_tracer_loop as runner

    class Robot:
        radius = 0.25
        v_pref = 1.0

        def get_position(self):
            return (0.0, 0.0)

        def get_velocity(self):
            return (0.0, 0.0)

        def get_goal_position(self):
            return (10.0, 0.0)

    means = torch.zeros(1, 2, 2, 2, 2)
    current = torch.zeros(1, 2, 1, 4)
    paths = torch.zeros(1, 2, 1, 2, 4)
    predictions = (means, means, means, current, paths)
    monkeypatch.setattr(runner, "_kinematic_safe_mask", lambda *args: np.ones(2, dtype=bool))
    monkeypatch.setattr(runner, "_static_obstacle_safe_mask", lambda *args: np.ones(2, dtype=bool))
    monkeypatch.setattr(runner, "_sage_dynamic_safe_mask", lambda *args: np.ones(2, dtype=bool))
    monkeypatch.setattr(runner, "_kinematic_surface_clearance", lambda *args: np.ones(2))

    def fake_cost(mode_means, belief, *args, **kwargs):
        human = torch.tensor([0.0, 0.1]) - torch.tensor([0.0, 0.1]) * (belief[0] - 0.5)
        zero = torch.zeros(2)
        return {"task": zero, "safety": zero, "human": human, "smoothness": zero,
                "violation_probability": zero, "safe_mask": torch.ones(2, dtype=torch.bool)}

    monkeypatch.setattr(runner, "expected_candidate_costs", fake_cost)
    args = SimpleNamespace(device="cpu", safety_weight=2.0, human_cost_weight=1.0,
                           collision_margin=0.65, execute_steps=1, hard_risk_filter=False,
                           risk_samples=4, risk_weight=0.5, smoothness_weight=0.0,
                           model_dt=0.25, env_dt=0.25, kinematic_safety_margin=0.0,
                           static_safety_margin=0.0, fallback_profile="none")
    results = []
    for leverage in [1.0, 4.0]:
        args.belief_cost_leverage = leverage
        diagnostics = {}
        results.append(runner._score_candidates(
            None, None, None, np.zeros((1, 2, 2, 4)), np.zeros((2, 2, 4)), Robot(), [], [],
            torch.tensor([[0.9, 0.1]]).log(), args, executor_safe=np.asarray(admissible),
            cached_predictions=predictions, diagnostics=diagnostics,
        ))
        components = [torch.as_tensor(diagnostics[name]) for name in PLANNER_COMPONENTS]
        torch.testing.assert_close(sum(components), results[-1][5])
        assert all(np.asarray(value).shape == (2,) for value in diagnostics.values())
    assert results[0][0] == 0
    assert results[1][0] == (1 if admissible[1] else 0)
    torch.testing.assert_close(results[0][7], results[1][7])
