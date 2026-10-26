import pytest
import torch

from sage.response.closed_loop import PlanningCostConfig, expected_candidate_costs


def test_short_response_cost_ignores_unreliable_suffix_without_changing_task_cost():
    means = torch.zeros(2, 1, 4, 2)
    references = torch.zeros_like(means)
    log_stds = torch.full_like(means, -5.0)
    current = torch.zeros(2, 1, 4)
    current[..., 0] = 2.0
    candidates = torch.zeros(2, 1, 4, 4)
    candidates[..., 0] = 2.0
    settings = dict(task_steps=2, risk_weight=0.5, risk_samples=16, violation_probability_alpha=0.1)

    def evaluate(response_steps):
        return expected_candidate_costs(
            means, torch.ones(1), current, candidates, references, 0.4,
            PlanningCostConfig(**settings, response_steps=response_steps),
            mode_log_stds=log_stds, task_direction=torch.tensor([1.0, 0.0]),
        )

    prefix = evaluate(2)
    full = evaluate(None)
    means[0, :, 2:, 0] = 2.5
    candidates[0, :, 2:, 0] = 1.0
    altered_prefix = evaluate(2)
    altered_full = evaluate(None)
    for field in ("total", "task", "safety", "human", "violation_probability", "safe_mask"):
        torch.testing.assert_close(prefix[field], altered_prefix[field])
    torch.testing.assert_close(full["task"], altered_full["task"])
    assert altered_full["violation_probability"][0] > full["violation_probability"][0]
    assert not altered_full["safe_mask"][0]
    assert altered_prefix["safe_mask"][0]


@pytest.mark.parametrize("steps", [0, 5])
def test_response_cost_horizon_must_fit_predictions(steps):
    means = torch.zeros(1, 1, 4, 2)
    with pytest.raises(ValueError, match="response_steps"):
        expected_candidate_costs(
            means, torch.ones(1), torch.zeros(1, 1, 4), torch.zeros(1, 1, 4, 4),
            means, 0.4, PlanningCostConfig(response_steps=steps),
        )
