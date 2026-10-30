import torch

from tracer.response.closed_loop import PlanningCostConfig, expected_candidate_costs


def _inputs(mode_means):
    candidates = mode_means.shape[0]
    horizon = mode_means.shape[-2]
    reference = torch.zeros_like(mode_means)
    current = torch.zeros(candidates, 1, 4)
    robot_candidates = torch.zeros(candidates, 1, horizon, 4)
    robot_candidates[..., 0] = torch.arange(candidates, dtype=torch.float32)[:, None, None]
    config = PlanningCostConfig(
        task_steps=1,
        response_steps=horizon,
        risk_weight=0.0,
        task_speed_scale=1.7,
    )
    return reference, current, robot_candidates, config


def _score(mode_means, belief):
    reference, current, robot_candidates, config = _inputs(mode_means)
    return expected_candidate_costs(
        mode_means,
        belief,
        current,
        robot_candidates,
        reference,
        0.25,
        config,
        mode_log_stds=torch.full_like(mode_means, -1.0),
        task_direction=torch.tensor([1.0, 0.0]),
    )


def test_belief_only_changes_mode_dependent_cost_terms():
    mode_means = torch.zeros(2, 3, 2, 2)
    mode_means[0, 1, :, 0] = 1.0
    mode_means[1, 2, :, 0] = 1.0
    uniform = _score(mode_means, torch.tensor([1 / 3, 1 / 3, 1 / 3]))
    confident = _score(mode_means, torch.tensor([0.02, 0.96, 0.02]))

    torch.testing.assert_close(uniform["task"], confident["task"])
    torch.testing.assert_close(uniform["smoothness"], confident["smoothness"])
    assert not torch.allclose(uniform["human"], confident["human"])
    assert not torch.allclose(uniform["total"], confident["total"])


def test_mode_blind_response_makes_belief_inert():
    mode_means = torch.zeros(2, 3, 2, 2)
    mode_means[0, :, :, 0] = 0.25
    mode_means[1, :, :, 0] = 0.75
    uniform = _score(mode_means, torch.tensor([1 / 3, 1 / 3, 1 / 3]))
    confident = _score(mode_means, torch.tensor([0.02, 0.96, 0.02]))

    for name in ("total", "task", "safety", "human", "smoothness"):
        torch.testing.assert_close(uniform[name], confident[name])
