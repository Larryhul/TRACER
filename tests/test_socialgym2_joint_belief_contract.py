from __future__ import annotations

from itertools import product
from types import SimpleNamespace

import numpy as np
import torch

import tracer.experiments.run_socialgym2_tracer_loop as socialgym_loop
from tracer.experiments.run_socialgym2_tracer_loop import HumanTrack
from tracer.response.closed_loop import PlanningCostConfig, expected_candidate_costs


def _human_costs(
    mode_means: torch.Tensor,
    belief: torch.Tensor,
    robot_current: torch.Tensor,
    robot_candidates: torch.Tensor,
    reference_means: torch.Tensor,
    mode_log_stds: torch.Tensor,
) -> dict[str, torch.Tensor]:
    task_direction = torch.zeros(robot_candidates.shape[1], 2)
    task_direction[:, 0] = 1.0
    return expected_candidate_costs(
        mode_means,
        belief,
        robot_current,
        robot_candidates,
        reference_means,
        dt=0.4,
        config=PlanningCostConfig(
            task_weight=1.0,
            safety_weight=2.0,
            human_weight=0.6,
            collision_margin=0.65,
            task_steps=1,
            violation_probability_alpha=0.35,
            risk_samples=8,
            risk_weight=0.2,
            smoothness_weight=0.1,
        ),
        mode_log_stds=mode_log_stds,
        task_direction=task_direction,
    )


def test_factorized_marginals_equal_explicit_joint_mode_expectation() -> None:
    generator = torch.Generator().manual_seed(20270906)
    humans, candidates, modes, robots, horizon = 2, 8, 3, 3, 3
    means = 0.35 * torch.randn(
        humans, candidates, modes, horizon, 2, generator=generator
    )
    references = 0.15 * torch.randn(
        humans, candidates, modes, horizon, 2, generator=generator
    )
    log_stds = torch.full_like(means, -1.2)
    robot_current = 0.25 * torch.randn(
        humans, candidates, robots, 4, generator=generator
    )
    robot_candidates = 0.25 * torch.randn(
        humans, candidates, robots, horizon, 4, generator=generator
    )
    beliefs = torch.tensor([[0.65, 0.25, 0.10], [0.15, 0.35, 0.50]])

    marginalized = torch.stack(
        [
            _human_costs(
                means[human],
                beliefs[human],
                robot_current[human],
                robot_candidates[human],
                references[human],
                log_stds[human],
            )["unconstrained_total"]
            for human in range(humans)
        ]
    ).mean(dim=0)

    explicitly_joint = torch.zeros(candidates)
    for assignment in product(range(modes), repeat=humans):
        probability = torch.prod(
            torch.stack([beliefs[human, mode] for human, mode in enumerate(assignment)])
        )
        assignment_cost = torch.stack(
            [
                _human_costs(
                    means[human],
                    torch.nn.functional.one_hot(
                        torch.tensor(mode), num_classes=modes
                    ).to(torch.float32),
                    robot_current[human],
                    robot_candidates[human],
                    references[human],
                    log_stds[human],
                )["unconstrained_total"]
                for human, mode in enumerate(assignment)
            ]
        ).mean(dim=0)
        explicitly_joint += probability * assignment_cost

    torch.testing.assert_close(marginalized, explicitly_joint, rtol=1e-5, atol=1e-6)


def test_deployed_safety_rule_intersects_per_human_masks(monkeypatch) -> None:
    candidates, modes, robots, horizon = 8, 3, 3, 3
    response_shape = (2, candidates, modes, horizon, 2)
    current_shape = (2, candidates, robots, 4)
    monkeypatch.setattr(
        socialgym_loop,
        "_response_predictions",
        lambda *args, **kwargs: (
            np.zeros(response_shape, dtype=np.float32),
            np.zeros(response_shape, dtype=np.float32),
            np.zeros(response_shape, dtype=np.float32),
            np.zeros(current_shape, dtype=np.float32),
        ),
    )
    calls = 0

    def fake_costs(*args, **kwargs):
        nonlocal calls
        human = calls % 2
        calls += 1
        safe = torch.zeros(candidates, dtype=torch.bool)
        safe[human + 1] = True
        safe[0] = True
        total = torch.arange(candidates, dtype=torch.float32)
        total[1:3] = -10.0
        return {
            "unconstrained_total": total,
            "safe_mask": safe,
            "violation_probability": (~safe).to(torch.float32),
        }

    monkeypatch.setattr(socialgym_loop, "expected_candidate_costs", fake_costs)
    tracks = [
        HumanTrack(
            track_id=index,
            position=np.zeros(2, dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
        )
        for index in range(2)
    ]
    selected, *_rest, diagnostics = socialgym_loop._select_action(
        "tracer-persistent",
        SimpleNamespace(config=SimpleNamespace(num_modes=modes)),
        tracks,
        np.zeros((robots, 2), dtype=np.float32),
        np.zeros((robots, 2), dtype=np.float32),
        np.ones((robots, 2), dtype=np.float32),
        np.zeros((candidates, robots), dtype=np.int64),
        np.zeros((candidates, robots, horizon, 2), dtype=np.float32),
        np.zeros((candidates, robots, horizon, 2), dtype=np.float32),
        history=2,
        model_dt=0.4,
        args=SimpleNamespace(
            env_dt=0.4,
            device="cpu",
            safety_weight=2.0,
            human_cost_weight=0.6,
            collision_margin=0.65,
            risk_alpha=0.35,
            hard_risk_filter=True,
            risk_samples=8,
            risk_weight=0.2,
            smoothness_weight=0.1,
            belief_selection_audit=False,
            belief_cost_leverage=1.0,
        ),
    )

    assert calls == 4
    assert selected == 0
    assert diagnostics["safe_fraction"] == 1.0 / candidates
