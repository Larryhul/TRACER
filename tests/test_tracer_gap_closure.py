from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from sage.belief.filter import (
    adaptive_categorical_update,
    apply_belief_forgetting,
    detect_change_point,
    hazard_belief_update,
    uniform_log_belief,
)
from sage.experiments.run_tracer_gap_closure import _static_safe_mask
from sage.response.closed_loop import CandidatePool


def test_forgetting_rate_one_returns_uniform_belief() -> None:
    belief = torch.log(torch.tensor([0.9, 0.08, 0.02]))
    forgotten = apply_belief_forgetting(belief, 1.0)
    torch.testing.assert_close(forgotten.exp(), torch.full((3,), 1.0 / 3.0))


def test_forgetting_rate_zero_preserves_posterior() -> None:
    belief = torch.log(torch.tensor([0.9, 0.08, 0.02]))
    preserved = apply_belief_forgetting(belief, 0.0)
    torch.testing.assert_close(preserved.exp(), belief.exp())


def test_hazard_update_preserves_normalization() -> None:
    prior = uniform_log_belief(3)
    likelihood = torch.tensor([3.0, -1.0, -2.0])
    posterior = hazard_belief_update(prior, likelihood, 0.2)
    torch.testing.assert_close(posterior.exp().sum(), torch.tensor(1.0))
    assert int(posterior.argmax()) == 0


def test_change_point_detector_uses_predictive_surprise() -> None:
    prior = torch.log(torch.tensor([0.98, 0.01, 0.01]))
    surprising = torch.tensor([-8.0, -8.0, -8.0])
    expected = torch.tensor([3.0, -1.0, -2.0])
    assert bool(detect_change_point(prior, surprising, 5.0))
    assert not bool(detect_change_point(prior, expected, 5.0))


def test_adaptive_update_reopens_a_stale_mode_after_surprise() -> None:
    prior = torch.log(torch.tensor([0.999999, 0.000001, 0.000000001]))
    likelihood = torch.tensor([-9.0, 4.0, -9.0])
    posterior, hazard, surprise, changed = adaptive_categorical_update(
        prior,
        likelihood,
        base_hazard=0.01,
        max_hazard=0.8,
        surprise_threshold=2.0,
        surprise_scale=0.75,
    )
    assert bool(changed)
    assert float(hazard) > 0.5
    assert float(surprise) > 2.0
    assert int(posterior.argmax()) == 1
    torch.testing.assert_close(posterior.exp().sum(), torch.tensor(1.0))


def test_static_safe_mask_rejects_obstacle_collision() -> None:
    trajectories = torch.zeros(2, 1, 2, 2, 4)
    trajectories[0, 0, 0, :, 0] = 2.0
    trajectories[0, 0, 1, :, 0] = -2.0
    trajectories[1, 0, 1, :, 0] = 1.0
    pool = CandidatePool(
        robot_current=torch.zeros(2, 1, 2, 4),
        robot_candidates=trajectories,
        activities=torch.zeros(2, 2),
    )
    simulator = SimpleNamespace(entity_positions=np.zeros((1, 2), dtype=np.float32))
    obstacles = np.asarray([[0.0, 0.0, 0.2]], dtype=np.float32)
    safe, obstacle, robot = _static_safe_mask(simulator, pool, obstacles, 0.1)
    assert safe.tolist() == [True, False]
    assert obstacle.tolist() == [False, True]
    assert robot.tolist() == [False, False]
