import copy
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sage.response.closed_loop import CandidatePool
from sage.response.coordinated_candidates import CoordinatedCandidateAdapter, complete_joint_speed_candidates


def original_pool(robots=3, count=16):
    current = torch.zeros(count, robots, 4)
    current[..., :2] = torch.arange(robots)[:, None] + 2.
    velocity = torch.full((count, robots, 5, 2), .1)
    candidates = torch.cat((current[:, :, None, :2]+(velocity*.4).cumsum(2), velocity), -1)
    return CandidatePool(current, candidates, torch.full((count, robots), .1), torch.ones(count, 2, 5, 2))


def test_joint_completion_retains_budget_and_all_robot_combinations():
    pool = original_pool()
    before = copy.deepcopy(pool)
    offsets = torch.tensor([[2., 0.], [0., 2.], [-2., 0.]])
    result = complete_joint_speed_candidates(pool, offsets, .4, 1.2, (.4, 1.), 3, torch.zeros(2, 2))
    assert result.robot_candidates.shape == pool.robot_candidates.shape
    expected = torch.tensor(list(itertools.product((.4, 1.), repeat=3))) * 1.2
    torch.testing.assert_close(result.robot_candidates[-8:, :, 0, 2:].norm(dim=-1), expected)
    torch.testing.assert_close(result.robot_candidates[..., :2], result.robot_current[:, :, None, :2]+(result.robot_candidates[..., 2:]*.4).cumsum(2))
    torch.testing.assert_close(result.robot_candidates[:8], pool.robot_candidates[:8])
    torch.testing.assert_close(pool.robot_candidates, before.robot_candidates)
    assert result.entity_candidates.shape == (16, 2, 5, 2)
    assert torch.count_nonzero(result.entity_candidates[-8:]) == 0


def test_joint_completion_stops_at_goal_and_supports_zero_speed():
    pool = original_pool()
    offsets = torch.tensor([[.1, 0.], [0., .2], [-.3, 0.]])
    result = complete_joint_speed_candidates(pool, offsets, .4, 1.2, (0., 1.), 3, torch.zeros(2, 2), True)
    displacement = result.robot_candidates[-8:, ..., :2]-result.robot_current[-8:, :, None, :2]
    assert (displacement.norm(dim=-1) <= offsets.norm(dim=-1)[None, :, None]+1e-6).all()
    assert torch.count_nonzero(result.robot_candidates[-8, ..., 2:]) == 0


@pytest.mark.parametrize("scales", [(1.,), (.4, .4), (-.4, 1.), (.4, float('nan')), (.2, .4, 1.)])
def test_joint_completion_rejects_invalid_or_overbudget_combinations(scales):
    with pytest.raises(ValueError):
        complete_joint_speed_candidates(original_pool(), torch.ones(3, 2), .4, 1.2, scales, 3, torch.zeros(2, 2))


def test_joint_completion_is_translation_consistent():
    pool = original_pool()
    offsets = torch.ones(3, 2)
    first = complete_joint_speed_candidates(pool, offsets, .4, 1.2, (.4, 1.), 3, torch.zeros(2, 2))
    shifted = copy.deepcopy(pool)
    shifted.robot_current[..., :2] += 3
    shifted.robot_candidates[..., :2] += 3
    shifted.entity_candidates[:] += 3
    second = complete_joint_speed_candidates(shifted, offsets, .4, 1.2, (.4, 1.), 3, torch.full((2, 2), 3.))
    torch.testing.assert_close(second.robot_candidates[..., :2], first.robot_candidates[..., :2]+3)
    torch.testing.assert_close(second.robot_candidates[..., 2:], first.robot_candidates[..., 2:])
    torch.testing.assert_close(second.entity_candidates, first.entity_candidates+3)


def test_adapter_single_and_batch_have_identical_observable_completion():
    pool = original_pool()
    generator = SimpleNamespace(config=SimpleNamespace(goal_speed=1.2, task_prefix_steps=3, stop_task_anchors_at_goal=False),
        candidate_pool=lambda *args: copy.deepcopy(pool), candidate_pools=lambda scenes, sample_seeds: [copy.deepcopy(pool) for scene in scenes])
    adapter = CoordinatedCandidateAdapter(generator, (.4, 1.))
    history = np.zeros((2, 4, 2), dtype=np.float32)
    positions = pool.robot_current[0, :, :2].numpy()
    velocities = np.zeros_like(positions)
    goals = positions+np.ones_like(positions)
    single = adapter.candidate_pool(history, positions, velocities, .4, 7, goals)
    batch = adapter.candidate_pools([(history, positions, velocities, .4, goals)], [7])[0]
    torch.testing.assert_close(single.robot_candidates, batch.robot_candidates)
    torch.testing.assert_close(single.entity_candidates, batch.entity_candidates)


def test_completion_configuration_is_opt_in(monkeypatch):
    from sage.experiments.finetune_h3_response_ranking import _adapter_from_config
    from sage.response.sage_candidate_adapter import SAGECandidateAdapter

    generator = SimpleNamespace()
    monkeypatch.setattr(SAGECandidateAdapter, "from_checkpoint", lambda *args, **kwargs: generator)
    raw = {"simulation": {"num_candidates": 16}, "candidate_source": {"checkpoint": "unused.pt"}}
    assert _adapter_from_config(raw, "cpu") is generator
    raw["candidate_source"]["joint_speed_completion"] = [.4, 1.]
    adapter = _adapter_from_config(raw, "cpu")
    assert isinstance(adapter, CoordinatedCandidateAdapter)
    assert adapter.generator is generator
    assert adapter.speed_scales == (.4, 1.)
