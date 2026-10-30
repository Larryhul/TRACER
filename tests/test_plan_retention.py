import pytest
import torch

from tracer.response.closed_loop import ClosedLoopSpec, PlanningCostConfig, SyntheticClosedLoopSimulator
from tracer.response.closed_loop_contract import validate_candidate_state
from tracer.response.plan_retention import plan_context_sha256, retain_plan_candidate, shifted_plan_candidate


def test_retained_controls_reanchor_to_actual_state_and_preserve_future_world_path():
    simulator = SyntheticClosedLoopSimulator(ClosedLoopSpec(num_robots=2, num_candidates=6, horizon=5, execute_steps=2), 3)
    pool = simulator.candidate_pool()
    previous = pool.robot_candidates[1].clone()
    expected_world = torch.from_numpy(simulator.entity_position)[None, None] + previous[:, 2:, :2]
    simulator.step(pool, 1, PlanningCostConfig())
    following = simulator.candidate_pool()
    original_candidates = following.robot_candidates.clone()
    updated, index = retain_plan_candidate(following, previous, 2, simulator.spec.dt, 1.2)
    actual_world = torch.from_numpy(simulator.entity_position)[None, None] + updated.robot_candidates[index, :, :3, :2]
    torch.testing.assert_close(actual_world, expected_world)
    torch.testing.assert_close(updated.robot_candidates[index, :, :3, 2:], previous[:, 2:, 2:])
    torch.testing.assert_close(following.robot_candidates, original_candidates)
    assert len(updated.robot_candidates) == len(following.robot_candidates)
    assert index == len(following.robot_candidates) - 1
    assert max(validate_candidate_state(simulator, updated).values()) < 1e-5
    repeated, repeated_index = retain_plan_candidate(updated, previous, 2, simulator.spec.dt, 1.2)
    assert repeated is updated
    assert repeated_index == index


def test_retention_rejects_invalid_execution_prefix():
    simulator = SyntheticClosedLoopSimulator(ClosedLoopSpec(), 0)
    pool = simulator.candidate_pool()
    with pytest.raises(ValueError, match="prefix"):
        shifted_plan_candidate(pool, pool.robot_candidates[0], simulator.spec.horizon, simulator.spec.dt)


@pytest.mark.parametrize("invalid", [0., -1., float("nan"), float("inf")])
def test_retention_rejects_invalid_timestep_and_activity_scale(invalid):
    simulator = SyntheticClosedLoopSimulator(ClosedLoopSpec(), 0)
    pool = simulator.candidate_pool()
    with pytest.raises(ValueError, match="timestep"):
        shifted_plan_candidate(pool, pool.robot_candidates[0], 1, invalid)
    with pytest.raises(ValueError, match="speed scale"):
        retain_plan_candidate(pool, pool.robot_candidates[0], 1, simulator.spec.dt, invalid)


def test_plan_context_binds_only_remaining_controls():
    previous = torch.zeros(2, 5, 4)
    original = plan_context_sha256(previous, 2)
    assert original != plan_context_sha256(None, 2)
    previous[:, :2, 2:] = 3.
    previous[..., :2] = 6.
    assert plan_context_sha256(previous, 2) == original
    previous[:, 3, 2:] = 1.
    assert plan_context_sha256(previous, 2) != original
