import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from tracer.experiments.closed_loop_protocol import audit_candidate_protocol, prepare_manifest
from tracer.experiments.eval_persistent_closed_loop import (
    Policy, _blend_gated_response_outputs, _evaluate_policy, _realized_total_cost,
    _write_csv, _write_episode_csv,
)
from tracer.response.closed_loop import (
    CandidatePool, ClosedLoopSpec, PlanningCostConfig, SyntheticClosedLoopSimulator,
    expected_candidate_costs,
)
from tracer.response.closed_loop_contract import task_goals, validate_candidate_state
from tracer.response.persistent_models import (
    BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig,
)
from tracer.response.persistent_trajectory import (
    MultiEntityPersistentTrajectoryEpisodeDataset,
    MultiEntityPersistentTrajectorySpec,
    PersistentTrajectoryEpisodeDataset,
    PersistentTrajectorySpec,
)


def test_stale_candidate_pool_is_rejected_without_mutating_simulator():
    simulator = SyntheticClosedLoopSimulator(ClosedLoopSpec(), 0)
    pool = simulator.candidate_pool()
    simulator.step(pool, 1, PlanningCostConfig())
    state_before = copy.deepcopy(simulator.__dict__)
    for operation in (simulator.counterfactuals, lambda pool, cost: simulator.step(pool, 0, cost)):
        with pytest.raises(ValueError, match="candidate/state contract"):
            operation(pool, PlanningCostConfig())
        np.testing.assert_array_equal(simulator.robot_positions, state_before["robot_positions"])
        assert simulator.rng.bit_generator.state == state_before["rng"].bit_generator.state


def test_candidate_integration_contract_rejects_inconsistent_positions():
    simulator = SyntheticClosedLoopSimulator(ClosedLoopSpec(), 0)
    pool = simulator.candidate_pool()
    altered = pool.robot_candidates.clone()
    altered[0, 0, 1, 0] += 0.1
    with pytest.raises(ValueError, match="candidate_integration_error"):
        validate_candidate_state(simulator, replace(pool, robot_candidates=altered))
    assert max(validate_candidate_state(simulator, pool).values()) < 1e-5


def test_candidate_generation_does_not_consume_environment_noise():
    first = SyntheticClosedLoopSimulator(ClosedLoopSpec(), 0)
    second = copy.deepcopy(first)
    first.candidate_pool()
    first.candidate_pool()
    assert first.rng.bit_generator.state == second.rng.bit_generator.state
    pool = second.candidate_pool()
    noise_before = copy.deepcopy(first.rng.bit_generator.state)
    first.counterfactuals(pool, PlanningCostConfig())
    assert first.rng.bit_generator.state == noise_before


def test_interaction_residual_dataset_is_optional_and_mode_conditioned():
    baseline = PersistentTrajectoryEpisodeDataset(
        PersistentTrajectorySpec(num_episodes=1, cache_samples=True)
    )[0]
    assert "interaction_residual" not in baseline
    dataset = MultiEntityPersistentTrajectoryEpisodeDataset(
        MultiEntityPersistentTrajectorySpec(
            num_episodes=1,
            num_entities=4,
            active_entities=2,
            response_noise=0.0,
            shared_team_actions=True,
            include_interaction_residual=True,
            cache_samples=True,
        )
    )
    sample = dataset[0]
    assert tuple(sample["interaction_residual"].shape) == (4, 8, 3, 12, 2)
    assert torch.isfinite(sample["interaction_residual"]).all()
    assert float(sample["interaction_residual"].abs().sum()) > 0.0


def test_response_gate_trusts_joint_only_with_confident_belief_and_low_uncertainty():
    additive = {
        "mean": torch.zeros(4, 1, 2, 2),
        "log_std": torch.full((4, 1, 2, 2), -1.0),
    }
    joint = {
        "mean": torch.ones(4, 1, 2, 2),
        "log_std": torch.full((4, 1, 2, 2), -1.0),
    }
    _, uniform_gate, _, _ = _blend_gated_response_outputs(
        additive,
        joint,
        torch.tensor([0.5, 0.5]),
        belief_threshold=0.35,
        belief_scale=0.10,
        uncertainty_threshold=0.45,
        uncertainty_scale=0.08,
    )
    blended, confident_gate, _, _ = _blend_gated_response_outputs(
        additive,
        joint,
        torch.tensor([0.98, 0.02]),
        belief_threshold=0.35,
        belief_scale=0.10,
        uncertainty_threshold=0.45,
        uncertainty_scale=0.08,
    )
    assert float(confident_gate.mean()) > float(uniform_gate.mean())
    assert float(blended["mean"].mean()) > 0.5


def test_response_gate_reduces_joint_residual_when_uncertainty_is_high():
    additive = {
        "mean": torch.zeros(3, 1, 2, 2),
        "log_std": torch.full((3, 1, 2, 2), -1.0),
    }
    joint = {
        "mean": torch.ones(3, 1, 2, 2),
        "log_std": torch.full((3, 1, 2, 2), 1.0),
    }
    blended, gate, _, uncertainty_confidence = _blend_gated_response_outputs(
        additive,
        joint,
        torch.tensor([0.98, 0.02]),
        belief_threshold=0.35,
        belief_scale=0.10,
        uncertainty_threshold=0.45,
        uncertainty_scale=0.08,
    )
    assert float(gate.mean()) < 0.1
    assert float(uncertainty_confidence.mean()) < 0.1
    assert float(blended["mean"].abs().mean()) < 0.1


@pytest.mark.parametrize("task_type", ["entity_approach", "fixed_goal"])
@pytest.mark.parametrize("episode", [0, 3, 12])
def test_prefix_perfect_prediction_matches_realized_cost(task_type, episode):
    spec = ClosedLoopSpec(task_type=task_type, response_noise=0.0)
    simulator = SyntheticClosedLoopSimulator(spec, episode)
    pool = simulator.candidate_pool()
    config = PlanningCostConfig(
        task_steps=spec.execute_steps, response_steps=spec.execute_steps,
        task_speed_scale=2.3, smoothness_weight=0.27, risk_weight=0.8,
        violation_probability_alpha=None,
    )
    current = pool.robot_current.numpy()
    candidates = pool.robot_candidates.numpy()
    stationary = np.zeros_like(candidates)
    stationary[..., :2] = current[:, :, None, :2]
    zero_noise = np.zeros((spec.horizon, 2), dtype=np.float32)
    means = torch.from_numpy(simulator._response_batch(current, candidates, noise=zero_noise))[:, None]
    reference = torch.from_numpy(simulator._response_batch(current, stationary, noise=zero_noise))[:, None]
    offsets = torch.from_numpy(task_goals(simulator) - simulator.robot_positions)
    predicted = expected_candidate_costs(
        means, torch.ones(1), pool.robot_current, pool.robot_candidates,
        reference, spec.dt, config, mode_log_stds=torch.full_like(means, -30.0),
        task_direction=offsets, task_goal_offsets=offsets if task_type == "fixed_goal" else None,
    )["unconstrained_total"]
    outcomes = simulator.counterfactuals(pool, config)
    actual = torch.tensor([_realized_total_cost(outcome, spec, config) for outcome in outcomes])
    torch.testing.assert_close(predicted, actual, rtol=1e-5, atol=1e-5)


def test_fixed_task_goal_is_not_the_moving_pedestrian():
    simulator = SyntheticClosedLoopSimulator(ClosedLoopSpec(task_type="fixed_goal"), 0)
    goals = task_goals(simulator)
    simulator.step(simulator.candidate_pool(), 1, PlanningCostConfig())
    np.testing.assert_array_equal(task_goals(simulator), goals)
    assert not np.allclose(goals, simulator.entity_position)


@pytest.mark.parametrize("sampling", ["shared", "independent"])
def test_unexecuted_suffix_does_not_change_prefix_diagnostic(sampling):
    simulator = SyntheticClosedLoopSimulator(ClosedLoopSpec(), 0)
    pool = simulator.candidate_pool()
    means = torch.zeros(pool.robot_candidates.shape[0], 2, simulator.spec.horizon, 2)
    config = PlanningCostConfig(task_steps=3, response_steps=3, risk_weight=0.3, temporal_risk_sampling=sampling)
    def score(response):
        return expected_candidate_costs(response, torch.tensor([0.5, 0.5]), pool.robot_current,
                                        pool.robot_candidates, means, simulator.spec.dt, config,
                                        mode_log_stds=torch.zeros_like(means))
    before = score(means)
    changed = means.clone()
    changed[:, :, 3:] = 1000.0
    after = score(changed)
    for key in before:
        torch.testing.assert_close(before[key], after[key])


def test_masked_candidates_keep_finite_diagnostic_costs():
    simulator = SyntheticClosedLoopSimulator(ClosedLoopSpec(), 0)
    pool = simulator.candidate_pool()
    means = torch.zeros(pool.robot_candidates.shape[0], 2, simulator.spec.horizon, 2)
    costs = expected_candidate_costs(
        means, torch.tensor([0.5, 0.5]), pool.robot_current, pool.robot_candidates,
        means, simulator.spec.dt, PlanningCostConfig(collision_margin=1000, violation_probability_alpha=0.1),
        mode_log_stds=torch.zeros_like(means),
    )
    assert not costs["safe_mask"].any()
    assert torch.isinf(costs["total"]).all()
    assert torch.isfinite(costs["unconstrained_total"]).all()


@pytest.mark.parametrize("evaluation_mode", ["closed_loop", "same_state"])
@pytest.mark.parametrize("retention", [False, True])
def test_policy_rollouts_respect_state_and_evidence_contract(evaluation_mode, retention, tmp_path):
    torch.manual_seed(2)
    spec = ClosedLoopSpec(windows=3, history=4, horizon=4, execute_steps=2, num_candidates=5)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        history=4, horizon=4, num_robots=2, hidden_dim=16, dropout=0.0,
    )).eval()
    policies = (Policy("task", "uniform", True, "task"), Policy("human", "persistent", True, "human"))
    trace = [] if evaluation_mode == "same_state" else None
    runs = {"2": {"id": {}}}
    for index, policy in enumerate(policies):
        runs["2"]["id"][policy.name] = _evaluate_policy(
            model, policy, spec, 2, "cpu", PlanningCostConfig(), 0.05,
            evaluation_mode=evaluation_mode,
            captured_reference_trace=trace if index == 0 else None,
            reference_trace=trace if index else None,
            retain_previous_plan=retention,
        )
    audit = audit_candidate_protocol(runs, [policy.name for policy in policies], evaluation_mode)
    assert audit["status"] == "pass"
    if evaluation_mode == "closed_loop":
        assert audit["divergent_state_comparisons"] > 0
    else:
        assert audit["divergent_state_comparisons"] == 0
        records = runs["2"]["id"]["human"]["window_records"]
        assert any(record["selected_candidate"] != record["executed_candidate"] for windows in records for record in windows)
    _write_csv(runs, tmp_path / "runs.csv")
    _write_episode_csv(runs, tmp_path / "episodes.csv")


def test_candidate_audit_distinguishes_physical_and_planner_state():
    reference = dict.fromkeys(("candidate_position_error", "candidate_velocity_error", "candidate_integration_error", "execution_position_error"), 0.)
    reference.update(candidate_seed=1, environment_rng_hash="rng", state_hash="physical", candidate_hash="pool", retained_plan_enabled=True, candidate_context_hash="plan")
    changed = {**reference, "candidate_context_hash": "other-plan", "candidate_hash": "other-pool"}
    runs = {"seed": {"id": {"first": {"window_records": [[reference]]}, "second": {"window_records": [[changed]]}}}}
    result = audit_candidate_protocol(runs, ["first", "second"], "closed_loop")
    assert result["divergent_candidate_context_comparisons"] == 1
    with pytest.raises(ValueError, match="planning context"):
        audit_candidate_protocol(runs, ["first", "second"], "same_state")
    changed["candidate_context_hash"] = reference["candidate_context_hash"]
    with pytest.raises(ValueError, match="different candidate pools"):
        audit_candidate_protocol(runs, ["first", "second"], "closed_loop")
    changed.pop("candidate_context_hash")
    with pytest.raises(ValueError, match="recorded planning context"):
        audit_candidate_protocol(runs, ["first", "second"], "closed_loop")


def test_resume_requires_exact_protocol_weights_and_sources(tmp_path):
    output = tmp_path / "results"
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    source = tmp_path / "source.py"
    source.write_text("version = 1\n")
    manifest = prepare_manifest(output, {"phase": "validation"}, [checkpoint], [source], False)
    assert prepare_manifest(output, {"phase": "validation"}, [checkpoint], [source], True) == manifest
    for config in ({"phase": "test"},):
        with pytest.raises(ValueError, match="mismatch"):
            prepare_manifest(output, config, [checkpoint], [source], True)
    checkpoint.write_bytes(b"updated weights")
    with pytest.raises(ValueError, match="mismatch"):
        prepare_manifest(output, {"phase": "validation"}, [checkpoint], [source], True)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "runs.json").write_text("{}")
    with pytest.raises(ValueError, match="legacy"):
        prepare_manifest(legacy, {}, [], [], True)


def test_response_gate_respects_configured_maximum_mix():
    additive = {
        "mean": torch.zeros(2, 1, 3, 2, 2),
        "log_std": torch.zeros(2, 1, 3, 2, 2),
    }
    joint = {
        "mean": torch.ones(2, 1, 3, 2, 2),
        "log_std": torch.zeros(2, 1, 3, 2, 2),
    }
    blended, gate, _, _ = _blend_gated_response_outputs(
        additive,
        joint,
        torch.tensor([1.0, 0.0, 0.0]),
        0.0,
        0.1,
        1.0,
        0.1,
        0.0,
        0.4,
    )
    assert torch.all(gate <= 0.4)
    assert torch.all(blended["mean"] <= 0.4)
