import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from tracer.experiments.eval_component_factorial import (
    FACTORIAL_POLICIES, METRICS, audit_shared_emissions, residual_weight,
    summarize_factorial,
)
from tracer.experiments.eval_persistent_closed_loop import _evaluate_policy, _prefix_objective_with_full_safety, _select_response_costs
from tracer.experiments.planner_sensitivity import candidate_score_decomposition
from tracer.experiments.train_component_residual import collect_uniform_rollouts, flatten_rollouts, freeze_noninteraction_parameters, prefix_cost_loss, residual_training_loss
from tracer.experiments.calibrate_component_memory import component_protocol_id, load_component_calibration, memory_decision_beliefs, score_scales
from tracer.experiments.formal_manifest import sha256_file
from tracer.experiments.audit_h2_memory_strata import memory_strata
from tracer.experiments.calibrate_component_residual_weight import choose_residual_weight, load_residual_weight
from tracer.experiments.train_component_mode_contrast import grouped_prefix_costs, mode_candidate_contrast
from tracer.response.closed_loop import ClosedLoopSpec, PlanningCostConfig
from tracer.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig


@pytest.mark.parametrize("retain_previous_plan", [False, True])
def test_factorial_holds_inference_fixed_when_planning_residual_changes(retain_previous_plan):
    torch.manual_seed(73)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=4, num_robots=2,
        hidden_dim=16, num_heads=2, dropout=0.0,
    )).eval()
    shared = dict(
        model=model, spec=ClosedLoopSpec(history=4, horizon=4, windows=3,
        num_robots=2, num_candidates=6, execute_steps=1, seed=73),
        episodes=2, device="cpu", planning_cost=PlanningCostConfig(task_steps=1),
        likelihood_scale=0.1,
        retain_previous_plan=retain_previous_plan,
    )
    trace = []
    results = {}
    for policy in FACTORIAL_POLICIES:
        weight = residual_weight(policy.name, 0.35)
        result = _evaluate_policy(
            **shared, policy=policy, evaluation_mode="same_state",
            captured_reference_trace=trace if not trace else None,
            reference_trace=trace if trace else None,
            response_gate_min=weight, response_gate_max=weight,
        )
        assert result["response_gate_weight"]["mean"] == pytest.approx(weight)
        results[policy.name] = result
    audit = audit_shared_emissions(results)
    assert audit["checked_emission_pairs"] == 18
    for memory in ("uniform", "reset", "persistent"):
        for rows in results[f"{memory}-joint"]["window_records"]:
            for row in rows:
                assert row["belief_gate_selection_flip"] == 0
                assert row["belief_gate_cost_delta"] == 0
    corrupted = copy.deepcopy(results)
    corrupted["persistent-additive"]["window_records"][0][1]["posterior_mode_probabilities"][0] += 0.01
    with pytest.raises(ValueError, match="changed inference"):
        audit_shared_emissions(corrupted)


def test_scene_bootstrap_does_not_count_training_seeds_as_new_scenes():
    runs = {}
    for seed in (2027, 2031, 2039):
        policies = {}
        for policy in FACTORIAL_POLICIES:
            value = float(policy.belief_type != "persistent")
            policies[policy.name] = {"episode_records": [
                {"episode": episode, **{metric: value for metric in METRICS}}
                for episode in range(4)
            ]}
        runs[str(seed)] = {"id": policies}
    summary = summarize_factorial(runs, resamples=100)["id"]
    assert summary["independent_scenes"] == 4
    assert summary["model_seeds"] == 3
    contrast = summary["contrasts"]["memory_at_joint"]["realized_total_cost"]
    assert contrast["mean_difference"] == -1
    assert contrast["case_bootstrap_ci95"] == [-1, -1]
    interaction = summary["contrasts"]["memory_joint_interaction"]["realized_total_cost"]
    assert interaction["mean_difference"] == 0
    runs["2039"]["id"]["persistent-additive"]["episode_records"][0]["episode"] = 4
    with pytest.raises(ValueError, match="align"):
        summarize_factorial(runs, resamples=100)


def test_navigation_metrics_are_explicit_and_reject_nonfinite_task_mixing():
    policies = {}
    for policy in FACTORIAL_POLICIES:
        policies[policy.name] = {"episode_records": [
            {"episode": episode, **dict.fromkeys(METRICS, 0.), "terminal_goal_distance": float(episode), "terminal_goal_success": float(episode == 0)}
            for episode in range(2)
        ]}
    runs = {"2027": {"id": policies}}
    summary = summarize_factorial(runs, 100, additional_metrics=("terminal_goal_distance", "terminal_goal_success"))["id"]
    assert summary["policy_means"]["persistent-joint"]["terminal_goal_success"] == .5
    policies["persistent-joint"]["episode_records"][0]["terminal_goal_distance"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        summarize_factorial(runs, 100, additional_metrics=("terminal_goal_distance",))


def test_source_manifest_includes_the_actual_runpy_entrypoint(tmp_path, monkeypatch):
    from tracer.experiments.eval_component_factorial import _local_sources

    entrypoint = tmp_path / "entrypoint.py"
    entrypoint.write_text("pass\n")
    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace(__file__=str(entrypoint)))
    assert entrypoint.resolve() in _local_sources()


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan")])
def test_invalid_residual_weights_fail(weight):
    with pytest.raises(ValueError, match="weight"):
        residual_weight("persistent-joint", weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_candidate_attribution_accepts_cuda_safety_mask():
    scores = torch.tensor([0.1, 0.2], device="cuda")
    result = candidate_score_decomposition(
        {}, scores, scores, torch.tensor([True, False], device="cuda"), 0, False,
    )
    assert result["admissible_fraction"] == 0.5


def test_residual_training_preserves_additive_prediction_and_uncertainty():
    torch.manual_seed(31)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=4, num_robots=2,
        hidden_dim=16, num_heads=2, dropout=0.2,
    ))
    trainable = freeze_noninteraction_parameters(model)
    assert not model.training
    history = torch.randn(3, 1, 4, 2)
    current = torch.randn(3, 1, 2, 4)
    candidates = torch.randn(3, 1, 2, 4, 4)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    additive_before = model(history, current, candidates, include_pair_interactions=False)
    joint_before = model(history, current, candidates)
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
    loss = residual_training_loss(joint_before, torch.randn(3, 4, 2), torch.tensor([0, 1, 2]), 0.4, 2)
    loss.backward()
    optimizer.step()
    additive_after = model(history, current, candidates, include_pair_interactions=False)
    joint_after = model(history, current, candidates)
    torch.testing.assert_close(additive_before["mean"], additive_after["mean"], rtol=0, atol=0)
    torch.testing.assert_close(joint_before["log_std"], joint_after["log_std"], rtol=0, atol=0)
    assert not torch.allclose(joint_before["mean"], joint_after["mean"])
    changed = {name for name, value in model.state_dict().items() if not torch.equal(before[name], value)}
    assert changed and changed.issubset(trainable)


def test_memory_calibration_scores_predictions_before_current_evidence():
    likelihoods = torch.tensor([[[0., -4., -8.], [-8., 0., -4.], [-4., -8., 0.]]], dtype=torch.float64)
    original = memory_decision_beliefs(likelihoods, 0.2)
    changed = likelihoods.clone()
    changed[:, 1] = torch.tensor([100., -100., -100.])
    updated = memory_decision_beliefs(changed, 0.2)
    for memory in ("persistent", "reset"):
        torch.testing.assert_close(original[memory][:, :2], updated[memory][:, :2])
        assert not torch.allclose(original[memory][:, 2], updated[memory][:, 2])
        torch.testing.assert_close(original[memory][:, 0], original["uniform"][:, 0])
    shifted = memory_decision_beliefs(likelihoods - 1000, 0.2)
    torch.testing.assert_close(original["persistent"], shifted["persistent"])
    score = score_scales(likelihoods, [0.2], coordinates=2)[0]
    expected = 0.5 * (score["decision_response_nll"]["persistent"] + score["decision_response_nll"]["reset"])
    assert score["selection_score"] == expected
    forgotten = memory_decision_beliefs(likelihoods, 0.2, forgetting_rate=1.0)
    torch.testing.assert_close(forgotten["persistent"], forgotten["reset"])
    scores = score_scales(likelihoods, [0.2], coordinates=2, forgetting_rates=[0.0, 0.5, 1.0])
    assert len(scores) == 3


def test_prefix_objective_preserves_full_horizon_safety_and_risk_fallback():
    prefix = {"unconstrained_total": torch.tensor([-10., 0., 1.]),
              "violation_probability": torch.tensor([0., 0., 0.]),
              "safe_mask": torch.tensor([True, True, True])}
    full = {"violation_probability": torch.tensor([0.4, 0.05, 0.02]),
            "safe_mask": torch.tensor([False, True, True])}
    result = _prefix_objective_with_full_safety(prefix, full, risk_weight=2.0)
    assert _select_response_costs(result, "total") == (1, False)
    assert torch.isinf(result["total"][0])
    torch.testing.assert_close(result["violation_probability"], full["violation_probability"])
    torch.testing.assert_close(result["unconstrained_total"], prefix["unconstrained_total"] + 2.0 * full["violation_probability"])
    full["safe_mask"][:] = False
    result = _prefix_objective_with_full_safety(prefix, full, risk_weight=2.0)
    assert _select_response_costs(result, "total") == (2, True)


def test_calibration_is_bound_to_checkpoint_cadence_and_distinct_scenes(tmp_path):
    checkpoint = tmp_path / "response.pt"
    checkpoint.write_bytes(b"frozen response")
    spec = ClosedLoopSpec(num_robots=3)
    record = {
        "schema": "component-memory-factual-v1", "phase": "validation", "status": "COMPLETED",
        "checkpoint_sha256": sha256_file(checkpoint), "data_seed": 11,
        "dt": spec.dt, "execute_steps": spec.execute_steps, "num_robots": spec.num_robots,
        "horizon": spec.horizon, "prefix_objective": True,
        "selected": {"likelihood_scale": 0.2},
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(record))
    assert load_component_calibration(path, checkpoint, spec, 12, True) == 0.2
    with pytest.raises(ValueError, match="separate"):
        load_component_calibration(path, checkpoint, spec, 11, True)
    with pytest.raises(ValueError, match="objective"):
        load_component_calibration(path, checkpoint, spec, 12, False)
    with pytest.raises(ValueError, match="execute_steps"):
        load_component_calibration(path, checkpoint, ClosedLoopSpec(num_robots=3, execute_steps=2), 12, True)
    checkpoint.write_bytes(b"different model")
    with pytest.raises(ValueError, match="checkpoint"):
        load_component_calibration(path, checkpoint, spec, 12, True)


def test_uniform_training_collection_preserves_factual_noise_and_reference():
    torch.manual_seed(41)
    raw = {
        "simulation": {"windows": 2, "history": 4, "horizon": 4, "num_robots": 2,
                       "num_candidates": 6, "execute_steps": 1, "dt": 0.4, "data_seed": 41},
        "planning": {"task_steps": 1},
    }
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=4, num_robots=2,
        hidden_dim=16, num_heads=2, dropout=0,
    )).eval()
    first = collect_uniform_rollouts(raw, None, model, 2, "cpu")
    second = collect_uniform_rollouts(raw, None, model, 2, "cpu")
    assert first["responses"].shape == (2, 2, 6, 4, 2)
    assert first["reference_responses"].shape == first["responses"].shape
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    assert (first["responses"] - first["reference_responses"]).abs().max() > 0
    values = flatten_rollouts(first)
    output = {"mean": values["target"][:, None, None].expand(-1, 1, 3, -1, -1)}
    reference = {"mean": values["reference_target"][:, None, None].expand(-1, 1, 3, -1, -1)}
    error = prefix_cost_loss(output, reference, values, PlanningCostConfig(**raw["planning"]), 0.4, 1)
    assert float(error) < 1e-12
    values["true_cost"] = values["true_cost"] + 0.5
    error = prefix_cost_loss(output, reference, values, PlanningCostConfig(**raw["planning"]), 0.4, 1)
    assert float(error) == pytest.approx(0.25)
    modes = collect_uniform_rollouts(raw, None, model, 2, "cpu", all_mode_supervision=True)
    assert modes["all_mode_responses"].shape == (2, 2, 3, 6, 4, 2)
    torch.testing.assert_close(modes["responses"], first["responses"], rtol=0, atol=0)
    for episode, mode in enumerate(modes["mode"]):
        torch.testing.assert_close(modes["all_mode_responses"][episode, :, mode], first["responses"][episode], rtol=0, atol=0)
    values = flatten_rollouts(modes)
    assert len(values["mode"]) == 2 * 2 * 3 * 6
    output = {"mean": values["target"][:, None, None].expand(-1, 1, 3, -1, -1)}
    reference = {"mean": values["reference_target"][:, None, None].expand(-1, 1, 3, -1, -1)}
    assert float(prefix_cost_loss(output, reference, values, PlanningCostConfig(**raw["planning"]), 0.4, 1)) < 1e-12
    groups = {name: tensor.flatten(0, 1) for name, tensor in modes.items() if tensor.ndim > 1}
    predicted = grouped_prefix_costs(
        groups["all_mode_responses"].transpose(1, 2), groups["all_mode_references"].transpose(1, 2),
        groups["current"], groups["candidates"], {"task_weight": 1., "human_weight": .6, "safety_weight": 2., "comfort_scale": 1.}, .4, 1,
    )
    torch.testing.assert_close(predicted, groups["all_mode_costs"], rtol=1e-5, atol=1e-6)


def test_planning_residual_head_cannot_rewrite_frozen_emissions():
    torch.manual_seed(71)
    emission = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=4, num_robots=2,
        hidden_dim=16, num_heads=2, dropout=0,
    )).eval()
    planning = copy.deepcopy(emission)
    with torch.no_grad():
        planning.pair_decoder[-1].weight.add_(0.2)
    shared = dict(
        spec=ClosedLoopSpec(history=4, horizon=4, windows=3, num_robots=2,
                            num_candidates=6, execute_steps=1, seed=71),
        episodes=2, device="cpu", planning_cost=PlanningCostConfig(task_steps=1),
        likelihood_scale=0.2, evaluation_mode="same_state",
    )
    policy = next(policy for policy in FACTORIAL_POLICIES if policy.name == "persistent-joint")
    trace = []
    original = _evaluate_policy(model=emission, policy=policy, **shared, captured_reference_trace=trace)
    corrected = _evaluate_policy(model=planning, emission_model=emission, policy=policy, **shared, reference_trace=trace)
    for original_rows, corrected_rows in zip(original["window_records"], corrected["window_records"]):
        for original_row, corrected_row in zip(original_rows, corrected_rows):
            assert original_row["decision_prefix_nll"] == pytest.approx(corrected_row["decision_prefix_nll"])
            torch.testing.assert_close(torch.tensor(original_row["posterior_mode_probabilities"]), torch.tensor(corrected_row["posterior_mode_probabilities"]))
            assert corrected_row["separate_frozen_emission_residual"]
    reset_policy = next(policy for policy in FACTORIAL_POLICIES if policy.name == "reset-joint")
    forgotten = _evaluate_policy(model=planning, emission_model=emission, policy=policy, **shared, reference_trace=trace, persistent_forgetting_rate=1.0)
    reset = _evaluate_policy(model=planning, emission_model=emission, policy=reset_policy, **shared, reference_trace=trace)
    assert forgotten["selected_candidates"] == reset["selected_candidates"]
    with torch.no_grad():
        planning.base_decoder[-1].weight.add_(0.01)
    with pytest.raises(ValueError, match="noninteraction"):
        _evaluate_policy(model=planning, emission_model=emission, policy=policy, **shared, reference_trace=trace)


def test_h2_memory_strata_do_not_confuse_accumulation_with_identity_reentry():
    labels = memory_strata([False, True, True, True, False, True])
    assert labels[0] == []
    assert "no_previous_observation" in labels[1]
    assert "one_previous_observation" in labels[2]
    assert "accumulation_without_reentry" in labels[3]
    assert labels[4] == []
    assert "identity_reentry" in labels[5]


def test_component_protocol_binding_ignores_scene_ids_but_rejects_changed_candidates():
    raw = {"simulation": {"data_seed": 11, "episodes": 4},
           "planning": {"human_weight": 0.6},
           "candidate_source": {"type": "sage", "diversity_weight": 0.0}}
    original = component_protocol_id(raw, True)
    changed = copy.deepcopy(raw)
    changed["simulation"]["data_seed"] = 123
    changed["simulation"]["episodes"] = 64
    assert component_protocol_id(changed, True) == original
    changed["candidate_source"]["diversity_weight"] = 0.35
    assert component_protocol_id(changed, True) != original
    assert component_protocol_id(raw, False) != original


def test_residual_weight_selection_does_not_trade_collisions_for_cost():
    records = [
        {"weight": 0., "cost": 1., "collision": 0.},
        {"weight": .5, "cost": .9, "collision": 0.},
        {"weight": 1., "cost": .2, "collision": .1},
    ]
    assert choose_residual_weight(records)["weight"] == .5
    records[1]["cost"] = 1.
    assert choose_residual_weight(records)["weight"] == 0.


def test_residual_weight_artifact_rejects_mismatched_source_and_scene(tmp_path):
    raw = {"simulation": {"data_seed": 11}, "planning": {}, "candidate_source": {"type": "sage"}}
    checkpoint = tmp_path / "response.pt"
    checkpoint.write_bytes(b"source model")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "status": "COMPLETED", "phase": "validation", "data_seed": 11,
        "checkpoint_sha256": sha256_file(checkpoint),
        "component_protocol_id": component_protocol_id(raw, True),
        "selected": {"weight": .75},
    }))
    assert load_residual_weight(calibration, checkpoint, raw, 12) == .75
    with pytest.raises(ValueError, match="separate"):
        load_residual_weight(calibration, checkpoint, raw, 11)
    changed = copy.deepcopy(raw)
    changed["planning"]["human_weight"] = 2.
    with pytest.raises(ValueError, match="protocol"):
        load_residual_weight(calibration, checkpoint, changed, 12)
    checkpoint.write_bytes(b"new weights")
    with pytest.raises(ValueError, match="checkpoint"):
        load_residual_weight(calibration, checkpoint, raw, 12)


def test_mode_candidate_contrast_removes_mode_and_candidate_only_offsets():
    costs = torch.tensor([[[0., 1., 2.], [2., 2., 0.], [3., 1., 1.]]])
    shifted = costs + torch.tensor([[[5., 10., -4.]]]) + torch.tensor([[[1.], [20.], [-7.]]])
    torch.testing.assert_close(mode_candidate_contrast(costs), mode_candidate_contrast(shifted), atol=2e-6, rtol=1e-5)
    assert mode_candidate_contrast(costs).abs().sum() > 0


def test_executable_regret_respects_safety_and_ignores_mode_only_cost_offsets():
    from tracer.experiments.train_component_mode_contrast import executable_candidate_regret

    truth = torch.tensor([[[0., 1., -10.], [0., 2., -20.]]])
    prediction = torch.tensor([[[0., 1., -100.], [0., 2., -100.]]], requires_grad=True)
    support = torch.tensor([[True, True, False]])
    correct = executable_candidate_regret(prediction, truth, support, .05)
    incorrect = executable_candidate_regret(prediction[..., [1, 0, 2]], truth, support, .05)
    assert correct < 1e-6
    assert incorrect > 1.
    offset = torch.tensor([[[5.], [-3.]]])
    torch.testing.assert_close(correct, executable_candidate_regret(prediction + offset, truth + offset, support, .05))
    correct.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[..., 2].abs().sum() == 0
    with pytest.raises(ValueError, match="nonempty"):
        executable_candidate_regret(prediction, truth, torch.zeros_like(support), .05)
