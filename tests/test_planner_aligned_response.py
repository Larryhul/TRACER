import json

import numpy as np
import pytest
import torch

from tracer.experiments.collect_external_planner_supervision import collection_cases, expanded_case_splits
from tracer.experiments.eval_external_information_value import information_evaluation_cases, summarize_information
from tracer.experiments.merge_planner_supervision import merge
from tracer.experiments.run_sicnav_tracer_loop import _differentiable_response_predictions, _response_predictions
from tracer.experiments.train_executable_value import continuation_cases
from tracer.response.closed_loop import PlanningCostConfig, expected_candidate_costs
from tracer.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig
from tracer.response.planner_supervision import (
    add_action_conditioned_uncertainty, configure_planner_finetuning, entity_risk_calibration_loss, group_regret,
    expected_prefix_regret_loss, planner_gap_calibration_loss, prefix_equivalence, top_prefix_ranking_loss,
)


def risk_example(temperature):
    means = torch.zeros(2, 3, 4, 2, requires_grad=True)
    deviations = torch.full_like(means, -1.0, requires_grad=True)
    candidates = torch.zeros(2, 1, 4, 4)
    candidates[..., 0] = .65
    costs = expected_candidate_costs(
        means, torch.full((3,), 1 / 3), torch.zeros(2, 1, 4), candidates,
        torch.zeros_like(means), .25,
        PlanningCostConfig(risk_weight=.5, violation_probability_alpha=.4), deviations,
        risk_gradient_temperature=temperature,
    )
    return costs, means, deviations


def test_risk_gradient_leaves_deployed_values_and_safety_masks_exactly_unchanged():
    original, _, _ = risk_example(None)
    surrogate, means, deviations = risk_example(.1)
    for name in original:
        torch.testing.assert_close(original[name], surrogate[name], rtol=0, atol=0)
    surrogate["violation_probability"].sum().backward()
    assert torch.isfinite(means.grad).all() and means.grad.abs().sum() > 0
    assert torch.isfinite(deviations.grad).all() and deviations.grad.abs().sum() > 0


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_risk_gradient_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="risk_gradient_temperature"):
        risk_example(temperature)


def test_planner_supervision_cannot_collect_test_or_overlapping_cases():
    splits = {"train": [112, 113], "val": [136], "test": [88]}
    assert collection_cases(splits, "train") == [112, 113]
    with pytest.raises(ValueError, match="excludes test"):
        collection_cases(splits, "test")
    with pytest.raises(ValueError, match="disjoint"):
        collection_cases({**splits, "val": [112]}, "train")


def test_confirmation_cannot_reuse_training_validation_or_previous_test_cases():
    splits = {"train": [112, 113], "val": [136], "test": [88]}
    assert information_evaluation_cases(splits) == ("validation", [136])
    assert information_evaluation_cases(splits, 160, 3) == ("confirmation", [160, 161, 162])
    for start in (88, 112, 136):
        with pytest.raises(ValueError, match="disjoint"):
            information_evaluation_cases(splits, start, 1)


def test_expanded_training_cannot_reassign_original_validation_or_test_cases():
    original = {"train": [112], "val": [136], "test": [88]}
    proposed = {"train": [0, 112], "val": [64, 136], "test": [88, 160]}
    assert expanded_case_splits(original, proposed) == proposed
    with pytest.raises(ValueError, match="retain the original train cases"):
        expanded_case_splits(proposed, original)
    with pytest.raises(ValueError):
        expanded_case_splits(original, {"train": [0, 88, 112], "val": [64, 136], "test": [160]})


def test_diagnostic_continuations_remain_outside_frozen_splits():
    protocol = {"cases": [698, 701], "case_splits": {"train": [112], "val": [136], "test": [88]}}
    assert continuation_cases(protocol, "diagnostic") == [698, 701]
    with pytest.raises(ValueError, match="outside frozen"):
        continuation_cases({**protocol, "cases": [88]}, "diagnostic")


def test_merge_requires_complete_disjoint_immutable_shards(tmp_path):
    splits = {"train": [0, 112], "val": [136], "test": [88]}
    shards = []
    for case in (0, 112):
        shard = tmp_path / f"shard{case}"
        (shard / "audit").mkdir(parents=True)
        (shard / "inputs").mkdir()
        protocol = {"split": "train", "cases": [case], "planning": {}, "dt": .25,
                    "execute_steps": 1, "checkpoint_sha256": "parent", "seed": 2027,
                    "parameters": {"cases": 1, "case_start": case, "safety_weight": 2.}}
        (shard / "protocol.json").write_text(json.dumps(protocol))
        (shard / "manifest.json").write_text(json.dumps({"status": "COMPLETED"}))
        row = {"seed": 2027, "case": case, "window": 0, "live_prefix_verified": True}
        (shard / "audit/windows.jsonl").write_text(json.dumps(row) + "\n")
        for directory in ("audit", "inputs"):
            np.savez(shard / directory / f"seed2027_case{case}_window0.npz", values=np.zeros(1))
        shards.append(shard)
    with pytest.raises(ValueError, match="entire declared split"):
        merge(tmp_path / "missing", shards[:1], splits, "train")
    with pytest.raises(ValueError, match="non-overlapping"):
        merge(tmp_path / "duplicate", [shards[0], shards[0]], splits, "train")
    output = tmp_path / "complete"
    merge(output, shards, splits, "train")
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["groups"] == 2
    assert (output / "inputs/seed2027_case0_window0.npz").read_bytes() == (shards[0] / "inputs/seed2027_case0_window0.npz").read_bytes()


def test_paired_checkpoint_summary_uses_cases_and_requires_common_support():
    row = {"status": "matched", "live_prefix_verified": True, "case": 160,
           "uniform_cost": 3., "persistent_cost": 2., "true_mode_cost": 2., "candidate_cost": 1., "oracle_cost": 0.,
           "uniform_prefix_regret": 1., "persistent_prefix_regret": 1., "candidate_prefix_regret": 0.}
    for name in ("uniform", "persistent", "true_mode", "candidate", "oracle"):
        row[f"{name}_supported"] = True
        row[f"{name}_action_change_vs_uniform"] = 0
    other = {**row, "case": 161, "candidate_cost": 3., "candidate_prefix_regret": 2.}
    unsupported = {**row, "case": 162, "candidate_supported": False, "candidate_cost": -100.}
    summary = summarize_information([dict(row) for _ in range(12)] + [other, unsupported])
    assert summary["independent_cases"] == 2
    assert summary["paired_model_comparison"]["cost"]["parent_minus_candidate"] == 0
    assert summary["paired_model_comparison"]["prefix_regret"]["parent_minus_candidate"] == 0


def test_prefix_loss_ignores_tail_only_differences_and_unsafe_candidates():
    scores = torch.tensor([0., 2., -10.], requires_grad=True)
    factual = torch.tensor([2., 0., -20.])
    same = torch.tensor([[True, True, False], [True, True, False], [False, False, True]])
    loss = top_prefix_ranking_loss(scores, factual, torch.tensor([True, True, False]), same)
    assert float(loss.detach()) == 0
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.zeros_like(scores))


def test_prefix_loss_corrects_the_order_of_executable_actions():
    scores = torch.tensor([0., .2], requires_grad=True)
    factual = torch.tensor([.2, 0.])
    loss = top_prefix_ranking_loss(scores, factual, torch.ones(2, dtype=torch.bool), torch.eye(2, dtype=torch.bool))
    loss.backward()
    assert scores.grad[0] < 0 and scores.grad[1] > 0


def test_gap_calibration_preserves_cost_scale_instead_of_exaggerating_separation():
    factual = torch.tensor([.0, .1, .2])
    scores = (factual + 3).requires_grad_()
    supported = torch.ones(3, dtype=torch.bool)
    same = torch.eye(3, dtype=torch.bool)
    assert float(planner_gap_calibration_loss(scores, factual, supported, same).detach()) < 1e-10
    exaggerated = torch.tensor([0., 1., 2.], requires_grad=True)
    loss = planner_gap_calibration_loss(exaggerated, factual, supported, same)
    loss.backward()
    assert exaggerated.grad[0] < 0 and exaggerated.grad[2] > 0


def test_gap_calibration_excludes_nonfinite_and_nonexecutable_branches():
    scores = torch.tensor([.1, .2, float("inf"), -100.], requires_grad=True)
    factual = torch.tensor([.2, .1, float("inf"), -200.])
    eligible = torch.tensor([True, True, False, False])
    loss = planner_gap_calibration_loss(scores, factual, eligible, torch.eye(4, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[2] == 0 and scores.grad[3] == 0


def test_entity_risk_supervision_ignores_unsupported_candidates():
    probabilities = torch.tensor([.1, .7, .9], requires_grad=True)
    loss = entity_risk_calibration_loss([{"violation_probability": probabilities}],
                                       [[1., 0., 0.]], torch.tensor([True, True, False]))
    loss.backward()
    assert probabilities.grad[0] < 0 and probabilities.grad[1] > 0 and probabilities.grad[2] == 0


def test_expected_prefix_regret_respects_true_cost_scale_and_prefix_duplicates():
    scores = torch.tensor([0., .1], requires_grad=True)
    factual = torch.tensor([.2, 0.])
    support = torch.ones(2, dtype=torch.bool)
    same = torch.eye(2, dtype=torch.bool)
    loss = expected_prefix_regret_loss(scores, factual, support, same)
    doubled = expected_prefix_regret_loss(scores, 2 * factual, support, same)
    torch.testing.assert_close(doubled, 2 * loss)
    loss.backward()
    assert scores.grad[0] < 0 and scores.grad[1] > 0
    repeated = expected_prefix_regret_loss(
        torch.tensor([0., .1, .2]), torch.tensor([.2, 0., .3]), torch.ones(3, dtype=torch.bool),
        torch.tensor([[True, False, True], [False, True, False], [True, False, True]]),
    )
    torch.testing.assert_close(repeated, loss)


def test_expected_prefix_regret_has_no_incentive_when_all_actions_are_equivalent():
    scores = torch.tensor([0., .1, -.3], requires_grad=True)
    loss = expected_prefix_regret_loss(scores, torch.tensor([2., 1., 3.]), torch.ones(3, dtype=torch.bool),
                                      torch.ones(3, 3, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.zeros_like(scores))


def test_prefix_regret_uses_best_tail_with_same_heading_and_velocity():
    snapshots = np.zeros((3, 2, 2, 8))
    snapshots[1, 1, 0, 0] = 1
    snapshots[2, 0, 0, 4] = .1
    same = torch.from_numpy(prefix_equivalence(snapshots, 1))
    assert same[0, 1] and not same[0, 2]
    result = group_regret(torch.tensor([0., 1., 2.]), torch.tensor([2., .5, 0.]), torch.ones(3, dtype=torch.bool), same)
    assert result["regret"] == 2 and result["prefix_regret"] == .5


def test_training_and_deployment_share_predictions_without_deployment_gradients():
    torch.manual_seed(2027)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=3, num_robots=1, hidden_dim=16,
        num_heads=2, graph_layers=1, dropout=0, dt=.25,
    )).eval()
    generator = np.random.default_rng(2027)
    history = generator.normal(size=(2, 4, 2)).astype(np.float32)
    current = generator.normal(size=(2, 4)).astype(np.float32)
    candidates = generator.normal(size=(2, 3, 3, 4)).astype(np.float32)
    deployed = _response_predictions(model, history, current, candidates, "cpu")
    trainable = _differentiable_response_predictions(model, history, current, candidates, "cpu")
    for original, actual in zip(deployed, trainable):
        torch.testing.assert_close(original, actual, rtol=1e-5, atol=1e-6)
        assert not original.requires_grad
    assert trainable[0].requires_grad
    trainable[0].square().mean().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())


def test_prefix_preserving_optimization_changes_future_but_not_belief_likelihood():
    torch.manual_seed(2027)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=3, num_robots=1, hidden_dim=16,
        num_heads=2, graph_layers=1, dropout=0, dt=.25, response_delay_steps=1,
        factual_prefix_dynamics=True,
    )).eval()
    names = configure_planner_finetuning(model, True)
    assert not any(name.startswith("factual_prefix_") for name in names)
    history = torch.randn(2, 1, 4, 2)
    current = torch.randn(2, 1, 1, 4)
    candidates = torch.randn(2, 1, 1, 3, 4)
    with torch.no_grad():
        before = model(history, current, candidates)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=.01)
    model.train()
    outputs = model(history, current, candidates)
    loss = outputs["mean"][..., 1:, :].square().mean() + outputs["log_std"][..., 1:, :].square().mean()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        after = model(history, current, candidates)
    for name in ("mean", "log_std"):
        torch.testing.assert_close(before[name][..., :1, :], after[name][..., :1, :], rtol=0, atol=0)
        assert not torch.allclose(before[name][..., 1:, :], after[name][..., 1:, :])


def test_action_uncertainty_upgrade_starts_identical_and_preserves_factual_prefix():
    torch.manual_seed(2027)
    parent = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=3, num_robots=1, hidden_dim=16,
        num_heads=2, graph_layers=1, dropout=0, dt=.25, response_delay_steps=1,
        factual_prefix_dynamics=True,
    )).eval()
    candidate = add_action_conditioned_uncertainty(parent)
    names = configure_planner_finetuning(candidate, True)
    assert any(name.startswith("candidate_std_decoder.") for name in names)
    history = torch.randn(2, 1, 4, 2)
    current = torch.randn(2, 1, 1, 4)
    actions = torch.randn(2, 1, 1, 3, 4)
    with torch.no_grad():
        original = parent(history, current, actions)
        before = candidate(history, current, actions)
    for name in ("mean", "log_std"):
        torch.testing.assert_close(original[name], before[name], rtol=0, atol=0)
    optimizer = torch.optim.Adam((parameter for parameter in candidate.parameters() if parameter.requires_grad), lr=.01)
    candidate.train()
    prediction = candidate(history, current, actions)
    prediction["log_std"][..., 1:, :].square().mean().backward()
    optimizer.step()
    candidate.eval()
    with torch.no_grad():
        after = candidate(history, current, actions)
    for name in ("mean", "log_std"):
        torch.testing.assert_close(original[name][..., :1, :], after[name][..., :1, :], rtol=0, atol=0)
    assert not torch.allclose(original["log_std"][..., 1:, :], after["log_std"][..., 1:, :])
