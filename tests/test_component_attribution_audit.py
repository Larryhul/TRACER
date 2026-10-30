import copy
import json
import shutil

import pytest
import torch

from tracer.experiments.audit_component_attribution import audit_directory, compare_inference_only, compare_inference_settings, decision_contribution, paired_pipeline_contrasts
from tracer.experiments.eval_component_factorial import FACTORIAL_POLICIES, summarize_factorial
from tracer.experiments.eval_persistent_closed_loop import _evaluate_policy
from tracer.experiments.merge_component_factorial import compatible_shard_configuration, merge_factorial_shards
from tracer.response.closed_loop import ClosedLoopSpec, PlanningCostConfig
from tracer.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig


def policies():
    row = {
        "window": 0, "state_hash": "state", "candidate_hash": "candidates", "candidate_seed": 11,
        "environment_rng_hash": "rng", "executed_candidate": 0, "executed_response_hash": "response",
        "selected_candidate": 0, "selected_true_cost": .2,
    }
    return {name: {"window_records": [[copy.deepcopy(row), dict(row, window=1)]]}
            for name in ("uniform-joint", "reset-joint", "persistent-joint")}


def test_decision_contribution_counts_only_matched_candidate_changes():
    data = policies()
    row = data["persistent-joint"]["window_records"][0][1]
    row.update(selected_candidate=1, selected_true_cost=.1)
    result = decision_contribution(data, "persistent-joint", "reset-joint")
    assert result["candidate_changes"] == 1
    assert result["candidate_change_fraction"] == .5
    assert result["beneficial_changes"] == 1
    assert result["mean_cost_difference"] == pytest.approx(-.05)
    assert result["mean_cost_difference_when_changed"] == pytest.approx(-.1)
    row["state_hash"] = "diverged"
    with pytest.raises(ValueError, match="shared factual states"):
        decision_contribution(data, "persistent-joint", "reset-joint")


def test_same_selection_cannot_have_an_apparent_gain():
    data = policies()
    result = decision_contribution(data, "persistent-joint", "reset-joint")
    assert result["candidate_changes"] == 0
    assert result["mean_cost_difference_when_changed"] is None
    data["persistent-joint"]["window_records"][0][0]["selected_true_cost"] = .1
    with pytest.raises(ValueError, match="identical costs"):
        decision_contribution(data, "persistent-joint", "reset-joint")


def test_inference_change_may_change_memory_rollouts_but_not_uniform():
    old = {"closed_loop": {"2027": {"id": policies()}}}
    new = copy.deepcopy(old)
    changed = new["closed_loop"]["2027"]["id"]["persistent-joint"]["window_records"][0][1]
    changed.update(selected_candidate=1, state_hash="diverged", candidate_hash="new_candidates")
    assert compare_inference_only(old, new)["unchanged_uniform_windows"] == 2
    new["closed_loop"]["2027"]["id"]["uniform-joint"]["window_records"][0][1]["selected_true_cost"] = .1
    with pytest.raises(ValueError, match="altered Uniform"):
        compare_inference_only(old, new)


@pytest.mark.parametrize("key", ["state_hash", "candidate_hash", "environment_rng_hash"])
def test_inference_comparison_rejects_unpaired_initial_conditions(key):
    old = {"closed_loop": {"2027": {"id": policies()}}}
    new = copy.deepcopy(old)
    new["closed_loop"]["2027"]["id"]["reset-joint"]["window_records"][0][0][key] = "different"
    with pytest.raises(ValueError, match="initial state mismatch"):
        compare_inference_only(old, new)


@pytest.fixture
def factorial_evidence(tmp_path):
    torch.manual_seed(99)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=4, num_robots=2,
        hidden_dim=16, num_heads=2, dropout=0.,
    )).eval()
    modes = ("same_state", "closed_loop")
    runs = {mode: {"99": {"id": {}}} for mode in modes}
    spec = ClosedLoopSpec(history=4, horizon=4, windows=2, num_robots=2, num_candidates=4, execute_steps=1, seed=99)
    for mode in modes:
        trace = []
        for policy in FACTORIAL_POLICIES:
            weight = 0. if policy.name.endswith("-additive") else .5
            result = _evaluate_policy(model, policy, spec, 1, "cpu", PlanningCostConfig(task_steps=1), .1,
                evaluation_mode=mode, captured_reference_trace=trace if mode == "same_state" and not trace else None,
                reference_trace=trace if mode == "same_state" and trace else None,
                response_gate_min=weight, response_gate_max=weight)
            runs[mode]["99"]["id"][policy.name] = result
        (tmp_path / f"seed99_id_{mode}.json").write_text(json.dumps(runs[mode]["99"]["id"]))
    arguments = {"evaluation_modes": list(modes), "seeds": [99], "splits": ["id"], "episodes": 1,
                 "bootstrap_resamples": 20, "likelihood_scale": .1, "joint_weight": .5,
                 "data_seed": 99, "device": "cpu", "num_threads": 2, "phase": "validation"}
    configuration = {"arguments": arguments, "config": {"simulation": vars(spec), "planning": {}, "candidate_source": {}}}
    manifest = {"sources": {}, "checkpoints": {}, "config": configuration}
    settings = {"99": {"likelihood_scale": .1, "joint_weight": .5, "planning_checkpoint": "shared.pt",
                       "prefix_objective": False, "retain_previous_plan": False,
                       "continuation_value_checkpoint": None, "continuation_value_weight": 0.}}
    summary = {"status": "COMPLETED", "phase": "validation", "changed_sources": [], "changed_checkpoints": [],
               "effective_settings": settings, "results": {mode: summarize_factorial(data, 20) for mode, data in runs.items()}}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    return tmp_path


def test_audit_reconstructs_evidence_and_rejects_unsupported_summary(factorial_evidence):
    result, _ = audit_directory(factorial_evidence)
    assert result["status"] == "COMPLETED"
    changed = copy.deepcopy(result)
    changed["effective_settings"]["99"]["planning_checkpoint"] = "other.pt"
    with pytest.raises(ValueError, match="planning_checkpoint"):
        compare_inference_settings(result, changed)
    path = factorial_evidence / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["results"]["closed_loop"]["id"]["policy_means"]["persistent-joint"]["realized_total_cost"] -= 1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="reconstructed episode evidence"):
        audit_directory(factorial_evidence)


def test_audit_rejects_unused_runtime_weight(factorial_evidence):
    path = factorial_evidence / "seed99_id_closed_loop.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["persistent-joint"]["window_records"][0][0]["response_gate_weight"] = 1.
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="runtime Joint weight"):
        audit_directory(factorial_evidence)


def test_pipeline_contrast_resamples_scenes_not_seed_repeats(factorial_evidence):
    _, runs = audit_directory(factorial_evidence)
    for mode in runs:
        runs[mode]["100"] = copy.deepcopy(runs[mode]["99"])
    candidate = copy.deepcopy(runs)
    for mode in candidate:
        for seed in candidate[mode]:
            candidate[mode][seed]["id"]["persistent-joint"]["episode_records"][0]["realized_total_cost"] -= .1
    result = paired_pipeline_contrasts(runs, candidate, resamples=50)
    contrast = result["closed_loop"]["id"]["persistent-joint"]["realized_total_cost"]
    assert contrast["mean_difference"] == pytest.approx(-.1)
    assert contrast["case_bootstrap_ci95"] == pytest.approx([-.1, -.1])
    assert contrast["independent_scenes"] == 1
    candidate["closed_loop"]["100"]["id"]["persistent-joint"]["episode_records"][0]["episode"] = 2
    with pytest.raises(ValueError, match="episodes must align"):
        paired_pipeline_contrasts(runs, candidate, resamples=50)


def test_shard_compatibility_allows_only_partition_and_output_changes():
    first = {"arguments": {"seeds": [1], "splits": ["id"], "output_dir": "first", "data_seed": 9}, "config": {"dt": .4}}
    second = copy.deepcopy(first)
    second["arguments"].update(seeds=[2], splits=["ood"], output_dir="second")
    compatible_shard_configuration(first, second)
    second["config"]["dt"] = .3
    with pytest.raises(ValueError, match="incompatible"):
        compatible_shard_configuration(first, second)


def test_shard_merge_preserves_bytes_and_scene_count(factorial_evidence):
    output = factorial_evidence.parent / f"{factorial_evidence.name}_merged"
    result = merge_factorial_shards([factorial_evidence], output)
    assert result["status"] == "COMPLETED"
    assert result["results"]["closed_loop"]["id"]["independent_scenes"] == 1
    assert (output / "seed99_id_closed_loop.json").read_bytes() == (factorial_evidence / "seed99_id_closed_loop.json").read_bytes()
    duplicate = factorial_evidence.parent / f"{factorial_evidence.name}_duplicate"
    shutil.copytree(factorial_evidence, duplicate)
    with pytest.raises(ValueError, match="duplicate seed"):
        merge_factorial_shards([factorial_evidence, duplicate], output.parent / "invalid_merge")
