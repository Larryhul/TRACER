import json

import pytest
import torch

from sage.experiments.collect_component_policy_coverage import collect_policy_coverage
from sage.experiments.train_component_residual import collect_uniform_rollouts, trace_to_rollouts
from sage.experiments.train_component_mode_contrast import grouped_training_inputs
from sage.experiments.formal_manifest import sha256_file
from sage.response.closed_loop import ClosedLoopSpec, PlanningCostConfig
from sage.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig


@pytest.mark.parametrize("retention", [False, True])
def test_coverage_preserves_initial_states_noise_and_uniform_collector(retention):
    torch.manual_seed(62)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(history=4, horizon=4, num_robots=2, hidden_dim=16, dropout=0.)).eval()
    raw = {"simulation": {"windows": 3, "history": 4, "horizon": 4, "num_robots": 2, "num_candidates": 6, "execute_steps": 1, "dt": .4, "data_seed": 123},
           "candidate_source": {"type": "synthetic", "retain_previous_plan": retention}, "planning": {"task_steps": 1}}
    uniform, uniform_audit = collect_policy_coverage(raw, None, model, model, 6, "cpu", "uniform")
    original = collect_uniform_rollouts(raw, None, model, 6, "cpu", all_mode_supervision=True)
    for name, value in original.items():
        torch.testing.assert_close(uniform[name], value, rtol=0., atol=0.)
    factorial, factorial_audit = collect_policy_coverage(raw, None, model, model, 6, "cpu", "factorial")
    assert factorial["behavior_policy_index"].tolist() == list(range(6))
    for first, second in zip(uniform_audit, factorial_audit):
        assert first["candidate_seed"] == second["candidate_seed"]
        assert first["environment_rng_hash"] == second["environment_rng_hash"]
        if first["window"] == 0:
            assert first["state_hash"] == second["state_hash"]
            assert first["candidate_hash"] == second["candidate_hash"]
    for episode in range(6):
        torch.testing.assert_close(factorial["responses"][episode], factorial["all_mode_responses"][episode, :, factorial["mode"][episode]], rtol=0., atol=0.)
    with pytest.raises(ValueError, match="nonempty"):
        trace_to_rollouts([], ClosedLoopSpec(), PlanningCostConfig(), 3)


def test_coverage_training_rejects_changed_labels_and_missing_completion(tmp_path):
    source = tmp_path / "uniform"
    source.mkdir()
    files = [source / name for name in ("train_rollouts.pt", "validation_rollouts.pt")]
    for path in files:
        path.write_bytes(b"frozen-labels")
    raw = {"simulation": {"data_seed": 5}}
    origin = {"config": {"schema": "component-policy-coverage-v1", "all_mode_supervision": True, "config": raw, "arguments": {}}}
    (tmp_path / "run_manifest.json").write_text(json.dumps(origin))
    summary = {"status": "COMPLETED", "changed_inputs": [], "trajectory_sha256": {str(path): sha256_file(path) for path in files}, "shared_validation_sha256": sha256_file(files[1])}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    inputs, metadata = grouped_training_inputs(source, raw)
    assert len(inputs) == 4 and metadata["collection_profile"] == "uniform"
    files[0].write_bytes(b"altered-labels")
    with pytest.raises(ValueError, match="hash mismatch"):
        grouped_training_inputs(source, raw)
