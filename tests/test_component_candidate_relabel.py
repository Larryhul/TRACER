import copy
import json

import pytest
import torch

from tracer.experiments.collect_component_candidate_relabel import GROUP_FIELDS, relabel_observed_states, response_labels, validate_relabel_protocol
from tracer.experiments.eval_component_factorial import FACTORIAL_POLICIES
from tracer.experiments.eval_persistent_closed_loop import _evaluate_policy
from tracer.experiments.formal_manifest import sha256_file
from tracer.experiments.train_component_mode_contrast import grouped_training_inputs
from tracer.experiments.train_component_residual import trace_to_rollouts
from tracer.response.closed_loop import ClosedLoopSpec, PlanningCostConfig, SyntheticClosedLoopSimulator
from tracer.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig


@pytest.fixture
def source_data():
    torch.manual_seed(23)
    spec = ClosedLoopSpec(history=4, horizon=4, windows=3, num_robots=2, num_candidates=6, execute_steps=1, seed=23)
    planning = PlanningCostConfig(task_steps=1)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(architecture="tracer", history=4, horizon=4,
        num_robots=2, hidden_dim=16, num_heads=2, dropout=0.)).eval()
    trace = []
    result = _evaluate_policy(model, FACTORIAL_POLICIES[0], spec, 2, "cpu", planning, .05,
        captured_reference_trace=trace, response_gate_min=1., response_gate_max=1., prefix_objective=True)
    data = trace_to_rollouts(trace, spec, planning, 3, all_mode_supervision=True)
    rows = [dict(row, episode=episode) for episode, windows in enumerate(result["window_records"]) for row in windows]
    return data, rows, spec, planning


def test_candidate_relabel_preserves_factual_states_budget_and_original_inputs(source_data):
    data, rows, spec, planning = source_data
    before = {name: value.clone() for name, value in data.items()}
    original, completed, audit = relabel_observed_states(data, rows, spec, planning, (.4, 1.), 1.2)
    assert len(audit) == 6
    assert set(completed) == set(GROUP_FIELDS)
    for name in GROUP_FIELDS:
        torch.testing.assert_close(data[name], before[name], rtol=0, atol=0)
        torch.testing.assert_close(original[name], data[name], rtol=0, atol=0)
        assert original[name].shape == completed[name].shape
    torch.testing.assert_close(completed["history"], original["history"], rtol=0, atol=0)
    assert not torch.equal(completed["candidates"], original["candidates"])
    assert "responses" not in completed and "mode" not in completed
    broken = copy.deepcopy(rows)
    broken[1]["state_hash"] = "incorrect"
    with pytest.raises(ValueError, match="factual state"):
        relabel_observed_states(data, broken, spec, planning, (.4, 1.), 1.2)
    with pytest.raises(ValueError, match="exactly once"):
        relabel_observed_states(data, rows+rows[:1], spec, planning, (.4, 1.), 1.2)


def test_counterfactual_labels_restore_hidden_mode_and_rng():
    spec = ClosedLoopSpec(horizon=4, history=4, num_robots=2, num_candidates=6)
    simulator = SyntheticClosedLoopSimulator(spec, 0)
    pool = simulator.candidate_pool()
    rng = json.dumps(simulator.rng.bit_generator.state, sort_keys=True)
    mode = simulator.mode
    first = response_labels(simulator, pool, PlanningCostConfig(), 3)
    second = response_labels(simulator, pool, PlanningCostConfig(), 3)
    assert simulator.mode == mode
    assert json.dumps(simulator.rng.bit_generator.state, sort_keys=True) == rng
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_relabel_protocol_rejects_changed_cadence_or_cost():
    source = {"simulation": {"data_seed": 1}, "planning": {"human_weight": .6},
              "candidate_source": {"goal_speed": 1.2, "goal_direction": "task_goal", "task_prefix_steps": 3}}
    target = copy.deepcopy(source)
    target["simulation"]["data_seed"] = 2
    target["candidate_source"]["joint_speed_completion"] = [.4, 1.]
    validate_relabel_protocol(source, target)
    target["planning"]["human_weight"] = .9
    with pytest.raises(ValueError, match="cadence or planning cost"):
        validate_relabel_protocol(source, target)


def test_relabel_training_loader_binds_dataset_and_marks_nonfactual_labels(tmp_path):
    directory = tmp_path / "completed"
    directory.mkdir()
    config = {"simulation": {"dt": .4}}
    files = [directory / f"{split}_rollouts.pt" for split in ("train", "validation")]
    for path in files:
        path.write_bytes(b"bound dataset")
    origin = {"schema": "component-candidate-relabel-v1", "config": config, "all_mode_supervision": True,
              "source_arguments": {"train_data_seed": 1, "validation_data_seed": 2},
              "state_scope": "fixed original states", "label_scope": "offline counterfactual labels, not an executed observation stream"}
    (tmp_path / "run_manifest.json").write_text(json.dumps({"config": origin}))
    summary = {"schema": origin["schema"], "status": "COMPLETED", "changed_inputs": [],
               "trajectory_sha256": {str(path): sha256_file(path) for path in files},
               "shared_validation_sha256": sha256_file(files[1]), "validation_context_repetitions": 2,
               "validation_independent_scenes": 32}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    _, arguments = grouped_training_inputs(directory, config)
    assert arguments["validation_independent_scenes"] == 32
    assert arguments["validation_context_repetitions"] == 2
    assert "not an executed" in arguments["label_scope"]
    files[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="trajectory hash mismatch"):
        grouped_training_inputs(directory, config)
