import json

import numpy as np
import pytest
import torch

from sage.experiments.eval_component_factorial import FACTORIAL_POLICIES, audit_shared_emissions
from sage.experiments.eval_persistent_closed_loop import _evaluate_policy, _select_response_costs
from sage.experiments.train_component_value import fit_value_head, value_error
from sage.experiments.calibrate_component_value import choose_value_weight
from sage.experiments.component_value_protocol import load_value_weight
from sage.experiments.calibrate_component_memory import component_protocol_id, load_component_calibration
from sage.experiments.formal_manifest import sha256_file
from sage.response.closed_loop import ClosedLoopSpec, PlanningCostConfig, SyntheticClosedLoopSimulator
from sage.response.continuation_value import FrozenContinuationValue, add_continuation_value, observable_value_features, predicted_next_value_features
from sage.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig


def test_value_features_match_executed_prefix_and_ignore_unexecuted_tail():
    simulator = SyntheticClosedLoopSimulator(ClosedLoopSpec(history=4, horizon=4, execute_steps=1, num_candidates=6), 42)
    pool = simulator.candidate_pool()
    outcome = simulator.counterfactuals(pool, PlanningCostConfig())[0]
    history = torch.from_numpy(simulator.history)
    means = outcome.response[None, None].expand(1, 3, -1, -1).clone()
    predicted = predicted_next_value_features(history, pool.robot_current[:1], pool.robot_candidates[:1], means, 1, 7, 8, .4)
    simulator.step(pool, 0, PlanningCostConfig())
    current = torch.from_numpy(np.concatenate([simulator.robot_positions - simulator.entity_position, simulator.robot_velocities], -1))
    expected = observable_value_features(torch.from_numpy(simulator.history), current, 7, 8, .4)
    torch.testing.assert_close(predicted[0, 0], expected, atol=2e-5, rtol=1e-5)
    means[..., 1:, :] += 1000
    again = predicted_next_value_features(history, pool.robot_current[:1], pool.robot_candidates[:1], means, 1, 7, 8, .4)
    torch.testing.assert_close(predicted, again)


def test_value_features_are_rigid_motion_and_robot_permutation_invariant():
    torch.manual_seed(71)
    history, current = torch.randn(5, 4, 2), torch.randn(5, 3, 4)
    reference = observable_value_features(history, current, 3, 8, .4)
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    moved = torch.cat([current[..., :2] @ rotation, current[..., 2:] @ rotation], -1)[:, [2, 0, 1]]
    updated = observable_value_features(history @ rotation + 7., moved, 3, 8, .4)
    torch.testing.assert_close(reference, updated)


def test_continuation_keeps_hard_safety_and_risk_fallback():
    costs = {"unconstrained_total": torch.tensor([0., 1., 2.]), "total": torch.tensor([float("inf"), 1., 2.]), "safe_mask": torch.tensor([False, True, True]), "violation_probability": torch.tensor([.8, .1, .05])}
    values = torch.tensor([-100., 0., -.5])
    updated = add_continuation_value(costs, values, 1.)
    assert _select_response_costs(updated, "total") == (1, False)
    torch.testing.assert_close(updated["violation_probability"], costs["violation_probability"])
    updated["safe_mask"] = torch.zeros(3, dtype=torch.bool)
    assert _select_response_costs(updated, "total") == (2, True)
    original = add_continuation_value(costs, values, 0.)
    torch.testing.assert_close(original["total"], costs["total"])


def test_value_weight_selection_and_provenance(tmp_path):
    records = [{"weight": 0., "cost": 1., "collision": 0.}, {"weight": .5, "cost": .8, "collision": 0.}, {"weight": 1., "cost": .1, "collision": .1}]
    assert choose_value_weight(records)["weight"] == .5
    records[1]["cost"] = 1.
    assert choose_value_weight(records)["weight"] == 0.
    checkpoint = tmp_path / "value.pt"
    checkpoint.write_bytes(b"value")
    raw = {"simulation": {}, "planning": {}, "candidate_source": {"type": "synthetic"}}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"status": "COMPLETED", "phase": "validation", "changed_inputs": [], "value_sha256": sha256_file(checkpoint), "component_protocol_id": component_protocol_id(raw, True), "data_seed": 12, "selected": {"weight": .5}}))
    assert load_value_weight(path, checkpoint, raw, 13) == .5
    with pytest.raises(ValueError, match="separate"):
        load_value_weight(path, checkpoint, raw, 12)
    checkpoint.write_bytes(b"modified")
    with pytest.raises(ValueError, match="mismatch"):
        load_value_weight(path, checkpoint, raw, 13)


def test_memory_calibration_binds_continuation_value(tmp_path):
    checkpoint, value_checkpoint = tmp_path / "response.pt", tmp_path / "value.pt"
    checkpoint.write_bytes(b"response")
    value_checkpoint.write_bytes(b"value")
    spec = ClosedLoopSpec()
    record = {"schema": "component-memory-factual-v1", "status": "COMPLETED", "phase": "validation", "checkpoint_sha256": sha256_file(checkpoint), "data_seed": 3,
              "selected": {"likelihood_scale": .5}, "continuation_value_sha256": sha256_file(value_checkpoint), "continuation_value_weight": .5,
              **{name: getattr(spec, name) for name in ("dt", "execute_steps", "num_robots", "horizon")}}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(record))
    assert load_component_calibration(path, checkpoint, spec, 4, continuation_value_checkpoint=value_checkpoint, continuation_value_weight=.5) == .5
    with pytest.raises(ValueError, match="continuation"):
        load_component_calibration(path, checkpoint, spec, 4)
    with pytest.raises(ValueError, match="continuation"):
        load_component_calibration(path, checkpoint, spec, 4, continuation_value_checkpoint=value_checkpoint, continuation_value_weight=.25)


def test_frozen_value_fit_and_shared_state_evidence_contract():
    torch.manual_seed(72)
    features = torch.randn(60, 27, dtype=torch.float64)
    modes = torch.arange(60) % 3
    remaining = torch.ones(60, dtype=torch.float64) * 3
    data = {"features": features, "modes": modes, "remaining": remaining, "returns": (features[:, 0] + modes) * remaining}
    center, scale, coefficients = fit_value_head(data, 3, .01)
    head = FrozenContinuationValue(center, scale, coefficients, 3, .4, 1)
    assert value_error(head, data) < .001
    assert head.mode_values(features[:, None].expand(-1, 3, -1), 0).eq(0).all()
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(history=4, horizon=4, num_robots=2, hidden_dim=16, dropout=0.)).eval()
    shared = dict(model=model, spec=ClosedLoopSpec(history=4, horizon=4, windows=3, num_robots=2, num_candidates=6, execute_steps=1), episodes=2,
                  device="cpu", planning_cost=PlanningCostConfig(task_steps=1), likelihood_scale=.05, prefix_objective=True, evaluation_mode="same_state")
    trace = []
    reference = _evaluate_policy(**shared, policy=FACTORIAL_POLICIES[0], captured_reference_trace=trace)
    zero = _evaluate_policy(**shared, policy=FACTORIAL_POLICIES[0], reference_trace=trace, continuation_value_model=head, continuation_value_weight=0.)
    assert reference["selected_candidates"] == zero["selected_candidates"]
    policies = {}
    for policy in FACTORIAL_POLICIES:
        weight = float(policy.name.endswith("-joint"))
        policies[policy.name] = _evaluate_policy(**shared, policy=policy, reference_trace=trace, response_gate_min=weight, response_gate_max=weight, continuation_value_model=head)
    assert audit_shared_emissions(policies)["status"] == "pass"
    for result in policies.values():
        for rows in result["window_records"]:
            assert rows[-1]["continuation_value_adjustment"] == [0.] * 6
    assert any(not np.allclose(left["planning_candidate_utility"], right["planning_candidate_utility"]) for left_rows, right_rows in zip(policies["uniform-joint"]["window_records"], policies["persistent-joint"]["window_records"]) for left, right in zip(left_rows, right_rows))
    with pytest.raises(ValueError, match="memoryless"):
        _evaluate_policy(**shared, policy=FACTORIAL_POLICIES[0], reference_trace=trace, continuation_value_model=head, retain_previous_plan=True)


def test_value_uses_current_mixture_without_updating_belief():
    torch.manual_seed(93)
    simulator = SyntheticClosedLoopSimulator(ClosedLoopSpec(history=4, horizon=4, execute_steps=1, num_candidates=6), 2)
    pool = simulator.candidate_pool()
    history = torch.from_numpy(simulator.history)
    means = torch.randn(6, 3, 4, 2)
    head = FrozenContinuationValue(torch.zeros(27), torch.ones(27), torch.randn(3, 28), 8, .4, 1)
    first = torch.tensor([1., 0., 0.])
    second = torch.tensor([0., 0., 1.])
    mixture = .3 * first + .7 * second
    before = mixture.clone()
    values = [head.candidate_values(history, pool.robot_current, pool.robot_candidates, means, belief, 7) for belief in [first, second, mixture]]
    torch.testing.assert_close(values[2], .3 * values[0] + .7 * values[1])
    torch.testing.assert_close(mixture, before, rtol=0., atol=0.)
    with pytest.raises(ValueError, match="normalized"):
        head.candidate_values(history, pool.robot_current, pool.robot_candidates, means, mixture * 2, 7)
    with pytest.raises(ValueError, match="positive feature scales"):
        FrozenContinuationValue(torch.zeros(27), torch.zeros(27), torch.randn(3, 28), 8, .4, 1)
