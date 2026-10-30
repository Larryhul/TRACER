import numpy as np

import pytest

from tracer.experiments.audit_reliable_value_opportunity import audit_opportunity, parse_reference_actions
from tracer.response.executable_value import FEATURE_VERSION
from tracer.response.reliable_executable_value import DEFAULT_MINIMUM_GAINS, RELIABILITY_TARGETS


def model():
    member = {"feature_version": FEATURE_VERSION, "scale": [1.], "coefficients": np.zeros((1, 9)).tolist()}
    return {
        "feature_version": FEATURE_VERSION,
        "selection_objective": "safe-completion-lexicographic-v1",
        "target_names": list(RELIABILITY_TARGETS),
        "ensemble": [member] * 8,
        "uncertainty_multiplier": 2.,
        "minimum_gains": list(DEFAULT_MINIMUM_GAINS),
        "training_support": {"feature_indices": [0], "center": [0.], "scale": [1.],
                             "anchors": [[0.]], "maximum_distance": 1.},
        "response_risk_guard": True,
    }


def test_audit_separates_real_opportunity_from_risk_and_model_abstention():
    targets = np.zeros((3, 9))
    targets[1, 8] = -1.
    targets[2, 0] = -.5
    group = {"case": 64, "window": 0, "rerank_allowed": True, "parent": 0,
             "candidate_indices": [10, 11, 12], "features": np.array([[0.], [.5], [2.]]),
             "response_risk_support": np.array([True, False, True]), "targets": targets}
    report = audit_opportunity([group], model())
    assert report["actual_opportunity_groups"] == 1
    assert report["actual_risk_supported_opportunity_groups"] == 0
    assert report["predicted_risk_supported_opportunity_groups"] == 0
    assert report["deployed_change_groups"] == 0
    assert report["records"][0]["deployed_prediction_upper"]["elapsed_time"] == 0.


def test_audit_counts_deployed_stagnation_as_task_harm():
    changed = model()
    changed["ensemble"][0]["coefficients"][0][7] = -1.
    for member in changed["ensemble"][1:]:
        member["coefficients"][0][7] = -1.
    targets = np.zeros((2, 9))
    targets[1, 5] = .25
    group = {"case": 64, "window": 0, "rerank_allowed": True, "parent": 0,
             "candidate_indices": [10, 11], "features": np.array([[0.], [.5]]),
             "response_risk_support": np.array([True, True]), "targets": targets}
    assert audit_opportunity([group], changed)["deployed_safe_or_task_harm_groups"] == 1


def test_audit_counts_parent_outside_training_support():
    group = {"case": 64, "window": 0, "rerank_allowed": True, "parent": 0,
             "candidate_indices": [10, 11], "features": np.array([[3.], [0.]]),
             "response_risk_support": np.array([True, True]), "targets": np.zeros((2, 9))}
    report = audit_opportunity([group], model())
    assert report["parent_out_of_training_support_groups"] == 1
    assert report["deployed_change_groups"] == 0


def test_audit_can_reconstruct_an_overridden_online_reference_action():
    group = {"case": 698, "window": 4, "rerank_allowed": True, "parent": 0,
             "candidate_indices": [2, 18], "features": np.array([[0.], [.5]]),
             "diagnostics": {"weighted_risk": np.zeros(19), "weighted_safety": np.zeros(19)},
             "response_risk_support": np.array([True, True]), "targets": np.zeros((2, 9))}
    report = audit_opportunity([group], model(), {(698, 4): 18})
    assert report["records"][0]["parent"] == 18
    with pytest.raises(ValueError, match="absent"):
        audit_opportunity([group], model(), {(698, 4): 7})
    assert parse_reference_actions(["698:4:18"]) == {(698, 4): 18}


def test_audit_separates_risk_supported_from_planner_supported_opportunity():
    targets = np.zeros((2, 9))
    targets[1, 8] = -1.
    group = {
        "case": 64, "window": 0, "rerank_allowed": True, "parent": 0,
        "candidate_indices": [10, 11], "features": np.array([[0.], [.5]]),
        "planner_scores": np.array([0., 1.]),
        "response_risk_support": np.array([True, True]), "targets": targets,
    }
    report = audit_opportunity([group], model())
    assert report["actual_risk_supported_opportunity_groups"] == 1
    assert report["actual_deployment_supported_opportunity_groups"] == 0
    assert report["actual_risk_opportunities_rejected_by_planner_support"] == 1
    assert report["planner_supported_alternative_pairs"] == 0
