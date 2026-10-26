from copy import deepcopy
import math

import pytest
import torch

from sage.experiments.calibrate_external_belief_cadence import (
    DEFAULT_SCALES,
    _controller_args,
    build_track_tensors,
    calibrate_external_cadence,
    select_validation_scale,
    validate_evidence,
    validation_case_splits,
)


def _checkpoint():
    return {"provenance": {"case_splits": {"train": list(range(112, 136)), "val": list(range(136, 144))}}}


def _splits():
    return validation_case_splits(_checkpoint(), {"case_start": 88, "cases": 24})


def _evidence():
    episodes, rows, labels = [], [], []
    for case in _splits()["val"]:
        episodes.append({"case": case, "seed": 2027, "policy": "tracer-uniform", "human_num": 1,
                         "windows": 3, "steps": 3, "belief_update_windows": 3,
                         "environment_dt": 0.25, "robot_kinematics": "unicycle"})
        labels.append({"case": case, "human": 0, "mode": 0})
        for window in range(3):
            rows.append({"case": case, "seed": 2027, "window": window, "policy": "tracer-uniform",
                         "factual_response_steps": 1, "belief_updated_this_window": 1,
                         "factual_mode_log_likelihood": [[2.0, 0.0, -1.0]],
                         "decision_log_beliefs": [[-math.log(3)] * 3],
                         "posterior_log_beliefs": [[-math.log(3)] * 3]})
    return rows, episodes, labels


def test_cases_come_from_checkpoint_val_not_metadata_test_range():
    splits = _splits()
    assert splits["val"] == list(range(136, 144))
    assert splits["test"] == list(range(88, 112))
    assert splits["train"] == list(range(112, 136))


@pytest.mark.parametrize("cases", [[88], [112], [136], list(range(136, 144)) + [88], [136] * 8])
def test_reject_test_training_partial_or_duplicate_collection_cases(cases):
    with pytest.raises(ValueError):
        validation_case_splits(_checkpoint(), {"case_start": 88, "cases": 24}, cases)


@pytest.mark.parametrize("changed_split", ["train", "test"])
def test_reject_overlapping_checkpoint_splits(changed_split):
    checkpoint = _checkpoint()
    checkpoint["provenance"]["case_splits"][changed_split] = [136]
    with pytest.raises(ValueError, match="overlap"):
        validation_case_splits(checkpoint, {"case_start": 88, "cases": 24})


def test_reject_checkpoint_without_validation_split():
    checkpoint = _checkpoint()
    del checkpoint["provenance"]["case_splits"]["val"]
    with pytest.raises(ValueError, match="val"):
        validation_case_splits(checkpoint, {"case_start": 88, "cases": 24})


def test_raw_evidence_keeps_identical_valued_but_distinct_ticks():
    rows, episodes, labels = _evidence()
    ordered = validate_evidence(rows, episodes, _splits(), 2027)
    likelihoods, modes, active, sequences = build_track_tensors(ordered, labels)
    assert likelihoods.shape == (8, 3, 3)
    assert active.sum() == 24
    assert modes.tolist() == [0] * 8
    assert all(sequence["windows"] == [0, 1, 2] for sequence in sequences)


def test_reject_duplicate_factual_evidence():
    rows, episodes, _ = _evidence()
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate factual evidence"):
        validate_evidence(rows, episodes, _splits(), 2027)


@pytest.mark.parametrize("removed", [0, 1, 2])
def test_reject_missing_windows_including_last_tick(removed):
    rows, episodes, _ = _evidence()
    rows.pop(removed)
    with pytest.raises(ValueError, match="missing windows"):
        validate_evidence(rows, episodes, _splits(), 2027)


@pytest.mark.parametrize("field,value", [
    ("case", 88), ("case", 112), ("factual_response_steps", 2), ("belief_updated_this_window", 0),
    ("policy", "tracer-persistent"), ("factual_mode_log_likelihood", [[float("nan")] * 3]),
    ("factual_mode_log_likelihood", [[0.0, 1.0]]), ("posterior_log_beliefs", [[-0.1, -2.0, -3.0]]),
    ("branch", 0),
])
def test_reject_invalid_or_nonfactual_evidence(field, value):
    rows, episodes, _ = _evidence()
    rows[0][field] = value
    with pytest.raises(ValueError):
        validate_evidence(rows, episodes, _splits(), 2027)


def test_reject_missing_evidence_interface():
    rows, episodes, _ = _evidence()
    del rows[0]["factual_mode_log_likelihood"]
    with pytest.raises(ValueError, match="belief_evidence_audit"):
        validate_evidence(rows, episodes, _splits(), 2027)


@pytest.mark.parametrize("phase,split", [("test", "val"), ("validation", "test"), ("train", "train")])
def test_selection_and_collection_only_accept_validation(phase, split, tmp_path):
    rows, _, labels = _evidence()
    likelihoods, modes, active, _ = build_track_tensors(rows, labels)
    with pytest.raises(ValueError, match="validation"):
        select_validation_scale(likelihoods, modes, active, phase=phase, split=split,
                                cases=_splits()["val"], case_splits=_splits())
    with pytest.raises(ValueError, match="validation"):
        calibrate_external_cadence(tmp_path / "missing.pt", tmp_path / "missing.json", tmp_path / "output",
                                   phase=phase, split=split)
    assert not (tmp_path / "output").exists()


def test_validation_phase_cannot_relabel_test_cases():
    rows, _, labels = _evidence()
    likelihoods, modes, active, _ = build_track_tensors(rows, labels)
    with pytest.raises(ValueError, match="validation cases"):
        select_validation_scale(likelihoods, modes, active, phase="validation", split="val",
                                cases=[88], case_splits=_splits())


def test_selection_uses_pre_observation_nll_not_current_or_final_labels():
    rows, episodes, labels = _evidence()
    likelihoods, modes, active, _ = build_track_tensors(validate_evidence(rows, episodes, _splits(), 2027), labels)
    arguments = {"phase": "validation", "split": "val", "cases": _splits()["val"], "case_splits": _splits()}
    result = select_validation_scale(likelihoods, modes, active, **arguments)
    assert [record["likelihood_scale"] for record in result["records"]] == list(DEFAULT_SCALES)
    assert result["selected"]["likelihood_scale"] == 1.0
    assert result["selected"]["decisions"] == 16
    altered = likelihoods.clone()
    altered[:, -1, 0] = -1000.0
    changed = select_validation_scale(altered, modes, active, **arguments)
    assert [record["decision_mode_nll"] for record in result["records"]] == [
        record["decision_mode_nll"] for record in changed["records"]
    ]
    assert changed["selected"]["likelihood_scale"] == result["selected"]["likelihood_scale"]
    assert changed["selected"]["posterior_mode_nll"] > result["selected"]["posterior_mode_nll"]


def test_collection_namespace_does_not_inherit_old_scale_or_counterfactual_audits(tmp_path):
    parameters = {"execute_steps": 1, "env_dt": 0.25, "model_dt": 0.25,
                  "motion_profile": "sicnav-unicycle", "response_backend": "crowdsimplus-persistent",
                  "static_braking_guard": True, "geometric_safety_prefix_steps": 1,
                  "likelihood_scale": 0.05, "belief_update_stride": 6,
                  "ranking_audit_windows": 4, "response_audit_stride": 1,
                  "static_route_weight": 2.0, "kinematic_safety_margin": 0.02}
    original = deepcopy(parameters)
    args = _controller_args(parameters, tmp_path / "seed2027/response.pt", tmp_path / "output", _splits(), 2027, "cpu")
    assert parameters == original
    assert args.policies == ["tracer-uniform"]
    assert args.belief_update_stride == args.execute_steps == 1
    assert args.belief_evidence_audit
    assert args.likelihood_scale == 1.0
    assert args.ranking_audit_windows == args.response_audit_stride == 0
    assert (args.case_start, args.cases) == (136, 8)
    assert args.static_route_weight == 2.0
    assert args.kinematic_safety_margin == 0.02


def test_reject_noncontiguous_track_masks():
    likelihoods = torch.zeros(1, 3, 3)
    with pytest.raises(ValueError, match="contiguous"):
        select_validation_scale(likelihoods, torch.tensor([0]), torch.tensor([[True, False, True]]),
                                phase="validation", split="val", cases=_splits()["val"], case_splits=_splits())


def test_response_calibration_scores_predictions_before_current_observation():
    from sage.belief.filter import run_categorical_filter
    likelihoods = torch.tensor([[[2., 0., -1.], [1., 0., -1.], [-2., 2., 0.]]])
    active = torch.ones(1, 3, dtype=torch.bool)
    arguments = {"phase": "validation", "split": "val", "cases": _splits()["val"], "case_splits": _splits(),
                 "selection_metric": "decision_response_nll", "scales": [.1, 1.]}
    result = select_validation_scale(likelihoods, torch.tensor([0]), active, **arguments)
    changed_labels = select_validation_scale(likelihoods, torch.tensor([2]), active, **arguments)
    assert result["selected"]["likelihood_scale"] == changed_labels["selected"]["likelihood_scale"]
    for record in result["records"]:
        before, _ = run_categorical_filter(likelihoods * record["likelihood_scale"])
        expected = -torch.logsumexp(before[:, 1:] + likelihoods[:, 1:], -1).mean() / 2
        assert record["decision_response_nll"] == pytest.approx(float(expected))
