from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import yaml

from tracer.experiments import calibrate_persistent_belief as calibration_module
from tracer.experiments.calibrate_persistent_belief import (
    _calibration_prediction,
    calibrate_checkpoint,
    calibration_scores,
    load_belief_calibration,
    validate_belief_calibration,
)
from tracer.experiments.formal_manifest import sha256_file
from tracer.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
)


@pytest.fixture
def artifact(tmp_path):
    checkpoint = tmp_path / "response.pt"
    checkpoint.write_bytes(b"frozen checkpoint bytes")
    payload = {
        "phase": "validation",
        "split": "val",
        "selection_metric": "decision_mode_nll",
        "checkpoint_sha256": sha256_file(checkpoint),
        "execute_steps": 3,
        "dt": 0.4,
        "num_robots": 3,
        "selected": {"likelihood_scale": 0.5, "decision_mode_nll": 0.3},
    }
    path = tmp_path / "calibration.json"
    return path, checkpoint, payload


def _load_artifact(artifact):
    path, checkpoint, payload = artifact
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_belief_calibration(
        path, checkpoint, execute_steps=3, dt=0.4, num_robots=3
    )


def test_loader_returns_consumable_selected_record(artifact):
    loaded = _load_artifact(artifact)
    assert loaded["selected"]["likelihood_scale"] == 0.5
    assert loaded["checkpoint_sha256"] == sha256_file(artifact[1])


def test_loader_preserves_external_extensions_and_accepts_keyword_path(artifact):
    path, checkpoint, payload = artifact
    payload.update({
        "execute_steps": 1,
        "num_robots": 1,
        "observation_stride": 1,
        "num_modes": 3,
        "case_splits": {"train": [1, 2], "val": [3, 4], "test": [5, 6]},
    })
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_belief_calibration(
        path=path, checkpoint_path=checkpoint, execute_steps=1, dt=0.4, num_robots=1
    )
    assert loaded == payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "test"),
        ("split", "test"),
        ("split", "ood"),
        ("selection_metric", "posterior_mode_nll"),
        ("selection_metric", "predictive_trajectory_nll"),
        ("checkpoint_sha256", "0" * 64),
        ("execute_steps", 1),
        ("execute_steps", 6),
        ("execute_steps", 3.5),
        ("execute_steps", True),
        ("dt", 0.25),
        ("dt", float("nan")),
        ("num_robots", 1),
        ("num_robots", 0),
    ],
)
def test_loader_rejects_contract_mismatch(artifact, field, value):
    artifact[2][field] = value
    with pytest.raises(ValueError, match=field):
        _load_artifact(artifact)


@pytest.mark.parametrize(
    "field",
    ["phase", "split", "selection_metric", "checkpoint_sha256", "execute_steps", "dt", "num_robots", "selected"],
)
def test_loader_rejects_missing_contract_fields(artifact, field):
    del artifact[2][field]
    with pytest.raises(ValueError, match=field):
        _load_artifact(artifact)


@pytest.mark.parametrize("scale", [0, -0.2, float("nan"), float("inf"), -float("inf"), True, "0.2", None])
def test_loader_rejects_invalid_scale(artifact, scale):
    artifact[2]["selected"]["likelihood_scale"] = scale
    with pytest.raises(ValueError, match="likelihood_scale"):
        _load_artifact(artifact)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.5, None])
def test_loader_rejects_invalid_selection_score(artifact, score):
    artifact[2]["selected"]["decision_mode_nll"] = score
    with pytest.raises(ValueError, match="decision_mode_nll"):
        _load_artifact(artifact)


def test_loader_hashes_checkpoint_bytes_not_just_filename(artifact):
    artifact[1].write_bytes(b"different checkpoint at the same path")
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        _load_artifact(artifact)


@pytest.mark.parametrize("overrides", [{"execute_steps": 6}, {"dt": 0.25}, {"num_robots": 1}])
def test_validator_checks_consumer_runtime_contract(artifact, overrides):
    expected = {"execute_steps": 3, "dt": 0.4, "num_robots": 3, **overrides}
    with pytest.raises(ValueError, match="mismatch"):
        validate_belief_calibration(artifact[2], artifact[1], **expected)


def test_loader_rejects_non_object_json(artifact):
    path, checkpoint, _ = artifact
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_belief_calibration(path, checkpoint, execute_steps=3, dt=0.4, num_robots=3)


def test_scores_preserve_causal_calls_and_full_multiclass_brier():
    likelihoods = torch.tensor([[[0.0, -2.0, -4.0], [-9.0, 0.0, 0.0], [0.0, -1.0, -1.0]]])
    active = torch.tensor([[True, False, True]])
    modes = torch.tensor([0])
    scales = [0.05, 0.5]
    original = calibration_scores(likelihoods, modes, active, scales)
    altered = likelihoods.clone()
    altered[:, 1:] = torch.tensor([100.0, -100.0, 0.0])
    changed = calibration_scores(altered, modes, active, scales)
    for first, second, scale in zip(original, changed, scales):
        assert first["decision_mode_nll"] == second["decision_mode_nll"]
        assert first["decisions"] == 1
        probability = (likelihoods[0, 0] * scale).softmax(-1)
        expected_brier = (probability - torch.tensor([1.0, 0.0, 0.0])).square().sum()
        assert first["decision_mode_brier"] == pytest.approx(float(expected_brier))


class EvidenceSpy:
    def __init__(self):
        self.config = SimpleNamespace(evidence_conditioning=True)
        self.received = None

    def __call__(self, *args, **kwargs):
        self.received = kwargs
        return {"mean": torch.zeros(1), "log_std": torch.zeros(1)}


@pytest.mark.parametrize("missing", ["evidence_candidates", "evidence_responses", "evidence_mask"])
def test_evidence_conditioned_model_rejects_each_missing_input(missing):
    model = EvidenceSpy()
    batch = {name: torch.zeros(1) for name in ("evidence_candidates", "evidence_responses", "evidence_mask")}
    del batch[missing]
    with pytest.raises(ValueError, match=missing):
        _calibration_prediction(model, batch)
    assert model.received is None


def test_evidence_is_forwarded_without_silent_omission():
    model = EvidenceSpy()
    batch = {
        name: torch.zeros(1)
        for name in (
            "entity_history", "robot_current", "robot_candidates",
            "evidence_candidates", "evidence_responses", "evidence_mask",
        )
    }
    _calibration_prediction(model, batch)
    for field in ("evidence_candidates", "evidence_responses", "evidence_mask"):
        assert model.received[field] is batch[field]


@pytest.fixture
def tiny_checkpoint(tmp_path):
    config = PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=4, num_robots=3,
        hidden_dim=16, num_heads=2, graph_layers=1, dt=0.4, dropout=0.0,
    )
    checkpoint_path = tmp_path / "response.pt"
    torch.manual_seed(17)
    model = BeliefConditionedTeamResponseModel(config)
    torch.save(
        {"model_config": vars(config), "model": model.state_dict(), "seed": 17, "data_seed": 418731, "id_split": "test"},
        checkpoint_path,
    )
    raw = {
        "evaluation": {"phase": "validation", "split": "val", "selection_metric": "decision_mode_nll"},
        "dataset": {
            "val_episodes": 2, "windows": 5, "history": 4, "horizon": 4,
            "num_robots": 3, "num_entities": 4, "active_entities": 2,
            "shared_team_actions": True, "dt": 0.4, "execute_steps": 3,
            "data_seed": 418731, "response_noise": 0.18,
        },
        "training": {"batch_size": 2, "num_threads": 1},
        "calibration": {"scales": [0.05, 0.5]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return checkpoint_path, config_path, raw


def test_calibration_uses_only_val_and_emits_matching_contract(tiny_checkpoint, tmp_path, monkeypatch):
    checkpoint_path, config_path, raw = tiny_checkpoint
    original_hash = sha256_file(checkpoint_path)
    calls = []
    make_dataset = calibration_module._make_dataset

    def val_only(config, split, seed):
        calls.append((split, seed))
        assert split == "val"
        assert "test_episodes" not in config["dataset"]
        assert "ood_episodes" not in config["dataset"]
        return make_dataset(config, split, seed)

    monkeypatch.setattr(calibration_module, "_make_dataset", val_only)
    output = tmp_path / "selected.json"
    result = calibrate_checkpoint(config_path, checkpoint_path, output)
    loaded = load_belief_calibration(output, checkpoint_path, execute_steps=3, dt=0.4, num_robots=3)
    assert calls == [("val", raw["dataset"]["data_seed"])]
    assert result == loaded
    assert result["episodes"] == 2
    assert result["tracks"] == 8
    assert result["active_observations"] == 20
    assert result["selected"]["decisions"] == 16
    assert result["selected"] == min(result["records"], key=lambda row: (row["decision_mode_nll"], row["likelihood_scale"]))
    assert sha256_file(checkpoint_path) == original_hash
    with pytest.raises(ValueError, match="output exists"):
        calibrate_checkpoint(config_path, checkpoint_path, output)


@pytest.mark.parametrize("field,value", [("phase", "test"), ("split", "test"), ("selection_metric", "posterior_mode_nll")])
def test_script_rejects_non_validation_selection(tiny_checkpoint, tmp_path, field, value):
    checkpoint, config_path, raw = tiny_checkpoint
    raw["evaluation"][field] = value
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        calibrate_checkpoint(config_path, checkpoint, tmp_path / "invalid.json")


def test_script_refuses_evidence_checkpoint_with_plain_h2_dataset(tiny_checkpoint, tmp_path):
    checkpoint, config_path, _ = tiny_checkpoint
    payload = torch.load(checkpoint, weights_only=True)
    payload["model_config"]["evidence_conditioning"] = True
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(**payload["model_config"]))
    payload["model"] = model.state_dict()
    torch.save(payload, checkpoint)
    output = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="evidence_conditioning=True requires"):
        calibrate_checkpoint(config_path, checkpoint, output)
    assert not output.exists()
