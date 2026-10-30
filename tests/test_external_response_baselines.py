from __future__ import annotations

import torch
import yaml
from pathlib import Path

from tracer.response.data import EntityResponseBatch
from tracer.response.external_baselines import METHOD_TAXONOMY, external_method_metadata
from tracer.response.models import (
    Eiffert2019StyleResponsePredictor,
    KediaStyleResponsePredictor,
    ResponseModelConfig,
)


def _batch(padded_value: float = 0.0) -> EntityResponseBatch:
    return EntityResponseBatch(
        entity_history=torch.zeros(2, 4, 2),
        entity_current=torch.zeros(2, 2),
        robot_current=torch.tensor([[[1.0, 0.0, 0.1, 0.0], [2.0, 0.0, 0.2, 0.0]], [[1.0, 0.0, 0.1, 0.0], [2.0, 0.0, 0.2, 0.0]]]),
        robot_candidates=torch.tensor(
            [
                [[[1.0, 0.0, 0.1, 0.0], [1.1, 0.0, 0.1, 0.0]], [[2.0, 0.0, 0.2, 0.0], [2.1, 0.0, 0.2, 0.0]]],
                [[[1.0, 0.0, 0.1, 0.0], [1.1, 0.0, 0.1, 0.0]], [[padded_value, padded_value, padded_value, padded_value], [padded_value, padded_value, padded_value, padded_value]]],
            ], dtype=torch.float32,
        ),
        robot_mask=torch.tensor([[True, True], [True, False]]),
        target_future_vel=torch.zeros(2, 2, 2),
        high_interaction=torch.zeros(2, dtype=torch.bool),
        scene_index=torch.arange(2),
        entity_index=torch.zeros(2, dtype=torch.long),
        dt=0.4,
    )


def test_eiffert_style_uses_shared_gaussian_response_contract_and_mask() -> None:
    torch.manual_seed(7)
    model = Eiffert2019StyleResponsePredictor(
        ResponseModelConfig(history=4, horizon=2, num_robots=2, hidden_dim=8, dropout=0.0)
    ).eval()
    first = model(_batch(0.0))
    second = model(_batch(999.0))
    assert first["mean"].shape == (2, 2, 2)
    assert first["log_std"].shape == (2, 2, 2)
    assert torch.allclose(first["mean"][1], second["mean"][1])
    assert torch.isfinite(first["mean"]).all()
    assert torch.isfinite(first["log_std"]).all()


def test_external_taxonomy_marks_method_level_claims() -> None:
    assert "eiffert2019-style" in METHOD_TAXONOMY
    assert METHOD_TAXONOMY["ScePT-style"]["implementation"] == "team-concat"
    model = Eiffert2019StyleResponsePredictor(ResponseModelConfig(hidden_dim=4, horizon=2))
    metadata = external_method_metadata(model)
    assert metadata["method"] == "eiffert2019-style"
    assert metadata["type"].startswith("external action-conditioned")
    assert metadata["parameter_count"] > 0


def test_kedia_style_uses_masked_latent_hypotheses_and_gaussian_contract() -> None:
    torch.manual_seed(11)
    model = KediaStyleResponsePredictor(
        ResponseModelConfig(history=4, horizon=2, num_robots=2, hidden_dim=8, dropout=0.0)
    ).eval()
    first = model(_batch(0.0))
    second = model(_batch(999.0))
    assert first["mean"].shape == (2, 2, 2)
    assert first["log_std"].shape == (2, 2, 2)
    assert first["hypothesis_weights"].shape == (2, 4)
    assert torch.allclose(first["mean"][1], second["mean"][1])
    assert torch.allclose(first["hypothesis_weights"][1], second["hypothesis_weights"][1])
    assert torch.isfinite(first["mean"]).all()
    assert torch.isfinite(first["log_std"]).all()


def test_external_protocol_contains_required_methods_and_validation_only_selection() -> None:
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/h1_external_response_baselines.yaml").read_text(encoding="utf-8")
    )
    variants = config["training"]["variants"]
    assert "eiffert2019-style" in variants
    assert "team-concat" in variants
    assert config["training"]["checkpoint_rule"] == "best_val_nll"
    assert config["protocol"]["test_selection"] == "validation-only"
