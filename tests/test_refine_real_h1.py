import argparse
import json

import numpy as np
import pytest
import torch

from sage.experiments import refine_real_h1
from sage.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    TrajectoryRecurrentBelief,
    TrajectoryRecurrentConfig,
)


@pytest.mark.parametrize("evaluate_only", [False, True])
def test_refinement_preserves_pairing_and_accesses_only_requested_splits(tmp_path, monkeypatch, evaluate_only):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    hashes = {}
    for split in ("train", "val"):
        path = data / f"navware_{split}.npz"
        path.write_bytes(split.encode())
        hashes[split] = refine_real_h1._sha256(path)
    config = PersistentTrajectoryModelConfig(hidden_dim=8, num_heads=2, graph_layers=1)
    recurrent_config = TrajectoryRecurrentConfig(hidden_dim=8)
    model = BeliefConditionedTeamResponseModel(config)
    recurrent = TrajectoryRecurrentBelief(recurrent_config)
    for name in ("response.pt", "history_only_response.pt"):
        torch.save({"model_config": vars(config), "model": model.state_dict(), "velocity_residual": True}, baseline / name)
    torch.save({"model_config": vars(recurrent_config), "model": recurrent.state_dict()}, baseline / "recurrent.pt")
    metadata = {
        "dataset": "navware", "data_dir": str(data), "model_seed": 2027,
        "input_sha256": hashes, "velocity_residual": True,
        "history": config.history, "horizon": config.horizon, "windows": 8, "execute_steps": 3,
        "training_arguments": {
            "weight_decay": .0001, "mode_discrimination_weight": .1,
            "mode_temperature": .2, "prefix_loss_weight": 1.,
            "likelihood_scale": .05, "confidence_threshold": .8,
        },
    }
    (baseline / "run_metadata.json").write_text(json.dumps(metadata))
    (baseline / "mode_clustering.json").write_text("{}")
    loaded = []
    trained = []

    def load_arrays(path):
        loaded.append(path.stem)
        return {name: np.zeros(1) for name in ("mode_features", "entity_id", "active_mask")}

    def train(*arguments, **keywords):
        trained.append((arguments[3:9], keywords))

    monkeypatch.setattr(refine_real_h1, "_load_arrays", load_arrays)
    monkeypatch.setattr(refine_real_h1, "_validate_protocol_arrays", lambda *arguments, **keywords: None)
    monkeypatch.setattr(refine_real_h1, "assign_response_modes", lambda *arguments: np.zeros(1))
    monkeypatch.setattr(refine_real_h1, "ExternalResponseEpisodeDataset", lambda *arguments: [0])
    monkeypatch.setattr(refine_real_h1, "_train_response", train)
    monkeypatch.setattr(refine_real_h1, "_evaluate", lambda *arguments, **keywords: ({"no-belief": {}, "oracle": {}}, []))
    args = argparse.Namespace(
        baseline=str(baseline), data_dir=None, output_dir=str(tmp_path / "out"),
        split="val", evaluate_only=evaluate_only, device="cpu", num_threads=1,
        epochs=5, lr=.0001, batch_size=2, mean_trajectory_weight=10., mean_endpoint_weight=.25,
    )
    refine_real_h1.run(args)
    assert loaded == (["navware_val"] if evaluate_only else ["navware_train", "navware_val"])
    if evaluate_only:
        assert not trained
        for name in ("response.pt", "history_only_response.pt"):
            assert (tmp_path / "out" / name).read_bytes() == (baseline / name).read_bytes()
    else:
        assert len(trained) == 2
        assert trained[0] == trained[1]
        assert trained[0][1]["mean_trajectory_weight"] == 10.
    result = json.loads((tmp_path / "out" / "val_metrics.json").read_text())
    assert "history-only" in result
    with pytest.raises(FileExistsError):
        refine_real_h1.run(args)
