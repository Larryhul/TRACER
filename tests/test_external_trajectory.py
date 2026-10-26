from __future__ import annotations

import json

import numpy as np
import torch

from sage.data.external_trajectory import UnifiedTrajectoryDataset, collate_unified_trajectory_batch
from sage.experiments.build_external_sage_dataset import _make_windows


def test_external_windows_keep_robot_entity_contract() -> None:
    positions = np.zeros((40, 3, 2), dtype=np.float32)
    positions[:, 0, 0] = np.arange(40, dtype=np.float32) * 0.1
    positions[:, 1, 1] = np.arange(40, dtype=np.float32) * 0.05
    positions[:, 2, 0] = 2.0
    roles = np.asarray([0, 1, 3], dtype=np.int64)
    arrays = _make_windows("unit", positions, roles, 0.1, 8, 12, 2, 4, 5.0)
    assert arrays["history"].shape[1:] == (3, 8, 2)
    assert arrays["future"].shape[1:] == (3, 12, 2)
    assert np.all(arrays["robot_mask"][:, 0])
    assert np.all(arrays["entity_mask"][:, 1:3])
    assert np.all(arrays["loss_mask"][:, :3])
    assert np.all(np.isfinite(arrays["future"]))


def test_unified_dataset_round_trip(tmp_path) -> None:
    arrays = {
        "history": np.zeros((2, 3, 8, 2), dtype=np.float32),
        "future": np.zeros((2, 3, 12, 2), dtype=np.float32),
        "current_pos": np.zeros((2, 3, 2), dtype=np.float32),
        "goals": np.zeros((2, 3, 2), dtype=np.float32),
        "goal_mask": np.asarray([[True, False, False]] * 2),
        "roles": np.asarray([[0, 1, 4]] * 2, dtype=np.int64),
        "agent_mask": np.asarray([[True, True, False]] * 2),
        "robot_mask": np.asarray([[True, False, False]] * 2),
        "entity_mask": np.asarray([[False, True, False]] * 2),
        "loss_mask": np.asarray([[True, True, False]] * 2),
        "dt": np.asarray([0.1], dtype=np.float32),
    }
    path = tmp_path / "sample.npz"
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps({"dataset": "unit"}), encoding="utf-8")
    dataset = UnifiedTrajectoryDataset(path)
    batch = collate_unified_trajectory_batch([dataset[0], dataset[1]])
    batch.validate()
    assert len(dataset) == 2
    assert batch.history.shape == (2, 3, 8, 2)
    assert batch.num_robots == 2
    assert batch.num_entities == 2
    assert torch.equal(batch.goal_mask[:, 0], torch.ones(2, dtype=torch.bool))
