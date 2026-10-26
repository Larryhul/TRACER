"""Unified trajectory samples for external SAGE portability experiments.

The external releases do not share a file format.  The converter writes a
compact, split-specific NPZ file and this module exposes it through the same
sample contract as the MID adapter.  A sample contains positions in history
and future velocities as required by :class:`HeterogeneousSceneBatch`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from sage.data.heterogeneous_scene import HeterogeneousSceneBatch


class UnifiedTrajectoryDataset(Dataset):
    """Load pre-built external trajectory windows without format-specific logic."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with np.load(self.path, allow_pickle=False) as data:
            required = (
                "history",
                "future",
                "current_pos",
                "goals",
                "goal_mask",
                "roles",
                "agent_mask",
                "robot_mask",
                "entity_mask",
                "loss_mask",
            )
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{self.path} is missing fields: {missing}")
            self.history = np.asarray(data["history"], dtype=np.float32)
            self.future = np.asarray(data["future"], dtype=np.float32)
            self.current_pos = np.asarray(data["current_pos"], dtype=np.float32)
            self.goals = np.asarray(data["goals"], dtype=np.float32)
            self.goal_mask = np.asarray(data["goal_mask"], dtype=np.bool_)
            self.roles = np.asarray(data["roles"], dtype=np.int64)
            self.agent_mask = np.asarray(data["agent_mask"], dtype=np.bool_)
            self.robot_mask = np.asarray(data["robot_mask"], dtype=np.bool_)
            self.entity_mask = np.asarray(data["entity_mask"], dtype=np.bool_)
            self.loss_mask = np.asarray(data["loss_mask"], dtype=np.bool_)
            dt_value = np.asarray(data["dt"], dtype=np.float32).reshape(-1)
            self.dt = float(dt_value[0]) if len(dt_value) else 0.1
        self._validate()
        manifest_path = self.path.with_suffix(".json")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    def _validate(self) -> None:
        if self.history.ndim != 4 or self.history.shape[-1] < 2:
            raise ValueError(f"history must be (N,A,H,F>=2), got {self.history.shape}")
        if self.future.ndim != 4 or self.future.shape[-1] != 2:
            raise ValueError(f"future must be (N,A,T,2), got {self.future.shape}")
        n, agents = self.history.shape[:2]
        for name, value, shape in (
            ("current_pos", self.current_pos, (n, agents, 2)),
            ("goals", self.goals, (n, agents, 2)),
            ("goal_mask", self.goal_mask, (n, agents)),
            ("roles", self.roles, (n, agents)),
            ("agent_mask", self.agent_mask, (n, agents)),
            ("robot_mask", self.robot_mask, (n, agents)),
            ("entity_mask", self.entity_mask, (n, agents)),
            ("loss_mask", self.loss_mask, (n, agents)),
        ):
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {value.shape}")
        if self.future.shape[2] <= 0:
            raise ValueError("future horizon must be positive")

    def __len__(self) -> int:
        return int(self.history.shape[0])

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "history": torch.from_numpy(self.history[index]),
            "future": torch.from_numpy(self.future[index]),
            "current_pos": torch.from_numpy(self.current_pos[index]),
            "goals": torch.from_numpy(self.goals[index]),
            "goal_mask": torch.from_numpy(self.goal_mask[index]),
            "roles": torch.from_numpy(self.roles[index]),
            "agent_mask": torch.from_numpy(self.agent_mask[index]),
            "robot_mask": torch.from_numpy(self.robot_mask[index]),
            "entity_mask": torch.from_numpy(self.entity_mask[index]),
            "loss_mask": torch.from_numpy(self.loss_mask[index]),
            "dt": self.dt,
        }


def collate_unified_trajectory_batch(samples: Sequence[dict[str, object]]) -> HeterogeneousSceneBatch:
    """Collate samples into the exact batch contract consumed by SAGE/baselines."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    dt = float(samples[0]["dt"])
    batch = HeterogeneousSceneBatch(
        history=torch.stack([sample["history"] for sample in samples]),  # type: ignore[arg-type]
        future=torch.stack([sample["future"] for sample in samples]),  # type: ignore[arg-type]
        current_pos=torch.stack([sample["current_pos"] for sample in samples]),  # type: ignore[arg-type]
        goals=torch.stack([sample["goals"] for sample in samples]),  # type: ignore[arg-type]
        goal_mask=torch.stack([sample["goal_mask"] for sample in samples]),  # type: ignore[arg-type]
        roles=torch.stack([sample["roles"] for sample in samples]),  # type: ignore[arg-type]
        agent_mask=torch.stack([sample["agent_mask"] for sample in samples]),  # type: ignore[arg-type]
        robot_mask=torch.stack([sample["robot_mask"] for sample in samples]),  # type: ignore[arg-type]
        entity_mask=torch.stack([sample["entity_mask"] for sample in samples]),  # type: ignore[arg-type]
        loss_mask=torch.stack([sample["loss_mask"] for sample in samples]),  # type: ignore[arg-type]
        dt=dt,
    )
    batch.validate()
    return batch
