"""Track-continuous external episodes for response and belief evaluation.

The public recordings contain one executed robot trajectory rather than a
resettable set of multi-robot counterfactual candidates.  This module keeps
that limitation explicit while preserving the pieces needed by the current
paper: an executed action, an observed entity response, a recording-local
entity identity, and multiple receding-horizon windows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_FIELDS = (
    "entity_history",
    "robot_current",
    "robot_candidates",
    "target_future_vel",
    "active_mask",
    "high_interaction",
    "episode_id",
    "entity_id",
    "mode_features",
)


def stable_entity_id(sequence_id: str, track_index: int) -> int:
    """Return a deterministic positive int64 ID for one recording-local track."""

    digest = hashlib.sha256(f"{sequence_id}:{track_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def response_mode_features(
    robot_positions: np.ndarray,
    entity_positions: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Describe one short track without assigning a psychological meaning."""

    robot_velocity = np.diff(robot_positions, axis=0) / float(dt)
    entity_velocity = np.diff(entity_positions, axis=0) / float(dt)
    relative_position = entity_positions[1:] - robot_positions[1:]
    distance = np.linalg.norm(relative_position, axis=-1)
    unit = relative_position / np.maximum(distance[:, None], 1e-6)
    relative_velocity = entity_velocity - robot_velocity
    radial = np.sum(relative_velocity * unit, axis=-1)
    tangent = np.stack([-unit[:, 1], unit[:, 0]], axis=-1)
    tangential = np.abs(np.sum(relative_velocity * tangent, axis=-1))
    return np.asarray(
        [
            np.linalg.norm(entity_velocity, axis=-1).mean(),
            radial.mean(),
            tangential.mean(),
            np.linalg.norm(robot_velocity, axis=-1).mean(),
            distance.mean(),
        ],
        dtype=np.float32,
    )


def build_sequence_episodes(
    sequence_id: str,
    positions: np.ndarray,
    *,
    dt: float,
    history: int,
    horizon: int,
    windows: int,
    execute_steps: int,
    max_entities: int,
    min_entities: int,
    episode_stride: int,
    interaction_radius: float,
    episode_offset: int = 0,
) -> dict[str, np.ndarray] | None:
    """Build fixed-size multi-entity episodes from one recorded sequence."""

    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (time, agents, 2)")
    if positions.shape[1] < min_entities + 1:
        return None
    if not 1 <= min_entities <= max_entities:
        raise ValueError("min_entities must lie in [1, max_entities]")
    if not 1 <= execute_steps <= horizon:
        raise ValueError("execute_steps must lie in [1, horizon]")

    span = history + (windows - 1) * execute_steps + horizon
    starts = range(0, len(positions) - span + 1, max(1, episode_stride))
    records: dict[str, list[np.ndarray | int]] = {
        key: [] for key in REQUIRED_FIELDS
    }
    for start in starts:
        segment = positions[start : start + span]
        if not np.isfinite(segment[:, 0]).all():
            continue
        available: list[tuple[float, int]] = []
        anchors = history - 1 + np.arange(windows) * execute_steps
        for track in range(1, positions.shape[1]):
            if not np.isfinite(segment[:, track]).all():
                continue
            distances = np.linalg.norm(
                segment[anchors, track] - segment[anchors, 0], axis=-1
            )
            available.append((float(distances.mean()), track))
        if len(available) < min_entities:
            continue
        available.sort(key=lambda item: (item[0], item[1]))
        selected = [track for _, track in available[:max_entities]]
        entities = max_entities

        entity_history = np.zeros(
            (entities, windows, history, 2), dtype=np.float32
        )
        robot_current = np.zeros(
            (entities, windows, 1, 4), dtype=np.float32
        )
        robot_candidates = np.zeros(
            (entities, windows, 1, horizon, 4), dtype=np.float32
        )
        target_future_vel = np.zeros(
            (entities, windows, horizon, 2), dtype=np.float32
        )
        active_mask = np.zeros((entities, windows), dtype=np.bool_)
        high_interaction = np.zeros((entities, windows), dtype=np.bool_)
        entity_ids = np.zeros((entities,), dtype=np.int64)
        mode_features = np.zeros((entities, 5), dtype=np.float32)

        for slot, track in enumerate(selected):
            active_mask[slot] = True
            entity_ids[slot] = stable_entity_id(sequence_id, track)
            mode_features[slot] = response_mode_features(
                segment[:, 0], segment[:, track], dt
            )
            for window, anchor in enumerate(anchors):
                entity_current = segment[anchor, track]
                robot_now = segment[anchor, 0]
                robot_previous = segment[anchor - 1, 0]
                robot_future = segment[anchor + 1 : anchor + 1 + horizon, 0]
                entity_future = segment[anchor + 1 : anchor + 1 + horizon, track]
                robot_velocity = np.diff(
                    np.concatenate([robot_now[None], robot_future], axis=0), axis=0
                ) / float(dt)
                entity_velocity = np.diff(
                    np.concatenate([entity_current[None], entity_future], axis=0),
                    axis=0,
                ) / float(dt)
                entity_history[slot, window] = segment[
                    anchor - history + 1 : anchor + 1, track
                ]
                robot_current[slot, window, 0, :2] = robot_now - entity_current
                robot_current[slot, window, 0, 2:] = (
                    robot_now - robot_previous
                ) / float(dt)
                robot_candidates[slot, window, 0, :, :2] = (
                    robot_future - entity_current
                )
                robot_candidates[slot, window, 0, :, 2:] = robot_velocity
                target_future_vel[slot, window] = entity_velocity
                high_interaction[slot, window] = bool(
                    np.linalg.norm(robot_now - entity_current) < interaction_radius
                )

        records["entity_history"].append(entity_history)
        records["robot_current"].append(robot_current)
        records["robot_candidates"].append(robot_candidates)
        records["target_future_vel"].append(target_future_vel)
        records["active_mask"].append(active_mask)
        records["high_interaction"].append(high_interaction)
        records["episode_id"].append(episode_offset + len(records["episode_id"]))
        records["entity_id"].append(entity_ids)
        records["mode_features"].append(mode_features)

    if not records["episode_id"]:
        return None
    return {
        key: np.asarray(value, dtype=np.int64)
        if key == "episode_id"
        else np.stack(value)
        for key, value in records.items()
    }


def concatenate_episode_records(
    records: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("no external response episodes were generated")
    return {
        key: np.concatenate([record[key] for record in records], axis=0)
        for key in REQUIRED_FIELDS
    }


def fit_response_modes(
    features: np.ndarray,
    entity_ids: np.ndarray,
    active_mask: np.ndarray,
    *,
    num_modes: int,
    seed: int,
    iterations: int = 100,
) -> dict[str, np.ndarray]:
    """Fit deterministic train-only k-means on unique active track descriptors."""

    active_entities = active_mask.any(axis=-1)
    flat_features = features[active_entities]
    flat_ids = entity_ids[active_entities]
    if len(flat_features) < num_modes:
        raise ValueError("not enough active tracks to fit response modes")
    unique_ids = np.unique(flat_ids)
    samples = np.stack(
        [flat_features[flat_ids == entity_id].mean(axis=0) for entity_id in unique_ids]
    )
    mean = samples.mean(axis=0)
    scale = samples.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    normalized = (samples - mean) / scale

    rng = np.random.default_rng(seed)
    centers = normalized[rng.choice(len(normalized), size=num_modes, replace=False)]
    labels = np.zeros(len(normalized), dtype=np.int64)
    for _ in range(iterations):
        distances = ((normalized[:, None] - centers[None]) ** 2).sum(axis=-1)
        updated_labels = distances.argmin(axis=1)
        updated_centers = centers.copy()
        for mode in range(num_modes):
            selected = normalized[updated_labels == mode]
            if len(selected):
                updated_centers[mode] = selected.mean(axis=0)
        if np.array_equal(updated_labels, labels) and np.allclose(
            updated_centers, centers
        ):
            centers = updated_centers
            break
        labels = updated_labels
        centers = updated_centers

    # Stable labels: low radial response first, then speed and tangential motion.
    center_order = np.lexsort((centers[:, 2], centers[:, 0], centers[:, 1]))
    centers = centers[center_order]
    return {"mean": mean, "scale": scale, "centers": centers}


def assign_response_modes(
    features: np.ndarray,
    entity_ids: np.ndarray,
    active_mask: np.ndarray,
    clustering: dict[str, np.ndarray],
) -> np.ndarray:
    """Assign one stable pseudo-mode to every recording-local track."""

    active_entities = active_mask.any(axis=-1)
    if (
        features.shape[:-1] != entity_ids.shape
        or entity_ids.shape != active_entities.shape
    ):
        raise ValueError("features, entity_ids, and active_mask are not aligned")
    modes = np.zeros(entity_ids.shape, dtype=np.int64)
    for entity_id in np.unique(entity_ids[active_entities]):
        occurrences = active_entities & (entity_ids == entity_id)
        descriptor = features[occurrences].mean(axis=0)
        normalized = (descriptor - clustering["mean"]) / clustering["scale"]
        distances = ((normalized[None] - clustering["centers"]) ** 2).sum(axis=-1)
        modes[occurrences] = int(distances.argmin())
    return modes


class ExternalResponseEpisodeDataset(Dataset):
    """Load cached external episodes with train-fitted pseudo-mode labels."""

    def __init__(self, path: str | Path, modes: np.ndarray) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_FIELDS if key not in data]
            if missing:
                raise ValueError(f"{self.path} is missing fields: {missing}")
            for key in REQUIRED_FIELDS:
                setattr(self, key, np.asarray(data[key]))
        expected = tuple(self.entity_id.shape)
        if tuple(modes.shape) != expected:
            raise ValueError(f"modes must have shape {expected}, got {modes.shape}")
        self.modes = np.asarray(modes, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.episode_id))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "entity_history": torch.from_numpy(
                self.entity_history[index].astype(np.float32, copy=False)
            ),
            "robot_current": torch.from_numpy(
                self.robot_current[index].astype(np.float32, copy=False)
            ),
            "robot_candidates": torch.from_numpy(
                self.robot_candidates[index].astype(np.float32, copy=False)
            ),
            "target_future_vel": torch.from_numpy(
                self.target_future_vel[index].astype(np.float32, copy=False)
            ),
            "active_mask": torch.from_numpy(
                self.active_mask[index].astype(np.bool_, copy=False)
            ),
            "presence_mask": torch.from_numpy(
                self.active_mask[index].astype(np.bool_, copy=False)
            ),
            "high_interaction": torch.from_numpy(
                self.high_interaction[index].astype(np.bool_, copy=False)
            ),
            "episode_id": torch.tensor(int(self.episode_id[index])),
            "entity_id": torch.from_numpy(
                self.entity_id[index].astype(np.int64, copy=False)
            ),
            "mode": torch.from_numpy(self.modes[index]),
        }
