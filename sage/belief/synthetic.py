from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


MODE_NAMES = ("yielding", "neutral", "assertive")
MODE_GAINS = torch.tensor(
    [
        [1.25, 0.95, 0.75],
        [0.85, 0.50, 0.40],
        [0.45, 0.18, 0.15],
    ],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class BeliefSequenceSpec:
    split: str = "train"
    num_episodes: int = 2048
    windows: int = 10
    seed: int = 2027
    response_noise: float = 0.18
    num_modes: int = 3
    action_policy: str = "iid"


def mode_response_means(actions: torch.Tensor) -> torch.Tensor:
    """Return mode-conditioned response means (..., windows, modes, 2).

    Each action contains robot-0 activity, robot-1 activity, opposing geometry,
    and a signed passing side. The two responses are slowdown and lateral shift.
    """

    if actions.shape[-1] != 4:
        raise ValueError(f"actions must end in four features, got {tuple(actions.shape)}")
    gains = MODE_GAINS.to(device=actions.device, dtype=actions.dtype)
    activity_0, activity_1, opposition, side = actions.unbind(dim=-1)
    main = activity_0 + activity_1
    pair = activity_0 * activity_1 * opposition
    asymmetry = activity_0 - activity_1

    main_gain = gains[:, 0]
    joint_gain = gains[:, 1]
    lateral_gain = gains[:, 2]
    slowdown = main[..., None] * main_gain + pair[..., None] * joint_gain
    lateral = lateral_gain * (
        0.30 * asymmetry[..., None] + side[..., None] * pair[..., None]
    )
    return torch.stack([slowdown, lateral], dim=-1)


def mode_log_likelihood(
    actions: torch.Tensor,
    responses: torch.Tensor,
    response_noise: float,
) -> torch.Tensor:
    """Evaluate p(response | action, mode) as a diagonal Gaussian."""

    if response_noise <= 0.0:
        raise ValueError("response_noise must be positive")
    if responses.shape != (*actions.shape[:-1], 2):
        raise ValueError(
            f"responses must have shape {(*actions.shape[:-1], 2)}, got {tuple(responses.shape)}"
        )
    means = mode_response_means(actions)
    residual = responses[..., None, :] - means
    variance = float(response_noise) ** 2
    return -0.5 * (
        residual.square().sum(dim=-1) / variance
        + 2.0 * math.log(2.0 * math.pi * variance)
    )


class PersistentBeliefSequenceDataset(Dataset):
    """Stable identity/mode episodes for validating cross-window belief updates."""

    def __init__(self, spec: BeliefSequenceSpec) -> None:
        if spec.num_modes != len(MODE_NAMES):
            raise ValueError(f"this diagnostic defines exactly {len(MODE_NAMES)} modes")
        if spec.num_episodes < 1 or spec.windows < 2:
            raise ValueError("num_episodes must be positive and windows must be at least two")
        if spec.response_noise <= 0.0:
            raise ValueError("response_noise must be positive")
        if spec.action_policy not in {"iid", "correlated"}:
            raise ValueError("action_policy must be 'iid' or 'correlated'")
        self.spec = spec
        split_offsets = {"train": 0, "val": 1_000_000, "test": 2_000_000}
        self.split_offset = split_offsets.get(spec.split.lower(), 3_000_000)
        self.seed = int(spec.seed) + self.split_offset

    def __len__(self) -> int:
        return int(self.spec.num_episodes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        rng = np.random.default_rng(self.seed + int(index) * 104729)
        mode = int(rng.integers(0, self.spec.num_modes))
        if self.spec.action_policy == "iid":
            activity = rng.uniform(0.15, 1.0, size=(self.spec.windows, 2)).astype(np.float32)
            opposition = rng.uniform(0.20, 1.0, size=(self.spec.windows, 1)).astype(np.float32)
        else:
            common = rng.uniform(0.15, 1.0, size=(self.spec.windows, 1))
            delta = rng.normal(0.0, 0.04, size=(self.spec.windows, 1))
            activity = np.concatenate([common + delta, common - delta], axis=-1)
            activity = np.clip(activity, 0.15, 1.0).astype(np.float32)
            opposition = rng.uniform(0.65, 1.0, size=(self.spec.windows, 1)).astype(np.float32)
        side = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(self.spec.windows, 1))
        actions = torch.from_numpy(np.concatenate([activity, opposition, side], axis=-1))
        means = mode_response_means(actions)
        noise = torch.from_numpy(
            rng.normal(0.0, self.spec.response_noise, size=(self.spec.windows, 2)).astype(np.float32)
        )
        responses = means[:, mode] + noise
        return {
            "entity_id": self.split_offset + int(index),
            "mode": torch.tensor(mode, dtype=torch.long),
            "actions": actions,
            "responses": responses,
            "action_policy": self.spec.action_policy,
        }
