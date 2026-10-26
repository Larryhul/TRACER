from __future__ import annotations

import numpy as np
import pytest
import torch

from sage.data.external_response import (
    assign_response_modes,
    build_sequence_episodes,
    fit_response_modes,
    stable_entity_id,
)
from sage.response.external_checkpoint_adapter import ExternalResponseCheckpointAdapter
from sage.response.models import ResponseModelConfig, TeamResponsePredictor


def _positions(time: int = 72, entities: int = 3) -> np.ndarray:
    clock = np.arange(time, dtype=np.float32)[:, None]
    robot = np.concatenate([0.08 * clock, 0.02 * clock], axis=1)
    tracks = [
        np.concatenate([1.0 + 0.02 * clock, 0.4 + 0.04 * clock], axis=1),
        np.concatenate([-1.0 + 0.03 * clock, 0.6 - 0.02 * clock], axis=1),
        np.concatenate([0.4 + 0.01 * clock, -1.0 + 0.03 * clock], axis=1),
    ]
    return np.stack([robot, *tracks[:entities]], axis=1)


def test_build_external_response_episodes_preserves_tracks_and_windows() -> None:
    result = build_sequence_episodes(
        "sequence-a",
        _positions(),
        dt=0.1,
        history=4,
        horizon=4,
        windows=3,
        execute_steps=2,
        max_entities=2,
        min_entities=2,
        episode_stride=8,
        interaction_radius=4.0,
    )
    assert result is not None
    assert result["entity_history"].shape[1:] == (2, 3, 4, 2)
    assert result["robot_current"].shape[1:] == (2, 3, 1, 4)
    assert result["robot_candidates"].shape[1:] == (2, 3, 1, 4, 4)
    assert result["target_future_vel"].shape[1:] == (2, 3, 4, 2)
    assert result["active_mask"].all()
    assert len(np.unique(result["entity_id"])) == 2
    assert len(np.unique(result["episode_id"])) == len(result["episode_id"])


def test_build_requires_two_entities_for_identity_comparison() -> None:
    result = build_sequence_episodes(
        "sequence-a",
        _positions(entities=1),
        dt=0.1,
        history=4,
        horizon=4,
        windows=3,
        execute_steps=2,
        max_entities=2,
        min_entities=2,
        episode_stride=8,
        interaction_radius=4.0,
    )
    assert result is None


def test_mode_assignment_uses_train_fit_clustering() -> None:
    features = np.asarray(
        [
            [[0.1, -0.1, 0.0, 0.5, 1.0]],
            [[0.2, -0.2, 0.0, 0.6, 1.1]],
            [[1.0, 0.8, 0.5, 1.5, 3.0]],
            [[1.1, 0.9, 0.4, 1.6, 3.1]],
            [[2.0, 1.8, 1.0, 2.5, 5.0]],
            [[2.1, 1.9, 1.1, 2.6, 5.1]],
        ],
        dtype=np.float32,
    )
    ids = np.asarray([[stable_entity_id(f"s-{i}", 0)] for i in range(6)])
    active = np.ones((6, 1, 1), dtype=bool)
    clustering = fit_response_modes(features, ids, active, num_modes=3, seed=7)
    modes = assign_response_modes(features, ids, active, clustering)
    assert modes.shape == (6, 1)
    assert len(np.unique(modes)) == 3


def test_mode_assignment_is_stable_for_overlapping_track_windows() -> None:
    features = np.asarray(
        [
            [[0.1, -0.2, 0.0, 0.5, 1.0]],
            [[2.0, 1.8, 1.0, 2.5, 5.0]],
            [[1.8, 1.5, 0.8, 2.2, 4.5]],
            [[0.2, -0.1, 0.1, 0.6, 1.2]],
        ],
        dtype=np.float32,
    )
    shared_id = stable_entity_id("shared", 1)
    ids = np.asarray(
        [
            [shared_id],
            [shared_id],
            [stable_entity_id("other", 1)],
            [stable_entity_id("third", 1)],
        ]
    )
    active = np.ones((4, 1, 1), dtype=bool)
    clustering = fit_response_modes(features, ids, active, num_modes=3, seed=5)
    modes = assign_response_modes(features, ids, active, clustering)
    assert modes[0, 0] == modes[1, 0]


def test_invalid_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="execute_steps"):
        build_sequence_episodes(
            "sequence-a",
            _positions(),
            dt=0.1,
            history=4,
            horizon=4,
            windows=3,
            execute_steps=5,
            max_entities=2,
            min_entities=2,
            episode_stride=8,
            interaction_radius=4.0,
        )


def test_external_checkpoint_adapter_exposes_closed_loop_single_mode(tmp_path) -> None:
    config = ResponseModelConfig(
        history=4,
        horizon=3,
        num_robots=1,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        architecture="tracer",
        graph_layers=1,
        dt=0.4,
    )
    predictor = TeamResponsePredictor("joint", config)
    checkpoint = tmp_path / "joint_seed7.pt"
    torch.save(
        {
            "model": predictor.state_dict(),
            "model_config": vars(config),
            "variant": "joint",
        },
        checkpoint,
    )
    adapter = ExternalResponseCheckpointAdapter.from_checkpoint(
        checkpoint, "cpu", expected_dt=0.25
    )
    output = adapter(
        torch.zeros(2, 1, 4, 2),
        torch.zeros(2, 1, 1, 4),
        torch.zeros(2, 1, 1, 5, 4),
        robot_mask=torch.ones(2, 1, 1, dtype=torch.bool),
        neighbor_history=torch.zeros(2, 1, 2, 4, 2),
        neighbor_mask=torch.ones(2, 1, 2, dtype=torch.bool),
    )
    assert adapter.config.num_modes == 2
    assert adapter.config.dt == 0.25
    assert output["mean"].shape == (2, 1, 2, 3, 2)
    assert output["log_std"].shape == output["mean"].shape
    torch.testing.assert_close(output["mean"][:, :, 0], output["mean"][:, :, 1])


def test_external_checkpoint_preserves_legacy_capacity_width(tmp_path):
    config = ResponseModelConfig(history=4, horizon=3, num_robots=1, hidden_dim=8,
                                 num_heads=2, dropout=0.0, architecture="tracer", graph_layers=1)
    predictor = TeamResponsePredictor("joint", config)
    predictor.tracer_capacity_decoder = torch.nn.Sequential(
        torch.nn.Linear(16, 16), torch.nn.GELU(), torch.nn.Dropout(0.0), torch.nn.Linear(16, 6),
    )
    path = tmp_path / "legacy.pt"
    torch.save({"model": predictor.state_dict(), "model_config": vars(config), "variant": "joint"}, path)
    with pytest.warns(RuntimeWarning, match="legacy capacity"):
        adapter = ExternalResponseCheckpointAdapter.from_checkpoint(path, "cpu", expected_dt=.4)
    assert adapter.restored_legacy_capacity_width == 16
    for name, parameter in predictor.state_dict().items():
        torch.testing.assert_close(parameter, adapter.predictor.state_dict()[name])


def test_external_response_delay_removes_impossible_first_step_action_effect():
    config = ResponseModelConfig(history=4, horizon=3, num_robots=1, hidden_dim=8,
                                 num_heads=2, dropout=0.0, architecture="tracer", graph_layers=1)
    adapter = ExternalResponseCheckpointAdapter(TeamResponsePredictor("joint", config), .4).eval()
    adapter.response_delay_steps = 1
    history = torch.zeros(2, 1, 4, 2)
    current = torch.zeros(2, 1, 1, 4)
    candidates = torch.randn(2, 1, 1, 3, 4)
    output = adapter(history, current, candidates)
    torch.testing.assert_close(output["mean"][0, ..., 0, :], output["mean"][1, ..., 0, :])
    torch.testing.assert_close(output["log_std"][0, ..., 0, :], output["log_std"][1, ..., 0, :])
