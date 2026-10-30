from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from tracer.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig
from tracer.response.static_context import segment_context_features
from tracer.experiments.add_live_static_context import relative_static_segments, validate_rolling_reset


def test_segment_context_is_endpoint_order_invariant():
    segments = torch.tensor([[1., -2., 1., 2.], [2., 0., 3., 1.], [1., 1., 1., 1.]])
    features = segment_context_features(segments)
    reverse = torch.cat([segments[:, 2:], segments[:, :2]], -1)
    torch.testing.assert_close(features, segment_context_features(reverse))
    assert features.shape == (3, 8)
    assert features[0, -1] == 1


def test_static_map_context_preserves_translation_invariance():
    obstacles = np.array([[[1., -2.], [1., 2.]]])
    origins = np.array([[0., 0.], [1., 1.]])
    original = relative_static_segments(obstacles, origins)
    moved = relative_static_segments(obstacles + np.array([3., -2.]), origins + np.array([3., -2.]))
    np.testing.assert_array_equal(original, moved)
    np.testing.assert_array_equal(original[1, 0], [0., -3., 0., 1.])


def test_rolling_map_reset_requires_matching_observed_humans_and_robot():
    env = SimpleNamespace(robot=SimpleNamespace(get_position=lambda: (2., 1.)),
                          humans=[SimpleNamespace(get_position=lambda: (1., 1.))])
    data = {"case": np.array([10]), "human": np.array([0]), "window": np.array([0]),
            "history": np.array([[[1., 1.], [1., 1.]]]), "current": np.array([[[1., 0., 0., 0.]]])}
    validate_rolling_reset(data, 10, env)
    data["current"][0, 0, 0] = 4.
    with pytest.raises(ValueError, match="robot positions"):
        validate_rolling_reset(data, 10, env)
    data["history"][0, -1, 0] = 9.
    with pytest.raises(ValueError, match="human positions"):
        validate_rolling_reset(data, 10, env)
    data["window"][0] = 1
    with pytest.raises(ValueError, match="initial observations"):
        validate_rolling_reset(data, 10, env)


def test_map_conditioned_model_uses_map_without_breaking_first_step_causality():
    torch.manual_seed(7)
    config = PersistentTrajectoryModelConfig(architecture="tracer", history=4, horizon=3, num_robots=1,
                                            hidden_dim=16, num_heads=2, dropout=0, response_delay_steps=1,
                                            static_geometry_features=True)
    model = BeliefConditionedTeamResponseModel(config).eval()
    history, current, candidates = torch.randn(2, 1, 4, 2), torch.randn(2, 1, 1, 4), torch.randn(2, 1, 1, 3, 4)
    segments = torch.tensor([1., -2., 1., 2.]).expand(2, 1, 1, 4)
    original = model(history, current, candidates, static_segments=segments)
    reversed_map = segments[..., [2, 3, 0, 1]]
    reversed_prediction = model(history, current, candidates, static_segments=reversed_map)
    moved_prediction = model(history, current, candidates, static_segments=segments + 3)
    changed_action = model(history, current, candidates + 2, static_segments=segments)
    torch.testing.assert_close(original["mean"], reversed_prediction["mean"])
    assert not torch.allclose(original["mean"], moved_prediction["mean"])
    torch.testing.assert_close(original["mean"][..., 0, :], changed_action["mean"][..., 0, :])
    with pytest.raises(ValueError, match="static_segments"):
        model(history, current, candidates)
    no_map = BeliefConditionedTeamResponseModel(replace(config, static_geometry_features=False))
    assert not hasattr(no_map, "static_encoder")
