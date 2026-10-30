import numpy as np
import pytest
import torch

from tracer.experiments.audit_real_h1 import prediction_metrics, summarize_recordings
from tracer.experiments.train_persistent_trajectory_response import constant_velocity_target, mixture_mean_trajectory_loss
from tracer.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig


def test_recording_macro_mean_does_not_treat_windows_as_independent_recordings():
    values = np.array([[1., 1.], [1., 1.], [10., 10.]])
    recordings = np.array(["first", "first", "second"])
    result = summarize_recordings(values, np.ones_like(values, dtype=bool), recordings)
    assert result["count"] == 6
    assert result["window_weighted_mean"] == 4.
    assert result["recording_macro_mean"] == 5.5


def test_empty_stratum_has_no_fabricated_zero_error():
    result = summarize_recordings(np.ones((1, 2)), np.zeros((1, 2), dtype=bool), np.array(["recording"]))
    assert result["window_weighted_mean"] is None
    assert result["recording_macro_mean"] is None


def test_recording_summary_rejects_misaligned_masks():
    with pytest.raises(ValueError, match="align"):
        summarize_recordings(np.zeros((2, 3)), np.ones((3, 2)), np.array(["first", "second"]))


def test_prediction_errors_integrate_velocity_in_seconds():
    prediction = torch.ones(1, 1, 3, 2)
    prediction[..., 1] = 0.
    target = torch.zeros_like(prediction)
    measured = prediction_metrics(prediction, target, .4)
    np.testing.assert_allclose(measured["ade"], .8)
    np.testing.assert_allclose(measured["fde"], 1.2)


def test_constant_velocity_target_repeats_last_observed_velocity():
    history = torch.tensor([[[[0., 0.], [0.4, 0.], [0.8, 0.]]]])
    target = constant_velocity_target(history, horizon=2, dt=.4)
    assert target.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(target.numpy(), [[[[1., 0.], [1., 0.]]]])


def test_external_relative_robot_geometry_is_translation_invariant():
    torch.manual_seed(42)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(
        history=3, horizon=2, num_robots=1, hidden_dim=16, num_heads=4,
        architecture="tracer", dropout=0., geometry_features=True,
        dynamic_interaction_features=True, robot_positions_target_relative=True,
    )).eval()
    history = torch.randn(2, 2, 3, 2)
    neighbors = torch.randn(2, 2, 1, 3, 2)
    current = torch.randn(2, 2, 1, 4)
    candidates = torch.randn(2, 2, 1, 2, 4)
    translation = torch.tensor([17., -23.])
    with torch.no_grad():
        original = model(history, current, candidates, neighbor_history=neighbors)
        translated = model(history + translation, current, candidates,
                           neighbor_history=neighbors + translation)
    for field in ("mean", "log_std"):
        torch.testing.assert_close(original[field], translated[field], atol=1e-5, rtol=1e-5)


def test_mixture_trajectory_loss_integrates_then_averages():
    modes = torch.zeros(1, 1, 2, 3, 2)
    modes[..., 0] = 1.
    target = torch.zeros(1, 1, 3, 2)
    loss = mixture_mean_trajectory_loss(modes, target, torch.ones(1, 1, dtype=torch.bool), .4)
    torch.testing.assert_close(loss, torch.tensor((.8 + .25 * 1.2) / 1.2))


def test_mixture_trajectory_loss_masks_padding_and_backpropagates():
    modes = torch.ones(1, 2, 3, 2, 2, requires_grad=True)
    target = torch.zeros(1, 2, 2, 2)
    loss = mixture_mean_trajectory_loss(modes, target, torch.tensor([[True, False]]), .1)
    loss.backward()
    assert modes.grad[:, 0].abs().sum() > 0
    assert modes.grad[:, 1].abs().sum() == 0


def test_mixture_trajectory_loss_matches_residual_representation():
    torch.manual_seed(42)
    modes = torch.randn(2, 3, 3, 4, 2)
    target = torch.randn(2, 3, 4, 2)
    offset = torch.randn(2, 3, 4, 2)
    mask = torch.ones(2, 3, dtype=torch.bool)
    torch.testing.assert_close(
        mixture_mean_trajectory_loss(modes, target, mask, .4),
        mixture_mean_trajectory_loss(modes - offset.unsqueeze(2), target - offset, mask, .4),
    )


def test_mixture_trajectory_loss_does_not_penalize_canceling_modes():
    modes = torch.tensor([[[[[1., 0.]], [[-1., 0.]]]]])
    loss = mixture_mean_trajectory_loss(modes, torch.zeros(1, 1, 1, 2), torch.ones(1, 1, dtype=torch.bool), .4)
    assert loss == 0


@pytest.mark.parametrize("dt, endpoint_weight", [(0., .25), (-.1, .25), (float("nan"), .25), (.1, -.1), (.1, float("inf"))])
def test_mixture_trajectory_loss_rejects_invalid_scaling(dt, endpoint_weight):
    with pytest.raises(ValueError, match="dt must be positive"):
        mixture_mean_trajectory_loss(
            torch.zeros(1, 1, 3, 2, 2), torch.zeros(1, 1, 2, 2),
            torch.ones(1, 1, dtype=torch.bool), dt, endpoint_weight,
        )


def test_mixture_trajectory_loss_rejects_empty_observation_mask():
    with pytest.raises(ValueError, match="select an observation"):
        mixture_mean_trajectory_loss(
            torch.zeros(1, 1, 3, 2, 2), torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, dtype=torch.bool), .1,
        )
