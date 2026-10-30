from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tracer.experiments.run_sicnav_tracer_loop import (
    _belief_observation_due, _update_belief, _validate_external_belief_cases,
)
from tracer.belief.adaptive_persistent import AdaptivePersistentBelief
from tracer.experiments.train_rolling_external_belief import (
    first_prefix_rows,
    prefix_mode_discrimination_loss,
    speed_thresholds,
    track_tensors,
)
from tracer.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig


@pytest.mark.parametrize("candidate_uncertainty", [False, True])
def test_external_delayed_mode_model_has_candidate_independent_prefix(candidate_uncertainty):
    torch.manual_seed(7)
    config = PersistentTrajectoryModelConfig(
        architecture="tracer", history=4, horizon=3, num_robots=1,
        hidden_dim=16, num_heads=2, dropout=0, dt=.25,
        response_delay_steps=1, candidate_uncertainty=candidate_uncertainty)
    model = BeliefConditionedTeamResponseModel(config).eval()
    history, current = torch.randn(2, 1, 4, 2), torch.randn(2, 1, 1, 4)
    candidates = torch.randn(2, 1, 1, 3, 4)
    original = model(history, current, candidates)
    changed = model(history, current, candidates + 2)
    for name in ("mean", "log_std"):
        torch.testing.assert_close(original[name][..., 0, :], changed[name][..., 0, :], rtol=0, atol=0)
    assert not torch.allclose(original["mean"][..., 1:, :], changed["mean"][..., 1:, :])
    assert not torch.allclose(original["mean"][:, :, 0], original["mean"][:, :, 1])
    original["mean"].square().sum().backward()
    assert model.mode_embedding.weight.grad.abs().sum() > 0


def test_external_delay_configuration_rejects_unsupported_values():
    with pytest.raises(ValueError, match="prediction horizon"):
        BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(response_delay_steps=-1))
    with pytest.raises(ValueError, match="requires architecture"):
        BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(response_delay_steps=1))


def test_speed_bins_use_training_identities_not_frequency_or_validation_labels():
    data = {"case": np.array([1, 1, 1, 2, 3]), "human": np.zeros(5, dtype=int)}
    speeds = {(1, 0): .5, (2, 0): 1., (3, 0): 1.5, (4, 0): 999.}
    expected = np.quantile([.5, 1., 1.5], [1 / 3, 2 / 3])
    np.testing.assert_allclose(speed_thresholds(data, speeds), expected)
    speeds[4, 0] = -999.
    np.testing.assert_allclose(speed_thresholds(data, speeds), expected)


def test_first_prefix_calibration_uses_one_factual_observation_per_state():
    data = {"case": np.ones(4, dtype=int), "human": np.zeros(4, dtype=int),
            "window": np.array([6, 0, 0, 6]), "mode": np.ones(4, dtype=int),
            "target": np.zeros((4, 3, 2))}
    rows = first_prefix_rows(data)
    np.testing.assert_array_equal(rows, [1, 0])
    likelihoods, modes, active = track_tensors(data, rows, torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
    assert likelihoods.shape == (1, 2, 3)
    assert active.sum() == 2 and modes.tolist() == [1]
    data["target"][2, 0, 0] = 1
    with pytest.raises(ValueError, match="responses differ"):
        first_prefix_rows(data)


def test_uniform_control_does_not_accumulate_mode_evidence():
    belief = torch.tensor([.1, .2, .7]).log()
    likelihood = torch.tensor([3., -2., 1.])
    args = SimpleNamespace(likelihood_scale=.5)
    uniform, changed = _update_belief(belief, likelihood, "tracer-uniform", args)
    torch.testing.assert_close(uniform.exp(), torch.full((3,), 1 / 3))
    posterior, _ = _update_belief(belief, likelihood, "tracer-persistent", args)
    assert not torch.allclose(posterior, uniform)
    assert not changed


def test_unspecified_external_belief_cadence_uses_execution_clock():
    assert _belief_observation_due(0, 1, None)
    assert _belief_observation_due(5, 3, None)
    with pytest.raises(ValueError, match="execute_steps"):
        _belief_observation_due(0, 0, None)


def test_adaptive_persistent_external_policy_uses_streaming_state():
    args = SimpleNamespace(
        likelihood_scale=0.5,
        adaptive_base_hazard=0.0,
        adaptive_max_hazard=0.2,
        adaptive_surprise_threshold=2.0,
        adaptive_surprise_scale=0.1,
        adaptive_hard_reset=False,
        adaptive_threshold_hysteresis=0.2,
        adaptive_cooldown_steps=2,
    )
    state = AdaptivePersistentBelief(3)
    prior = torch.full((3,), -torch.log(torch.tensor(3.0)))
    first, _ = _update_belief(
        prior, torch.tensor([4.0, -2.0, -2.0]),
        "tracer-adaptive-persistent", args, state,
    )
    second, _ = _update_belief(
        first, torch.tensor([0.0, 0.0, 0.0]),
        "tracer-adaptive-persistent", args, state,
    )
    assert second[0] > second[1]
    torch.testing.assert_close(second, state.log_belief)


def test_prefix_mode_discrimination_loss_uses_causal_first_step():
    prediction = {
        "mean": torch.zeros(2, 3, 4, 2, requires_grad=True),
        "log_std": torch.zeros(2, 3, 4, 2),
    }
    target = torch.zeros(2, 4, 2)
    labels = torch.tensor([0, 2])
    loss = prefix_mode_discrimination_loss(prediction, target, labels, temperature=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction["mean"].grad is not None
    assert prediction["mean"].grad[..., 1:, :].abs().sum() == 0


def test_external_belief_evidence_cadence_matches_calibration():
    due = [window for window in range(14) if _belief_observation_due(window, 1, 6)]
    assert due == [0, 6, 12]
    with pytest.raises(ValueError, match="positive multiple"):
        _belief_observation_due(0, 4, 6)


def test_external_belief_navigation_refuses_training_or_calibration_overlap():
    checkpoint = {"provenance": {"case_splits": {"train": [112, 113], "val": [136]}}}
    _validate_external_belief_cases(checkpoint, 80, 8)
    with pytest.raises(ValueError, match="overlap"):
        _validate_external_belief_cases(checkpoint, 111, 2)
    with pytest.raises(ValueError, match="overlap"):
        _validate_external_belief_cases(checkpoint, 136, 1)
