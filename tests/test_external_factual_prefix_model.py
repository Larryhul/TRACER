import pytest
import torch
from dataclasses import replace

from sage.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig
from sage.experiments.train_rolling_external_belief import prefix_mode_discrimination_loss


def _model(**overrides):
    config = dict(architecture="tracer", history=4, horizon=3, num_robots=1, num_modes=3,
                  hidden_dim=16, num_heads=2, dropout=0, dt=0.25, response_delay_steps=1,
                  observed_velocity_anchor=True, factual_prefix_dynamics=True,
                  mode_speed_values=(0.6, 1.0, 1.4))
    config.update(overrides)
    return BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(**config)).eval()


def test_factual_prefix_uses_previous_action_not_unexecuted_candidate():
    torch.manual_seed(7)
    model = _model()
    history = torch.randn(2, 1, 4, 2)
    current = torch.randn(2, 1, 1, 4)
    candidates = torch.randn(2, 1, 1, 3, 4)
    original = model(history, current, candidates)
    changed_candidate = model(history, current, candidates + 9)
    for name in ("mean", "log_std"):
        torch.testing.assert_close(original[name][..., 0, :], changed_candidate[name][..., 0, :], rtol=0, atol=0)
    changed_current = current.clone()
    changed_current[..., 2:] += 1.0
    changed_action = model(history, changed_current, candidates)
    assert not torch.allclose(original["mean"][..., 0, :], changed_action["mean"][..., 0, :])
    torch.testing.assert_close(original["log_std"][:, :, 0, 0], original["log_std"][:, :, 2, 0])
    assert original["preferred_speed_mean"].shape == (2, 1)
    loss = original["mean"].square().mean() + original["preferred_speed_mean"].square().mean()
    loss.backward()
    assert model.factual_prefix_mean[-1].weight.grad.abs().sum() > 0
    assert model.speed_condition_encoder[0].weight.grad.abs().sum() > 0


def test_observed_velocity_anchor_is_exact_and_has_no_new_parameters():
    torch.manual_seed(3)
    anchored = _model(factual_prefix_dynamics=False, mode_speed_values=())
    plain = _model(factual_prefix_dynamics=False, mode_speed_values=(), observed_velocity_anchor=False)
    plain.load_state_dict(anchored.state_dict())
    history = torch.randn(2, 1, 4, 2)
    current = torch.randn(2, 1, 1, 4)
    candidates = torch.randn(2, 1, 1, 3, 4)
    difference = anchored(history, current, candidates)["mean"] - plain(history, current, candidates)["mean"]
    velocity = (history[..., -1, :] - history[..., -2, :]) / 0.25
    torch.testing.assert_close(difference, velocity[..., None, None, :].expand_as(difference), atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("overrides", [dict(mode_speed_values=(0.5, 0.5, 1.0)), dict(mode_speed_values=(0.5, 1.0)), dict(response_delay_steps=0)])
def test_invalid_factual_conditioning_rejected(overrides):
    with pytest.raises(ValueError):
        _model(**overrides)


def test_continuous_conditioning_uses_configured_mode_grid_at_inference():
    model = _model(conditioning="continuous")
    output = model(torch.randn(2, 1, 4, 2), torch.randn(2, 1, 1, 4), torch.randn(2, 1, 1, 3, 4))
    assert output["mean"].shape[2] == 3
    assert output["mean"].requires_grad


def test_target_relative_external_geometry_matches_centered_inputs():
    torch.manual_seed(19)
    aligned = _model(geometry_features=True, dynamic_interaction_features=True,
                     robot_positions_target_relative=True)
    centered = _model(geometry_features=True, dynamic_interaction_features=True)
    centered.load_state_dict(aligned.state_dict())
    history = torch.randn(2, 1, 4, 2)
    neighbors = torch.randn(2, 1, 3, 4, 2)
    current = torch.randn(2, 1, 1, 4)
    candidates = torch.randn(2, 1, 1, 3, 4)
    origin = history[..., -1:, :]
    expected = centered(history - origin, current, candidates,
                        neighbor_history=neighbors - origin.unsqueeze(-3))
    actual = aligned(history, current, candidates, neighbor_history=neighbors)
    translation = torch.tensor([17., -31.])
    translated = aligned(history + translation, current, candidates,
                         neighbor_history=neighbors + translation)
    for name in ("mean", "log_std", "preferred_speed_mean"):
        torch.testing.assert_close(actual[name], expected[name])
        torch.testing.assert_close(actual[name], translated[name], atol=2e-5, rtol=2e-5)


def test_factual_discrimination_does_not_use_unobserved_future():
    torch.manual_seed(25)
    means = torch.randn(2, 3, 4, 2, requires_grad=True)
    prediction = {"mean": means, "log_std": torch.zeros_like(means)}
    target = torch.randn(2, 4, 2)
    labels = torch.tensor([0, 2])
    loss = prefix_mode_discrimination_loss(prediction, target, labels, .5, end_step=1)
    changed = target.clone()
    changed[:, 1:] += 100.
    torch.testing.assert_close(loss, prefix_mode_discrimination_loss(prediction, changed, labels, .5, end_step=1))
    loss.backward()
    assert means.grad[..., 0, :].abs().sum() > 0
    assert means.grad[..., 1:, :].abs().sum() == 0


def test_closed_mode_gate_removes_spurious_evidence_without_changing_uniform_mean():
    torch.manual_seed(29)
    model = _model(mode_response_gate=True)
    history = torch.randn(2, 1, 4, 2)
    current = torch.randn(2, 1, 1, 4)
    candidates = torch.randn(2, 1, 1, 3, 4)
    model.config = replace(model.config, mode_response_gate=False)
    ungated = model(history, current, candidates)
    model.config = replace(model.config, mode_response_gate=True)
    with torch.no_grad():
        for decoder in (model.factual_mode_gate, model.candidate_mode_gate):
            decoder[-1].weight.zero_()
            decoder[-1].bias.fill_(-100.)
    gated = model(history, current, candidates)
    for name in ("mean", "log_std"):
        expected = ungated[name].mean(2, keepdim=True).expand_as(gated[name])
        torch.testing.assert_close(gated[name], expected)
    assert gated["mode_gate_logits"].shape == (2, 1, 3)


def test_mode_gate_prefix_is_independent_of_unexecuted_candidate():
    model = _model(mode_response_gate=True)
    history = torch.randn(2, 1, 4, 2)
    current = torch.randn(2, 1, 1, 4)
    candidates = torch.randn(2, 1, 1, 3, 4)
    original = model(history, current, candidates)
    changed = model(history, current, candidates + 9)
    torch.testing.assert_close(original["mode_gate_logits"][..., 0], changed["mode_gate_logits"][..., 0], atol=0, rtol=0)
    assert not torch.allclose(original["mode_gate_logits"][..., 1:], changed["mode_gate_logits"][..., 1:])
    original["mode_gate_logits"].square().sum().backward()
    assert model.factual_mode_gate[-1].weight.grad.abs().sum() > 0
    assert model.candidate_mode_gate[-1].weight.grad.abs().sum() > 0
