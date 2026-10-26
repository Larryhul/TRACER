from dataclasses import replace

import pytest
import torch

from sage.response.h1_transfer import initialize_h1_encoder, initialize_single_robot_response
from sage.response.models import ResponseModelConfig, TeamResponsePredictor
from sage.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig
from sage.experiments.calibrate_persistent_belief import calibration_scores
from sage.response.external_checkpoint_adapter import delayed_response_prediction
from sage.response.data import EntityResponseBatch


def test_h1_encoder_transfer_preserves_h2_mode_decoder_and_supports_extra_robot(tmp_path):
    source_config = ResponseModelConfig(
        architecture="tracer", hidden_dim=16, num_heads=2, num_robots=2,
        geometry_features=True, dynamic_interaction_features=True, temporal_pooling="attention",
    )
    source = TeamResponsePredictor("joint", source_config)
    path = tmp_path / "h1.pt"
    torch.save({"model": source.state_dict(), "model_config": vars(source_config)}, path)
    target_config = PersistentTrajectoryModelConfig(
        architecture="tracer", hidden_dim=16, num_heads=2, num_robots=3,
        geometry_features=True, dynamic_interaction_features=True, temporal_pooling="attention",
    )
    target = BeliefConditionedTeamResponseModel(target_config)
    mode_embedding = target.mode_embedding.weight.detach().clone()
    metadata = initialize_h1_encoder(target, path)
    assert metadata["source_robots"] == 2 and metadata["target_robots"] == 3
    for name, parameter in source.tracer_encoder.state_dict().items():
        torch.testing.assert_close(parameter, target.tracer_encoder.state_dict()[name])
    torch.testing.assert_close(mode_embedding, target.mode_embedding.weight)
    output = target(torch.randn(2, 1, 8, 2), torch.randn(2, 1, 3, 4), torch.randn(2, 1, 3, 12, 4))
    assert output["mean"].shape == (2, 1, 3, 12, 2)
    output["mean"].sum().backward()
    assert any(parameter.grad is not None for parameter in target.tracer_encoder.parameters())
    with pytest.raises(ValueError, match="configuration mismatch"):
        initialize_h1_encoder(BeliefConditionedTeamResponseModel(replace(target_config, dynamic_interaction_features=False)), path)


def test_calibration_uses_only_previous_observations_and_active_tracks():
    likelihoods = torch.tensor([[[0.0, -4.0], [-2.0, 0.0], [0.0, -9.0]]])
    active = torch.ones(1, 3, dtype=torch.bool)
    before = calibration_scores(likelihoods, torch.tensor([0]), active, [0.05, 0.5])
    altered = likelihoods.clone()
    altered[:, -1] = torch.tensor([-1000.0, 1000.0])
    after = calibration_scores(altered, torch.tensor([0]), active, [0.05, 0.5])
    for original, changed in zip(before, after):
        assert original["decision_mode_nll"] == changed["decision_mode_nll"]
        assert original["posterior_mode_nll"] != changed["posterior_mode_nll"]
    with pytest.raises(ValueError, match="positive"):
        calibration_scores(likelihoods, torch.tensor([0]), active, [0.0])


def test_full_single_robot_transfer_preserves_every_predictive_mode(tmp_path):
    torch.manual_seed(7)
    source_config = ResponseModelConfig(architecture="tracer", hidden_dim=16, num_heads=2, num_robots=1,
                                        history=4, horizon=3, dt=.25, dropout=0,
                                        geometry_features=True, dynamic_interaction_features=True, temporal_pooling="attention")
    source = TeamResponsePredictor("joint", source_config).eval()
    path = tmp_path / "source.pt"
    torch.save({"model": source.state_dict(), "model_config": vars(source_config), "variant": "joint", "response_delay_steps": 1}, path)
    config = PersistentTrajectoryModelConfig(architecture="tracer", hidden_dim=16, num_heads=2, num_robots=1,
                                             history=4, horizon=3, dt=.25, dropout=0, num_modes=3,
                                             geometry_features=True, dynamic_interaction_features=True, temporal_pooling="attention",
                                             min_log_std=source_config.min_log_std, max_log_std=source_config.max_log_std,
                                             response_delay_steps=1, joint_uncertainty_decoder=True)
    target = BeliefConditionedTeamResponseModel(config).eval()
    metadata = initialize_single_robot_response(target, path)
    history = torch.randn(2, 4, 2)
    current, candidates = torch.randn(2, 1, 4), torch.randn(2, 1, 3, 4)
    batch = EntityResponseBatch(history, history[:, -1], current, candidates, torch.ones(2, 1, dtype=torch.bool),
                                torch.randn(2, 3, 2), torch.ones(2, dtype=torch.bool), torch.arange(2), torch.zeros(2, dtype=torch.long), .25)
    expected = delayed_response_prediction(source, batch, 1)
    prediction = target(history[:, None], current[:, None], candidates[:, None])
    for name in ("mean", "log_std"):
        torch.testing.assert_close(prediction[name][:, 0], expected[name][:, None].expand(-1, 3, -1, -1))
    assert metadata["mode_conditioning_columns_initially_zero"]
    prediction["mean"][:, :, 0].square().sum().backward()
    assert target.base_decoder[0].weight.grad[:, 16:32].abs().sum() > 0
    with pytest.raises(ValueError, match="one-robot"):
        initialize_single_robot_response(BeliefConditionedTeamResponseModel(replace(config, num_robots=2)), path)
