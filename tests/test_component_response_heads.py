import pytest
import torch

from tracer.experiments.train_component_response_heads import mean_reference_effect_loss, response_head_loss, select_response_head_parameters
from tracer.experiments.train_component_mode_contrast import grouped_prefix_costs
from tracer.response.closed_loop import PlanningCostConfig
from tracer.response.persistent_models import stationary_reference_candidates
from tracer.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig


def test_reference_supervision_detects_hidden_reference_error():
    target = torch.randn(4, 3, 6, 2)
    reference = torch.randn_like(target)
    perfect, _ = mean_reference_effect_loss(target, reference, target, reference, 3)
    assert perfect == 0
    loss, metrics = mean_reference_effect_loss(target, reference + 1, target, reference, 3)
    assert metrics["candidate_prefix_mse"] == 0
    assert metrics["reference_prefix_mse"] == pytest.approx(1.)
    assert metrics["effect_prefix_mse"] == pytest.approx(1.)
    assert loss == pytest.approx(2.)
    with pytest.raises(ValueError, match="align"):
        mean_reference_effect_loss(target, reference, target, reference, 7)


def test_mean_head_training_preserves_encoder_uncertainty_and_mode_embedding():
    torch.manual_seed(517)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(history=4, horizon=4, num_robots=2, hidden_dim=16, dropout=0., architecture="tracer")).eval()
    pair = select_response_head_parameters(model, "pair")
    assert not any(name.startswith(("base_decoder.", "single_decoder.")) for name in pair)
    heads = select_response_head_parameters(model, "mean-heads")
    assert set(pair).issubset(heads)
    assert any(name.startswith("base_decoder.") for name in heads)
    assert any(name.startswith("single_decoder.") for name in heads)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    batch = {"history": torch.randn(2, 4, 2), "current": torch.randn(2, 5, 2, 4), "candidates": torch.randn(2, 5, 2, 4, 4),
             "all_mode_responses": torch.randn(2, 3, 5, 4, 2), "all_mode_references": torch.randn(2, 3, 5, 4, 2)}
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.001)
    loss, _ = response_head_loss(model, batch, 2)
    loss.backward()
    optimizer.step()
    changed = {name for name, value in model.state_dict().items() if not torch.equal(value, initial[name])}
    assert changed and changed.issubset(heads)
    assert any(name.startswith("base_decoder.") for name in changed)
    assert not any(name.startswith(("mode_embedding.", "log_std_decoder.", "tracer_encoder.")) for name in changed)


def test_prefix_cost_supervision_matches_deployed_grouped_costs():
    torch.manual_seed(853)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(history=4, horizon=4, num_robots=2, hidden_dim=16, dropout=0., architecture="tracer")).eval()
    batch = {"history": torch.randn(2, 4, 2), "current": torch.randn(2, 5, 2, 4), "candidates": torch.randn(2, 5, 2, 4, 4),
             "all_mode_responses": torch.randn(2, 3, 5, 4, 2), "all_mode_references": torch.randn(2, 3, 5, 4, 2)}
    history = batch["history"][:, None].expand(-1, 5, -1, -1).reshape(10, 1, 4, 2)
    current = batch["current"].reshape(10, 1, 2, 4)
    candidates = batch["candidates"].reshape(10, 1, 2, 4, 4)
    planning = vars(PlanningCostConfig(task_steps=2))
    with torch.no_grad():
        means = model(history, current, candidates)["mean"][:, 0].reshape(2, 5, 3, 4, 2)
        references = model(history, current, stationary_reference_candidates(current, 4))["mean"][:, 0].reshape_as(means)
        batch["all_mode_costs"] = grouped_prefix_costs(means, references, batch["current"], batch["candidates"], planning, model.config.dt, 2)
    baseline, _ = response_head_loss(model, batch, 2)
    aligned, metrics = response_head_loss(model, batch, 2, planning, 64.)
    torch.testing.assert_close(aligned, baseline)
    assert metrics["cost_mse"] == pytest.approx(0., abs=1e-10)
    batch["all_mode_costs"] += .5
    changed, metrics = response_head_loss(model, batch, 2, planning, 64.)
    torch.testing.assert_close(changed - baseline, torch.tensor(16.))
    assert metrics["cost_mse"] == pytest.approx(.25)
