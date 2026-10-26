import pytest
import torch
from torch.utils.data import DataLoader

from sage.belief.filter import run_categorical_filter
from sage.experiments.audit_h2_scale_calibration import _brier_from_rows
from sage.experiments.audit_component_identity_replay import METRICS, METHODS, aggregate_contrasts, recording_scores
from sage.experiments.train_persistent_trajectory_response import _evaluate, _shuffled_identity_filter, causal_training_beliefs, mixture_mean_trajectory_loss
from sage.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    TrajectoryRecurrentBelief,
    TrajectoryRecurrentConfig,
)
from sage.response.persistent_trajectory import MultiEntityPersistentTrajectoryEpisodeDataset, MultiEntityPersistentTrajectorySpec


@pytest.mark.parametrize("entities", [2, 3, 4])
def test_identity_keys_change_across_windows_for_every_entity_count(entities):
    likelihoods = torch.eye(entities)[:, None].expand(-1, 4, -1) * 8.
    mask = torch.ones(entities, 4, dtype=torch.bool)
    prior = torch.full((entities, entities), 1. / entities).log()
    before, after = _shuffled_identity_filter(likelihoods, mask, torch.zeros(entities, dtype=torch.long), torch.arange(entities), prior)
    persistent_before, _ = run_categorical_filter(likelihoods, prior_log_belief=prior, observation_mask=mask)
    torch.testing.assert_close(before[:, 0], persistent_before[:, 0])
    assert torch.diagonal(before[:, 1].exp()).max() < .01
    assert torch.diagonal(persistent_before[:, 1].exp()).min() > .99
    torch.testing.assert_close(after.exp().sum(-1), torch.ones(entities, 4))


def test_identity_corruption_keeps_single_entities_and_episode_boundaries():
    likelihoods = torch.tensor([[[6., 0.], [6., 0.], [6., 0.]], [[0., 6.], [0., 6.], [0., 6.]]])
    mask = torch.tensor([[True, False, True], [True, True, True]])
    prior = torch.full((2, 2), .5).log()
    expected = run_categorical_filter(likelihoods, prior_log_belief=prior, observation_mask=mask)
    actual = _shuffled_identity_filter(likelihoods, mask, torch.tensor([12, 24]), torch.zeros(2, dtype=torch.long), prior)
    for expected_value, actual_value in zip(expected, actual):
        torch.testing.assert_close(expected_value, actual_value)


def test_identity_corruption_is_causal_and_independent_of_input_row_order():
    likelihoods = torch.tensor([[[8., 0.]] * 4, [[0., 8.]] * 4, [[4., 0.]] * 4, [[0., 4.]] * 4])
    mask = torch.ones(4, 4, dtype=torch.bool)
    episodes = torch.tensor([10, 10, 20, 20])
    slots = torch.tensor([0, 1, 0, 1])
    prior = torch.full((4, 2), .5).log()
    original = _shuffled_identity_filter(likelihoods, mask, episodes, slots, prior)
    permutation = torch.tensor([3, 1, 2, 0])
    shuffled = _shuffled_identity_filter(likelihoods[permutation], mask[permutation], episodes[permutation], slots[permutation], prior[permutation])
    changed = likelihoods.clone()
    changed[:, 2:] = changed[:, 2:].flip(-1)
    future_changed = _shuffled_identity_filter(changed, mask, episodes, slots, prior)
    for original_value, shuffled_value, changed_value in zip(original, shuffled, future_changed):
        torch.testing.assert_close(original_value[permutation], shuffled_value)
        torch.testing.assert_close(original_value[:, :2], changed_value[:, :2])


def test_brier_uses_all_classes_and_rejects_legacy_true_probability_only_rows():
    rows = [{"active": 1, "window": 1, "posterior_true_probability": .5, "posterior_brier": .375}]
    assert _brier_from_rows(rows) == .375
    del rows[0]["posterior_brier"]
    with pytest.raises(ValueError, match="full multiclass Brier"):
        _brier_from_rows(rows)


def test_identity_replay_clusters_recordings_instead_of_overlapping_windows():
    rows = []
    for method in METHODS:
        for entity, repetitions in [(10, 20), (20, 1)]:
            value = 1. if method == "persistent" else (2. if entity == 10 else 4.)
            rows.extend({"main_evaluation": 1, "method": method, "entity_id": entity, **dict.fromkeys(METRICS, value)} for _ in range(repetitions))
    scores = recording_scores(rows, {10: "lab/experiment1/condition", 20: "lab/experiment2/condition"})
    records = aggregate_contrasts([{"seed": 7, "recording_scores": scores}, {"seed": 8, "recording_scores": scores}], "bi3")
    assert len(records[0]["clusters"]) == 2
    assert records[0]["mean_difference"] == -2.
    with pytest.raises(ValueError, match="absent"):
        recording_scores(rows, {10: "first"})


def test_causal_training_belief_never_uses_current_response_or_unexecuted_suffix():
    means = torch.zeros(1, 4, 2, 3, 2, requires_grad=True)
    with torch.no_grad():
        means[:, :, 1] = 2.
    deviations = torch.zeros_like(means)
    target = torch.zeros(1, 4, 3, 2)
    mask = torch.ones(1, 4, dtype=torch.bool)
    original = causal_training_beliefs(means, deviations, target, mask, 1, .5)
    assert not original.requires_grad
    torch.testing.assert_close(original[:, 0], torch.full((1, 2), .5))
    assert original[0, 1, 0] > .85
    changed = target.clone()
    changed[:, 2:] = 2.
    modified = causal_training_beliefs(means, deviations, changed, mask, 1, .5)
    torch.testing.assert_close(original[:, :3], modified[:, :3])
    suffix = target.clone()
    suffix[:, :, 1:] = 100.
    torch.testing.assert_close(original, causal_training_beliefs(means, deviations, suffix, mask, 1, .5))
    mask[:, 0] = False
    unobserved = causal_training_beliefs(means, deviations, target, mask, 1, .5)
    torch.testing.assert_close(unobserved[:, 1], torch.full((1, 2), .5))


def test_belief_weighted_trajectory_loss_preserves_uniform_default_and_gradients():
    means = torch.tensor([[[[[1., 0.]], [[-1., 0.]]]]], requires_grad=True)
    target = torch.tensor([[[[1., 0.]]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    uniform = torch.full((1, 1, 2), .5)
    default = mixture_mean_trajectory_loss(means, target, mask, .4)
    weighted = mixture_mean_trajectory_loss(means, target, mask, .4, mode_weights=uniform)
    torch.testing.assert_close(default, weighted)
    informed = mixture_mean_trajectory_loss(means, target, mask, .4, mode_weights=torch.tensor([[[.9, .1]]]))
    assert informed < weighted
    informed.backward()
    assert torch.isfinite(means.grad).all()
    with pytest.raises(ValueError, match="sum to one"):
        mixture_mean_trajectory_loss(means, target, mask, .4, mode_weights=torch.ones_like(uniform))


def test_frozen_training_support_preserves_deployed_least_risk_fallback():
    from sage.experiments.train_component_mode_contrast import frozen_executable_support

    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(history=4, horizon=4, hidden_dim=16)).eval()
    current = torch.zeros(2, 4, 2, 4)
    current[..., :2] = 5.
    candidates = current.unsqueeze(-2).expand(-1, -1, -1, 4, -1).clone()
    data = {"history": torch.zeros(2, 4, 2), "current": current, "candidates": candidates}
    parameters = {name: value.clone() for name, value in model.state_dict().items()}
    support = frozen_executable_support(model, data, "cpu", {"violation_probability_alpha": .1, "risk_samples": 8, "collision_margin": 100.})
    assert support.shape == (2, 4)
    assert support.dtype == torch.bool
    assert (support.sum(-1) == 1).all()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, parameters[name])


def test_h2_evaluator_records_full_multiclass_brier_without_changing_uniform_predictions():
    torch.manual_seed(61)
    dataset = MultiEntityPersistentTrajectoryEpisodeDataset(MultiEntityPersistentTrajectorySpec(
        num_episodes=2, windows=5, history=4, horizon=4, num_robots=2, num_entities=4, active_entities=2,
    ))
    response = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(history=4, horizon=4, hidden_dim=16, num_heads=2, graph_layers=1, dropout=0.))
    recurrent = TrajectoryRecurrentBelief(TrajectoryRecurrentConfig(history=4, horizon=4, hidden_dim=16))
    _, rows = _evaluate(response, recurrent, DataLoader(dataset, batch_size=2), "cpu", .4, .05, .8, 2, "test")
    uniform = [row for row in rows if row["method"] == "no-belief"]
    oracle = [row for row in rows if row["method"] == "oracle"]
    assert _brier_from_rows(uniform, "decision") == pytest.approx(2. / 3.)
    assert _brier_from_rows(uniform, "posterior") == pytest.approx(2. / 3.)
    assert _brier_from_rows(oracle, "decision") == 0.
    assert all(0. <= row["posterior_brier"] <= 2. for row in rows)
