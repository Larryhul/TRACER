import numpy as np
import pytest

from tracer.response.reliable_executable_value import DEFAULT_MINIMUM_GAINS, reliable_action, reliability_targets
from tracer.response.reliable_executable_value import (
    PAIRWISE_EFFECT_OBJECTIVE, PAIRWISE_RELIABILITY_OBJECTIVE, RELIABILITY_TARGETS,
    pairwise_reliable_model_action, reliable_model_action,
)
from tracer.experiments.train_reliable_executable_value import evaluate_reliable_value


def test_safety_improvement_is_not_itself_a_deployment_gain():
    upper = np.zeros((3, 9))
    upper[1, 0] = -.1
    upper[1, 8] = 8.
    upper[2, 8] = -10.
    assert reliable_action(upper, 0, DEFAULT_MINIMUM_GAINS) == 2


def test_time_gain_cannot_offset_higher_collision_timeout_stall_or_human_variation():
    for metric in range(8):
        upper = np.zeros((2, 9))
        upper[1, metric] = .001
        upper[1, 8] = -10.
        assert reliable_action(upper, 0, DEFAULT_MINIMUM_GAINS) == 0


def test_uncertain_gain_and_unsupported_action_preserve_reference():
    upper = np.zeros((3, 9))
    upper[1, 8] = -.1
    upper[2, 0] = -.5
    assert reliable_action(upper, 0, DEFAULT_MINIMUM_GAINS, [True, True, False]) == 0


def test_goal_shortfall_gain_precedes_human_variation_and_time():
    upper = np.zeros((3, 9))
    upper[1, 6] = -.2
    upper[1, 8] = 1.
    upper[2, 6] = -.5
    assert reliable_action(upper, 0, DEFAULT_MINIMUM_GAINS) == 2


def test_missing_legacy_exposure_is_not_silently_zero_filled():
    with pytest.raises(ValueError, match="restored full-episode"):
        reliability_targets({"success": 1.}, .3)


def test_reliability_targets_keep_time_separate_from_goal_shortfall():
    branch = dict(outcome_history_restored=True, collision_free_success=0., collision=0., wall_collision=0.,
                  success=0., timeout=1., stagnation_time=2., terminal_goal_distance=3.,
                  human_velocity_variation=.7, elapsed_time=15.)
    np.testing.assert_allclose(reliability_targets(branch, .3), [1, 0, 0, 1, 1, 2, 2.7, .7, 15])


def test_out_of_training_support_abstains_even_when_predicted_gain_is_large():
    model = {"target_names": list(RELIABILITY_TARGETS), "ensemble": [{}] * 8,
             "uncertainty_multiplier": 2., "minimum_gains": list(DEFAULT_MINIMUM_GAINS),
             "training_support": {"feature_indices": [0], "center": [0.], "scale": [1.],
                                  "anchors": [[0.]], "maximum_distance": 1.}}
    upper = np.zeros((2, 9))
    upper[1, 6] = -.5
    assert reliable_model_action(model, [[0.], [3.]], 0, upper) == 0
    assert reliable_model_action(model, [[2.], [0.]], 0, upper) == 0
    assert reliable_model_action(model, [[0.], [.5]], 0, upper) == 1
    with pytest.raises(ValueError, match="frozen reference"):
        reliable_model_action(model, [[0.], [.5]], 0, upper, [False, True])


def test_single_model_cannot_claim_resampled_confidence():
    model = {"target_names": list(RELIABILITY_TARGETS), "ensemble": [{}], "uncertainty_multiplier": 2.}
    with pytest.raises(ValueError, match="scene-resampled uncertainty"):
        reliable_model_action(model, [[0.], [.5]], 0, np.zeros((2, 9)))


def test_reliable_development_acceptance_allows_unchanged_safety():
    member = {"feature_version": "observable-prefix-response-value-v3-temporal-geometry",
              "scale": [1.], "coefficients": [[0., 0., 0., 0., 0., 0., 0., 0., -1.]]}
    model_data = {"feature_version": "observable-prefix-response-value-v3-temporal-geometry",
                  "selection_objective": "safe-completion-lexicographic-v1",
                  "target_names": list(RELIABILITY_TARGETS), "ensemble": [member] * 8,
                  "uncertainty_multiplier": 2., "minimum_gains": list(DEFAULT_MINIMUM_GAINS),
                  "training_support": {"feature_indices": [0], "center": [0.], "scale": [1.],
                                       "anchors": [[0.]], "maximum_distance": 10.}}
    groups = []
    for case in (1, 2):
        targets = np.zeros((2, 9))
        targets[1, 8] = -1.
        groups.append({"case": case, "window": 0, "rerank_allowed": True,
                       "candidate_indices": [0, 1], "parent": 0,
                       "features": np.array([[0.], [1.]]),
                       "response_risk_support": np.array([True, True]), "targets": targets})
    result = evaluate_reliable_value(model_data, groups)
    assert result["eligible_for_development_navigation"]
    assert result["first_improved_priority_tier"] == 1


def test_reliable_development_evaluation_matches_deployment_schedule():
    member = {"feature_version": "observable-prefix-response-value-v3-temporal-geometry",
              "scale": [1.], "coefficients": [[0., 0., 0., 0., 0., 0., 0., 0., -1.]]}
    model_data = {"feature_version": "observable-prefix-response-value-v3-temporal-geometry",
                  "selection_objective": "safe-completion-lexicographic-v1",
                  "target_names": list(RELIABILITY_TARGETS), "ensemble": [member] * 8,
                  "uncertainty_multiplier": 2., "minimum_gains": list(DEFAULT_MINIMUM_GAINS),
                  "training_support": {"feature_indices": [0], "center": [0.], "scale": [1.],
                                       "anchors": [[0.]], "maximum_distance": 10.},
                  "deployment_stride": 4, "maximum_episode_interventions": 1}
    groups = []
    for window in (4, 0):
        targets = np.zeros((2, 9))
        targets[1, 8] = -1.
        groups.append({"case": 1, "window": window, "rerank_allowed": True,
                       "candidate_indices": [0, 1], "parent": 0,
                       "features": np.array([[0.], [1.]]),
                       "response_risk_support": np.array([True, True]), "targets": targets})
    result = evaluate_reliable_value(model_data, groups)
    assert [row["window"] for row in result["records"]] == [0, 4]
    assert [row["changed"] for row in result["records"]] == [True, False]


def pairwise_model(human_bias=3., harm_bias=-5.):
    member = {
        "center": [0., 0., 0.], "scale": [1., 1., 1.],
        "input_weight": [[0., 1., 0.]], "input_bias": [0.],
        "layer_norm_weight": [1.], "layer_norm_bias": [0.], "layer_norm_epsilon": 1e-5,
        "hidden_weight": [[0.]], "hidden_bias": [1.],
        "output_weight": [[0.], [0.]], "output_bias": [human_bias, harm_bias],
    }
    return {
        "selection_objective": PAIRWISE_RELIABILITY_OBJECTIVE,
        "basic_feature_count": 1, "ensemble": [member] * 8,
        "uncertainty_multiplier": 1.,
        "probability_calibration": [{"slope": 1., "intercept": 0.}] * 2,
        "human_score_threshold": .8, "harm_score_threshold": .05,
        "training_support": {"center": [0., 0., 0.], "scale": [1., 1., 1.],
                             "anchors": [[0., 1., 1.]], "maximum_distance": 2.},
    }


def test_pairwise_reliability_abstains_for_zero_head_and_respects_support():
    features = np.array([[0.], [1.], [8.]])
    selected, predictions = pairwise_reliable_model_action(pairwise_model(), features, 0, [True, True, True])
    assert selected == 1 and predictions.shape == (3, 2)
    assert pairwise_reliable_model_action(pairwise_model(0., 0.), features, 0)[0] == 0
    assert pairwise_reliable_model_action(pairwise_model(), features, 0, [True, False, True])[0] == 0


def test_pairwise_effect_head_uses_effect_upper_bound_and_harm_veto():
    model = pairwise_model()
    model.update({
        "selection_objective": PAIRWISE_EFFECT_OBJECTIVE,
        "effect_transform_scale": .2, "minimum_human_gain": .05,
        "harm_probability_calibration": {"slope": 1., "intercept": 0.},
    })
    for member in model["ensemble"]:
        member["output_bias"] = [-1., -5.]
    features = np.array([[0.], [1.]])
    assert pairwise_reliable_model_action(model, features, 0)[0] == 1
    for member in model["ensemble"]:
        member["output_bias"] = [-1., 2.]
    assert pairwise_reliable_model_action(model, features, 0)[0] == 0
