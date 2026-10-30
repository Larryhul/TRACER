from __future__ import annotations

import numpy as np


RELIABILITY_OBJECTIVE = "safe-completion-lexicographic-v1"
PAIRWISE_RELIABILITY_OBJECTIVE = "pairwise-reliability-classifier-v1"
PAIRWISE_EFFECT_OBJECTIVE = "pairwise-reliability-effect-v1"
RELIABILITY_TARGETS = (
    "safe_failure", "collision", "wall_collision", "task_failure", "timeout",
    "stagnation_time", "goal_shortfall", "human_velocity_variation", "elapsed_time",
)
RELIABILITY_TIERS = ((0, 1, 2), (3, 4, 5), (6, 7), (8,))
RELIABILITY_VETO_INDICES = (0, 1, 2, 3, 4, 5)
RELIABILITY_IMPROVEMENT_TIERS = ((6, 7), (8,))
DEFAULT_MINIMUM_GAINS = (.02, .01, .01, .02, .02, .25, .025, .05, .25)


def _sigmoid(values):
    values = np.asarray(values, dtype=float)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1. / (1. + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1. + exponential)
    return result


def _silu(values):
    return values * _sigmoid(values)


def pairwise_reliability_features(features, parent, basic_feature_count):
    features = np.asarray(features, dtype=float)
    if (features.ndim != 2 or not 0 <= parent < len(features)
            or not isinstance(basic_feature_count, int) or basic_feature_count < 1
            or basic_feature_count > features.shape[1] or not np.isfinite(features).all()):
        raise ValueError("pairwise reliability requires finite candidate features and a valid reference")
    basic = features[:, :basic_feature_count]
    reference = np.broadcast_to(basic[parent], basic.shape)
    return np.concatenate([reference, basic, basic - reference], axis=1)


def _pairwise_member_logits(member, pair_features):
    values = (pair_features - np.asarray(member["center"])) / np.asarray(member["scale"])
    values = values @ np.asarray(member["input_weight"]).T + np.asarray(member["input_bias"])
    mean = values.mean(axis=1, keepdims=True)
    variance = np.square(values - mean).mean(axis=1, keepdims=True)
    values = (values - mean) / np.sqrt(variance + float(member["layer_norm_epsilon"]))
    values = values * np.asarray(member["layer_norm_weight"]) + np.asarray(member["layer_norm_bias"])
    values = _silu(values)
    values = _silu(values @ np.asarray(member["hidden_weight"]).T + np.asarray(member["hidden_bias"]))
    return values @ np.asarray(member["output_weight"]).T + np.asarray(member["output_bias"])


def _pairwise_member_probability(member, pair_features):
    return _sigmoid(_pairwise_member_logits(member, pair_features))


def _calibrate_probability(probability, calibration):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1. - 1e-6)
    logits = np.log(probability / (1. - probability))
    return _sigmoid(float(calibration["slope"]) * logits + float(calibration["intercept"]))


def pairwise_reliability_predictions(model, features, parent):
    if model.get("selection_objective") != PAIRWISE_RELIABILITY_OBJECTIVE:
        raise ValueError("unsupported pairwise reliability objective")
    members = model.get("ensemble", [])
    if len(members) < 8 or float(model.get("uncertainty_multiplier", 0.)) <= 0:
        raise ValueError("pairwise reliability deployment requires at least eight uncertainty members")
    pair_features = pairwise_reliability_features(features, parent, model["basic_feature_count"])
    probabilities = np.asarray([
        _pairwise_member_probability(member, pair_features) for member in members
    ])
    if probabilities.shape != (len(members), len(features), 2) or not np.isfinite(probabilities).all():
        raise ValueError("pairwise reliability members must return finite human/harm probabilities")
    multiplier = float(model["uncertainty_multiplier"])
    mean = probabilities.mean(axis=0)
    deviation = probabilities.std(axis=0, ddof=1)
    lower_human = np.clip(mean[:, 0] - multiplier * deviation[:, 0], 0., 1.)
    upper_harm = np.clip(mean[:, 1] + multiplier * deviation[:, 1], 0., 1.)
    calibration = model["probability_calibration"]
    calibrated_mean = np.column_stack([
        _calibrate_probability(mean[:, index], calibration[index]) for index in range(2)
    ])
    return pair_features, calibrated_mean, _calibrate_probability(
        lower_human, calibration[0]
    ), _calibrate_probability(upper_harm, calibration[1])


def pairwise_effect_predictions(model, features, parent):
    if model.get("selection_objective") != PAIRWISE_EFFECT_OBJECTIVE:
        raise ValueError("unsupported pairwise effect objective")
    members = model.get("ensemble", [])
    if len(members) < 8 or float(model.get("uncertainty_multiplier", 0.)) <= 0:
        raise ValueError("pairwise effect deployment requires at least eight uncertainty members")
    pair_features = pairwise_reliability_features(features, parent, model["basic_feature_count"])
    raw = np.asarray([_pairwise_member_logits(member, pair_features) for member in members])
    if raw.shape != (len(members), len(features), 2) or not np.isfinite(raw).all():
        raise ValueError("pairwise effect members must return finite effect/harm outputs")
    effect_scale = float(model["effect_transform_scale"])
    effects = effect_scale * np.sinh(raw[:, :, 0])
    harm_probability = _sigmoid(raw[:, :, 1])
    multiplier = float(model["uncertainty_multiplier"])
    mean_effect = effects.mean(axis=0)
    upper_effect = mean_effect + multiplier * effects.std(axis=0, ddof=1)
    mean_harm = harm_probability.mean(axis=0)
    upper_harm = np.clip(
        mean_harm + multiplier * harm_probability.std(axis=0, ddof=1), 0., 1.
    )
    calibration = model["harm_probability_calibration"]
    return (
        pair_features,
        np.column_stack([mean_effect, _calibrate_probability(mean_harm, calibration)]),
        upper_effect,
        _calibrate_probability(upper_harm, calibration),
    )


def pairwise_training_support(model, pair_features):
    domain = model["training_support"]
    normalized = (pair_features - np.asarray(domain["center"])) / np.asarray(domain["scale"])
    anchors = np.asarray(domain["anchors"])
    if (anchors.ndim != 2 or anchors.shape[1] != normalized.shape[1]
            or not np.isfinite(normalized).all() or not np.isfinite(anchors).all()):
        raise ValueError("pairwise reliability training support has an invalid feature contract")
    distances = np.sqrt(np.square(normalized[:, None] - anchors[None]).mean(axis=-1)).min(axis=1)
    return distances <= float(domain["maximum_distance"]), distances


def pairwise_reliable_model_action(model, features, parent, admissible_mask=None):
    if model.get("selection_objective") == PAIRWISE_EFFECT_OBJECTIVE:
        pair_features, predictions, upper_effect, upper_harm = pairwise_effect_predictions(
            model, features, parent
        )
        eligible_human = upper_effect <= -float(model["minimum_human_gain"])
        human_score = upper_effect
    else:
        pair_features, predictions, lower_human, upper_harm = pairwise_reliability_predictions(
            model, features, parent
        )
        eligible_human = lower_human >= float(model["human_score_threshold"])
        human_score = -lower_human
    support, _ = pairwise_training_support(model, pair_features)
    if admissible_mask is not None:
        supplied = np.asarray(admissible_mask, dtype=bool)
        if supplied.shape != support.shape or not supplied[parent]:
            raise ValueError("pairwise reliability support must preserve the frozen reference")
        support &= supplied
    eligible = (
        support
        & eligible_human
        & (upper_harm <= float(model["harm_score_threshold"]))
    )
    eligible[parent] = False
    alternatives = np.flatnonzero(eligible)
    if not len(alternatives):
        return int(parent), predictions
    selected = min(
        alternatives,
        key=lambda index: (human_score[index], upper_harm[index], index),
    )
    return int(selected), predictions


def reliability_targets(branch, robot_radius):
    if not branch.get("outcome_history_restored"):
        raise ValueError("reliability targets require restored full-episode outcomes, not legacy or missing telemetry")
    values = np.asarray([
        1. - branch["collision_free_success"], branch["collision"], branch["wall_collision"],
        1. - branch["success"], branch["timeout"], branch["stagnation_time"],
        max(0., branch["terminal_goal_distance"] - robot_radius),
        branch["human_velocity_variation"], branch["elapsed_time"],
    ], dtype=float)
    if not np.isfinite(values).all() or (values < -1e-8).any() or not np.isin(values[:5], [0., 1.]).all():
        raise ValueError("reliability outcomes must be finite, nonnegative, with binary safety/failure labels")
    return np.maximum(values, 0.)


def reliable_action(upper, parent, minimum_gains, admissible_mask=None):
    upper = np.asarray(upper, dtype=float)
    gains = np.asarray(minimum_gains, dtype=float)
    if (upper.ndim != 2 or upper.shape[1] != len(RELIABILITY_TARGETS)
            or not 0 <= parent < len(upper) or not np.isfinite(upper).all()
            or gains.shape != (len(RELIABILITY_TARGETS),) or not np.isfinite(gains).all()
            or (gains <= 0).any() or not np.allclose(upper[parent], 0., atol=1e-10, rtol=0)):
        raise ValueError("reliable action selection requires finite reference-relative outcomes and positive gains")
    support = np.ones(len(upper), dtype=bool) if admissible_mask is None else np.asarray(admissible_mask, dtype=bool).copy()
    if support.shape != (len(upper),) or not support[parent]:
        raise ValueError("reliable action support must preserve the frozen reference")
    support &= (upper[:, RELIABILITY_VETO_INDICES] <= 1e-10).all(axis=1)
    for tier in RELIABILITY_IMPROVEMENT_TIERS:
        support &= (upper[:, tier] <= 1e-10).all(axis=1)
        improving = support & (upper[:, tier] < -gains[list(tier)]).any(axis=1)
        alternatives = np.flatnonzero(improving)
        if len(alternatives):
            return int(min(alternatives, key=lambda index: tuple(upper[index, list(tier)] / gains[list(tier)])))
    return int(parent)


def reliable_model_action(model, features, parent, upper, admissible_mask=None):
    if tuple(model.get("target_names", ())) != RELIABILITY_TARGETS:
        raise ValueError("reliable model target contract differs from training")
    if len(model.get("ensemble", [])) < 8 or model.get("uncertainty_multiplier", 0.) <= 0:
        raise ValueError("reliable deployment requires scene-resampled uncertainty, not a single point estimate")
    domain = model["training_support"]
    selected_features = np.asarray(features, dtype=float)[:, domain["feature_indices"]]
    normalized = (selected_features - np.asarray(domain["center"])) / np.asarray(domain["scale"])
    anchors = np.asarray(domain["anchors"])
    distances = np.sqrt(np.square(normalized[:, None] - anchors[None]).mean(axis=-1)).min(axis=1)
    support = distances <= domain["maximum_distance"]
    if admissible_mask is not None:
        supplied = np.asarray(admissible_mask, dtype=bool)
        if supplied.shape != support.shape or not supplied[parent]:
            raise ValueError("reliable action support must preserve the frozen reference")
        support &= supplied
    support[parent] = True
    if distances[parent] > domain["maximum_distance"]:
        return int(parent)
    return reliable_action(upper, parent, model["minimum_gains"], support)
