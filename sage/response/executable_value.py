from __future__ import annotations

import numpy as np


FEATURE_VERSION = "observable-prefix-response-value-v3-temporal-geometry"
COMPONENTS = (
    "weighted_task", "weighted_safety", "weighted_human", "weighted_risk",
    "observable_penalty", "weighted_belief_adjustment", "mode_mean_spread",
    "action_response_delta_mse",
)


def prefix_value_feature_indices(names):
    bases = [name for name in names if not name.startswith("signed_log_") and "_scene" not in name]
    selected = {name for name in bases if name.startswith(("prefix_", "linear_", "response_prefix_")) or name in
                ("closest_human_closing_speed", "maximum_human_closing_speed", "mean_human_closing_speed")}
    return [index for index, name in enumerate(names) if name in selected or
            (name.startswith("signed_log_") and name.removeprefix("signed_log_") in selected) or
            ("_scene" in name and name.rsplit("_scene", 1)[0] in selected)]


def response_risk_admissible(diagnostics, parent):
    risk = np.asarray(diagnostics["weighted_risk"], dtype=float)
    safety = np.asarray(diagnostics["weighted_safety"], dtype=float)
    if risk.shape != safety.shape or risk.ndim != 1 or not np.isfinite(risk).all() or not np.isfinite(safety).all():
        raise ValueError("response-risk guard requires aligned finite original cost components")
    if not 0 <= parent < len(risk):
        raise ValueError("response-risk guard requires the original supported selection")
    return (risk <= risk[parent] + 1e-8) & (safety <= safety[parent] + 1e-8)


def executable_value_features(*, candidates, headings, robot_position, robot_velocity,
                              robot_goal, robot_heading, human_positions, human_velocities,
                              human_radii, robot_radius, beliefs, means, log_stds, diagnostics,
                              dt, execute_steps):
    candidates = np.asarray(candidates, dtype=np.float64)
    humans = np.asarray(human_positions, dtype=np.float64)
    velocities = np.asarray(human_velocities, dtype=np.float64)
    weights = np.asarray(beliefs, dtype=np.float64)
    count, horizon = candidates.shape[:2]
    if candidates.shape != (count, horizon, 4) or not 1 <= execute_steps <= horizon:
        raise ValueError("value features require candidate trajectories and a valid executable prefix")
    if weights.ndim != 2 or weights.shape[0] != len(humans) or np.any(weights < 0) or not np.allclose(weights.sum(-1), 1., atol=1e-5):
        raise ValueError("value features require normalized observable decision beliefs")
    means = np.asarray(means, dtype=np.float64)
    log_stds = np.asarray(log_stds, dtype=np.float64)
    if means.shape != (len(humans), count, weights.shape[1], horizon, 2) or log_stds.shape != means.shape:
        raise ValueError("value features require candidate-conditioned response distributions")
    if not np.isfinite(dt) or dt <= 0 or not all(np.isfinite(values).all() for values in
            (candidates, humans, velocities, weights, means, log_stds, robot_position, robot_velocity, robot_goal)):
        raise ValueError("executable value inputs and timestep must be finite, with positive timestep")
    direction = np.asarray(robot_goal) - robot_position
    distance = float(np.linalg.norm(direction))
    direction = direction / max(distance, 1e-8)
    lateral = np.array([-direction[1], direction[0]])
    prefix_velocity = candidates[:, :execute_steps, 2:].mean(1)
    features, names = [], []

    def add(name, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (count,):
            raise ValueError(f"candidate feature {name} is not candidate-aligned")
        features.append(values)
        names.append(name)

    add("prefix_forward_speed", prefix_velocity @ direction)
    add("prefix_lateral_speed", prefix_velocity @ lateral)
    add("prefix_speed", np.linalg.norm(prefix_velocity, axis=-1))
    add("prefix_acceleration", np.linalg.norm(prefix_velocity - robot_velocity, axis=-1) / (dt * execute_steps))
    if headings is None:
        angle = np.arctan2(prefix_velocity[:, 1], prefix_velocity[:, 0])
    else:
        angle = np.asarray(headings)[:, execute_steps - 1]
    turn = (angle - robot_heading + np.pi) % (2 * np.pi) - np.pi
    add("prefix_turn", turn)
    add("prefix_turn_abs", np.abs(turn))
    goal_heading = np.arctan2(direction[1], direction[0])
    add("prefix_goal_heading_cos", np.cos(angle - goal_heading))
    add("prefix_goal_heading_sin", np.sin(angle - goal_heading))
    add("prefix_goal_alignment_gain", np.cos(angle - goal_heading) - np.cos(robot_heading - goal_heading))
    terminal = candidates[:, execute_steps - 1, :2]
    add("prefix_goal_progress", distance - np.linalg.norm(terminal - robot_goal, axis=-1))
    for name in COMPONENTS:
        add(name, diagnostics[name])
    relative_humans = humans - robot_position
    relative_velocity = velocities[None] - prefix_velocity[:, None]
    squared_speed = np.maximum(np.square(relative_velocity).sum(-1), 1e-8)
    closest_time = np.clip(-np.sum(relative_humans[None] * relative_velocity, axis=-1) / squared_speed, 0., horizon * dt)
    closest = relative_humans[None] + closest_time[..., None] * relative_velocity
    surface = np.linalg.norm(closest, axis=-1) - (np.asarray(human_radii) + robot_radius)[None]
    add("linear_closest_surface", surface.min(1))
    add("linear_mean_surface", surface.mean(1))
    closest_weights = np.isclose(surface, surface.min(1, keepdims=True), atol=1e-8, rtol=0).astype(float)
    closest_weights /= closest_weights.sum(1, keepdims=True)
    add("linear_closest_time", (closest_time * closest_weights).sum(1))
    current_distance = np.maximum(np.linalg.norm(relative_humans, axis=-1), 1e-6)
    closing_speed = -np.sum(relative_humans[None] * relative_velocity, axis=-1) / current_distance[None]
    add("closest_human_closing_speed", (closing_speed * closest_weights).sum(1))
    add("maximum_human_closing_speed", closing_speed.max(1))
    add("mean_human_closing_speed", closing_speed.mean(1))
    for lookahead in (.25, .5, 1., 2., 3.):
        separation = relative_humans[None] + relative_velocity * lookahead
        surface_at_time = np.linalg.norm(separation, axis=-1) - (np.asarray(human_radii) + robot_radius)[None]
        add(f"linear_surface_at_{lookahead}", surface_at_time.min(1))
        exposure = np.exp(-np.maximum(surface_at_time, 0.) / .5)
        add(f"linear_exposure_at_{lookahead}", exposure.mean(1))
    mean_paths = humans[:, None, None, None] + np.cumsum(means * dt, axis=-2)
    clearance = np.linalg.norm(mean_paths - candidates[None, :, None, :, :2], axis=-1)
    clearance -= (np.asarray(human_radii) + robot_radius)[:, None, None, None]
    for label, steps in (("prefix", execute_steps), ("one_second", min(horizon, max(execute_steps, round(1. / dt)))), ("horizon", horizon)):
        closest = clearance[..., :steps].min(-1)
        expected = (closest * weights[:, None]).sum(-1)
        add(f"response_{label}_closest", expected.min(0))
        add(f"response_{label}_mean", expected.mean(0))
        add(f"response_{label}_mode_range", np.ptp(closest, axis=-1).max(0))
    closest_step = clearance.argmin(-1).astype(np.float64) * dt
    expected_time = (closest_step * weights[:, None]).sum(-1)
    expected_clearance = (clearance.min(-1) * weights[:, None]).sum(-1)
    response_weights = np.isclose(expected_clearance, expected_clearance.min(0, keepdims=True), atol=1e-8, rtol=0).astype(float)
    response_weights /= response_weights.sum(0, keepdims=True)
    add("response_closest_time", (expected_time * response_weights).sum(0))
    for step in sorted({execute_steps - 1, min(horizon - 1, max(0, round(1. / dt) - 1)), horizon - 1}):
        exposure = np.exp(-np.maximum(clearance[..., step], 0.) / .5)
        add(f"response_exposure_step{step}", (exposure * weights[:, None]).sum(-1).mean(0))
    variance = np.exp(2 * np.clip(log_stds, -20, 5)).mean((-1, -2))
    add("response_variance", (variance * weights[:, None]).sum(-1).mean(0))
    basic = np.stack(features, axis=-1)
    nearest_weights = np.isclose(current_distance, current_distance.min(), atol=1e-8, rtol=0).astype(float)
    nearest_weights /= nearest_weights.sum()
    nearby_velocity = (velocities * nearest_weights[:, None]).sum(0)
    scene = np.asarray([
        np.clip(distance, 0., 15.), np.linalg.norm(robot_velocity),
        np.clip(np.linalg.norm(relative_humans, axis=-1).min(), 0., 15.),
        np.clip((relative_humans @ direction).min(), -15., 15.),
        -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=-1).mean(),
        np.cos(robot_heading - goal_heading),
        float(nearby_velocity @ direction),
        float(nearby_velocity @ lateral),
    ])
    expanded = np.concatenate([basic, np.sign(basic) * np.log1p(np.abs(basic)),
                               (basic[:, :, None] * scene[None, None]).reshape(count, -1)], axis=-1)
    feature_names = names + [f"signed_log_{name}" for name in names]
    feature_names += [f"{name}_scene{index}" for name in names for index in range(len(scene))]
    if not np.isfinite(expanded).all():
        raise ValueError("executable value features must contain only finite observable values")
    return expanded, feature_names


def value_representation(features, model):
    if model.get("feature_indices") is not None:
        features = features[:, model["feature_indices"]]
    projection = model.get("projection")
    if projection is None:
        return features
    normalized = (features - np.asarray(projection["center"])) / np.asarray(projection["scale"])
    nonlinear = np.cos(normalized @ np.asarray(projection["frequencies"]) + np.asarray(projection["phases"]))
    return np.concatenate([normalized, nonlinear * np.sqrt(2. / nonlinear.shape[1])], axis=-1)


def fit_paired_value(groups, regularization, nonlinear_bandwidth=None, feature_indices=None):
    if regularization <= 0 or not groups:
        raise ValueError("paired value fitting requires data and positive regularization")
    case_counts = {case: sum(group["case"] == case for group in groups) for case in {group["case"] for group in groups}}
    model = {"feature_version": FEATURE_VERSION, "regularization": float(regularization),
             "projection": None, "feature_indices": feature_indices}
    if nonlinear_bandwidth is not None:
        if nonlinear_bandwidth <= 0:
            raise ValueError("nonlinear feature bandwidth must be positive")
        all_features = np.concatenate([group["features"] for group in groups if group.get("rerank_allowed", True)])
        if feature_indices is not None:
            all_features = all_features[:, feature_indices]
        feature_weights = np.concatenate([
            np.full(len(group["features"]), 1. / (case_counts[group["case"]] * len(group["features"])))
            for group in groups if group.get("rerank_allowed", True)
        ])
        feature_weights /= feature_weights.sum()
        center = (all_features * feature_weights[:, None]).sum(0)
        scale = np.sqrt((np.square(all_features - center) * feature_weights[:, None]).sum(0))
        scale = np.maximum(scale, 1e-3)
        generator = np.random.default_rng(931527)
        frequencies = generator.normal(size=(all_features.shape[1], 128)) / (nonlinear_bandwidth * np.sqrt(all_features.shape[1]))
        model["projection"] = {"center": center.tolist(), "scale": scale.tolist(),
                               "frequencies": frequencies.tolist(), "phases": generator.uniform(0., 2 * np.pi, 128).tolist(),
                               "bandwidth": float(nonlinear_bandwidth)}
    differences, targets, weights = [], [], []
    for group in groups:
        if not group.get("rerank_allowed", True):
            continue
        reference = group["parent"]
        features = value_representation(group["features"], model)
        for index in range(len(group["features"])):
            if index == reference:
                continue
            differences.append(features[index] - features[reference])
            targets.append(group["targets"][index] - group["targets"][reference])
            weights.append(1. / (case_counts[group["case"]] * max(len(group["features"]) - 1, 1)))
    if not differences:
        raise ValueError("paired value fitting needs distinct alternative prefixes")
    design, targets = np.asarray(differences), np.asarray(targets)
    weights = np.asarray(weights) / np.sum(weights)
    scale = np.sqrt((np.square(design) * weights[:, None]).sum(0))
    scale = np.maximum(scale, 1e-4)
    design = design / scale
    gram = design.T @ (design * weights[:, None])
    rhs = design.T @ (targets * weights[:, None])
    coefficients = np.linalg.solve(gram + regularization * np.eye(gram.shape[0]), rhs)
    model.update({"scale": scale.tolist(), "coefficients": coefficients.tolist()})
    return model


def value_action(model, features, parent, minimum_gain=.25, admissible_mask=None):
    if model["feature_version"] != FEATURE_VERSION or minimum_gain < 0:
        raise ValueError("unsupported value feature contract or action gain threshold")
    if model.get("selection_objective") in {
        "pairwise-reliability-classifier-v1", "pairwise-reliability-effect-v1",
    }:
        from sage.response.reliable_executable_value import pairwise_reliable_model_action

        return pairwise_reliable_model_action(model, features, parent, admissible_mask)
    if model.get("ensemble") is not None:
        members = []
        for member in model["ensemble"]:
            represented = value_representation(features, member)
            relative = (represented - represented[parent]) / np.asarray(member["scale"])
            members.append(relative @ np.asarray(member["coefficients"]))
        members = np.asarray(members)
        if len(members) < 2:
            raise ValueError("value uncertainty requires at least two independently resampled scene fits")
        predictions = members.mean(0)
        upper = predictions + float(model["uncertainty_multiplier"]) * members.std(0, ddof=1)
    else:
        represented = value_representation(features, model)
        relative = (represented - represented[parent]) / np.asarray(model["scale"])
        predictions = relative @ np.asarray(model["coefficients"])
        upper = predictions
    if model.get("selection_objective") == "safe-completion-lexicographic-v1":
        from sage.response.reliable_executable_value import reliable_model_action

        selected = reliable_model_action(model, features, parent, upper, admissible_mask)
        return selected, predictions
    if predictions.shape != (len(features), 3) or not np.isfinite(predictions).all():
        raise ValueError("executable value predictions must be candidate-aligned finite time/failure/collision differences")
    if not np.isfinite(upper).all():
        raise ValueError("value uncertainty upper estimates must be finite")
    admissible = (upper[:, 1] <= 1e-10) & (upper[:, 2] <= 1e-10)
    if admissible_mask is not None:
        supplied = np.asarray(admissible_mask, dtype=bool)
        if supplied.shape != admissible.shape or not supplied[parent]:
            raise ValueError("value response-risk support must preserve the parent action")
        admissible &= supplied
    admissible[parent] = True
    scores = np.where(admissible, upper[:, 0], np.inf)
    selected = int(np.argmin(scores))
    if scores[selected] >= -minimum_gain:
        selected = int(parent)
    return selected, predictions
