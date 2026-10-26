"""Observable short-prefix motion hypotheses and joint geometric checks."""

from __future__ import annotations

import numpy as np

from sage.experiments.socialgym2_execution_diagnostics import point_segment_distances


def observable_route_directions(
    robot_positions,
    robot_headings,
    local_targets,
    goals,
    minimum_target_distance: float = 0.25,
):
    positions = np.asarray(robot_positions, dtype=np.float64)
    headings = np.asarray(robot_headings, dtype=np.float64).reshape(-1)
    targets = np.asarray(local_targets, dtype=np.float64)
    final_goals = np.asarray(goals, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("route directions require robot xy positions")
    if headings.shape != (len(positions),) or targets.shape != positions.shape or final_goals.shape != positions.shape:
        raise ValueError("route directions require robot-aligned headings, targets, and goals")
    if not all(np.isfinite(value).all() for value in (positions, headings, targets, final_goals)):
        raise ValueError("route direction observations must be finite")
    if minimum_target_distance < 0 or not np.isfinite(minimum_target_distance):
        raise ValueError("minimum target distance must be finite and nonnegative")
    target_world = np.empty_like(targets)
    for robot_index in range(len(positions)):
        cosine, sine = np.cos(headings[robot_index]), np.sin(headings[robot_index])
        local = targets[robot_index]
        target_world[robot_index] = positions[robot_index] + np.asarray([
            cosine * local[0] - sine * local[1], sine * local[0] + cosine * local[1]
        ])
    target_delta = target_world - positions
    fallback_delta = final_goals - positions
    target_norm = np.linalg.norm(target_delta, axis=-1, keepdims=True)
    fallback_norm = np.linalg.norm(fallback_delta, axis=-1, keepdims=True)
    route_direction = np.where(
        target_norm >= minimum_target_distance,
        target_delta / np.maximum(target_norm, 1e-8),
        fallback_delta / np.maximum(fallback_norm, 1e-8),
    )
    return np.where(
        np.linalg.norm(route_direction, axis=-1, keepdims=True) > 1e-5,
        route_direction,
        np.stack([np.cos(headings), np.sin(headings)], axis=-1),
    )


def executable_candidate_prefix(
    robot_positions,
    robot_headings,
    robot_world_velocities,
    robot_angular_velocities,
    local_targets,
    goals,
    actions,
    horizon: int,
    dt: float,
    max_speed: float = 1.5,
    nominal_start_speed: float = 1.5,
    linear_acceleration: float = 5.0,
    angular_acceleration: float = 5.0,
    max_angular_speed: float = 5.0,
    turn_in_place_angle: float = np.deg2rad(30.0),
    measured_prefix_duration: float | None = None,
    integration_dt: float = 0.025,
):
    """Build an observable, conservative approximation of GoAlone/Halt.

    The first model interval is the measured current twist (or immediate
    stop for Halt), which is the only prefix directly supported by the
    current observation.  Later intervals use the currently exposed graph
    navigation target and bounded acceleration/turning; they are explicitly
    an extrapolation rather than hidden simulator rollout.
    """
    positions = np.asarray(robot_positions, dtype=np.float64)
    headings = np.asarray(robot_headings, dtype=np.float64).reshape(-1)
    world_velocities = np.asarray(robot_world_velocities, dtype=np.float64)
    angular_velocities = np.asarray(robot_angular_velocities, dtype=np.float64).reshape(-1)
    targets = np.asarray(local_targets, dtype=np.float64)
    final_goals = np.asarray(goals, dtype=np.float64)
    action_array = np.asarray(actions)
    robot_count = len(positions)
    if positions.shape != (robot_count, 2) or world_velocities.shape != positions.shape:
        raise ValueError("candidate prefix requires robot xy positions and world velocities")
    if headings.shape != (robot_count,) or angular_velocities.shape != (robot_count,):
        raise ValueError("candidate prefix requires one heading and angular velocity per robot")
    if targets.shape != positions.shape or final_goals.shape != positions.shape:
        raise ValueError("candidate prefix requires robot-aligned local targets and goals")
    if action_array.ndim != 2 or action_array.shape[1] != robot_count or not np.isin(action_array, [0, 1]).all():
        raise ValueError("joint actions must be robot-aligned GoAlone/Halt vectors")
    if horizon < 1 or dt <= 0 or not np.isfinite(dt):
        raise ValueError("candidate prefix requires a positive horizon and timestep")
    if measured_prefix_duration is None:
        measured_prefix_duration = float(dt)
    if measured_prefix_duration <= 0 or not np.isfinite(measured_prefix_duration):
        raise ValueError("measured prefix duration must be finite and positive")
    if integration_dt <= 0 or not np.isfinite(integration_dt):
        raise ValueError("integration_dt must be finite and positive")
    if not 0 < nominal_start_speed <= max_speed:
        raise ValueError("nominal start speed must be positive and no larger than max speed")
    values = [positions, headings, world_velocities, angular_velocities, targets, final_goals]
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("candidate prefix observations must be finite")
    route_direction = observable_route_directions(positions, headings, targets, final_goals)
    target_world = positions + route_direction * np.maximum(
        np.linalg.norm(targets, axis=-1, keepdims=True), 1.0
    )

    candidate_positions = np.empty((len(action_array), robot_count, horizon, 2), dtype=np.float32)
    candidate_velocities = np.empty_like(candidate_positions)
    for candidate_index, action in enumerate(action_array):
        for robot_index in range(robot_count):
            position = positions[robot_index].copy()
            velocity = world_velocities[robot_index].copy()
            heading = float(headings[robot_index])
            angular_velocity = float(angular_velocities[robot_index])
            initial_cosine, initial_sine = np.cos(heading), np.sin(heading)
            body_velocity = np.asarray([
                initial_cosine * velocity[0] + initial_sine * velocity[1],
                -initial_sine * velocity[0] + initial_cosine * velocity[1],
            ])
            cruise_speed = min(
                float(max_speed),
                max(float(nominal_start_speed), float(np.linalg.norm(velocity))),
            )
            initially_moving = float(np.linalg.norm(velocity)) > 0.05
            halted = bool(action[robot_index])
            for step in range(horizon):
                if halted:
                    velocity[:] = 0.0
                    angular_velocity = 0.0
                    candidate_positions[candidate_index, robot_index, step] = position
                    candidate_velocities[candidate_index, robot_index, step] = velocity
                    continue
                if initially_moving and (step + 1) * dt <= measured_prefix_duration + 1e-9:
                    # Match the measured deployed prefix instead of inventing
                    # the nominal graph-navigation speed at rest.
                    previous_position = position.copy()
                    cosine, sine = np.cos(heading), np.sin(heading)
                    if abs(angular_velocity) > 1e-8:
                        local_offset = np.asarray([
                            (np.sin(angular_velocity * dt) * body_velocity[0]
                             + (np.cos(angular_velocity * dt) - 1.0) * body_velocity[1]) / angular_velocity,
                            ((1.0 - np.cos(angular_velocity * dt)) * body_velocity[0]
                             + np.sin(angular_velocity * dt) * body_velocity[1]) / angular_velocity,
                        ])
                        position = position + np.asarray([
                            cosine * local_offset[0] - sine * local_offset[1],
                            sine * local_offset[0] + cosine * local_offset[1],
                        ])
                    else:
                        position = position + np.asarray([
                            cosine * body_velocity[0] - sine * body_velocity[1],
                            sine * body_velocity[0] + cosine * body_velocity[1],
                        ]) * dt
                    heading = heading + angular_velocity * dt
                    next_cosine, next_sine = np.cos(heading), np.sin(heading)
                    velocity = np.asarray([
                        next_cosine * body_velocity[0] - next_sine * body_velocity[1],
                        next_sine * body_velocity[0] + next_cosine * body_velocity[1],
                    ])
                    candidate_positions[candidate_index, robot_index, step] = position
                    candidate_velocities[candidate_index, robot_index, step] = (
                        position - previous_position
                    ) / dt
                    previous_position = position.copy()
                    continue
                previous_position = position.copy()
                integration_steps = max(1, int(np.ceil(dt / integration_dt)))
                sub_dt = dt / integration_steps
                for _ in range(integration_steps):
                    delta = target_world[robot_index] - position
                    if np.linalg.norm(delta) <= 1e-5:
                        delta = final_goals[robot_index] - position
                    desired_angle = (
                        float(np.arctan2(delta[1], delta[0]))
                        if np.linalg.norm(delta) > 1e-5 else heading
                    )
                    angle_error = float(np.arctan2(
                        np.sin(desired_angle - heading), np.cos(desired_angle - heading),
                    ))
                    desired_angular_velocity = np.clip(
                        angle_error / sub_dt, -max_angular_speed, max_angular_speed,
                    )
                    angular_velocity += np.clip(
                        desired_angular_velocity - angular_velocity,
                        -angular_acceleration * sub_dt,
                        angular_acceleration * sub_dt,
                    )
                    angular_velocity = float(np.clip(
                        angular_velocity, -max_angular_speed, max_angular_speed,
                    ))
                    heading += angular_velocity * sub_dt
                    current_speed = float(np.linalg.norm(velocity))
                    desired_speed = 0.0 if abs(angle_error) > turn_in_place_angle else cruise_speed
                    next_speed = current_speed + np.clip(
                        desired_speed - current_speed,
                        -linear_acceleration * sub_dt,
                        linear_acceleration * sub_dt,
                    )
                    velocity = next_speed * np.asarray([np.cos(heading), np.sin(heading)])
                    position = position + velocity * sub_dt
                candidate_positions[candidate_index, robot_index, step] = position
                candidate_velocities[candidate_index, robot_index, step] = (
                    position - previous_position
                ) / dt
    return candidate_positions, candidate_velocities


def constant_twist_prefix(robot_poses, body_velocities, actions, tick_count: int, dt: float):
    poses = np.asarray(robot_poses, dtype=np.float64)
    velocities = np.asarray(body_velocities, dtype=np.float64)
    actions = np.asarray(actions)
    if poses.ndim != 2 or poses.shape[1] != 3 or velocities.shape != poses.shape:
        raise ValueError("robot pose and body twist must have matching robot, xyz axes")
    if actions.ndim != 2 or actions.shape[1] != len(poses) or not np.isin(actions, [0, 1]).all():
        raise ValueError("joint actions must be robot-aligned GoAlone/Halt vectors")
    if tick_count < 1 or dt <= 0 or not np.isfinite(dt) or not np.isfinite(poses).all() or not np.isfinite(velocities).all():
        raise ValueError("motion hypotheses require finite observations and positive time")
    times = np.arange(1, tick_count + 1, dtype=np.float64) * dt
    moving = 1.0 - actions[..., None]
    angular_velocity = velocities[None, :, 2, None] * moving
    angles = poses[None, :, 2, None] + angular_velocity * times
    initial_angles = poses[None, :, 2, None]
    nonzero = np.abs(angular_velocity) > 1e-10
    denominator = np.where(nonzero, angular_velocity, 1.0)
    integrated_cosine = np.where(nonzero, (np.sin(angles) - np.sin(initial_angles)) / denominator, np.cos(initial_angles) * times)
    integrated_sine = np.where(nonzero, (np.cos(initial_angles) - np.cos(angles)) / denominator, np.sin(initial_angles) * times)
    longitudinal = velocities[None, :, 0, None] * moving
    lateral = velocities[None, :, 1, None] * moving
    offsets = np.stack([longitudinal * integrated_cosine - lateral * integrated_sine,
                        longitudinal * integrated_sine + lateral * integrated_cosine], axis=-1)
    positions = poses[None, :, None, :2] + offsets
    return positions, np.broadcast_to(angles, positions.shape[:-1]).copy()


def joint_prefix_clearances(
    initial_positions,
    candidate_positions,
    segments,
    robot_separation: float = 0.4,
    static_radius: float = 0.22,
    recovery_improvement: float = 0.01,
    hard_robot_separation: float | None = None,
    hard_static_radius: float | None = None,
):
    initial = np.asarray(initial_positions, dtype=np.float64)
    candidates = np.asarray(candidate_positions, dtype=np.float64)
    walls = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    if candidates.ndim != 4 or candidates.shape[-1] != 2 or initial.shape != candidates.shape[1::2]:
        raise ValueError("joint geometry requires candidate, robot, tick, xy trajectories")
    if not np.isfinite(candidates).all() or not np.isfinite(initial).all() or not np.isfinite(walls).all():
        raise ValueError("joint geometry must be finite")
    if robot_separation <= 0 or static_radius < 0 or recovery_improvement <= 0:
        raise ValueError("geometric thresholds must be nonnegative with positive separation")
    if hard_robot_separation is None:
        hard_robot_separation = robot_separation
    if hard_static_radius is None:
        hard_static_radius = static_radius
    if hard_robot_separation <= 0 or hard_static_radius < 0:
        raise ValueError("hard geometric thresholds must be positive/nonnegative")
    if hard_robot_separation > robot_separation or hard_static_radius > static_radius:
        raise ValueError("hard geometric thresholds cannot exceed robust thresholds")
    full = np.concatenate([np.broadcast_to(initial, (len(candidates),) + initial.shape)[:, :, None], candidates], axis=2)
    robot_minimum = np.full(len(candidates), np.inf)
    terminal_minimum = np.full(len(candidates), np.inf)
    initial_robot_minimum = np.inf
    pair_recoverable = np.ones(len(candidates), dtype=bool)
    for first_robot in range(initial.shape[0]):
        for second_robot in range(first_robot + 1, initial.shape[0]):
            relative = full[:, first_robot] - full[:, second_robot]
            start = relative[:, :-1]
            delta = np.diff(relative, axis=1)
            fraction = -np.sum(start * delta, axis=-1) / np.maximum(np.sum(delta**2, axis=-1), 1e-15)
            distance = np.linalg.norm(start + np.clip(fraction, 0.0, 1.0)[..., None] * delta, axis=-1)
            pair_minimum = distance.min(axis=1) - robot_separation
            pair_terminal = np.linalg.norm(relative[:, -1], axis=-1) - robot_separation
            pair_initial = float(np.linalg.norm(relative[0, 0]) - robot_separation)
            robot_minimum = np.minimum(robot_minimum, pair_minimum)
            terminal_minimum = np.minimum(terminal_minimum, pair_terminal)
            initial_robot_minimum = min(initial_robot_minimum, pair_initial)
            if pair_initial >= 0.0:
                pair_recoverable &= pair_minimum >= 0.0
            elif pair_initial + robot_separation - hard_robot_separation >= 0.0:
                pair_recoverable &= (
                    (pair_minimum >= pair_initial - 1e-6)
                    & (pair_terminal >= pair_initial - 1e-6)
                )
            else:
                pair_recoverable &= (
                    (pair_minimum >= pair_initial - 1e-6)
                    & (pair_terminal >= pair_initial + recovery_improvement)
                )
    static_minimum = np.full(len(candidates), np.inf)
    static_terminal = np.full(len(candidates), np.inf)
    initial_static = np.inf
    static_recoverable = np.ones(len(candidates), dtype=bool)
    if len(walls):
        initial_static = float(point_segment_distances(initial, walls).min()) - static_radius
        for candidate_index, trajectory in enumerate(full):
            sampled = np.concatenate([trajectory, (trajectory[:, :-1] + trajectory[:, 1:]) * 0.5], axis=1)
            distances = point_segment_distances(sampled.reshape(-1, 2), walls)
            motion_bound = np.linalg.norm(np.diff(trajectory, axis=1), axis=-1).max() / 4.0
            static_minimum[candidate_index] = float(distances.min()) - static_radius - motion_bound
            static_terminal[candidate_index] = float(
                point_segment_distances(trajectory[:, -1], walls).min()
            ) - static_radius
            if initial_static >= 0.0:
                static_recoverable[candidate_index] = static_minimum[candidate_index] >= 0.0
            elif initial_static + static_radius - hard_static_radius >= 0.0:
                static_recoverable[candidate_index] = (
                    static_minimum[candidate_index] >= initial_static - motion_bound - 1e-6
                    and static_terminal[candidate_index] >= initial_static - 1e-6
                )
            else:
                static_recoverable[candidate_index] = (
                    static_minimum[candidate_index] >= initial_static - motion_bound - 1e-6
                    and static_terminal[candidate_index] >= initial_static + recovery_improvement
                )
    feasible = pair_recoverable & static_recoverable
    hard_robot_clearance = robot_minimum + robot_separation - hard_robot_separation
    hard_static_clearance = static_minimum + static_radius - hard_static_radius
    return {
        "initial_robot_clearance": np.full(len(candidates), initial_robot_minimum),
        "robot_clearance": robot_minimum,
        "terminal_robot_clearance": terminal_minimum,
        "initial_static_clearance": np.full(len(candidates), initial_static),
        "static_clearance": static_minimum,
        "terminal_static_clearance": static_terminal,
        "hard_robot_clearance": hard_robot_clearance,
        "hard_static_clearance": hard_static_clearance,
        "hard_feasible": (hard_robot_clearance >= 0.0) & (hard_static_clearance >= 0.0),
        "recovering_initial_overlap": feasible & (
            (initial_robot_minimum < 0.0) | (initial_static < 0.0)
        ),
        "feasible": feasible,
    }


def add_observed_human_clearances(
    geometry,
    initial_robot_positions,
    candidate_positions,
    human_positions,
    human_velocities,
    dt: float,
    robot_human_separation: float = 0.5,
    hard_robot_human_separation: float = 0.4,
    recovery_improvement: float = 0.01,
):
    result = {key: np.asarray(value).copy() for key, value in geometry.items()}
    initial = np.asarray(initial_robot_positions, dtype=np.float64)
    candidates = np.asarray(candidate_positions, dtype=np.float64)
    humans = np.asarray(human_positions, dtype=np.float64).reshape(-1, 2)
    velocities = np.asarray(human_velocities, dtype=np.float64).reshape(-1, 2)
    if candidates.ndim != 4 or initial.shape != candidates.shape[1::2]:
        raise ValueError("observed-human geometry requires candidate, robot, tick, xy trajectories")
    if humans.shape != velocities.shape:
        raise ValueError("observed-human positions and velocities must align")
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("observed-human geometry requires a positive finite timestep")
    if not 0 < hard_robot_human_separation <= robot_human_separation:
        raise ValueError("hard human separation must be positive and no larger than robust separation")
    if recovery_improvement <= 0 or not all(
        np.isfinite(value).all() for value in (initial, candidates, humans, velocities)
    ):
        raise ValueError("observed-human geometry must be finite with positive recovery")
    count = len(candidates)
    minimum = np.full(count, np.inf)
    terminal = np.full(count, np.inf)
    initial_minimum = np.inf
    feasible = np.ones(count, dtype=bool)
    hard_feasible = np.ones(count, dtype=bool)
    if len(humans):
        robot_full = np.concatenate([
            np.broadcast_to(initial, (count,) + initial.shape)[:, :, None], candidates,
        ], axis=2)
        times = np.arange(candidates.shape[2] + 1, dtype=np.float64) * dt
        human_full = humans[:, None] + velocities[:, None] * times[None, :, None]
        for robot_index in range(initial.shape[0]):
            for human_index in range(len(humans)):
                relative = robot_full[:, robot_index] - human_full[human_index][None]
                start = relative[:, :-1]
                delta = np.diff(relative, axis=1)
                fraction = -np.sum(start * delta, axis=-1) / np.maximum(
                    np.sum(delta**2, axis=-1), 1e-15,
                )
                swept_distance = np.linalg.norm(
                    start + np.clip(fraction, 0.0, 1.0)[..., None] * delta,
                    axis=-1,
                ).min(axis=1)
                pair_minimum = swept_distance - robot_human_separation
                pair_terminal = np.linalg.norm(relative[:, -1], axis=-1) - robot_human_separation
                pair_initial = float(np.linalg.norm(relative[0, 0]) - robot_human_separation)
                hard_minimum = swept_distance - hard_robot_human_separation
                minimum = np.minimum(minimum, pair_minimum)
                terminal = np.minimum(terminal, pair_terminal)
                initial_minimum = min(initial_minimum, pair_initial)
                hard_feasible &= hard_minimum >= 0.0
                if pair_initial >= 0.0:
                    feasible &= pair_minimum >= 0.0
                elif pair_initial + robot_human_separation - hard_robot_human_separation >= 0.0:
                    feasible &= (
                        (pair_minimum >= pair_initial - 1e-6)
                        & (pair_terminal >= pair_initial - 1e-6)
                    )
                else:
                    feasible &= (
                        (pair_minimum >= pair_initial - 1e-6)
                        & (pair_terminal >= pair_initial + recovery_improvement)
                    )
    result.update({
        "initial_human_clearance": np.full(count, initial_minimum),
        "human_clearance": minimum,
        "terminal_human_clearance": terminal,
        "human_feasible": feasible,
        "hard_human_feasible": hard_feasible,
        "feasible": np.asarray(result["feasible"], dtype=bool) & feasible,
        "hard_feasible": np.asarray(result["hard_feasible"], dtype=bool) & hard_feasible,
    })
    return result


def stagnation_recovery_mask(
    geometry,
    actions,
    stagnant_robots,
    allow_hard_safe_approach: bool = False,
    require_robust_robot_clearance: bool = False,
    require_nonworsening_robot_clearance: bool = False,
    robot_clearance_floor: float | None = None,
):
    action_array = np.asarray(actions)
    stagnant = np.asarray(stagnant_robots, dtype=bool)
    count = len(np.asarray(geometry["feasible"]))
    if action_array.ndim != 2 or action_array.shape[0] != count:
        raise ValueError("joint actions must align with geometry candidates")
    if stagnant.shape != (action_array.shape[1],):
        raise ValueError("stagnation state must align with robot actions")
    if not stagnant.any():
        return np.zeros(count, dtype=bool)
    recovery = np.asarray(geometry["hard_feasible"], dtype=bool).copy()
    recovery &= np.any(action_array[:, stagnant] == 0, axis=1)
    if require_robust_robot_clearance and "robot_clearance" in geometry:
        recovery &= np.asarray(geometry["robot_clearance"]) >= 0.0
    if require_nonworsening_robot_clearance:
        if not {
            "initial_robot_clearance",
            "terminal_robot_clearance",
        }.issubset(geometry):
            raise ValueError("robot nonworsening recovery requires initial and terminal clearance")
        recovery &= np.asarray(geometry["terminal_robot_clearance"]) >= (
            np.asarray(geometry["initial_robot_clearance"]) - 1e-6
        )
    if robot_clearance_floor is not None:
        if not np.isfinite(robot_clearance_floor):
            raise ValueError("robot clearance floor must be finite")
        if "robot_clearance" not in geometry:
            raise ValueError("robot clearance floor requires robot clearance")
        recovery &= np.asarray(geometry["robot_clearance"]) >= float(robot_clearance_floor)
    if not allow_hard_safe_approach:
        for initial_key, terminal_key in (
            ("initial_robot_clearance", "terminal_robot_clearance"),
            ("initial_static_clearance", "terminal_static_clearance"),
            ("initial_human_clearance", "terminal_human_clearance"),
        ):
            if initial_key in geometry and terminal_key in geometry:
                recovery &= np.asarray(geometry[terminal_key]) >= np.asarray(geometry[initial_key]) - 1e-6
    return recovery


def select_with_joint_geometry(
    costs,
    human_safe,
    human_violation,
    geometry,
    actions=None,
    stagnant_robots=None,
    allow_hard_safe_approach: bool = False,
    require_robust_robot_clearance: bool = False,
    require_nonworsening_robot_clearance: bool = False,
    robot_clearance_floor: float | None = None,
    recovery_safety_first: bool = False,
):
    costs = np.asarray(costs, dtype=np.float64)
    human_safe = np.asarray(human_safe, dtype=bool)
    violation = np.asarray(human_violation, dtype=np.float64)
    feasible = np.asarray(geometry["feasible"], dtype=bool)
    recovery = np.zeros_like(feasible)
    if actions is not None and stagnant_robots is not None and "hard_feasible" in geometry:
        recovery = stagnation_recovery_mask(
            geometry,
            actions,
            stagnant_robots,
            allow_hard_safe_approach=allow_hard_safe_approach,
            require_robust_robot_clearance=require_robust_robot_clearance,
            require_nonworsening_robot_clearance=require_nonworsening_robot_clearance,
            robot_clearance_floor=robot_clearance_floor,
        )
        feasible = feasible | recovery
    if costs.ndim != 1 or not (costs.shape == human_safe.shape == violation.shape == feasible.shape):
        raise ValueError("geometry and response scores must share candidate identities")
    finite = np.isfinite(costs) & np.isfinite(violation)
    if not finite.any():
        raise ValueError("no finite response scores for geometric fallback")
    if recovery.any():
        release = np.any(
            np.asarray(actions)[:, np.asarray(stagnant_robots, dtype=bool)] == 0,
            axis=1,
        )
        recovery_admitted = finite & human_safe & feasible & release
        if recovery_admitted.any():
            if not recovery_safety_first:
                return int(np.argmin(np.where(recovery_admitted, costs, np.inf))), False
            indices = np.flatnonzero(recovery_admitted)
            cost_choice = int(indices[np.argmin(costs[indices])])
            robot_clearance_all = np.asarray(
                geometry.get("robot_clearance", np.full(len(costs), np.inf)),
                dtype=np.float64,
            )
            if np.isfinite(robot_clearance_all[cost_choice]) and robot_clearance_all[cost_choice] >= 0.0:
                return cost_choice, False
            robot_clearance = np.nan_to_num(
                robot_clearance_all[indices],
                nan=-np.inf,
                posinf=np.finfo(np.float64).max,
                neginf=-np.finfo(np.float64).max,
            )
            terminal_robot_clearance = np.nan_to_num(
                np.asarray(geometry.get("terminal_robot_clearance", robot_clearance))[indices],
                nan=-np.inf,
                posinf=np.finfo(np.float64).max,
                neginf=-np.finfo(np.float64).max,
            )
            human_clearance = np.nan_to_num(
                np.asarray(geometry.get("human_clearance", np.full(len(costs), np.inf)))[indices],
                nan=-np.inf,
                posinf=np.finfo(np.float64).max,
                neginf=-np.finfo(np.float64).max,
            )
            order = np.lexsort(
                (
                    indices,
                    costs[indices],
                    -human_clearance,
                    -terminal_robot_clearance,
                    -robot_clearance,
                )
            )
            return int(indices[order[0]]), False
    admitted = finite & human_safe & feasible
    if admitted.any():
        return int(np.argmin(np.where(admitted, costs, np.inf))), False
    geometrically_feasible = finite & feasible
    if geometrically_feasible.any():
        indices = np.flatnonzero(geometrically_feasible)
        order = np.lexsort((indices, costs[indices], violation[indices]))
        return int(indices[order[0]]), True
    if actions is not None and "hard_feasible" in geometry:
        action_array = np.asarray(actions)
        if action_array.ndim != 2 or action_array.shape[0] != len(costs):
            raise ValueError("joint actions must align with geometry candidates")
        halt = np.all(action_array == 1, axis=1)
        hard_safe_halt = finite & human_safe & halt & np.asarray(geometry["hard_feasible"], dtype=bool)
        if hard_safe_halt.any():
            indices = np.flatnonzero(hard_safe_halt)
            return int(indices[np.argmin(costs[indices])]), True
    indices = np.flatnonzero(finite)
    bottleneck = np.minimum(geometry["robot_clearance"], geometry["static_clearance"])
    terminal = np.minimum(
        geometry["terminal_robot_clearance"],
        geometry.get("terminal_static_clearance", geometry["static_clearance"]),
    )
    if "human_clearance" in geometry:
        bottleneck = np.minimum(bottleneck, geometry["human_clearance"])
        terminal = np.minimum(terminal, geometry["terminal_human_clearance"])
    order = np.lexsort((indices, costs[indices], violation[indices], -terminal[indices], -bottleneck[indices]))
    return int(indices[order[0]]), True


def stagnation_adjusted_costs(costs, actions, stagnant_robots, penalty: float):
    values = np.asarray(costs, dtype=np.float64)
    action_array = np.asarray(actions)
    stagnant = np.asarray(stagnant_robots, dtype=bool)
    if values.ndim != 1 or action_array.ndim != 2 or action_array.shape[0] != len(values):
        raise ValueError("stagnation costs and joint actions must align")
    if stagnant.shape != (action_array.shape[1],) or not np.isin(action_array, [0, 1]).all():
        raise ValueError("stagnation state must be robot-aligned with GoAlone/Halt actions")
    if penalty < 0 or not np.isfinite(penalty):
        raise ValueError("stagnation penalty must be finite and nonnegative")
    if not stagnant.any() or penalty == 0:
        return values.copy()
    return values + penalty * action_array[:, stagnant].mean(axis=1)
