from __future__ import annotations

import hashlib
import json

import numpy as np


def validate_response_clock(model, environment_dt, allow_temporal_transfer=False):
    training_dt = float(getattr(model, "training_dt", model.config.dt))
    if not np.isfinite(environment_dt) or environment_dt <= 0.0:
        raise ValueError("environment dt must be positive and finite")
    if not np.isclose(float(model.config.dt), environment_dt, rtol=0, atol=1e-7):
        raise ValueError("response inference dt and environment dt must agree")
    mismatch = not np.isclose(training_dt, environment_dt, rtol=0, atol=1e-7)
    if mismatch and not allow_temporal_transfer:
        raise ValueError("response decoder training dt differs from environment dt; inference-time velocity rescaling is not temporal retraining")
    return {"training_dt": training_dt, "environment_dt": float(environment_dt), "temporal_transfer": bool(mismatch)}


def navigation_outcome(reached, environment_timeout, steps, step_budget, collisions, wall_collisions):
    budget_exhausted = not bool(reached) and int(steps) >= int(step_budget)
    return {
        "success": float(bool(reached)),
        "collision_free_success": float(bool(reached) and not collisions and not wall_collisions),
        "timeout": float(bool(environment_timeout) or budget_exhausted),
        "environment_timeout": float(bool(environment_timeout)),
        "budget_exhausted": float(budget_exhausted),
    }


def initial_navigation_state(env, robot, static_obstacles):
    def state(agent):
        return [*agent.get_position(), *agent.get_velocity(), *agent.get_goal_position(),
                float(agent.radius), float(agent.v_pref)]
    payload = {
        "robot": state(robot), "humans": [state(human) for human in env.humans],
        "static_obstacles": np.asarray(static_obstacles).tolist(),
        "dt": float(env.time_step), "time_limit": float(env.time_limit),
    }
    return {
        "initial_scene_hash": hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "robot_kinematics": str(robot.kinematics), "robot_initial_heading": float(robot.theta),
        "environment_dt": float(env.time_step),
    }
