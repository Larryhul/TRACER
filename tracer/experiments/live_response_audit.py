from __future__ import annotations

import copy
from pathlib import Path

import numpy as np


def choose_audit_branches(count, selected, budget, seed):
    if count < 1 or budget < 1 or not 0 <= selected < count:
        raise ValueError("response audit needs positive candidate count/budget and a valid selection")
    remaining = np.delete(np.arange(count), selected)
    generator = np.random.default_rng(seed)
    additional = generator.choice(remaining, size=min(budget - 1, len(remaining)), replace=False)
    return np.asarray([selected, *additional], dtype=int)


def _observable_snapshot(env):
    agents = [env.robot, *env.humans]
    return np.asarray([[*agent.get_position(), *agent.get_velocity()] for agent in agents])


def collect_live_responses(env, candidates, headings, histories, selected, budget, seed,
                           case, window, action_type, velocity_action):
    """Audit cloned response branches after selection without feeding outcomes to the controller."""
    indices = choose_audit_branches(len(candidates), selected, budget, seed)
    original = _observable_snapshot(env)
    random_state = np.random.get_state()
    targets, executed, projected = [], [], []
    try:
        for index in indices:
            np.random.set_state(random_state)
            clone = copy.deepcopy(env)
            target, states, changes = [], [], []
            for step, desired in enumerate(candidates[index]):
                heading = None if headings is None else headings[index, step]
                raw = velocity_action(clone.robot, action_type, desired[2:], heading)
                action = clone.constrain_agent_action_exact(clone.robot, raw)
                clone.step(action)
                velocity = np.asarray(clone.robot.get_velocity())
                changes.append(not np.allclose(velocity, desired[2:], atol=1e-6, rtol=0))
                target.append([human.get_velocity() for human in clone.humans])
                states.append([*clone.robot.get_position(), *velocity])
            targets.append(target)
            executed.append(states)
            projected.append(changes)
    finally:
        np.random.set_state(random_state)
    if not np.array_equal(original, _observable_snapshot(env)):
        raise ValueError("response audit mutated the live environment")
    target = np.asarray(targets, dtype=np.float32).transpose(0, 2, 1, 3)
    trajectories = np.asarray(executed, dtype=np.float32)
    if not np.allclose(trajectories[..., :2], original[0, :2] + np.cumsum(trajectories[..., 2:] * env.time_step, axis=1), atol=2e-5):
        raise ValueError("response audit execution violated the candidate integration contract")
    if not np.allclose(target[:, :, 0], target[0, :, 0], atol=1e-6):
        raise ValueError("response audit first step is not candidate independent")
    human_count = len(env.humans)
    origins = original[1:, :2].astype(np.float32)
    current = np.concatenate([original[0, :2] - origins, np.broadcast_to(original[0, 2:], (human_count, 2))], -1)
    relative = np.broadcast_to(trajectories[:, None, None], (len(indices), human_count, 1, *trajectories.shape[1:])).copy()
    relative[..., :2] -= origins[None, :, None, None]
    rows = len(indices) * human_count
    return {
        "history": np.broadcast_to(histories[None], (len(indices), *histories.shape)).reshape(rows, *histories.shape[1:]).copy(),
        "current": np.broadcast_to(current[None, :, None], (len(indices), human_count, 1, 4)).reshape(rows, 1, 4).copy(),
        "candidates": relative.reshape(rows, 1, *trajectories.shape[1:]),
        "neighbors": np.broadcast_to(histories, (rows, *histories.shape)).copy(),
        "neighbor_mask": np.tile(~np.eye(human_count, dtype=bool), (len(indices), 1)),
        "target": target.reshape(rows, *target.shape[2:]),
        "human_current_velocity": np.tile(original[1:, 2:].astype(np.float32), (len(indices), 1)),
        "previous_executed_robot_velocity": np.broadcast_to(original[0, 2:].astype(np.float32), (rows, 1, 2)).copy(),
        "case": np.full(rows, case, dtype=np.int64),
        "window": np.full(rows, window, dtype=np.int64),
        "human": np.tile(np.arange(human_count), len(indices)),
        "branch": np.repeat(indices, human_count),
        "selected_branch": np.repeat(indices == selected, human_count),
        "branch_projected": np.repeat(np.asarray(projected).any(axis=1), human_count),
    }


def save_response_audit(directory, seed, case, policy, records):
    if not records:
        return None
    path = Path(directory) / "response_audit" / f"seed{seed}_case{case}_{policy}.npz"
    if path.exists():
        raise ValueError("response audit output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {name: np.concatenate([record[name] for record in records]) for name in records[0]}
    np.savez_compressed(path, **data)
    return path
