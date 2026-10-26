from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sage.response.sage_candidate_adapter import (
    SAGECandidateAdapter,
    SAGECandidateAdapterConfig,
)


class _GoalRecordingSAGE:
    config = SimpleNamespace(diffusion=SimpleNamespace(horizon=4))

    def __init__(self) -> None:
        self.seen_goals: list[torch.Tensor] = []
        self.generate_calls = 0

    def eval(self):
        return self

    def generate(self, batch, guided=True, step_stride=1):
        self.seen_goals.append(batch.goals.detach().cpu().clone())
        self.generate_calls += 1
        velocity = torch.zeros(batch.batch_size, batch.num_agents, 4, 2)
        position = batch.current_pos[:, :, None] + torch.cumsum(
            velocity * batch.dt, dim=2
        )
        return {"future_vel": velocity, "future_pos": position}


def _scene():
    history = np.zeros((5, 2), dtype=np.float32)
    positions = np.asarray([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    goals = np.asarray([[7.0, 1.0], [-7.0, -1.0]], dtype=np.float32)
    return history, positions, velocities, 0.4, goals


def test_task_goal_conditioning_reaches_sage_batch_and_task_anchor():
    model = _GoalRecordingSAGE()
    adapter = SAGECandidateAdapter(
        model,
        "cpu",
        SAGECandidateAdapterConfig(
            num_candidates=2,
            oversample_factor=2,
            goal_direction="task_goal",
            task_prefilter=True,
            include_task_anchor=True,
            task_anchor_speed_scales=(0.5,),
        ),
    )

    history, positions, velocities, dt, goals = _scene()
    pool = adapter.candidate_pool(
        history, positions, velocities, dt, sample_seed=17, robot_goals=goals
    )

    assert model.seen_goals
    seen = model.seen_goals[0][0, :2].numpy()
    np.testing.assert_allclose(seen, goals)
    relative_current = pool.robot_current[:, :, :2]
    displacement = pool.robot_candidates[:, :, 1, :2] - relative_current
    task_direction = goals - positions
    task_direction /= np.linalg.norm(task_direction, axis=-1, keepdims=True)
    progress = (
        (displacement * torch.from_numpy(task_direction)[None])
        .sum(dim=-1)
        .mean(dim=-1)
        .numpy()
    )
    assert np.any(progress > 0.0)


def test_task_goal_conditioning_requires_explicit_robot_goals():
    adapter = SAGECandidateAdapter(
        _GoalRecordingSAGE(),
        "cpu",
        SAGECandidateAdapterConfig(goal_direction="task_goal"),
    )
    history, positions, velocities, dt, _ = _scene()
    with pytest.raises(ValueError, match="robot_goals are required"):
        adapter.candidate_pool(history, positions, velocities, dt, sample_seed=17)


def test_goal_stopping_anchor_reaches_goal_without_overshoot():
    adapter = SAGECandidateAdapter(
        _GoalRecordingSAGE(), "cpu", SAGECandidateAdapterConfig(
            num_candidates=1, goal_direction="task_goal", include_task_anchor=True,
            task_prefilter=True, goal_distance_prefilter=True, stop_task_anchors_at_goal=True,
        ),
    )
    history, positions, velocities, dt, _ = _scene()
    goals = positions + np.asarray([[0.1, 0.0], [-0.1, 0.0]], dtype=np.float32)
    pool = adapter.candidate_pool(history, positions, velocities, dt, sample_seed=17, robot_goals=goals)
    candidate = pool.robot_candidates[0]
    torch.testing.assert_close(candidate[:, 0, :2], torch.from_numpy(goals))
    torch.testing.assert_close(candidate[:, -1, :2], torch.from_numpy(goals))
    torch.testing.assert_close(candidate[:, 1:, 2:], torch.zeros_like(candidate[:, 1:, 2:]))
    integrated = pool.robot_current[0, :, None, :2] + torch.cumsum(candidate[..., 2:] * dt, dim=1)
    torch.testing.assert_close(candidate[..., :2], integrated)


def test_cache_key_includes_robot_goals():
    model = _GoalRecordingSAGE()
    adapter = SAGECandidateAdapter(
        model,
        "cpu",
        SAGECandidateAdapterConfig(
            num_candidates=2,
            cache_candidate_pools=True,
            goal_direction="task_goal",
        ),
    )
    history, positions, velocities, dt, goals = _scene()
    adapter.candidate_pool(
        history, positions, velocities, dt, sample_seed=17, robot_goals=goals
    )
    changed_goals = goals + np.asarray([[0.5, 0.0], [0.0, -0.5]], dtype=np.float32)
    adapter.candidate_pool(
        history, positions, velocities, dt, sample_seed=17, robot_goals=changed_goals
    )
    assert model.generate_calls == 2


def test_batch_scene_goal_alignment_is_preserved():
    model = _GoalRecordingSAGE()
    adapter = SAGECandidateAdapter(
        model,
        "cpu",
        SAGECandidateAdapterConfig(
            num_candidates=2,
            oversample_factor=1,
            goal_direction="task_goal",
            batch_size=2,
        ),
    )
    history, positions, velocities, dt, goals = _scene()
    shifted_goals = goals + np.asarray([[0.0, 2.0], [0.0, -2.0]], dtype=np.float32)
    batch = adapter._batch_many(
        [
            (history, positions, velocities, dt, goals),
            (history, positions, velocities, dt, shifted_goals),
        ]
    )
    seen = batch.goals[:, :2].numpy()
    np.testing.assert_allclose(seen[:2], np.repeat(goals[None], 2, axis=0))
    np.testing.assert_allclose(
        seen[2:], np.repeat(shifted_goals[None], 2, axis=0)
    )


def test_multi_entity_history_returns_world_frame_robot_candidates():
    model = _GoalRecordingSAGE()
    adapter = SAGECandidateAdapter(
        model,
        "cpu",
        SAGECandidateAdapterConfig(num_candidates=2, oversample_factor=1),
    )
    entity_history = np.stack(
        [
            np.zeros((5, 2), dtype=np.float32),
            np.ones((5, 2), dtype=np.float32),
        ],
        axis=0,
    )
    robot_positions = np.asarray([[2.0, 0.0]], dtype=np.float32)
    robot_velocities = np.zeros_like(robot_positions)
    pool = adapter.candidate_pool(
        entity_history,
        robot_positions,
        robot_velocities,
        0.25,
        sample_seed=9,
    )
    assert model.seen_goals[0].shape[-2] == 3  # one robot plus two entities
    np.testing.assert_allclose(
        pool.robot_current[:, 0, :2], np.repeat(robot_positions, 2, axis=0)
    )
    np.testing.assert_allclose(
        pool.robot_candidates[:, 0, :, :2],
        np.broadcast_to(robot_positions[0][None, None, :], (2, 4, 2)),
    )
