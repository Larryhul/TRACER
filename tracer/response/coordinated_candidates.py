import itertools
import math

import numpy as np
import torch

from tracer.response.closed_loop import CandidatePool


def complete_joint_speed_candidates(pool, task_offsets, dt, goal_speed, speed_scales, prefix_steps, entity_positions=None, stop_at_goal=False):
    candidates, current = pool.robot_candidates, pool.robot_current
    if candidates.ndim != 4 or candidates.shape[-1] != 4 or tuple(current.shape) != tuple(candidates.shape[:2]) + (4,):
        raise ValueError("joint candidates and current robot states must align")
    count, robots, horizon = candidates.shape[:3]
    if pool.activities.shape != (count, robots):
        raise ValueError("activities must align with joint candidates")
    if count < 1 or not 1 <= prefix_steps <= horizon:
        raise ValueError("nonempty candidate budget and valid prefix required")
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(goal_speed) or goal_speed <= 0:
        raise ValueError("positive finite timestep and goal speed required")
    scales = tuple(float(scale) for scale in speed_scales)
    if len(scales) < 2 or len(set(scales)) != len(scales) or any(not math.isfinite(scale) or scale < 0 for scale in scales):
        raise ValueError("at least two distinct finite nonnegative speed scales required")
    anchor_count = len(scales) ** robots
    if anchor_count >= count:
        raise ValueError("joint speed combinations must leave room for a generated candidate")
    offsets = torch.as_tensor(task_offsets, dtype=candidates.dtype, device=candidates.device)
    if offsets.shape != (robots, 2) or not all(torch.isfinite(value).all() for value in (candidates, current, offsets, pool.activities)):
        raise ValueError("finite aligned observable states required")
    if not torch.allclose(current, current[:1].expand_as(current), rtol=0, atol=1e-6):
        raise ValueError("all candidates must share the current observable robot state")
    distance = offsets.norm(dim=-1)
    direction = offsets / distance[:, None].clamp_min(1e-6)
    combinations = candidates.new_tensor(list(itertools.product(scales, repeat=robots)))
    velocity = (direction[None, :, None] * goal_speed * combinations[:, :, None, None]).expand(-1, -1, horizon, -1).clone()
    displacement = (velocity * dt).cumsum(2)
    if stop_at_goal:
        traveled = displacement.norm(dim=-1).minimum(distance[None, :, None])
        displacement = traveled[..., None] * direction[None, :, None]
        velocity = torch.diff(displacement, dim=2, prepend=torch.zeros_like(displacement[:, :, :1])) / dt
    position = current[0, :, None, :2][None] + displacement
    anchors = torch.cat((position, velocity), dim=-1)
    progress = ((candidates[:, :, prefix_steps-1, :2]-current[..., :2]) * direction[None]).sum(-1).mean(-1)
    if stop_at_goal:
        generated_displacement = candidates[:, :, prefix_steps-1, :2]-current[..., :2]
        progress = (distance[None]-(offsets[None]-generated_displacement).norm(dim=-1)).mean(-1)
    retained = torch.argsort(progress, descending=True, stable=True)[:count-anchor_count]
    entity_candidates = None
    if pool.entity_candidates is not None:
        generated_entities = pool.entity_candidates
        positions = torch.as_tensor(entity_positions, dtype=generated_entities.dtype, device=generated_entities.device) if entity_positions is not None else None
        if generated_entities.ndim != 4 or generated_entities.shape[0] != count or generated_entities.shape[-2:] != (horizon, 2):
            raise ValueError("entity candidates must have candidate/entity/horizon/coordinate axes")
        if positions is None or positions.shape != generated_entities.shape[1:2] + (2,) or not torch.isfinite(positions).all():
            raise ValueError("observed entity positions required for anchor entity geometry")
        stationary = positions[None, :, None].expand(anchor_count, -1, horizon, -1)
        entity_candidates = torch.cat((generated_entities[retained], stationary), 0)
    return CandidatePool(
        robot_current=current[:1].expand(count, -1, -1).clone(),
        robot_candidates=torch.cat((candidates[retained], anchors), 0),
        activities=torch.cat((pool.activities[retained], (velocity.norm(dim=-1).mean(-1)/goal_speed).clamp(0, 1)), 0),
        entity_candidates=entity_candidates,
    )


class CoordinatedCandidateAdapter:
    def __init__(self, generator, speed_scales):
        self.generator = generator
        self.speed_scales = tuple(speed_scales)

    def __getattr__(self, name):
        return getattr(self.generator, name)

    def _complete(self, pool, entity_history, robot_positions, robot_velocities, dt, robot_goals=None):
        positions = np.asarray(robot_positions)
        history = np.asarray(entity_history)
        entity_positions = history[-1:,:] if history.ndim == 2 else history[:, -1, :]
        if robot_goals is not None:
            targets = np.asarray(robot_goals)
        else:
            nearest = np.linalg.norm(positions[:, None]-entity_positions[None], axis=-1).argmin(-1)
            targets = entity_positions[nearest]
        return complete_joint_speed_candidates(pool, targets-positions, dt, self.config.goal_speed,
            self.speed_scales, self.config.task_prefix_steps, entity_positions, self.config.stop_task_anchors_at_goal)

    def candidate_pool(self, entity_history, robot_positions, robot_velocities, dt, sample_seed, robot_goals=None):
        pool = self.generator.candidate_pool(entity_history, robot_positions, robot_velocities, dt, sample_seed, robot_goals)
        return self._complete(pool, entity_history, robot_positions, robot_velocities, dt, robot_goals)

    def candidate_pools(self, scenes, sample_seeds):
        pools = self.generator.candidate_pools(scenes, sample_seeds)
        return [self._complete(pool, *scene) for pool, scene in zip(pools, scenes, strict=True)]
