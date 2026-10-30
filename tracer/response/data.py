from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from tracer.data.heterogeneous_scene import HeterogeneousSceneBatch


@dataclass
class EntityResponseBatch:
    """Entity-level response samples extracted from scene-level batches."""

    entity_history: torch.Tensor
    entity_current: torch.Tensor
    robot_current: torch.Tensor
    robot_candidates: torch.Tensor
    robot_mask: torch.Tensor
    target_future_vel: torch.Tensor
    high_interaction: torch.Tensor
    scene_index: torch.Tensor
    entity_index: torch.Tensor
    dt: float
    neighbor_history: torch.Tensor | None = None
    neighbor_mask: torch.Tensor | None = None
    robot_actions: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "EntityResponseBatch":
        values = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in vars(self).items()
        }
        return EntityResponseBatch(**values)

    @property
    def num_samples(self) -> int:
        return int(self.entity_history.shape[0])


def _current_velocity(history: torch.Tensor, dt: float) -> torch.Tensor:
    if history.shape[-2] < 2:
        return torch.zeros_like(history[..., -1, :2])
    return (history[..., -1, :2] - history[..., -2, :2]) / dt


def build_entity_response_batch(
    scene: HeterogeneousSceneBatch,
    num_robots: int = 2,
    interaction_radius: float = 3.0,
    robot_actions: torch.Tensor | None = None,
) -> EntityResponseBatch:
    """Flatten valid entities and attach their nearest robot candidate futures.

    Robot full histories are deliberately excluded. Each robot is represented
    by its current relative position/current velocity and candidate future.
    """

    if num_robots < 1:
        raise ValueError("num_robots must be positive")
    scene.validate()
    dt = float(scene.dt)
    supervision = scene.supervision_mask().bool()
    valid_entities = scene.agent_mask.bool() & scene.entity_mask.bool() & supervision
    valid_robots = scene.agent_mask.bool() & scene.robot_mask.bool()
    robot_velocity = _current_velocity(scene.history, dt)

    horizon = scene.future.shape[2]
    dtype = scene.history.dtype
    device = scene.history.device

    if not valid_robots.any():
        raise ValueError("scene batch contains no supervised entity with an active robot")
    entity_rows = torch.where(valid_entities)
    if entity_rows[0].numel() == 0:
        raise ValueError("scene batch contains no supervised entity with an active robot")

    # Select nearest valid robots for every entity in one tensor operation. Invalid
    # robot slots receive infinite distance and are removed by the mask below.
    batch_size, num_agents = scene.current_pos.shape[:2]
    if robot_actions is None:
        robot_actions = scene.robot_actions
    if robot_actions is not None:
        if tuple(robot_actions.shape) != (batch_size, num_agents):
            raise ValueError(
                f"robot_actions must be {(batch_size, num_agents)}, got {tuple(robot_actions.shape)}"
            )
        robot_actions = robot_actions.to(device=device)
    distances = torch.linalg.vector_norm(
        scene.current_pos[:, :, None, :] - scene.current_pos[:, None, :, :], dim=-1
    )
    distances = distances.masked_fill(~valid_robots[:, None, :], float("inf"))
    selected_distances, selected_ids = torch.topk(
        distances, k=min(num_robots, num_agents), dim=-1, largest=False
    )
    selected_valid = torch.isfinite(selected_distances)

    selected_current = torch.gather(
        scene.current_pos[:, None, :, :].expand(batch_size, num_agents, num_agents, 2),
        2,
        selected_ids[..., None].expand(batch_size, num_agents, selected_ids.shape[-1], 2),
    )
    selected_velocity = torch.gather(
        robot_velocity[:, None, :, :].expand(batch_size, num_agents, num_agents, 2),
        2,
        selected_ids[..., None].expand(batch_size, num_agents, selected_ids.shape[-1], 2),
    )
    selected_future = torch.gather(
        scene.future[:, None, :, :, :].expand(
            batch_size, num_agents, num_agents, horizon, 2
        ),
        2,
        selected_ids[..., None, None].expand(
            batch_size, num_agents, selected_ids.shape[-1], horizon, 2
        ),
    )
    entity_current = scene.current_pos
    relative_current = selected_current - entity_current[:, :, None, :]
    future_position = selected_current[:, :, :, None, :] + torch.cumsum(
        selected_future * dt, dim=3
    )
    relative_future = future_position - entity_current[:, :, None, None, :]
    robot_current = torch.cat(
        [relative_current, selected_velocity], dim=-1
    ) * selected_valid[..., None].to(dtype)
    robot_candidates = torch.cat(
        [relative_future, selected_future], dim=-1
    ) * selected_valid[..., None, None].to(dtype)
    high_interaction = (
        selected_valid[..., :2].all(dim=-1)
        & (selected_distances[..., :2] < interaction_radius).all(dim=-1)
    )

    flat_batch, flat_entity = entity_rows
    flat_index = flat_batch * num_agents + flat_entity
    entity_history = scene.history[:, :, :, :2].reshape(
        batch_size * num_agents, scene.history.shape[2], 2
    )[flat_index]
    entity_current_flat = entity_current.reshape(batch_size * num_agents, 2)[flat_index]
    target_future = scene.future[:, :, :, :2].reshape(
        batch_size * num_agents, horizon, 2
    )[flat_index]
    flat_robot_current = robot_current.reshape(
        batch_size * num_agents, selected_ids.shape[-1], 4
    )[flat_index]
    flat_robot_candidates = robot_candidates.reshape(
        batch_size * num_agents, selected_ids.shape[-1], horizon, 4
    )[flat_index]
    flat_robot_mask = selected_valid.reshape(
        batch_size * num_agents, selected_ids.shape[-1]
    )[flat_index]
    flat_robot_actions = None
    if robot_actions is not None:
        selected_actions = torch.gather(
            robot_actions[:, None, :].expand(batch_size, num_agents, num_agents),
            2,
            selected_ids,
        )
        flat_robot_actions = selected_actions.reshape(
            batch_size * num_agents, selected_ids.shape[-1]
        )[flat_index]
    all_entity_history = scene.history[:, :, :, :2]
    flat_neighbor_history = all_entity_history[flat_batch]
    flat_neighbor_mask = valid_entities[flat_batch].clone()
    flat_neighbor_mask[torch.arange(flat_entity.numel(), device=device), flat_entity] = False
    return EntityResponseBatch(
        entity_history=entity_history,
        entity_current=entity_current_flat,
        robot_current=flat_robot_current,
        robot_candidates=flat_robot_candidates,
        robot_mask=flat_robot_mask,
        target_future_vel=target_future,
        high_interaction=high_interaction.reshape(batch_size * num_agents)[flat_index],
        scene_index=flat_batch.long(),
        entity_index=flat_entity.long(),
        dt=dt,
        neighbor_history=flat_neighbor_history,
        neighbor_mask=flat_neighbor_mask,
        robot_actions=flat_robot_actions,
    )


def shuffle_robot_candidates(
    batch: EntityResponseBatch,
    permutation: torch.Tensor | None = None,
) -> EntityResponseBatch:
    """Shuffle joint candidate velocities while retaining receiver current states."""

    n = batch.num_samples
    if n < 2:
        return batch
    if permutation is None:
        permutation = torch.arange(n, device=batch.robot_candidates.device)
        # Shuffle across scenes with the same robot availability pattern. A
        # within-scene shuffle would often preserve the identical team action.
        for mask_pattern in torch.unique(batch.robot_mask, dim=0):
            group = torch.where((batch.robot_mask == mask_pattern).all(dim=1))[0]
            scenes = torch.unique(batch.scene_index[group])
            if scenes.numel() < 2:
                continue
            for scene_offset, scene_id in enumerate(scenes):
                targets = group[batch.scene_index[group] == scene_id]
                source_scene = scenes[(scene_offset + 1) % scenes.numel()]
                sources = group[batch.scene_index[group] == source_scene]
                source_indices = torch.arange(targets.numel(), device=group.device) % sources.numel()
                permutation[targets] = sources[source_indices]
    if tuple(permutation.shape) != (n,):
        raise ValueError(f"permutation must have shape {(n,)}, got {tuple(permutation.shape)}")

    shuffled_vel = batch.robot_candidates[permutation, :, :, 2:]
    relative_pos = batch.robot_current[:, :, None, :2] + torch.cumsum(shuffled_vel * batch.dt, dim=2)
    candidates = torch.cat([relative_pos, shuffled_vel], dim=-1)
    candidates = candidates * batch.robot_mask[:, :, None, None].to(candidates.dtype)
    actions = batch.robot_actions[permutation] if batch.robot_actions is not None else None
    return replace(batch, robot_candidates=candidates, robot_actions=actions)


def shuffle_robot_candidates_within_groups(
    batch: EntityResponseBatch,
    group_size: int,
    branch_size: int,
) -> EntityResponseBatch:
    """Permute candidate branches within each factorial base-scene group.

    This is the paired negative control for intervention datasets. It preserves
    the receiver state, robot availability, and base-scene geometry while
    breaking the candidate-to-response pairing. ``group_size`` is the number
    of flattened entity rows in one base scene and ``branch_size`` is the
    number of rows in one intervention branch.
    """

    n = batch.num_samples
    if group_size <= 0 or branch_size <= 0:
        raise ValueError("group_size and branch_size must be positive")
    if group_size % branch_size != 0:
        raise ValueError("group_size must be divisible by branch_size")
    if n % group_size != 0:
        raise ValueError(
            f"batch has {n} rows, which is not divisible by group_size={group_size}"
        )
    branch_count = group_size // branch_size
    if branch_count < 2:
        raise ValueError("within-group shuffle requires at least two branches")

    permutation = torch.arange(n, device=batch.robot_candidates.device)
    for group_start in range(0, n, group_size):
        for branch in range(branch_count):
            target_start = group_start + branch * branch_size
            source_branch = (branch + 1) % branch_count
            source_start = group_start + source_branch * branch_size
            permutation[target_start : target_start + branch_size] = torch.arange(
                source_start,
                source_start + branch_size,
                device=permutation.device,
            )
    return shuffle_robot_candidates(batch, permutation=permutation)
