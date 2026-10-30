from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tracer.config import sage_config_from_dict
from tracer.data.heterogeneous_scene import AgentRole, HeterogeneousSceneBatch
from tracer.experiments.checkpoint_compat import apply_checkpoint_compatibility
from tracer.models.sage_model import SAGE
from tracer.response.closed_loop import CandidatePool
from tracer.response.planning_support import diverse_candidate_indices


@dataclass(frozen=True)
class SAGECandidateAdapterConfig:
    num_candidates: int = 16
    guided: bool = True
    step_stride: int = 10
    goal_speed: float = 1.2
    oversample_factor: int = 1
    task_prefilter: bool = False
    task_prefix_steps: int = 3
    # ``task_goal`` uses the same robot-goal contract as SAGE training.  The
    # legacy ``velocity``/``entity`` modes remain available for old artifacts.
    goal_direction: str = "velocity"
    include_task_anchor: bool = False
    task_anchor_speed_scales: tuple[float, ...] = (1.0,)
    cache_candidate_pools: bool = False
    batch_size: int = 4
    guidance_step_clip_norm: float | None = None
    diversity_weight: float = 0.0
    goal_distance_prefilter: bool = False
    stop_task_anchors_at_goal: bool = False


class SAGECandidateAdapter:
    """Convert a receding-horizon scene to SAGE joint robot candidates."""

    def __init__(
        self,
        model: SAGE,
        device: str,
        config: SAGECandidateAdapterConfig,
    ) -> None:
        if config.num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        if config.step_stride < 1:
            raise ValueError("step_stride must be positive")
        if config.oversample_factor < 1:
            raise ValueError("oversample_factor must be positive")
        if config.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= config.diversity_weight <= 1.0:
            raise ValueError("diversity_weight must lie in [0, 1]")
        if not 1 <= config.task_prefix_steps <= int(model.config.diffusion.horizon):
            raise ValueError("task_prefix_steps must lie within the model horizon")
        if config.goal_direction not in {"velocity", "entity", "task_goal"}:
            raise ValueError(
                "goal_direction must be 'velocity', 'entity', or 'task_goal'"
            )
        if not config.task_anchor_speed_scales or any(
            scale <= 0.0 for scale in config.task_anchor_speed_scales
        ):
            raise ValueError("task_anchor_speed_scales must contain positive values")
        self.model = model.eval()
        self.device = torch.device(device)
        self.config = config
        self.horizon = int(model.config.diffusion.horizon)
        self._candidate_cache: dict[str, CandidatePool] = {}

    def _cache_key(
        self,
        entity_history: np.ndarray,
        robot_positions: np.ndarray,
        robot_velocities: np.ndarray,
        dt: float,
        sample_seed: int,
        robot_goals: np.ndarray | None = None,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(str(int(sample_seed)).encode("ascii"))
        digest.update(np.asarray(float(dt), dtype=np.float64).tobytes())
        for name, value in (
            ("entity_history", entity_history),
            ("robot_positions", robot_positions),
            ("robot_velocities", robot_velocities),
        ):
            array = np.ascontiguousarray(value)
            digest.update(name.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes())
        if robot_goals is not None:
            array = np.ascontiguousarray(robot_goals)
            digest.update(b"robot_goals")
            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str,
        config: SAGECandidateAdapterConfig,
    ) -> "SAGECandidateAdapter":
        checkpoint = torch.load(
            Path(checkpoint_path), map_location=device, weights_only=False
        )
        if "config" not in checkpoint:
            raise ValueError("SAGE checkpoint must contain its training config")
        sage_config = sage_config_from_dict(checkpoint["config"])
        sage_config = apply_checkpoint_compatibility(
            sage_config, checkpoint.get("model", {})
        )
        if config.guidance_step_clip_norm is not None:
            sage_config.guidance.guidance_step_clip_norm = float(
                config.guidance_step_clip_norm
            )
        model = SAGE(sage_config).to(device)
        model.load_state_dict(checkpoint["model"])
        return cls(model, device, config)

    def _batch(
        self,
        entity_history: np.ndarray,
        robot_positions: np.ndarray,
        robot_velocities: np.ndarray,
        dt: float,
        robot_goals: np.ndarray | None = None,
    ) -> HeterogeneousSceneBatch:
        entity_history = np.asarray(entity_history, dtype=np.float32)
        robot_positions = np.asarray(robot_positions, dtype=np.float32)
        robot_velocities = np.asarray(robot_velocities, dtype=np.float32)
        if entity_history.ndim not in {2, 3} or entity_history.shape[-1] != 2:
            raise ValueError(
                "entity_history must have shape (history, 2) or (entities, history, 2)"
            )
        if entity_history.ndim == 2:
            entity_history = entity_history[None]
        if entity_history.shape[1] < 1:
            raise ValueError("entity_history must contain at least one history step")
        if robot_positions.ndim != 2 or robot_positions.shape[-1] != 2:
            raise ValueError("robot_positions must have shape (robots, 2)")
        if robot_velocities.shape != robot_positions.shape:
            raise ValueError("robot_velocities must align with robot_positions")
        if robot_goals is not None:
            robot_goals = np.asarray(robot_goals, dtype=np.float32)
            if robot_goals.shape != robot_positions.shape:
                raise ValueError("robot_goals must align with robot_positions")
            if not np.isfinite(robot_goals).all():
                raise ValueError("robot_goals must contain finite values")
        if not np.isfinite(entity_history).all() or not np.isfinite(
            robot_positions
        ).all() or not np.isfinite(robot_velocities).all():
            raise ValueError("scene inputs must contain finite values")

        robots = robot_positions.shape[0]
        entity_count, history_steps = entity_history.shape[:2]
        offsets = (
            np.arange(1 - history_steps, 1, dtype=np.float32) * float(dt)
        )
        robot_history = (
            robot_positions[:, None, :]
            + robot_velocities[:, None, :] * offsets[None, :, None]
        )
        scene_history = np.concatenate([robot_history, entity_history], axis=0)
        current = np.concatenate([robot_positions, entity_history[:, -1]], axis=0)
        candidates = self.config.num_candidates * self.config.oversample_factor
        agents = robots + entity_count
        history = torch.from_numpy(scene_history)[None].expand(
            candidates, -1, -1, -1
        ).clone()
        current_pos = torch.from_numpy(current)[None].expand(
            candidates, -1, -1
        ).clone()
        future = torch.zeros(candidates, agents, self.horizon, 2)
        roles = torch.full(
            (candidates, agents), int(AgentRole.PHA), dtype=torch.long
        )
        roles[:, :robots] = int(AgentRole.ROBOT)
        agent_mask = torch.ones(candidates, agents, dtype=torch.bool)
        robot_mask = torch.zeros_like(agent_mask)
        robot_mask[:, :robots] = True
        entity_mask = ~robot_mask

        entity_positions = entity_history[:, -1]
        distances = np.linalg.norm(
            entity_positions[None] - robot_positions[:, None], axis=-1
        )
        nearest_entity = entity_positions[distances.argmin(axis=1)]
        entity_direction = nearest_entity - robot_positions
        entity_norm = np.linalg.norm(entity_direction, axis=-1, keepdims=True)
        entity_direction /= np.maximum(entity_norm, 1e-6)
        speed = np.linalg.norm(robot_velocities, axis=-1, keepdims=True)
        velocity_direction = np.where(
            speed > 1e-6,
            robot_velocities / np.maximum(speed, 1e-6),
            entity_direction,
        )
        if self.config.goal_direction == "task_goal":
            if robot_goals is None:
                raise ValueError(
                    "robot_goals are required when goal_direction='task_goal'"
                )
            task_direction = robot_goals - robot_positions
            task_norm = np.linalg.norm(task_direction, axis=-1, keepdims=True)
            direction = np.where(
                task_norm > 1e-6,
                task_direction / np.maximum(task_norm, 1e-6),
                velocity_direction,
            )
        elif self.config.goal_direction == "entity":
            direction = entity_direction
        else:
            direction = velocity_direction
        fallback_goals = robot_positions + (
            direction
            * float(self.config.goal_speed)
            * self.horizon
            * float(dt)
        )
        goals = current_pos.clone()
        goals[:, :robots] = torch.from_numpy(
            robot_goals if robot_goals is not None else fallback_goals
        )[None]
        goal_mask = robot_mask.clone()
        batch = HeterogeneousSceneBatch(
            history=history,
            future=future,
            current_pos=current_pos,
            roles=roles,
            agent_mask=agent_mask,
            robot_mask=robot_mask,
            entity_mask=entity_mask,
            loss_mask=agent_mask,
            goals=goals,
            goal_mask=goal_mask,
            dt=float(dt),
        )
        batch.validate()
        return batch.to(self.device)

    def _batch_many(
        self,
        scenes: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    ) -> HeterogeneousSceneBatch:
        """Build one diffusion batch while keeping each scene's candidate block contiguous."""

        batches = [self._batch(*scene) for scene in scenes]
        return HeterogeneousSceneBatch(
            history=torch.cat([batch.history for batch in batches], dim=0),
            future=torch.cat([batch.future for batch in batches], dim=0),
            current_pos=torch.cat([batch.current_pos for batch in batches], dim=0),
            roles=torch.cat([batch.roles for batch in batches], dim=0),
            agent_mask=torch.cat([batch.agent_mask for batch in batches], dim=0),
            robot_mask=torch.cat([batch.robot_mask for batch in batches], dim=0),
            entity_mask=torch.cat([batch.entity_mask for batch in batches], dim=0),
            loss_mask=torch.cat([batch.loss_mask for batch in batches], dim=0),
            goals=torch.cat([batch.goals for batch in batches], dim=0),
            goal_mask=torch.cat([batch.goal_mask for batch in batches], dim=0),
            dt=batches[0].dt,
        )

    def _noise_schedule(self, shape: tuple[int, int, int, int], sample_seed: int) -> list[torch.Tensor]:
        """Reproduce one scalar adapter call's random stream for batched generation."""

        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        schedule: list[torch.Tensor] = []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(sample_seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(sample_seed))
            schedule.append(torch.randn(shape, device=self.device))
            for step in range(self.model.config.diffusion.num_steps, 0, -self.config.step_stride):
                if max(step - self.config.step_stride, 0) != 0:
                    schedule.append(torch.randn_like(schedule[0]))
        return schedule

    def _finalize_pool(
        self,
        entity_history: np.ndarray,
        robot_positions: np.ndarray,
        robot_velocities: np.ndarray,
        dt: float,
        output: dict[str, torch.Tensor],
        robot_goals: np.ndarray | None = None,
    ) -> CandidatePool:
        """Convert a generated scene block into the closed-loop candidate contract."""

        robots = robot_positions.shape[0]
        if entity_history.ndim == 3:
            entity_positions = np.asarray(entity_history[:, -1], dtype=np.float32)
            distances = np.linalg.norm(
                entity_positions[None] - robot_positions[:, None], axis=-1
            )
            nearest_entity = entity_positions[distances.argmin(axis=1)]
        else:
            nearest_entity = np.asarray(entity_history[-1], dtype=np.float32)[None]
        entity_current = torch.as_tensor(
            entity_history[-1], device=self.device, dtype=output["future_pos"].dtype
        )
        robot_future_pos = output["future_pos"][:, :robots]
        robot_future_vel = output["future_vel"][:, :robots]
        entity_future_pos = output["future_pos"][:, robots:]
        if entity_history.ndim == 2:
            relative_future = robot_future_pos - entity_current[None, None, None, :]
            candidate_position = relative_future
            relative_current = torch.as_tensor(
                robot_positions - entity_history[-1][None, :],
                device=self.device,
                dtype=robot_future_pos.dtype,
            )
        else:
            # With a complete entity set there is no single valid origin. The
            # public candidate contract therefore returns world-frame robot
            # positions and velocities directly.
            candidate_position = robot_future_pos
            relative_current = torch.as_tensor(
                robot_positions,
                device=self.device,
                dtype=robot_future_pos.dtype,
            )
        candidates = torch.cat([candidate_position, robot_future_vel], dim=-1)
        current_velocity = torch.as_tensor(
            robot_velocities, device=self.device, dtype=candidates.dtype
        )
        current = torch.cat([relative_current, current_velocity], dim=-1)
        generated_candidates = candidates.shape[0]
        current = current[None].expand(generated_candidates, -1, -1)
        activities = torch.linalg.vector_norm(robot_future_vel, dim=-1).mean(dim=-1)
        activities = (activities / max(float(self.config.goal_speed), 1e-6)).clamp(
            0.0, 1.0
        )
        if self.config.include_task_anchor:
            anchor_target = (
                robot_goals
                if robot_goals is not None and self.config.goal_direction == "task_goal"
                else np.broadcast_to(
                    nearest_entity if entity_history.ndim == 3 else entity_history[-1],
                    robot_positions.shape,
                )
            )
            anchor_direction = torch.as_tensor(
                anchor_target - robot_positions,
                device=self.device,
                dtype=robot_future_pos.dtype,
            )
            anchor_distance = torch.linalg.vector_norm(anchor_direction, dim=-1)
            anchor_direction = anchor_direction / torch.linalg.vector_norm(
                anchor_direction, dim=-1, keepdim=True
            ).clamp_min(1e-6)
            speed_scales = torch.as_tensor(
                self.config.task_anchor_speed_scales,
                device=self.device,
                dtype=robot_future_pos.dtype,
            )
            anchor_velocity = (
                anchor_direction[None, :, None, :]
                * float(self.config.goal_speed)
                * speed_scales[:, None, None, None]
            ).expand(-1, -1, self.horizon, -1)
            anchor_position = torch.as_tensor(
                robot_positions, device=self.device, dtype=robot_future_pos.dtype
            )[None, :, None, :] + torch.cumsum(anchor_velocity * float(dt), dim=2)
            if self.config.stop_task_anchors_at_goal:
                distances = (
                    speed_scales[:, None, None] * float(self.config.goal_speed) * float(dt)
                    * torch.arange(1, self.horizon + 1, device=self.device)[None, None]
                ).minimum(anchor_distance[None, :, None])
                displacement = distances[..., None] * anchor_direction[None, :, None]
                initial = torch.as_tensor(robot_positions, device=self.device, dtype=robot_future_pos.dtype)
                anchor_position = initial[None, :, None] + displacement
                anchor_velocity = torch.diff(
                    displacement, dim=2, prepend=torch.zeros_like(displacement[:, :, :1])
                ) / float(dt)
            anchor_relative = (
                anchor_position
                if entity_history.ndim == 3
                else anchor_position - entity_current[None, None, None, :]
            )
            anchors = torch.cat([anchor_relative, anchor_velocity], dim=-1)
            candidates = torch.cat([candidates, anchors], dim=0)
            current = torch.cat(
                [current, current[:1].expand(len(speed_scales), -1, -1)], dim=0
            )
            # Keep candidate-wise alignment for the optional joint entity
            # futures. Anchors have no diffusion response; use the observed
            # entity positions as a neutral placeholder and let callers
            # disable the dynamic SAGE mask when anchors are mixed in.
            anchor_entity_current = (
                torch.as_tensor(
                    entity_history[:, -1] if entity_history.ndim == 3 else entity_history[-1][None],
                    device=self.device,
                    dtype=robot_future_pos.dtype,
                )
                .expand(len(speed_scales), -1, -1)
                .unsqueeze(2)
                .expand(-1, -1, self.horizon, -1)
            )
            entity_future_pos = torch.cat(
                [entity_future_pos, anchor_entity_current], dim=0
            )
            activities = torch.cat(
                [
                    activities,
                    speed_scales[:, None].expand(-1, robots).clamp(0.0, 1.0),
                ]
            )
        generated_candidates = candidates.shape[0]
        if generated_candidates > self.config.num_candidates:
            if self.config.task_prefilter:
                prefix = self.config.task_prefix_steps - 1
                prefilter_target = (
                    torch.as_tensor(
                        robot_goals,
                        device=self.device,
                        dtype=robot_future_pos.dtype,
                    )
                    if robot_goals is not None
                    and self.config.goal_direction == "task_goal"
                    else (
                        torch.as_tensor(
                            nearest_entity,
                            device=self.device,
                            dtype=robot_future_pos.dtype,
                        )
                        if entity_history.ndim == 3
                        else entity_current[None].expand(robots, -1)
                    )
                )
                toward = prefilter_target - torch.as_tensor(
                    robot_positions,
                    device=self.device,
                    dtype=robot_future_pos.dtype,
                )
                goal_offsets = toward.clone()
                toward = toward / torch.linalg.vector_norm(
                    toward, dim=-1, keepdim=True
                ).clamp_min(1e-6)
                displacement = candidates[:, :, prefix, :2] - relative_current[None]
                task_progress = (displacement * toward[None]).sum(dim=-1).mean(dim=-1)
                if self.config.goal_distance_prefilter:
                    task_progress = (
                        torch.linalg.vector_norm(goal_offsets, dim=-1)
                        - torch.linalg.vector_norm(goal_offsets[None] - displacement, dim=-1)
                    ).mean(dim=-1)
                if self.config.diversity_weight > 0.0:
                    selected = diverse_candidate_indices(
                        candidates, task_progress, self.config.num_candidates,
                        self.config.diversity_weight,
                    )
                else:
                    selected = task_progress.topk(self.config.num_candidates).indices
            else:
                selected = torch.arange(self.config.num_candidates, device=self.device)
            candidates = candidates[selected]
            current = current[selected]
            activities = activities[selected]
            entity_future_pos = entity_future_pos[selected]
        return CandidatePool(
            robot_current=current.cpu(),
            robot_candidates=candidates.cpu(),
            activities=activities.cpu(),
            entity_candidates=entity_future_pos.cpu(),
        )

    @torch.no_grad()
    def candidate_pool(
        self,
        entity_history: np.ndarray,
        robot_positions: np.ndarray,
        robot_velocities: np.ndarray,
        dt: float,
        sample_seed: int,
        robot_goals: np.ndarray | None = None,
    ) -> CandidatePool:
        """Generate a candidate pool under the explicit trajectory contract.

        ``entity_history`` may be one track ``(history, 2)`` or a complete
        entity set ``(entities, history, 2)``. The former preserves the legacy
        entity-relative output; the latter returns world-frame robot
        trajectories so no arbitrary entity can be used as an origin.
        """
        cache_key = None
        if self.config.cache_candidate_pools:
            cache_key = self._cache_key(
                entity_history,
                robot_positions,
                robot_velocities,
                dt,
                sample_seed,
                robot_goals=robot_goals,
            )
            cached = self._candidate_cache.get(cache_key)
            if cached is not None:
                return cached
        batch = self._batch(
            entity_history,
            robot_positions,
            robot_velocities,
            dt,
            robot_goals=robot_goals,
        )
        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(sample_seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(sample_seed))
            output = self.model.generate(
                batch,
                guided=self.config.guided,
                step_stride=self.config.step_stride,
            )
        pool = self._finalize_pool(
            entity_history,
            robot_positions,
            robot_velocities,
            dt,
            output,
            robot_goals=robot_goals,
        )
        if cache_key is not None:
            self._candidate_cache[cache_key] = pool
        return pool

    @torch.no_grad()
    def candidate_pools(
        self,
        scenes: list[tuple[np.ndarray, np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]],
        sample_seeds: list[int],
    ) -> list[CandidatePool]:
        """Generate independent candidate pools in one GPU batch.

        Each scene receives the same diffusion random stream as an isolated
        ``candidate_pool`` call with its own seed. Scene blocks stay contiguous
        so post-processing and candidate ordering remain unchanged.
        """

        if len(scenes) != len(sample_seeds):
            raise ValueError("scenes and sample_seeds must have equal length")
        if not scenes:
            return []
        normalized_scenes = []
        for scene in scenes:
            if len(scene) == 4:
                normalized_scenes.append((*scene, None))
            elif len(scene) == 5:
                normalized_scenes.append(tuple(scene))
            else:
                raise ValueError(
                    "each scene must contain history, positions, velocities, dt, and optional robot_goals"
                )
        results: list[CandidatePool | None] = [None] * len(scenes)
        pending: list[int] = []
        for index, (scene, seed) in enumerate(zip(normalized_scenes, sample_seeds)):
            if self.config.cache_candidate_pools:
                key = self._cache_key(
                    *scene[:4], int(seed), robot_goals=scene[4]
                )
                cached = self._candidate_cache.get(key)
                if cached is not None:
                    results[index] = cached
                    continue
            pending.append(index)
        if not pending:
            return [pool for pool in results if pool is not None]

        if self.model.generation_mode == "decoupled":
            raise NotImplementedError("batched candidate generation requires joint SAGE checkpoints")
        candidates_per_scene = self.config.num_candidates * self.config.oversample_factor
        for start in range(0, len(pending), self.config.batch_size):
            chunk = pending[start : start + self.config.batch_size]
            pending_scenes = [normalized_scenes[index] for index in chunk]
            batch = self._batch_many(pending_scenes)
            agents = batch.num_agents
            shape = (candidates_per_scene, agents, self.horizon, 2)
            schedules = [
                self._noise_schedule(shape, int(sample_seeds[index])) for index in chunk
            ]
            schedule_count = len(schedules[0])
            noise_schedule = [
                torch.cat([schedule[noise_index] for schedule in schedules], dim=0)
                for noise_index in range(schedule_count)
            ]
            output = self.model.generate(
                batch,
                guided=self.config.guided,
                step_stride=self.config.step_stride,
                noise_schedules=noise_schedule,
            )
            offset = 0
            for index, scene in zip(chunk, pending_scenes):
                count = candidates_per_scene
                scene_output = {
                    name: value[offset : offset + count]
                    for name, value in output.items()
                    if torch.is_tensor(value)
                }
                pool = self._finalize_pool(
                    *scene[:4], scene_output, robot_goals=scene[4]
                )
                if self.config.cache_candidate_pools:
                    key = self._cache_key(
                        *scene[:4],
                        int(sample_seeds[index]),
                        robot_goals=scene[4],
                    )
                    self._candidate_cache[key] = pool
                results[index] = pool
                offset += count
        return [pool for pool in results if pool is not None]
