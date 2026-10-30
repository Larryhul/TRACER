"""Retain an executed plan's remaining controls in the next candidate pool."""

from dataclasses import replace
import hashlib
import math

import torch

from tracer.response.closed_loop import CandidatePool


def plan_context_sha256(previous_plan: torch.Tensor | None, execute_steps: int) -> str:
    if previous_plan is None:
        return hashlib.sha256(b"no_previous_plan").hexdigest()
    controls = previous_plan[:, execute_steps:, 2:].detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(controls.shape).encode("ascii"))
    digest.update(str(controls.dtype).encode("ascii"))
    digest.update(controls.tobytes())
    return digest.hexdigest()


def shifted_plan_candidate(pool: CandidatePool, previous_plan: torch.Tensor, execute_steps: int, dt: float) -> torch.Tensor:
    if previous_plan.shape != pool.robot_candidates.shape[1:] or not 0 < execute_steps < previous_plan.shape[1]:
        raise ValueError("retained plan and executed prefix must match the candidate horizon")
    if not math.isfinite(dt) or dt <= 0 or not torch.isfinite(previous_plan).all():
        raise ValueError("finite previous controls and positive timestep are required")
    velocities = torch.cat([previous_plan[:, execute_steps:, 2:], previous_plan[:, -1:, 2:].expand(-1, execute_steps, -1)], dim=1).to(pool.robot_candidates)
    positions = pool.robot_current[0, :, None, :2] + (velocities * dt).cumsum(dim=1)
    return torch.cat([positions, velocities], dim=-1)


def retained_candidate_index(pool: CandidatePool, candidate: torch.Tensor) -> int:
    matches = torch.isclose(pool.robot_candidates, candidate[None], rtol=1e-5, atol=2e-5).flatten(1).all(1)
    return int(torch.where(matches)[0][0]) if matches.any() else -1


def retain_plan_candidate(pool: CandidatePool, previous_plan: torch.Tensor, execute_steps: int, dt: float, speed_scale: float) -> tuple[CandidatePool, int]:
    if not math.isfinite(speed_scale) or speed_scale <= 0:
        raise ValueError("retained activity speed scale must be positive")
    candidate = shifted_plan_candidate(pool, previous_plan, execute_steps, dt)
    existing = retained_candidate_index(pool, candidate)
    if existing >= 0:
        return pool, existing
    candidates, activities = pool.robot_candidates.clone(), pool.activities.clone()
    candidates[-1] = candidate
    activities[-1] = (torch.linalg.vector_norm(candidate[..., 2:], dim=-1).mean(-1) / speed_scale).clamp(0., 1.)
    return replace(pool, robot_candidates=candidates, activities=activities, entity_candidates=None), len(candidates) - 1
