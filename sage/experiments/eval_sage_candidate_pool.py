from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from sage.response.closed_loop import (
    ClosedLoopSpec,
    PlanningCostConfig,
    SyntheticClosedLoopSimulator,
)
from sage.response.sage_candidate_adapter import (
    SAGECandidateAdapter,
    SAGECandidateAdapterConfig,
)


def _candidate_metrics(
    pool,
    execute_steps: int,
    robot_goals: np.ndarray | None = None,
) -> tuple[dict[str, float], torch.Tensor]:
    candidates = pool.robot_candidates.float()
    current = pool.robot_current.float()
    finite = bool(torch.isfinite(candidates).all() and torch.isfinite(current).all())
    step = execute_steps - 1
    displacement = candidates[:, :, step, :2] - current[:, :, :2]
    if robot_goals is None:
        relative = current[:, :, :2]
        toward = -relative / torch.linalg.vector_norm(
            relative, dim=-1, keepdim=True
        ).clamp_min(1e-6)
    else:
        goals = torch.as_tensor(robot_goals, dtype=current.dtype)
        if tuple(goals.shape) != (int(current.shape[1]), 2):
            raise ValueError("robot_goals must have shape (robots, 2)")
        toward = goals[None] - current[:, :, :2]
        toward = toward / torch.linalg.vector_norm(
            toward, dim=-1, keepdim=True
        ).clamp_min(1e-6)
    progress = (displacement * toward).sum(dim=-1).mean(dim=-1)
    flat = candidates.flatten(start_dim=1)
    diversity = float(torch.pdist(flat).mean()) if flat.shape[0] > 1 else 0.0
    rounded = torch.round(flat * 1000.0).to(torch.int64)
    unique_fraction = float(torch.unique(rounded, dim=0).shape[0] / flat.shape[0])
    return (
        {
            "all_finite": float(finite),
            "candidate_diversity": diversity,
            "unique_candidate_fraction": unique_fraction,
            "positive_progress_fraction": float((progress > 0.0).float().mean()),
            "has_positive_progress": float((progress > 0.0).any()),
            "max_progress": float(progress.max()),
            "mean_progress": float(progress.mean()),
            "progress_standard_deviation": float(progress.std(unbiased=False)),
            "activity_standard_deviation": float(
                pool.activities.float().std(unbiased=False)
            ),
        },
        progress,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit SAGE joint-candidate diversity and task availability."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--oversample-factor", type=int, default=4)
    parser.add_argument("--disable-task-prefilter", action="store_true")
    parser.add_argument(
        "--goal-direction",
        choices=("velocity", "entity", "task_goal"),
        default="task_goal",
    )
    parser.add_argument("--seed", type=int, default=32027)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    adapter = SAGECandidateAdapter.from_checkpoint(
        args.checkpoint,
        args.device,
        SAGECandidateAdapterConfig(
            num_candidates=args.num_candidates,
            guided=True,
            step_stride=10,
            goal_speed=1.2,
            oversample_factor=args.oversample_factor,
            task_prefilter=not args.disable_task_prefilter,
            task_prefix_steps=args.execute_steps,
            goal_direction=args.goal_direction,
        ),
    )
    records: list[dict[str, float | int | str]] = []
    planning_cost = PlanningCostConfig()
    for split in ("id", "ood"):
        spec = ClosedLoopSpec(
            windows=args.windows,
            num_robots=args.num_robots,
            num_candidates=args.num_candidates,
            execute_steps=args.execute_steps,
            seed=args.seed,
            split=split,
        )
        for episode in range(args.episodes):
            simulator = SyntheticClosedLoopSimulator(spec, episode)
            for window in range(args.windows):
                sample_seed = (
                    args.seed
                    + (0 if split == "id" else 7_000_003)
                    + episode * 104729
                    + window * 15485863
                )
                pool = adapter.candidate_pool(
                    simulator.history,
                    simulator.robot_positions,
                    simulator.robot_velocities,
                    simulator.spec.dt,
                    sample_seed,
                    robot_goals=(
                        np.broadcast_to(
                            simulator.entity_position,
                            simulator.robot_positions.shape,
                        ).copy()
                        if args.goal_direction == "task_goal"
                        else None
                    ),
                )
                metrics, progress = _candidate_metrics(
                    pool,
                    args.execute_steps,
                    robot_goals=(
                        np.broadcast_to(
                            simulator.entity_position,
                            simulator.robot_positions.shape,
                        ).copy()
                        if args.goal_direction == "task_goal"
                        else None
                    ),
                )
                records.append(
                    {
                        "split": split,
                        "episode": episode,
                        "window": window,
                        **metrics,
                    }
                )
                simulator.step(pool, int(progress.argmax()), planning_cost)

    metric_names = [
        "all_finite",
        "candidate_diversity",
        "unique_candidate_fraction",
        "positive_progress_fraction",
        "has_positive_progress",
        "max_progress",
        "mean_progress",
        "progress_standard_deviation",
        "activity_standard_deviation",
    ]
    summary = {
        split: {
            name: float(
                np.mean([record[name] for record in records if record["split"] == split])
            )
            for name in metric_names
        }
        for split in ("id", "ood")
    }
    checks = {
        "all_candidates_finite": all(
            math.isclose(summary[split]["all_finite"], 1.0)
            for split in ("id", "ood")
        ),
        "mean_unique_candidate_fraction_at_least_0_5": all(
            summary[split]["unique_candidate_fraction"] >= 0.5
            for split in ("id", "ood")
        ),
        "positive_candidate_available_in_at_least_90pct_windows": all(
            summary[split]["has_positive_progress"] >= 0.9
            for split in ("id", "ood")
        ),
        "mean_best_prefix_progress_above_0_1m": all(
            summary[split]["max_progress"] > 0.1 for split in ("id", "ood")
        ),
        "mean_progress_std_above_0_05m": all(
            summary[split]["progress_standard_deviation"] > 0.05
            for split in ("id", "ood")
        ),
        "mean_trajectory_diversity_positive": all(
            summary[split]["candidate_diversity"] > 0.05
            for split in ("id", "ood")
        ),
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "rule_status": "post-pilot operational adapter gate; not preregistered",
        "checkpoint": args.checkpoint,
        "episodes_per_split": args.episodes,
        "windows": args.windows,
        "num_robots": args.num_robots,
        "num_candidates": args.num_candidates,
        "oversample_factor": args.oversample_factor,
        "task_prefilter": not args.disable_task_prefilter,
        "goal_direction": args.goal_direction,
        "checks": checks,
        "summary": summary,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "window_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
