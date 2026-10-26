"""Evaluate joint versus matched-additive TRACER on independent test episodes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sage.experiments.eval_persistent_closed_loop import _load_model
from sage.response.independent_social_force import (
    IndependentSocialForceDataset,
    IndependentSocialForceDatasetSpec,
)
from sage.response.persistent_models import trajectory_mode_log_likelihood


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    dt: float,
) -> dict[str, dict[str, float | int]]:
    model.eval()
    values: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in ("ade", "fde", "nll")}
        for name in ("joint", "additive")
    }
    for batch in loader:
        history = batch["entity_history"].to(device)
        current = batch["robot_current"].to(device)
        candidates = batch["robot_candidates"].to(device)
        target_velocity = batch["target_future_vel"].to(device)
        mode = batch["mode"].to(device)
        for name, include_pair_interactions in (("joint", True), ("additive", False)):
            output = model(
                history,
                current,
                candidates,
                include_pair_interactions=include_pair_interactions,
            )
            selected_mean = output["mean"].gather(
                2,
                mode[:, None, None, None, None].expand(
                    -1, history.shape[1], 1, model.config.horizon, 2
                ),
            ).squeeze(2)
            prediction = torch.cumsum(selected_mean * dt, dim=-2)
            target = torch.cumsum(target_velocity * dt, dim=-2)
            error = torch.linalg.vector_norm(prediction - target, dim=-1)
            values[name]["ade"].extend(error.mean(dim=-1).flatten().cpu().tolist())
            values[name]["fde"].extend(error[..., -1].flatten().cpu().tolist())
            log_likelihood = trajectory_mode_log_likelihood(
                output["mean"], output["log_std"], target_velocity
            )
            selected_nll = -log_likelihood.gather(
                -1, mode[:, None, None].expand(-1, history.shape[1], 1)
            ).squeeze(-1)
            values[name]["nll"].extend(selected_nll.flatten().cpu().tolist())
    return {
        name: {
            metric: {"mean": _mean(metric_values), "count": len(metric_values)}
            for metric, metric_values in metrics.items()
        }
        for name, metrics in values.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = IndependentSocialForceDataset(
        IndependentSocialForceDatasetSpec(
            split="test",
            num_episodes=args.test_episodes,
            windows=args.windows,
            history=args.history,
            horizon=args.horizon,
            num_robots=args.num_robots,
            num_candidates=args.num_candidates,
            execute_steps=args.execute_steps,
            dt=args.dt,
            response_noise=args.response_noise,
            entity_max_speed=args.entity_max_speed,
            scene_family=args.scene_family,
            seed=args.data_seed,
            cache_samples=True,
        )
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_results: dict[str, Any] = {}
    for seed in args.seeds:
        model = _load_model(
            Path(args.checkpoint_dir) / f"seed{seed}" / "response.pt",
            args.device,
        )
        result = evaluate_model(
            model,
            DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0),
            args.device,
            args.dt,
        )
        result["additive_minus_joint"] = {
            metric: {
                "mean": result["additive"][metric]["mean"]
                - result["joint"][metric]["mean"],
                "count": result["joint"][metric]["count"],
            }
            for metric in ("ade", "fde", "nll")
        }
        seed_results[str(seed)] = result
        (output_dir / f"seed{seed}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    aggregate: dict[str, Any] = {}
    for method in ("joint", "additive", "additive_minus_joint"):
        aggregate[method] = {}
        for metric in ("ade", "fde", "nll"):
            seed_values = [seed_results[seed][method][metric]["mean"] for seed in seed_results]
            aggregate[method][metric] = {
                "mean": _mean(seed_values),
                "std": float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0,
                "count": len(seed_values),
            }
    result = {
        "protocol": "independent-social-force-h1-response-ablation-v1",
        "backend": "independent-social-force-v2-nonlinear",
        "seeds": args.seeds,
        "test_episodes_per_seed": args.test_episodes,
        "aggregate": aggregate,
        "per_seed": seed_results,
        "interpretation": "positive additive_minus_joint favors the joint pair residual",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Independent H1 Response Ablation",
        "",
        "Positive additive-minus-joint values favor TRACER's joint pair residual.",
        "",
        "| Metric | Joint | Matched additive | Additive - joint |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("ade", "fde", "nll"):
        lines.append(
            f"| {metric.upper()} | "
            f"{aggregate['joint'][metric]['mean']:.6f} +/- {aggregate['joint'][metric]['std']:.6f} | "
            f"{aggregate['additive'][metric]['mean']:.6f} +/- {aggregate['additive'][metric]['std']:.6f} | "
            f"{aggregate['additive_minus_joint'][metric]['mean']:.6f} |"
        )
    lines += [
        "",
        "This is an independent-environment response-structure diagnostic. It is "
        "not a real-human causal result and does not imply universal planner superiority.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="experiments/independent_social_force_formal_20260902")
    parser.add_argument("--output-dir", default="experiments/independent_social_force_formal_20260902/h1_response_ablation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2031, 2039, 2053, 2063])
    parser.add_argument("--data-seed", type=int, default=418731)
    parser.add_argument("--test-episodes", type=int, default=256)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.3)
    parser.add_argument("--response-noise", type=float, default=0.08)
    parser.add_argument("--entity-max-speed", type=float, default=2.2)
    parser.add_argument("--scene-family", default="crossing")
    parser.add_argument("--batch-size", type=int, default=64)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
