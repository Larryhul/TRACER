"""Evaluate response models on continuous, unseen team-action interventions.

The formal H1 corpus uses a two-level factorial intervention. This benchmark
keeps the same resettable simulator and response noise, but interpolates each
robot between its yielding and active trajectory at five continuous levels.
It tests support interpolation, candidate-response ranking, and a generic
direct-concatenation team baseline without changing the Formal-v1 gate.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from sage.data.synthetic_navigation import collate_synthetic_scene_batch
from sage.response.causal_synthetic import CausalResponseSpec, CausalTeamResponseDataset
from sage.response.data import (
    build_entity_response_batch,
    shuffle_robot_candidates_within_groups,
)
from sage.response.metrics import response_metrics
from sage.response.models import ResponseModelConfig, TeamResponsePredictor


STANDARD_VARIANTS = ("history", "nearest", "additive", "matched-additive", "joint")
VARIANTS = (*STANDARD_VARIANTS, "team-concat")
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _load_checkpoint(path: Path, device: str) -> TeamResponsePredictor:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TeamResponsePredictor(
        str(checkpoint["variant"]), ResponseModelConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _config(args: argparse.Namespace, output_dir: Path, mode: str, data_seed: int) -> dict[str, Any]:
    return {
        "dataset": {
            "source": "causal_response",
            "history": args.history,
            "horizon": args.horizon,
            "dt": args.dt,
            "causal_response": {
                "train_base_scenes": args.train_scenes,
                "val_base_scenes": args.val_scenes,
                "test_base_scenes": args.test_scenes,
                "interventions_per_scene": 4,
                "num_robots": 2,
                "num_entities": args.entities,
                "seed": data_seed,
                "area_size": args.area_size,
                "influence_radius": args.influence_radius,
                "individual_gain": 1.0,
                "joint_gain": args.joint_gain,
                "joint_slowdown": args.joint_slowdown,
                "joint_lateral_gain": args.joint_lateral_gain,
                "entity_noise": args.entity_noise,
                "entity_max_speed": 3.0,
                "cache_samples": True,
            },
        },
        "model": {
            "num_robots": 2,
            "hidden_dim": args.hidden_dim,
            "num_heads": 4,
            "dropout": args.dropout,
            "min_log_std": -4.0,
            "max_log_std": 2.0,
        },
        "training": {
            "variants": ["team-concat"],
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "num_workers": 0,
            "nested_residual_training": False,
            "checkpoint_rule": "best_val_nll",
            "output_dir": str(output_dir),
        },
        "evaluation": {"interaction_radius": args.influence_radius},
        "benchmark": {
            "mode": mode,
            "data_seed": data_seed,
            "levels": list(LEVELS),
        },
    }


def _train_team_concat(
    config_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    mode: str,
    seed: int,
    data_seed: int,
) -> None:
    checkpoint = output_dir / "team-concat.pt"
    if checkpoint.exists() and not args.force:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "sage.experiments.train_response_baselines",
        "--config",
        str(config_path),
        "--effect-mode",
        mode,
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--data-seed",
        str(data_seed),
        "--variants",
        "team-concat",
        "--output-dir",
        str(output_dir),
    ]
    log_path = output_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env={**__import__("os").environ, "PYTHONPATH": str(Path.cwd())},
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"team-concat training failed; inspect {log_path}")


def _continuous_actions(
    dataset: CausalTeamResponseDataset,
    base_scene_id: int,
    base: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return all 5^R interpolated action branches and their levels."""

    rng = dataset._action_rng(base_scene_id)
    robot_current = base["robot_current"]
    base_direction = -robot_current
    base_direction /= np.maximum(
        np.linalg.norm(base_direction, axis=-1, keepdims=True), 1e-6
    )
    base_angle = np.arctan2(base_direction[:, 1], base_direction[:, 0])
    active_offsets = rng.normal(0.0, 0.12, size=dataset.num_robots)
    yielding_offsets = rng.choice([-1.0, 1.0], size=dataset.num_robots) * rng.uniform(
        0.75, 1.10, size=dataset.num_robots
    )
    active_speeds = rng.uniform(1.35, 1.75, size=dataset.num_robots)
    yielding_speeds = rng.uniform(0.55, 0.85, size=dataset.num_robots)
    turns = rng.uniform(-0.10, 0.10, size=dataset.num_robots)
    levels = np.asarray(list(itertools.product(LEVELS, repeat=dataset.num_robots)), dtype=np.float32)
    offsets = yielding_offsets[None] + levels * (active_offsets - yielding_offsets)[None]
    speeds = yielding_speeds[None] + levels * (active_speeds - yielding_speeds)[None]
    steps = np.arange(dataset.horizon, dtype=np.float32)[None, None, :]
    angles = base_angle[None, :, None] + offsets[:, :, None] + turns[None, :, None] * steps
    velocity = np.stack(
        [np.cos(angles) * speeds[:, :, None], np.sin(angles) * speeds[:, :, None]],
        axis=-1,
    )
    return velocity.astype(np.float32), levels


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = _rank(first)
    second_rank = _rank(second)
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = float(np.linalg.norm(first_rank) * np.linalg.norm(second_rank))
    return float(np.dot(first_rank, second_rank) / denominator) if denominator > 1e-12 else 0.0


def _finite_mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    return float(finite.mean()) if finite.numel() else float("nan")


def _build_scene(
    dataset: CausalTeamResponseDataset,
    base_scene_id: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    base = dataset._base_scene(base_scene_id)
    robot_future, levels = _continuous_actions(dataset, base_scene_id, base)
    responses = np.stack(
        [dataset._entity_response(base, future, dataset.effect_mode) for future in robot_future],
        axis=0,
    )
    samples = tuple(
        dataset._make_sample(base_scene_id, branch, base, robot_future[branch], responses[branch])
        for branch in range(len(levels))
    )
    scene = collate_synthetic_scene_batch(samples)
    return scene, responses, levels


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["effect_mode"]), int(row["model_seed"]), str(row["variant"]))
        grouped.setdefault(key, []).append(row)
    metrics = (
        "nll",
        "ade",
        "fde",
        "action_effect_rmse",
        "action_effect_rank_spearman",
        "action_effect_top1",
        "action_effect_top3",
    )
    summary = []
    for (mode, seed, variant), values in sorted(grouped.items()):
        result: dict[str, Any] = {
            "effect_mode": mode,
            "model_seed": seed,
            "variant": variant,
            "n_scenes": len({int(value["base_scene_id"]) for value in values}),
            "n_rows": len(values),
        }
        for metric in metrics:
            array = np.asarray([float(value[metric]) for value in values], dtype=np.float64)
            finite = array[np.isfinite(array)]
            result[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            result[f"{metric}_sd"] = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
        summary.append(result)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def _evaluate(
    dataset: CausalTeamResponseDataset,
    models: dict[str, TeamResponsePredictor],
    mode: str,
    model_seed: int,
    data_seed: int,
    device: str,
    scenes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_scene_id in range(scenes):
        scene, targets_np, levels = _build_scene(dataset, base_scene_id)
        batch = build_entity_response_batch(scene, num_robots=2, interaction_radius=4.0).to(device)
        targets = batch.target_future_vel.reshape(len(levels), dataset.num_entities, dataset.horizon, 2)
        for variant, model in models.items():
            output = model(batch)
            metrics = response_metrics(output["mean"], output["log_std"], batch.target_future_vel, batch.dt)
            prediction = output["mean"].reshape_as(targets)
            target_delta = targets - targets[:1]
            prediction_delta = prediction - prediction[:1]
            true_score = target_delta.square().mean(dim=(1, 2, 3)).detach().cpu().numpy()
            pred_score = prediction_delta.square().mean(dim=(1, 2, 3)).detach().cpu().numpy()
            rank = _spearman(true_score, pred_score)
            # Rank the strongest response effect. Ranking the smallest effect
            # would make the zero-level reference branch a trivial top-1.
            true_order = np.argsort(-true_score)
            pred_order = np.argsort(-pred_score)
            rows.append(
                {
                    "effect_mode": mode,
                    "model_seed": model_seed,
                    "data_seed": data_seed,
                    "base_scene_id": int(base_scene_id),
                    "variant": variant,
                    "nll": _finite_mean(metrics["nll"]),
                    "ade": _finite_mean(metrics["ade"]),
                    "fde": _finite_mean(metrics["fde"]),
                    "action_effect_rmse": float(torch.sqrt(
                        (prediction_delta - target_delta).square().mean()
                    )),
                    "action_effect_rank_spearman": rank,
                    "action_effect_top1": float(pred_order[0] == true_order[0]),
                    "action_effect_top3": float(
                        len(set(pred_order[:3].tolist()) & set(true_order[:3].tolist())) / 3.0
                    ),
                    "mean_level": float(levels.mean()),
                }
            )
        shuffled_batch = shuffle_robot_candidates_within_groups(
            batch,
            group_size=len(levels) * dataset.num_entities,
            branch_size=dataset.num_entities,
        )
        shuffled = models["joint"](shuffled_batch)
        shuffled_metrics = response_metrics(
            shuffled["mean"], shuffled["log_std"], batch.target_future_vel, batch.dt
        )
        rows.append(
            {
                "effect_mode": mode,
                "model_seed": model_seed,
                "data_seed": data_seed,
                "base_scene_id": int(base_scene_id),
                "variant": "shuffled-joint",
                "nll": _finite_mean(shuffled_metrics["nll"]),
                "ade": _finite_mean(shuffled_metrics["ade"]),
                "fde": _finite_mean(shuffled_metrics["fde"]),
                "action_effect_rmse": float("nan"),
                "action_effect_rank_spearman": float("nan"),
                "action_effect_top1": float("nan"),
                "action_effect_top3": float("nan"),
                "mean_level": float(levels.mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous action-support response benchmark.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-root", default="experiments/pilot/h1_paired_rescue_stage1_v2/checkpoints")
    parser.add_argument("--context-checkpoint-root", default="experiments/pilot/h1_context_baselines_medium_20260819/checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modes", nargs="+", choices=["none", "additive", "nonlinear"], default=["none", "additive", "nonlinear"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--train-scenes", type=int, default=4096)
    parser.add_argument("--val-scenes", type=int, default=256)
    parser.add_argument("--test-scenes", type=int, default=256)
    parser.add_argument("--entities", type=int, default=12)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--area-size", type=float, default=10.0)
    parser.add_argument("--influence-radius", type=float, default=4.0)
    parser.add_argument("--joint-gain", type=float, default=3.0)
    parser.add_argument("--joint-slowdown", type=float, default=1.0)
    parser.add_argument("--joint-lateral-gain", type=float, default=1.25)
    parser.add_argument("--entity-noise", type=float, default=0.015)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "running",
        "device": args.device,
        "levels": list(LEVELS),
        "branches_per_scene": len(LEVELS) ** 2,
        "independent_unit": "base_scene",
        "variants": [*VARIANTS, "shuffled-joint"],
        "standard_checkpoint_root": args.checkpoint_root,
        "continuous_test_data_seed": 328731,
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    all_rows: list[dict[str, Any]] = []
    for mode in args.modes:
        for model_seed in args.seeds:
            data_seed = 328731
            config = _config(
                args,
                output_root / "team_concat_checkpoints" / mode / f"seed{model_seed}",
                mode,
                318731,
            )
            config_path = output_root / "configs" / mode / f"seed{model_seed}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            concat_dir = output_root / "team_concat_checkpoints" / mode / f"seed{model_seed}"
            _train_team_concat(config_path, concat_dir, args, mode, model_seed, 318731)
            response_root = Path(args.checkpoint_root) / mode / f"seed{model_seed}"
            context_root = Path(args.context_checkpoint_root) / mode / f"seed{model_seed}"
            models = {
                variant: _load_checkpoint(
                    (context_root if variant in {"history", "nearest"} else response_root)
                    / f"{variant}.pt",
                    args.device,
                )
                for variant in STANDARD_VARIANTS
            }
            models["team-concat"] = _load_checkpoint(concat_dir / "team-concat.pt", args.device)
            dataset = CausalTeamResponseDataset(
                CausalResponseSpec(
                    split="test",
                    num_base_scenes=args.test_scenes,
                    interventions_per_scene=4,
                    history=args.history,
                    horizon=args.horizon,
                    num_robots=2,
                    num_entities=args.entities,
                    dt=args.dt,
                    seed=data_seed,
                    effect_mode=mode,
                    area_size=args.area_size,
                    influence_radius=args.influence_radius,
                    individual_gain=1.0,
                    joint_gain=args.joint_gain,
                    joint_slowdown=args.joint_slowdown,
                    joint_lateral_gain=args.joint_lateral_gain,
                    entity_noise=args.entity_noise,
                    entity_max_speed=3.0,
                    cache_samples=False,
                )
            )
            print(f"evaluating mode={mode} seed={model_seed} device={args.device}", flush=True)
            rows = _evaluate(dataset, models, mode, model_seed, data_seed, args.device, args.test_scenes)
            _write_csv(output_root / mode / f"seed{model_seed}" / "records.csv", rows)
            all_rows.extend(rows)
            _write_csv(output_root / "records.csv", all_rows)
            _write_csv(output_root / "summary.csv", _summarize(all_rows))
    protocol["status"] = "completed"
    protocol["completed_rows"] = len(all_rows)
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps({"protocol": protocol, "summary": _summarize(all_rows)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved={output_root} rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
