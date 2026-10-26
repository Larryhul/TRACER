from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from sage.data.synthetic_navigation import collate_synthetic_scene_batch
from sage.response.causal_synthetic import make_causal_response_dataset_from_config
from sage.response.data import (
    build_entity_response_batch,
    shuffle_robot_candidates_within_groups,
)
from sage.response.factorial import (
    interaction_recovery_metrics,
    pairwise_factorial_interactions,
)
from sage.response.metrics import response_metrics
from sage.response.models import ResponseModelConfig, TeamResponsePredictor


VARIANTS = ("history", "nearest", "additive", "matched-additive", "joint")
EVALUATED_METHODS = (*VARIANTS, "joint-without-pair")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _cell_config(
    args: argparse.Namespace,
    root: Path,
    robots: int,
    scene_family: str,
    response_family: str,
    mode: str,
    data_seed: int,
) -> dict[str, Any]:
    cell_name = f"R{robots}_{scene_family}_{response_family}_{mode}"
    cell_root = root / cell_name
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
                "interventions_per_scene": 2**robots,
                "num_robots": robots,
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
                "scene_family": scene_family,
                "response_family": response_family,
                "cache_samples": True,
            },
        },
        "model": {
            "num_robots": robots,
            "hidden_dim": args.hidden_dim,
            "num_heads": 4,
            "architecture": "tracer",
            "graph_layers": 2,
            "dropout": 0.1,
            "min_log_std": -4.0,
            "max_log_std": 2.0,
        },
        "training": {
            "variants": list(VARIANTS),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "num_workers": 0,
            "nested_residual_training": True,
            "prefix_loss_weight": 1.0,
            "checkpoint_rule": "best_val_nll",
            "output_dir": str(cell_root),
        },
        "evaluation": {"interaction_radius": args.influence_radius},
        "benchmark": {
            "cell": cell_name,
            "mode": mode,
            "scene_family": scene_family,
            "response_family": response_family,
            "num_robots": robots,
            "data_seed": data_seed,
        },
    }


def _train_cell(
    config_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    seed: int,
    data_seed: int,
) -> None:
    expected = [output_dir / f"{variant}.pt" for variant in VARIANTS]
    if not args.force and all(path.exists() for path in expected):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "sage.experiments.train_response_baselines",
        "--config",
        str(config_path),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--data-seed",
        str(data_seed),
        "--output-dir",
        str(output_dir),
        "--variants",
        *VARIANTS,
    ]
    log_path = output_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env={**os.environ, "PYTHONPATH": str(Path.cwd())},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"training failed for {output_dir}; inspect {log_path}")


def _load_models(
    output_dir: Path,
    device: str,
    robots: int,
) -> dict[str, TeamResponsePredictor]:
    models: dict[str, TeamResponsePredictor] = {}
    for variant in VARIANTS:
        checkpoint = torch.load(
            output_dir / f"{variant}.pt", map_location=device, weights_only=True
        )
        config = ResponseModelConfig(**checkpoint["model_config"])
        if config.num_robots != robots:
            raise ValueError(
                f"checkpoint {variant} has R={config.num_robots}, expected R={robots}"
            )
        model = TeamResponsePredictor(variant, config).to(device)
        model.load_state_dict(checkpoint["model"])
        models[variant] = model.eval()
    return models


def _mean_finite(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    return float(finite.mean()) if finite.numel() else float("nan")


def _scene_rows(
    dataset,
    models: dict[str, TeamResponsePredictor],
    device: str,
    robots: int,
    interaction_radius: float,
    mode: str,
    scene_family: str,
    response_family: str,
    model_seed: int,
    data_seed: int,
    test_chunk: int,
) -> list[dict[str, Any]]:
    intervention_count = dataset.interventions_per_scene
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, dataset.num_base_scenes, test_chunk):
            stop = min(start + test_chunk, dataset.num_base_scenes)
            samples = []
            for base_scene_id in range(start, stop):
                first = base_scene_id * intervention_count
                samples.extend(dataset[first + intervention] for intervention in range(intervention_count))
            scene = collate_synthetic_scene_batch(samples)
            batch = build_entity_response_batch(
                scene, num_robots=robots, interaction_radius=interaction_radius
            ).to(device)
            entities = dataset.num_entities
            group_count = stop - start
            target = batch.target_future_vel.reshape(
                group_count, intervention_count, entities, dataset.horizon, 2
            )
            predictions: dict[str, torch.Tensor] = {}
            log_stds: dict[str, torch.Tensor] = {}
            for method, model in models.items():
                output = model(batch)
                predictions[method] = output["mean"].reshape(
                    group_count, intervention_count, entities, dataset.horizon, 2
                )
                log_stds[method] = output["log_std"].reshape(
                    group_count, intervention_count, entities, dataset.horizon, 2
                )
            without_pair = models["joint"](batch, include_pair_interactions=False)
            predictions["joint-without-pair"] = models["joint"](
                batch, include_pair_interactions=False
            )["mean"].reshape(group_count, intervention_count, entities, dataset.horizon, 2)
            log_stds["joint-without-pair"] = without_pair["log_std"].reshape(
                group_count, intervention_count, entities, dataset.horizon, 2
            )
            shuffled_batch = shuffle_robot_candidates_within_groups(
                batch,
                group_size=intervention_count * entities,
                branch_size=entities,
            )
            shuffled = models["joint"](shuffled_batch)
            predictions["shuffled-joint"] = shuffled["mean"].reshape(
                group_count, intervention_count, entities, dataset.horizon, 2
            )
            log_stds["shuffled-joint"] = shuffled["log_std"].reshape(
                group_count, intervention_count, entities, dataset.horizon, 2
            )
            for local in range(group_count):
                target_scene = target[local]
                target_pair = pairwise_factorial_interactions(target_scene, robots, dim=0)
                base_scene_id = int(samples[local * intervention_count]["base_scene_id"])
                for method, prediction_all in predictions.items():
                    prediction = prediction_all[local]
                    metric = response_metrics(
                        prediction.reshape(-1, dataset.horizon, 2),
                        log_stds[method][local].reshape(-1, dataset.horizon, 2),
                        target_scene.reshape(-1, dataset.horizon, 2),
                        batch.dt,
                    )
                    pred_pair = pairwise_factorial_interactions(prediction, robots, dim=0)
                    interaction = interaction_recovery_metrics(pred_pair, target_pair)
                    rows.append(
                        {
                            "mode": mode,
                            "scene_family": scene_family,
                            "response_family": response_family,
                            "num_robots": robots,
                            "model_seed": model_seed,
                            "data_seed": data_seed,
                            "base_scene_id": base_scene_id,
                            "method": method,
                            "nll": _mean_finite(metric["nll"]),
                            "ade": _mean_finite(metric["ade"]),
                            "fde": _mean_finite(metric["fde"]),
                            "interaction_rmse": interaction["rmse"],
                            "interaction_r2": interaction["zero_baseline_r2"],
                            "interaction_target_rms": interaction["target_rms"],
                            "interaction_prediction_rms": interaction["prediction_rms"],
                            "interaction_energy_ratio": interaction["energy_ratio"],
                        }
                    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["mode"],
                row["scene_family"],
                row["response_family"],
                row["num_robots"],
                row["method"],
            )
        ].append(row)
    summaries = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        result = dict(zip(("mode", "scene_family", "response_family", "num_robots", "method"), key))
        result["n_scenes"] = len(values)
        for metric in (
            "nll",
            "ade",
            "fde",
            "interaction_rmse",
            "interaction_r2",
            "interaction_target_rms",
            "interaction_prediction_rms",
            "interaction_energy_ratio",
        ):
            array = np.asarray([float(row[metric]) for row in values], dtype=float)
            finite = array[np.isfinite(array)]
            result[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            result[f"{metric}_sd"] = (
                float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
            )
        summaries.append(result)
    return summaries


def _write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Causal Simulation Benchmark Matrix",
        "",
        "All rows are paired held-out base scenes. `joint` is the proposed joint response model;",
        "`matched-additive` controls for residual parameter capacity; `shuffled-joint` rotates",
        "candidate branches within each base scene as an action-permutation negative control. Results",
        "are diagnostic until the confirmatory selected cells",
        "are rerun with the registered scene count and model seeds.",
        "",
        "| R | Scene | Mechanism | Method | NLL | ADE | FDE | Pair RMSE | Pair R2 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {num_robots} | {scene_family} | {response_family} | {method} | "
            "{nll_mean:.4f} | {ade_mean:.4f} | {fde_mean:.4f} | "
            "{interaction_rmse_mean:.4f} | {interaction_r2_mean:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-R causal simulation benchmark matrix.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="reuse frozen cell checkpoints from a separate matrix directory",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--robots", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--scene-families", nargs="+", default=["radial", "corridor", "crossing", "bottleneck"])
    parser.add_argument("--response-families", nargs="+", default=["pincer", "speed", "lateral", "threshold"])
    parser.add_argument("--modes", nargs="+", choices=["none", "additive", "nonlinear"], default=["nonlinear"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027])
    parser.add_argument("--train-scenes", type=int, default=512)
    parser.add_argument("--val-scenes", type=int, default=128)
    parser.add_argument("--test-scenes", type=int, default=128)
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
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--test-chunk", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    cells = [
        (robots, scene, family, mode)
        for mode in args.modes
        for robots in args.robots
        for scene in args.scene_families
        for family in args.response_families
    ]
    if args.max_cells > 0:
        cells = cells[: args.max_cells]
    protocol = {
        "status": "running",
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cells": [
            {
                "num_robots": robots,
                "scene_family": scene,
                "response_family": family,
                "mode": mode,
            }
            for robots, scene, family, mode in cells
        ],
        "model_variants": list(VARIANTS),
        "independent_unit": "base_scene",
        "pairwise_interaction": "average factorial contrast over complement bits",
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for cell_index, (robots, scene_family, response_family, mode) in enumerate(cells, start=1):
        for model_seed in args.seeds:
            data_seed = 700000 + cell_index * 1000 + model_seed
            cell_name = f"R{robots}_{scene_family}_{response_family}_{mode}"
            cell_root = output_dir / cell_name / f"seed{model_seed}"
            checkpoint_cell_root = checkpoint_root / cell_name / f"seed{model_seed}"
            config = _cell_config(
                args,
                checkpoint_cell_root,
                robots,
                scene_family,
                response_family,
                mode,
                data_seed,
            )
            config_path = cell_root / "config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(
                f"[{cell_index}/{len(cells)}] cell={cell_name} seed={model_seed} device={args.device}",
                flush=True,
            )
            _set_seed(model_seed)
            _train_cell(config_path, checkpoint_cell_root, args, model_seed, data_seed)
            dataset = make_causal_response_dataset_from_config(
                config, split="test", effect_mode=mode, seed=data_seed
            )
            models = _load_models(checkpoint_cell_root, args.device, robots)
            rows = _scene_rows(
                dataset,
                models,
                args.device,
                robots,
                args.influence_radius,
                mode,
                scene_family,
                response_family,
                model_seed,
                data_seed,
                args.test_chunk,
            )
            _write_csv(cell_root / "paired_records.csv", rows)
            all_rows.extend(rows)
            summaries = _summary(all_rows)
            _write_csv(output_dir / "summary.csv", summaries)
            _write_markdown(output_dir / "summary.md", summaries)
            (output_dir / "records.csv").unlink(missing_ok=True)
            _write_csv(output_dir / "records.csv", all_rows)
            print(f"completed cell={cell_name} seed={model_seed} rows={len(rows)}", flush=True)

    protocol["status"] = "completed"
    protocol["completed_rows"] = len(all_rows)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps({"protocol": protocol, "summary": _summary(all_rows)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved={output_dir} rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
