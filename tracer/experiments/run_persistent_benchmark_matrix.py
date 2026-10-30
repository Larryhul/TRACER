from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


METHODS = ("no-belief", "reset", "persistent", "shuffled-id", "recurrent", "oracle")


def _cell_name(robots: int, scene: str, family: str) -> str:
    return f"R{robots}_{scene}_{family}"


def _config(
    args: argparse.Namespace,
    output_dir: Path,
    robots: int,
    scene: str,
    family: str,
    data_seed: int,
) -> dict[str, Any]:
    return {
        "dataset": {
            "train_episodes": args.train_episodes,
            "val_episodes": args.val_episodes,
            "ood_episodes": args.ood_episodes,
            "windows": args.windows,
            "history": args.history,
            "horizon": args.horizon,
            "num_robots": robots,
            "num_entities": args.entities,
            "active_entities": args.active_entities,
            "shared_team_actions": True,
            "dt": args.dt,
            "seed": data_seed,
            "data_seed": data_seed,
            "response_noise": args.response_noise,
            "influence_radius": args.influence_radius,
            "execute_steps": args.execute_steps,
            "entity_max_speed": args.entity_max_speed,
            "scene_family": scene,
            "response_family": family,
            "ood_action_policy": args.ood_action_policy,
            "cache_samples": True,
        },
        "model": {
            "hidden_dim": args.hidden_dim,
            "recurrent_hidden_dim": args.recurrent_hidden_dim,
            "dropout": args.dropout,
            "recurrent_use_entity_history": args.recurrent_use_entity_history,
        },
        "training": {
            "epochs": args.epochs,
            "response_epochs": args.epochs,
            "recurrent_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "recurrent_lr": args.recurrent_lr,
            "weight_decay": args.weight_decay,
            "mode_discrimination_weight": args.mode_discrimination_weight,
            "mode_temperature": args.mode_temperature,
            "num_workers": 0,
            "num_threads": args.num_threads,
        },
        "evaluation": {
            "id_split": "val",
            "likelihood_scale": args.likelihood_scale,
            "confidence_threshold": args.confidence_threshold,
        },
        "output_dir": str(output_dir),
        "benchmark": {
            "cell": _cell_name(robots, scene, family),
            "num_robots": robots,
            "scene_family": scene,
            "response_family": family,
            "data_seed": data_seed,
            "ood_action_policy": args.ood_action_policy,
        },
    }


def _train(
    config_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    seed: int,
) -> None:
    required = (output_dir / "response.pt", output_dir / "recurrent.pt")
    if not args.force and all(path.exists() for path in required):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tracer.experiments.train_persistent_trajectory_response",
        "--config",
        str(config_path),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    log_path = output_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env={**os.environ, "PYTHONPATH": str(Path.cwd())},
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"persistent training failed; inspect {log_path}")


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _rows(
    output_dir: Path,
    robots: int,
    scene: str,
    family: str,
    seed: int,
    data_seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in ("id", "ood"):
        payload = json.loads(
            (output_dir / ("metrics.json" if split == "id" else "ood_metrics.json")).read_text(
                encoding="utf-8"
            )
        )
        for method in METHODS:
            metrics = payload[method]
            records.append(
                {
                    "split": split,
                    "num_robots": robots,
                    "scene_family": scene,
                    "response_family": family,
                    "model_seed": seed,
                    "data_seed": data_seed,
                    "method": method,
                    "predictive_trajectory_nll": _finite(metrics["predictive_trajectory_nll"]),
                    "trajectory_ade": _finite(metrics["trajectory_ade"]),
                    "trajectory_fde": _finite(metrics["trajectory_fde"]),
                    "decision_mode_nll": _finite(metrics["decision_mode_nll"]),
                    "decision_mode_accuracy": _finite(metrics["decision_mode_accuracy"]),
                    "mode_nll": _finite(metrics["mode_nll"]),
                    "mode_accuracy": _finite(metrics["mode_accuracy"]),
                    "mode_brier": _finite(metrics["mode_brier"]),
                    "posterior_entropy": _finite(metrics["posterior_entropy"]),
                    "final_mode_accuracy": _finite(metrics["final_mode_accuracy"]),
                    "windows_to_confidence": _finite(metrics["windows_to_confidence"]),
                    "high_trajectory_ade": _finite(metrics["high_trajectory_ade"]),
                    "main_evaluation_count": int(metrics["main_evaluation_count"]),
                    "active_observation_count": int(metrics["active_observation_count"]),
                }
            )
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("split", "num_robots", "scene_family", "response_family", "method")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        grouped.setdefault(key, []).append(row)
    metrics = (
        "predictive_trajectory_nll",
        "trajectory_ade",
        "trajectory_fde",
        "decision_mode_nll",
        "decision_mode_accuracy",
        "mode_nll",
        "mode_accuracy",
        "mode_brier",
        "posterior_entropy",
        "final_mode_accuracy",
        "windows_to_confidence",
        "high_trajectory_ade",
    )
    result: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        row = dict(zip(keys, key))
        row["n_runs"] = len(values)
        for metric in metrics:
            array = np.asarray([_finite(value[metric]) for value in values])
            finite = array[np.isfinite(array)]
            row[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[f"{metric}_sd"] = (
                float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
            )
        result.append(row)
    return result


def _write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Persistent-Belief Simulation Matrix",
        "",
        "Positive `reset-minus-persistent` and `recurrent-minus-persistent` values",
        "mean lower predictive trajectory NLL for persistent identity-bound belief.",
        "All ID/OOD rows retain the full baseline set and are diagnostic until the",
        "selected cells are rerun under the frozen confirmatory protocol.",
        "",
        "| R | Split | Scene | Mechanism | Method | Pred NLL | ADE | Mode Acc | Entropy | W80 |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {num_robots} | {split} | {scene_family} | {response_family} | {method} | "
            "{predictive_trajectory_nll_mean:.4f} | {trajectory_ade_mean:.4f} | "
            "{mode_accuracy_mean:.4f} | {posterior_entropy_mean:.4f} | "
            "{windows_to_confidence_mean:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-condition persistent-belief benchmark matrix.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--robots", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--scene-families", nargs="+", default=["radial", "corridor", "crossing", "bottleneck"])
    parser.add_argument("--response-families", nargs="+", default=["pincer", "speed", "lateral", "threshold"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027])
    parser.add_argument("--train-episodes", type=int, default=512)
    parser.add_argument("--val-episodes", type=int, default=128)
    parser.add_argument("--ood-episodes", type=int, default=128)
    parser.add_argument("--entities", type=int, default=10)
    parser.add_argument("--active-entities", type=int, default=3)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--response-noise", type=float, default=0.12)
    parser.add_argument("--influence-radius", type=float, default=4.0)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--entity-max-speed", type=float, default=3.0)
    parser.add_argument("--ood-action-policy", choices=["iid", "correlated"], default="correlated")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--recurrent-use-entity-history", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--recurrent-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mode-discrimination-weight", type=float, default=0.0)
    parser.add_argument("--mode-temperature", type=float, default=0.2)
    parser.add_argument("--likelihood-scale", type=float, default=0.2)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cells = [
        (robots, scene, family)
        for robots in args.robots
        for scene in args.scene_families
        for family in args.response_families
    ]
    if args.max_cells > 0:
        cells = cells[: args.max_cells]
    protocol: dict[str, Any] = {
        "status": "running",
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "independent_unit": "episode",
        "methods": list(METHODS),
        "cells": [
            {
                "num_robots": robots,
                "scene_family": scene,
                "response_family": family,
            }
            for robots, scene, family in cells
        ],
        "ood_action_policy": args.ood_action_policy,
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    all_rows: list[dict[str, Any]] = []
    for cell_index, (robots, scene, family) in enumerate(cells, start=1):
        for model_seed in args.seeds:
            data_seed = 810000 + cell_index * 1000 + model_seed
            name = _cell_name(robots, scene, family)
            cell_dir = output_root / name / f"seed{model_seed}"
            config = _config(args, cell_dir, robots, scene, family, data_seed)
            config_path = cell_dir / "config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(
                f"[{cell_index}/{len(cells)}] cell={name} seed={model_seed} device={args.device}",
                flush=True,
            )
            _train(config_path, cell_dir, args, model_seed)
            rows = _rows(cell_dir, robots, scene, family, model_seed, data_seed)
            with (cell_dir / "matrix_records.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            all_rows.extend(rows)
            summaries = _summary(all_rows)
            _write_csv(output_root / "summary.csv", summaries)
            _write_markdown(output_root / "summary.md", summaries)
            _write_csv(output_root / "records.csv", all_rows)
            print(f"completed cell={name} seed={model_seed} rows={len(rows)}", flush=True)
    protocol["status"] = "completed"
    protocol["completed_rows"] = len(all_rows)
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "summary.json").write_text(
        json.dumps({"protocol": protocol, "summary": _summary(all_rows)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved={output_root} rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
