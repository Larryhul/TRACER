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


POLICIES = (
    "task-only",
    "no-belief-joint",
    "reset-additive",
    "reset-joint",
    "persistent-additive",
    "persistent-joint",
    "recurrent-joint",
    "oracle-joint",
)
METRICS = (
    "realized_total_cost",
    "task_progress",
    "safety_cost",
    "human_cost",
    "min_distance",
    "collision_rate",
    "oracle_regret",
    "oracle_hit_rate",
    "candidate_rank_spearman",
    "candidate_rank_kendall",
    "final_mode_accuracy",
    "runtime_ms_per_window",
)


def _cell_name(robots: int, scene: str, family: str) -> str:
    return f"R{robots}_{scene}_{family}"


def _config(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    output_dir: Path,
    robots: int,
    scene: str,
    family: str,
    data_seed: int,
    model_seed: int,
) -> dict[str, Any]:
    return {
        "simulation": {
            "episodes": args.episodes,
            "windows": args.windows,
            "history": args.history,
            "horizon": args.horizon,
            "num_robots": robots,
            "num_candidates": args.num_candidates,
            "execute_steps": args.execute_steps,
            "dt": args.dt,
            "response_noise": args.response_noise,
            "influence_radius": args.influence_radius,
            "entity_max_speed": args.entity_max_speed,
            "scene_family": scene,
            "response_family": family,
            "data_seed": data_seed,
            "splits": ["id", "ood"],
        },
        "planning": {
            "task_weight": args.task_weight,
            "safety_weight": args.safety_weight,
            "human_weight": args.human_weight,
            "comfort_scale": args.comfort_scale,
            "collision_margin": args.collision_margin,
            "task_steps": args.execute_steps,
        },
        "belief": {"likelihood_scale": args.likelihood_scale},
        "candidate_source": {"type": "synthetic"},
        "additional_policies": ["recurrent-joint"],
        "suite": {
            "seeds": [model_seed],
            # eval_persistent_closed_loop appends seed{seed} itself.
            "checkpoint_dir": str(checkpoint_dir.parent),
            "output_dir": str(output_dir),
            "num_threads_per_seed": args.num_threads,
        },
        "benchmark": {
            "cell": _cell_name(robots, scene, family),
            "num_robots": robots,
            "scene_family": scene,
            "response_family": family,
            "model_seed": model_seed,
            "data_seed": data_seed,
        },
    }


def _run(config_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    if not args.force and (output_dir / "summary.json").exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "sage.experiments.eval_persistent_closed_loop",
        "--config",
        str(config_path),
        "--device",
        args.device,
        "--output-dir",
        str(output_dir),
    ]
    log_path = output_dir / "closed_loop.log"
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
        raise RuntimeError(f"closed-loop evaluation failed; inspect {log_path}")


def _finite(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rows(
    summary_path: Path,
    robots: int,
    scene: str,
    family: str,
    model_seed: int,
    data_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for split in ("id", "ood"):
        split_payload = payload[split]
        for policy in POLICIES:
            if policy not in split_payload:
                raise ValueError(f"summary is missing policy {policy!r} in {summary_path}")
            metrics = split_payload[policy]
            row = {
                "split": split,
                "num_robots": robots,
                "scene_family": scene,
                "response_family": family,
                "model_seed": model_seed,
                "data_seed": data_seed,
                "policy": policy,
            }
            row.update({metric: _finite(metrics[metric]["mean"]) for metric in METRICS})
            records.append(row)
        for name, value in split_payload["contrasts"].items():
            contrasts.append(
                {
                    "split": split,
                    "num_robots": robots,
                    "scene_family": scene,
                    "response_family": family,
                    "model_seed": model_seed,
                    "data_seed": data_seed,
                    "contrast": name,
                    "value": _finite(value),
                }
            )
    return records, contrasts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_names = ("split", "num_robots", "scene_family", "response_family", "policy")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in key_names)
        groups.setdefault(key, []).append(row)
    result = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        row = dict(zip(key_names, key))
        row["n_runs"] = len(values)
        for metric in METRICS:
            array = np.asarray([_finite(value[metric]) for value in values])
            finite = array[np.isfinite(array)]
            row[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[f"{metric}_sd"] = (
                float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
            )
        result.append(row)
    return result


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Closed-Loop Simulation Benchmark Matrix",
        "",
        "The matrix evaluates finite joint candidate selection under matched",
        "scene/response mechanisms. Positive `reset_minus_persistent_total_cost`,",
        "`recurrent_minus_persistent_total_cost`, and `additive_minus_joint_total_cost`",
        "favor the persistent joint policy. These are bounded integration results,",
        "not universal planner superiority claims.",
        "",
        "| R | Split | Scene | Mechanism | Policy | Cost | Task | Human | Regret | Hit |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {num_robots} | {split} | {scene_family} | {response_family} | {policy} | "
            "{realized_total_cost_mean:.4f} | {task_progress_mean:.4f} | "
            "{human_cost_mean:.4f} | {oracle_regret_mean:.4f} | "
            "{oracle_hit_rate_mean:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-condition closed-loop benchmark matrix.")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--robots", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--scene-families", nargs="+", default=["radial", "corridor", "crossing", "bottleneck"])
    parser.add_argument("--response-families", nargs="+", default=["pincer", "speed", "lateral", "threshold"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027])
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--response-noise", type=float, default=0.18)
    parser.add_argument("--influence-radius", type=float, default=4.0)
    parser.add_argument("--entity-max-speed", type=float, default=3.0)
    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument("--safety-weight", type=float, default=2.0)
    parser.add_argument("--human-weight", type=float, default=0.6)
    parser.add_argument("--comfort-scale", type=float, default=1.0)
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument(
        "--likelihood-scale",
        type=float,
        default=0.05,
        help="Frozen Formal-v1 H2/H3 likelihood scale; override only for diagnostics.",
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
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
        "policies": list(POLICIES),
        "likelihood_scale": args.likelihood_scale,
        "cells": [
            {"num_robots": robots, "scene_family": scene, "response_family": family}
            for robots, scene, family in cells
        ],
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    all_rows: list[dict[str, Any]] = []
    all_contrasts: list[dict[str, Any]] = []
    for cell_index, (robots, scene, family) in enumerate(cells, start=1):
        for model_seed in args.seeds:
            data_seed = 910000 + cell_index * 1000 + model_seed
            name = _cell_name(robots, scene, family)
            checkpoint_dir = checkpoint_root / name / f"seed{model_seed}"
            if not (checkpoint_dir / "response.pt").exists():
                raise FileNotFoundError(
                    f"missing H2 checkpoint for {name}/seed{model_seed}: {checkpoint_dir}"
                )
            cell_dir = output_root / name / f"seed{model_seed}"
            config = _config(
                args,
                checkpoint_dir,
                cell_dir,
                robots,
                scene,
                family,
                data_seed,
                model_seed,
            )
            config_path = cell_dir / "config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(
                f"[{cell_index}/{len(cells)}] cell={name} seed={model_seed} device={args.device}",
                flush=True,
            )
            _run(config_path, cell_dir, args)
            rows, contrasts = _rows(cell_dir / "summary.json", robots, scene, family, model_seed, data_seed)
            _write_csv(cell_dir / "matrix_records.csv", rows)
            _write_csv(cell_dir / "matrix_contrasts.csv", contrasts)
            all_rows.extend(rows)
            all_contrasts.extend(contrasts)
            aggregate = _aggregate(all_rows)
            _write_csv(output_root / "summary.csv", aggregate)
            _write_markdown(output_root / "summary.md", aggregate)
            _write_csv(output_root / "records.csv", all_rows)
            _write_csv(output_root / "contrasts.csv", all_contrasts)
            print(f"completed cell={name} seed={model_seed} rows={len(rows)}", flush=True)
    protocol["status"] = "completed"
    protocol["completed_rows"] = len(all_rows)
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "summary.json").write_text(
        json.dumps(
            {"protocol": protocol, "summary": _aggregate(all_rows), "contrasts": all_contrasts},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved={output_root} rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
