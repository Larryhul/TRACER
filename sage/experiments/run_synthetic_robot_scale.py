"""Run a fixed-budget SAGE robot-count scale and single-robot stress sweep."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROBOT_COUNTS = (1, 3, 6)
BASELINES = ("constant_velocity", "direct_goal")


def _run(command: list[str], root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, text=True)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    print(f"completed_in_seconds={time.monotonic() - started:.1f} log={log_path}", flush=True)


def _read_single(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, got {len(rows)}")
    return rows[0]


def run(
    root: Path,
    output_dir: Path,
    run_dir: Path,
    robot_counts: tuple[int, ...] = ROBOT_COUNTS,
    dataset_seed: int = 6061,
    model_seed: int = 6061,
    epochs: int = 10,
    train_samples: int = 512,
    test_samples: int = 512,
    num_entities: int = 20,
    device: str = "cuda",
) -> dict[str, Any]:
    config = "configs/sage_synthetic.yaml"
    rows: list[dict[str, str]] = []
    for num_robots in robot_counts:
        scale_dir = run_dir / f"r{num_robots}"
        checkpoint = scale_dir / f"synthetic_sage_epoch{epochs}.pt"
        sage_csv = scale_dir / "sage.csv"
        train_command = [
            sys.executable,
            "-u",
            "-m",
            "sage.experiments.train_synthetic_sage",
            "--config",
            config,
            "--epochs",
            str(epochs),
            "--batch-size",
            "64",
            "--limit-samples",
            str(train_samples),
            "--num-robots",
            str(num_robots),
            "--num-entities",
            str(num_entities),
            "--save-every",
            str(epochs),
            "--seed",
            str(model_seed),
            "--dataset-seed",
            str(dataset_seed),
            "--device",
            device,
            "--save-dir",
            str(scale_dir),
        ]
        _run(train_command, root, scale_dir / "train.log")
        eval_command = [
            sys.executable,
            "-u",
            "-m",
            "sage.experiments.eval_synthetic_sage",
            "--config",
            config,
            "--split",
            "test",
            "--checkpoint",
            str(checkpoint),
            "--limit-samples",
            str(test_samples),
            "--num-robots",
            str(num_robots),
            "--num-entities",
            str(num_entities),
            "--batch-size",
            "64",
            "--samples",
            "5",
            "--stride",
            "10",
            "--role-metrics",
            "--seed",
            str(model_seed),
            "--dataset-seed",
            str(dataset_seed),
            "--device",
            device,
            "--save-csv",
            str(sage_csv),
        ]
        _run(eval_command, root, scale_dir / "eval.log")
        sage_row = _read_single(sage_csv)
        sage_row["scale_method"] = "sage"
        rows.append(sage_row)
        for baseline in BASELINES:
            baseline_csv = scale_dir / f"{baseline}.csv"
            baseline_command = [
                sys.executable,
                "-u",
                "-m",
                "sage.experiments.eval_baseline_comparison",
                "--baseline",
                baseline,
                "--config",
                config,
                "--dataset",
                "synthetic",
                "--split",
                "test",
                "--limit-samples",
                str(test_samples),
                "--num-robots",
                str(num_robots),
                "--num-entities",
                str(num_entities),
                "--batch-size",
                "64",
                "--role-metrics",
                "--device",
                device,
                "--dataset-seed",
                str(dataset_seed),
                "--save-csv",
                str(baseline_csv),
            ]
            _run(baseline_command, root, scale_dir / f"{baseline}.log")
            baseline_row = _read_single(baseline_csv)
            baseline_row["scale_method"] = baseline
            rows.append(baseline_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with (output_dir / "scale_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "COMPLETED",
        "protocol": {
            "config": config,
            "robot_counts": list(robot_counts),
            "dataset_seed": dataset_seed,
            "model_seed": model_seed,
            "epochs": epochs,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "num_entities": num_entities,
            "evaluation_samples": 5,
            "interpretation": "single-/multi-robot scale stress; not a formal superiority gate",
        },
        "rows": rows,
    }
    (output_dir / "scale_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Synthetic Robot-Scale Stress",
        "",
        "Fixed-budget diagnostic across one, three, and six robot proxies. The experiment reports scaling and risk profiles; it is not a new confirmatory gate.",
        "",
        "| Robots | Method | Robot ADE | Robot FDE | Goal success | R-E collision | R-R collision | Runtime ms/scene |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('num_robots', '')} | {row['scale_method']} | {float(row['robot_ade']):.4f} | "
            f"{float(row['robot_fde']):.4f} | {float(row['goal_success_rate']):.4f} | "
            f"{float(row['robot_entity_collision_rate']):.4f} | {float(row['robot_robot_collision_rate']):.4f} | "
            f"{float(row.get('runtime_ms_per_scene', row.get('runtime_ms_per_window', 0.0)) or 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "The R=1 row is a single-robot degeneration check; R=3 is the primary system size; R=6 is a bounded stress point. Results should be interpreted together with the fixed sample/epoch budget.",
        ]
    )
    (output_dir / "scale_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_summary={output_dir / 'scale_summary.json'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="reports/synthetic_robot_scale_2026-08-21")
    parser.add_argument("--run-dir", default="experiments/pilot/synthetic_robot_scale_20260821")
    parser.add_argument("--robot-counts", nargs="+", type=int, default=list(ROBOT_COUNTS))
    parser.add_argument("--dataset-seed", type=int, default=6061)
    parser.add_argument("--model-seed", type=int, default=6061)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--num-entities", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(
        Path(args.root).resolve(),
        Path(args.output_dir),
        Path(args.run_dir),
        tuple(args.robot_counts),
        args.dataset_seed,
        args.model_seed,
        args.epochs,
        args.train_samples,
        args.test_samples,
        args.num_entities,
        args.device,
    )


if __name__ == "__main__":
    main()
