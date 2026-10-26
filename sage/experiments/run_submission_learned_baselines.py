"""Run matched multi-seed learned baselines for the submission comparison."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = (2027, 2028, 2029, 2030, 2031)
LEARNED_BASELINES = ("trajectron_plus_plus", "mid")
ALL_METHODS = (*LEARNED_BASELINES, "sage_medium")
METRICS = (
    "robot_ade",
    "robot_fde",
    "entity_ade",
    "goal_fde",
    "goal_success_rate",
    "robot_entity_collision_rate",
    "robot_robot_collision_rate",
)


def _run(command: list[str], root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
    elapsed = time.monotonic() - started
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    print(f"completed_in_seconds={elapsed:.1f} log={log_path}", flush=True)


def _read_single(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def _write_union_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        name = row.get("baseline", "") or row.get("model_variant", "")
        if name == "joint_heterogeneous_semantic":
            name = "sage_medium"
        grouped.setdefault(name, []).append(row)
    methods: dict[str, Any] = {}
    for method, method_rows in sorted(grouped.items()):
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in method_rows if row.get(metric, "") != ""]
            metrics[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        seed_values = {
            row.get("train_seed", "") or row.get("eval_seed", "") or row.get("dataset_seed", "")
            for row in method_rows
        }
        methods[method] = {
            "n_rows": len(method_rows),
            "seeds": sorted(value for value in seed_values if value != ""),
            "metrics": metrics,
        }
    return {
        "protocol": {
            "dataset": "controlled_synthetic",
            "config": "configs/sage_synthetic_medium_20260819.yaml",
            "dataset_seed": 2027,
            "model_seeds": list(DEFAULT_SEEDS),
            "epochs": 10,
            "train_samples": 2048,
            "test_samples": 1024,
            "evaluation_samples": 20,
            "evaluation_stride": 10,
            "num_robots": 3,
            "num_entities": 50,
            "interpretation": "supporting navigation-family robustness, separate from Formal-v1 gates",
        },
        "methods": methods,
    }


def _write_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Submission Learned-Baseline Multi-Seed Summary",
        "",
        "This is supporting navigation-family evidence under the controlled synthetic protocol; it is separate from the Formal-v1 H1/H2/H3 gates.",
        "All learned methods use the same fixed dataset seed (2027), five model seeds, ten training epochs, 2,048 training samples, 1,024 test samples, and 20-sample evaluation with stride 10.",
        "",
        "| Method | n | Seeds | Robot ADE | Robot FDE | Entity ADE | Goal success | R-E collision | R-R collision |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, record in payload["methods"].items():
        def cell(metric: str) -> str:
            value = record["metrics"][metric]
            return f"{value['mean']:.4f} +/- {value['std']:.4f}"

        lines.append(
            f"| {method} | {record['n_rows']} | {', '.join(record['seeds'])} | "
            f"{cell('robot_ade')} | {cell('robot_fde')} | {cell('entity_ade')} | "
            f"{cell('goal_success_rate')} | {cell('robot_entity_collision_rate')} | "
            f"{cell('robot_robot_collision_rate')} |"
        )
    lines.extend(
        [
            "",
            "The learned rows are repository implementations under a matched budget, not official checkpoint reproductions. They remain supporting evidence and do not establish universal superiority.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    root: Path,
    output_dir: Path,
    run_dir: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    dataset_seed: int = 2027,
    epochs: int = 10,
    train_samples: int = 2048,
    test_samples: int = 1024,
    device: str = "cuda",
    resume_existing: bool = False,
    methods: tuple[str, ...] = ALL_METHODS,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("at least one model seed is required")
    config = "configs/sage_synthetic_medium_20260819.yaml"
    rows: list[dict[str, str]] = []
    source_path = root / "experiments/baselines/synthetic_multiseed_20260820/results.csv"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows.extend(
            row
            for row in csv.DictReader(handle)
            if (row.get("baseline", "") or row.get("model_variant", ""))
            in {"constant_velocity", "constant_acceleration", "direct_goal", "social_force"}
        )

    requested = set(methods)
    for baseline in LEARNED_BASELINES:
        if baseline not in requested:
            continue
        for seed in seeds:
            seed_dir = run_dir / f"{baseline}_seed{seed}"
            checkpoint = seed_dir / f"synthetic_{baseline}_epoch{epochs}.pt"
            eval_csv = seed_dir / "evaluation.csv"
            if not (resume_existing and eval_csv.exists() and checkpoint.exists()):
                train_command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "sage.experiments.train_learned_baseline",
                    "--baseline",
                    baseline,
                    "--config",
                    config,
                    "--dataset",
                    "synthetic",
                    "--epochs",
                    str(epochs),
                    "--batch-size",
                    "64",
                    "--limit-samples",
                    str(train_samples),
                    "--save-every",
                    str(epochs),
                    "--seed",
                    str(seed),
                    "--dataset-seed",
                    str(dataset_seed),
                    "--device",
                    device,
                    "--save-dir",
                    str(seed_dir),
                ]
                _run(train_command, root, seed_dir / "train.log")
                eval_command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "sage.experiments.eval_learned_baseline",
                    "--baseline",
                    baseline,
                    "--config",
                    config,
                    "--dataset",
                    "synthetic",
                    "--split",
                    "test",
                    "--checkpoint",
                    str(checkpoint),
                    "--limit-samples",
                    str(test_samples),
                    "--batch-size",
                    "128",
                    "--samples",
                    "20",
                    "--stride",
                    "10",
                    "--role-metrics",
                    "--seed",
                    str(seed),
                    "--dataset-seed",
                    str(dataset_seed),
                    "--device",
                    device,
                    "--save-csv",
                    str(eval_csv),
                ]
                _run(eval_command, root, seed_dir / "eval.log")
            rows.append(_read_single(eval_csv))

    if "sage_medium" not in requested:
        combined_path = output_dir / "combined_results.csv"
        _write_union_csv(combined_path, rows)
        payload = _summary(rows)
        payload["sources"] = [str(combined_path)]
        _write_summary(output_dir, payload)
        print(f"saved_combined={combined_path}", flush=True)
        print(f"saved_summary={output_dir / 'summary.json'}", flush=True)
        return payload

    for seed in seeds:
        seed_dir = run_dir / f"sage_medium_seed{seed}"
        checkpoint = seed_dir / f"synthetic_sage_epoch{epochs}.pt"
        eval_csv = seed_dir / "evaluation.csv"
        if not (resume_existing and eval_csv.exists() and checkpoint.exists()):
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
                "--save-every",
                str(epochs),
                "--seed",
                str(seed),
                "--dataset-seed",
                str(dataset_seed),
                "--device",
                device,
                "--save-dir",
                str(seed_dir),
            ]
            _run(train_command, root, seed_dir / "train.log")
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
                "--batch-size",
                "128",
                "--samples",
                "20",
                "--stride",
                "10",
                "--role-metrics",
                "--seed",
                str(seed),
                "--dataset-seed",
                str(dataset_seed),
                "--device",
                device,
                "--save-csv",
                str(eval_csv),
            ]
            _run(eval_command, root, seed_dir / "eval.log")
        rows.append(_read_single(eval_csv))

    combined_path = output_dir / "combined_results.csv"
    _write_union_csv(combined_path, rows)
    payload = _summary(rows)
    payload["sources"] = [str(combined_path)]
    _write_summary(output_dir, payload)
    print(f"saved_combined={combined_path}", flush=True)
    print(f"saved_summary={output_dir / 'summary.json'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="reports/baseline_multiseed_2026-08-21")
    parser.add_argument("--run-dir", default="experiments/pilot/submission_learned_multiseed_20260821")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--dataset-seed", type=int, default=2027)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    args = parser.parse_args()
    run(
        Path(args.root).resolve(),
        Path(args.output_dir),
        Path(args.run_dir),
        seeds=tuple(args.seeds),
        dataset_seed=args.dataset_seed,
        epochs=args.epochs,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        device=args.device,
        resume_existing=args.resume_existing,
        methods=tuple(args.methods),
    )


if __name__ == "__main__":
    main()
