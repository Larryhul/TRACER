"""Run SAGE across multiple synthetic dataset seeds and model seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


METRICS = (
    "robot_ade",
    "robot_fde",
    "entity_ade",
    "goal_success_rate",
    "robot_entity_collision_rate",
    "robot_robot_collision_rate",
)
DEFAULT_DATASET_SEEDS = (2027, 2028, 2029)
DEFAULT_MODEL_SEEDS = (2027, 2028, 2029)


def _run_job(
    root: Path,
    run_dir: Path,
    dataset_seed: int,
    model_seed: int,
    epochs: int,
    train_samples: int,
    test_samples: int,
    device: str,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / f"synthetic_sage_epoch{epochs}.pt"
    evaluation = run_dir / "evaluation.csv"
    train_command = [
        sys.executable,
        "-u",
        "-m",
        "sage.experiments.train_synthetic_sage",
        "--config",
        "configs/sage_synthetic_medium_20260819.yaml",
        "--epochs",
        str(epochs),
        "--batch-size",
        "64",
        "--limit-samples",
        str(train_samples),
        "--save-every",
        str(epochs),
        "--seed",
        str(model_seed),
        "--dataset-seed",
        str(dataset_seed),
        "--device",
        device,
        "--save-dir",
        str(run_dir),
    ]
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(train_command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    eval_command = [
        sys.executable,
        "-u",
        "-m",
        "sage.experiments.eval_synthetic_sage",
        "--config",
        "configs/sage_synthetic_medium_20260819.yaml",
        "--split",
        "test",
        "--checkpoint",
        str(checkpoint),
        "--limit-samples",
        str(test_samples),
        "--batch-size",
        "128",
        "--samples",
        "10",
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
        str(evaluation),
    ]
    with (run_dir / "eval.log").open("w", encoding="utf-8") as log:
        subprocess.run(eval_command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    with evaluation.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {evaluation}, got {len(rows)}")
    row = rows[0]
    row["dataset_seed_requested"] = str(dataset_seed)
    row["model_seed_requested"] = str(model_seed)
    return row


def _summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["dataset_seed_requested"], []).append(row)
    by_dataset_seed: dict[str, Any] = {}
    for dataset_seed, seed_rows in sorted(grouped.items()):
        by_dataset_seed[dataset_seed] = {
            "n_model_seeds": len(seed_rows),
            "metrics": {
                metric: {
                    "mean": statistics.fmean(float(row[metric]) for row in seed_rows),
                    "std": statistics.stdev(float(row[metric]) for row in seed_rows)
                    if len(seed_rows) > 1
                    else 0.0,
                }
                for metric in METRICS
            },
        }
    pooled = {
        metric: {
            "mean": statistics.fmean(float(row[metric]) for row in rows),
            "std": statistics.stdev(float(row[metric]) for row in rows)
            if len(rows) > 1
            else 0.0,
        }
        for metric in METRICS
    }
    return {"by_dataset_seed": by_dataset_seed, "pooled": pooled}


def run(
    root: Path,
    output_dir: Path,
    run_root: Path,
    dataset_seeds: tuple[int, ...] = DEFAULT_DATASET_SEEDS,
    model_seeds: tuple[int, ...] = DEFAULT_MODEL_SEEDS,
    epochs: int = 10,
    train_samples: int = 1024,
    test_samples: int = 1024,
    device: str = "cuda",
    workers: int = 3,
) -> dict[str, Any]:
    jobs = [
        (dataset_seed, model_seed)
        for dataset_seed in dataset_seeds
        for model_seed in model_seeds
    ]
    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = {
            executor.submit(
                _run_job,
                root,
                run_root / f"dataset{dataset_seed}_model{model_seed}",
                dataset_seed,
                model_seed,
                epochs,
                train_samples,
                test_samples,
                device,
            ): (dataset_seed, model_seed)
            for dataset_seed, model_seed in jobs
        }
        for future in as_completed(futures):
            dataset_seed, model_seed = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"dataset_sensitivity_completed dataset_seed={dataset_seed} model_seed={model_seed}",
                flush=True,
            )
    rows.sort(key=lambda row: (int(row["dataset_seed_requested"]), int(row["model_seed_requested"])))
    output_dir.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with (output_dir / "dataset_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows)
    payload = {
        "status": "SUPPLEMENTARY_DATASET_SEED_SENSITIVITY",
        "protocol": {
            "config": "configs/sage_synthetic_medium_20260819.yaml",
            "dataset_seeds": list(dataset_seeds),
            "model_seeds": list(model_seeds),
            "epochs": epochs,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "evaluation_samples": 10,
            "interpretation": "proposed SAGE robustness to synthetic dataset seed; separate from Formal-v1 gates",
        },
        **summary,
    }
    (output_dir / "dataset_sensitivity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# SAGE Dataset-Seed Sensitivity",
        "",
        "SAGE is retrained under three synthetic dataset seeds and three model seeds per dataset seed. This separates model-seed variation from dataset-seed variation; it is supplementary evidence, not a Formal-v1 gate.",
        "",
        "| Dataset seed | n model seeds | Robot ADE | Robot FDE | Entity ADE | Goal success | R-E collision | R-R collision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_seed, record in summary["by_dataset_seed"].items():
        def cell(metric: str) -> str:
            value = record["metrics"][metric]
            return f"{value['mean']:.4f} +/- {value['std']:.4f}"

        lines.append(
            f"| {dataset_seed} | {record['n_model_seeds']} | {cell('robot_ade')} | "
            f"{cell('robot_fde')} | {cell('entity_ade')} | {cell('goal_success_rate')} | "
            f"{cell('robot_entity_collision_rate')} | {cell('robot_robot_collision_rate')} |"
        )
    pooled = summary["pooled"]
    lines.append(
        f"| pooled | {len(rows)} | {pooled['robot_ade']['mean']:.4f} +/- {pooled['robot_ade']['std']:.4f} | "
        f"{pooled['robot_fde']['mean']:.4f} +/- {pooled['robot_fde']['std']:.4f} | "
        f"{pooled['entity_ade']['mean']:.4f} +/- {pooled['entity_ade']['std']:.4f} | "
        f"{pooled['goal_success_rate']['mean']:.4f} +/- {pooled['goal_success_rate']['std']:.4f} | "
        f"{pooled['robot_entity_collision_rate']['mean']:.4f} +/- {pooled['robot_entity_collision_rate']['std']:.4f} | "
        f"{pooled['robot_robot_collision_rate']['mean']:.4f} +/- {pooled['robot_robot_collision_rate']['std']:.4f} |"
    )
    (output_dir / "dataset_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_summary={output_dir / 'dataset_sensitivity.json'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="reports/sage_dataset_sensitivity_2026-08-21")
    parser.add_argument("--run-root", default="experiments/pilot/sage_dataset_sensitivity_20260821")
    parser.add_argument("--dataset-seeds", nargs="+", type=int, default=list(DEFAULT_DATASET_SEEDS))
    parser.add_argument("--model-seeds", nargs="+", type=int, default=list(DEFAULT_MODEL_SEEDS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(
        Path(args.root).resolve(),
        Path(args.output_dir),
        Path(args.run_root),
        tuple(args.dataset_seeds),
        tuple(args.model_seeds),
        args.epochs,
        args.train_samples,
        args.test_samples,
        args.device,
        args.workers,
    )


if __name__ == "__main__":
    main()
