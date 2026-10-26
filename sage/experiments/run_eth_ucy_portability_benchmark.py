"""Run a matched ETH/UCY portability benchmark for SAGE and learned references.

This track evaluates observational forecasting compatibility only. The target
pedestrian is represented as the repository's robot proxy and neighboring
pedestrians are entities; no joint intervention or executed-belief label is
available in ETH/UCY. The runner keeps the training budget and held-out cap
fixed across the SAGE, MID-style, and Trajectron++-style implementations.
"""

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


DATASETS = ("eth", "hotel", "univ", "zara1", "zara2")
METHODS = ("sage", "mid", "trajectron_plus_plus")
DEFAULT_SEEDS = (2027, 2028, 2029)
METRICS = (
    "robot_ade",
    "robot_fde",
    "entity_ade",
    "entity_fde",
    "goal_success_rate",
    "collision_rate",
    "robot_entity_collision_rate",
    "robot_robot_collision_rate",
    "min_distance",
)


def _run_command(command: list[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no evaluation row in {path}")
    return rows[-1]


def _run_one(
    root: Path,
    output_root: Path,
    method: str,
    dataset: str,
    seed: int,
    config: str,
    data_dir: str,
    device: str,
    epochs: int,
    batch_size: int,
    train_samples: int,
    test_samples: int,
    samples: int,
    stride: int,
) -> dict[str, Any]:
    run_dir = output_root / method / dataset / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / (
        f"{dataset}_sage_epoch{epochs}.pt"
        if method == "sage"
        else f"{dataset}_{method}_epoch{epochs}.pt"
    )
    evaluation = run_dir / "evaluation.csv"
    if not checkpoint.exists():
        if method == "sage":
            train_command = [
                sys.executable,
                "-u",
                "-m",
                "sage.experiments.train_real_sage",
                "--config",
                config,
                "--data-dir",
                data_dir,
                "--dataset",
                dataset,
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--limit-samples",
                str(train_samples),
                "--save-every",
                str(epochs),
                "--seed",
                str(seed),
                "--device",
                device,
                "--save-dir",
                str(run_dir),
            ]
        else:
            train_command = [
                sys.executable,
                "-u",
                "-m",
                "sage.experiments.train_learned_baseline",
                "--baseline",
                method,
                "--config",
                config,
                "--dataset",
                dataset,
                "--data-dir",
                data_dir,
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--limit-samples",
                str(train_samples),
                "--save-every",
                str(epochs),
                "--seed",
                str(seed),
                "--device",
                device,
                "--save-dir",
                str(run_dir),
            ]
        _run_command(train_command, run_dir / "train.log", root)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    if not evaluation.exists():
        if method == "sage":
            eval_command = [
                sys.executable,
                "-u",
                "-m",
                "sage.experiments.eval_real_sage",
                "--config",
                config,
                "--data-dir",
                data_dir,
                "--dataset",
                dataset,
                "--split",
                "test",
                "--checkpoint",
                str(checkpoint),
                "--limit-samples",
                str(test_samples),
                "--samples",
                str(samples),
                "--stride",
                str(stride),
                "--seed",
                str(seed),
                "--device",
                device,
                "--save-csv",
                str(evaluation),
            ]
        else:
            eval_command = [
                sys.executable,
                "-u",
                "-m",
                "sage.experiments.eval_learned_baseline",
                "--baseline",
                method,
                "--config",
                config,
                "--dataset",
                dataset,
                "--data-dir",
                data_dir,
                "--split",
                "test",
                "--checkpoint",
                str(checkpoint),
                "--limit-samples",
                str(test_samples),
                "--samples",
                str(samples),
                "--stride",
                str(stride),
                "--seed",
                str(seed),
                "--device",
                device,
                "--save-csv",
                str(evaluation),
            ]
        _run_command(eval_command, run_dir / "eval.log", root)

    raw = _last_csv_row(evaluation)
    row: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "model_seed": seed,
        "device": device,
        "epochs": epochs,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "samples": samples,
        "checkpoint": str(checkpoint),
    }
    for metric in METRICS:
        value = raw.get(metric, "")
        row[metric] = float(value) if value not in ("", "None", "nan") else None
    return row


def _aggregate(
    rows: list[dict[str, Any]],
    datasets: tuple[str, ...] = DATASETS,
    methods: tuple[str, ...] = METHODS,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["method"])), []).append(row)
    result: dict[str, Any] = {}
    for (dataset, method), values in sorted(grouped.items()):
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            finite = [float(row[metric]) for row in values if row[metric] is not None]
            metrics[metric] = {
                "mean": statistics.fmean(finite) if finite else None,
                "std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
            }
        result[f"{dataset}/{method}"] = {
            "dataset": dataset,
            "method": method,
            "n": len(values),
            "seeds": sorted(int(row["model_seed"]) for row in values),
            "metrics": metrics,
        }

    for method in methods:
        macro: dict[str, Any] = {}
        for metric in METRICS:
            values = [
                result[f"{dataset}/{method}"]["metrics"][metric]["mean"]
                for dataset in datasets
                if f"{dataset}/{method}" in result
                and result[f"{dataset}/{method}"]["metrics"][metric]["mean"] is not None
            ]
            macro[metric] = statistics.fmean(values) if values else None
        result[f"macro/{method}"] = {
            "dataset": "macro",
            "method": method,
            "n": len(datasets),
            "seeds": [],
            "metrics": {metric: {"mean": value, "std": 0.0} for metric, value in macro.items()},
        }
    return result


def _write_markdown(
    path: Path, aggregate: dict[str, Any], methods: tuple[str, ...] = METHODS
) -> None:
    lines = [
        "# ETH/UCY Portability Benchmark",
        "",
        "Observational forecasting compatibility only; no intervention or persistent-belief labels are present.",
        "",
        "| Dataset | Method | Seeds | Robot ADE | Robot FDE | Entity ADE | Goal success | Collision | R-E collision |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, record in aggregate.items():
        if record["dataset"] == "macro":
            continue
        def cell(metric: str) -> str:
            value = record["metrics"][metric]["mean"]
            return "-" if value is None else f"{value:.4f}"
        lines.append(
            f"| {record['dataset']} | {record['method']} | {','.join(map(str, record['seeds']))} | "
            f"{cell('robot_ade')} | {cell('robot_fde')} | {cell('entity_ade')} | "
            f"{cell('goal_success_rate')} | {cell('collision_rate')} | {cell('robot_entity_collision_rate')} |"
        )
    lines.extend(("", "## Macro mean", "", "| Method | Robot ADE | Robot FDE | Entity ADE | Goal success | Collision |", "|---|---:|---:|---:|---:|---:|"))
    for method in methods:
        record = aggregate[f"macro/{method}"]
        def macro_cell(metric: str) -> str:
            value = record["metrics"][metric]["mean"]
            return "-" if value is None else f"{value:.4f}"
        lines.append(
            f"| {method} | {macro_cell('robot_ade')} | {macro_cell('robot_fde')} | "
            f"{macro_cell('entity_ade')} | {macro_cell('goal_success_rate')} | {macro_cell('collision_rate')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    root: Path,
    output_dir: Path,
    run_root: Path,
    config: str = "configs/sage.yaml",
    data_dir: str = "models/processed_data_noise",
    datasets: tuple[str, ...] = DATASETS,
    methods: tuple[str, ...] = METHODS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    device: str = "cuda",
    epochs: int = 5,
    batch_size: int = 64,
    train_samples: int = 1024,
    test_samples: int = 512,
    samples: int = 10,
    stride: int = 10,
    source_revision: str = "",
    workers: int = 1,
) -> dict[str, Any]:
    jobs = [
        (dataset, method, seed)
        for dataset in datasets
        for method in methods
        for seed in seeds
    ]

    def run_job(job: tuple[str, str, int]) -> dict[str, Any]:
        dataset, method, seed = job
        print(
            f"eth_ucy_portability dataset={dataset} method={method} seed={seed} device={device}",
            flush=True,
        )
        return _run_one(
            root,
            run_root,
            method,
            dataset,
            seed,
            config,
            data_dir,
            device,
            epochs,
            batch_size,
            train_samples,
            test_samples,
            samples,
            stride,
        )

    rows: list[dict[str, Any]] = []
    if workers <= 1:
        rows = [run_job(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            futures = {executor.submit(run_job, job): job for job in jobs}
            for future in as_completed(futures):
                rows.append(future.result())
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (row["dataset"], row["method"], row["model_seed"]))
    fields = [
        "dataset",
        "method",
        "model_seed",
        "device",
        "epochs",
        "train_samples",
        "test_samples",
        "samples",
        "checkpoint",
        *METRICS,
    ]
    with (output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    aggregate = _aggregate(rows, datasets=datasets, methods=methods)
    payload = {
        "status": "completed",
        "protocol": {
            "dataset_family": "ETH/UCY",
            "source_repository": "https://github.com/gutianpei/MID",
            "source_revision": source_revision,
            "datasets": list(datasets),
            "methods": list(methods),
            "model_seeds": list(seeds),
            "device": device,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "samples": samples,
            "stride": stride,
            "history": 8,
            "horizon": 12,
            "independent_unit": "held-out test window; dataset-level macro means are descriptive",
            "claim_boundary": "observational portability and forecasting compatibility only",
        },
        "aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(output_dir / "summary.md", aggregate, methods=methods)
    print(f"saved_summary={output_dir / 'summary.json'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="reports/external_validation_20260824/eth_ucy_portability")
    parser.add_argument("--run-root", default="experiments/pilot/eth_ucy_portability_20260824")
    parser.add_argument("--config", default="configs/sage.yaml")
    parser.add_argument("--data-dir", default="models/processed_data_noise")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(
        Path(args.root).resolve(),
        Path(args.output_dir),
        Path(args.run_root),
        args.config,
        args.data_dir,
        tuple(args.datasets),
        tuple(args.methods),
        tuple(args.seeds),
        args.device,
        args.epochs,
        args.batch_size,
        args.train_samples,
        args.test_samples,
        args.samples,
        args.stride,
        args.source_revision,
        args.workers,
    )


if __name__ == "__main__":
    main()
