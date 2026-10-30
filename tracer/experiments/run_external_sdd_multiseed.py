"""Run and summarize a bounded multi-seed SDD adapter compatibility smoke."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    "robot_ade",
    "robot_fde",
    "entity_ade",
    "goal_fde",
    "goal_success_rate",
    "robot_entity_collision_rate",
    "robot_robot_collision_rate",
)
DEFAULT_SEEDS = (202061, 202062, 202063)


def _run_streaming(command: list[str], root: Path, log_path: Path) -> None:
    """Run a child experiment while preserving both live output and a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    elapsed = time.monotonic() - started
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    print(f"completed_in_seconds={elapsed:.1f}", flush=True)


def _read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one evaluation row in {path}, got {len(rows)}")
    return rows[0]


def aggregate(rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    """Return per-seed rows and mean/std summaries for the tracked metrics."""
    normalized = [dict(row) for row in rows]
    metrics: dict[str, dict[str, float | int]] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in normalized]
        metrics[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
    return {"rows": normalized, "metrics": metrics}


def _write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sdd_sage_multiseed.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = ["seed", *METRICS]
    with (output_dir / "sdd_sage_multiseed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for source_row in payload["rows"]:
            writer.writerow({field: source_row.get(field, "") for field in fields})

    lines = [
        "# SDD SAGE Multi-Seed Compatibility Smoke",
        "",
        "This is bounded external portability evidence, not a formal causal or benchmark-superiority result.",
        (
            "The protocol uses four SDD train scenes, two held-out test scenes, "
            f"{payload['protocol']['train_windows']} train/{payload['protocol']['test_windows']} test windows, "
            f"{payload['protocol']['epochs']} CUDA training epochs, and "
            f"{payload['protocol']['samples']} stochastic evaluation samples per seed."
        ),
        "",
        "| Metric | Mean | SD | Range | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, summary in payload["metrics"].items():
        lines.append(
            f"| {metric} | {summary['mean']:.6f} | {summary['std']:.6f} | "
            f"[{summary['min']:.6f}, {summary['max']:.6f}] | {summary['n']} |"
        )
    lines.extend(
        [
            "",
            "Per-seed evaluation rows are in `sdd_sage_multiseed.csv`; child logs and checkpoints remain under the ignored `experiments/pilot/external_sdd_multiseed_20260821/` directory.",
            "The result only establishes that the existing SDD adapter and SAGE training/evaluation path execute consistently on the public annotations subset; the protocol uses "
            f"{payload['protocol']['num_robot_proxies']} robot proxy slot(s).",
        ]
    )
    (output_dir / "sdd_sage_multiseed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    root: Path,
    output_dir: Path,
    run_dir: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    device: str = "cuda",
    epochs: int = 5,
    limit_train_samples: int = 256,
    limit_test_samples: int = 256,
    samples: int = 1,
    stride: int = 10,
    num_robot_proxies: int = 3,
) -> dict[str, Any]:
    """Train/evaluate one SAGE checkpoint per seed under a fixed SDD protocol."""
    if not seeds:
        raise ValueError("at least one seed is required")
    if num_robot_proxies < 1:
        raise ValueError("num_robot_proxies must be at least one")
    data_dir = "raw_data/external/sdd_annotations_subset"
    common = [
        "--config",
        "configs/sage_sdd_multirobot.yaml",
        "--data-dir",
        data_dir,
        "--source",
        "official",
        "--sdd-root",
        data_dir,
        "--device",
        device,
        "--batch-size",
        "32",
        "--num-workers",
        "0",
        "--num-robot-proxies",
        str(num_robot_proxies),
        "--role-metrics",
    ]
    rows: list[dict[str, str]] = []
    run_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        checkpoint = seed_dir / f"sdd_sage_epoch{epochs}.pt"
        eval_csv = seed_dir / "eval.csv"
        train_command = [
            sys.executable,
            "-m",
            "tracer.experiments.train_sdd_sage",
            *common[: common.index("--role-metrics")],
            "--epochs",
            str(epochs),
            "--save-every",
            str(epochs),
            "--limit-scenes",
            "4",
            "--limit-samples",
            str(limit_train_samples),
            "--save-dir",
            str(seed_dir),
            "--seed",
            str(seed),
        ]
        _run_streaming(train_command, root, seed_dir / "train.log")
        if not checkpoint.exists():
            raise FileNotFoundError(f"training did not produce {checkpoint}")
        eval_command = [
            sys.executable,
            "-m",
            "tracer.experiments.eval_sdd_sage",
            *common,
            "--split",
            "test",
            "--checkpoint",
            str(checkpoint),
            "--samples",
            str(samples),
            "--stride",
            str(stride),
            "--limit-scenes",
            "2",
            "--limit-samples",
            str(limit_test_samples),
            "--seed",
            str(seed),
            "--save-csv",
            str(eval_csv),
        ]
        _run_streaming(eval_command, root, seed_dir / "eval.log")
        row = _read_single_row(eval_csv)
        row["seed"] = str(seed)
        rows.append(row)
        payload = {
            "protocol": {
                "dataset": "SDD",
                "train_scenes": 4,
                "test_scenes": 2,
                "train_windows": limit_train_samples,
                "test_windows": limit_test_samples,
                "epochs": epochs,
                "device": device,
                "samples": samples,
                "stride": stride,
                "num_robot_proxies": num_robot_proxies,
            },
            "seeds": list(seeds),
            **aggregate(rows),
        }
        _write_outputs(output_dir, payload)
    payload = {
        "protocol": {
            "dataset": "SDD",
            "train_scenes": 4,
            "test_scenes": 2,
            "train_windows": limit_train_samples,
            "test_windows": limit_test_samples,
            "epochs": epochs,
            "device": device,
            "samples": samples,
            "stride": stride,
            "num_robot_proxies": num_robot_proxies,
        },
        "seeds": list(seeds),
        **aggregate(rows),
    }
    _write_outputs(output_dir, payload)
    print(f"saved_summary={output_dir / 'sdd_sage_multiseed.json'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="reports/external_validation_2026-08-21")
    parser.add_argument("--run-dir", default="experiments/pilot/external_sdd_multiseed_20260821")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit-train-samples", type=int, default=256)
    parser.add_argument("--limit-test-samples", type=int, default=256)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--num-robot-proxies", type=int, default=3)
    args = parser.parse_args()
    run(
        Path(args.root).resolve(),
        Path(args.output_dir),
        Path(args.run_dir),
        seeds=tuple(args.seeds),
        device=args.device,
        epochs=args.epochs,
        limit_train_samples=args.limit_train_samples,
        limit_test_samples=args.limit_test_samples,
        samples=args.samples,
        stride=args.stride,
        num_robot_proxies=args.num_robot_proxies,
    )


if __name__ == "__main__":
    main()
