"""Run matched learned-baseline references on the external SDD adapter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


METHODS = ("trajectron_plus_plus", "mid")
DEFAULT_SEEDS = (2027, 2028, 2029, 2030, 2031)
METRICS = (
    "robot_ade",
    "robot_fde",
    "entity_ade",
    "goal_success_rate",
    "robot_entity_collision_rate",
    "robot_robot_collision_rate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _run(
    root: Path,
    method: str,
    seed: int,
    run_root: Path,
    epochs: int,
    train_windows: int,
    test_windows: int,
    device: str,
    samples: int,
    num_robot_proxies: int,
    resume_existing: bool,
) -> dict[str, str]:
    seed_dir = run_root / f"{method}_seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = seed_dir / f"sdd_{method}_epoch{epochs}.pt"
    evaluation = seed_dir / "evaluation.csv"
    if resume_existing and checkpoint.is_file() and evaluation.is_file():
        with evaluation.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(f"expected one evaluation row in {evaluation}, got {len(rows)}")
        row = rows[0]
        row["method"] = method
        row["model_seed"] = str(seed)
        return row
    common = [
        "--config",
        "configs/sage_sdd_multirobot.yaml",
        "--dataset",
        "sdd",
        "--data-dir",
        "raw_data/external/sdd_annotations_subset",
        "--source",
        "official",
        "--sdd-root",
        "raw_data/external/sdd_annotations_subset",
        "--num-robot-proxies",
        str(num_robot_proxies),
        "--device",
        device,
    ]
    train_command = [
        sys.executable,
        "-u",
        "-m",
        "tracer.experiments.train_learned_baseline",
        "--baseline",
        method,
        *common,
        "--epochs",
        str(epochs),
        "--batch-size",
        "32",
        "--limit-scenes",
        "4",
        "--limit-samples",
        str(train_windows),
        "--save-every",
        str(epochs),
        "--seed",
        str(seed),
        "--save-dir",
        str(seed_dir),
    ]
    with (seed_dir / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(train_command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    eval_command = [
        sys.executable,
        "-u",
        "-m",
        "tracer.experiments.eval_learned_baseline",
        "--baseline",
        method,
        *common,
        "--split",
        "test",
        "--checkpoint",
        str(checkpoint),
        "--batch-size",
        "64",
        "--limit-scenes",
        "2",
        "--num-robot-proxies",
        str(num_robot_proxies),
        "--limit-samples",
        str(test_windows),
        "--samples",
        str(samples),
        "--stride",
        "10",
        "--role-metrics",
        "--seed",
        str(seed),
        "--save-csv",
        str(evaluation),
    ]
    with (seed_dir / "eval.log").open("w", encoding="utf-8") as log:
        subprocess.run(eval_command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    with evaluation.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one evaluation row in {evaluation}, got {len(rows)}")
    row = rows[0]
    row["method"] = method
    row["model_seed"] = str(seed)
    return row


def _aggregate(rows: list[dict[str, str]]) -> dict[str, Any]:
    methods: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        methods.setdefault(row["method"], []).append(row)
    return {
        method: {
            "n": len(method_rows),
            "seeds": sorted(int(row["model_seed"]) for row in method_rows),
            "metrics": {
                metric: {
                    "mean": sum(float(row[metric]) for row in method_rows) / len(method_rows),
                    "std": (
                        statistics.stdev(float(row[metric]) for row in method_rows)
                        if len(method_rows) > 1
                        else 0.0
                    ),
                }
                for metric in METRICS
            },
        }
        for method, method_rows in sorted(methods.items())
    }


def run(
    root: Path,
    output_dir: Path,
    run_root: Path,
    methods: tuple[str, ...] = METHODS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    epochs: int = 15,
    train_windows: int = 2048,
    test_windows: int = 1024,
    samples: int = 20,
    device: str = "cuda",
    workers: int = 2,
    num_robot_proxies: int = 1,
    resume_existing: bool = False,
) -> dict[str, Any]:
    if num_robot_proxies < 1:
        raise ValueError("num_robot_proxies must be at least one")
    jobs = [(method, seed) for method in methods for seed in seeds]
    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = {
            executor.submit(
                _run,
                root,
                method,
                seed,
                run_root,
                epochs,
                train_windows,
                test_windows,
                device,
                samples,
                num_robot_proxies,
                resume_existing,
            ): (method, seed)
            for method, seed in jobs
        }
        for future in as_completed(futures):
            method, seed = futures[future]
            rows.append(future.result())
            print(f"sdd_learned_completed method={method} seed={seed}", flush=True)
    rows.sort(key=lambda row: (row["method"], int(row["model_seed"])))
    output_dir.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with (output_dir / "sdd_learned_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "SUPPLEMENTARY_EXTERNAL_SDD_LEARNED_REFERENCE",
        "git": _git_state(root),
        "protocol": {
            "dataset": "SDD",
            "train_scenes": 4,
            "test_scenes": 2,
            "train_windows": train_windows,
            "test_windows": test_windows,
            "epochs": epochs,
            "samples": samples,
            "methods": list(methods),
            "model_seeds": list(seeds),
            "num_robot_proxies": num_robot_proxies,
            "interpretation": "repository learned references under the external SDD adapter; not official checkpoint reproduction",
        },
        "methods": _aggregate(rows),
    }
    (output_dir / "sdd_learned_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# External SDD Learned References",
        "",
        "Matched repository MID-style and Trajectron++-style references trained and evaluated through the same SDD adapter as SAGE.",
        "",
        "| Method | n | Seeds | Robot ADE | Robot FDE | Entity ADE | Goal success | R-E collision | R-R collision |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, record in payload["methods"].items():
        def cell(metric: str) -> str:
            value = record["metrics"][metric]
            return f"{value['mean']:.4f} +/- {value['std']:.4f}"

        lines.append(
            f"| {method} | {record['n']} | {', '.join(map(str, record['seeds']))} | "
            f"{cell('robot_ade')} | {cell('robot_fde')} | {cell('entity_ade')} | "
            f"{cell('goal_success_rate')} | {cell('robot_entity_collision_rate')} | "
            f"{cell('robot_robot_collision_rate')} |"
        )
    lines.append("")
    lines.append("These are repository implementations and adapter references, not official external checkpoint reproductions or transfer-superiority claims.")
    (output_dir / "sdd_learned_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifact_names = (
        "sdd_learned_results.csv",
        "sdd_learned_summary.json",
        "sdd_learned_summary.md",
    )
    manifest = {
        "status": "COMPLETED",
        "protocol": payload["protocol"],
        "git": payload["git"],
        "artifacts": {
            name: _sha256(output_dir / name) for name in artifact_names
        },
        "run_entries": [
            {
                "method": row["method"],
                "model_seed": int(row["model_seed"]),
                "evaluation": str(
                    run_root / f"{row['method']}_seed{row['model_seed']}" / "evaluation.csv"
                ),
            }
            for row in rows
        ],
        "claim_boundary": "supplementary repository learned-reference results through the external SDD adapter; not TRACER closed-loop evidence, official checkpoint reproduction, or transfer-superiority evidence",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved_summary={output_dir / 'sdd_learned_summary.json'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="reports/external_validation_2026-08-21/sdd_strong_20260821")
    parser.add_argument("--run-root", default="experiments/pilot/external_sdd_learned_20260821")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train-windows", type=int, default=2048)
    parser.add_argument("--test-windows", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--num-robot-proxies", type=int, default=1)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    run(
        Path(args.root).resolve(),
        Path(args.output_dir),
        Path(args.run_root),
        tuple(args.methods),
        tuple(args.seeds),
        args.epochs,
        args.train_windows,
        args.test_windows,
        args.samples,
        args.device,
        args.workers,
        args.num_robot_proxies,
        args.resume_existing,
    )


if __name__ == "__main__":
    main()
