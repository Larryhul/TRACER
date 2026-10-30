"""Run and aggregate repeated external SAGE portability experiments.

The external datasets are observational, so this runner is a robustness
benchmark rather than a causal test.  Seeds are kept at the recording split
and training level; the aggregate reports seed mean/std and preserves every
per-seed JSON report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = ("uf_retail", "thor_magni", "huron_sacson")
LEARNED_METHODS = ("sage", "mid", "trajectron_plus_plus")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(dataset: str, report_paths: list[Path]) -> dict[str, Any]:
    reports = [_read(path) for path in report_paths]
    methods: dict[str, Any] = {}
    for method in LEARNED_METHODS:
        rows = [report["methods"][method]["metrics"] for report in reports if method in report.get("methods", {})]
        if not rows:
            continue
        metric_names = sorted(set().union(*(row.keys() for row in rows)))
        summary: dict[str, Any] = {"seeds": [int(report["seed"]) for report in reports]}
        for metric in metric_names:
            values = [float(row[metric]) for row in rows if metric in row and np.isfinite(float(row[metric]))]
            if not values:
                continue
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary[f"{metric}_min"] = float(np.min(values))
            summary[f"{metric}_max"] = float(np.max(values))
        methods[method] = summary
    return {
        "dataset": dataset,
        "n_seeds": len(reports),
        "seeds": [int(report["seed"]) for report in reports],
        "source_reports": [str(path) for path in report_paths],
        "methods": methods,
        "claim_boundary": "observational forecasting portability only; no real-human causal or persistent-effect conclusion",
    }


def _write_markdown(path: Path, aggregate: dict[str, Any]) -> None:
    lines = [
        f"# External SAGE portability: {aggregate['dataset']}",
        "",
        f"Repeated recording-level split and training-seed benchmark over {aggregate['n_seeds']} seeds: "
        f"{', '.join(str(seed) for seed in aggregate['seeds'])}.",
        "",
        "This is observational forecasting portability evidence only; it does not identify real-human causal or persistent effects.",
        "",
        "| Method | Robot ADE | Entity ADE | Robot-entity collision |",
        "|---|---:|---:|---:|",
    ]
    for method, metrics in aggregate["methods"].items():
        lines.append(
            f"| {method} | {metrics['robot_ade_mean']:.4f} +/- {metrics['robot_ade_std']:.4f} | "
            f"{metrics['entity_ade_mean']:.4f} +/- {metrics['entity_ade_std']:.4f} | "
            f"{metrics['robot_entity_collision_rate_mean']:.4f} +/- "
            f"{metrics['robot_entity_collision_rate_std']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated external SAGE/baseline portability experiments.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029])
    parser.add_argument("--data-dir", default="experiments/external_sage_data")
    parser.add_argument("--run-root", default="experiments/external_sage_multiseed")
    parser.add_argument("--report-root", default="reports/external_validation_20260825/external_sage_multiseed")
    parser.add_argument("--config", default="configs/tracer.yaml")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        paths: list[Path] = []
        for seed in args.seeds:
            seed_report_dir = report_root / f"seed{seed}"
            seed_report = seed_report_dir / f"{dataset}_sage_baseline.json"
            paths.append(seed_report)
            if args.resume_existing and seed_report.exists():
                print(f"skip dataset={dataset} seed={seed}", flush=True)
                continue
            command = [
                sys.executable,
                "-u",
                "-m",
                "tracer.experiments.run_external_sage_benchmark",
                "--dataset",
                dataset,
                "--data-dir",
                args.data_dir,
                "--output-dir",
                str(run_root / f"seed{seed}"),
                "--report-dir",
                str(seed_report_dir),
                "--config",
                args.config,
                "--methods",
                *LEARNED_METHODS,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--num-workers",
                str(args.num_workers),
                "--stride",
                str(args.stride),
                "--seed",
                str(seed),
            ]
            print(f"run dataset={dataset} seed={seed}", flush=True)
            subprocess.run(command, check=True)
        if not all(path.exists() for path in paths):
            missing = [str(path) for path in paths if not path.exists()]
            raise FileNotFoundError(f"missing per-seed reports: {missing}")
        aggregate = _aggregate(dataset, paths)
        output = report_root / f"{dataset}_multiseed.json"
        output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_markdown(report_root / f"{dataset}_multiseed.md", aggregate)
        print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
