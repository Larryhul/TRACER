"""Summarize matched SDD baseline and SAGE compatibility smoke rows."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    metrics: dict[str, dict[str, float | int]] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        metrics[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return {"n": len(rows), "metrics": metrics}


def summarize(
    baseline_paths: list[Path],
    sage_path: Path,
    *,
    test_windows: int = 256,
    sage_train_windows: int = 256,
    sage_epochs: int = 5,
    sage_seeds: int = 3,
    sage_samples: int = 20,
    learned_paths: list[Path] | None = None,
) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    for path in baseline_paths:
        rows = _read_rows(path)
        baseline_names = {row.get("baseline", "") for row in rows}
        baseline_names.discard("")
        if len(baseline_names) != 1:
            raise ValueError(f"expected one baseline in {path}, got {sorted(baseline_names)}")
        methods[next(iter(baseline_names))] = _summarize(rows)
    for path in learned_paths or []:
        rows = _read_rows(path)
        method_names = {row.get("method", "") for row in rows}
        method_names.discard("")
        if not method_names:
            raise ValueError(f"no learned method names found in {path}")
        for method_name in sorted(method_names):
            method_rows = [row for row in rows if row.get("method") == method_name]
            methods[method_name] = _summarize(method_rows)
    methods["sage"] = _summarize(_read_rows(sage_path))
    return {
        "protocol": {
            "dataset": "SDD",
            "test_scenes": 2,
            "test_windows": test_windows,
            "sage_train_scenes": 4,
            "sage_train_windows": sage_train_windows,
            "sage_epochs": sage_epochs,
            "sage_seeds": sage_seeds,
            "sage_samples": sage_samples,
            "interpretation": "matched external compatibility/risk profile only",
        },
        "methods": methods,
        "sources": [str(path) for path in [*baseline_paths, sage_path, *(learned_paths or [])]],
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sdd_external_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = ["method", "n", *[f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")]]
    with (output_dir / "sdd_external_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method, record in payload["methods"].items():
            row: dict[str, Any] = {"method": method, "n": record["n"]}
            for metric in METRICS:
                row[f"{metric}_mean"] = record["metrics"][metric]["mean"]
                row[f"{metric}_std"] = record["metrics"][metric]["std"]
            writer.writerow(row)

    lines = [
        "# Matched SDD External Compatibility Comparison",
        "",
        "This is a bounded external compatibility and risk-profile comparison. It is not evidence for the paper's joint-intervention or persistent-belief causal claim, and it is not an external transfer-superiority benchmark.",
        "",
        "| Method | n | Robot ADE | Robot FDE | Entity ADE | Goal success | R-E collision | R-R collision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, record in payload["methods"].items():
        def cell(metric: str) -> str:
            value = record["metrics"][metric]
            return f"{value['mean']:.4f} +/- {value['std']:.4f}"

        lines.append(
            f"| {method} | {record['n']} | {cell('robot_ade')} | {cell('robot_fde')} | "
            f"{cell('entity_ade')} | {cell('goal_success_rate')} | "
            f"{cell('robot_entity_collision_rate')} | {cell('robot_robot_collision_rate')} |"
        )
    lines.extend(
        [
            "",
            (
                "The SAGE row uses "
                f"{payload['protocol']['sage_seeds']} seeds, "
                f"{payload['protocol']['sage_epochs']} epochs, and the configured stochastic "
                "evaluation sample count; deterministic and learned reference rows use the "
                "same held-out windows. The learned rows are repository implementations, "
                "not official MID or Trajectron++ checkpoint reproductions. Lower ADE/FDE "
                "and lower collision rates are preferable, while goal success is higher-is-better."
            ),
            "The external numbers are reported to expose compatibility and failure modes. They do not override the formal synthetic claim boundary.",
        ]
    )
    (output_dir / "sdd_external_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", nargs="+", required=True)
    parser.add_argument("--sage-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-windows", type=int, default=256)
    parser.add_argument("--sage-train-windows", type=int, default=256)
    parser.add_argument("--sage-epochs", type=int, default=5)
    parser.add_argument("--sage-seeds", type=int, default=3)
    parser.add_argument("--sage-samples", type=int, default=20)
    parser.add_argument("--learned-csv", nargs="*", default=[])
    args = parser.parse_args()
    payload = summarize(
        [Path(path) for path in args.baseline_csv],
        Path(args.sage_csv),
        test_windows=args.test_windows,
        sage_train_windows=args.sage_train_windows,
        sage_epochs=args.sage_epochs,
        sage_seeds=args.sage_seeds,
        sage_samples=args.sage_samples,
        learned_paths=[Path(path) for path in args.learned_csv],
    )
    write_outputs(Path(args.output_dir), payload)
    print(f"saved_summary={Path(args.output_dir) / 'sdd_external_comparison.json'}")


if __name__ == "__main__":
    main()
