"""Merge parallel learned-baseline partials into one submission comparison CSV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


DETERMINISTIC = (
    "constant_velocity",
    "constant_acceleration",
    "direct_goal",
    "social_force",
)
LEARNED = ("trajectron_plus_plus", "mid", "sage_medium")
PARTIAL_DIRS = {
    "trajectron_plus_plus": "partial_trajectron",
    "mid": "partial_mid",
    "sage_medium": "partial_sage",
}
METRICS = (
    "robot_ade",
    "robot_fde",
    "entity_ade",
    "goal_fde",
    "goal_success_rate",
    "robot_entity_collision_rate",
    "robot_robot_collision_rate",
)


def _method(row: dict[str, str]) -> str:
    method = row.get("baseline", "") or row.get("model_variant", "")
    return "sage_medium" if method == "joint_heterogeneous_semantic" else method


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, str]], dataset_seed: int, seeds: tuple[int, ...]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_method(row), []).append(row)
    methods: dict[str, Any] = {}
    for method, method_rows in sorted(grouped.items()):
        methods[method] = {
            "n_rows": len(method_rows),
            "seeds": sorted(
                {
                    row.get("train_seed", "") or row.get("eval_seed", "") or row.get("dataset_seed", "")
                    for row in method_rows
                    if row.get("train_seed", "") or row.get("eval_seed", "") or row.get("dataset_seed", "")
                }
            ),
            "metrics": {
                metric: {
                    "mean": statistics.fmean(float(row[metric]) for row in method_rows),
                    "std": statistics.stdev(float(row[metric]) for row in method_rows)
                    if len(method_rows) > 1
                    else 0.0,
                    "n": len(method_rows),
                }
                for metric in METRICS
            },
        }
    return {
        "protocol": {
            "dataset": "controlled_synthetic",
            "dataset_seed": dataset_seed,
            "model_seeds": list(seeds),
            "learned_methods": list(LEARNED),
            "interpretation": "supporting navigation-family robustness, separate from Formal-v1 gates",
        },
        "methods": methods,
    }


def run(
    root: Path,
    partial_root: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
    dataset_seed: int,
) -> dict[str, Any]:
    partial_root = partial_root if partial_root.is_absolute() else root / partial_root
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    deterministic_path = root / "experiments/baselines/synthetic_multiseed_20260820/results.csv"
    rows = [
        row for row in _read(deterministic_path)
        if _method(row) in DETERMINISTIC
    ]
    for method in LEARNED:
        partial_path = partial_root / PARTIAL_DIRS[method] / "combined_results.csv"
        if not partial_path.exists():
            raise FileNotFoundError(partial_path)
        method_rows = [row for row in _read(partial_path) if _method(row) == method]
        if len(method_rows) != len(seeds):
            raise ValueError(f"{method} expected {len(seeds)} rows, found {len(method_rows)}")
        rows.extend(method_rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[_method(row)] = counts.get(_method(row), 0) + 1
    expected = {method: len(seeds) for method in (*DETERMINISTIC, *LEARNED)}
    if counts != expected:
        raise ValueError(f"unexpected method counts: {counts}; expected {expected}")
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "combined_results.csv"
    _write(combined_path, rows)
    payload = _summary(rows, dataset_seed, seeds)
    payload["sources"] = [
        str(deterministic_path.relative_to(root)),
        *[
            str(
                (
                    partial_root / PARTIAL_DIRS[method]
                    / "combined_results.csv"
                ).relative_to(root)
            )
            for method in LEARNED
        ],
    ]
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Submission Baseline Multi-Seed Summary",
        "",
        "Deterministic rows come from the frozen five-seed comparison; learned rows are merged from matched five-model-seed partial runs on the same fixed synthetic dataset seed.",
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
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_combined={combined_path}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--partial-root", default="reports/baseline_multiseed_2026-08-21")
    parser.add_argument("--output-dir", default="reports/baseline_multiseed_2026-08-21")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--dataset-seed", type=int, default=2027)
    args = parser.parse_args()
    run(
        Path(args.root).resolve(),
        Path(args.partial_root),
        Path(args.output_dir),
        tuple(args.seeds),
        args.dataset_seed,
    )


if __name__ == "__main__":
    main()
