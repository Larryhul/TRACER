"""Summarize supporting baseline CSVs without mixing seed blocks silently."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = (
    "robot_ade",
    "robot_fde",
    "entity_ade",
    "goal_fde",
    "goal_success_rate",
    "robot_entity_collision_rate",
    "robot_robot_collision_rate",
)


def _baseline_name(row: dict[str, str]) -> str:
    name = row.get("baseline", "") or row.get("model_variant", "")
    if name == "joint_heterogeneous_semantic":
        return "sage_medium"
    return name


def summarize(path: Path, additional_paths: list[Path] | None = None) -> dict[str, Any]:
    source_paths = [path, *(additional_paths or [])]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    primary_names: set[str] = set()
    for source_index, source_path in enumerate(source_paths):
        with source_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            name = _baseline_name(row)
            if source_index == 0:
                primary_names.add(name)
            elif name in primary_names and name in {
                "constant_velocity",
                "constant_acceleration",
                "direct_goal",
                "social_force",
            }:
                continue
            grouped[name].append(row)
    output = {}
    for baseline, baseline_rows in sorted(grouped.items()):
        metrics = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in baseline_rows if row.get(metric, "") != ""]
            metrics[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        output[baseline] = {
            "n_rows": len(baseline_rows),
            "dataset_seeds": sorted({row.get("dataset_seed", "") for row in baseline_rows}),
            "metrics": metrics,
        }
    return {
        "sources": [str(source_path) for source_path in source_paths],
        "baselines": output,
    }


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = ["baseline", "n_rows", "dataset_seeds"]
    for metric in METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_std", f"{metric}_n"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for baseline, record in payload["baselines"].items():
            row = {
                "baseline": baseline,
                "n_rows": record["n_rows"],
                "dataset_seeds": ",".join(record["dataset_seeds"]),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = record["metrics"][metric]["mean"]
                row[f"{metric}_std"] = record["metrics"][metric]["std"]
                row[f"{metric}_n"] = record["metrics"][metric]["n"]
            writer.writerow(row)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Supporting Baseline Seed Summary",
        "",
        "This table is exploratory supporting evidence and is separate from the Formal-v1 gates. Values are mean +/- SD over the rows available for each method; `n` is reported to expose unequal seed coverage.",
        "",
        "| Method | n rows | Dataset seeds | Robot ADE | Robot FDE | Entity ADE | Goal success | R-E collision | R-R collision |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for baseline, record in payload["baselines"].items():
        def cell(metric: str) -> str:
            value = record["metrics"][metric]
            return f"{value['mean']:.4f} +/- {value['std']:.4f} (n={value['n']})"

        lines.append(
            f"| {baseline} | {record['n_rows']} | {', '.join(record['dataset_seeds'])} | "
            f"{cell('robot_ade')} | {cell('robot_fde')} | {cell('entity_ade')} | "
            f"{cell('goal_success_rate')} | {cell('robot_entity_collision_rate')} | "
            f"{cell('robot_robot_collision_rate')} |"
        )
    lines.extend(
        [
            "",
            "The deterministic and social-force rows were rerun over five controlled seed blocks (dataset seeds 202027--202031). The learned Trajectron++-style, MID-style, and SAGE-medium rows come from the previously recorded single-seed supporting run and are therefore not multi-seed robustness claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize supporting baseline rows.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--additional-csv", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = summarize(Path(args.csv), [Path(path) for path in args.additional_csv])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "summary.csv", payload)
    _write_markdown(output_dir / "summary.md", payload)
    print(f"saved_summary={output_dir / 'summary.json'}")
    print(f"saved_report={output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
