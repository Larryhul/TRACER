from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_METRICS = [
    "goal_fde",
    "robot_ade",
    "robot_entity_collision_rate",
    "robot_robot_collision_rate",
    "robot_entity_min_distance",
    "robot_robot_min_distance",
    "pir",
    "svr",
    "energy",
    "runtime_ms_per_scene",
]
GROUP_FIELDS = [
    "dataset",
    "split",
    "config",
    "num_robot_proxies",
    "max_entities",
    "guided",
    "samples",
    "stride",
    "limit_samples",
    "model_variant",
    "generation_mode",
    "encoder_variant",
    "use_robot_robot_edges",
    "guidance_scale",
    "w_collision",
    "w_robot_robot_collision",
    "w_social",
    "w_kinematic",
    "w_goal",
]


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def fmt(value: float | None, digits: int) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_pm(mean: float | None, std: float | None, digits: int) -> str:
    if mean is None:
        return ""
    if std is None:
        std = 0.0
    return f"{mean:.{digits}f}+-{std:.{digits}f}"


def guided_sort_value(value: str) -> int:
    return 1 if str(value).lower() == "true" else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize synthetic scalability evaluation seeds as mean/std rows.")
    parser.add_argument("--input", required=True, help="Raw per-seed CSV produced by eval_synthetic_sage.py")
    parser.add_argument("--output", required=True, help="Output mean/std CSV")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS, help="Metric columns to summarize")
    parser.add_argument("--digits", type=int, default=4, help="Decimal places for *_pm_std display columns")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with input_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows found in {input_path}")

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        groups[key].append(row)

    output_rows: list[dict[str, object]] = []
    for key, group_rows in groups.items():
        out: dict[str, object] = {field: value for field, value in zip(GROUP_FIELDS, key)}
        seeds = sorted({row.get("eval_seed", "") for row in group_rows if row.get("eval_seed", "") != ""})
        out["num_seeds"] = len(seeds) if seeds else len(group_rows)
        out["eval_seeds"] = " ".join(seeds)
        out["num_rows"] = len(group_rows)

        for metric in args.metrics:
            values = [parse_float(row.get(metric)) for row in group_rows]
            values = [value for value in values if value is not None]
            if not values:
                continue
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_pm_std"] = fmt_pm(mean, std, args.digits)
        output_rows.append(out)

    def sort_key(row: dict[str, object]) -> tuple[int, int]:
        try:
            nr = int(float(str(row.get("num_robot_proxies", 0))))
        except ValueError:
            nr = 0
        return nr, guided_sort_value(str(row.get("guided", "")))

    output_rows.sort(key=sort_key)

    base_fields = GROUP_FIELDS + ["num_seeds", "eval_seeds", "num_rows"]
    metric_fields: list[str] = []
    for metric in args.metrics:
        metric_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_pm_std"])
    fieldnames = base_fields + [field for field in metric_fields if any(field in row for row in output_rows)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    print(f"groups={len(output_rows)} rows={len(rows)} output={output_path}")


if __name__ == "__main__":
    main()
