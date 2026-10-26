from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


CHECKPOINT_EPOCH = re.compile(r"epoch(?P<epoch>\d+)\.pt$")
REQUIRED_METRICS = (
    "robot_ade",
    "robot_fde",
    "goal_fde",
    "goal_progress",
    "normalized_goal_progress",
    "goal_success_rate",
    "robot_entity_collision_rate",
    "robot_robot_collision_rate",
    "mean_robot_speed",
    "max_robot_speed_observed",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _epoch(row: dict[str, str]) -> int:
    match = CHECKPOINT_EPOCH.search(row["checkpoint"])
    if match is None:
        raise ValueError(f"cannot parse checkpoint epoch: {row['checkpoint']}")
    return int(match.group("epoch"))


def _metric(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name} for {row['checkpoint']}")
    return value


def _setting(row: dict[str, str]) -> dict[str, object]:
    setting = {
        "checkpoint": row["checkpoint"],
        "epoch": _epoch(row),
        "guidance_scale": float(row["guidance_scale"]),
        "w_collision": float(row["w_collision"]),
        "w_robot_robot_collision": float(row["w_robot_robot_collision"]),
    }
    if "guidance_step_clip_norm" in row:
        setting["guidance_step_clip_norm"] = float(row["guidance_step_clip_norm"])
    return setting


def _same_setting(first: dict[str, str], second: dict[str, str]) -> bool:
    keys = ("guidance_scale", "w_collision", "w_robot_robot_collision")
    same = first["checkpoint"] == second["checkpoint"] and all(
        math.isclose(float(first[key]), float(second[key]), abs_tol=1e-12)
        for key in keys
    )
    if "guidance_step_clip_norm" in first or "guidance_step_clip_norm" in second:
        same = same and "guidance_step_clip_norm" in first and "guidance_step_clip_norm" in second
        if same:
            same = math.isclose(
                float(first["guidance_step_clip_norm"]),
                float(second["guidance_step_clip_norm"]),
                abs_tol=1e-12,
            )
    return same


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and gate the three-robot SAGE P0 generator."
    )
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--test-csv")
    parser.add_argument("--reference-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    validation_rows = _read_rows(Path(args.validation_csv))
    for row in validation_rows:
        for metric in REQUIRED_METRICS:
            _metric(row, metric)
    validation_rows.sort(key=_epoch)
    reference = None
    if args.reference_json:
        reference = json.loads(Path(args.reference_json).read_text(encoding="utf-8"))[
            "metrics"
        ]
        robot_entity_limit = float(reference["robot_entity_collision_rate"]) + 0.02
        robot_robot_limit = float(reference["robot_robot_collision_rate"])
    else:
        robot_entity_limit = 0.05
        robot_robot_limit = 0.05
    eligible = [
        row
        for row in validation_rows
        if _metric(row, "normalized_goal_progress") > 0.0
        and _metric(row, "robot_entity_collision_rate") <= robot_entity_limit
        and _metric(row, "robot_robot_collision_rate") <= robot_robot_limit
    ]
    validation_feasible = bool(eligible)
    if not eligible:
        eligible = validation_rows
    selected = min(
        eligible,
        key=lambda row: (
            _metric(row, "robot_ade"),
            _metric(row, "goal_fde"),
            _epoch(row),
        ),
    )
    report: dict[str, object] = {
        "status": "selection_only" if not args.test_csv else "evaluated",
        "decision_rule_scope": "post-pilot operational gate; not preregistered",
        "selection_split": "validation",
        "selected_setting": _setting(selected),
        "validation_feasible": validation_feasible,
        "validation_limits": {
            "robot_entity_collision_rate": robot_entity_limit,
            "robot_robot_collision_rate": robot_robot_limit,
        },
        "validation_rows": [
            {
                **_setting(row),
                **{metric: _metric(row, metric) for metric in REQUIRED_METRICS},
            }
            for row in validation_rows
        ],
    }

    if args.test_csv:
        test_rows = _read_rows(Path(args.test_csv))
        matching = [row for row in test_rows if _same_setting(row, selected)]
        if len(matching) != 1:
            raise ValueError(
                "test CSV must contain exactly one row for the selected checkpoint"
            )
        test = matching[0]
        finite = True
        try:
            for metric in REQUIRED_METRICS:
                _metric(test, metric)
        except (KeyError, ValueError):
            finite = False
        checks = {
            "all_required_metrics_finite": finite,
            "validation_setting_is_feasible": validation_feasible,
            "test_makes_positive_goal_progress": (
                _metric(test, "normalized_goal_progress") > 0.0
            ),
            "test_robot_entity_within_reference_tolerance": (
                _metric(test, "robot_entity_collision_rate") <= robot_entity_limit
            ),
            "test_robot_robot_no_worse_than_reference": (
                _metric(test, "robot_robot_collision_rate") <= robot_robot_limit
            ),
            "test_mean_robot_speed_positive": _metric(test, "mean_robot_speed") > 0.1,
            "test_max_robot_speed_within_configured_limit": (
                _metric(test, "max_robot_speed_observed") <= 3.05
            ),
        }
        report.update(
            {
                "checks": checks,
                "passed": all(checks.values()),
                "test_metrics": {
                    metric: _metric(test, metric) for metric in REQUIRED_METRICS
                },
                "reference_metrics": reference,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
