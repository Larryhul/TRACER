"""Summarize window-level B3 metrics from a completed fixed-pool run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean


def _mean(records: list[dict[str, object]], key: str) -> float:
    values = [
        float(record[key])
        for record in records
        if key in record and math.isfinite(float(record[key]))
    ]
    return fmean(values) if values else float("nan")


def _entropy_bin(entropy: float) -> str:
    if entropy < 0.5:
        return "low"
    if entropy < 1.0:
        return "mid"
    return "high"


def _flip_mean(records: list[dict[str, object]], key: str, fallback: str) -> float:
    values = []
    for record in records:
        if not float(record.get("belief_selection_flip", 0.0)):
            continue
        value = record.get(key, record.get(fallback))
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return fmean(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    runs = json.loads((run_dir / "runs.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    entropy_diagnostics: list[dict[str, object]] = []
    for seed, split_runs in sorted(runs.items(), key=lambda item: int(item[0])):
        for split, policy_runs in sorted(split_runs.items()):
            for policy, result in policy_runs.items():
                windows = [record for episode in result["window_records"] for record in episode]
                late_windows = [
                    record for record in windows if int(record["window"]) >= 4
                ]
                episode_records = result["episode_records"]
                row = {
                    "seed": int(seed),
                    "split": split,
                    "policy": policy,
                    "windows": len(windows),
                    "selected_next_window_response_cost": fmean(
                        float(record["selected_true_cost"]) for record in windows
                    ),
                    "oracle_candidate_cost": fmean(
                        float(record["oracle_true_cost"]) for record in windows
                    ),
                    "window_oracle_regret": fmean(
                        float(record["oracle_regret"]) for record in windows
                    ),
                    "late_window_response_cost": fmean(
                        float(record["selected_true_cost"]) for record in late_windows
                    ),
                    "late_window_oracle_regret": fmean(
                        float(record["oracle_regret"]) for record in late_windows
                    ),
                    "late_window_fallback": fmean(
                        float(record["safety_fallback"]) for record in late_windows
                    ),
                    "candidate_true_cost_range": fmean(
                        float(record["candidate_true_cost_range"])
                        for record in episode_records
                    ),
                    "candidate_response_spread": _mean(
                        windows, "candidate_response_spread"
                    ),
                    "belief_entropy": _mean(windows, "belief_entropy"),
                    "belief_max_probability": _mean(
                        windows, "belief_max_probability"
                    ),
                    "belief_selection_flip_rate": _mean(
                        windows, "belief_selection_flip"
                    ),
                    "belief_selection_predicted_cost_delta": _mean(
                        windows, "belief_selection_predicted_cost_delta"
                    ),
                    "belief_selection_true_cost_delta": _mean(
                        windows, "belief_selection_cost_delta"
                    ),
                    "belief_flip_windows": sum(
                        float(record.get("belief_selection_flip", 0.0))
                        for record in windows
                    ),
                    "belief_flip_predicted_cost_delta": _flip_mean(
                        windows,
                        "belief_flip_predicted_cost_delta",
                        "belief_selection_predicted_cost_delta",
                    ),
                    "belief_flip_true_cost_delta": _flip_mean(
                        windows,
                        "belief_flip_true_cost_delta",
                        "belief_selection_cost_delta",
                    ),
                    "belief_flip_oracle_regret_delta": _flip_mean(
                        windows,
                        "belief_flip_oracle_regret_delta",
                        "belief_selection_oracle_regret_delta",
                    ),
                    "selected_violation_probability": fmean(
                        float(record["selected_violation_probability"])
                        for record in windows
                    ),
                    "safety_fallback": fmean(
                        float(record["safety_fallback"]) for record in windows
                    ),
                    "task_progress": fmean(
                        float(record["task_progress"]) for record in episode_records
                    ),
                    "entity_disturbance": fmean(
                        float(record["human_cost"]) for record in episode_records
                    ),
                    "near_miss_distance": fmean(
                        float(record["min_distance"]) for record in episode_records
                    ),
                    "collision_rate": fmean(
                        float(record["collision_rate"]) for record in episode_records
                    ),
                    "realized_total_cost": fmean(
                        float(record["realized_total_cost"])
                        for record in episode_records
                    ),
                }
                rows.append(row)
                for entropy_bin in ("low", "mid", "high"):
                    bin_records = [
                        record
                        for record in windows
                        if "belief_entropy" in record
                        and _entropy_bin(float(record["belief_entropy"]))
                        == entropy_bin
                    ]
                    if not bin_records:
                        continue
                    entropy_diagnostics.append(
                        {
                            "seed": int(seed),
                            "split": split,
                            "policy": policy,
                            "entropy_bin": entropy_bin,
                            "windows": len(bin_records),
                            "belief_entropy": _mean(bin_records, "belief_entropy"),
                            "belief_selection_flip_rate": _mean(
                                bin_records, "belief_selection_flip"
                            ),
                            "belief_selection_predicted_cost_delta": _mean(
                                bin_records,
                                "belief_selection_predicted_cost_delta",
                            ),
                            "belief_selection_true_cost_delta": _mean(
                                bin_records, "belief_selection_cost_delta"
                            ),
                            "belief_selection_oracle_regret_delta": _mean(
                                bin_records,
                                "belief_selection_oracle_regret_delta",
                            ),
                            "belief_flip_windows": sum(
                                float(record.get("belief_selection_flip", 0.0))
                                for record in bin_records
                            ),
                            "belief_flip_true_cost_delta": _flip_mean(
                                bin_records,
                                "belief_flip_true_cost_delta",
                                "belief_selection_cost_delta",
                            ),
                            "belief_flip_oracle_regret_delta": _flip_mean(
                                bin_records,
                                "belief_flip_oracle_regret_delta",
                                "belief_selection_oracle_regret_delta",
                            ),
                        }
                    )
    fields = list(rows[0])
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output_json = Path(args.output_json)
    output_json.write_text(
        json.dumps(
            {
                "source": str(run_dir),
                "candidate_protocol": json.loads(
                    (run_dir / "candidate_protocol.json").read_text(encoding="utf-8")
                ),
                "rows": rows,
                "entropy_diagnostics": entropy_diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
