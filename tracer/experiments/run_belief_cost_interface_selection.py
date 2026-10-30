"""Select TRACER's belief-to-cost weight on validation-only data.

The selection parameter changes only the planner's predicted human-response
cost.  Realized evaluation cost remains fixed at the submission objective, so
the validation sweep cannot change the metric definition.  Test data use a
fresh simulator seed and are never used to choose the weight.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tracer.experiments.eval_persistent_closed_loop import _load_model
from tracer.experiments.run_tracer_gap_closure import (
    FaultScenario,
    _ci,
    _git_state,
    _run_policy,
    _sha256,
)


VALIDATION_SCENARIOS = (
    FaultScenario("clean"),
    FaultScenario("nonstationary-hazard", nonstationary=True),
    FaultScenario("combined-realism", partial_observation=True, missed_detection_rate=0.25, id_switch=True, communication_delay=2, asynchronous_execution=True, nonstationary=True, static_obstacles=True),
    FaultScenario("hard-safety-obstacles", static_obstacles=True),
)


def _copy_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def _evaluate(
    args: argparse.Namespace,
    policy: str,
    weight: float,
    data_seed: int,
    episodes: int,
    split: str,
    role: str,
) -> list[dict[str, Any]]:
    local_args = _copy_args(
        args,
        belief_cost_weight=weight,
        data_seed=data_seed,
        episodes=episodes,
    )
    records: list[dict[str, Any]] = []
    for seed in args.seeds:
        model = _load_model(
            Path(args.checkpoint_dir) / f"seed{seed}" / "response.pt",
            args.device,
        )
        for scenario in VALIDATION_SCENARIOS:
            rows = _run_policy(model, policy, scenario, local_args, split, seed)
            for row in rows:
                row.update({
                    "interface_weight": float(weight),
                    "evaluation_role": role,
                    "evaluation_data_seed": int(data_seed),
                })
            records.extend(rows)
            print(
                f"belief_cost_{role} weight={weight:g} scenario={scenario.name} rows={len(rows)}",
                flush=True,
            )
    return records


def _contrast(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    left_map = {(row["seed"], row["scenario"], row["episode"]): row for row in left}
    right_map = {(row["seed"], row["scenario"], row["episode"]): row for row in right}
    values = [
        float(right_map[key]["realized_total_cost"] - left_map[key]["realized_total_cost"])
        for key in left_map.keys() & right_map.keys()
    ]
    return _ci(values, seed)


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path.cwd().resolve()
    git = _git_state()
    if git["dirty"]:
        raise RuntimeError("belief-cost interface selection requires a clean Git worktree")
    validation_records: list[dict[str, Any]] = []
    candidates = []
    for weight in args.weight_grid:
        rows = _evaluate(
            args, "tracer-persistent", weight, args.validation_data_seed,
            args.validation_episodes, "id", "validation",
        )
        validation_records.extend(rows)
        candidates.append({
            "weight": float(weight),
            "validation_cost": float(np.mean([row["realized_total_cost"] for row in rows])),
            "validation_collision": float(np.mean([row["collision_rate"] for row in rows])),
            "validation_rows": len(rows),
        })
    selected = min(candidates, key=lambda item: item["validation_cost"])
    selected_weight = float(selected["weight"])
    test_records = []
    test_records.extend(_evaluate(
        args, "tracer-persistent", 0.6, args.test_data_seed,
        args.test_episodes, "ood", "test-default-persistent",
    ))
    test_records.extend(_evaluate(
        args, "tracer-persistent", selected_weight, args.test_data_seed,
        args.test_episodes, "ood", "test-selected-persistent",
    ))
    test_records.extend(_evaluate(
        args, "tracer-reset", 0.6, args.test_data_seed,
        args.test_episodes, "ood", "test-reset",
    ))
    all_records = validation_records + test_records
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    test_variants = sorted({row["evaluation_role"] for row in test_records})
    summary: dict[str, Any] = {"validation_candidates": candidates, "selected": selected, "test": {}, "paired_contrasts": {}}
    for index, variant in enumerate(test_variants):
        rows = [row for row in test_records if row["evaluation_role"] == variant]
        summary["test"][variant] = {
            metric: _ci(
                [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))],
                args.bootstrap_seed + index * 17 + metric_index,
            )
            for metric_index, metric in enumerate(("realized_total_cost", "task_progress", "collision_rate", "hard_safety_fallback", "hard_safety_violation_without_fallback"))
        }
    default_rows = [row for row in test_records if row["evaluation_role"] == "test-default-persistent"]
    selected_rows = [row for row in test_records if row["evaluation_role"] == "test-selected-persistent"]
    reset_rows = [row for row in test_records if row["evaluation_role"] == "test-reset"]
    summary["paired_contrasts"]["selected-persistent_minus_default-persistent"] = _contrast(default_rows, selected_rows, args.bootstrap_seed + 100)
    summary["paired_contrasts"]["reset_minus_selected-persistent"] = _contrast(selected_rows, reset_rows, args.bootstrap_seed + 101)
    metadata = {
        "protocol": "tracer-belief-to-cost-validation-selection-v1",
        "claim_boundary": "validation-only selection of a planner response-cost weight; held-out test is an interface stability diagnostic, not a superiority claim",
        "git": git,
        "device": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "validation_data_seed": args.validation_data_seed,
        "test_data_seed": args.test_data_seed,
        "data_seeds_disjoint": args.validation_data_seed != args.test_data_seed,
        "parameters": vars(args),
        "runtime_seconds": time.perf_counter() - start,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "selection.json").write_text(json.dumps(selected, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# TRACER belief-to-cost interface selection", "",
        f"Validation seed `{args.validation_data_seed}` and test seed `{args.test_data_seed}` are disjoint; selected weight `{selected_weight:g}` by minimum validation realized cost.",
        "", "## Validation grid", "",
    ]
    lines.extend(f"- weight `{item['weight']:g}`: validation cost `{item['validation_cost']:.5f}`, collision `{item['validation_collision']:.4f}`." for item in candidates)
    lines.extend(["", "## Held-out test", ""])
    for variant in test_variants:
        item = summary["test"][variant]["realized_total_cost"]
        lines.append(f"- `{variant}`: cost `{item['mean']:.5f}` CI `[{item['ci95_low']:.5f}, {item['ci95_high']:.5f}]`.")
    lines.extend(["", "## Paired contrasts", ""])
    for name, item in summary["paired_contrasts"].items():
        lines.append(f"- `{name}`: `{item['mean']:.5f}` CI `[{item['ci95_low']:.5f}, {item['ci95_high']:.5f}]`.")
    lines.append("\nThe test set is used only after the weight is frozen; all contrasts remain diagnostic and do not establish universal superiority.\n")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "status": "COMPLETED",
        "protocol": metadata["protocol"],
        "git": git,
        "selection_role": "validation-only",
        "test_role": "independent held-out data seed",
        "artifacts": {name: _sha256(output / name) for name in ("episode_metrics.csv", "metadata.json", "selection.json", "summary.json", "report.md")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"metadata": metadata, "summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/tracer_belief_cost_selection_20260903")
    parser.add_argument("--checkpoint-dir", default="experiments/independent_multientity_suite_20260902_final/checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2031, 2039, 2053, 2063])
    parser.add_argument("--validation-data-seed", type=int, default=979731)
    parser.add_argument("--test-data-seed", type=int, default=989731)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=16)
    parser.add_argument("--weight-grid", nargs="+", type=float, default=[0.0, 0.15, 0.3, 0.6, 1.0, 1.5])
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--num-entities", type=int, default=6)
    parser.add_argument("--active-entities", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--switch-window", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.3)
    parser.add_argument("--response-noise", type=float, default=0.08)
    parser.add_argument("--entity-max-speed", type=float, default=2.2)
    parser.add_argument("--scene-family", default="crossing")
    parser.add_argument("--likelihood-scale", type=float, default=0.05)
    parser.add_argument("--hazard-rate", type=float, default=0.18)
    parser.add_argument("--forgetting-rate", type=float, default=0.18)
    parser.add_argument("--change-point-threshold", type=float, default=5.0)
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument("--hard-safety-margin", type=float, default=0.08)
    parser.add_argument("--risk-alpha", type=float, default=0.1)
    parser.add_argument("--risk-samples", type=int, default=32)
    parser.add_argument("--mpogames-temperature", type=float, default=0.15)
    parser.add_argument("--bootstrap-seed", type=int, default=999731)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
