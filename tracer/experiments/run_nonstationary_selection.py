"""Select non-stationary belief parameters on validation and freeze for test."""

from __future__ import annotations

import argparse
import csv
import json
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
    _jsonable,
    _run_policy,
    _sha256,
)


NONSTATIONARY = FaultScenario("nonstationary-hazard", nonstationary=True)


def _copy_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def _evaluate_variant(
    args: argparse.Namespace,
    policy: str,
    variant: str,
    data_seed: int,
    episodes: int,
    split: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    local_args = _copy_args(args, data_seed=data_seed, episodes=episodes)
    for seed in args.seeds:
        checkpoint = Path(args.checkpoint_dir) / f"seed{seed}" / "response.pt"
        model = _load_model(checkpoint, args.device)
        rows = _run_policy(model, policy, NONSTATIONARY, local_args, split, seed)
        for row in rows:
            row["variant"] = variant
            row["evaluation_data_seed"] = data_seed
            row["evaluation_role"] = "validation" if data_seed == args.validation_data_seed else "test"
        records.extend(rows)
    return records


def _selection_grid(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    families = {
        "hazard": [("tracer-hazard", "hazard_rate", value) for value in args.hazard_grid],
        "forgetting": [("tracer-forgetting", "forgetting_rate", value) for value in args.forgetting_grid],
        "change-point": [("tracer-change-point", "change_point_threshold", value) for value in args.change_point_grid],
    }
    selection: dict[str, Any] = {}
    for family, variants in families.items():
        candidates = []
        for policy, field, value in variants:
            tuned = _copy_args(args, **{field: value})
            variant = f"{family}:{field}={value:g}"
            rows = _evaluate_variant(
                tuned,
                policy,
                variant,
                args.validation_data_seed,
                args.validation_episodes,
                "id",
            )
            records.extend(rows)
            candidates.append({
                "variant": variant,
                "policy": policy,
                "parameter": field,
                "value": value,
                "validation_cost": float(np.mean([row["realized_total_cost"] for row in rows])),
                "validation_mode_accuracy": float(np.mean([row["final_mode_accuracy"] for row in rows])),
            })
            print(f"nonstationary_validation variant={variant} cost={candidates[-1]['validation_cost']:.6f}", flush=True)
        selection[family] = min(candidates, key=lambda item: item["validation_cost"])
        selection[family]["selection_rule"] = "minimum validation realized_total_cost"
    return records, selection


def _test_selected(
    args: argparse.Namespace, selection: dict[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    fixed = [
        ("tracer-persistent", "persistent", {}),
        ("tracer-reset", "reset", {}),
    ]
    for family, selected in selection.items():
        fixed.append((
            selected["policy"],
            f"selected-{family}:{selected['parameter']}={selected['value']:g}",
            {selected["parameter"]: selected["value"]},
        ))
    for policy, variant, updates in fixed:
        tuned = _copy_args(args, **updates)
        rows = _evaluate_variant(
            tuned,
            policy,
            variant,
            args.test_data_seed,
            args.test_episodes,
            "ood",
        )
        records.extend(rows)
        print(f"nonstationary_test variant={variant} episodes={len(rows)}", flush=True)
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    git = _git_state()
    if git["dirty"]:
        raise RuntimeError("non-stationary selection requires a clean Git worktree")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation_records, selection = _selection_grid(args)
    test_records = _test_selected(args, selection)
    all_records = validation_records + test_records
    with (output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    summary: dict[str, Any] = {"test": {}, "paired_vs_persistent": {}}
    test_variants = sorted({row["variant"] for row in test_records})
    for index, variant in enumerate(test_variants):
        rows = [row for row in test_records if row["variant"] == variant]
        summary["test"][variant] = {
            metric: _ci(
                [float(row[metric]) for row in rows],
                args.bootstrap_seed + index * 10 + metric_index,
            )
            for metric_index, metric in enumerate((
                "realized_total_cost", "task_progress", "collision_rate",
                "oracle_regret", "final_mode_accuracy", "final_belief_entropy",
                "belief_selection_flip", "change_point_detected",
            ))
        }
    persistent = {
        (row["seed"], row["episode"]): row
        for row in test_records if row["variant"] == "persistent"
    }
    for index, variant in enumerate(test_variants):
        if variant == "persistent":
            continue
        compared = {
            (row["seed"], row["episode"]): row
            for row in test_records if row["variant"] == variant
        }
        deltas = [
            float(compared[key]["realized_total_cost"] - persistent[key]["realized_total_cost"])
            for key in persistent.keys() & compared.keys()
        ]
        summary["paired_vs_persistent"][f"{variant}-persistent"] = _ci(
            deltas, args.bootstrap_seed + 100 + index
        )
    metadata = {
        "protocol": "tracer-nonstationary-validation-selection-v1",
        "git": git,
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "validation_data_seed": args.validation_data_seed,
        "test_data_seed": args.test_data_seed,
        "data_seeds_disjoint": args.validation_data_seed != args.test_data_seed,
        "model_seeds": list(args.seeds),
        "validation_episodes_per_seed_variant": args.validation_episodes,
        "test_episodes_per_seed_variant": args.test_episodes,
        "selection": selection,
        "parameters": vars(args),
        "runtime_seconds": time.perf_counter() - start,
        "claim_boundary": "validation-selected nonstationary robustness in a resettable independent simulator; no real-human persistence claim",
    }
    payload = {"metadata": _jsonable(metadata), "summary": _jsonable(summary)}
    (output / "selection.json").write_text(json.dumps(_jsonable(selection), indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# TRACER Non-Stationary Belief Selection", "",
        f"Validation data seed `{args.validation_data_seed}` and test data seed `{args.test_data_seed}` are disjoint.",
        f"Five model seeds are nuisance blocks; validation/test episodes per seed are `{args.validation_episodes}/{args.test_episodes}`.",
        "", "## Frozen Selection", "",
    ]
    for family, selected in selection.items():
        lines.append(
            f"- `{family}` selected `{selected['parameter']}={selected['value']:g}` at validation cost `{selected['validation_cost']:.4f}`."
        )
    lines += ["", "## Independent Test", ""]
    for variant in test_variants:
        result = summary["test"][variant]["realized_total_cost"]
        lines.append(
            f"- `{variant}` cost `{result['mean']:.4f}` CI `[{result['ci95_low']:.4f}, {result['ci95_high']:.4f}]`."
        )
    lines += ["", "## Paired Test Contrasts", ""]
    for name, result in summary["paired_vs_persistent"].items():
        lines.append(
            f"- `{name}`: `{result['mean']:.4f}` CI `[{result['ci95_low']:.4f}, {result['ci95_high']:.4f}]`."
        )
    lines += ["", "Negative policy-minus-Persistent cost favors the selected policy. Test data were not used for parameter selection.", ""]
    report = output / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "status": "COMPLETED",
        "protocol": metadata["protocol"],
        "git": git,
        "selection_role": "validation-only",
        "test_role": "independent held-out data seed",
        "artifacts": {
            name: _sha256(output / name)
            for name in ("episode_metrics.csv", "selection.json", "summary.json", "report.md")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/tracer_nonstationary_selection_20260902")
    parser.add_argument("--checkpoint-dir", default="experiments/independent_multientity_suite_20260902_final/checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2031, 2039, 2053, 2063])
    parser.add_argument("--validation-data-seed", type=int, default=728731)
    parser.add_argument("--test-data-seed", type=int, default=828731)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=16)
    parser.add_argument("--hazard-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.35])
    parser.add_argument("--forgetting-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.35])
    parser.add_argument("--change-point-grid", nargs="+", type=float, default=[1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--hazard-rate", type=float, default=0.2)
    parser.add_argument("--forgetting-rate", type=float, default=0.2)
    parser.add_argument("--change-point-threshold", type=float, default=3.0)
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
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument("--hard-safety-margin", type=float, default=0.08)
    parser.add_argument("--risk-alpha", type=float, default=0.1)
    parser.add_argument("--risk-samples", type=int, default=32)
    parser.add_argument("--bootstrap-seed", type=int, default=828731)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
