"""Select SAGE inference guidance from validation, then evaluate one test setting.

This audit is deliberately separate from the primary external benchmark.  It
compares guided and unguided inference on validation entity ADE, chooses one
setting per dataset using pooled model seeds, and evaluates that fixed setting
on the untouched test split.  The test split is never used for selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tracer.config import load_sage_config
from tracer.experiments.eval_real_sage import seed_everything
from tracer.experiments.run_external_sage_benchmark import evaluate


DATASETS = ("thor_magni", "uf_retail", "huron_sacson")
SEEDS = (2027, 2028, 2029)
SETTINGS = (True, False)


def _metrics(
    dataset: str,
    seed: int,
    guided: bool,
    split: str,
    data_dir: Path,
    checkpoint_root: Path,
    config_path: str,
    device: str,
    batch_size: int,
    workers: int,
    stride: int,
) -> dict[str, float]:
    cfg, _ = load_sage_config(config_path)
    train_path = data_dir / f"{dataset}_train.npz"
    split_path = data_dir / f"{dataset}_{split}.npz"
    with np.load(train_path, allow_pickle=False) as data:
        cfg.diffusion.horizon = int(data["future"].shape[2])
    checkpoint = checkpoint_root / f"seed{seed}" / dataset / f"sage_seed{seed}_epoch6.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    # Separate deterministic streams for each validation/test setting.
    seed_everything(seed + (100000 if not guided else 0) + (200000 if split == "test" else 0))
    return evaluate(
        "sage",
        checkpoint,
        split_path,
        cfg,
        device,
        batch_size,
        workers,
        stride,
        guided=guided,
    )


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    checkpoint_root = Path(args.checkpoint_root)
    results: dict[str, Any] = {}
    for dataset in DATASETS:
        results[dataset] = {"validation": {}, "selection": {}, "test": {}}
        for guided in SETTINGS:
            label = "guided" if guided else "unguided"
            val_rows = []
            for seed in SEEDS:
                val_rows.append(
                    _metrics(
                        dataset,
                        seed,
                        guided,
                        "val",
                        data_dir,
                        checkpoint_root,
                        args.config,
                        args.device,
                        args.batch_size,
                        args.workers,
                        args.stride,
                    )
                )
            results[dataset]["validation"][label] = {
                "seeds": list(SEEDS),
                "entity_ade_mean": _mean(val_rows, "entity_ade"),
                "robot_entity_collision_rate_mean": _mean(val_rows, "robot_entity_collision_rate"),
                "rows": val_rows,
            }
        selected = min(
            ("guided", "unguided"),
            key=lambda label: (
                results[dataset]["validation"][label]["entity_ade_mean"],
                results[dataset]["validation"][label]["robot_entity_collision_rate_mean"],
            ),
        )
        results[dataset]["selection"] = {
            "setting": selected,
            "criterion": "pooled validation entity ADE, then validation robot-entity collision rate",
        }
        guided = selected == "guided"
        test_rows = []
        for seed in SEEDS:
            test_rows.append(
                _metrics(
                    dataset,
                    seed,
                    guided,
                    "test",
                    data_dir,
                    checkpoint_root,
                    args.config,
                    args.device,
                    args.batch_size,
                    args.workers,
                    args.stride,
                )
            )
        results[dataset]["test"] = {
            "setting": selected,
            "seeds": list(SEEDS),
            "robot_ade_mean": _mean(test_rows, "robot_ade"),
            "entity_ade_mean": _mean(test_rows, "entity_ade"),
            "robot_entity_collision_rate_mean": _mean(test_rows, "robot_entity_collision_rate"),
            "goal_progress_mean": _mean(test_rows, "goal_progress"),
            "rows": test_rows,
        }
    return {
        "protocol": {
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "candidate_settings": ["guided", "unguided"],
            "selection_split": "validation",
            "evaluation_split": "test",
            "checkpoint_budget": 6,
            "claim_boundary": "external forecasting portability only; validation-selected inference setting, no causal or persistent-effect conclusion",
        },
        "results": results,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# External SAGE Guidance Selection",
        "",
        "Guided versus unguided SAGE inference was selected on the validation split",
        "using pooled entity ADE, with collision rate as a tie-break. The selected",
        "setting was then evaluated once on the untouched test split for all three",
        "model seeds. No test metric was used for selection.",
        "",
        "## Validation Selection",
        "",
        "| Dataset | Guided entity ADE | Unguided entity ADE | Selected |",
        "|---|---:|---:|---|",
    ]
    for dataset, item in payload["results"].items():
        val = item["validation"]
        lines.append(
            f"| {dataset} | {val['guided']['entity_ade_mean']:.6f} | "
            f"{val['unguided']['entity_ade_mean']:.6f} | {item['selection']['setting']} |"
        )
    lines += [
        "",
        "## Held-Out Test",
        "",
        "| Dataset | Setting | Robot ADE | Entity ADE | Robot-entity collision | Goal progress |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, item in payload["results"].items():
        test = item["test"]
        lines.append(
            f"| {dataset} | {test['setting']} | {test['robot_ade_mean']:.6f} | "
            f"{test['entity_ade_mean']:.6f} | {test['robot_entity_collision_rate_mean']:.6f} | "
            f"{test['goal_progress_mean']:.6f} |"
        )
    lines += [
        "",
        "This is a validation-selected inference audit, not a replacement for the",
        "primary external rows. The setting is retained together with all metrics",
        "because lower entity error and lower collision do not necessarily move",
        "together across datasets.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="experiments/external_sage_data")
    parser.add_argument("--checkpoint-root", default="experiments/external_sage_multiseed")
    parser.add_argument("--config", default="configs/tracer.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--output-json", default="reports/external_validation_20260904/external_sage_guidance_selection.json")
    parser.add_argument("--output-report", default="reports/external_validation_20260904/external_sage_guidance_selection.md")
    args = parser.parse_args()
    payload = run(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_markdown(payload), encoding="utf-8")
    print(f"json={output_json}")
    print(f"report={output_report}")
    for dataset, item in payload["results"].items():
        print(f"selected dataset={dataset} setting={item['selection']['setting']}")


if __name__ == "__main__":
    main()
