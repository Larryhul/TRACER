"""Evaluate frozen H2 checkpoints across executed-prefix lengths."""

from __future__ import annotations

import argparse
import csv
import copy
import json
import statistics
from pathlib import Path
from typing import Any

import torch
import yaml

from tracer.experiments.eval_persistent_trajectory_checkpoints import _load_checkpoint_models
from tracer.experiments.train_persistent_trajectory_response import _evaluate, _make_dataset, _set_seed


DEFAULT_PREFIXES = (1, 3, 6)
DEFAULT_SEEDS = (2027, 2028, 2029, 2030, 2031)
METHOD_PAIRS = (
    ("reset", "persistent", "reset_minus_persistent_nll"),
    ("shuffled-id", "persistent", "shuffled_id_minus_persistent_nll"),
    ("recurrent", "persistent", "recurrent_minus_persistent_nll"),
)


def _paired_mean(rows: list[dict[str, Any]], first: str, second: str) -> tuple[float, int]:
    grouped: dict[tuple[int, int, int], dict[str, float]] = {}
    for row in rows:
        if int(row.get("main_evaluation", 0)) != 1:
            continue
        key = (int(row["episode_id"]), int(row["entity_id"]), int(row["window"]))
        grouped.setdefault(key, {})[str(row["method"])] = float(row["predictive_trajectory_nll"])
    values = [
        methods[first] - methods[second]
        for methods in grouped.values()
        if first in methods and second in methods
    ]
    if not values:
        raise ValueError(f"no paired rows for {first} vs {second}")
    return statistics.fmean(values), len(values)


def run(
    root: Path,
    config_path: Path,
    checkpoint_root: Path,
    output_dir: Path,
    prefixes: tuple[int, ...] = DEFAULT_PREFIXES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    device: str = "cuda",
    batch_size: int = 8,
) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        if prefix < 1:
            raise ValueError("executed prefix must be positive")
        for seed in seeds:
            _set_seed(seed)
            seed_dir = root / checkpoint_root / f"seed{seed}"
            response_model, recurrent_model, metadata = _load_checkpoint_models(seed_dir, device)
            eval_config = copy.deepcopy(raw)
            eval_config["dataset"]["execute_steps"] = prefix
            data_seed = metadata["data_seed"]
            dataset_cfg = eval_config["dataset"]
            evaluation_cfg = eval_config.get("evaluation", {})
            id_dataset = _make_dataset(eval_config, metadata["id_split"], data_seed)
            ood_dataset = _make_dataset(
                eval_config,
                "ood",
                data_seed,
                action_policy=str(dataset_cfg.get("ood_action_policy", "correlated")),
            )
            loader_kwargs = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}
            for split, dataset in (("id", id_dataset), ("ood", ood_dataset)):
                _, eval_rows = _evaluate(
                    response_model,
                    recurrent_model,
                    torch.utils.data.DataLoader(dataset, **loader_kwargs),
                    device=device,
                    dt=float(dataset_cfg.get("dt", 0.4)),
                    likelihood_scale=float(evaluation_cfg.get("likelihood_scale", 0.05)),
                    confidence_threshold=float(evaluation_cfg.get("confidence_threshold", 0.8)),
                    execute_steps=prefix,
                    split_name=split,
                )
                for first, second, contrast in METHOD_PAIRS:
                    mean, paired_rows = _paired_mean(eval_rows, first, second)
                    rows.append(
                        {
                            "prefix": prefix,
                            "split": split,
                            "seed": seed,
                            "contrast": contrast,
                            "mean": mean,
                            "paired_rows": paired_rows,
                        }
                    )
            del response_model, recurrent_model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    summary_rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        for split in ("id", "ood"):
            for _, _, contrast in METHOD_PAIRS:
                values = [
                    float(row["mean"])
                    for row in rows
                    if row["prefix"] == prefix
                    and row["split"] == split
                    and row["contrast"] == contrast
                ]
                summary_rows.append(
                    {
                        "prefix": prefix,
                        "split": split,
                        "contrast": contrast,
                        "mean": statistics.fmean(values),
                        "sd_across_seeds": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "n_seeds": len(values),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "SUPPLEMENTARY_PREFIX_SENSITIVITY",
        "protocol": {
            "config": str(config_path),
            "checkpoint_root": str(checkpoint_root),
            "prefixes": list(prefixes),
            "model_seeds": list(seeds),
            "data_seed_source": "frozen checkpoint metadata",
            "interpretation": "executed-prefix sensitivity diagnostic; separate from Formal-v1 gate",
        },
        "rows": summary_rows,
    }
    (output_dir / "prefix_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "prefix_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(summary_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    lines = [
        "# H2 Executed-Prefix Sensitivity",
        "",
        "Frozen H2 checkpoints are evaluated with one-, three-, and six-step executed prefixes. Positive values favor Persistent because they are reset/shuffled/recurrent minus Persistent predictive NLL.",
        "",
        "| Prefix | Split | Contrast | Mean | SD across seeds | n |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['prefix']} | {row['split']} | {row['contrast']} | "
            f"{row['mean']:.6f} | {row['sd_across_seeds']:.6f} | {row['n_seeds']} |"
        )
    lines.append("")
    lines.append("This diagnostic tests sensitivity to the disclosed execution-prefix choice; it is not a new confirmatory gate.")
    (output_dir / "prefix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_summary={output_dir / 'prefix_summary.json'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/persistent_trajectory_formal.yaml")
    parser.add_argument("--checkpoint-root", default="experiments/formal_v1/h2/persistent_trajectory_suite_fast")
    parser.add_argument("--output-dir", default="reports/h2_prefix_sensitivity_2026-08-21")
    parser.add_argument("--prefixes", nargs="+", type=int, default=list(DEFAULT_PREFIXES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run(
        Path(args.root).resolve(),
        Path(args.config),
        Path(args.checkpoint_root),
        Path(args.output_dir),
        tuple(args.prefixes),
        tuple(args.seeds),
        args.device,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
