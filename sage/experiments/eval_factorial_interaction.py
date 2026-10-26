from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
import yaml

from sage.data.synthetic_navigation import collate_synthetic_scene_batch
from sage.response.causal_synthetic import make_causal_response_dataset_from_config
from sage.response.data import build_entity_response_batch
from sage.response.factorial import factorial_second_difference, interaction_recovery_metrics
from sage.response.models import ResponseModelConfig, TeamResponsePredictor


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_models(
    run_dir: Path,
    variants: list[str],
    device: str,
) -> dict[str, TeamResponsePredictor]:
    models: dict[str, TeamResponsePredictor] = {}
    for variant in variants:
        checkpoint_path = run_dir / f"{variant}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = TeamResponsePredictor(
            variant,
            ResponseModelConfig(**checkpoint["model_config"]),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        models[variant] = model.eval()
    return models


def evaluate_run(
    raw_config: dict[str, Any],
    run_dir: Path,
    effect_mode: str,
    seed: int,
    variants: list[str],
    device: str,
) -> dict[str, Any]:
    dataset = make_causal_response_dataset_from_config(
        raw_config,
        split="val",
        effect_mode=effect_mode,
        seed=seed,
    )
    if dataset.interventions_per_scene != 4:
        raise ValueError("factorial interaction evaluation requires exactly four interventions")

    model_cfg = raw_config.get("model", {})
    evaluation_cfg = raw_config.get("evaluation", {})
    num_robots = int(model_cfg.get("num_robots", 2))
    interaction_radius = float(evaluation_cfg.get("interaction_radius", 4.0))
    models = _load_models(run_dir, variants, device)

    targets: list[torch.Tensor] = []
    high_masks: list[torch.Tensor] = []
    predictions: dict[str, list[torch.Tensor]] = {variant: [] for variant in variants}
    with torch.inference_mode():
        for base_scene_id in range(dataset.num_base_scenes):
            first = base_scene_id * dataset.interventions_per_scene
            samples = [dataset[first + intervention] for intervention in range(4)]
            scene = collate_synthetic_scene_batch(samples)
            batch = build_entity_response_batch(
                scene,
                num_robots=num_robots,
                interaction_radius=interaction_radius,
            ).to(device)
            num_entities = batch.num_samples // 4
            target = batch.target_future_vel.reshape(4, num_entities, dataset.horizon, 2)
            targets.append(factorial_second_difference(target).cpu())
            high_masks.append(batch.high_interaction.reshape(4, num_entities)[0].cpu())
            for variant, model in models.items():
                mean = model(batch)["mean"].reshape(4, num_entities, dataset.horizon, 2)
                predictions[variant].append(factorial_second_difference(mean).cpu())

    target = torch.stack(targets)
    high_mask = torch.stack(high_masks)
    result: dict[str, Any] = {
        "effect_mode": effect_mode,
        "seed": seed,
        "num_base_scenes": dataset.num_base_scenes,
        "num_entities": int(target.shape[0] * target.shape[1]),
        "high_interaction_fraction": float(high_mask.float().mean()),
        "variants": {},
    }
    for variant in variants:
        prediction = torch.stack(predictions[variant])
        result["variants"][variant] = {
            "all": interaction_recovery_metrics(prediction, target),
            "high_interaction": interaction_recovery_metrics(
                prediction[high_mask], target[high_mask]
            ),
        }

    additive_rmse = result["variants"]["additive"]["all"]["rmse"]
    joint_rmse = result["variants"]["joint"]["all"]["rmse"]
    high_additive_rmse = result["variants"]["additive"]["high_interaction"]["rmse"]
    high_joint_rmse = result["variants"]["joint"]["high_interaction"]["rmse"]
    result["contrasts"] = {
        "additive_minus_joint_rmse": additive_rmse - joint_rmse,
        "joint_relative_rmse_reduction": (
            1.0 - joint_rmse / additive_rmse if additive_rmse > 1e-6 else float("nan")
        ),
        "high_additive_minus_joint_rmse": high_additive_rmse - high_joint_rmse,
        "high_joint_relative_rmse_reduction": (
            1.0 - high_joint_rmse / high_additive_rmse
            if high_additive_rmse > 1e-6
            else float("nan")
        ),
    }
    return result


def _aggregate(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for run_key, run in runs.items():
        mode, _seed = run_key.split("/", maxsplit=1)
        flattened = {
            "target_rms": run["variants"]["joint"]["all"]["target_rms"],
            "additive_prediction_rms": run["variants"]["additive"]["all"]["prediction_rms"],
            "joint_prediction_rms": run["variants"]["joint"]["all"]["prediction_rms"],
            "additive_rmse": run["variants"]["additive"]["all"]["rmse"],
            "joint_rmse": run["variants"]["joint"]["all"]["rmse"],
            "joint_pearson": run["variants"]["joint"]["all"]["pearson"],
            "joint_cosine": run["variants"]["joint"]["all"]["cosine"],
            "joint_zero_baseline_r2": run["variants"]["joint"]["all"]["zero_baseline_r2"],
            "high_target_rms": run["variants"]["joint"]["high_interaction"]["target_rms"],
            "high_additive_rmse": run["variants"]["additive"]["high_interaction"]["rmse"],
            "high_joint_rmse": run["variants"]["joint"]["high_interaction"]["rmse"],
            "high_joint_pearson": run["variants"]["joint"]["high_interaction"]["pearson"],
            "high_joint_zero_baseline_r2": run["variants"]["joint"]["high_interaction"][
                "zero_baseline_r2"
            ],
            **run["contrasts"],
        }
        for name, value in flattened.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                grouped.setdefault(mode, {}).setdefault(name, []).append(float(value))

    return {
        mode: {
            name: {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "count": len(values),
            }
            for name, values in metrics.items()
        }
        for mode, metrics in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recovery of 2x2 response interactions.")
    parser.add_argument("--config", default="configs/response_causal_suite.yaml")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modes", nargs="+", choices=["none", "additive", "nonlinear"])
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()

    raw = _load_config(args.config)
    suite_cfg = raw.get("suite", {})
    modes = args.modes or list(suite_cfg.get("modes", ["none", "additive", "nonlinear"]))
    seeds = args.seeds or [int(seed) for seed in suite_cfg.get("seeds", [2027, 2028, 2029])]
    checkpoint_root = Path(args.checkpoint_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = ["additive", "joint"]

    runs: dict[str, dict[str, Any]] = {}
    for mode in modes:
        for seed in seeds:
            run_key = f"{mode}/{seed}"
            print(f"factorial_eval mode={mode} seed={seed} device={args.device}", flush=True)
            runs[run_key] = evaluate_run(
                raw,
                checkpoint_root / mode / f"seed{seed}",
                mode,
                seed,
                variants,
                args.device,
            )
            (output_dir / "runs.json").write_text(
                json.dumps(runs, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    summary = _aggregate(runs)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for mode in modes:
        values = summary[mode]
        print(
            f"factorial_summary mode={mode} "
            f"target_rms={values['target_rms']['mean']:.6g} "
            f"additive_rmse={values['additive_rmse']['mean']:.6g} "
            f"joint_rmse={values['joint_rmse']['mean']:.6g} "
            f"joint_pearson={values.get('joint_pearson', {}).get('mean', float('nan')):.6g}",
            flush=True,
        )
    print(f"saved_results={output_dir}", flush=True)


if __name__ == "__main__":
    main()
