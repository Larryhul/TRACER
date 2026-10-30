from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from tracer.data.synthetic_navigation import collate_synthetic_scene_batch
from tracer.experiments.formal_manifest import mark_formal_run, prepare_formal_run
from tracer.response.causal_synthetic import make_causal_response_dataset_from_config
from tracer.response.data import (
    build_entity_response_batch,
    shuffle_robot_candidates_within_groups,
)
from tracer.response.factorial import factorial_second_difference, interaction_recovery_metrics
from tracer.response.metrics import response_metrics
from tracer.response.models import ResponseModelConfig, TeamResponsePredictor


TRAINED_VARIANTS = ("additive", "matched-additive", "joint")
EVALUATED_METHODS = (*TRAINED_VARIANTS, "joint-without-pair", "shuffled-joint")
CONTEXT_BASELINES = ("history", "nearest")


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_models(
    run_dir: Path,
    device: str,
    variants: tuple[str, ...] = TRAINED_VARIANTS,
) -> tuple[dict[str, TeamResponsePredictor], dict]:
    models: dict[str, TeamResponsePredictor] = {}
    counts: dict[str, dict[str, int]] = {}
    data_seeds: set[int] = set()
    for variant in variants:
        checkpoint = torch.load(
            run_dir / f"{variant}.pt", map_location=device, weights_only=True
        )
        model = TeamResponsePredictor(
            variant, ResponseModelConfig(**checkpoint["model_config"])
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        models[variant] = model.eval()
        counts[variant] = {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": int(checkpoint.get("trainable_parameters", sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ))),
        }
        if checkpoint.get("data_seed") is not None:
            data_seeds.add(int(checkpoint["data_seed"]))
    if len(data_seeds) > 1:
        raise ValueError(f"checkpoint data seeds disagree in {run_dir}: {sorted(data_seeds)}")
    joint_count = counts["joint"]["trainable"]
    matched_count = counts["matched-additive"]["trainable"]
    total_joint_count = counts["joint"]["total"]
    total_matched_count = counts["matched-additive"]["total"]
    trainable_difference = abs(matched_count - joint_count) / max(joint_count, 1)
    return models, {
        "counts": counts,
        "matched_vs_joint_relative_difference": trainable_difference,
        "matched_vs_joint_trainable_relative_difference": trainable_difference,
        "matched_vs_joint_total_relative_difference": abs(
            total_matched_count - total_joint_count
        ) / max(total_joint_count, 1),
        "checkpoint_data_seed": next(iter(data_seeds), None),
    }


def _method_output(
    method: str,
    models: dict[str, TeamResponsePredictor],
    batch,
    shuffled_batch=None,
) -> dict[str, torch.Tensor]:
    if method == "joint-without-pair":
        return models["joint"](batch, include_pair_interactions=False)
    if method == "shuffled-joint":
        return models["joint"](shuffled_batch if shuffled_batch is not None else batch)
    return models[method](batch)


def _finite_mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    return float(finite.mean()) if finite.numel() else float("nan")


def _evaluate_run(
    raw: dict[str, Any],
    run_dir: Path,
    effect_mode: str,
    model_seed: int,
    data_seed: int,
    device: str,
    split: str,
    record_scope: str,
    allow_data_seed_mismatch: bool,
    include_context_baselines: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = make_causal_response_dataset_from_config(
        raw, split=split, effect_mode=effect_mode, seed=data_seed
    )
    if dataset.num_robots != 2 or dataset.interventions_per_scene != 4:
        raise ValueError("H1 paired pilot requires R=2 and interventions U00/U10/U01/U11")
    model_cfg = raw.get("model", {})
    evaluation_cfg = raw.get("evaluation", {})
    num_robots = int(model_cfg.get("num_robots", 2))
    interaction_radius = float(evaluation_cfg.get("interaction_radius", 4.0))
    model_variants = (
        (*CONTEXT_BASELINES, *TRAINED_VARIANTS)
        if include_context_baselines
        else TRAINED_VARIANTS
    )
    evaluated_methods = (
        (*CONTEXT_BASELINES, *EVALUATED_METHODS)
        if include_context_baselines
        else EVALUATED_METHODS
    )
    models, parameter_audit = _load_models(run_dir, device, model_variants)
    checkpoint_data_seed = parameter_audit["checkpoint_data_seed"]
    if (
        checkpoint_data_seed is not None
        and checkpoint_data_seed != data_seed
        and not allow_data_seed_mismatch
    ):
        raise ValueError(
            f"evaluation data_seed={data_seed} differs from checkpoint data_seed={checkpoint_data_seed}; "
            "pass --allow-data-seed-mismatch only for an explicitly independent evaluation corpus"
        )

    records: list[dict[str, Any]] = []
    diagnostics = {
        "robot_permutation_max_abs_error": 0.0,
        "pair_zero_condition_max_abs_error": 0.0,
        "pair_one_robot_mask_max_abs_error": 0.0,
        "matched_additive_second_difference_max_abs": 0.0,
        "joint_without_pair_matches_additive_max_abs_error": 0.0,
    }
    with torch.inference_mode():
        for local_base_id in range(dataset.num_base_scenes):
            first = local_base_id * dataset.interventions_per_scene
            samples = [dataset[first + intervention] for intervention in range(4)]
            scene = collate_synthetic_scene_batch(samples)
            batch = build_entity_response_batch(
                scene,
                num_robots=num_robots,
                interaction_radius=interaction_radius,
            ).to(device)
            num_entities = batch.num_samples // 4
            shuffled_batch = shuffle_robot_candidates_within_groups(
                batch,
                group_size=4 * num_entities,
                branch_size=num_entities,
            )
            base_scene_id = int(samples[0]["base_scene_id"])
            target = batch.target_future_vel.reshape(
                4, num_entities, dataset.horizon, 2
            )
            target_interaction = factorial_second_difference(target)

            outputs = {
                method: _method_output(method, models, batch, shuffled_batch)
                for method in evaluated_methods
            }
            diagnostics["joint_without_pair_matches_additive_max_abs_error"] = max(
                diagnostics["joint_without_pair_matches_additive_max_abs_error"],
                float(
                    (outputs["joint-without-pair"]["mean"] - outputs["additive"]["mean"])
                    .abs()
                    .max()
                ),
            )
            for method, output in outputs.items():
                sample_metrics = response_metrics(
                    output["mean"], output["log_std"], batch.target_future_vel, batch.dt
                )
                high_mask = batch.high_interaction
                predicted = output["mean"].reshape(
                    4, num_entities, dataset.horizon, 2
                )
                interaction = interaction_recovery_metrics(
                    factorial_second_difference(predicted), target_interaction
                )
                records.append(
                    {
                        "effect_mode": effect_mode,
                        "model_seed": model_seed,
                        "data_seed": data_seed,
                        "split": f"{record_scope}_{split}",
                        "base_scene_id": base_scene_id,
                        "method": method,
                        "num_entities": num_entities,
                        "nll": _finite_mean(sample_metrics["nll"]),
                        "ade": _finite_mean(sample_metrics["ade"]),
                        "fde": _finite_mean(sample_metrics["fde"]),
                        "high_nll": _finite_mean(sample_metrics["nll"][high_mask]),
                        "high_ade": _finite_mean(sample_metrics["ade"][high_mask]),
                        "interaction_rmse": interaction["rmse"],
                        "interaction_mae": interaction["mae"],
                        "interaction_bias": interaction["bias"],
                        "interaction_pearson": interaction["pearson"],
                        "interaction_target_rms": interaction["target_rms"],
                        "interaction_prediction_rms": interaction["prediction_rms"],
                        "interaction_energy_ratio": interaction["energy_ratio"],
                    }
                )

            swapped = replace(
                batch,
                robot_current=batch.robot_current.flip(1),
                robot_candidates=batch.robot_candidates.flip(1),
                robot_mask=batch.robot_mask.flip(1),
            )
            swapped_mean = models["joint"](swapped)["mean"]
            diagnostics["robot_permutation_max_abs_error"] = max(
                diagnostics["robot_permutation_max_abs_error"],
                float((outputs["joint"]["mean"] - swapped_mean).abs().max()),
            )

            zero_second = replace(
                batch,
                robot_current=batch.robot_current.clone(),
                robot_candidates=batch.robot_candidates.clone(),
            )
            zero_second.robot_current[:, 1] = 0.0
            zero_second.robot_candidates[:, 1] = 0.0
            zero_with_pair = models["joint"](zero_second)["mean"]
            zero_without_pair = models["joint"](
                zero_second, include_pair_interactions=False
            )["mean"]
            diagnostics["pair_zero_condition_max_abs_error"] = max(
                diagnostics["pair_zero_condition_max_abs_error"],
                float((zero_with_pair - zero_without_pair).abs().max()),
            )

            one_robot = replace(
                batch,
                robot_mask=torch.stack(
                    [batch.robot_mask[:, 0], torch.zeros_like(batch.robot_mask[:, 1])],
                    dim=1,
                ),
            )
            one_with_pair = models["joint"](one_robot)["mean"]
            one_without_pair = models["joint"](
                one_robot, include_pair_interactions=False
            )["mean"]
            diagnostics["pair_one_robot_mask_max_abs_error"] = max(
                diagnostics["pair_one_robot_mask_max_abs_error"],
                float((one_with_pair - one_without_pair).abs().max()),
            )

            matched_means = outputs["matched-additive"]["mean"].reshape(
                4, num_entities, dataset.horizon, 2
            )
            diagnostics["matched_additive_second_difference_max_abs"] = max(
                diagnostics["matched_additive_second_difference_max_abs"],
                float(factorial_second_difference(matched_means).abs().max()),
            )

    diagnostics.update(parameter_audit)
    diagnostics["num_base_scenes"] = dataset.num_base_scenes
    diagnostics["num_records"] = len(records)
    return records, diagnostics


def _effect_summary(values: list[float], seed: int, bootstrap_resamples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
    dz = mean / sd if math.isfinite(sd) and sd > 0.0 else float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(bootstrap_resamples, len(array)))
    samples = array[indices]
    means = samples.mean(axis=1)
    sample_sd = samples.std(axis=1, ddof=1)
    standardized = np.divide(
        means,
        sample_sd,
        out=np.full_like(means, np.nan),
        where=sample_sd > 0.0,
    )

    def interval(values_: np.ndarray, alpha: float) -> list[float]:
        finite = values_[np.isfinite(values_)]
        if finite.size == 0:
            return [float("nan"), float("nan")]
        return [
            float(np.quantile(finite, alpha / 2.0)),
            float(np.quantile(finite, 1.0 - alpha / 2.0)),
        ]

    return {
        "n_base_scenes": len(array),
        "mean": mean,
        "sd": sd,
        "dz": dz,
        "mean_ci95": interval(means, 0.05),
        "dz_ci90": interval(standardized, 0.10),
    }


def _paired_values(
    records: list[dict[str, Any]],
    metric: str,
    first_method: str,
    second_method: str,
) -> list[float]:
    by_scene: dict[int, dict[str, float]] = {}
    for row in records:
        by_scene.setdefault(int(row["base_scene_id"]), {})[str(row["method"])] = float(
            row[metric]
        )
    values = []
    for methods in by_scene.values():
        if first_method in methods and second_method in methods:
            values.append(methods[first_method] - methods[second_method])
    return values


def _summarize_cell(
    records: list[dict[str, Any]], seed: int, bootstrap_resamples: int
) -> dict[str, Any]:
    contrasts = {
        "matched_additive_minus_joint_nll": ("nll", "matched-additive", "joint"),
        "strict_additive_minus_joint_nll": ("nll", "additive", "joint"),
        "joint_without_pair_minus_joint_nll": ("nll", "joint-without-pair", "joint"),
        "shuffled_minus_joint_nll": ("nll", "shuffled-joint", "joint"),
        "matched_additive_minus_joint_interaction_rmse": (
            "interaction_rmse",
            "matched-additive",
            "joint",
        ),
    }
    return {
        name: _effect_summary(
            _paired_values(records, metric, first_method, second_method),
            seed + index * 104729,
            bootstrap_resamples,
        )
        for index, (name, (metric, first_method, second_method)) in enumerate(
            contrasts.items()
        )
    }


def _average_seeds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in records:
        key = (str(row["effect_mode"]), int(row["base_scene_id"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    averaged: list[dict[str, Any]] = []
    metric_fields = [
        "nll",
        "ade",
        "fde",
        "high_nll",
        "high_ade",
        "interaction_rmse",
        "interaction_mae",
        "interaction_bias",
        "interaction_pearson",
        "interaction_target_rms",
        "interaction_prediction_rms",
        "interaction_energy_ratio",
    ]
    for (mode, base_scene_id, method), rows in grouped.items():
        result: dict[str, Any] = {
            "effect_mode": mode,
            "base_scene_id": base_scene_id,
            "method": method,
        }
        for field in metric_fields:
            finite = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
            result[field] = statistics.fmean(finite) if finite else float("nan")
        averaged.append(result)
    return averaged


def _summarize_absolute_metrics(
    records: list[dict[str, Any]], seed: int, bootstrap_resamples: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    methods = sorted({str(row["method"]) for row in records})
    for method_index, method in enumerate(methods):
        method_records = [row for row in records if row["method"] == method]
        result[method] = {}
        for metric_index, metric in enumerate(("nll", "ade", "fde", "high_nll", "high_ade")):
            values = [
                float(row[metric])
                for row in method_records
                if math.isfinite(float(row[metric]))
            ]
            summary = _effect_summary(
                values,
                seed + method_index * 104729 + metric_index * 49979687,
                bootstrap_resamples,
            )
            result[method][metric] = {
                "n_base_scenes": summary["n_base_scenes"],
                "mean": summary["mean"],
                "sd": summary["sd"],
                "mean_ci95": summary["mean_ci95"],
            }
    return result


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("cannot write an empty H1 record table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the paired, parameter-matched H1 pilot.")
    parser.add_argument("--config", default="configs/h1_paired_pilot.yaml")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modes", nargs="+", choices=["none", "additive", "nonlinear"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--allow-data-seed-mismatch", action="store_true")
    parser.add_argument("--include-context-baselines", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()

    raw = _load_config(args.config)
    suite_cfg = raw.get("suite", {})
    causal_cfg = raw.get("dataset", {}).get("causal_response", {})
    modes = args.modes or list(suite_cfg.get("modes", ["none", "additive", "nonlinear"]))
    seeds = args.seeds or [int(value) for value in suite_cfg.get("seeds", [2027, 2028, 2029])]
    data_seed = int(args.data_seed if args.data_seed is not None else causal_cfg.get("seed", 81173))
    checkpoint_root = Path(args.checkpoint_root)
    output_dir = Path(args.output_dir)
    formal = "formal_protocol" in raw
    if formal:
        if args.include_context_baselines:
            raise RuntimeError(
                "formal H1 method set is frozen; context baselines are exploratory only"
            )
        protocol = raw["formal_protocol"]
        expected_modes = list(suite_cfg["modes"])
        expected_seeds = [int(value) for value in suite_cfg["seeds"]]
        expected_data_seed = int(protocol["formal_test_data_seed"])
        expected_split = str(suite_cfg["evaluation_split"])
        expected_checkpoint_root = Path(suite_cfg["checkpoint_root"]).resolve()
        expected_output_dir = Path(suite_cfg["output_dir"]).resolve()
        expected_resamples = int(raw.get("evaluation", {}).get("bootstrap_resamples", 10000))
        if protocol.get("execution") != "evaluation_only_frozen_checkpoints":
            raise RuntimeError("formal H1 must use evaluation_only_frozen_checkpoints")
        if modes != expected_modes or seeds != expected_seeds:
            raise RuntimeError("formal H1 modes and model seeds are frozen by the config")
        if data_seed != expected_data_seed or args.split != expected_split:
            raise RuntimeError("formal H1 data seed and split are frozen by the config")
        if checkpoint_root.resolve() != expected_checkpoint_root:
            raise RuntimeError("formal H1 checkpoint root is frozen by the config")
        if output_dir.resolve() != expected_output_dir:
            raise RuntimeError("formal H1 output directory is frozen by the config")
        if args.bootstrap_resamples != expected_resamples:
            raise RuntimeError("formal H1 bootstrap count is frozen by the config")
        if not args.allow_data_seed_mismatch:
            raise RuntimeError(
                "formal H1 requires the explicit independent-corpus mismatch flag"
            )
        checkpoint_artifacts = [
            checkpoint_root / mode / f"seed{model_seed}" / f"{variant}.pt"
            for mode in modes
            for model_seed in seeds
            for variant in TRAINED_VARIANTS
        ]
        prepare_formal_run(
            Path(args.config),
            output_dir,
            Path.cwd(),
            command=list(sys.argv),
            overrides={
                "device": args.device,
                "modes": modes,
                "seeds": seeds,
                "data_seed": data_seed,
                "split": args.split,
                "bootstrap_resamples": args.bootstrap_resamples,
                "checkpoint_root": str(checkpoint_root),
                "output_dir": str(output_dir),
            },
            input_artifacts=checkpoint_artifacts,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    run_diagnostics: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}
    for mode in modes:
        for model_seed in seeds:
            run_key = f"{mode}/{model_seed}"
            run_dir = checkpoint_root / mode / f"seed{model_seed}"
            print(f"h1_eval mode={mode} model_seed={model_seed} data_seed={data_seed}", flush=True)
            records, diagnostics = _evaluate_run(
                raw,
                run_dir,
                mode,
                model_seed,
                data_seed,
                args.device,
                args.split,
                "formal" if formal else "pilot",
                args.allow_data_seed_mismatch,
                args.include_context_baselines,
            )
            all_records.extend(records)
            run_diagnostics[run_key] = diagnostics
            per_seed[run_key] = _summarize_cell(
                records, model_seed + data_seed, args.bootstrap_resamples
            )
            _write_records(output_dir / mode / f"seed{model_seed}" / "paired_records.csv", records)
            (output_dir / mode / f"seed{model_seed}" / "diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    _write_records(output_dir / "paired_records.csv", all_records)
    averaged = _average_seeds(all_records)
    pooled: dict[str, Any] = {}
    baseline_metrics: dict[str, Any] = {}
    for mode in modes:
        mode_records = [row for row in averaged if row["effect_mode"] == mode]
        pooled[mode] = _summarize_cell(
            mode_records, data_seed + modes.index(mode) * 49979687, args.bootstrap_resamples
        )
        if args.include_context_baselines:
            baseline_metrics[mode] = _summarize_absolute_metrics(
                mode_records,
                data_seed + modes.index(mode) * 32452843,
                args.bootstrap_resamples,
            )

    max_diagnostics = {
        name: max(float(values[name]) for values in run_diagnostics.values())
        for name in (
            "robot_permutation_max_abs_error",
            "pair_zero_condition_max_abs_error",
            "pair_one_robot_mask_max_abs_error",
            "matched_additive_second_difference_max_abs",
            "joint_without_pair_matches_additive_max_abs_error",
            "matched_vs_joint_relative_difference",
            "matched_vs_joint_trainable_relative_difference",
        )
    }
    nonlinear = pooled.get("nonlinear", {})
    corrected_paired_shuffle = str(
        raw.get("formal_protocol", {}).get("version", "")
    ).endswith("corrected-paired-shuffle")
    if corrected_paired_shuffle:
        none_shuffle_low, none_shuffle_high = pooled["none"][
            "shuffled_minus_joint_nll"
        ]["mean_ci95"]
        negative_equivalence = [
            none_shuffle_low > -0.003 and none_shuffle_high < 0.003
        ]
    else:
        negative_equivalence = []
        for mode in ("none", "additive"):
            if mode not in pooled:
                continue
            low, high = pooled[mode]["matched_additive_minus_joint_nll"]["dz_ci90"]
            negative_equivalence.append(low > -0.2 and high < 0.2)
    nonlinear_nll = nonlinear.get("matched_additive_minus_joint_nll", {})
    nonlinear_pair = nonlinear.get("matched_additive_minus_joint_interaction_rmse", {})
    gates = {
        "parameter_match_within_1pct": max_diagnostics[
            "matched_vs_joint_trainable_relative_difference"
        ] < 0.01,
        "structural_checks_below_1e-5": max(
            value
            for name, value in max_diagnostics.items()
            if name not in {
                "matched_vs_joint_relative_difference",
                "matched_vs_joint_trainable_relative_difference",
            }
        ) < 1e-5,
        "negative_controls_equivalent_dz_0p2": bool(negative_equivalence)
        and all(negative_equivalence),
        "nonlinear_nll_ci_positive_and_dz_0p2": bool(nonlinear_nll)
        and nonlinear_nll["mean_ci95"][0] > 0.0
        and nonlinear_nll["dz"] >= 0.2,
        "nonlinear_interaction_rmse_improves": bool(nonlinear_pair)
        and nonlinear_pair["mean_ci95"][0] > 0.0,
    }
    if corrected_paired_shuffle:
        gates["none_paired_shuffle_raw_nll_ci_within_0p003"] = bool(
            negative_equivalence[0]
        )
        additive_shuffle = pooled["additive"]["shuffled_minus_joint_nll"]
        gates["additive_paired_shuffle_degrades"] = (
            additive_shuffle["mean_ci95"][0] > 0.0
        )
    gate_name = "formal_go" if formal else "pilot_go"
    gates[gate_name] = all(gates.values())
    summary = {
        "status": "FORMAL_CONFIRMATORY" if formal else "PILOT_ONLY_NOT_CONFIRMATORY",
        "data_seed": data_seed,
        "allow_data_seed_mismatch": args.allow_data_seed_mismatch,
        "split": args.split,
        "model_seeds": seeds,
        "modes": modes,
        "bootstrap_resamples": args.bootstrap_resamples,
        "independent_unit": "base_scene",
        "seed_handling": "average method metrics over model seeds within base scene, then bootstrap base scenes",
        "per_seed": per_seed,
        "pooled_across_seeds": pooled,
        "context_baselines_included": args.include_context_baselines,
        "baseline_metrics": baseline_metrics,
        "diagnostic_maxima": max_diagnostics,
        "parameter_audit": {
            key: value["counts"] for key, value in run_diagnostics.items()
        },
        "gates": gates,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if formal:
        mark_formal_run(output_dir, "COMPLETED", Path.cwd())
    print(f"h1_{gate_name}={gates[gate_name]} saved_summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
