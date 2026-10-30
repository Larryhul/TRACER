"""Evaluation-only H1 baseline and response-family transfer summary.

The script reuses frozen checkpoints from the causal benchmark matrix and
evaluates one training family on several disjoint response families.  It is
deliberately separate from the Formal-v2 H1 evaluator: no formal checkpoint,
test split, or gate is modified.  ``independent`` is the exact structural
reduction of ``joint`` with its pair residual disabled; this is recorded
explicitly instead of pretending it is a separately trained architecture.

In addition to continuous Gaussian metrics, the report includes two
calibration diagnostics.  ``coverage_1sigma``/``coverage_2sigma`` are the
coordinate-wise empirical Gaussian interval coverages.  ``velocity_sign_brier``
and ``velocity_sign_ece`` evaluate the fixed binary event ``v_coordinate > 0``
using the Gaussian predictive probability.  This makes Brier/ECE well-defined
for a continuous response without turning the four paired interventions into
a spurious classification task.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from tracer.data.synthetic_navigation import collate_synthetic_scene_batch
from tracer.response.causal_synthetic import make_causal_response_dataset_from_config
from tracer.response.data import (
    build_entity_response_batch,
    shuffle_robot_candidates_within_groups,
)
from tracer.response.factorial import factorial_second_difference, interaction_recovery_metrics
from tracer.response.metrics import gaussian_nll_per_sample, response_metrics
from tracer.response.models import ResponseModelConfig, TeamResponsePredictor, build_response_predictor
from tracer.response.external_baselines import METHOD_TAXONOMY, external_method_metadata


METHODS = (
    "history-only",
    "nearest-robot",
    "independent",
    "additive",
    "matched-additive",
    "joint",
    "shuffled-joint",
    "eiffert2019-style",
    "ScePT-style",
)
MODEL_VARIANTS = ("history", "nearest", "additive", "matched-additive", "joint", "eiffert2019-style", "team-concat")
KEDIA_MODEL_VARIANT = "kedia-style"
KEDIA_METHOD_NAME = "Kedia-style"
METHOD_DISPLAY = {
    "history": "history-only",
    "nearest": "nearest-robot",
    "team-concat": "ScePT-style",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_models(
    checkpoint_dir: Path,
    device: str,
    include_kedia: bool = False,
) -> dict[str, TeamResponsePredictor]:
    models: dict[str, TeamResponsePredictor] = {}
    variants = MODEL_VARIANTS + ((KEDIA_MODEL_VARIANT,) if include_kedia else ())
    for variant in variants:
        checkpoint_path = checkpoint_dir / f"{variant}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        config = ResponseModelConfig(**checkpoint["model_config"])
        model = build_response_predictor(variant, config).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        models[variant] = model
    return models


def _finite_mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    return float(finite.mean()) if finite.numel() else float("nan")


def _probabilistic_diagnostics(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
    groups: int,
    entities: int,
    horizon: int,
) -> tuple[float, float, float, float, float]:
    """Return velocity-sign Brier/ECE and Gaussian calibration diagnostics."""

    sigma = torch.exp(log_std).clamp_min(1e-8)
    # P(v > 0) under each predicted scalar Gaussian.  The event threshold is
    # fixed before seeing any target family and is not fitted on test data.
    probabilities = 0.5 * (1.0 + torch.erf(mean / (sigma * math.sqrt(2.0))))
    labels = (target > 0.0).to(probabilities.dtype)
    brier = (probabilities - labels).square().mean()
    confidence = torch.maximum(probabilities, 1.0 - probabilities)
    prediction = probabilities >= 0.5
    correctness = prediction.eq(labels.bool()).to(probabilities.dtype)
    ece = torch.zeros((), device=probabilities.device, dtype=probabilities.dtype)
    for lower in torch.arange(0.0, 1.0, 0.1, device=probabilities.device):
        upper = lower + 0.1
        mask = (confidence >= lower) & (
            (confidence < upper) if upper < 1.0 else (confidence <= upper)
        )
        if mask.any():
            ece = ece + mask.to(probabilities.dtype).mean() * (
                confidence[mask].mean() - correctness[mask].mean()
            ).abs()
    z = (target - mean) / torch.exp(log_std).clamp_min(1e-8)
    coverage_1 = (z.abs() <= 1.0).to(z.dtype).mean()
    coverage_2 = (z.abs() <= 2.0).to(z.dtype).mean()
    abs_z = z.abs().mean()
    return float(brier), float(ece), float(coverage_1), float(coverage_2), float(abs_z)


def _evaluate_target_family(
    dataset,
    models: dict[str, TeamResponsePredictor],
    device: str,
    train_family: str,
    target_family: str,
    data_seed: int,
    test_chunk: int,
    methods: tuple[str, ...] = METHODS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entities = int(dataset.num_entities)
    horizon = int(dataset.horizon)
    intervention_count = int(dataset.interventions_per_scene)
    if intervention_count != 4:
        raise ValueError("the unified H1 report requires the four paired R=2 interventions")
    with torch.inference_mode():
        for start in range(0, dataset.num_base_scenes, test_chunk):
            stop = min(start + test_chunk, dataset.num_base_scenes)
            samples = []
            for base_scene_id in range(start, stop):
                first = base_scene_id * intervention_count
                samples.extend(dataset[first + intervention] for intervention in range(intervention_count))
            scene = collate_synthetic_scene_batch(samples)
            batch = build_entity_response_batch(
                scene,
                num_robots=2,
                interaction_radius=float(dataset.spec.influence_radius),
            ).to(device)
            groups = stop - start
            target = batch.target_future_vel.reshape(groups, 4, entities, horizon, 2)
            outputs: dict[str, dict[str, torch.Tensor]] = {}
            for variant, model in models.items():
                outputs[variant] = model(batch)
            # The exact no-pair reduction is used for the independent baseline.
            outputs["independent"] = models["joint"](batch, include_pair_interactions=False)
            shuffled_batch = shuffle_robot_candidates_within_groups(
                batch,
                group_size=intervention_count * entities,
                branch_size=entities,
            )
            outputs["shuffled-joint"] = models["joint"](shuffled_batch)
            outputs["history-only"] = outputs.pop("history")
            outputs["nearest-robot"] = outputs.pop("nearest")
            outputs["ScePT-style"] = outputs.pop("team-concat")
            if KEDIA_MODEL_VARIANT in outputs:
                outputs[KEDIA_METHOD_NAME] = outputs.pop(KEDIA_MODEL_VARIANT)

            for method in methods:
                output = outputs[method]
                mean = output["mean"].reshape(groups, 4, entities, horizon, 2)
                log_std = output["log_std"].reshape(groups, 4, entities, horizon, 2)
                metric = response_metrics(
                    mean.reshape(-1, horizon, 2),
                    log_std.reshape(-1, horizon, 2),
                    target.reshape(-1, horizon, 2),
                    batch.dt,
                )
                interaction = interaction_recovery_metrics(
                    factorial_second_difference(mean, dim=1),
                    factorial_second_difference(target, dim=1),
                )
                brier, ece, coverage_1, coverage_2, abs_z = _probabilistic_diagnostics(
                    mean,
                    log_std,
                    target,
                    groups,
                    entities,
                    horizon,
                )
                for local in range(groups):
                    # Keep scene-level rows as the independent statistical unit.
                    local_mean = mean[local].reshape(-1, horizon, 2)
                    local_log_std = log_std[local].reshape(-1, horizon, 2)
                    local_target = target[local].reshape(-1, horizon, 2)
                    local_metric = response_metrics(
                        local_mean, local_log_std, local_target, batch.dt
                    )
                    local_interaction = interaction_recovery_metrics(
                        factorial_second_difference(mean[local], dim=0),
                        factorial_second_difference(target[local], dim=0),
                    )
                    local_brier, local_ece, local_cov1, local_cov2, local_abs_z = _probabilistic_diagnostics(
                        mean[local : local + 1],
                        log_std[local : local + 1],
                        target[local : local + 1],
                        1,
                        entities,
                        horizon,
                    )
                    predicted_position = torch.cumsum(mean[local] * batch.dt, dim=2)
                    target_position = torch.cumsum(target[local] * batch.dt, dim=2)
                    action_effect_error = torch.linalg.vector_norm(
                        (predicted_position - predicted_position[0:1])
                        - (target_position - target_position[0:1]), dim=-1
                    ).mean()
                    display_method = METHOD_DISPLAY.get(method, method)
                    rows.append(
                        {
                            "train_family": train_family,
                            "target_family": target_family,
                            "data_seed": data_seed,
                            "base_scene_id": int(samples[local * intervention_count]["base_scene_id"]),
                            "method": display_method,
                            "nll": _finite_mean(local_metric["nll"]),
                            "ade": _finite_mean(local_metric["ade"]),
                            "fde": _finite_mean(local_metric["fde"]),
                            "interaction_rmse": local_interaction["rmse"],
                            "pair_rmse": local_interaction["rmse"],
                            "action_effect_error": float(action_effect_error),
                            "interaction_pearson": local_interaction["pearson"],
                            "interaction_r2": local_interaction["zero_baseline_r2"],
                            "velocity_sign_brier": local_brier,
                            "velocity_sign_ece": local_ece,
                            "brier": local_brier,
                            "ece": local_ece,
                            "coverage_1sigma": local_cov1,
                            "coverage_2sigma": local_cov2,
                            "abs_z": local_abs_z,
                        }
                    )
    return rows


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["target_family"]), str(row["method"])), []).append(row)
    metrics = (
        "nll",
        "ade",
        "fde",
        "interaction_rmse",
        "pair_rmse",
        "action_effect_error",
        "interaction_pearson",
        "interaction_r2",
        "velocity_sign_brier",
        "velocity_sign_ece",
        "brier",
        "ece",
        "coverage_1sigma",
        "coverage_2sigma",
        "abs_z",
    )
    result: list[dict[str, Any]] = []
    for (target, method), values in sorted(grouped.items()):
        item: dict[str, Any] = {
            "target_family": target,
            "method": method,
            "n_scenes": len(values),
        }
        for metric in metrics:
            array = np.asarray([float(row[metric]) for row in values], dtype=float)
            finite = array[np.isfinite(array)]
            item[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            item[f"{metric}_sd"] = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
        item["brier_mean"] = item["velocity_sign_brier_mean"]
        item["brier_sd"] = item["velocity_sign_brier_sd"]
        item["ece_mean"] = item["velocity_sign_ece_mean"]
        item["ece_sd"] = item["velocity_sign_ece_sd"]
        result.append(item)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _contrasts(rows: list[dict[str, Any]], bootstrap_resamples: int = 5000) -> list[dict[str, Any]]:
    """Compute paired base-scene contrasts without treating entities as repeats."""

    contrasts = (
        ("eiffert_minus_history_nll", "nll", "eiffert2019-style", "history-only"),
        ("eiffert_minus_joint_nll", "nll", "eiffert2019-style", "joint"),
        ("scept_minus_joint_nll", "nll", "ScePT-style", "joint"),
        ("matched_additive_minus_joint_nll", "nll", "matched-additive", "joint"),
        ("shuffled_minus_joint_nll", "nll", "shuffled-joint", "joint"),
        ("matched_additive_minus_joint_interaction_rmse", "interaction_rmse", "matched-additive", "joint"),
        ("shuffled_minus_joint_action_effect_error", "action_effect_error", "shuffled-joint", "joint"),
    )
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["target_family"]), int(row["base_scene_id"])), {})[
            str(row["method"])
        ] = row
    output: list[dict[str, Any]] = []
    for target_family in sorted({key[0] for key in grouped}):
        target_rows = {
            scene_id: methods
            for (family, scene_id), methods in grouped.items()
            if family == target_family
        }
        for index, (name, metric, first, second) in enumerate(contrasts):
            values = np.asarray(
                [
                    float(methods[first][metric]) - float(methods[second][metric])
                    for methods in target_rows.values()
                    if first in methods and second in methods
                ],
                dtype=float,
            )
            rng = np.random.default_rng(991 + index * 104729 + len(target_family))
            bootstrap = values[rng.integers(0, len(values), size=(bootstrap_resamples, len(values)))].mean(axis=1)
            output.append(
                {
                    "target_family": target_family,
                    "contrast": name,
                    "n_scenes": int(values.size),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                    "ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "ci95_high": float(np.quantile(bootstrap, 0.975)),
                }
            )
    return output


def _write_markdown(
    path: Path,
    protocol: dict[str, Any],
    summary: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
) -> None:
    lines = [
        "# Unified H1 Response Baselines and Unseen-Mechanism Transfer",
        "",
        "This is an evaluation-only supplementary audit. A frozen model trained on the `pincer` response family is evaluated on the paired held-out `pincer`, `speed`, `lateral`, and `threshold` families. The latter three are mechanism-transfer tests; their response generators and test scenes are not used to train the checkpoint.",
        "",
        f"Protocol: `{protocol['num_test_scenes']}` base scenes per target family, R=2, four paired interventions, model seed `{protocol['model_seed']}`, and data seeds `{', '.join(str(v) for v in protocol['target_data_seeds'].values())}`.",
        "",
        "`independent` is the exact `joint(include_pair_interactions=False)` reduction and is shown separately for the requested baseline taxonomy. `velocity_sign_brier`/`velocity_sign_ece` calibrate the fixed event `v_coordinate > 0` under each Gaussian prediction. Gaussian coverage is reported against the nominal 1-sigma (0.6827) and 2-sigma (0.9545) reference values.",
        "",
        "| Target family | Method | NLL | ADE | FDE | Action-effect | Pair RMSE | Sign Brier | Sign ECE | 1-sigma cov. | 2-sigma cov. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['target_family']} | {row['method']} | {row['nll_mean']:.4f} +/- {row['nll_sd']:.4f} | "
            f"{row['ade_mean']:.4f} | {row['fde_mean']:.4f} | {row['action_effect_error_mean']:.4f} | {row['pair_rmse_mean']:.4f} | "
            f"{row['velocity_sign_brier_mean']:.4f} | {row['velocity_sign_ece_mean']:.4f} | "
            f"{row['coverage_1sigma_mean']:.4f} | {row['coverage_2sigma_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
            "Differences are computed per held-out base scene before bootstrapping. Positive matched-additive-minus-joint values favor Joint; positive shuffled-minus-joint values indicate degradation after breaking candidate-response pairing.",
            "",
            "| Target family | Contrast | Mean | 95% CI | N |",
            "|---|---|---:|---|---:|",
        ]
    )
    for row in contrasts:
        lines.append(
            f"| {row['target_family']} | {row['contrast']} | {row['mean']:.5f} | "
            f"[{row['ci95_low']:.5f}, {row['ci95_high']:.5f}] | {row['n_scenes']} |"
        )
    lines.extend(
        [
            "",
            "Positive matched-additive-minus-joint NLL or pair-RMSE contrasts favor Joint. Cross-family rows are transfer diagnostics, not a replacement for the frozen Formal-v2 H1 gate. All rows and complete metrics are retained in `records.csv`; no test split or seed was selected after observing outcomes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    train_dir = root / args.checkpoint_dir
    train_config_path = train_dir / "config.yaml"
    train_config = _load_yaml(train_config_path)
    checkpoint = torch.load(train_dir / "joint.pt", map_location="cpu", weights_only=True)
    model_seed = int(checkpoint.get("seed", args.model_seed))
    train_family = str(train_config.get("benchmark", {}).get("response_family", args.train_family))
    include_kedia = "kedia-style" in train_config.get("training", {}).get("variants", [])
    models = _load_models(train_dir, args.device, include_kedia=include_kedia)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_families = tuple(args.target_families)
    target_seeds = {family: int(args.target_data_seed_start + index * 1000 + model_seed) for index, family in enumerate(target_families)}
    all_rows: list[dict[str, Any]] = []
    for family in target_families:
        raw = dict(train_config)
        raw.setdefault("dataset", {}).setdefault("causal_response", {})["response_family"] = family
        raw["dataset"]["causal_response"]["seed"] = target_seeds[family]
        dataset = make_causal_response_dataset_from_config(
            raw,
            split="test",
            effect_mode="nonlinear",
            seed=target_seeds[family],
            cache_samples=True,
        )
        rows = _evaluate_target_family(
            dataset,
            models,
            args.device,
            train_family,
            family,
            target_seeds[family],
            args.test_chunk,
            methods=METHODS + ((KEDIA_METHOD_NAME,) if include_kedia else ()),
        )
        all_rows.extend(rows)
        print(f"evaluated train={train_family} target={family} scenes={dataset.num_base_scenes}", flush=True)

    summary = _summary(all_rows)
    contrasts = _contrasts(all_rows)
    protocol = {
        "status": "completed",
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_data_seed": int(checkpoint.get("data_seed", -1)),
        "train_family": train_family,
        "target_families": list(target_families),
        "target_data_seeds": target_seeds,
        "model_seed": model_seed,
        "num_test_scenes": int(train_config.get("dataset", {}).get("causal_response", {}).get("test_base_scenes", 128)),
        "num_robots": 2,
        "interventions_per_scene": 4,
        "methods": list(METHODS),
        "method_taxonomy": METHOD_TAXONOMY,
        "model_metadata": {
            name: external_method_metadata(model, name)
            for name, model in models.items()
        },
        "independent_definition": "joint checkpoint evaluated with include_pair_interactions=False; exact strict-additive reduction",
        "velocity_sign_brier_definition": "Brier score for the fixed event v_coordinate > 0 under each scalar Gaussian prediction",
        "selection": "frozen before target-family evaluation; no test-driven model/seed selection",
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "records.csv", all_rows)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "contrasts.csv", contrasts)
    _write_markdown(output_dir / "summary.md", protocol, summary, contrasts)
    print(f"saved={output_dir} rows={len(all_rows)}", flush=True)
    return {"protocol": protocol, "summary": summary, "contrasts": contrasts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--checkpoint-dir",
        default="experiments/causal_benchmark_matrix_20260823_mechanisms/R2_radial_pincer_nonlinear/seed2027",
    )
    parser.add_argument("--output-dir", default="reports/h1_unified_baselines_20260904")
    parser.add_argument("--train-family", default="pincer")
    parser.add_argument("--target-families", nargs="+", default=["pincer", "speed", "lateral", "threshold"])
    parser.add_argument("--target-data-seed-start", type=int, default=910000)
    parser.add_argument("--model-seed", type=int, default=2027)
    parser.add_argument("--test-chunk", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
