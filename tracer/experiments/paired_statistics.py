from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def validate_formal_episode_analysis(
    manifest_path: Path,
    expected_stage: str,
    input_paths: list[Path],
    bootstrap_samples: int,
    analysis_seed: int,
) -> dict[str, Any]:
    """Fail closed unless inputs belong to an active Formal-v1 episode run."""

    manifest_file = manifest_path.resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    checks = {
        "schema": manifest.get("schema") == "sage-formal-run/v1",
        "status": manifest.get("status") == "STARTED",
        "stage": manifest.get("stage") == expected_stage,
        "independent_unit": manifest.get("independent_unit") == "episode",
        "model_seed_role": manifest.get("model_seed_role") == "nuisance_block",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"formal {expected_stage} analysis manifest failed checks: {failed}"
        )

    run_root = manifest_file.parent
    frozen_config_path = run_root / "config_frozen.yaml"
    if not frozen_config_path.is_file():
        raise RuntimeError(
            f"formal analysis requires frozen config: {frozen_config_path}"
        )
    frozen_config = yaml.safe_load(
        frozen_config_path.read_text(encoding="utf-8")
    ) or {}
    analysis = frozen_config.get("analysis", {})
    parameter_checks = {
        "bootstrap_samples": int(analysis.get("bootstrap_samples", -1))
        == bootstrap_samples,
        "analysis_seed": int(analysis.get("seed", -1)) == analysis_seed,
    }
    failed_parameters = [
        name for name, passed in parameter_checks.items() if not passed
    ]
    if failed_parameters:
        raise RuntimeError(
            f"formal {expected_stage} analysis parameters differ from frozen config: "
            f"{failed_parameters}"
        )
    for input_path in input_paths:
        resolved = input_path.resolve()
        if resolved != run_root and run_root not in resolved.parents:
            raise RuntimeError(
                f"formal analysis input must stay inside run root {run_root}: {resolved}"
            )
        if not resolved.is_file():
            raise RuntimeError(f"formal analysis input is missing: {resolved}")
    return {"manifest": manifest, "config": frozen_config}


def validate_formal_episode_records(
    records: list[dict[str, Any]],
    frozen_config: dict[str, Any],
    expected_stage: str,
    observed_seeds: set[int],
) -> None:
    """Verify Formal-v1 contrasts use every frozen seed and episode."""

    expected_seeds = {int(seed) for seed in frozen_config.get("suite", {}).get("seeds", [])}
    if len(expected_seeds) < 5 or observed_seeds != expected_seeds:
        raise RuntimeError(
            "formal episode analysis requires the complete frozen five-seed set; "
            f"expected={sorted(expected_seeds)} observed={sorted(observed_seeds)}"
        )
    if expected_stage == "H2":
        dataset = frozen_config["dataset"]
        expected_units = {
            "id": int(dataset[f"{frozen_config['evaluation']['id_split']}_episodes"]),
            "ood": int(dataset["ood_episodes"]),
        }
    elif expected_stage == "H3":
        simulation = frozen_config["simulation"]
        expected_units = {
            str(split): int(simulation["episodes"])
            for split in simulation["splits"]
        }
    else:
        raise ValueError(f"unsupported formal episode stage: {expected_stage}")

    observed_splits = {str(record["split"]) for record in records}
    if observed_splits != set(expected_units):
        raise RuntimeError(
            f"formal {expected_stage} splits differ from frozen config; "
            f"expected={sorted(expected_units)} observed={sorted(observed_splits)}"
        )
    failures = []
    for record in records:
        split = str(record["split"])
        expected = expected_units[split]
        if int(record["independent_units"]) != expected:
            failures.append(
                f"{split}/{record['contrast']}/{record['metric']}: "
                f"units={record['independent_units']} expected={expected}"
            )
        if int(record["model_seed_blocks"]) != len(expected_seeds):
            failures.append(
                f"{split}/{record['contrast']}/{record['metric']}: "
                f"seed_blocks={record['model_seed_blocks']} expected={len(expected_seeds)}"
            )
        if int(record["paired_observations"]) != expected * len(expected_seeds):
            failures.append(
                f"{split}/{record['contrast']}/{record['metric']}: "
                f"observations={record['paired_observations']} "
                f"expected={expected * len(expected_seeds)}"
            )
    if failures:
        raise RuntimeError(
            "formal episode analysis dimensions failed: " + "; ".join(failures[:5])
        )


def blocked_paired_bootstrap(
    values_by_seed_and_unit: Mapping[int, Mapping[int, float]],
    rng: np.random.Generator,
    samples: int,
) -> dict[str, object]:
    """Bootstrap paired units after averaging over nuisance model-seed blocks."""

    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not values_by_seed_and_unit:
        raise ValueError("paired analysis requires at least one model-seed block")

    seed_ids = sorted(int(seed) for seed in values_by_seed_and_unit)
    reference_units: tuple[int, ...] | None = None
    columns: list[np.ndarray] = []
    seed_block_means: dict[str, float] = {}
    for seed in seed_ids:
        values = values_by_seed_and_unit[seed]
        unit_ids = tuple(sorted(int(unit) for unit in values))
        if reference_units is None:
            reference_units = unit_ids
        elif unit_ids != reference_units:
            missing = sorted(set(reference_units) - set(unit_ids))
            extra = sorted(set(unit_ids) - set(reference_units))
            raise ValueError(
                "model-seed blocks must contain identical paired units; "
                f"seed={seed} missing={missing[:5]} extra={extra[:5]}"
            )
        column = np.asarray([values[unit] for unit in unit_ids], dtype=np.float64)
        if not np.isfinite(column).all():
            raise ValueError(f"paired values must be finite; seed={seed}")
        columns.append(column)
        seed_block_means[str(seed)] = float(column.mean())

    assert reference_units is not None
    if len(reference_units) < 2:
        raise ValueError("paired analysis requires at least two independent units")

    matrix = np.column_stack(columns)
    unit_effects = matrix.mean(axis=1)
    bootstrap_indices = rng.integers(
        0,
        len(reference_units),
        size=(samples, len(reference_units)),
    )
    bootstrap_means = unit_effects[bootstrap_indices].mean(axis=1)
    standard_deviation = float(unit_effects.std(ddof=1))
    mean = float(unit_effects.mean())
    return {
        "model_seed_blocks": len(seed_ids),
        "seeds": len(seed_ids),
        "independent_units": len(reference_units),
        "episodes": len(reference_units),
        "paired_observations": int(matrix.size),
        "mean": mean,
        "median": float(np.median(unit_effects)),
        "standard_deviation": standard_deviation,
        "paired_dz": (
            mean / standard_deviation
            if standard_deviation > 0.0
            else float("nan")
        ),
        "bootstrap_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "bootstrap_probability_positive": float((bootstrap_means > 0.0).mean()),
        "seed_block_means": seed_block_means,
    }
