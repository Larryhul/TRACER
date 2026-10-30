from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from tracer.experiments.formal_manifest import sha256_file


EXPECTED_SEEDS = (2027, 2028, 2029, 2030, 2031)
EXPECTED_MODES = ("none", "additive", "nonlinear")
EXPECTED_METHODS = (
    "additive",
    "matched-additive",
    "joint",
    "joint-without-pair",
    "shuffled-joint",
)
TRAINED_VARIANTS = ("additive", "matched-additive", "joint")
RAW_NLL_SESOI = 0.003
MECHANISM_LEAKAGE_RATIO = 0.05
STRUCTURAL_TOLERANCE = 1e-5
PARAMETER_MATCH_TOLERANCE = 0.01


def _read_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError("paired H1 records must not be empty")
    return records


def _root_mean_square(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    return math.sqrt(sum(value * value for value in finite) / len(finite))


def _mechanism_leakage(
    records: list[dict[str, str]],
    seeds: Iterable[int],
) -> dict[str, Any]:
    joint = [row for row in records if row["method"] == "joint"]
    nonlinear_target_rms = _root_mean_square(
        float(row["interaction_target_rms"])
        for row in joint
        if row["effect_mode"] == "nonlinear"
    )
    result: dict[str, Any] = {
        "reference_nonlinear_target_rms": nonlinear_target_rms,
        "threshold_ratio": MECHANISM_LEAKAGE_RATIO,
        "modes": {},
    }
    for mode in ("none", "additive"):
        mode_rows = [row for row in joint if row["effect_mode"] == mode]
        prediction_rms = _root_mean_square(
            float(row["interaction_prediction_rms"]) for row in mode_rows
        )
        ratio = (
            prediction_rms / nonlinear_target_rms
            if nonlinear_target_rms > 0.0
            else float("nan")
        )
        per_seed = {}
        for seed in seeds:
            seed_prediction_rms = _root_mean_square(
                float(row["interaction_prediction_rms"])
                for row in mode_rows
                if int(row["model_seed"]) == seed
            )
            per_seed[str(seed)] = {
                "prediction_rms": seed_prediction_rms,
                "ratio_to_nonlinear_target": (
                    seed_prediction_rms / nonlinear_target_rms
                    if nonlinear_target_rms > 0.0
                    else float("nan")
                ),
            }
        result["modes"][mode] = {
            "prediction_rms": prediction_rms,
            "ratio_to_nonlinear_target": ratio,
            "passes_pooled_5pct_rule": math.isfinite(ratio)
            and ratio <= MECHANISM_LEAKAGE_RATIO,
            "per_seed_supporting": per_seed,
        }
    return result


def _record_completeness(
    records: list[dict[str, str]],
    expected_base_scenes: int,
    expected_data_seed: int = 318731,
    expected_split: str = "pilot_test",
) -> dict[str, Any]:
    counts: Counter[tuple[str, int, str]] = Counter()
    scene_ids: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    duplicate_keys: Counter[tuple[str, int, int, str]] = Counter()
    nonfinite_required = 0
    observed_data_seeds: set[int] = set()
    observed_splits: set[str] = set()
    observed_seeds: set[int] = set()
    observed_modes: set[str] = set()
    observed_methods: set[str] = set()

    for row in records:
        mode = row["effect_mode"]
        seed = int(row["model_seed"])
        scene = int(row["base_scene_id"])
        method = row["method"]
        cell = (mode, seed, method)
        counts[cell] += 1
        scene_ids[cell].add(scene)
        duplicate_keys[(mode, seed, scene, method)] += 1
        observed_data_seeds.add(int(row["data_seed"]))
        observed_splits.add(row["split"])
        observed_seeds.add(seed)
        observed_modes.add(mode)
        observed_methods.add(method)
        for field in (
            "nll",
            "interaction_rmse",
            "interaction_target_rms",
            "interaction_prediction_rms",
        ):
            nonfinite_required += int(not math.isfinite(float(row[field])))

    expected_cells = {
        (mode, seed, method)
        for mode in EXPECTED_MODES
        for seed in EXPECTED_SEEDS
        for method in EXPECTED_METHODS
    }
    complete_cells = all(
        counts[cell] == expected_base_scenes
        and len(scene_ids[cell]) == expected_base_scenes
        for cell in expected_cells
    )
    reference_scene_ids = scene_ids[next(iter(sorted(expected_cells)))]
    identical_scene_ids = bool(reference_scene_ids) and all(
        scene_ids[cell] == reference_scene_ids for cell in expected_cells
    )
    duplicate_count = sum(count - 1 for count in duplicate_keys.values() if count > 1)
    expected_rows = (
        len(EXPECTED_MODES)
        * len(EXPECTED_SEEDS)
        * len(EXPECTED_METHODS)
        * expected_base_scenes
    )
    checks = {
        "expected_record_count": len(records) == expected_rows,
        "all_expected_cells_have_unique_base_scenes": complete_cells,
        "identical_base_scene_ids_across_all_cells": identical_scene_ids,
        "no_duplicate_method_scene_keys": duplicate_count == 0,
        "required_metrics_finite": nonfinite_required == 0,
        f"data_seed_is_{expected_data_seed}": observed_data_seeds
        == {expected_data_seed},
        f"split_is_{expected_split}": observed_splits == {expected_split},
        "expected_seeds_modes_methods_only": (
            observed_seeds == set(EXPECTED_SEEDS)
            and observed_modes == set(EXPECTED_MODES)
            and observed_methods == set(EXPECTED_METHODS)
        ),
    }
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "observed_rows": len(records),
        "expected_rows": expected_rows,
        "observed_model_seeds": sorted(observed_seeds),
        "missing_model_seeds": sorted(set(EXPECTED_SEEDS) - observed_seeds),
        "duplicate_keys": duplicate_count,
        "nonfinite_required_metrics": nonfinite_required,
    }


def _checkpoint_audit(checkpoint_root: Path) -> dict[str, Any]:
    expected = [
        checkpoint_root / mode / f"seed{seed}" / f"{variant}.pt"
        for mode in EXPECTED_MODES
        for seed in EXPECTED_SEEDS
        for variant in TRAINED_VARIANTS
    ]
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    hashes = {}
    metadata_errors: list[str] = []
    if not missing:
        for path in expected:
            relative = path.relative_to(checkpoint_root).as_posix()
            hashes[relative] = sha256_file(path)
            mode = path.parents[1].name
            seed = int(path.parent.name.removeprefix("seed"))
            variant = path.stem
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            except (
                OSError,
                RuntimeError,
                ValueError,
                EOFError,
                pickle.UnpicklingError,
            ) as error:
                metadata_errors.append(f"{relative}: unreadable checkpoint: {error}")
                continue
            expected_metadata = {
                "effect_mode": mode,
                "seed": seed,
                "data_seed": 318731,
                "variant": variant,
            }
            for name, expected_value in expected_metadata.items():
                if checkpoint.get(name) != expected_value:
                    metadata_errors.append(
                        f"{relative}: {name}={checkpoint.get(name)!r}, "
                        f"expected={expected_value!r}"
                    )
    aggregate_hash = None
    if hashes:
        digest = hashlib.sha256()
        for name, checksum in sorted(hashes.items()):
            digest.update(name.encode("utf-8"))
            digest.update(checksum.encode("ascii"))
        aggregate_hash = digest.hexdigest()
    return {
        "complete": not missing and not metadata_errors,
        "expected_checkpoints": len(expected),
        "missing_checkpoints": missing,
        "metadata_errors": metadata_errors,
        "aggregate_sha256": aggregate_hash,
        "sha256": hashes,
    }


def _summary_completeness(
    summary: dict[str, Any],
    expected_base_scenes: int,
    expected_status: str = "PILOT_ONLY_NOT_CONFIRMATORY",
    expected_data_seed: int = 318731,
    require_data_seed_mismatch: bool = False,
) -> dict[str, Any]:
    observed_seed_keys = set(summary.get("per_seed", {}))
    expected_seed_keys = {
        f"{mode}/{seed}" for mode in EXPECTED_MODES for seed in EXPECTED_SEEDS
    }
    pooled = summary.get("pooled_across_seeds", {})
    pooled_scene_counts = [
        int(contrast.get("n_base_scenes", -1))
        for mode in EXPECTED_MODES
        for contrast in pooled.get(mode, {}).values()
    ]
    per_seed_scene_counts = [
        int(contrast.get("n_base_scenes", -1))
        for cell in summary.get("per_seed", {}).values()
        for contrast in cell.values()
    ]
    checks = {
        "expected_status_declared": summary.get("status") == expected_status,
        "independent_unit_is_base_scene": summary.get("independent_unit") == "base_scene",
        f"data_seed_is_{expected_data_seed}": int(summary.get("data_seed", -1))
        == expected_data_seed,
        "data_seed_mismatch_flag_matches_profile": bool(
            summary.get("allow_data_seed_mismatch", False)
        )
        == require_data_seed_mismatch,
        "split_is_test": summary.get("split") == "test",
        "bootstrap_resamples_is_10000": int(summary.get("bootstrap_resamples", -1))
        == 10_000,
        "all_five_model_seeds_present": summary.get("model_seeds")
        == list(EXPECTED_SEEDS),
        "all_modes_present": summary.get("modes") == list(EXPECTED_MODES),
        "all_seed_mode_cells_present": observed_seed_keys == expected_seed_keys,
        "pooled_base_scene_count_matches_expected": bool(pooled_scene_counts)
        and all(count == expected_base_scenes for count in pooled_scene_counts),
        "per_seed_base_scene_count_matches_expected": bool(per_seed_scene_counts)
        and all(count == expected_base_scenes for count in per_seed_scene_counts),
    }
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "missing_seed_mode_cells": sorted(expected_seed_keys - observed_seed_keys),
    }


def _formal_manifest_audit(
    manifest_path: Path,
    summary_path: Path,
    records_path: Path,
    checkpoint_root: Path,
    checkpoint_audit: dict[str, Any],
) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {
            "complete": False,
            "checks": {"manifest_exists": False},
            "manifest": {"path": str(manifest_path), "sha256": None},
            "frozen_config": {"path": None, "sha256": None},
            "observed_checkpoint_artifacts": 0,
        }
    manifest_file = manifest_path.resolve()
    run_root = manifest_file.parent
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    frozen_config = run_root / "config_frozen.yaml"
    overrides = manifest.get("overrides", {})
    environment_path = run_root / "environment.json"
    environment = (
        json.loads(environment_path.read_text(encoding="utf-8"))
        if environment_path.is_file()
        else {}
    )

    expected_hashes = checkpoint_audit.get("sha256", {})
    artifact_hashes: dict[str, str] = {}
    artifact_records_valid = True
    for artifact in manifest.get("input_artifacts", []):
        try:
            relative = Path(str(artifact["path"]))
            suffix = "/".join(relative.parts[-3:])
            checksum = str(artifact["sha256"])
            exists = bool(artifact["exists"])
        except (KeyError, TypeError):
            artifact_records_valid = False
            continue
        if suffix in artifact_hashes:
            artifact_records_valid = False
        artifact_hashes[suffix] = checksum
        artifact_records_valid = artifact_records_valid and exists

    def inside_run(path: Path) -> bool:
        resolved = path.resolve()
        return resolved == run_root or run_root in resolved.parents

    checks = {
        "manifest_schema": manifest.get("schema") == "sage-formal-run/v1",
        "manifest_completed": manifest.get("status") == "COMPLETED",
        "manifest_stage_h1": manifest.get("stage") == "H1",
        "manifest_independent_unit_base_scene": manifest.get("independent_unit")
        == "base_scene",
        "manifest_seed_role_nuisance_block": manifest.get("model_seed_role")
        == "nuisance_block",
        "formal_inputs_inside_run_root": inside_run(summary_path)
        and inside_run(records_path),
        "frozen_config_hash_matches_manifest": frozen_config.is_file()
        and manifest.get("config", {}).get("sha256") == sha256_file(frozen_config),
        "rescue_gate_was_passed": manifest.get("gate", {}).get("content", {}).get(
            "status"
        )
        == "passed",
        "all_checkpoint_hashes_match_manifest": artifact_records_valid
        and artifact_hashes == expected_hashes,
        "formal_overrides_are_frozen": (
            overrides.get("device") == "cuda"
            and overrides.get("modes") == list(EXPECTED_MODES)
            and overrides.get("seeds") == list(EXPECTED_SEEDS)
            and int(overrides.get("data_seed", -1)) == 328731
            and overrides.get("split") == "test"
            and int(overrides.get("bootstrap_resamples", -1)) == 10_000
            and Path(str(overrides.get("checkpoint_root", ""))).resolve()
            == checkpoint_root.resolve()
            and Path(str(overrides.get("output_dir", ""))).resolve() == run_root
        ),
        "explicit_independent_corpus_flag_recorded": "--allow-data-seed-mismatch"
        in manifest.get("command", []),
        "cuda_environment_recorded": environment.get("torch", {}).get(
            "cuda_available"
        )
        is True,
    }
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "frozen_config": {
            "path": str(frozen_config),
            "sha256": sha256_file(frozen_config) if frozen_config.is_file() else None,
        },
        "observed_checkpoint_artifacts": len(artifact_hashes),
    }


def _contrast(summary: dict[str, Any], mode: str, name: str) -> dict[str, Any]:
    try:
        return summary["pooled_across_seeds"][mode][name]
    except KeyError:
        return {}


def _ci_inside(contrast: dict[str, Any], bound: float) -> bool:
    interval = contrast.get("mean_ci95", [float("nan"), float("nan")])
    return len(interval) == 2 and float(interval[0]) >= -bound and float(interval[1]) <= bound


def _ci_positive(contrast: dict[str, Any]) -> bool:
    interval = contrast.get("mean_ci95", [float("nan"), float("nan")])
    return len(interval) == 2 and float(interval[0]) > 0.0


def review_h1_rescue(
    summary_path: Path,
    records_path: Path,
    checkpoint_root: Path,
    expected_base_scenes: int = 256,
    review_kind: str = "rescue",
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if review_kind not in {"rescue", "formal"}:
        raise ValueError(f"unsupported H1 review kind: {review_kind}")
    formal = review_kind == "formal"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = _read_records(records_path)
    summary_audit = _summary_completeness(
        summary,
        expected_base_scenes,
        expected_status=(
            "FORMAL_CONFIRMATORY" if formal else "PILOT_ONLY_NOT_CONFIRMATORY"
        ),
        expected_data_seed=328731 if formal else 318731,
        require_data_seed_mismatch=formal,
    )
    record_audit = _record_completeness(
        records,
        expected_base_scenes,
        expected_data_seed=328731 if formal else 318731,
        expected_split="formal_test" if formal else "pilot_test",
    )
    checkpoint_audit = _checkpoint_audit(checkpoint_root)
    observed_seeds = record_audit["observed_model_seeds"]
    leakage = _mechanism_leakage(records, observed_seeds)

    none_nll = _contrast(summary, "none", "matched_additive_minus_joint_nll")
    additive_nll = _contrast(
        summary, "additive", "matched_additive_minus_joint_nll"
    )
    nonlinear_nll = _contrast(
        summary, "nonlinear", "matched_additive_minus_joint_nll"
    )
    nonlinear_interaction = _contrast(
        summary,
        "nonlinear",
        "matched_additive_minus_joint_interaction_rmse",
    )
    diagnostics = summary.get("diagnostic_maxima", {})
    structural_names = (
        "robot_permutation_max_abs_error",
        "pair_zero_condition_max_abs_error",
        "pair_one_robot_mask_max_abs_error",
        "matched_additive_second_difference_max_abs",
        "joint_without_pair_matches_additive_max_abs_error",
    )
    structural_values = [
        float(diagnostics.get(name, float("inf"))) for name in structural_names
    ]
    observed_direction_checks = []
    for seed in observed_seeds:
        cell = summary.get("per_seed", {}).get(f"nonlinear/{seed}", {})
        observed_direction_checks.extend(
            [
                float(
                    cell.get("matched_additive_minus_joint_nll", {}).get(
                        "mean", -math.inf
                    )
                )
                > 0.0,
                float(
                    cell.get(
                        "matched_additive_minus_joint_interaction_rmse", {}
                    ).get("mean", -math.inf)
                )
                > 0.0,
            ]
        )

    revised_checks = {
        "none_raw_nll_ci_within_0p003": _ci_inside(none_nll, RAW_NLL_SESOI),
        "additive_raw_nll_ci_within_0p003": _ci_inside(
            additive_nll, RAW_NLL_SESOI
        ),
        "none_mechanism_leakage_at_most_5pct": bool(
            leakage["modes"]["none"]["passes_pooled_5pct_rule"]
        ),
        "additive_mechanism_leakage_at_most_5pct": bool(
            leakage["modes"]["additive"]["passes_pooled_5pct_rule"]
        ),
        "nonlinear_joint_nll_ci_positive": _ci_positive(nonlinear_nll),
        "nonlinear_interaction_rmse_ci_positive": _ci_positive(
            nonlinear_interaction
        ),
        "parameter_match_within_1pct": float(
            diagnostics.get(
                "matched_vs_joint_trainable_relative_difference",
                diagnostics.get("matched_vs_joint_relative_difference", math.inf),
            )
        )
        < PARAMETER_MATCH_TOLERANCE,
        "structural_checks_below_1e_minus_5": bool(structural_values)
        and max(structural_values) < STRUCTURAL_TOLERANCE,
        "nonlinear_direction_positive_for_every_seed": (
            observed_seeds == list(EXPECTED_SEEDS)
            and bool(observed_direction_checks)
            and all(observed_direction_checks)
        ),
    }
    completeness = {
        "summary": summary_audit,
        "paired_records": record_audit,
        "checkpoints": checkpoint_audit,
    }
    if formal:
        completeness["formal_manifest"] = _formal_manifest_audit(
            manifest_path or summary_path.parent / "manifest.json",
            summary_path,
            records_path,
            checkpoint_root,
            checkpoint_audit,
        )
    inputs_complete = all(section["complete"] for section in completeness.values())
    revised_pass = all(revised_checks.values())
    if not inputs_complete:
        recommendation = (
            "INCOMPLETE_FORMAL_H1_INPUTS"
            if formal
            else "INCOMPLETE_AWAITING_ALL_FIVE_SEEDS"
        )
    elif revised_pass:
        recommendation = (
            "ELIGIBLE_FOR_H1_PASS_GATE_REVIEW"
            if formal
            else "ELIGIBLE_FOR_MANUAL_GATE_REVIEW"
        )
    else:
        recommendation = "DO_NOT_PASS_H1_GATE" if formal else "DO_NOT_PASS_RESCUE_GATE"

    return {
        "review_kind": review_kind,
        "review_status": recommendation,
        "evidence_boundary": (
            "Formal-v1 confirmatory H1 evidence on the frozen independent test corpus"
            if formal
            else "pilot-derived rescue replication; not Formal-v1 confirmatory evidence"
        ),
        "thresholds": {
            "raw_nll_sesoi": RAW_NLL_SESOI,
            "mechanism_leakage_ratio": MECHANISM_LEAKAGE_RATIO,
            "parameter_match_relative": PARAMETER_MATCH_TOLERANCE,
            "structural_absolute": STRUCTURAL_TOLERANCE,
        },
        "input_artifacts": {
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "paired_records": {
                "path": str(records_path),
                "sha256": sha256_file(records_path),
            },
            "checkpoint_root": str(checkpoint_root),
        },
        "completeness": completeness,
        "primary_contrasts": {
            "none_matched_additive_minus_joint_nll": none_nll,
            "additive_matched_additive_minus_joint_nll": additive_nll,
            "nonlinear_matched_additive_minus_joint_nll": nonlinear_nll,
            "nonlinear_matched_additive_minus_joint_interaction_rmse": nonlinear_interaction,
        },
        "mechanism_leakage": leakage,
        "original_dz_equivalence_diagnostic": {
            "passed": bool(
                summary.get("gates", {}).get(
                    "negative_controls_equivalent_dz_0p2", False
                )
            ),
            "required_by_revised_gate": False,
        },
        "revised_gate_checks": revised_checks,
        "revised_gate_pass": inputs_complete and revised_pass,
        "gate_created": False,
        "formal_h1_authorized": False if not formal else None,
        "h2_authorized": False,
        "required_next_action": (
            "Wait for complete inputs."
            if not inputs_complete
            else (
                "A human must inspect this report before creating h1_passed.json."
                if formal
                else "A human must inspect this report before creating any rescue gate."
            )
        ),
    }


def review_h1_formal(
    summary_path: Path,
    records_path: Path,
    checkpoint_root: Path,
    manifest_path: Path,
    expected_base_scenes: int = 256,
) -> dict[str, Any]:
    return review_h1_rescue(
        summary_path,
        records_path,
        checkpoint_root,
        expected_base_scenes=expected_base_scenes,
        review_kind="formal",
        manifest_path=manifest_path,
    )


def _format_check(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    formal = payload["review_kind"] == "formal"
    lines = [
        "# H1 Formal-v1 Pass Gate Review"
        if formal
        else "# H1 Rescue Manual Gate Review",
        "",
        f"- Recommendation: `{payload['review_status']}`",
        f"- Evidence boundary: {payload['evidence_boundary']}",
        "- Gate created: `NO`",
        "- H2 authorized: `NO`" if formal else "- Formal H1 authorized: `NO`",
        "",
        "## Completeness",
        "",
    ]
    for section_name, section in payload["completeness"].items():
        lines.append(
            f"- `{section_name}`: `{_format_check(bool(section['complete']))}`"
        )
    lines.extend(["", "## Revised Gate Checks", ""])
    for name, passed in payload["revised_gate_checks"].items():
        lines.append(f"- `{name}`: `{_format_check(bool(passed))}`")

    lines.extend(["", "## Negative-Control Raw NLL", ""])
    for mode in ("none", "additive"):
        contrast = payload["primary_contrasts"][
            f"{mode}_matched_additive_minus_joint_nll"
        ]
        interval = contrast.get("mean_ci95", [float("nan"), float("nan")])
        lines.append(
            f"- `{mode}`: mean `{float(contrast.get('mean', float('nan'))):.6f}`, "
            f"95% CI `[{float(interval[0]):.6f}, {float(interval[1]):.6f}]`"
        )

    lines.extend(["", "## Mechanism Leakage", ""])
    leakage = payload["mechanism_leakage"]
    lines.append(
        f"- Nonlinear target interaction RMS: "
        f"`{float(leakage['reference_nonlinear_target_rms']):.6f}`"
    )
    for mode in ("none", "additive"):
        values = leakage["modes"][mode]
        lines.append(
            f"- `{mode}` predicted interaction RMS: "
            f"`{float(values['prediction_rms']):.6f}`; ratio "
            f"`{float(values['ratio_to_nonlinear_target']):.4f}`"
        )

    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            f"{payload['required_next_action']} This tool never writes a gate file.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a fail-closed manual review for H1 rescue or Formal-v1."
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--paired-records", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--review-kind", choices=("rescue", "formal"), default="rescue"
    )
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    payload = review_h1_rescue(
        Path(args.summary),
        Path(args.paired_records),
        Path(args.checkpoint_root),
        review_kind=args.review_kind,
        manifest_path=Path(args.manifest) if args.manifest else None,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "review.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(output_dir / "review.md", payload)
    print(
        json.dumps(
            {
                "review_status": payload["review_status"],
                "gate_created": False,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
