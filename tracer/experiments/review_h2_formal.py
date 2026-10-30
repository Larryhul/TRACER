from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (2027, 2028, 2029, 2030, 2031)
METHODS = ("no-belief", "reset", "persistent", "shuffled-id", "recurrent", "oracle")
SPLITS = ("id", "ood")
PRIMARY_CONTRASTS = (
    "reset-minus-persistent",
    "shuffled-id-minus-persistent",
    "recurrent-minus-persistent",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record_completeness(raw_paths: list[Path], episodes: int) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    observed_seeds: list[int] = []
    rows_by_seed: dict[str, int] = {}
    duplicate_keys = 0
    nonfinite = 0
    expected_rows_per_seed = len(SPLITS) * episodes * 10 * 8 * len(METHODS)
    for path in raw_paths:
        seed_name = path.parent.name
        if not seed_name.startswith("seed"):
            raise ValueError(f"raw path must live in seed directory: {path}")
        seed = int(seed_name.removeprefix("seed"))
        observed_seeds.append(seed)
        seen: set[tuple[str, int, int, int, str]] = set()
        row_count = 0
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row_count += 1
                key = (
                    row["split"],
                    int(row["episode_id"]),
                    int(row["entity_slot"]),
                    int(row["window"]),
                    row["method"],
                )
                duplicate_keys += int(key in seen)
                seen.add(key)
                for field in (
                    "predictive_trajectory_nll",
                    "trajectory_ade",
                    "trajectory_fde",
                ):
                    nonfinite += int(not math.isfinite(float(row[field])))
        rows_by_seed[str(seed)] = row_count
        checks[f"seed{seed}_has_expected_rows"] = row_count == expected_rows_per_seed
        checks[f"seed{seed}_has_unique_rows"] = len(seen) == row_count
    checks["five_expected_seeds"] = sorted(observed_seeds) == list(EXPECTED_SEEDS)
    checks["all_required_metrics_finite"] = nonfinite == 0
    checks["no_duplicate_episode_entity_window_methods"] = duplicate_keys == 0
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "expected_rows_per_seed": expected_rows_per_seed,
        "rows_by_seed": rows_by_seed,
        "observed_seeds": sorted(observed_seeds),
        "nonfinite_required_values": nonfinite,
        "duplicate_keys": duplicate_keys,
    }


def _primary_records(analysis: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for record in analysis.get("contrasts", []):
        if (
            record.get("scope") == "all-post-initial"
            and record.get("metric") == "predictive_trajectory_nll"
            and record.get("contrast") in PRIMARY_CONTRASTS
        ):
            records.setdefault(str(record["split"]), {})[str(record["contrast"])] = record
    return records


def review_formal_h2(
    analysis_path: Path,
    raw_paths: list[Path],
    manifest_path: Path,
    episodes: int = 256,
    required_contrasts: tuple[str, ...] = PRIMARY_CONTRASTS,
) -> dict[str, Any]:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completeness = _record_completeness(raw_paths, episodes)
    records = _primary_records(analysis)
    primary_checks: dict[str, bool] = {}
    primary_values: dict[str, Any] = {}
    unknown = set(required_contrasts) - set(PRIMARY_CONTRASTS)
    if unknown:
        raise ValueError(f"unknown H2 gate contrasts: {sorted(unknown)}")
    for split in SPLITS:
        for contrast in required_contrasts:
            key = f"{split}_{contrast}_nll_ci_positive"
            record = records.get(split, {}).get(contrast)
            low = float(record["bootstrap_ci95_low"]) if record else float("nan")
            primary_values[f"{split}/{contrast}"] = record
            primary_checks[key] = record is not None and math.isfinite(low) and low > 0.0

    structural_checks = {
        "formal_confirmatory_status": analysis.get("analysis_status") == "FORMAL_CONFIRMATORY",
        "manifest_is_formal_run": manifest.get("status") in {"STARTED", "COMPLETED"},
        "window0_identical": float(analysis.get("window0_sanity", {}).get("max_episode_sum_nll_difference", math.inf)) <= 1e-6,
        "all_required_primary_records_present": all(primary_checks.values()) or all(
            records.get(split, {}).get(contrast) is not None
            for split in SPLITS
            for contrast in required_contrasts
        ),
    }
    checks = {**completeness["checks"], **structural_checks, **primary_checks}
    passed = completeness["complete"] and all(structural_checks.values()) and all(primary_checks.values())
    return {
        "review_kind": "formal",
        "review_status": "ELIGIBLE_FOR_H2_PASS_GATE_REVIEW" if passed else "DO_NOT_PASS_H2_GATE",
        "evidence_boundary": "Formal-v1 confirmatory H2 episode-level test/OOD evidence",
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "analysis": {"path": str(analysis_path), "sha256": _sha256(analysis_path)},
        "raw_inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in raw_paths
        ],
        "completeness": completeness,
        "structural_checks": structural_checks,
        "primary_checks": primary_checks,
        "required_primary_contrasts": list(required_contrasts),
        "primary_records": primary_values,
        "gate_pass": passed,
        "gate_created": False,
        "required_next_action": (
            "Inspect this review and create h2_passed.json only if the recorded checks are accepted."
            if passed
            else "Do not create h2_passed.json; diagnose the failed formal checks."
        ),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# H2 Formal-v1 Pass Gate Review",
        "",
        f"- Recommendation: `{payload['review_status']}`",
        f"- Evidence boundary: {payload['evidence_boundary']}",
        "- Gate created: `NO`",
        "- Required primary contrasts: "
        + ", ".join(f"`{name}`" for name in payload["required_primary_contrasts"]),
        "",
        "## Structural Checks",
        "",
    ]
    for name, value in {**payload["completeness"]["checks"], **payload["structural_checks"]}.items():
        lines.append(f"- `{name}`: `{'PASS' if value else 'FAIL'}`")
    lines.extend(["", "## Primary Predictive NLL Checks", ""])
    for name, value in payload["primary_checks"].items():
        lines.append(f"- `{name}`: `{'PASS' if value else 'FAIL'}`")
    lines.extend(["", "## Decision Boundary", "", payload["required_next_action"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a completed Formal-v1 H2 suite without writing a gate.")
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--raw", nargs="+", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument(
        "--required-contrasts",
        nargs="+",
        choices=PRIMARY_CONTRASTS,
        default=None,
        help=(
            "Contrasts required for the gate. Omit to require all reported "
            "references; use reset and shuffled-id when recurrent is an upper-bound reference."
        ),
    )
    args = parser.parse_args()
    payload = review_formal_h2(
        Path(args.analysis),
        [Path(path) for path in args.raw],
        Path(args.manifest),
        episodes=args.episodes,
        required_contrasts=tuple(args.required_contrasts or PRIMARY_CONTRASTS),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "review.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(output_dir / "review.md", payload)
    print(json.dumps({"review_status": payload["review_status"], "gate_created": False}, sort_keys=True))
    raise SystemExit(0 if payload["gate_pass"] else 2)


if __name__ == "__main__":
    main()
