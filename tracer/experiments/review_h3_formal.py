from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (2027, 2028, 2029, 2030, 2031)
SPLITS = ("id", "ood")
POLICIES = (
    "task-only",
    "no-belief-joint",
    "reset-additive",
    "reset-joint",
    "persistent-additive",
    "persistent-joint",
    "oracle-joint",
)
EPISODE_METRICS = (
    "realized_total_cost",
    "task_progress",
    "safety_cost",
    "human_cost",
    "min_distance",
    "collision_rate",
    "mean_activity",
    "oracle_regret",
    "oracle_hit_rate",
    "candidate_rank_spearman",
    "candidate_rank_kendall",
    "task_feasible_candidate_count",
    "task_feasible_candidate_fraction",
    "candidate_true_cost_range",
    "final_belief_entropy",
    "final_mode_accuracy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_completeness(
    path: Path, episodes: int, policies: tuple[str, ...]
) -> dict[str, Any]:
    expected_keys = {
        (seed, split, policy, episode)
        for seed in EXPECTED_SEEDS
        for split in SPLITS
        for policy in policies
        for episode in range(episodes)
    }
    observed: set[tuple[int, str, str, int]] = set()
    duplicate_keys = 0
    nonfinite = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seed"]), row["split"], row["policy"], int(row["episode"]))
            duplicate_keys += int(key in observed)
            observed.add(key)
            for metric in EPISODE_METRICS:
                nonfinite += int(not math.isfinite(float(row[metric])))
    checks = {
        "expected_episode_policy_rows": len(observed) == len(expected_keys),
        "all_expected_episode_policy_keys": observed == expected_keys,
        "no_duplicate_episode_policy_keys": duplicate_keys == 0,
        "all_required_episode_metrics_finite": nonfinite == 0,
    }
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "observed_rows": len(observed),
        "expected_rows": len(expected_keys),
        "duplicate_keys": duplicate_keys,
        "nonfinite_required_values": nonfinite,
    }


def _runs_completeness(
    path: Path,
    episodes: int,
    windows: int,
    candidates: int,
    policies: tuple[str, ...],
) -> dict[str, Any]:
    runs = json.loads(path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "five_expected_seed_blocks": sorted(int(seed) for seed in runs) == list(EXPECTED_SEEDS),
    }
    malformed_records = 0
    nonfinite_values = 0
    invalid_rejection_values = 0
    nonfinite_selected_values = 0
    for seed in EXPECTED_SEEDS:
        seed_run = runs.get(str(seed), {})
        checks[f"seed{seed}_has_expected_splits"] = set(seed_run) == set(SPLITS)
        for split in SPLITS:
            split_run = seed_run.get(split, {})
            checks[f"seed{seed}_{split}_has_expected_policies"] = set(split_run) == set(policies)
            for policy in policies:
                result = split_run.get(policy, {})
                episode_records = result.get("episode_records", [])
                selected = result.get("selected_candidates", [])
                window_records = result.get("window_records", [])
                key = f"seed{seed}_{split}_{policy}"
                checks[f"{key}_has_episode_records"] = len(episode_records) == episodes
                checks[f"{key}_has_window_records"] = len(window_records) == episodes
                checks[f"{key}_has_selected_records"] = len(selected) == episodes
                for episode in range(episodes):
                    if episode >= len(episode_records) or episode >= len(window_records) or episode >= len(selected):
                        malformed_records += 1
                        continue
                    windows_for_episode = window_records[episode]
                    selected_for_episode = selected[episode]
                    if len(windows_for_episode) != windows or len(selected_for_episode) != windows:
                        malformed_records += 1
                        continue
                    for window, record in enumerate(windows_for_episode):
                        if not isinstance(record, dict):
                            malformed_records += 1
                            continue
                        if record.get("window") != window:
                            malformed_records += 1
                        if not isinstance(record.get("candidate_hash"), str) or len(record["candidate_hash"]) != 64:
                            malformed_records += 1
                        if not isinstance(record.get("candidate_seed"), int):
                            malformed_records += 1
                        if not isinstance(record.get("candidate_generation_key"), str):
                            malformed_records += 1
                        if not 0 <= int(record.get("selected_candidate", -1)) < candidates:
                            malformed_records += 1
                        predicted = record.get("predicted_candidate_utility", [])
                        truth = record.get("ground_truth_candidate_utility", [])
                        if len(predicted) != candidates or len(truth) != candidates:
                            malformed_records += 1
                        # Rejected candidates are deliberately represented by -inf
                        # after the probabilistic safety filter. They are valid
                        # structural records, but must never be selected.
                        for value in predicted:
                            numeric = float(value)
                            if not math.isfinite(numeric) and numeric != -math.inf:
                                invalid_rejection_values += 1
                            if numeric == math.inf:
                                nonfinite_values += 1
                        nonfinite_values += sum(
                            not math.isfinite(float(value)) for value in truth
                        )
                        selected_candidate = int(record.get("selected_candidate", -1))
                        if 0 <= selected_candidate < len(predicted):
                            selected_is_finite = math.isfinite(
                                float(predicted[selected_candidate])
                            )
                            selected_is_explicit_fallback = bool(
                                record.get("safety_fallback", 0)
                            ) and float(record.get("selected_violation_probability", float("nan"))) >= 0.0
                            nonfinite_selected_values += int(
                                not selected_is_finite and not selected_is_explicit_fallback
                            )
    checks["all_window_candidate_records_structurally_valid"] = malformed_records == 0
    checks["all_ground_truth_candidate_values_finite"] = nonfinite_values == 0
    checks["rejected_candidate_values_are_negative_infinity"] = invalid_rejection_values == 0
    checks["selected_candidate_values_finite"] = nonfinite_selected_values == 0
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "malformed_records": malformed_records,
        "nonfinite_candidate_values": nonfinite_values,
        "invalid_rejection_values": invalid_rejection_values,
        "nonfinite_selected_values": nonfinite_selected_values,
    }


def review_formal_h3(
    analysis_path: Path,
    episode_metrics_path: Path,
    runs_path: Path,
    manifest_path: Path,
    candidate_gate_path: Path,
    episodes: int = 256,
    windows: int = 8,
    candidates: int = 16,
) -> dict[str, Any]:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_gate = json.loads(candidate_gate_path.read_text(encoding="utf-8"))
    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    has_recurrent = any(
        "recurrent-joint" in split_run
        for seed_run in runs.values()
        for split_run in seed_run.values()
    )
    policies = POLICIES + (("recurrent-joint",) if has_recurrent else ())
    episode_check = _episode_completeness(episode_metrics_path, episodes, policies)
    runs_check = _runs_completeness(runs_path, episodes, windows, candidates, policies)
    structural_checks = {
        "formal_confirmatory_analysis": analysis.get("analysis_status") == "FORMAL_CONFIRMATORY",
        "manifest_completed": manifest.get("status") == "COMPLETED",
        "manifest_independent_unit_episode": manifest.get("independent_unit") == "episode",
        "manifest_seed_role_nuisance_block": manifest.get("model_seed_role") == "nuisance_block",
        "analysis_uses_frozen_bootstrap": analysis.get("bootstrap_samples") == 10_000
        and analysis.get("seed") == 20270810,
        "candidate_gate_recorded": candidate_gate.get("status") == "evaluated",
        "optional_recurrent_reference_complete": (
            not has_recurrent
            or all(
                "recurrent-joint" in seed_run.get(split, {})
                for seed_run in runs.values()
                for split in SPLITS
            )
        ),
    }
    analysis_gate = analysis.get("gate", {})
    candidate_gate_passed = bool(candidate_gate.get("passed"))
    task_gate_passed = analysis_gate.get("status") == "pass"
    checks = {
        **episode_check["checks"],
        **runs_check["checks"],
        **structural_checks,
    }
    complete = episode_check["complete"] and runs_check["complete"] and all(structural_checks.values())
    if not candidate_gate_passed and not task_gate_passed:
        claim_decision = "BLOCKED_BY_SAGE_CANDIDATE_GATE_AND_TASK_GATE"
    elif not candidate_gate_passed:
        claim_decision = "BLOCKED_BY_SAGE_CANDIDATE_GATE"
    elif not task_gate_passed:
        claim_decision = "BLOCKED_BY_TASK_GATE"
    else:
        claim_decision = "ELIGIBLE_FOR_HUMAN_CLAIM_REVIEW"
    return {
        "review_kind": "formal_h3",
        "review_status": (
            "ELIGIBLE_FOR_H3_DESCRIPTIVE_REVIEW"
            if complete
            else "DO_NOT_ACCEPT_H3_RUN_AS_COMPLETE"
        ),
        "evidence_boundary": (
            "Formal-v1 complete closed-loop H3 run; descriptive system evidence only "
            "when the separate SAGE candidate/task gate fails"
        ),
        "claim_gate": {
            "candidate_gate_passed": candidate_gate_passed,
            "task_gate_passed": task_gate_passed,
            "full_closed_loop_claim_passed": complete and candidate_gate_passed and task_gate_passed,
            "decision": claim_decision,
        },
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "analysis": {"path": str(analysis_path), "sha256": _sha256(analysis_path)},
        "episode_metrics": {
            "path": str(episode_metrics_path),
            "sha256": _sha256(episode_metrics_path),
        },
        "runs": {"path": str(runs_path), "sha256": _sha256(runs_path)},
        "candidate_gate": {"path": str(candidate_gate_path), "sha256": _sha256(candidate_gate_path)},
        "episode_completeness": episode_check,
        "runs_completeness": runs_check,
        "structural_checks": structural_checks,
        "checks": checks,
        "gate_created": False,
        "required_next_action": (
            "Report H3 as descriptive only; do not create h3_passed.json. "
            "The SAGE candidate/task gate must be repaired in a separately declared run before a full closed-loop claim."
            if complete and (not candidate_gate_passed or not task_gate_passed)
            else "Inspect the complete formal evidence and perform the human claim review."
        ),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# H3 Formal-v1 Review",
        "",
        f"- Review: `{payload['review_status']}`",
        f"- Candidate/task claim: `{payload['claim_gate']['decision']}`",
        f"- Gate created: `NO`",
        "",
        "## Structural Checks",
        "",
    ]
    for name, value in payload["checks"].items():
        lines.append(f"- `{name}`: `{'PASS' if value else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            payload["required_next_action"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed review of a completed Formal-v1 H3 suite.")
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--episode-metrics", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=16)
    args = parser.parse_args()
    payload = review_formal_h3(
        Path(args.analysis),
        Path(args.episode_metrics),
        Path(args.runs),
        Path(args.manifest),
        Path(args.candidate_gate),
        episodes=args.episodes,
        windows=args.windows,
        candidates=args.candidates,
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
                "claim_gate": payload["claim_gate"],
                "gate_created": False,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if payload["review_status"] == "ELIGIBLE_FOR_H3_DESCRIPTIVE_REVIEW" else 2)


if __name__ == "__main__":
    main()
