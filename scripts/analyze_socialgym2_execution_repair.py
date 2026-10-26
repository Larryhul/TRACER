#!/usr/bin/env python3
"""Compare old and repaired SocialGym2 systems on paired generated scenes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sage.experiments.aggregate_socialgym2_tracer import _cluster_bootstrap
from sage.experiments.paired_binary_uncertainty import paired_binary_jeffreys


BINARY_METRICS = (
    "collision_free_team_success",
    "team_success",
    "collision",
    "initial_collision",
    "timeout",
)
CONTINUOUS_METRICS = (
    "task_progress",
    "robot_success_fraction",
    "min_distance",
    "fallback_rate",
)
PAIRING_FIELDS = (
    "episode_seed",
    "initial_state_hash",
    "initial_scene_snapshot",
    "initial_collision",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "1.0", "true"}:
        return True
    if normalized in {"0", "0.0", "false"}:
        return False
    raise ValueError(f"invalid binary value: {value!r}")


def _load_run(path: Path, episodes: int) -> tuple[list[dict[str, str]], dict[str, str]]:
    manifest_path = path / "manifest.json"
    metadata_path = path / "metadata.json"
    metrics_path = path / "episode_metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED":
        raise ValueError(f"run is not complete: {path}")
    for name in ("metadata.json", "episode_metrics.csv"):
        expected = manifest.get("artifacts", {}).get(name)
        if expected != _sha256(path / name):
            raise ValueError(f"artifact hash mismatch in {path}: {name}")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows if int(row["episode"]) < episodes]
    if len(rows) != episodes:
        raise ValueError(f"expected {episodes} episode rows in {path}, found {len(rows)}")
    if {row["policy"] for row in rows} != {"tracer-persistent"}:
        raise ValueError(f"repair comparison requires only tracer-persistent: {path}")
    if {int(row["episode"]) for row in rows} != set(range(episodes)):
        raise ValueError(f"episode lattice is incomplete in {path}")
    return rows, metadata.get("algorithm_sources", {})


def _load_runs(
    paths: list[Path],
    episodes: int,
) -> tuple[dict[tuple[str, str, str], dict[str, str]], dict[str, str]]:
    keyed: dict[tuple[str, str, str], dict[str, str]] = {}
    source_identity = None
    for path in paths:
        rows, algorithm_sources = _load_run(path, episodes)
        if source_identity is None:
            source_identity = algorithm_sources
        elif source_identity != algorithm_sources:
            raise ValueError(f"algorithm source identity changed across runs: {path}")
        for row in rows:
            key = (row["seed"], row["scenario"], row["episode"])
            if key in keyed:
                raise ValueError(f"duplicate run row: {key}")
            keyed[key] = row
    return keyed, source_identity or {}


def _value(row: dict[str, str], metric: str) -> float:
    if metric == "timeout":
        return float(not _as_bool(row["team_success"]) and not _as_bool(row["terminal_reached"]))
    if metric in BINARY_METRICS:
        return float(_as_bool(row[metric]))
    return float(row[metric])


def _metric_summary(
    old_rows: dict[tuple[str, str, str], dict[str, str]],
    new_rows: dict[tuple[str, str, str], dict[str, str]],
    keys: list[tuple[str, str, str]],
    metric: str,
    seed: int,
) -> dict[str, Any]:
    old = np.asarray([_value(old_rows[key], metric) for key in keys], dtype=np.float64)
    new = np.asarray([_value(new_rows[key], metric) for key in keys], dtype=np.float64)
    differences = new - old
    by_seed_scenario: dict[str, list[float]] = {}
    for key, difference in zip(keys, differences):
        by_seed_scenario.setdefault(f"{key[0]}::{key[1]}", []).append(float(difference))
    result: dict[str, Any] = {
        "old_mean": float(old.mean()),
        "repaired_mean": float(new.mean()),
        "paired_difference_repaired_minus_old": float(differences.mean()),
        "pairs": len(keys),
        "seed_scenario_cluster_bootstrap": _cluster_bootstrap(by_seed_scenario, seed),
    }
    if metric in BINARY_METRICS:
        result["paired_binary_jeffreys"] = paired_binary_jeffreys(
            new,
            old,
            seed=seed + 20_000,
        )
    return result


def analyze(
    old_paths: list[Path],
    repaired_paths: list[Path],
    *,
    episodes: int,
    expected_seeds: list[int],
    expected_scenarios: list[str],
) -> dict[str, Any]:
    old_rows, old_sources = _load_runs(old_paths, episodes)
    repaired_rows, repaired_sources = _load_runs(repaired_paths, episodes)
    expected_keys = {
        (str(seed), scenario, str(episode))
        for seed in expected_seeds
        for scenario in expected_scenarios
        for episode in range(episodes)
    }
    if set(old_rows) != expected_keys or set(repaired_rows) != expected_keys:
        raise ValueError("old/repaired inputs do not form the expected paired lattice")

    raw_scenario_hash_mismatches = []
    for key in sorted(expected_keys):
        old = old_rows[key]
        repaired = repaired_rows[key]
        for field in PAIRING_FIELDS:
            if field == "initial_scene_snapshot":
                matched = json.loads(old[field]) == json.loads(repaired[field])
            else:
                matched = old[field] == repaired[field]
            if not matched:
                raise ValueError(f"paired scene mismatch for {key}: {field}")
        if old["scenario_hash"] != repaired["scenario_hash"]:
            raw_scenario_hash_mismatches.append(key)

    scenario_results = {}
    all_metrics = BINARY_METRICS + CONTINUOUS_METRICS
    for scenario_index, scenario in enumerate(expected_scenarios):
        keys = sorted(key for key in expected_keys if key[1] == scenario)
        scenario_results[scenario] = {
            metric: _metric_summary(
                old_rows,
                repaired_rows,
                keys,
                metric,
                2_610_000 + 100 * scenario_index + metric_index,
            )
            for metric_index, metric in enumerate(all_metrics)
        }

    pooled_keys = sorted(expected_keys)
    pooled = {
        metric: _metric_summary(
            old_rows,
            repaired_rows,
            pooled_keys,
            metric,
            2_620_000 + metric_index,
        )
        for metric_index, metric in enumerate(all_metrics)
    }
    initial_clean_keys = [
        key for key in pooled_keys if not _as_bool(old_rows[key]["initial_collision"])
    ]
    initial_clean = {
        metric: _metric_summary(
            old_rows,
            repaired_rows,
            initial_clean_keys,
            metric,
            2_630_000 + metric_index,
        )
        for metric_index, metric in enumerate(
            ("collision_free_team_success", "team_success", "collision", "timeout", "task_progress")
        )
    }
    gate = {
        "collision_free_team_success_improves": pooled["collision_free_team_success"][
            "paired_difference_repaired_minus_old"
        ] > 0.0,
        "team_success_not_lower": pooled["team_success"][
            "paired_difference_repaired_minus_old"
        ] >= 0.0,
        "task_progress_not_lower": pooled["task_progress"][
            "paired_difference_repaired_minus_old"
        ] >= 0.0,
    }
    map_gates = {}
    for scenario, metrics in scenario_results.items():
        map_gates[scenario] = {
            "team_success_not_lower": metrics["team_success"][
                "paired_difference_repaired_minus_old"
            ] >= 0.0,
            "task_progress_not_lower": metrics["task_progress"][
                "paired_difference_repaired_minus_old"
            ] >= 0.0,
            "collision_not_higher": metrics["collision"][
                "paired_difference_repaired_minus_old"
            ] <= 0.0,
        }
        map_gates[scenario]["accepted"] = all(map_gates[scenario].values())
    gate["map_gates"] = map_gates
    gate["accepted"] = all(gate.values()) and all(
        item["accepted"] for item in map_gates.values()
    )
    return {
        "protocol": "socialgym2-execution-repair-paired-comparison-v1",
        "status": "COMPLETED",
        "episodes_per_seed_scenario": episodes,
        "seeds": expected_seeds,
        "scenarios": expected_scenarios,
        "pairs": len(expected_keys),
        "pairing_audit": {
            "verified_fields": list(PAIRING_FIELDS),
            "raw_scenario_hash_mismatch_count": len(raw_scenario_hash_mismatches),
            "raw_scenario_hash_boundary": (
                "Generated scenario hashes include worker-specific absolute paths. Matching canonical "
                "episode seeds, full reset snapshots, reset-state hashes, and reset collision labels "
                "are required across systems; raw hashes are retained but not used across worker roots."
            ),
            "hidden_human_goal_boundary": (
                "Human goals are not exposed by reset observations; equality relies on the shared "
                "deterministic scenario/reset seed contract."
            ),
        },
        "algorithm_source_identity": {
            "old": old_sources,
            "repaired": repaired_sources,
        },
        "pooled": pooled,
        "scenarios": scenario_results,
        "secondary_initial_clean": {
            "pairs": len(initial_clean_keys),
            "metrics": initial_clean,
        },
        "acceptance_gate": gate,
        "claim_boundary": (
            "Paired simulator evidence. Bootstrap units are environment seed by map; Bayesian binary "
            "intervals are descriptive posterior uncertainty, not frequentist confidence intervals."
        ),
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    rows = [
        "# SocialGym2 execution repair comparison",
        "",
        f"Pairs: `{result['pairs']}`; acceptance: `{result['acceptance_gate']['accepted']}`.",
        "",
        "| Scope | Metric | Old | Repaired | Difference |",
        "|---|---:|---:|---:|---:|",
    ]
    scopes = [("pooled", result["pooled"]), *result["scenarios"].items()]
    for scope, metrics in scopes:
        for metric in ("collision_free_team_success", "team_success", "collision", "timeout", "task_progress"):
            item = metrics[metric]
            rows.append(
                f"| {scope} | {metric} | {item['old_mean']:.4f} | "
                f"{item['repaired_mean']:.4f} | {item['paired_difference_repaired_minus_old']:+.4f} |"
            )
    rows.extend([
        "",
        "Raw generated-scenario hashes are not compared across worker roots because generated files "
        "embed absolute paths. Canonical episode seeds, full reset snapshots, reset hashes, and reset "
        "collision labels are matched instead.",
        "",
    ])
    path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", action="append", type=Path, required=True)
    parser.add_argument("--repaired-run", action="append", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--expected-scenarios", nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_json, args.output_md):
        if output.exists():
            raise FileExistsError(output)
    result = analyze(
        args.old_run,
        args.repaired_run,
        episodes=args.episodes,
        expected_seeds=args.expected_seeds,
        expected_scenarios=args.expected_scenarios,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_markdown(result, args.output_md)
    print(json.dumps({"status": result["status"], "pairs": result["pairs"], "accepted": result["acceptance_gate"]["accepted"]}))


if __name__ == "__main__":
    main()
