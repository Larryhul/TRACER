"""Quantify map-specific effects and heterogeneity in a frozen SocialGym2 aggregate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from sage.experiments.aggregate_socialgym2_tracer import _bootstrap, _cluster_bootstrap
from sage.experiments.paired_binary_uncertainty import paired_binary_jeffreys


POLICIES = (
    "tracer-persistent",
    "tracer-reset",
    "tracer-uniform",
    "tracer-no-pair",
    "official-goalone",
    "official-halt",
)
OUTCOMES = (
    "collision_free_team_success",
    "team_success",
    "collision",
    "task_progress",
)
BINARY_OUTCOMES = set(OUTCOMES[:3])
SCENARIOS = ("envs/scenario/intersection", "envs/scenario/open")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(aggregate: Path, output: Path, expected_pairs: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite map-stratified analysis: {output}")
    manifest_path = aggregate / "manifest.json"
    metadata_path = aggregate / "metadata.json"
    metrics_path = aggregate / "episode_metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED" or metadata.get("run_class") != "frozen_complete_lattice":
        raise ValueError("map-stratified analysis requires a completed frozen lattice")
    for name in ("metadata.json", "episode_metrics.csv"):
        if manifest.get("artifacts", {}).get(name) != sha256(aggregate / name):
            raise ValueError(f"aggregate artifact hash mismatch: {name}")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    policies = sorted({row["policy"] for row in rows})
    scenarios = sorted({row["scenario"] for row in rows})
    if policies != sorted(POLICIES):
        raise ValueError(f"unexpected policy lattice: {policies}")
    if scenarios != sorted(SCENARIOS):
        raise ValueError(f"unexpected scenario lattice: {scenarios}")
    key = lambda row: (row["seed"], row["scenario"], row["episode"])
    by_policy = {}
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        keyed_rows = {key(row): row for row in policy_rows}
        if len(keyed_rows) != len(policy_rows):
            raise ValueError(f"duplicate paired key within policy: {policy}")
        by_policy[policy] = keyed_rows
    reference = by_policy["tracer-persistent"]
    if any(set(values) != set(reference) for values in by_policy.values()):
        raise ValueError("map-stratified analysis requires identical policy keys")
    pairs = len(reference)
    if pairs != int(metadata.get("paired_keys", -1)) or pairs != int(metadata.get("independent_scene_clusters", -1)):
        raise ValueError("aggregate metadata does not match paired rows")
    if expected_pairs is not None and pairs != expected_pairs:
        raise ValueError(f"map-stratified analysis found {pairs} pairs, expected {expected_pairs}")

    results = {}
    comparators = [policy for policy in policies if policy != "tracer-persistent"]
    for policy_index, policy in enumerate(comparators):
        for outcome_index, outcome in enumerate(OUTCOMES):
            name = f"{policy}-persistent:{outcome}"
            scenario_results = {}
            seed_scenario_means = {}
            for scenario_index, scenario in enumerate(SCENARIOS):
                scenario_keys = sorted(item for item in reference if item[1] == scenario)
                differences = [
                    float(by_policy[policy][item][outcome]) - float(reference[item][outcome])
                    for item in scenario_keys
                ]
                by_seed = {}
                for item, difference in zip(scenario_keys, differences):
                    by_seed.setdefault(item[0], []).append(difference)
                seed_scenario_means[scenario] = {
                    seed: float(np.mean(values))
                    for seed, values in sorted(by_seed.items())
                }
                scenario_result = {
                    "pairs": len(scenario_keys),
                    "scene_bootstrap": _bootstrap(
                        differences,
                        2_001_001 + policy_index * 101 + outcome_index * 11 + scenario_index,
                    ),
                    "seed_cluster_bootstrap": _cluster_bootstrap(
                        by_seed,
                        2_101_001 + policy_index * 101 + outcome_index * 11 + scenario_index,
                    ),
                }
                if outcome in BINARY_OUTCOMES:
                    scenario_result["paired_binary_jeffreys"] = paired_binary_jeffreys(
                        [float(by_policy[policy][item][outcome]) for item in scenario_keys],
                        [float(reference[item][outcome]) for item in scenario_keys],
                        seed=2_201_001 + policy_index * 101 + outcome_index * 11 + scenario_index,
                    )
                scenario_results[scenario] = scenario_result
            common_seeds = sorted(set(seed_scenario_means[SCENARIOS[0]]) & set(seed_scenario_means[SCENARIOS[1]]))
            heterogeneity = [
                seed_scenario_means[SCENARIOS[1]][seed] - seed_scenario_means[SCENARIOS[0]][seed]
                for seed in common_seeds
            ]
            results[name] = {
                "estimand": "comparator minus Persistent within map",
                "scenarios": scenario_results,
                "map_heterogeneity": {
                    "estimand": "open effect minus intersection effect, paired by environment seed",
                    "seed_paired_bootstrap": _bootstrap(
                        heterogeneity,
                        2_301_001 + policy_index * 101 + outcome_index * 11,
                    ),
                    "per_seed": dict(zip(common_seeds, heterogeneity)),
                },
            }
    artifact = {
        "protocol": "socialgym2-map-stratified-supplement-v1",
        "status": "COMPLETED",
        "phase": "posthoc-supplement-after-partial-campaign-inspection",
        "primary_confirmatory_analysis": False,
        "aggregate": str(aggregate.resolve()),
        "aggregate_manifest_sha256": sha256(manifest_path),
        "aggregate_metadata_sha256": sha256(metadata_path),
        "episode_metrics_sha256": sha256(metrics_path),
        "pairs": pairs,
        "scenarios": list(SCENARIOS),
        "reference_policy": "tracer-persistent",
        "results": results,
        "claim_boundary": (
            "posthoc map-stratified paired uncertainty and heterogeneity; pointwise intervals, "
            "not a preregistered confirmatory, equivalence, or noninferiority analysis"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-pairs", type=int)
    args = parser.parse_args()
    artifact = analyze(args.aggregate, args.output, args.expected_pairs)
    print(json.dumps({"status": artifact["status"], "pairs": artifact["pairs"]}, indent=2))


if __name__ == "__main__":
    main()
