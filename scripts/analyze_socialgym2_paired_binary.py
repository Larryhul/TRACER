"""Add population uncertainty to paired binary SocialGym2 outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sage.experiments.paired_binary_uncertainty import paired_binary_jeffreys


BINARY_METRICS = ("success", "team_success", "collision_free_team_success", "collision")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(aggregate: Path, output: Path, expected_pairs: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite paired binary analysis: {output}")
    manifest_path = aggregate / "manifest.json"
    metadata_path = aggregate / "metadata.json"
    metrics_path = aggregate / "episode_metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED" or metadata.get("run_class") != "frozen_complete_lattice":
        raise ValueError("paired binary analysis requires a completed frozen lattice")
    for name in ("metadata.json", "episode_metrics.csv"):
        if manifest.get("artifacts", {}).get(name) != sha256(aggregate / name):
            raise ValueError(f"aggregate artifact hash mismatch: {name}")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    policies = sorted({row["policy"] for row in rows})
    if "tracer-persistent" not in policies:
        raise ValueError("aggregate lacks the Persistent reference policy")
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
        raise ValueError("paired binary analysis requires identical policy keys")
    pairs = len(reference)
    for field in ("paired_keys", "independent_scene_clusters"):
        if int(metadata.get(field, -1)) != pairs:
            raise ValueError(f"aggregate metadata {field} does not match paired rows")
    if expected_pairs is not None and pairs != expected_pairs:
        raise ValueError(f"paired binary analysis found {pairs} pairs, expected {expected_pairs}")
    results = {}
    for policy_index, policy in enumerate(policies):
        if policy == "tracer-persistent":
            continue
        ordered = sorted(reference)
        for metric_index, metric in enumerate(BINARY_METRICS):
            results[f"{policy}-persistent:{metric}"] = paired_binary_jeffreys(
                [float(by_policy[policy][item][metric]) for item in ordered],
                [float(reference[item][metric]) for item in ordered],
                seed=974_231 + 31 * policy_index + metric_index,
            )
    artifact = {
        "protocol": "socialgym2-paired-binary-supplement-v1",
        "phase": "posthoc-supplement-after-partial-campaign-inspection",
        "primary_confirmatory_analysis": False,
        "aggregate": str(aggregate.resolve()),
        "aggregate_manifest_sha256": sha256(manifest_path),
        "episode_metrics_sha256": sha256(metrics_path),
        "pairs": pairs,
        "reference_policy": "tracer-persistent",
        "results": results,
        "claim_boundary": "supplementary Bayesian paired binary uncertainty; does not establish noninferiority",
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
    print(json.dumps({"status": "COMPLETED", "pairs": artifact["pairs"]}, indent=2))


if __name__ == "__main__":
    main()
