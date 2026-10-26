"""Attach hash-bound execution roles to a completed SocialGym2 aggregate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sage.experiments.socialgym2_policy_roles import EXTERNAL_POLICIES, socialgym2_policy_role


EXPECTED_POLICIES = (
    "tracer-persistent",
    "tracer-reset",
    "tracer-uniform",
    "tracer-no-pair",
    "official-goalone",
    "official-halt",
    *sorted(EXTERNAL_POLICIES),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annotate(aggregate: Path, output: Path, expected_pairs: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite policy-role annotation: {output}")
    manifest_path = aggregate / "manifest.json"
    metadata_path = aggregate / "metadata.json"
    metrics_path = aggregate / "episode_metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED" or metadata.get("run_class") != "frozen_complete_lattice":
        raise ValueError("policy-role annotation requires a completed frozen lattice")
    for name in ("metadata.json", "episode_metrics.csv"):
        if manifest.get("artifacts", {}).get(name) != sha256(aggregate / name):
            raise ValueError(f"aggregate artifact hash mismatch: {name}")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    policies = sorted({row["policy"] for row in rows})
    if policies != sorted(EXPECTED_POLICIES):
        raise ValueError(f"unexpected policy lattice: {policies}")
    pairs = int(metadata.get("paired_keys", -1))
    if pairs <= 0 or int(metadata.get("independent_scene_clusters", -1)) != pairs:
        raise ValueError("aggregate metadata has inconsistent paired scene counts")
    if expected_pairs is not None and pairs != expected_pairs:
        raise ValueError(f"policy-role annotation found {pairs} pairs, expected {expected_pairs}")
    source_artifacts = metadata.get("source_artifacts")
    if not isinstance(source_artifacts, list) or len(source_artifacts) != len(EXPECTED_POLICIES) * len(metadata.get("seeds", [])) * len(metadata.get("scenarios", [])):
        raise ValueError("aggregate source-artifact lattice is incomplete")
    counterfactual_enabled = True
    artifact = {
        "protocol": "socialgym2-policy-role-annotation-v1",
        "status": "COMPLETED",
        "aggregate": str(aggregate.resolve()),
        "aggregate_manifest_sha256": sha256(manifest_path),
        "aggregate_metadata_sha256": sha256(metadata_path),
        "episode_metrics_sha256": sha256(metrics_path),
        "paired_scenes": pairs,
        "source_artifacts": len(source_artifacts),
        "policy_roles": {
            policy: socialgym2_policy_role(policy, counterfactual_enabled)
            for policy in EXPECTED_POLICIES
        },
        "correction": (
            "The frozen per-run v5 metadata labelled official controls as pair-interaction policies. "
            "Execution bypasses response selection and posterior updates for official-* policies; "
            "their pair-interaction and belief fields are therefore not applicable."
        ),
        "claim_boundary": (
            "This hash-bound annotation corrects descriptive policy roles only. It does not alter "
            "trajectories, metrics, pairing, or the aggregate statistical analysis."
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
    artifact = annotate(args.aggregate, args.output, args.expected_pairs)
    print(json.dumps({"status": artifact["status"], "paired_scenes": artifact["paired_scenes"]}, indent=2))


if __name__ == "__main__":
    main()
