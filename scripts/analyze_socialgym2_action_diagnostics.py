"""Reconstruct map-specific joint-action diagnostics from frozen source traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


POLICIES = (
    "tracer-persistent",
    "tracer-reset",
    "tracer-uniform",
    "tracer-no-pair",
)
SCENARIOS = ("envs/scenario/intersection", "envs/scenario/open")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_actions(source_record: dict) -> tuple[list[tuple], dict]:
    path = Path(source_record["path"])
    metadata_path = path / "metadata.json"
    manifest_path = path / "manifest.json"
    trace_path = path / "belief_trace.csv"
    if source_record.get("metadata_sha256") != sha256(metadata_path):
        raise ValueError(f"source metadata hash mismatch: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED":
        raise ValueError(f"source artifact is incomplete: {path}")
    if manifest.get("artifacts", {}).get("belief_trace.csv") != sha256(trace_path):
        raise ValueError(f"source trace hash mismatch: {path}")
    parameters = metadata.get("parameters", {})
    scenario = parameters.get("scenario")
    declared = parameters.get("policies", [])
    if scenario not in SCENARIOS or len(declared) != 1:
        raise ValueError(f"unexpected source role: {path}")
    policy = declared[0]
    if policy not in POLICIES:
        return [], {
            "path": str(path),
            "manifest_sha256": sha256(manifest_path),
            "trace_sha256": sha256(trace_path),
            "policy": policy,
            "scenario": scenario,
        }
    decisions = {}
    with trace_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] != policy or str(row["seed"]) != str(parameters.get("seed")):
                raise ValueError(f"source trace identity mismatch: {path}")
            action = tuple(json.loads(row["action_vector"]))
            if len(action) != 3 or any(value not in (0, 1) for value in action):
                raise ValueError(f"invalid joint action in {path}")
            key = (policy, str(row["seed"]), scenario, str(row["episode"]), str(row["step"]))
            previous = decisions.setdefault(key, action)
            if previous != action:
                raise ValueError(f"inconsistent per-track action in {path}")
    source = {
        "path": str(path),
        "manifest_sha256": sha256(manifest_path),
        "trace_sha256": sha256(trace_path),
        "policy": policy,
        "scenario": scenario,
    }
    return [(key, action) for key, action in decisions.items()], source


def analyze(aggregate: Path, output: Path, expected_pairs: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite action diagnostics: {output}")
    manifest_path = aggregate / "manifest.json"
    metadata_path = aggregate / "metadata.json"
    metrics_path = aggregate / "episode_metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED" or metadata.get("run_class") != "frozen_complete_lattice":
        raise ValueError("action diagnostics require a completed frozen lattice")
    for name in ("metadata.json", "episode_metrics.csv"):
        if manifest.get("artifacts", {}).get(name) != sha256(aggregate / name):
            raise ValueError(f"aggregate artifact hash mismatch: {name}")
    pairs = int(metadata.get("paired_keys", -1))
    if expected_pairs is not None and pairs != expected_pairs:
        raise ValueError(f"action diagnostics found {pairs} pairs, expected {expected_pairs}")

    decisions = {}
    sources = []
    for source_record in metadata.get("source_artifacts", []):
        rows, source = _source_actions(source_record)
        sources.append(source)
        for key, action in rows:
            if key in decisions:
                raise ValueError(f"duplicate decision key: {key}")
            decisions[key] = action
    results = {}
    for policy in POLICIES:
        results[policy] = {}
        for scenario in SCENARIOS:
            actions = [action for key, action in decisions.items() if key[0] == policy and key[2] == scenario]
            if not actions:
                raise ValueError(f"missing action trace for {policy} in {scenario}")
            halted = [sum(action) for action in actions]
            results[policy][scenario] = {
                "decision_windows": len(actions),
                "mean_halted_robots": sum(halted) / len(halted),
                "all_go_rate": sum(value == 0 for value in halted) / len(halted),
                "any_halt_rate": sum(value > 0 for value in halted) / len(halted),
                "all_halt_rate": sum(value == 3 for value in halted) / len(halted),
            }
    artifact = {
        "protocol": "socialgym2-map-action-diagnostics-v1",
        "status": "COMPLETED",
        "phase": "posthoc-mechanism-diagnostic-after-campaign-inspection",
        "primary_confirmatory_analysis": False,
        "aggregate": str(aggregate.resolve()),
        "aggregate_manifest_sha256": sha256(manifest_path),
        "aggregate_metadata_sha256": sha256(metadata_path),
        "episode_metrics_sha256": sha256(metrics_path),
        "pairs": pairs,
        "decision_key": ["policy", "seed", "scenario", "episode", "step"],
        "source_traces": sources,
        "results": results,
        "claim_boundary": (
            "posthoc action-frequency diagnostic reconstructed from hash-bound source traces; "
            "not a navigation effect estimate, causal mediation analysis, or confirmatory test"
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
