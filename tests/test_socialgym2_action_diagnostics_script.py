from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_socialgym2_action_diagnostics import POLICIES, SCENARIOS, analyze


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> Path:
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    source_records = []
    for policy in POLICIES:
        for scenario in SCENARIOS:
            source = tmp_path / f"{policy}-{scenario.rsplit('/', 1)[-1]}"
            source.mkdir()
            metadata = source / "metadata.json"
            metadata.write_text(json.dumps({"parameters": {"seed": 7, "scenario": scenario, "policies": [policy]}}))
            trace = source / "belief_trace.csv"
            with trace.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("policy", "seed", "episode", "step", "track_id", "action_vector"))
                writer.writeheader()
                for track in range(2):
                    writer.writerow({"policy": policy, "seed": 7, "episode": 0, "step": 0, "track_id": track,
                                     "action_vector": json.dumps([0, 0, int(scenario.endswith("open"))])})
            source_manifest = source / "manifest.json"
            source_manifest.write_text(json.dumps({"status": "COMPLETED", "artifacts": {"belief_trace.csv": _sha256(trace)}}))
            source_records.append({"path": str(source), "metadata_sha256": _sha256(metadata)})
    metadata = aggregate / "metadata.json"
    metadata.write_text(json.dumps({"run_class": "frozen_complete_lattice", "paired_keys": 2,
                                    "source_artifacts": source_records}))
    metrics = aggregate / "episode_metrics.csv"
    metrics.write_text("policy,seed,scenario,episode\n")
    (aggregate / "manifest.json").write_text(json.dumps({"status": "COMPLETED", "artifacts": {
        "metadata.json": _sha256(metadata), "episode_metrics.csv": _sha256(metrics)}}))
    return aggregate


def test_action_diagnostics_deduplicate_tracks_and_preserve_maps(tmp_path: Path) -> None:
    aggregate = _fixture(tmp_path)

    result = analyze(aggregate, tmp_path / "result.json", expected_pairs=2)

    persistent = result["results"]["tracer-persistent"]
    assert persistent[SCENARIOS[0]]["decision_windows"] == 1
    assert persistent[SCENARIOS[0]]["mean_halted_robots"] == 0
    assert persistent[SCENARIOS[1]]["mean_halted_robots"] == 1
    assert persistent[SCENARIOS[1]]["all_go_rate"] == 0


def test_action_diagnostics_reject_tampered_trace(tmp_path: Path) -> None:
    aggregate = _fixture(tmp_path)
    metadata = json.loads((aggregate / "metadata.json").read_text(encoding="utf-8"))
    source = Path(metadata["source_artifacts"][0]["path"])
    with (source / "belief_trace.csv").open("a") as handle:
        handle.write("tampered\n")

    with pytest.raises(ValueError, match="trace hash mismatch"):
        analyze(aggregate, tmp_path / "result.json")
