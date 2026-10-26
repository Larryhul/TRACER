from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_socialgym2_map_stratified import POLICIES, SCENARIOS, analyze


FIELDS = (
    "policy",
    "seed",
    "scenario",
    "episode",
    "collision_free_team_success",
    "team_success",
    "collision",
    "task_progress",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate(tmp_path: Path, rows: list[dict[str, str]], pairs: int) -> Path:
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    metrics = aggregate / "episode_metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = aggregate / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "run_class": "frozen_complete_lattice",
                "paired_keys": pairs,
                "independent_scene_clusters": pairs,
            }
        ),
        encoding="utf-8",
    )
    (aggregate / "manifest.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "artifacts": {
                    "metadata.json": _sha256(metadata),
                    "episode_metrics.csv": _sha256(metrics),
                },
            }
        ),
        encoding="utf-8",
    )
    return aggregate


def _rows() -> list[dict[str, str]]:
    rows = []
    for policy in POLICIES:
        for scenario in SCENARIOS:
            for episode in range(2):
                persistent = policy == "tracer-persistent"
                better_in_intersection = persistent and scenario.endswith("intersection")
                rows.append(
                    {
                        "policy": policy,
                        "seed": str(2309 + episode),
                        "scenario": scenario,
                        "episode": "0",
                        "collision_free_team_success": str(int(better_in_intersection)),
                        "team_success": str(int(better_in_intersection)),
                        "collision": str(int(not better_in_intersection)),
                        "task_progress": str(1.0 if better_in_intersection else 0.0),
                    }
                )
    return rows


def test_complete_lattice_reports_map_effects_and_heterogeneity(tmp_path: Path) -> None:
    aggregate = _aggregate(tmp_path, _rows(), pairs=4)

    result = analyze(aggregate, tmp_path / "result.json", expected_pairs=4)

    item = result["results"]["tracer-reset-persistent:team_success"]
    assert item["scenarios"][SCENARIOS[0]]["scene_bootstrap"]["mean"] == -1.0
    assert item["scenarios"][SCENARIOS[1]]["scene_bootstrap"]["mean"] == 0.0
    assert item["map_heterogeneity"]["seed_paired_bootstrap"]["mean"] == 1.0
    assert item["scenarios"][SCENARIOS[0]]["paired_binary_jeffreys"]["pairs"] == 2


def test_missing_map_is_rejected(tmp_path: Path) -> None:
    rows = [row for row in _rows() if row["scenario"] == SCENARIOS[0]]
    aggregate = _aggregate(tmp_path, rows, pairs=2)

    with pytest.raises(ValueError, match="scenario lattice"):
        analyze(aggregate, tmp_path / "result.json")


def test_wrong_pair_count_is_rejected(tmp_path: Path) -> None:
    aggregate = _aggregate(tmp_path, _rows(), pairs=4)

    with pytest.raises(ValueError, match="expected 320"):
        analyze(aggregate, tmp_path / "result.json", expected_pairs=320)
