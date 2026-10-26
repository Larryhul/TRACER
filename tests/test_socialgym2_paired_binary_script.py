from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_socialgym2_paired_binary import analyze


FIELDS = (
    "policy",
    "seed",
    "scenario",
    "episode",
    "success",
    "team_success",
    "collision_free_team_success",
    "collision",
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


def _row(policy: str, episode: int, value: int) -> dict[str, str]:
    return {
        "policy": policy,
        "seed": "2309",
        "scenario": "intersection",
        "episode": str(episode),
        "success": str(value),
        "team_success": str(value),
        "collision_free_team_success": str(value),
        "collision": str(1 - value),
    }


def test_completed_lattice_is_analyzed_with_expected_pair_count(tmp_path: Path) -> None:
    rows = [
        _row(policy, episode, int(policy == "tracer-persistent"))
        for policy in ("tracer-persistent", "tracer-reset")
        for episode in range(2)
    ]
    aggregate = _aggregate(tmp_path, rows, pairs=2)

    result = analyze(aggregate, tmp_path / "result.json", expected_pairs=2)

    assert result["pairs"] == 2
    assert len(result["results"]) == 4


def test_duplicate_policy_key_is_rejected(tmp_path: Path) -> None:
    rows = [
        _row("tracer-persistent", 0, 1),
        _row("tracer-persistent", 0, 1),
        _row("tracer-reset", 0, 0),
    ]
    aggregate = _aggregate(tmp_path, rows, pairs=1)

    with pytest.raises(ValueError, match="duplicate paired key"):
        analyze(aggregate, tmp_path / "result.json")


def test_incomplete_expected_pair_count_is_rejected(tmp_path: Path) -> None:
    rows = [_row(policy, 0, 1) for policy in ("tracer-persistent", "tracer-reset")]
    aggregate = _aggregate(tmp_path, rows, pairs=1)

    with pytest.raises(ValueError, match="expected 320"):
        analyze(aggregate, tmp_path / "result.json", expected_pairs=320)

