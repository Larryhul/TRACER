from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.annotate_socialgym2_policy_roles import EXPECTED_POLICIES, annotate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate(tmp_path: Path, *, pairs: int = 2) -> Path:
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    metrics = aggregate / "episode_metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("policy", "seed", "scenario", "episode"))
        writer.writeheader()
        for policy in EXPECTED_POLICIES:
            for episode in range(pairs):
                writer.writerow({"policy": policy, "seed": 2309, "scenario": "intersection", "episode": episode})
    metadata = aggregate / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "run_class": "frozen_complete_lattice",
                "paired_keys": pairs,
                "independent_scene_clusters": pairs,
                "seeds": ["2309"],
                "scenarios": ["intersection"],
                "source_artifacts": [{"path": policy} for policy in EXPECTED_POLICIES],
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


def test_annotation_distinguishes_component_and_external_policy_roles(tmp_path: Path) -> None:
    aggregate = _aggregate(tmp_path)

    result = annotate(aggregate, tmp_path / "roles.json", expected_pairs=2)

    persistent = result["policy_roles"]["tracer-persistent"]
    no_pair = result["policy_roles"]["tracer-no-pair"]
    official = result["policy_roles"]["official-goalone"]
    assert persistent["response_planner_evaluated"] is True
    assert persistent["posterior_updated"] is True
    assert persistent["pair_interactions"] is True
    assert no_pair["pair_interactions"] is False
    assert official["response_planner_evaluated"] is False
    assert official["posterior_updated"] is False
    assert official["pair_interactions"] is None
    assert official["same_state_uniform_counterfactual_evaluated"] is False


def test_annotation_rejects_tampered_aggregate(tmp_path: Path) -> None:
    aggregate = _aggregate(tmp_path)
    with (aggregate / "episode_metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        annotate(aggregate, tmp_path / "roles.json")


def test_annotation_rejects_wrong_expected_pair_count(tmp_path: Path) -> None:
    aggregate = _aggregate(tmp_path, pairs=1)

    with pytest.raises(ValueError, match="expected 320"):
        annotate(aggregate, tmp_path / "roles.json", expected_pairs=320)
