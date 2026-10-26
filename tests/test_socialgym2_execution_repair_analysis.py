from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_socialgym2_execution_repair import analyze


FIELDS = (
    "policy",
    "seed",
    "episode_seed",
    "scenario",
    "scenario_hash",
    "initial_state_hash",
    "initial_scene_snapshot",
    "episode",
    "terminal_reached",
    "team_success",
    "robot_success_fraction",
    "collision_free_team_success",
    "initial_collision",
    "collision",
    "task_progress",
    "min_distance",
    "fallback_rate",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(tmp_path: Path, name: str, *, repaired: bool, mismatch: bool = False) -> Path:
    path = tmp_path / name
    path.mkdir()
    metrics = path / "episode_metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for episode in range(2):
            initial_hash = f"state-{episode}"
            if mismatch and episode == 1:
                initial_hash = "different"
            writer.writerow({
                "policy": "tracer-persistent",
                "seed": "7",
                "episode_seed": str(7000 + episode),
                "scenario": "envs/scenario/open",
                "scenario_hash": f"{name}-raw-{episode}",
                "initial_state_hash": initial_hash,
                "initial_scene_snapshot": json.dumps({"scene": episode}),
                "episode": str(episode),
                "terminal_reached": str(repaired or episode == 0),
                "team_success": str(repaired or episode == 0),
                "robot_success_fraction": str(1.0 if repaired else 0.5),
                "collision_free_team_success": str(repaired),
                "initial_collision": "0",
                "collision": str(not repaired),
                "task_progress": str(0.9 if repaired else 0.7),
                "min_distance": str(0.8 if repaired else 0.6),
                "fallback_rate": "0.0",
            })
    metadata = path / "metadata.json"
    metadata.write_text(json.dumps({"algorithm_sources": {"controller.py": name}}), encoding="utf-8")
    (path / "manifest.json").write_text(json.dumps({
        "status": "COMPLETED",
        "artifacts": {
            "metadata.json": _sha256(metadata),
            "episode_metrics.csv": _sha256(metrics),
        },
    }), encoding="utf-8")
    return path


def test_paired_repair_analysis_accepts_improved_system(tmp_path: Path) -> None:
    old = _run(tmp_path, "old", repaired=False)
    repaired = _run(tmp_path, "repaired", repaired=True)

    result = analyze(
        [old],
        [repaired],
        episodes=2,
        expected_seeds=[7],
        expected_scenarios=["envs/scenario/open"],
    )

    assert result["acceptance_gate"]["accepted"] is True
    assert result["acceptance_gate"]["map_gates"]["envs/scenario/open"]["accepted"] is True
    assert result["pooled"]["collision_free_team_success"]["paired_difference_repaired_minus_old"] == 1.0
    assert result["pairing_audit"]["raw_scenario_hash_mismatch_count"] == 2


def test_paired_repair_analysis_rejects_reset_mismatch(tmp_path: Path) -> None:
    old = _run(tmp_path, "old", repaired=False)
    repaired = _run(tmp_path, "repaired", repaired=True, mismatch=True)

    with pytest.raises(ValueError, match="initial_state_hash"):
        analyze(
            [old],
            [repaired],
            episodes=2,
            expected_seeds=[7],
            expected_scenarios=["envs/scenario/open"],
        )
