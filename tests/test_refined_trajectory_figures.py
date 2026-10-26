from __future__ import annotations

import json

import numpy as np
import pytest

from sage.experiments.plot_refined_trajectories import POLICIES, interaction, select_h3


def test_additive_factorial_effect_is_zero():
    baseline = np.asarray([[0.3, -0.2], [0.6, 0.1]])
    first = np.asarray([[1.0, 0.0], [0.5, 0.1]])
    second = np.asarray([[-0.3, 0.2], [0.1, 0.4]])
    values = np.stack([baseline, baseline + first, baseline + second, baseline + first + second])
    np.testing.assert_allclose(interaction(values), 0, atol=1e-12)


def test_factorial_effect_retains_joint_term():
    values = np.zeros((4, 3, 2))
    values[3] = [0.5, -0.25]
    np.testing.assert_allclose(interaction(values), [[0.5, -0.25]] * 3)


def screening_fixture(directory, cases):
    (directory / "h3_data").mkdir()
    (directory / "h3_screening_protocol.json").write_text(json.dumps({"screening_cases": list(cases)}))
    for case, outcomes in cases.items():
        for policy, (safe, time) in zip(POLICIES, outcomes):
            stem = directory / "h3_data" / f"case{case}_{policy}"
            np.savez(stem.with_suffix(".npz"), humans=np.zeros((2, 8, 2)))
            metadata = {"metrics": {"initial_scene_hash": f"scene{case}", "success": 1.0,
                                    "collision_free_success": float(safe), "collision": float(not safe),
                                    "wall_collision": 0.0, "total_time": time},
                        "minimum_sampled_surface_gap_m": 0.1}
            stem.with_suffix(".json").write_text(json.dumps(metadata))


def test_selection_excludes_unsafe_tracer_even_when_fastest(tmp_path):
    screening_fixture(tmp_path, {
        144: [(False, 3), (True, 8), (True, 8), (True, 8)],
        145: [(True, 9), (True, 8), (True, 8), (True, 8)],
    })
    result = select_h3(tmp_path)
    assert result["selected"]["case"] == 145
    assert not result["selected"]["has_sicnav_advantage"]
    assert len(result["rows"]) == 8


def test_selection_prefers_safety_gain_over_time_gain(tmp_path):
    screening_fixture(tmp_path, {
        144: [(True, 3), (True, 8), (True, 8), (True, 8)],
        145: [(True, 9), (False, 8), (True, 8), (True, 8)],
    })
    result = select_h3(tmp_path)
    assert result["selected"]["case"] == 145
    assert result["selected"]["sicnav_safety_gain"] == 1


def test_incomplete_screen_cannot_be_silently_selected(tmp_path):
    screening_fixture(tmp_path, {144: [(True, 8)] * 4})
    (tmp_path / "h3_data/case144_sicnav-np.json").unlink()
    with pytest.raises(FileNotFoundError):
        select_h3(tmp_path)


def test_mismatched_initial_scene_is_rejected(tmp_path):
    screening_fixture(tmp_path, {144: [(True, 8)] * 4})
    path = tmp_path / "h3_data/case144_sicnav-np.json"
    metadata = json.loads(path.read_text())
    metadata["metrics"]["initial_scene_hash"] = "another-scene"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="scene mismatch"):
        select_h3(tmp_path)
