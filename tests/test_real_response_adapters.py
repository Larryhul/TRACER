import json
import zipfile

import numpy as np
import pytest

from tracer.data.real_response import (
    build_real_response_dataset,
    load_bi3_positions,
    load_navware_positions,
)
from tracer.data.real_response_download_bi3 import install_bi3_archive
from tracer.experiments.run_external_response_benchmark import _validate_protocol_arrays
from tracer.experiments.summarize_real_h1 import summarize


def test_navware_pose_csv_loader(tmp_path):
    path = tmp_path / "scene_robot_and_participants.csv"
    path.write_text(
        "timestamp,robot_x,robot_y,robot_yaw_rad,x1,y1,x2,y2\n"
        "1730000000000000000,0,0,0,1,0,2,0\n"
        "1730000000100000000,0.1,0,0,1.1,0,2.1,0\n",
        encoding="utf-8",
    )
    sequence_id, positions, dt = load_navware_positions(path)
    assert sequence_id == "scene_robot_and_participants"
    assert positions.shape == (2, 3, 2)
    assert np.isclose(dt, 0.1)
    assert not np.isnan(positions[:, 2]).any()


def test_bi3_json_loader(tmp_path):
    path = tmp_path / "condition.json"
    path.write_text(
        json.dumps(
            [
                {
                    "time": 0.0,
                    "robot_state": [0.0, 0.0],
                    "agent_states": [[1.0, 0.0], [2.0, 0.0]],
                },
                {
                    "time": 0.4,
                    "robot_state": [0.1, 0.0],
                    "agent_states": [[1.1, 0.0], [2.1, 0.0]],
                },
            ]
        ),
        encoding="utf-8",
    )
    sequence_id, positions, dt = load_bi3_positions(path)
    assert sequence_id == "condition"
    assert positions.shape == (2, 3, 2)
    assert np.isclose(dt, 0.4)


def test_bi3_split_conversion(tmp_path):
    root = tmp_path / "bi3"
    for site, experiment in (("um", "um1"), ("laas", "laas1"), ("um", "um14"), ("um", "um16")):
        source_dir = root / "jsons" / site / experiment
        source_dir.mkdir(parents=True)
        rows = [
            {
                "time": step * 0.1,
                "robot_state": [step * 0.1, 0.0],
                "agent_states": [
                    [1.0 + step * 0.1, 0.1],
                    [2.0 + step * 0.2, -0.1],
                ],
            }
            for step in range(241)
        ]
        (source_dir / "forward.json").write_text(json.dumps(rows), encoding="utf-8")
    output_dir = tmp_path / "converted"
    summary = build_real_response_dataset(
        "bi3",
        root,
        output_dir,
        history=9,
        horizon=12,
        windows=8,
        execute_steps=3,
        max_entities=2,
        min_entities=1,
        dt=0.4,
        episode_stride=2,
    )
    assert all(summary["splits"][split]["episodes"] > 0 for split in ("train", "val", "test"))
    train_ids = json.loads(
        (output_dir / "bi3_train.json").read_text(encoding="utf-8")
    )["recording_ids"]
    assert len(set(train_ids)) == len(train_ids)
    assert len(train_ids) == 2
    with np.load(output_dir / "bi3_train.npz") as arrays:
        assert arrays["entity_history"].shape[0] == 20
        np.testing.assert_allclose(arrays["target_future_vel"][:, 0, :, :, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(arrays["robot_candidates"][..., 2], 1.0, atol=1e-5)
        train_entities = set(arrays["entity_id"].flatten())
    with np.load(output_dir / "bi3_test.npz") as arrays:
        assert train_entities.isdisjoint(set(arrays["entity_id"].flatten()))


def test_bi3_local_archive_install(tmp_path):
    archive = tmp_path / "bi3.zip"
    source = tmp_path / "source" / "jsons" / "um" / "um1"
    source.mkdir(parents=True)
    (source / "condition.json").write_text("[]", encoding="utf-8")
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(source / "condition.json", "jsons/um/um1/condition.json")
    destination = tmp_path / "installed"
    result = install_bi3_archive(archive, destination)
    assert result["status"] == "installed"
    assert result["json_count"] == 1
    assert (destination / "download_manifest.json").is_file()


def test_bi3_rootless_official_archive_install(tmp_path):
    archive = tmp_path / "bi3_rootless.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("/", "")
        handle.writestr("um/um1/cv.json", "[]")
        handle.writestr("laas/laas1/cv.json", "[]")
    destination = tmp_path / "installed"
    result = install_bi3_archive(archive, destination)
    assert result["json_count"] == 2
    assert (destination / "jsons/um/um1/cv.json").is_file()
    assert (destination / "jsons/laas/laas1/cv.json").is_file()


def test_bi3_archive_merge_keeps_prior_site(tmp_path):
    first = tmp_path / "um.zip"
    second = tmp_path / "laas.zip"
    with zipfile.ZipFile(first, "w") as handle:
        handle.writestr("um/um1/cv.json", "[]")
    with zipfile.ZipFile(second, "w") as handle:
        handle.writestr("laas/laas1/cv.json", "[]")
    destination = tmp_path / "installed"
    install_bi3_archive(first, destination)
    result = install_bi3_archive(second, destination, merge=True)
    assert result["merge"] is True
    assert (destination / "jsons/um/um1/cv.json").is_file()
    assert (destination / "jsons/laas/laas1/cv.json").is_file()


def test_bi3_archive_merge_preflights_conflicts(tmp_path):
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    with zipfile.ZipFile(first, "w") as handle:
        handle.writestr("um/um1/cv.json", "one")
    with zipfile.ZipFile(second, "w") as handle:
        handle.writestr("um/um1/cv.json", "two")
        handle.writestr("laas/laas1/cv.json", "new")
    destination = tmp_path / "installed"
    install_bi3_archive(first, destination)
    with pytest.raises(ValueError, match="conflicting JSON"):
        install_bi3_archive(second, destination, merge=True)
    assert (destination / "jsons/um/um1/cv.json").read_text(encoding="utf-8") == "one"
    assert not (destination / "jsons/laas/laas1/cv.json").exists()


def test_bi3_invalid_archive_does_not_destroy_existing_data(tmp_path):
    archive = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("README.txt", "not the dataset")
    source = tmp_path / "installed" / "jsons" / "um" / "um1" / "condition.json"
    source.parent.mkdir(parents=True)
    source.write_text("[]")
    with pytest.raises(FileNotFoundError):
        install_bi3_archive(archive, tmp_path / "installed", overwrite=True)
    assert source.read_text(encoding="utf-8") == "[]"


def test_bi3_wrapped_archive_preserves_unrelated_files(tmp_path):
    archive = tmp_path / "wrapped.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Bi3/jsons/um/um1/condition.json", "[]")
        handle.writestr("Bi3/large_unused_video.mp4", "not required")
    destination = tmp_path / "installed"
    destination.mkdir()
    marker = destination / "unrelated.txt"
    marker.write_text("keep")
    install_bi3_archive(archive, destination, overwrite=True)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert (destination / "jsons/um/um1/condition.json").is_file()
    assert not (destination / "Bi3").exists()


def test_bi3_archive_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../jsons/um/um1/condition.json", "[]")
    with pytest.raises(ValueError, match="unsafe archive path"):
        install_bi3_archive(archive, tmp_path / "installed")


@pytest.fixture
def protocol_arrays():
    return {
        "entity_history": np.zeros((2, 1, 8, 9, 2), dtype=np.float32),
        "robot_current": np.zeros((2, 1, 8, 1, 4), dtype=np.float32),
        "robot_candidates": np.zeros((2, 1, 8, 1, 12, 4), dtype=np.float32),
        "target_future_vel": np.zeros((2, 1, 8, 12, 2), dtype=np.float32),
        "active_mask": np.ones((2, 1, 8), dtype=np.bool_),
    }
def test_runner_accepts_matching_protocol(tmp_path, protocol_arrays):
    _validate_protocol_arrays(
        protocol_arrays, tmp_path / "valid.npz", history=9, horizon=12, windows=8
    )


def test_runner_rejects_protocol_mismatch(tmp_path, protocol_arrays):
    with pytest.raises(ValueError, match="entity_history"):
        _validate_protocol_arrays(
            protocol_arrays, tmp_path / "mismatch.npz", history=8, horizon=12, windows=8
        )


@pytest.mark.parametrize("field", ["robot_current", "robot_candidates", "target_future_vel", "active_mask"])
def test_runner_rejects_unaligned_entities(tmp_path, protocol_arrays, field):
    protocol_arrays[field] = np.repeat(protocol_arrays[field], 2, axis=1)
    with pytest.raises(ValueError, match="unaligned"):
        _validate_protocol_arrays(
            protocol_arrays, tmp_path / "mismatch.npz", history=9, horizon=12, windows=8
        )


def test_bi3_sampling_does_not_fill_recording_gaps(tmp_path):
    path = tmp_path / "gaps.json"
    rows = [
        {"time": timestamp, "robot_state": [timestamp, 0], "agent_states": [[timestamp, 1], [timestamp, 2]]}
        for timestamp in (0.0, 0.1, 0.2, 1.6, 1.7, 2.0)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    _, positions, dt = load_bi3_positions(path, dt=0.4)
    assert dt == 0.4
    assert np.isnan(positions[2:4]).all()
    np.testing.assert_allclose(positions[4, 0], [1.6, 0])


def test_summary_rejects_misaligned_roots_and_seeds(tmp_path):
    with pytest.raises(ValueError, match="equal length"):
        summarize("Bi3", [tmp_path], [1, 2], tmp_path / "summary")


def test_summary_keeps_matched_action_ablation(tmp_path):
    roots = [tmp_path / "run1", tmp_path / "run2"]
    methods = ("history-only", "no-belief", "reset", "persistent", "recurrent")
    for root, seed in zip(roots, [1, 2]):
        directory = root / f"seed{seed}"
        directory.mkdir(parents=True)
        metrics = {
            method: {"predictive_trajectory_nll": float(seed), "trajectory_ade": float(seed), "trajectory_fde": float(seed)}
            for method in methods
        }
        for split in ("val", "test"):
            (directory / f"{split}_metrics.json").write_text(json.dumps(metrics))
    summary = summarize("fixture", roots, [1, 2], tmp_path / "summary")
    assert summary["splits"]["test"]["no-belief"]["mean"] == [1.5, 1.5, 1.5]
    assert "changes both action conditioning and belief" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_bi3_nearest_timestamp_sampling_preserves_missing_human(tmp_path):
    path = tmp_path / "irregular.json"
    rows = [
        {"time": timestamp, "robot_state": [timestamp, 0], "agent_states": [[timestamp, 1], [timestamp, 2]]}
        for timestamp in (0., .11, .29, .39, .63, .79, .9)
    ]
    rows[3]["agent_states"] = [[.39, 1]]
    path.write_text(json.dumps(list(reversed(rows))))
    _, positions, dt = load_bi3_positions(path, dt=.4)
    assert dt == .4
    np.testing.assert_allclose(positions[:, 0, 0], [0, .39, .79])
    assert np.isnan(positions[1, 2]).all()
    assert np.isfinite(positions[1, 1]).all()
