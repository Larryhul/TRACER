from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tracer.experiments.navigation_contract import initial_navigation_state
from tracer.experiments.run_sicnav_tracer_loop import _new_episode


def relative_static_segments(obstacles, origins):
    segments = np.asarray(obstacles, dtype=np.float32).reshape(-1, 4)
    origins = np.asarray(origins, dtype=np.float32)
    return segments[None] - np.tile(origins, (1, 2))[:, None]


def validate_rolling_reset(data, case, env):
    for human_index, human in enumerate(env.humans):
        rows = np.flatnonzero((data["case"] == case) & (data["human"] == human_index) & (data["window"] == 0))
        if not len(rows):
            raise ValueError("rolling map recovery requires initial observations for each identity")
        origin = np.asarray(human.get_position())
        relative_robot = np.asarray(env.robot.get_position()) - origin
        if not np.allclose(data["history"][rows, -1], origin, atol=1e-6, rtol=0):
            raise ValueError("rolling initial human positions differ from the map reset")
        if not np.allclose(data["current"][rows, 0, :2], relative_robot, atol=1e-6, rtol=0):
            raise ValueError("rolling initial robot positions differ from the map reset")


def main():
    parser = argparse.ArgumentParser(description="Attach observed static-map context to frozen live-controller data.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--collection-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source, output = Path(args.data_dir), Path(args.output_dir)
    if output.exists():
        raise ValueError("static-context data requires a fresh output directory")
    manifest = json.loads((source / "data_manifest.json").read_text(encoding="utf-8"))
    datasets = {}
    for split in ("train", "val"):
        with np.load(source / f"{split}.npz") as loaded:
            datasets[split] = dict(loaded)
    rolling = manifest["version"] == "rolling-prefix-aligned-v2"
    if rolling:
        root = Path(manifest["parameters"]["external_root"]).resolve()
        env_config = root / "sicnav/configs/env.config"
        rows = {int(case): None for data in datasets.values() for case in np.unique(data["case"])}
    else:
        if args.collection_dir is None:
            raise ValueError("live-controller data requires its collection directory")
        collection = Path(args.collection_dir)
        if manifest["collection_metadata_sha256"] != hashlib.sha256((collection / "metadata.json").read_bytes()).hexdigest():
            raise ValueError("static map source does not match the frozen collection")
        metadata = json.loads((collection / "metadata.json").read_text(encoding="utf-8"))
        with (collection / "episode_metrics.csv").open() as handle:
            rows = {int(row["case"]): row for row in csv.DictReader(handle)}
        root = Path(metadata["parameters"]["external_root"]).resolve()
        env_config = Path(metadata["parameters"]["env_config"]).resolve()
    maps = {}
    for case, row in rows.items():
        env, robot, _, obstacles, *_ = _new_episode(root, case, env_config, None, "sicnav-unicycle")
        if rolling:
            for data in datasets.values():
                if case in data["case"]:
                    validate_rolling_reset(data, case, env)
        elif initial_navigation_state(env, robot, obstacles)["initial_scene_hash"] != row["initial_scene_hash"]:
            raise ValueError("static map reset differs from the collected initial scene")
        maps[case] = obstacles
    if len({len(obstacles) for obstacles in maps.values()}) != 1:
        raise ValueError("static map batching requires a consistent segment count")
    for split, data in datasets.items():
        segments = np.zeros((len(data["target"]), len(next(iter(maps.values()))), 4), dtype=np.float32)
        for case in np.unique(data["case"]):
            selected = data["case"] == case
            segments[selected] = relative_static_segments(maps[int(case)], data["history"][selected, -1])
        data["static_segments"] = segments
        datasets[split] = data
    manifest["static_context"] = {
        "semantics": "observed static line segments relative to each entity's current observed position; no human hidden preferences or goals",
        "source_data_sha256": {split: hashlib.sha256((source / f"{split}.npz").read_bytes()).hexdigest() for split in datasets},
        "initial_state_hash_audit": "not_recorded_in_rolling_source" if rolling else "pass",
        "initial_observed_position_audit": "pass", "map_dynamics": "fixed official map within each episode",
        "map_source_config_sha256": hashlib.sha256(env_config.read_bytes()).hexdigest(),
        "maps_sha256": hashlib.sha256(json.dumps(maps, sort_keys=True).encode()).hexdigest(),
    }
    output.mkdir(parents=True)
    for split, data in datasets.items():
        np.savez_compressed(output / f"{split}.npz", **data)
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
