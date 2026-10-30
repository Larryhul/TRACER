"""Prepare a reproducible scene-level SDD reference split.

The official SDD annotation repository stores one annotations.txt per scene.
SAGE's official adapter uses a lightweight trajectory pickle only to obtain
the scene IDs, so this utility creates that reference index without copying
the annotation rows into a second large file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def discover_scene_files(root: Path) -> list[Path]:
    paths = sorted(root.glob("annotations/*/video*/annotations.txt"))
    if not paths:
        paths = sorted(root.glob("*/video*/annotations.txt"))
    return paths


def scene_id_for(path: Path) -> str:
    scene_name, video_name = path.parts[-3], path.parts[-2]
    return f"{scene_name}_{video_name.removeprefix('video')}"


def _reference_frame(scene_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sceneId": scene_ids,
            "trackId": [0] * len(scene_ids),
            "frame": [0] * len(scene_ids),
            "x": [0.0] * len(scene_ids),
            "y": [0.0] * len(scene_ids),
        }
    )


def prepare(root: Path, output_dir: Path, train_scenes: int) -> dict[str, object]:
    paths = discover_scene_files(root)
    if len(paths) < 2:
        raise FileNotFoundError(f"Expected at least two SDD scene annotation files under {root}")
    scene_ids = [scene_id_for(path) for path in paths]
    train_ids = scene_ids[: max(1, min(train_scenes, len(scene_ids) - 1))]
    test_ids = scene_ids[len(train_ids) :]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_trajnet.pkl"
    test_path = output_dir / "test_trajnet.pkl"
    _reference_frame(train_ids).to_pickle(train_path)
    _reference_frame(test_ids).to_pickle(test_path)
    manifest = {
        "source_root": str(root),
        "scene_files": [str(path) for path in paths],
        "scene_ids": scene_ids,
        "train_scene_ids": train_ids,
        "test_scene_ids": test_ids,
        "train_reference": str(train_path),
        "test_reference": str(test_path),
        "note": "Reference pickles contain scene IDs only; official annotations remain the source of trajectory rows and labels.",
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare scene-level SDD reference pickles.")
    parser.add_argument("--sdd-root", default="raw_data/external/sdd_annotations_subset")
    parser.add_argument("--output-dir", default="raw_data/external/sdd_annotations_subset")
    parser.add_argument("--train-scenes", type=int, default=4)
    args = parser.parse_args()
    manifest = prepare(Path(args.sdd_root), Path(args.output_dir), args.train_scenes)
    print(f"train_scenes={len(manifest['train_scene_ids'])}")
    print(f"test_scenes={len(manifest['test_scene_ids'])}")
    print(f"saved_manifest={Path(args.output_dir) / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
