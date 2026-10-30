from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


try:
    import dill as pickle_backend
except ModuleNotFoundError:
    import pickle as pickle_backend


ANNOTATION_COLUMNS = [
    "trackId",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "frame",
    "lost",
    "occluded",
    "generated",
    "label",
]


def load_trajnet_scene_ids(paths: list[Path]) -> set[str]:
    try:
        import pandas as pd  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("pandas is required to read SDD trajnet pickle files") from exc
    scene_ids: set[str] = set()
    for path in paths:
        with path.open("rb") as f:
            try:
                df = pickle_backend.load(f, encoding="latin1")
            except TypeError:
                f.seek(0)
                df = pickle_backend.load(f)
        if "sceneId" not in df.columns:
            raise ValueError(f"{path} does not contain sceneId")
        scene_ids.update(str(value) for value in df["sceneId"].dropna().unique())
    return scene_ids


def parse_scene_id(scene_id: str) -> tuple[str, str]:
    if "_" not in scene_id:
        raise ValueError(f"Cannot parse sceneId={scene_id!r}; expected e.g. bookstore_0")
    scene, video = scene_id.rsplit("_", 1)
    return scene, video


def candidate_annotation_paths(root: Path, scene: str, video: str) -> list[Path]:
    video_names = [video, f"video{video}", f"video_{video}"]
    candidates: list[Path] = []
    for video_name in video_names:
        candidates.extend(
            [
                root / "annotations" / scene / video_name / "annotations.txt",
                root / scene / video_name / "annotations.txt",
                root / "annotations" / scene / video_name / "annotation.txt",
                root / scene / video_name / "annotation.txt",
            ]
        )
    return candidates


def find_annotation_file(root: Path, scene_id: str) -> Path | None:
    scene, video = parse_scene_id(scene_id)
    for candidate in candidate_annotation_paths(root, scene, video):
        if candidate.exists():
            return candidate
    return None


def read_annotation_labels(path: Path) -> dict[str, str]:
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=" ")
        for raw in reader:
            parts = [p for p in raw if p != ""]
            if len(parts) < 10:
                continue
            track_id = parts[0]
            label = parts[9].strip().lower()
            if track_id and label:
                label_counts[track_id][label] += 1
    return {track_id: counts.most_common(1)[0][0] for track_id, counts in label_counts.items() if counts}


def write_label_map(root: Path, scene_ids: set[str], output: Path) -> tuple[int, list[str]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    rows: list[dict[str, str]] = []
    for scene_id in sorted(scene_ids):
        annotation_file = find_annotation_file(root, scene_id)
        if annotation_file is None:
            missing.append(scene_id)
            continue
        for track_id, label in sorted(read_annotation_labels(annotation_file).items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
            rows.append({"sceneId": scene_id, "trackId": track_id, "label": label})

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sceneId", "trackId", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sceneId/trackId/label map from official SDD annotations.")
    parser.add_argument("--sdd-root", required=True, help="Official Stanford Drone Dataset root containing annotations/*/*/annotations.txt")
    parser.add_argument("--train-pkl", default="raw_data/stanford/train_trajnet.pkl")
    parser.add_argument("--test-pkl", default="raw_data/stanford/test_trajnet.pkl")
    parser.add_argument("--output", default="raw_data/stanford/sdd_label_map.csv")
    args = parser.parse_args()

    root = Path(args.sdd_root)
    scene_ids = load_trajnet_scene_ids([Path(args.train_pkl), Path(args.test_pkl)])
    num_rows, missing = write_label_map(root, scene_ids, Path(args.output))
    print(f"sdd_root={root}")
    print(f"scene_ids={len(scene_ids)}")
    print(f"saved={args.output}")
    print(f"rows={num_rows}")
    print(f"missing_scenes={len(missing)}")
    if missing:
        print("missing_scene_ids=" + ",".join(missing[:50]))
        print("expected layouts include:")
        print("  <sdd-root>/annotations/bookstore/video0/annotations.txt")
        print("  <sdd-root>/bookstore/video0/annotations.txt")


if __name__ == "__main__":
    main()
