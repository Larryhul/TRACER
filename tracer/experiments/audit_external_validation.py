"""Run bounded external trajectory sanity checks.

The external datasets do not contain the paired joint interventions or the
executed-belief state required by the Formal-v1 causal claims. This runner
therefore measures data portability and a transparent constant-velocity
trajectory reference only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_HISTORY = 8
DEFAULT_HORIZON = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _track_windows(
    frames: np.ndarray,
    positions: np.ndarray,
    history: int,
    horizon: int,
) -> tuple[list[tuple[float, float]], float]:
    """Return constant-velocity ADE/FDE pairs and the contiguous-frame rate."""

    order = np.argsort(frames)
    frames = np.asarray(frames, dtype=np.int64)[order]
    positions = np.asarray(positions, dtype=np.float64)[order]
    if len(frames) < history + horizon:
        return [], 0.0
    gaps = np.diff(frames)
    continuity = float(np.mean(gaps == 1)) if len(gaps) else 1.0
    scores: list[tuple[float, float]] = []
    total = history + horizon
    for start in range(0, len(frames) - total + 1):
        window_frames = frames[start : start + total]
        if not np.all(np.diff(window_frames) == 1):
            continue
        window = positions[start : start + total]
        velocity = window[history - 1] - window[history - 2]
        steps = np.arange(1, horizon + 1, dtype=np.float64)[:, None]
        prediction = window[history - 1][None] + steps * velocity[None]
        target = window[history:]
        errors = np.linalg.norm(prediction - target, axis=-1)
        scores.append((float(errors.mean()), float(errors[-1])))
    return scores, continuity


def _sdd_scene_files(root: Path) -> list[Path]:
    candidates = sorted(root.glob("annotations/*/video*/annotations.txt"))
    if not candidates:
        candidates = sorted(root.glob("*/video*/annotations.txt"))
    return candidates


def _parse_sdd_file(path: Path, coordinate_scale: float) -> tuple[str, list[dict[str, Any]]]:
    relative = path.parts[-3:]
    scene_name = relative[0]
    video_name = relative[1]
    scene_id = f"{scene_name}_{video_name.removeprefix('video')}"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=9)
            if len(parts) < 10:
                continue
            try:
                _, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label = parts
                frame_i = int(float(frame))
                lost_i = int(float(lost))
                x = (float(xmin) + float(xmax)) / (2.0 * coordinate_scale)
                y = (float(ymin) + float(ymax)) / (2.0 * coordinate_scale)
            except ValueError:
                continue
            rows.append(
                {
                    "scene_id": scene_id,
                    "track_id": parts[0],
                    "frame": frame_i,
                    "x": x,
                    "y": y,
                    "label": label.strip("\"'").lower(),
                    "lost": lost_i,
                    "occluded": int(float(occluded)),
                    "generated": int(float(generated)),
                }
            )
    return scene_id, rows


def _audit_sdd(
    root: Path,
    history: int,
    horizon: int,
    frame_stride: int,
    coordinate_scale: float,
    limit_scenes: int,
) -> dict[str, Any]:
    files = _sdd_scene_files(root)
    selected = files[:limit_scenes] if limit_scenes > 0 else files
    if not selected:
        return {
            "status": "missing",
            "root": str(root),
            "scene_files": [],
            "reason": "No annotations/*/video*/annotations.txt files were found.",
        }

    scene_count = 0
    row_count = 0
    track_count = 0
    window_count = 0
    continuity_values: list[float] = []
    ade_values: list[float] = []
    fde_values: list[float] = []
    labels: Counter[str] = Counter()
    source_hashes: dict[str, str] = {}
    for path in selected:
        scene_id, rows = _parse_sdd_file(path, coordinate_scale)
        source_hashes[str(path)] = _sha256(path)
        if not rows:
            continue
        scene_count += 1
        row_count += len(rows)
        labels.update(row["label"] for row in rows)
        frame_df = pd.DataFrame(rows)
        frame_df["frame_bin"] = frame_df["frame"] // frame_stride
        grouped = (
            frame_df.groupby(["track_id", "frame_bin"], as_index=False)
            .agg({"x": "mean", "y": "mean", "label": "first"})
            .sort_values(["track_id", "frame_bin"])
        )
        for track_id, track_df in grouped.groupby("track_id", sort=True):
            track_count += 1
            scores, continuity = _track_windows(
                track_df["frame_bin"].to_numpy(),
                track_df[["x", "y"]].to_numpy(),
                history,
                horizon,
            )
            continuity_values.append(continuity)
            window_count += len(scores)
            ade_values.extend(score[0] for score in scores)
            fde_values.extend(score[1] for score in scores)

    return {
        "status": "complete",
        "root": str(root),
        "scene_files": [str(path) for path in selected],
        "source_sha256": source_hashes,
        "scenes": scene_count,
        "rows": row_count,
        "tracks": track_count,
        "windows": window_count,
        "history": history,
        "horizon": horizon,
        "frame_stride": frame_stride,
        "coordinate_scale": coordinate_scale,
        "mean_track_continuity": float(np.mean(continuity_values)) if continuity_values else 0.0,
        "constant_velocity_ade": float(np.mean(ade_values)) if ade_values else None,
        "constant_velocity_fde": float(np.mean(fde_values)) if fde_values else None,
        "label_counts": dict(sorted(labels.items())),
    }


def _audit_thor_magni(
    path: Path,
    history: int,
    horizon: int,
) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    frame = pd.read_csv(path)
    required = {
        "Time",
        "x",
        "y",
        "ag_id",
        "file_name",
        "tracklet_id",
        "dataset_name",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"status": "invalid", "path": str(path), "missing_columns": missing}
    frame = frame.dropna(subset=["Time", "x", "y", "ag_id", "file_name", "tracklet_id"]).copy()
    frame["episode_id"] = frame["dataset_name"].astype(str) + "/" + frame["file_name"].astype(str)
    frame["track_key"] = (
        frame["episode_id"] + "/" + frame["tracklet_id"].astype(str) + "/" + frame["ag_id"].astype(str)
    )
    ade_values: list[float] = []
    fde_values: list[float] = []
    continuity_values: list[float] = []
    track_count = 0
    window_count = 0
    action_counts: Counter[str] = Counter()
    for _, track in frame.groupby("track_key", sort=True):
        track = track.sort_values(["Time", "frame_id"]).drop_duplicates("Time")
        track_count += 1
        frames = np.arange(len(track), dtype=np.int64)
        scores, continuity = _track_windows(
            frames,
            track[["x", "y"]].to_numpy(dtype=np.float64),
            history,
            horizon,
        )
        continuity_values.append(continuity)
        window_count += len(scores)
        ade_values.extend(score[0] for score in scores)
        fde_values.extend(score[1] for score in scores)
        if "action" in track:
            action_counts.update(
                str(value)
                for value in track["action"].dropna().astype(str)
                if value.strip()
            )
    return {
        "status": "complete",
        "path": str(path),
        "source_sha256": _sha256(path),
        "rows": int(len(frame)),
        "episodes": int(frame["episode_id"].nunique()),
        "tracks": track_count,
        "windows": window_count,
        "history": history,
        "horizon": horizon,
        "mean_track_continuity": float(np.mean(continuity_values)) if continuity_values else 0.0,
        "constant_velocity_ade": float(np.mean(ade_values)) if ade_values else None,
        "constant_velocity_fde": float(np.mean(fde_values)) if fde_values else None,
        "action_classes": len(action_counts),
        "action_counts": dict(action_counts.most_common()),
        "agent_type_counts": {
            str(key): int(value)
            for key, value in frame["agent_type"].value_counts(dropna=False).items()
        }
        if "agent_type" in frame
        else {},
    }


def _audit_jrdb(root: Path) -> dict[str, Any]:
    required = {
        "train_dataset": root / "train_dataset",
        "test_dataset": root / "test_dataset",
        "detections": root / "detections",
        "processed_odometry": root / "processed" / "odometry",
        "processed_odoemtry_legacy": root / "processed" / "odoemtry",
        "processed_keypoints": root / "processed" / "labels" / "labels_3d_keypoints",
    }
    present = {name: path.exists() for name, path in required.items()}
    complete = present["train_dataset"] and present["test_dataset"] and (
        present["processed_odometry"] or present["processed_odoemtry_legacy"]
    )
    return {
        "status": "ready" if complete else "blocked_missing_or_authenticated_assets",
        "root": str(root),
        "required_paths": {name: str(path) for name, path in required.items()},
        "present": present,
        "download_instructions": "JRDB raw train/test, detections, odometry, and keypoints require the official JRDB account/download flow.",
        "does_not_claim_results": True,
    }


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = ["dataset", "status", "scenes_or_episodes", "tracks", "windows", "ade", "fde", "continuity"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for dataset, result in payload["datasets"].items():
            writer.writerow(
                {
                    "dataset": dataset,
                    "status": result.get("status"),
                    "scenes_or_episodes": result.get("scenes", result.get("episodes", "")),
                    "tracks": result.get("tracks", ""),
                    "windows": result.get("windows", ""),
                    "ade": result.get("constant_velocity_ade", ""),
                    "fde": result.get("constant_velocity_fde", ""),
                    "continuity": result.get("mean_track_continuity", ""),
                }
            )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# External Validation Audit",
        "",
        "This is a bounded trajectory portability and HRI compatibility sanity check. External datasets do not provide the paired joint interventions, executed response observations, or belief ground truth required by the Formal-v1 causal claims.",
        "",
        "| Dataset | Status | Scenes/episodes | Tracks | Windows | Constant-velocity ADE | Constant-velocity FDE | Mean continuity |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in payload["datasets"].items():
        lines.append(
            f"| {dataset} | {result.get('status')} | "
            f"{result.get('scenes', result.get('episodes', '-'))} | "
            f"{result.get('tracks', '-')} | {result.get('windows', '-')} | "
            f"{result.get('constant_velocity_ade', '-') if result.get('constant_velocity_ade') is not None else '-'} | "
            f"{result.get('constant_velocity_fde', '-') if result.get('constant_velocity_fde') is not None else '-'} | "
            f"{result.get('mean_track_continuity', '-') if result.get('mean_track_continuity') is not None else '-'} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: ADE/FDE are reference values for a transparent constant-velocity predictor in each dataset's native coordinate normalization. They are not claims that the SAGE model transfers without retraining.",
            "",
            "JRDB remains fail-closed until the official authenticated raw assets and preprocessing outputs are present.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(
    sdd_root: Path,
    thor_magni_csv: Path,
    jrdb_root: Path,
    output_dir: Path,
    history: int = DEFAULT_HISTORY,
    horizon: int = DEFAULT_HORIZON,
    sdd_frame_stride: int = 12,
    sdd_coordinate_scale: float = 50.0,
    sdd_limit_scenes: int = 0,
) -> dict[str, Any]:
    payload = {
        "source_registry": {
            "SDD": {
                "repository": "https://github.com/flclain/StanfordDroneDataset",
                "revision": "9179f3c0dd9ac0bd67a6223703d04a7577a9fb18",
                "asset_scope": "six annotation files: bookstore/video0-1, gates/video0-1, nexus/video0-1",
                "license_file": "https://github.com/flclain/StanfordDroneDataset/blob/master/LICENSE",
            },
            "THOR-MAGNI": {
                "repository": "https://github.com/tmralmeida/thor-magni-actions",
                "revision": "b49fa515707584756e2c9078fffed324c6f278af",
                "asset": "data/processed/thor_magni/merged_data.csv",
                "license_file": "https://github.com/tmralmeida/thor-magni-actions/blob/main/LICENSE",
            },
            "JRDB": {
                "official_download": "https://jrdb.erc.monash.edu/#downloads",
                "status": "authenticated download required",
            },
        },
        "protocol": {
            "history": history,
            "horizon": horizon,
            "sdd_frame_stride": sdd_frame_stride,
            "sdd_coordinate_scale": sdd_coordinate_scale,
            "claim_boundary": "external sanity only; no joint causal or real-human preference claim",
        },
        "datasets": {
            "SDD": _audit_sdd(
                sdd_root,
                history,
                horizon,
                sdd_frame_stride,
                sdd_coordinate_scale,
                sdd_limit_scenes,
            ),
            "THOR-MAGNI": _audit_thor_magni(thor_magni_csv, history, horizon),
            "JRDB": _audit_jrdb(jrdb_root),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "external_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "external_validation.csv", payload)
    _write_markdown(output_dir / "external_validation.md", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit public external trajectory data.")
    parser.add_argument("--sdd-root", default="raw_data/external/sdd_annotations_subset")
    parser.add_argument("--thor-magni-csv", default="raw_data/external/thor_magni_merged_data.csv")
    parser.add_argument("--jrdb-root", default="raw_data/external/jrdb")
    parser.add_argument("--output-dir", default="reports/external_validation_2026-08-21")
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--sdd-frame-stride", type=int, default=12)
    parser.add_argument("--sdd-coordinate-scale", type=float, default=50.0)
    parser.add_argument("--sdd-limit-scenes", type=int, default=0)
    args = parser.parse_args()
    payload = audit(
        Path(args.sdd_root),
        Path(args.thor_magni_csv),
        Path(args.jrdb_root),
        Path(args.output_dir),
        history=args.history,
        horizon=args.horizon,
        sdd_frame_stride=args.sdd_frame_stride,
        sdd_coordinate_scale=args.sdd_coordinate_scale,
        sdd_limit_scenes=args.sdd_limit_scenes,
    )
    print(f"saved_json={Path(args.output_dir) / 'external_validation.json'}")
    for dataset, result in payload["datasets"].items():
        print(f"dataset={dataset} status={result.get('status')} windows={result.get('windows', 0)}")


if __name__ == "__main__":
    main()
