"""Adapters for real-world robot--entity response recordings."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tracer.data.external_response import (
    build_sequence_episodes,
    concatenate_episode_records,
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _navware_csv_rows(path: Path) -> tuple[str, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        required = {"timestamp", "robot_x", "robot_y"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} missing NavWareSet columns: {sorted(missing)}")
        return path.stem, list(reader)


def load_navware_positions(path: Path) -> tuple[str, np.ndarray, float] | None:
    sequence_id, rows = _navware_csv_rows(path)
    if len(rows) < 2:
        return None
    participant_columns: list[tuple[str, str]] = []
    fieldnames = set(rows[0])
    for index in range(1, 64):
        x_name, y_name = f"x{index}", f"y{index}"
        if x_name in fieldnames and y_name in fieldnames:
            participant_columns.append((x_name, y_name))
    if not participant_columns:
        return None

    timestamps: list[float] = []
    values: list[list[list[float]]] = []
    for row in rows:
        timestamp = _finite(row.get("timestamp"))
        robot_x = _finite(row.get("robot_x"))
        robot_y = _finite(row.get("robot_y"))
        if timestamp is None or robot_x is None or robot_y is None:
            continue
        timestamp = timestamp / 1e9 if abs(timestamp) > 1e12 else timestamp
        tracks = [[robot_x, robot_y]]
        for x_name, y_name in participant_columns:
            x_value = _finite(row.get(x_name))
            y_value = _finite(row.get(y_name))
            tracks.append(
                [x_value, y_value]
                if x_value is not None and y_value is not None
                else [np.nan, np.nan]
            )
        timestamps.append(timestamp)
        values.append(tracks)
    if len(values) < 2:
        return None
    order = np.argsort(np.asarray(timestamps, dtype=np.float64))
    times = np.asarray(timestamps, dtype=np.float64)[order]
    positions = np.asarray(values, dtype=np.float32)[order]
    unique, indices = np.unique(times, return_index=True)
    times = unique
    positions = positions[indices]
    if len(times) < 2:
        return None
    dt_values = np.diff(times)
    dt_values = dt_values[dt_values > 1e-6]
    if not len(dt_values):
        return None
    dt = float(np.median(dt_values))
    return sequence_id, positions, dt


def _json_source_paths(root: Path) -> list[Path]:
    return sorted(root.glob("jsons/*/*/*.json"))


def _sample_bi3_positions(times: np.ndarray, positions: np.ndarray, dt: float) -> np.ndarray:
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    target_times = np.arange(times[0], times[-1] + dt * 0.5, dt)
    target_times = target_times[target_times <= times[-1]]
    next_indices = np.clip(np.searchsorted(times, target_times), 0, len(times) - 1)
    previous_indices = np.maximum(next_indices - 1, 0)
    use_previous = (
        np.abs(times[previous_indices] - target_times)
        <= np.abs(times[next_indices] - target_times)
    )
    indices = np.where(use_previous, previous_indices, next_indices)
    sampled = positions[indices].copy()
    sampled[np.abs(times[indices] - target_times) > dt * 0.5 + 1e-8] = np.nan
    return sampled


def load_bi3_positions(path: Path, dt: float | None = None) -> tuple[str, np.ndarray, float] | None:
    with path.open("r", encoding="utf-8") as handle:
        timesteps = json.load(handle)
    if not isinstance(timesteps, list) or len(timesteps) < 2:
        return None
    timestamps: list[float] = []
    values: list[list[list[float]]] = []
    for timestep in timesteps:
        if not isinstance(timestep, dict):
            continue
        timestamp = _finite(timestep.get("time"))
        robot_state = timestep.get("robot_state")
        agent_states = timestep.get("agent_states")
        if timestamp is None:
            continue
        tracks = [[np.nan, np.nan] for _ in range(3)]
        if isinstance(robot_state, list) and len(robot_state) >= 2:
            tracks[0] = [_finite(robot_state[0]), _finite(robot_state[1])]
        agents = agent_states if isinstance(agent_states, list) else []
        for index, agent in enumerate(agents[:2], start=1):
            if not isinstance(agent, list) or len(agent) < 2:
                continue
            tracks[index] = [_finite(agent[0]), _finite(agent[1])]
        timestamps.append(timestamp)
        values.append(
            [
                [float(value) if value is not None else np.nan for value in track]
                for track in tracks
            ]
        )
    if len(values) < 2:
        return None
    order = np.argsort(np.asarray(timestamps, dtype=np.float64), kind="stable")
    times = np.asarray(timestamps, dtype=np.float64)[order]
    positions = np.asarray(values, dtype=np.float32)[order]
    unique, indices = np.unique(times, return_index=True)
    times = unique
    positions = positions[indices]
    dt_values = np.diff(times)
    dt_values = dt_values[dt_values > 1e-6]
    if not len(dt_values):
        return None
    if len(path.parents) >= 3 and path.parents[2].name == "jsons":
        sequence_id = "/".join((path.parents[1].name, path.parent.name, path.stem))
    else:
        sequence_id = path.stem
    if dt is not None:
        positions = _sample_bi3_positions(times, positions, dt)
        if len(positions) < 2:
            return None
    return sequence_id, positions, float(dt if dt is not None else np.median(dt_values))


def _split_navware(paths: list[Path]) -> dict[str, list[Path]]:
    ordered = sorted(paths, key=lambda path: path.stem)
    count = len(ordered)
    train_end = max(1, int(round(count * 0.6)))
    val_end = max(train_end + 1, int(round(count * 0.8)))
    return {"train": ordered[:train_end], "val": ordered[train_end:val_end], "test": ordered[val_end:]}


def _split_bi3(path: Path) -> str | None:
    parts = path.parts
    if len(parts) < 3:
        return None
    site = parts[-3]
    experiment = parts[-2]
    try:
        index = int(experiment.replace(site, ""))
    except ValueError:
        return None
    if site == "um":
        if 1 <= index <= 13:
            return "train"
        if 14 <= index <= 15:
            return "val"
        if 16 <= index <= 17:
            return "test"
    if site == "laas":
        if 1 <= index <= 14:
            return "train"
        if 15 <= index <= 17:
            return "val"
        if 18 <= index <= 20:
            return "test"
    return None


def _source_paths(dataset: str, root: Path) -> dict[str, list[Path]]:
    if dataset == "navware":
        paths = sorted(root.rglob("*_robot_and_participants.csv"))
        if not paths:
            paths = sorted(root.rglob("*.csv"))
        return _split_navware(paths)
    if dataset == "bi3":
        paths = _json_source_paths(root)
        result = {"train": [], "val": [], "test": []}
        for path in paths:
            split = _split_bi3(path)
            if split is not None:
                result[split].append(path)
        return result
    raise ValueError(f"unsupported real response dataset: {dataset}")


def _read_source(dataset: str, path: Path, dt: float | None) -> tuple[str, np.ndarray, float] | None:
    if dataset == "navware":
        return load_navware_positions(path)
    return load_bi3_positions(path, dt=0.4 if dt is None else dt)


def build_real_response_dataset(
    dataset: str,
    root: Path,
    output_dir: Path,
    *,
    history: int = 9,
    horizon: int = 12,
    windows: int = 8,
    execute_steps: int = 3,
    max_entities: int = 2,
    min_entities: int = 1,
    dt: float | None = None,
    episode_stride: int = 1,
    interaction_radius: float = 4.0,
    limit_recordings: int = 0,
) -> dict[str, Any]:
    if dt is not None and (not math.isfinite(dt) or dt <= 0):
        raise ValueError("dt must be positive and finite")
    split_paths = _source_paths(dataset, root)
    if limit_recordings > 0:
        split_paths = {
            split: paths[:limit_recordings] for split, paths in split_paths.items()
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset": dataset,
        "source_root": str(root),
        "protocol": "real-world-single-robot-action-response-v2" if dataset == "bi3" else "real-world-single-robot-action-response-v1",
        "history": history,
        "horizon": horizon,
        "windows": windows,
        "execute_steps": execute_steps,
        "max_entities": max_entities,
        "min_entities": min_entities,
        "requested_dt": dt,
        "sampling": "timestamp-nearest-with-gap-masking" if dataset == "bi3" else "source-rows",
        "episode_stride": episode_stride,
        "interaction_radius": interaction_radius,
        "claim_boundary": (
            "held-out real-recording robot-conditioned response prediction; "
            "external validation only, without counterfactual causal proof, "
            "persistent latent-mode ground truth, or closed-loop execution"
        ),
        "splits": {},
    }
    episode_offset = 0
    for split, paths in split_paths.items():
        records: list[dict[str, np.ndarray]] = []
        used: list[str] = []
        skipped: list[dict[str, str]] = []
        for path in paths:
            try:
                parsed = _read_source(dataset, path, dt)
                if parsed is None:
                    skipped.append({"path": str(path), "reason": "invalid_or_short_recording"})
                    continue
                sequence_id, positions, inferred_dt = parsed
                effective_dt = float(dt if dt is not None else inferred_dt)
                if dataset == "bi3" and dt is None:
                    effective_dt = 0.4
                record = build_sequence_episodes(
                    sequence_id,
                    positions,
                    dt=effective_dt,
                    history=history,
                    horizon=horizon,
                    windows=windows,
                    execute_steps=execute_steps,
                    max_entities=max_entities,
                    min_entities=min_entities,
                    episode_stride=episode_stride,
                    interaction_radius=interaction_radius,
                    episode_offset=episode_offset,
                )
                if record is None:
                    skipped.append({"path": str(path), "reason": "no_complete_episode"})
                    continue
                episode_offset += len(record["episode_id"])
                records.append(record)
                used.append(sequence_id)
            except Exception as error:
                skipped.append({"path": str(path), "reason": f"{type(error).__name__}: {error}"})
        if not records:
            raise ValueError(f"no usable {dataset} recordings in {split}")
        arrays = concatenate_episode_records(records)
        output_path = output_dir / f"{dataset}_{split}.npz"
        np.savez_compressed(output_path, **arrays)
        active = arrays["active_mask"].any(axis=-1)
        manifest = {
            "dataset": dataset,
            "split": split,
            "recordings_seen": len(paths),
            "recordings_used": len(used),
            "recording_ids": used,
            "skipped": skipped,
            "episodes": int(len(arrays["episode_id"])),
            "active_entity_episodes": int(active.sum()),
            "decision_windows": int(arrays["active_mask"].sum()),
            "output": str(output_path),
            "claim_boundary": summary["claim_boundary"],
        }
        output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        summary["splits"][split] = manifest
    summary["conversion_id"] = hashlib.sha256(json.dumps(summary, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    (output_dir / f"{dataset}_conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build real-world H1 response episodes.")
    parser.add_argument("--dataset", required=True, choices=["navware", "bi3"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--history", type=int, default=9)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--max-entities", type=int, default=2)
    parser.add_argument("--min-entities", type=int, default=1)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument("--interaction-radius", type=float, default=4.0)
    parser.add_argument("--limit-recordings", type=int, default=0)
    args = parser.parse_args()
    result = build_real_response_dataset(
        args.dataset,
        args.root,
        args.output_dir,
        history=args.history,
        horizon=args.horizon,
        windows=args.windows,
        execute_steps=args.execute_steps,
        max_entities=args.max_entities,
        min_entities=args.min_entities,
        dt=args.dt,
        episode_stride=args.episode_stride,
        interaction_radius=args.interaction_radius,
        limit_recordings=args.limit_recordings,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
