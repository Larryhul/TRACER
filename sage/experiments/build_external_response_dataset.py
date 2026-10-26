"""Build track-continuous response/belief episodes from external recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sage.data.external_response import (
    build_sequence_episodes,
    concatenate_episode_records,
)
from sage.experiments.build_external_sage_dataset import (
    _read_sequence,
    _recording_paths,
    _split_paths,
)


def build_external_response_dataset(
    dataset: str,
    root: Path,
    output_dir: Path,
    *,
    history: int = 8,
    horizon: int = 12,
    windows: int = 8,
    execute_steps: int = 3,
    max_entities: int = 4,
    min_entities: int = 2,
    dt: float = 0.1,
    episode_stride: int = 24,
    interaction_radius: float = 4.0,
    limit_recordings: int = 0,
) -> dict[str, Any]:
    if dataset not in {"thor_magni", "huron_sacson"}:
        raise ValueError("response-belief episodes require recorded robot motion")
    paths = _recording_paths(dataset, root)
    if limit_recordings > 0:
        paths = paths[:limit_recordings]
    split_paths = _split_paths(paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset": dataset,
        "source_root": str(root),
        "protocol": "observational-single-robot-response-belief-v1",
        "history": history,
        "horizon": horizon,
        "windows": windows,
        "execute_steps": execute_steps,
        "max_entities": max_entities,
        "min_entities": min_entities,
        "dt": dt,
        "episode_stride": episode_stride,
        "interaction_radius": interaction_radius,
        "claim_boundary": (
            "held-out-recording action-conditioned response prediction and "
            "recording-local identity continuity only; no multi-robot "
            "counterfactual, causal, or long-term human claim"
        ),
        "splits": {},
    }
    episode_offset = 0
    for split, files in split_paths.items():
        split_records: list[dict[str, np.ndarray]] = []
        used: list[str] = []
        skipped: list[dict[str, str]] = []
        for path in files:
            try:
                parsed = _read_sequence(dataset, path, dt)
                if parsed is None:
                    skipped.append(
                        {"path": str(path), "reason": "insufficient_or_invalid_track"}
                    )
                    continue
                sequence_id, positions, _ = parsed
                record = build_sequence_episodes(
                    sequence_id,
                    positions,
                    dt=dt,
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
                    skipped.append(
                        {"path": str(path), "reason": "no_complete_track_episode"}
                    )
                    continue
                episode_offset += len(record["episode_id"])
                split_records.append(record)
                used.append(sequence_id)
            except Exception as error:
                skipped.append(
                    {"path": str(path), "reason": f"{type(error).__name__}: {error}"}
                )
        arrays = concatenate_episode_records(split_records)
        output_path = output_dir / f"{dataset}_{split}.npz"
        np.savez_compressed(output_path, **arrays)
        active = arrays["active_mask"].any(axis=-1)
        manifest = {
            "dataset": dataset,
            "split": split,
            "recordings_seen": len(files),
            "recordings_used": len(used),
            "recording_ids": used,
            "skipped": skipped,
            "episodes": int(len(arrays["episode_id"])),
            "active_entity_episodes": int(active.sum()),
            "decision_windows": int(arrays["active_mask"].sum()),
            "output": str(output_path),
            "claim_boundary": summary["claim_boundary"],
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        summary["splits"][split] = manifest
    summary["conversion_id"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    (output_dir / f"{dataset}_conversion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build external response and persistent-belief episodes."
    )
    parser.add_argument(
        "--dataset", required=True, choices=["thor_magni", "huron_sacson"]
    )
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--output-dir", default="experiments/external_response_data"
    )
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--max-entities", type=int, default=4)
    parser.add_argument("--min-entities", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--episode-stride", type=int, default=24)
    parser.add_argument("--interaction-radius", type=float, default=4.0)
    parser.add_argument("--limit-recordings", type=int, default=0)
    args = parser.parse_args()
    result = build_external_response_dataset(
        args.dataset,
        Path(args.root),
        Path(args.output_dir),
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
    for split, record in result["splits"].items():
        print(
            f"{args.dataset} {split}: recordings={record['recordings_used']} "
            f"episodes={record['episodes']} entities={record['active_entity_episodes']}",
            flush=True,
        )
    print(f"conversion_id={result['conversion_id']}", flush=True)


if __name__ == "__main__":
    main()
