"""Extract and audit the complete THOR-MAGNI motion-capture CSV release.

The Zenodo release is a large ZIP.  ``download_thor_csv_parts`` is kept out
of the repository because the byte-range manifest is generated from the
Zenodo central directory; this runner consumes that manifest and validates
each local ZIP member before extracting it.  The audit is deliberately
streaming: it uses body centroids, robot proximity, gaze columns, roles, and
recording conditions without loading the 2.5 GB of CSV text into memory.

The resulting evidence is an external persistent-response compatibility
audit.  Helmet IDs are persistent within a recording, but the public release
does not provide a cross-recording participant key or paired same-state
counterfactual robot intervention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


LOCAL_HEADER = struct.Struct("<4s5H3L2H")
SCENARIO_CONDITIONS = {
    "Scenario_1_A": "static_robot_obstacle_without_lane_marking",
    "Scenario_1_B": "static_robot_obstacle_with_lane_marking",
    "Scenario_2": "static_robot_obstacle_object_transport",
    "Scenario_3_A": "moving_robot_differential_drive",
    "Scenario_3_B": "moving_robot_omnidirectional_drive",
    "Scenario_4_A": "hri_verbal_only",
    "Scenario_4_B": "hri_speech_gesture_gaze",
    "Scenario_5": "hri_proactive_help",
}


def _finite(value: str | None) -> float | None:
    if value is None or value == "" or value == "N/A":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_rows(manifest: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index, start, end, compressed_size, name, output = line.split("\t")
        rows.append(
            {
                "index": int(index),
                "zip_start": int(start),
                "zip_end": int(end),
                "compressed_size": int(compressed_size),
                "name": name,
                "part": Path(output),
            }
        )
    return rows


def extract_csv_parts(manifest: Path, output_root: Path, force: bool = False) -> list[dict[str, Any]]:
    """Validate each byte-range ZIP member and extract its CSV payload."""

    extracted: list[dict[str, Any]] = []
    for row in _manifest_rows(manifest):
        part = row["part"]
        if not part.exists():
            raise FileNotFoundError(part)
        blob = part.read_bytes()
        if len(blob) != row["zip_end"] - row["zip_start"] + 1:
            raise ValueError(f"incomplete ZIP member {part}: {len(blob)} bytes")
        fields = LOCAL_HEADER.unpack_from(blob, 0)
        signature, version, flags, method, modified_time, modified_date, crc32, compressed, uncompressed, name_len, extra_len = fields
        if signature != b"PK\x03\x04" or method != 8:
            raise ValueError(f"unsupported ZIP local header in {part}")
        name = blob[LOCAL_HEADER.size : LOCAL_HEADER.size + name_len].decode("utf-8")
        data_start = LOCAL_HEADER.size + name_len + extra_len
        compressed_data = blob[data_start:]
        if len(compressed_data) != row["compressed_size"]:
            raise ValueError(f"compressed-size mismatch for {part}: {len(compressed_data)}")
        payload = zlib.decompress(compressed_data, -15)
        if uncompressed and len(payload) != uncompressed:
            raise ValueError(f"uncompressed-size mismatch for {part}: {len(payload)}")
        target = output_root / Path(name).relative_to("THOR_MAGNI/CSVs_Scenarios")
        target.parent.mkdir(parents=True, exist_ok=True)
        if force or not target.exists() or target.stat().st_size != len(payload):
            target.write_bytes(payload)
        extracted.append(
            {
                "index": row["index"],
                "name": name,
                "path": str(target),
                "bytes": len(payload),
                "sha256": _sha256(target),
                "zip_member_sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    return extracted


def _metadata(rows: list[list[str]]) -> dict[str, list[str]]:
    return {row[0]: row[1:] for row in rows if row and row[0]}


def _condition(scenario: str, file_id: str) -> str:
    if scenario in {"Scenario_1", "Scenario_3", "Scenario_4"}:
        match = re.search(r"SC\d([AB])(?:_|$)", file_id)
        suffix = match.group(1) if match else "unknown"
        key = f"{scenario}_{suffix}"
    else:
        key = scenario
    return SCENARIO_CONDITIONS.get(key, "unknown")


def _body_specs(metadata: dict[str, list[str]]) -> list[tuple[str, str]]:
    names = metadata.get("BODY_NAMES", [])
    roles = metadata.get("BODY_ROLES", [])
    return [(name, roles[index] if index < len(roles) else "unknown") for index, name in enumerate(names) if name]


def _position_columns(header: list[str], body: str) -> tuple[int, int, int] | None:
    columns = {name: index for index, name in enumerate(header)}
    names = tuple(f"{body} Centroid_{axis}" for axis in ("X", "Y", "Z"))
    if not all(name in columns for name in names):
        return None
    return tuple(columns[name] for name in names)  # type: ignore[return-value]


def _gaze_columns(header: list[str], body: str) -> dict[str, tuple[int, ...]]:
    columns = {name: index for index, name in enumerate(header)}
    result: dict[str, tuple[int, ...]] = {}
    for label, suffixes in {
        "gaze_2d": ("G2D_X", "G2D_Y"),
        "gaze_3d": ("G3D_X", "G3D_Y", "G3D_Z"),
        "movement": ("Movement",),
    }.items():
        names = []
        for suffix in suffixes:
            matches = [
                name for name in header
                if name.startswith(f"{body} ") and name.endswith(suffix)
            ]
            if not matches:
                names = []
                break
            names.append(matches[0])
        if names:
            result[label] = tuple(columns[name] for name in names)
    return result


def _track_record(
    scenario: str,
    condition: str,
    file_id: str,
    body: str,
    role: str,
    stats: dict[str, Any],
) -> dict[str, Any]:
    duration = stats["last_time"] - stats["first_time"] if stats["last_time"] is not None else None
    displacement = None
    if stats["first_xy"] is not None and stats["last_xy"] is not None:
        displacement = float(np.linalg.norm(np.asarray(stats["last_xy"]) - np.asarray(stats["first_xy"])))
    return {
        "scenario": scenario,
        "condition": condition,
        "file_id": file_id,
        "body_id": body,
        "role": role,
        "samples": stats["samples"],
        "duration_s": duration,
        "path_length_m": stats["path_length"] / 1000.0,
        "displacement_m": displacement / 1000.0 if displacement is not None else None,
        "mean_speed_mps": (stats["path_length"] / 1000.0 / duration) if duration and duration > 0 else None,
        "min_robot_distance_m": stats["min_robot_distance"] / 1000.0 if stats["min_robot_distance"] is not None else None,
        "gaze_2d_valid": stats["gaze_2d_valid"],
        "gaze_3d_valid": stats["gaze_3d_valid"],
        "movement_label_valid": stats["movement_label_valid"],
        "missing_centroid_rows": stats["missing_centroid_rows"],
    }


def _recording_pair_key(file_id: str) -> str | None:
    match = re.match(r"(?P<date>\d+)_SC(?P<scenario>[134])(?P<condition>[AB])_(?P<run>R\d+)$", file_id)
    if not match:
        return None
    return f"{match.group('date')}_SC{match.group('scenario')}_{match.group('run')}"


def audit_csv(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        metadata_rows = [next(reader) for _ in range(16)]
        header = next(reader)
        metadata = _metadata(metadata_rows)
        file_id = str(metadata.get("FILE_ID", [path.stem])[0])
        scenario = path.parent.name
        condition = _condition(scenario, file_id)
        bodies = _body_specs(metadata)
        body_columns = {body: _position_columns(header, body) for body, _ in bodies}
        gaze_columns = {body: _gaze_columns(header, body) for body, _ in bodies}
        robot_columns = body_columns.get("DARKO_Robot")
        stats: dict[str, dict[str, Any]] = {}
        for body, role in bodies:
            stats[body] = {
                "samples": 0,
                "first_time": None,
                "last_time": None,
                "first_xy": None,
                "last_xy": None,
                "previous_xy": None,
                "previous_time": None,
                "path_length": 0.0,
                "min_robot_distance": None,
                "gaze_2d_valid": 0,
                "gaze_3d_valid": 0,
                "movement_label_valid": 0,
                "missing_centroid_rows": 0,
                "role": role,
            }
        data_rows = 0
        robot_valid = 0
        robot_motion: list[tuple[float, float]] = []
        time_index = header.index("Time") if "Time" in header else 1
        for row in reader:
            if len(row) < len(header):
                row.extend([""] * (len(header) - len(row)))
            time = _finite(row[time_index])
            if time is None:
                continue
            data_rows += 1
            robot_xy: tuple[float, float] | None = None
            if robot_columns is not None:
                robot_values = [_finite(row[index]) for index in robot_columns]
                if robot_values[0] is not None and robot_values[1] is not None:
                    robot_xy = (robot_values[0], robot_values[1])
                    robot_valid += 1
                    robot_motion.append(robot_xy)
            for body, role in bodies:
                if body == "DARKO_Robot":
                    continue
                record = stats[body]
                columns = body_columns.get(body)
                if columns is None:
                    record["missing_centroid_rows"] += 1
                    continue
                values = [_finite(row[index]) for index in columns]
                if values[0] is None or values[1] is None:
                    record["missing_centroid_rows"] += 1
                else:
                    xy = (values[0], values[1])
                    record["samples"] += 1
                    if record["first_time"] is None:
                        record["first_time"] = time
                        record["first_xy"] = xy
                    if record["previous_xy"] is not None and record["previous_time"] is not None:
                        record["path_length"] += float(np.linalg.norm(np.asarray(xy) - np.asarray(record["previous_xy"])))
                    record["previous_xy"] = xy
                    record["previous_time"] = time
                    record["last_time"] = time
                    record["last_xy"] = xy
                    if robot_xy is not None:
                        distance = float(np.linalg.norm(np.asarray(xy) - np.asarray(robot_xy)))
                        if record["min_robot_distance"] is None or distance < record["min_robot_distance"]:
                            record["min_robot_distance"] = distance
                gaze = gaze_columns.get(body, {})
                if "gaze_2d" in gaze and all(_finite(row[index]) is not None for index in gaze["gaze_2d"]):
                    record["gaze_2d_valid"] += 1
                if "gaze_3d" in gaze and all(_finite(row[index]) is not None for index in gaze["gaze_3d"]):
                    record["gaze_3d_valid"] += 1
                if "movement" in gaze and row[gaze["movement"][0]] not in {"", "N/A"}:
                    record["movement_label_valid"] += 1
        tracks = [_track_record(scenario, condition, file_id, body, role, values) for (body, role), values in zip(bodies, stats.values()) if body != "DARKO_Robot" and values["samples"] > 0]
    robot_displacement = None
    if len(robot_motion) >= 2:
        robot_displacement = float(np.linalg.norm(np.asarray(robot_motion[-1]) - np.asarray(robot_motion[0]))) / 1000.0
    episode = {
        "scenario": scenario,
        "condition": condition,
        "file_id": file_id,
        "path": str(path),
        "data_rows": data_rows,
        "body_count": len(bodies),
        "human_bodies_with_position": len(tracks),
        "robot_position_valid_rows": robot_valid,
        "robot_position_coverage": robot_valid / data_rows if data_rows else 0.0,
        "robot_displacement_m": robot_displacement,
        "recording_pair_key": _recording_pair_key(file_id),
        "mean_track_path_length_m": float(np.mean([track["path_length_m"] for track in tracks])) if tracks else None,
        "mean_track_displacement_m": float(np.mean([track["displacement_m"] for track in tracks if track["displacement_m"] is not None])) if any(track["displacement_m"] is not None for track in tracks) else None,
        "mean_track_speed_mps": float(np.mean([track["mean_speed_mps"] for track in tracks if track["mean_speed_mps"] is not None])) if any(track["mean_speed_mps"] is not None for track in tracks) else None,
        "mean_track_min_robot_distance_m": float(np.mean([track["min_robot_distance_m"] for track in tracks if track["min_robot_distance_m"] is not None])) if any(track["min_robot_distance_m"] is not None for track in tracks) else None,
        "gaze_2d_valid_rows": int(sum(track["gaze_2d_valid"] for track in tracks)),
        "gaze_3d_valid_rows": int(sum(track["gaze_3d_valid"] for track in tracks)),
        "body_roles": dict(Counter(role for _, role in bodies)),
        "metadata": {
            "n_frames_qtm": metadata.get("N_FRAMES_QTM", [None])[0],
            "eyetracking_devices": metadata.get("EYETRACKING_DEVICES", []),
            "eyetracking_data_included": metadata.get("EYETRACKING_DATA_INCLUDED", []),
        },
    }
    return episode, tracks


def _aggregate_tracks(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in tracks:
        grouped[track["condition"]].append(track)
    result: dict[str, Any] = {}
    for condition, values in sorted(grouped.items()):
        def mean(key: str) -> float | None:
            numbers = [float(item[key]) for item in values if item.get(key) is not None and math.isfinite(float(item[key]))]
            return float(np.mean(numbers)) if numbers else None
        result[condition] = {
            "tracks": len(values),
            "episodes": len({item["file_id"] for item in values}),
            "mean_path_length_m": mean("path_length_m"),
            "mean_displacement_m": mean("displacement_m"),
            "mean_speed_mps": mean("mean_speed_mps"),
            "mean_min_robot_distance_m": mean("min_robot_distance_m"),
            "gaze_2d_valid": int(sum(item["gaze_2d_valid"] for item in values)),
            "gaze_3d_valid": int(sum(item["gaze_3d_valid"] for item in values)),
        }
    return result


def _condition_contrasts(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    contrasts = {
        "scenario_3_omni_minus_differential": (
            "moving_robot_differential_drive",
            "moving_robot_omnidirectional_drive",
        ),
        "scenario_4_speech_gesture_gaze_minus_verbal": (
            "hri_verbal_only",
            "hri_speech_gesture_gaze",
        ),
    }
    output: dict[str, Any] = {}
    for name, (condition_a, condition_b) in contrasts.items():
        by_key: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for episode in episodes:
            key = episode.get("recording_pair_key")
            if key and episode["condition"] in {condition_a, condition_b}:
                by_key[key][episode["condition"]] = episode
        pairs = [values for values in by_key.values() if condition_a in values and condition_b in values]
        metrics = ("mean_track_path_length_m", "mean_track_displacement_m", "mean_track_speed_mps", "mean_track_min_robot_distance_m")
        summary: dict[str, Any] = {"paired_recordings": len(pairs), "condition_a": condition_a, "condition_b": condition_b, "metrics": {}}
        for metric in metrics:
            deltas = [
                float(pair[condition_b][metric]) - float(pair[condition_a][metric])
                for pair in pairs
                if pair[condition_a].get(metric) is not None and pair[condition_b].get(metric) is not None
            ]
            if deltas:
                rng = np.random.default_rng(20260825 + len(metric))
                values = np.asarray(deltas, dtype=np.float64)
                bootstrap = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
                signs = rng.choice(np.asarray([-1.0, 1.0]), size=(20000, len(values)))
                null = np.abs((signs * values[None]).mean(axis=1))
                pvalue = float((1.0 + np.sum(null >= abs(values.mean()))) / 20001.0)
                summary["metrics"][metric] = {
                    "n": len(deltas),
                    "mean_delta_b_minus_a": float(values.mean()),
                    "bootstrap_95ci_mean_delta": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
                    "sign_flip_pvalue": pvalue,
                }
            else:
                summary["metrics"][metric] = {"n": 0, "mean_delta_b_minus_a": None}
        output[name] = summary
    return output


def _action_coverage(action_csv: Path, raw_files: Iterable[str]) -> dict[str, Any]:
    raw_names = {Path(name).name for name in raw_files}
    if not action_csv.exists():
        return {"status": "missing", "path": str(action_csv), "raw_files_with_actions": 0}
    frame_names: set[str] = set()
    actions: Counter[str] = Counter()
    rows = 0
    with action_csv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            name = str(row.get("file_name", ""))
            frame_names.add(name)
            action = str(row.get("action", ""))
            if action:
                actions[action] += 1
    matched = sorted(raw_names & frame_names)
    return {
        "status": "partial_action_release",
        "path": str(action_csv),
        "rows": rows,
        "raw_recordings": len(raw_names),
        "recordings_with_action_labels": len(matched),
        "coverage": len(matched) / len(raw_names) if raw_names else 0.0,
        "action_classes": len(actions),
        "action_counts": dict(actions.most_common()),
        "matched_files": matched,
        "claim_boundary": "action labels are available for the public processed Scenario 2/3 subset only; no action label is inferred for the remaining raw recordings",
    }


def run(
    manifest: Path,
    extracted_root: Path,
    output_dir: Path,
    action_csv: Path | None = None,
    force_extract: bool = False,
) -> dict[str, Any]:
    extraction = extract_csv_parts(manifest, extracted_root, force=force_extract)
    episodes: list[dict[str, Any]] = []
    tracks: list[dict[str, Any]] = []
    for item in extraction:
        episode, episode_tracks = audit_csv(Path(item["path"]))
        episode["source_sha256"] = item["sha256"]
        episodes.append(episode)
        tracks.extend(episode_tracks)
    action = _action_coverage(action_csv, (item["path"] for item in episodes)) if action_csv else {"status": "not_requested"}
    payload: dict[str, Any] = {
        "status": "complete",
        "dataset": "THÖR-MAGNI full public motion-capture CSV release",
        "source": "https://zenodo.org/records/13865754",
        "protocol": {
            "recordings": len(episodes),
            "scenarios": sorted({episode["scenario"] for episode in episodes}),
            "condition_mapping": SCENARIO_CONDITIONS,
            "identity_scope": "helmet identity is persistent within each recording; cross-recording participant identity is not published in the consumed CSV release",
            "claim_boundary": "full motion-capture, robot-condition, role, gaze, and action-coverage audit; no causal same-state intervention claim",
        },
        "summary": {
            "recordings": len(episodes),
            "data_rows": int(sum(episode["data_rows"] for episode in episodes)),
            "tracks": len(tracks),
            "roles": dict(Counter(track["role"] for track in tracks)),
            "conditions": dict(Counter(episode["condition"] for episode in episodes)),
            "robot_coverage_mean": float(np.mean([episode["robot_position_coverage"] for episode in episodes])) if episodes else None,
            "gaze_2d_valid_rows": int(sum(episode["gaze_2d_valid_rows"] for episode in episodes)),
            "gaze_3d_valid_rows": int(sum(episode["gaze_3d_valid_rows"] for episode in episodes)),
            "condition_track_summary": _aggregate_tracks(tracks),
            "condition_contrasts": _condition_contrasts(episodes),
        },
        "action_coverage": action,
        "episodes": episodes,
        "tracks": tracks,
        "extracted_members": extraction,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(episodes[0]) if episodes else []
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(episodes)
    with (output_dir / "tracks.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(tracks[0]) if tracks else []
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(tracks)
    lines = [
        "# THÖR-MAGNI Full Audit",
        "",
        f"Recordings: {len(episodes)}; rows: {payload['summary']['data_rows']}; tracks: {len(tracks)}.",
        "",
        "The audit uses the complete public CSV motion-capture release. Conditions are mapped from the dataset documentation: Scenario 3 A/B are differential-drive versus omni-directional moving robot, and Scenario 4 A/B are verbal-only versus speech+gesture+gaze HRI.",
        "",
        "Helmet IDs are persistent within a recording. The consumed public release does not provide a cross-recording participant key, so the result is not a longitudinal person-identity claim.",
        "",
        f"Action coverage: {action.get('recordings_with_action_labels', 0)}/{len(episodes)} recordings; no action labels are inferred for uncovered raw recordings.",
        "",
        "See summary.json for per-recording gaze validity, robot coverage, role counts, track metrics, and source hashes.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/tmp/thor_csv_ranges.tsv")
    parser.add_argument("--extracted-root", default="raw_data/external/thor_magni_full/csvs")
    parser.add_argument("--action-csv", default="raw_data/external/thor_magni_merged_data.csv")
    parser.add_argument("--output-dir", default="reports/external_validation_20260825/thor_magni_full")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()
    payload = run(Path(args.manifest), Path(args.extracted_root), Path(args.output_dir), Path(args.action_csv), args.force_extract)
    print(f"status={payload['status']} recordings={payload['summary']['recordings']} rows={payload['summary']['data_rows']} tracks={payload['summary']['tracks']}")
    print(f"action_coverage={payload['action_coverage'].get('recordings_with_action_labels', 0)}/{payload['summary']['recordings']}")


if __name__ == "__main__":
    main()
