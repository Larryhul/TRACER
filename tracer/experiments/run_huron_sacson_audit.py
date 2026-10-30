"""Audit the public HuRoN/SACSoN ROS1 bags for action-conditioned response.

HuRoN provides synchronized robot odometry, pedestrian detections, and laser
scans, but it does not provide the matched candidate-action intervention used
by Formal-v1.  This script therefore reports an action-conditioned descriptive
audit only.  Marker IDs are recording-local and are never treated as
cross-recording participant identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    Reader = None  # type: ignore[assignment,misc]
    Stores = None  # type: ignore[assignment,misc]
    get_typestore = None  # type: ignore[assignment]


SOURCE_ROOT = "https://rail.eecs.berkeley.edu/datasets/huron/"
ROBOT_TOPICS = ("/odometry",)
PEDESTRIAN_TOPICS = ("/pedestrians_pose",)
LASER_TOPICS = ("/laserscan",)
RECORDING_RE = re.compile(r"^(?P<condition>.+)_(?P<bag>\d{8}\.bag)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[float]) -> float | None:
    finite = [number for value in values if (number := _finite(value)) is not None]
    return float(np.mean(finite)) if finite else None


def _median(values: Iterable[float]) -> float | None:
    finite = [number for value in values if (number := _finite(value)) is not None]
    return float(np.median(finite)) if finite else None


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    finite = [number for value in values if (number := _finite(value)) is not None]
    return float(np.percentile(finite, percentile)) if finite else None


def _condition_fields(condition: str) -> dict[str, Any]:
    interaction_loss = condition.endswith("-intloss")
    base = condition.removesuffix("-intloss")
    parts = base.split("-")
    date = "-".join(parts[:3]) if len(parts) >= 3 else base
    robot = "-".join(parts[3:]) if len(parts) >= 4 else "unknown"
    return {
        "condition": condition,
        "date": date,
        "robot_platform": robot,
        "interaction_loss_condition": interaction_loss,
    }


def _distance_bin(distance: float | None) -> str:
    if distance is None:
        return "no_pedestrian_detection"
    if distance < 1.0:
        return "near_lt_1m"
    if distance < 2.0:
        return "mid_1_to_2m"
    if distance < 3.0:
        return "far_2_to_3m"
    return "over_3m"


def _motion_bin(pose_speed: float | None) -> str:
    if pose_speed is None:
        return "unknown_motion"
    if pose_speed < 0.05:
        return "stationary_lt_0.05mps"
    if pose_speed < 0.5:
        return "slow_0.05_to_0.5mps"
    return "moving_ge_0.5mps"


def _trajectory_metrics(times_ns: np.ndarray, xy: np.ndarray, linear: np.ndarray, angular: np.ndarray) -> dict[str, Any]:
    if len(times_ns) < 2:
        return {"odom_samples": int(len(times_ns)), "trajectory_status": "too_short"}
    dt = np.diff(times_ns.astype(np.float64)) / 1e9
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    valid = dt > 0
    duration = float((times_ns[-1] - times_ns[0]) / 1e9)
    pose_speed = steps[valid] / dt[valid] if np.any(valid) else np.asarray([], dtype=np.float64)
    quality_flag = "ok"
    if len(pose_speed) and float(np.max(pose_speed)) > 6.0:
        quality_flag = "suspect_odometry_jump"
    return {
        "odom_samples": int(len(times_ns)),
        "trajectory_status": "complete",
        "duration_s": duration if duration > 0 else None,
        "sampling_hz": float(1.0 / np.median(dt[valid])) if np.any(valid) else None,
        "path_length_m": float(np.sum(steps)),
        "displacement_m": float(np.linalg.norm(xy[-1] - xy[0])),
        "mean_pose_speed_mps": float(np.mean(pose_speed)) if len(pose_speed) else None,
        "mean_abs_linear_x_mps": _mean(np.abs(linear)),
        "mean_abs_angular_z_rps": _mean(np.abs(angular)),
        "max_abs_linear_x_mps": float(np.max(np.abs(linear))) if len(linear) else None,
        "max_abs_angular_z_rps": float(np.max(np.abs(angular))) if len(angular) else None,
        "raw_twist_quality_flag": (
            "raw_twist_outlier"
            if (len(linear) and float(np.max(np.abs(linear))) > 2.0)
            or (len(angular) and float(np.max(np.abs(angular))) > 4.0)
            else "within_audit_range"
        ),
        "quality_flag": quality_flag,
    }


def _synchronise_response(
    odom_times_ns: np.ndarray,
    pose_speed: np.ndarray,
    linear: np.ndarray,
    angular: np.ndarray,
    marker_times_ns: np.ndarray,
    nearest_distance: np.ndarray,
    marker_counts: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not len(marker_times_ns) or not len(odom_times_ns):
        return [], {"status": "no_synchronised_observations", "bins": {}}
    indices = np.searchsorted(odom_times_ns, marker_times_ns, side="right") - 1
    indices = np.clip(indices, 0, len(odom_times_ns) - 1)
    rows: list[dict[str, Any]] = []
    for index, distance, count in zip(indices, nearest_distance, marker_counts):
        distance_value = _finite(distance)
        rows.append(
            {
                "distance_bin": _distance_bin(distance_value),
                "robot_motion_bin": _motion_bin(float(pose_speed[index])),
                "nearest_pedestrian_distance_m": distance_value,
                "pedestrian_count": int(count),
                "pose_speed_mps": float(pose_speed[index]),
                "linear_x_mps": float(linear[index]),
                "angular_z_rps": float(angular[index]),
            }
        )
    bins: dict[str, Any] = {}
    for name in sorted({row["distance_bin"] for row in rows}):
        subset = [row for row in rows if row["distance_bin"] == name]
        bins[name] = {
            "observations": len(subset),
            "fraction": float(len(subset) / len(rows)),
            "mean_nearest_distance_m": _mean(
                row["nearest_pedestrian_distance_m"]
                for row in subset
                if row["nearest_pedestrian_distance_m"] is not None
            ),
            "mean_pedestrian_count": _mean(row["pedestrian_count"] for row in subset),
            "mean_pose_speed_mps": _mean(row["pose_speed_mps"] for row in subset),
            "median_pose_speed_mps": _median(row["pose_speed_mps"] for row in subset),
            "p95_pose_speed_mps": _percentile((row["pose_speed_mps"] for row in subset), 95),
            "mean_linear_x_mps": _mean(row["linear_x_mps"] for row in subset),
            "median_linear_x_mps": _median(row["linear_x_mps"] for row in subset),
            "p95_abs_linear_x_mps": _percentile((abs(row["linear_x_mps"]) for row in subset), 95),
            "mean_abs_linear_x_mps": _mean(abs(row["linear_x_mps"]) for row in subset),
            "median_abs_angular_z_rps": _median(abs(row["angular_z_rps"]) for row in subset),
            "p95_abs_angular_z_rps": _percentile((abs(row["angular_z_rps"]) for row in subset), 95),
            "mean_abs_angular_z_rps": _mean(abs(row["angular_z_rps"]) for row in subset),
        }
    return rows, {"status": "complete", "observations": len(rows), "bins": bins}


def _human_response_audit(
    marker_observations: list[tuple[int, int, float, float, float]],
    odom_times_ns: np.ndarray,
    pose_speed: np.ndarray,
    linear: np.ndarray,
    angular: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Estimate short within-recording pedestrian motion in robot base frame."""

    if not marker_observations or not len(odom_times_ns):
        return [], {"status": "no_marker_motion_observations", "bins": {}}
    by_marker: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    for timestamp_ns, marker_id, x, y, distance in marker_observations:
        by_marker[marker_id].append((timestamp_ns, x, y, distance))
    rows: list[dict[str, Any]] = []
    for marker_id, observations in sorted(by_marker.items()):
        observations.sort(key=lambda item: item[0])
        for previous, current in zip(observations, observations[1:]):
            previous_time, previous_x, previous_y, previous_distance = previous
            timestamp_ns, x, y, distance = current
            dt = (timestamp_ns - previous_time) / 1e9
            if dt <= 0.0 or dt > 2.0:
                continue
            relative_speed = math.hypot(x - previous_x, y - previous_y) / dt
            radial_velocity = (distance - previous_distance) / dt
            index = int(np.searchsorted(odom_times_ns, timestamp_ns, side="right") - 1)
            index = int(np.clip(index, 0, len(odom_times_ns) - 1))
            rows.append(
                {
                    "marker_id": int(marker_id),
                    "timestamp_ns": int(timestamp_ns),
                    "distance_m": float(distance),
                    "distance_bin": _distance_bin(distance),
                    "robot_motion_bin": _motion_bin(float(pose_speed[index])),
                    "relative_speed_mps": float(relative_speed),
                    "radial_velocity_mps": float(radial_velocity),
                    "quality_flag": (
                        "suspect_marker_jump"
                        if relative_speed > 10.0 or abs(radial_velocity) > 10.0
                        else "ok"
                    ),
                    "pose_speed_mps": float(pose_speed[index]),
                    "linear_x_mps": float(linear[index]),
                    "angular_z_rps": float(angular[index]),
                }
            )
    quality_excluded = [row for row in rows if row["quality_flag"] != "ok"]
    primary_rows = [row for row in rows if row["quality_flag"] == "ok"]
    bins: dict[str, Any] = {}
    for name in sorted({row["distance_bin"] for row in primary_rows}):
        subset = [row for row in primary_rows if row["distance_bin"] == name]
        bins[name] = {
            "observations": len(subset),
            "marker_tracks": len({row["marker_id"] for row in subset}),
            "mean_relative_speed_mps": _mean(row["relative_speed_mps"] for row in subset),
            "median_relative_speed_mps": _median(row["relative_speed_mps"] for row in subset),
            "p95_relative_speed_mps": _percentile((row["relative_speed_mps"] for row in subset), 95),
            "mean_radial_velocity_mps": _mean(row["radial_velocity_mps"] for row in subset),
            "median_radial_velocity_mps": _median(row["radial_velocity_mps"] for row in subset),
            "approaching_fraction": _mean(row["radial_velocity_mps"] < 0.0 for row in subset),
            "mean_pose_speed_mps": _mean(row["pose_speed_mps"] for row in subset),
            "moving_fraction": _mean(row["robot_motion_bin"] == "moving_ge_0.5mps" for row in subset),
        }
    motion_bins: dict[str, Any] = {}
    for name in sorted({row["robot_motion_bin"] for row in primary_rows}):
        subset = [row for row in primary_rows if row["robot_motion_bin"] == name]
        motion_bins[name] = {
            "observations": len(subset),
            "mean_distance_m": _mean(row["distance_m"] for row in subset),
            "mean_relative_speed_mps": _mean(row["relative_speed_mps"] for row in subset),
            "median_relative_speed_mps": _median(row["relative_speed_mps"] for row in subset),
            "mean_radial_velocity_mps": _mean(row["radial_velocity_mps"] for row in subset),
            "approaching_fraction": _mean(row["radial_velocity_mps"] < 0.0 for row in subset),
        }
    return rows, {
        "status": "complete" if rows else "no_marker_motion_observations",
        "observations": len(rows),
        "primary_observations": len(primary_rows),
        "quality_excluded_observations": len(quality_excluded),
        "marker_tracks": len(by_marker),
        "bins": bins,
        "robot_motion_bins": motion_bins,
    }


def _read_bag(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if Reader is None or Stores is None or get_typestore is None:
        return {"status": "missing_rosbags_dependency", "file": str(path)}, [], []
    match = RECORDING_RE.match(path.name)
    if not match:
        return {"status": "unrecognised_filename", "file": str(path)}, [], []
    fields = _condition_fields(match.group("condition"))
    fields.update(
        {
            "status": "invalid",
            "file": str(path),
            "source_url": f"{SOURCE_ROOT}{match.group('condition')}/{match.group('bag')}",
            "source_sha256": _sha256(path),
            "file_bytes": path.stat().st_size,
        }
    )
    typestore = get_typestore(Stores.ROS1_NOETIC)
    odom_times: list[int] = []
    odom_xy: list[tuple[float, float]] = []
    linear: list[float] = []
    angular: list[float] = []
    marker_times: list[int] = []
    nearest_distance: list[float] = []
    marker_counts: list[int] = []
    marker_observations: list[tuple[int, int, float, float, float]] = []
    laser_messages = 0
    pedestrian_ids: set[int] = set()
    try:
        with Reader(path) as reader:
            connections = {connection.topic: connection for connection in reader.connections}
            selected = [
                connection
                for topic in (*ROBOT_TOPICS, *PEDESTRIAN_TOPICS, *LASER_TOPICS)
                if (connection := connections.get(topic)) is not None
            ]
            if not selected:
                fields["status"] = "no_required_topics"
                return fields, [], []
            for connection, timestamp_ns, raw in reader.messages(connections=selected):
                if connection.topic == "/odometry":
                    message = typestore.deserialize_ros1(raw, connection.msgtype)
                    position = message.pose.pose.position
                    x = _finite(position.x)
                    y = _finite(position.y)
                    if x is None or y is None:
                        continue
                    odom_times.append(int(timestamp_ns))
                    odom_xy.append((x, y))
                    linear.append(_finite(message.twist.twist.linear.x) or 0.0)
                    angular.append(_finite(message.twist.twist.angular.z) or 0.0)
                elif connection.topic == "/pedestrians_pose":
                    message = typestore.deserialize_ros1(raw, connection.msgtype)
                    distances: list[float] = []
                    for marker in message.markers:
                        if int(marker.action) == 2:
                            continue
                        x = _finite(marker.pose.position.x)
                        y = _finite(marker.pose.position.y)
                        if x is None or y is None:
                            continue
                        pedestrian_ids.add(int(marker.id))
                        distance = float(math.hypot(x, y))
                        distances.append(distance)
                        marker_observations.append((int(timestamp_ns), int(marker.id), x, y, distance))
                    marker_times.append(int(timestamp_ns))
                    nearest_distance.append(min(distances) if distances else float("nan"))
                    marker_counts.append(len(distances))
                elif connection.topic == "/laserscan":
                    laser_messages += 1
    except Exception as error:  # Keep one corrupt bag from hiding the valid subset.
        fields.update({"status": "parse_error", "error": f"{type(error).__name__}: {error}"})
        return fields, [], []
    odom_time_array = np.asarray(odom_times, dtype=np.int64)
    order = np.argsort(odom_time_array)
    odom_time_array = odom_time_array[order]
    xy_array = np.asarray(odom_xy, dtype=np.float64)[order] if odom_xy else np.empty((0, 2))
    linear_array = np.asarray(linear, dtype=np.float64)[order]
    angular_array = np.asarray(angular, dtype=np.float64)[order]
    pose_steps = np.linalg.norm(np.diff(xy_array, axis=0), axis=1) if len(xy_array) > 1 else np.asarray([])
    pose_dt = np.diff(odom_time_array.astype(np.float64)) / 1e9 if len(odom_time_array) > 1 else np.asarray([])
    pose_speed_array = np.zeros(len(odom_time_array), dtype=np.float64)
    valid_pose_dt = pose_dt > 0
    if np.any(valid_pose_dt):
        valid_indices = np.flatnonzero(valid_pose_dt) + 1
        pose_speed_array[valid_indices] = pose_steps[valid_pose_dt] / pose_dt[valid_pose_dt]
    marker_time_array = np.asarray(marker_times, dtype=np.int64)
    marker_order = np.argsort(marker_time_array)
    marker_time_array = marker_time_array[marker_order]
    nearest_array = np.asarray(nearest_distance, dtype=np.float64)[marker_order]
    count_array = np.asarray(marker_counts, dtype=np.int64)[marker_order]
    response_rows, response = _synchronise_response(
        odom_time_array,
        pose_speed_array,
        linear_array,
        angular_array,
        marker_time_array,
        nearest_array,
        count_array,
    )
    human_response_rows, human_response = _human_response_audit(
        marker_observations,
        odom_time_array,
        pose_speed_array,
        linear_array,
        angular_array,
    )
    fields.update(_trajectory_metrics(odom_time_array, xy_array, linear_array, angular_array))
    fields.update(
        {
            "status": "complete",
            "pedestrian_messages": int(len(marker_time_array)),
            "pedestrian_observations_with_detection": int(np.sum(np.isfinite(nearest_array))),
            "recording_local_marker_ids": len(pedestrian_ids),
            "laser_messages": laser_messages,
            "response_audit": response,
            "human_response_audit": human_response,
        }
    )
    for row in response_rows:
        row.update(
            {
                "condition": fields["condition"],
                "robot_platform": fields["robot_platform"],
                "interaction_loss_condition": fields["interaction_loss_condition"],
                "file": str(path),
            }
        )
    for row in human_response_rows:
        row.update(
            {
                "condition": fields["condition"],
                "robot_platform": fields["robot_platform"],
                "interaction_loss_condition": fields["interaction_loss_condition"],
                "file": str(path),
            }
        )
    return fields, response_rows, human_response_rows


def audit_huron_sacson(root: Path) -> dict[str, Any]:
    bags = sorted(root.rglob("*.bag"))
    if Reader is None:
        return {
            "status": "blocked_missing_rosbags_dependency",
            "dataset": "HuRoN/SACSoN",
            "local_root": str(root),
            "bags_seen": len(bags),
            "claim_boundary": "no result until ROS1 bag parser is installed",
        }
    records: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    human_response_rows: list[dict[str, Any]] = []
    for path in bags:
        record, rows, human_rows = _read_bag(path)
        records.append(record)
        response_rows.extend(rows)
        human_response_rows.extend(human_rows)
    complete = [record for record in records if record.get("status") == "complete"]
    by_condition: dict[str, Any] = {}
    for condition in sorted({record.get("condition") for record in complete if record.get("condition")}):
        subset = [record for record in complete if record.get("condition") == condition]
        bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in response_rows:
            if row["condition"] == condition:
                bins[row["distance_bin"]].append(row)
        human_bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
        human_motion_bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
        human_all_rows = [row for row in human_response_rows if row["condition"] == condition]
        human_primary_rows = [row for row in human_all_rows if row.get("quality_flag") == "ok"]
        for row in human_primary_rows:
            human_bins[row["distance_bin"]].append(row)
            human_motion_bins[row["robot_motion_bin"]].append(row)
        by_condition[condition] = {
            "bags": len(subset),
            "robot_platform": subset[0].get("robot_platform"),
            "interaction_loss_condition": subset[0].get("interaction_loss_condition"),
            "mean_path_length_m": _mean(record.get("path_length_m") for record in subset if record.get("path_length_m") is not None),
            "mean_displacement_m": _mean(record.get("displacement_m") for record in subset if record.get("displacement_m") is not None),
            "mean_duration_s": _mean(record.get("duration_s") for record in subset if record.get("duration_s") is not None),
            "response_bins": {
                name: {
                    "observations": len(values),
                    "mean_pose_speed_mps": _mean(row["pose_speed_mps"] for row in values),
                    "median_pose_speed_mps": _median(row["pose_speed_mps"] for row in values),
                    "p95_pose_speed_mps": _percentile((row["pose_speed_mps"] for row in values), 95),
                    "mean_abs_linear_x_mps": _mean(abs(row["linear_x_mps"]) for row in values),
                    "median_abs_linear_x_mps": _median(abs(row["linear_x_mps"]) for row in values),
                    "p95_abs_linear_x_mps": _percentile((abs(row["linear_x_mps"]) for row in values), 95),
                    "mean_abs_angular_z_rps": _mean(abs(row["angular_z_rps"]) for row in values),
                    "median_abs_angular_z_rps": _median(abs(row["angular_z_rps"]) for row in values),
                    "p95_abs_angular_z_rps": _percentile((abs(row["angular_z_rps"]) for row in values), 95),
                }
                for name, values in sorted(bins.items())
            },
            "human_response_bins": {
                name: {
                    "observations": len(values),
                    "marker_tracks": len({row["marker_id"] for row in values}),
                    "mean_relative_speed_mps": _mean(row["relative_speed_mps"] for row in values),
                    "median_relative_speed_mps": _median(row["relative_speed_mps"] for row in values),
                    "p95_relative_speed_mps": _percentile((row["relative_speed_mps"] for row in values), 95),
                    "mean_radial_velocity_mps": _mean(row["radial_velocity_mps"] for row in values),
                    "median_radial_velocity_mps": _median(row["radial_velocity_mps"] for row in values),
                    "approaching_fraction": _mean(row["radial_velocity_mps"] < 0.0 for row in values),
                    "mean_pose_speed_mps": _mean(row["pose_speed_mps"] for row in values),
                    "moving_fraction": _mean(row["robot_motion_bin"] == "moving_ge_0.5mps" for row in values),
                }
                for name, values in sorted(human_bins.items())
            },
            "human_response_observations": len(human_all_rows),
            "human_response_primary_observations": len(human_primary_rows),
            "human_response_quality_excluded_observations": len(human_all_rows) - len(human_primary_rows),
            "human_response_by_robot_motion": {
                name: {
                    "observations": len(values),
                    "mean_distance_m": _mean(row["distance_m"] for row in values),
                    "mean_relative_speed_mps": _mean(row["relative_speed_mps"] for row in values),
                    "median_relative_speed_mps": _median(row["relative_speed_mps"] for row in values),
                    "mean_radial_velocity_mps": _mean(row["radial_velocity_mps"] for row in values),
                    "approaching_fraction": _mean(row["radial_velocity_mps"] < 0.0 for row in values),
                }
                for name, values in sorted(human_motion_bins.items())
            },
        }
    return {
        "status": "complete" if complete else "no_valid_bags",
        "dataset": "HuRoN/SACSoN",
        "source_root": SOURCE_ROOT,
        "local_root": str(root),
        "bags_seen": len(bags),
        "complete_bags": len(complete),
        "failed_bags": len(records) - len(complete),
        "total_odom_samples": int(sum(record.get("odom_samples", 0) for record in complete)),
        "total_pedestrian_messages": int(sum(record.get("pedestrian_messages", 0) for record in complete)),
        "total_laser_messages": int(sum(record.get("laser_messages", 0) for record in complete)),
        "conditions": by_condition,
        "bag_records": records,
        "response_observations": len(response_rows),
        "human_response_observations": len(human_response_rows),
        "human_response_primary_observations": int(
            sum(record.get("human_response_audit", {}).get("primary_observations", 0) for record in complete)
        ),
        "human_response_quality_excluded_observations": int(
            sum(record.get("human_response_audit", {}).get("quality_excluded_observations", 0) for record in complete)
        ),
        "claim_boundary": "single-robot action-conditioned descriptive response audit; recording-local marker IDs only; no causal or persistent-human identity claim",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = audit_huron_sacson(root)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "bags.csv", payload.get("bag_records", []))
    response_rows: list[dict[str, Any]] = []
    for record in payload.get("bag_records", []):
        response = record.get("response_audit", {})
        for distance_bin, summary in response.get("bins", {}).items():
            response_rows.append(
                {
                    "condition": record.get("condition"),
                    "robot_platform": record.get("robot_platform"),
                    "interaction_loss_condition": record.get("interaction_loss_condition"),
                    "file": record.get("file"),
                    "distance_bin": distance_bin,
                    **summary,
                }
            )
    _write_csv(output_dir / "response_bins.csv", response_rows)
    human_response_rows: list[dict[str, Any]] = []
    for record in payload.get("bag_records", []):
        response = record.get("human_response_audit", {})
        for distance_bin, summary in response.get("bins", {}).items():
            human_response_rows.append(
                {
                    "condition": record.get("condition"),
                    "robot_platform": record.get("robot_platform"),
                    "interaction_loss_condition": record.get("interaction_loss_condition"),
                    "file": record.get("file"),
                    "distance_bin": distance_bin,
                    **summary,
                }
            )
    _write_csv(output_dir / "human_response_bins.csv", human_response_rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local HuRoN/SACSoN ROS1 bags.")
    parser.add_argument("--root", default="raw_data/external/huron_sacson/sample_bags")
    parser.add_argument("--output-dir", default="reports/external_validation_20260825/huron_sacson")
    args = parser.parse_args()
    payload = run(Path(args.root), Path(args.output_dir))
    print(
        f"dataset={payload.get('dataset')} status={payload.get('status')} "
        f"bags={payload.get('complete_bags', 0)}/{payload.get('bags_seen', 0)} "
        f"response_observations={payload.get('response_observations', 0)}"
    )


if __name__ == "__main__":
    main()
