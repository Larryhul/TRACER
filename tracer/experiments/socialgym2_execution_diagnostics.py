"""Read-only physical-time diagnostics for the SocialGym2 deployment adapter."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "socialgym2-execution-diagnostics-v1"


def _vector(value: Any, size: int = 3) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(array) < size or not np.isfinite(array[:size]).all():
        raise ValueError("diagnostic observation has an invalid pose or velocity")
    return array[:size]


def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([cosine * vector[0] - sine * vector[1], sine * vector[0] + cosine * vector[1]])


def _angle_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = first - second
    return np.arctan2(np.sin(difference), np.cos(difference))


def load_vector_map(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        numbers = [float(part) for part in value.replace(",", " ").split()]
        if len(numbers) != 4 or not np.isfinite(numbers).all():
            raise ValueError("vector map must contain finite xy line segments")
        rows.append(numbers)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2, 2)


def point_segment_distances(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    direction = segments[:, 1] - segments[:, 0]
    offset = points[:, None] - segments[None, :, 0]
    fraction = np.sum(offset * direction[None], axis=-1) / np.maximum(np.sum(direction**2, axis=-1), 1e-15)
    closest = segments[None, :, 0] + np.clip(fraction, 0.0, 1.0)[..., None] * direction[None]
    return np.linalg.norm(points[:, None] - closest, axis=-1)


def _segment_intersections(start: np.ndarray, end: np.ndarray, segments: np.ndarray) -> np.ndarray:
    if not len(segments):
        return np.zeros(0, dtype=bool)
    direction = end - start
    walls = segments[:, 1] - segments[:, 0]
    offset = segments[:, 0] - start

    def cross(first, second):
        return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]

    denominator = cross(direction, walls)
    nonparallel = np.abs(denominator) > 1e-12
    safe_denominator = np.where(nonparallel, denominator, 1.0)
    fraction = cross(offset, walls) / safe_denominator
    wall_fraction = cross(offset, direction) / safe_denominator
    ordinary = nonparallel & (fraction >= 0) & (fraction <= 1) & (wall_fraction >= 0) & (wall_fraction <= 1)
    collinear = ~nonparallel & (np.abs(cross(offset, direction)) < 1e-12)
    squared = float(np.dot(direction, direction))
    if squared > 1e-15:
        projections = np.sum((segments - start) * direction, axis=-1) / squared
        collinear &= (projections.min(axis=-1) <= 1) & (projections.max(axis=-1) >= 0)
    else:
        collinear &= point_segment_distances(start[None], segments)[0] < 1e-12
    return ordinary | collinear


def observed_tick(responses: list[Any], partial_observation: bool, segments: np.ndarray) -> dict[str, Any]:
    poses = np.stack([_vector(response.robot_poses) for response in responses])
    body_velocities = np.stack([_vector(response.robot_vels) for response in responses])
    world_velocities = np.stack([_rotate(velocity[:2], pose[2]) for velocity, pose in zip(body_velocities, poses)])
    humans = []
    for robot_index, response in enumerate(responses):
        for observation_index, value in enumerate(response.human_poses or []):
            local_position = _vector(value)[:2]
            if partial_observation and np.linalg.norm(local_position) <= 1e-8:
                continue
            local_velocity = _vector(response.human_vels[observation_index])[:2] if observation_index < len(response.human_vels or []) else np.zeros(2)
            humans.append({
                "source_robot": robot_index,
                "observation_index_not_identity": observation_index,
                "position": (poses[robot_index, :2] + _rotate(local_position, poses[robot_index, 2])).tolist(),
                "velocity": _rotate(local_velocity + body_velocities[robot_index, :2], poses[robot_index, 2]).tolist(),
            })
    human_positions = np.asarray([human["position"] for human in humans], dtype=np.float64).reshape(-1, 2)
    robot_distances = np.linalg.norm(poses[:, None, :2] - poses[None, :, :2], axis=-1)
    np.fill_diagonal(robot_distances, np.inf)
    human_distances = np.linalg.norm(poses[:, None, :2] - human_positions[None], axis=-1)
    static_distances = point_segment_distances(poses[:, :2], segments)
    collisions = []
    for robot_index, response in enumerate(responses):
        pose = poses[robot_index]
        front = pose[:2] + 0.4 * np.asarray([math.cos(pose[2]), math.sin(pose[2])])
        labels = []
        if np.any(robot_distances[robot_index] < 0.4):
            labels.append("robot_robot")
        if np.any(human_distances[robot_index] < 0.4):
            labels.append("robot_human_observed")
        if np.any(_segment_intersections(pose[:2], front, segments)):
            labels.append("robot_static_front_segment")
        collisions.append({
            "robot": robot_index,
            "official_collision": bool(response.collision),
            "geometry_labels": labels,
            "official_unclassified": bool(response.collision) and not labels,
            "robot_distance": float(robot_distances[robot_index].min()) if len(poses) > 1 else None,
            "observed_human_distance": float(human_distances[robot_index].min()) if len(human_positions) else None,
            "static_center_distance": float(static_distances[robot_index].min()) if len(segments) else None,
        })
    return {
        "robot_poses": poses.tolist(), "robot_world_velocities": world_velocities.tolist(),
        "robot_body_velocities": body_velocities.tolist(),
        "goals": [_vector(response.goal_pose).tolist() for response in responses],
        "local_targets": [_vector(response.local_target).tolist() for response in responses],
        "local_targets_frame": "robot_local_xy",
        "robot_states": [int(response.robot_state) for response in responses],
        "follow_targets": [int(response.follow_target) for response in responses],
        "done": [bool(response.done) for response in responses],
        "success": [bool(response.success) for response in responses],
        "human_observations": humans, "collisions": collisions,
        "classification_source": "observed team geometry and supplied vector map; not simulator ground-truth type labels",
    }


def sample_planned_prefix(initial_positions, planned_positions, elapsed_time: float, model_dt: float):
    initial = np.asarray(initial_positions, dtype=np.float64)
    planned = np.asarray(planned_positions, dtype=np.float64)
    if not math.isfinite(elapsed_time) or model_dt <= 0 or elapsed_time < 0:
        raise ValueError("prefix timestamps must be finite and nonnegative")
    if planned.ndim != 3 or initial.shape != planned[:, 0].shape or planned.shape[-1] != 2:
        raise ValueError("prefix must have robot, model-step, xy axes")
    if elapsed_time > planned.shape[1] * model_dt + 1e-8:
        raise ValueError("observed time extends beyond the scored candidate")
    interval = min(int(max(elapsed_time - 1e-10, 0) / model_dt), planned.shape[1] - 1)
    lower = initial if interval == 0 else planned[:, interval - 1]
    upper = planned[:, interval]
    fraction = (elapsed_time - interval * model_dt) / model_dt
    return lower + fraction * (upper - lower), (upper - lower) / model_dt


def track_snapshot(tracks) -> list[dict[str, Any]]:
    return [{
        "track_id": int(track.track_id), "position": np.asarray(track.position).tolist(),
        "velocity": np.asarray(track.velocity).tolist(), "observed": bool(track.observed),
        "missed": int(track.missed),
        "log_belief": None if track.belief is None else np.asarray(track.belief).tolist(),
    } for track in tracks]


class ExecutionDiagnostics:
    def __init__(self, output: Path, map_path: Path, model_dt: float, sim_dt: float, partial_observation: bool):
        self.output = output
        self.map_path = map_path
        self.map_sha256 = hashlib.sha256(map_path.read_bytes()).hexdigest()
        self.segments = load_vector_map(map_path)
        self.model_dt = float(model_dt)
        self.sim_dt = float(sim_dt)
        self.partial_observation = partial_observation
        self.handle = (output / "execution_diagnostics.jsonl").open("x")
        self.counts = {}
        self.current = None
        self.window = None
        self.last_time = None
        self.previous_tick = None

    def _write(self, kind: str, payload: dict[str, Any]):
        record = {"schema": SCHEMA, "kind": kind, **(self.current or {}), **payload}
        self.handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def begin_episode(self, context, responses, tracks):
        if self.current is not None:
            raise ValueError("previous diagnostic episode is incomplete")
        self.current = dict(context)
        self.last_time = 0.0
        self.previous_tick = observed_tick(responses, self.partial_observation, self.segments)
        self._write("initial", {"time": 0.0, "observation": self.previous_tick, "tracks": track_snapshot(tracks)})

    def begin_window(self, step, start_time, positions, velocities, action, tracks, diagnostics):
        if self.current is None or self.window is not None or not math.isclose(start_time, self.last_time, abs_tol=1e-7):
            raise ValueError("diagnostic windows must cover consecutive physical time")
        self.window = {
            "step": int(step), "start_time": float(start_time),
            "initial_positions": np.asarray(self.previous_tick["robot_poses"])[:, :2].tolist(),
            "planned_positions": np.asarray(positions).tolist(), "planned_velocities": np.asarray(velocities).tolist(),
            "action": np.asarray(action).astype(int).tolist(), "tracks": track_snapshot(tracks),
            "fallback": bool(diagnostics["fallback"]),
            "joint_geometry": diagnostics.get("joint_geometry"),
            "selected_risk": float(diagnostics["min_violation_probability"]) if math.isfinite(float(diagnostics["min_violation_probability"])) else None,
        }
        self._write("window", self.window)

    def record_tick(self, timestamp, responses):
        if self.window is None or timestamp <= self.last_time or not math.isclose(timestamp - self.last_time, self.sim_dt, abs_tol=1e-7):
            raise ValueError("duplicate, missing, or nonmonotonic physical tick")
        observation = observed_tick(responses, self.partial_observation, self.segments)
        elapsed = timestamp - self.window["start_time"]
        predicted_position, predicted_velocity = sample_planned_prefix(
            self.window["initial_positions"], self.window["planned_positions"], elapsed, self.model_dt,
        )
        actual_pose = np.asarray(observation["robot_poses"])
        actual_velocity = np.asarray(observation["robot_world_velocities"])
        previous_pose = np.asarray(self.previous_tick["robot_poses"])
        previous_velocity = np.asarray(self.previous_tick["robot_world_velocities"])
        moving_prediction = np.linalg.norm(predicted_velocity, axis=-1) > 1e-6
        heading_error = np.abs(_angle_difference(actual_pose[:, 2], np.arctan2(predicted_velocity[:, 1], predicted_velocity[:, 0])))
        turns = _angle_difference(actual_pose[:, 2], previous_pose[:, 2]) / (timestamp - self.last_time)
        speed_change = (np.linalg.norm(actual_velocity, axis=-1) - np.linalg.norm(previous_velocity, axis=-1)) / (timestamp - self.last_time)
        local_targets = np.asarray(observation["local_targets"])[:, :2]
        target_delta = np.stack([_rotate(target, pose[2]) for target, pose in zip(local_targets, actual_pose)])
        goal_delta = np.asarray(observation["goals"])[:, :2] - actual_pose[:, :2]
        waypoint_bend = np.abs(_angle_difference(np.arctan2(target_delta[:, 1], target_delta[:, 0]), np.arctan2(goal_delta[:, 1], goal_delta[:, 0])))
        self._write("tick", {
            "step": self.window["step"], "time": float(timestamp), "elapsed_window_time": float(elapsed),
            "action": self.window["action"], "fallback": self.window["fallback"],
            "observation": observation,
            "predicted_position": predicted_position.tolist(), "predicted_velocity": predicted_velocity.tolist(),
            "position_error": np.linalg.norm(actual_pose[:, :2] - predicted_position, axis=-1).tolist(),
            "velocity_error": np.linalg.norm(actual_velocity - predicted_velocity, axis=-1).tolist(),
            "heading_error": [float(value) if valid else None for value, valid in zip(heading_error, moving_prediction)],
            "turn_rate": turns.tolist(), "speed_derivative": speed_change.tolist(),
            "waypoint_goal_angle": waypoint_bend.tolist(),
            "waypoint_target_valid": (np.linalg.norm(local_targets, axis=-1) > 1e-6).tolist(),
            "observed_motion": ((actual_pose[:, :2] - previous_pose[:, :2]) / (timestamp - self.last_time)).tolist(),
        })
        self.previous_tick = observation
        self.last_time = float(timestamp)

    def end_window(self, timestamp, tracks, evidence_rows):
        if self.window is None or not math.isclose(timestamp, self.last_time, abs_tol=1e-7):
            raise ValueError("window must end at the last observed tick")
        elapsed = timestamp - self.window["start_time"]
        evidence_ids = [row["track_id"] for row in evidence_rows]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("one observed identity was used more than once in a window")
        self._write("update", {
            "step": self.window["step"], "time": float(timestamp), "elapsed_window_time": float(elapsed),
            "complete_model_step": math.isclose(elapsed, self.model_dt, abs_tol=1e-7),
            "tracks": track_snapshot(tracks), "evidence_rows": evidence_rows,
            "executed_mean_robot_velocity": ((np.asarray(self.previous_tick["robot_poses"])[:, :2] - np.asarray(self.window["initial_positions"])) / elapsed).tolist(),
        })
        self.window = None
        self.handle.flush()

    def end_episode(self, metrics):
        if self.current is None or self.window is not None:
            raise ValueError("episode has an incomplete physical window")
        self._write("episode", {"time": self.last_time, "metrics": metrics})
        self.handle.flush()
        self.current = None

    def close(self, completed: bool):
        self.handle.close()
        if completed and (self.current is not None or self.window is not None):
            raise ValueError("cannot complete an unfinished diagnostic stream")
        if hashlib.sha256(self.map_path.read_bytes()).hexdigest() != self.map_sha256:
            raise ValueError("diagnostic vector map changed during execution")
        path = self.output / "execution_diagnostics.jsonl"
        manifest = {
            "schema": SCHEMA, "status": "COMPLETED" if completed else "INTERRUPTED",
            "counts": self.counts, "stream_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "map_path": str(self.map_path), "map_sha256": self.map_sha256,
            "model_dt": self.model_dt, "sim_dt": self.sim_dt,
            "partial_observation": self.partial_observation,
            "boundary": "Read-only deployment observations. Geometry labels may overlap and cannot establish unseen human collisions; retain official unknowns. No policy changes or safety guarantee.",
        }
        (self.output / "execution_diagnostics_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
