"""Summarize completed physical-tick evidence without changing the controller."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from tracer.experiments.socialgym2_execution_diagnostics import SCHEMA


def _summary(values):
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite physical diagnostic")
    return {"count": len(array), "mean": float(array.mean()), "p95": float(np.quantile(array, 0.95)), "max": float(array.max())}


def analyze(directory: Path):
    stream_path = directory / "execution_diagnostics.jsonl"
    manifest_path = directory / "execution_diagnostics_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    content = stream_path.read_bytes()
    if manifest["schema"] != SCHEMA or manifest["status"] != "COMPLETED":
        raise ValueError("only completed physical streams may be summarized")
    if hashlib.sha256(content).hexdigest() != manifest["stream_sha256"]:
        raise ValueError("physical stream content does not match its manifest")
    records = [json.loads(line) for line in content.splitlines()]
    counts = Counter(record["kind"] for record in records)
    if dict(counts) != manifest["counts"]:
        raise ValueError("physical stream record counts do not match")
    if any(record["schema"] != SCHEMA for record in records):
        raise ValueError("mixed physical diagnostic protocols")
    episodes = defaultdict(list)
    for record in records:
        episodes[(record["policy"], record["seed"], record["scenario"], record["episode"])].append(record)
    episode_results = []
    overall_errors = defaultdict(lambda: defaultdict(list))
    for key, episode in episodes.items():
        initial = [row for row in episode if row["kind"] == "initial"]
        final = [row for row in episode if row["kind"] == "episode"]
        ticks = [row for row in episode if row["kind"] == "tick"]
        windows = [row for row in episode if row["kind"] == "window"]
        updates = [row for row in episode if row["kind"] == "update"]
        if len(initial) != 1 or len(final) != 1 or len(windows) != len(updates):
            raise ValueError("episode lacks a complete initial, final, or update record")
        if final[0]["metrics"]["executed_sim_ticks"] != len(ticks):
            raise ValueError("physical ticks do not match the navigation record")
        timestamps = np.array([initial[0]["time"], *[row["time"] for row in ticks]])
        if not np.allclose(np.diff(timestamps), manifest["sim_dt"], rtol=0, atol=1e-7):
            raise ValueError("physical timeline contains missing or duplicated ticks")
        collision_types = Counter()
        collision_robots = set()
        first_collision = None
        unknown_ticks = 0
        near_miss_ticks = 0
        robot_halting = 0
        all_halting = 0
        fallback_ticks = 0
        stall_durations = np.zeros(len(initial[0]["observation"]["robot_poses"]))
        maximum_stall = np.zeros_like(stall_durations)
        human_speed = []
        observed_human_distance = []
        robot_distance = []
        static_distance = []
        for row in ticks:
            observation = row["observation"]
            action = np.asarray(row["action"])
            robot_halting += int(action.sum())
            all_halting += int(np.all(action == 1))
            fallback_ticks += int(row["fallback"])
            goals = np.asarray(observation["goals"])[:, :2]
            positions = np.asarray(observation["robot_poses"])[:, :2]
            remaining = np.linalg.norm(goals - positions, axis=-1) > 0.5
            stationary = np.linalg.norm(row["observed_motion"], axis=-1) < 0.05
            stall_durations = np.where(remaining & stationary, stall_durations + manifest["sim_dt"], 0.0)
            maximum_stall = np.maximum(maximum_stall, stall_durations)
            human_speed.extend(float(np.linalg.norm(human["velocity"])) for human in observation["human_observations"])
            for robot_index, collision in enumerate(observation["collisions"]):
                if collision["official_collision"]:
                    if first_collision is None:
                        first_collision = {"time": row["time"], "step": row["step"], "robot": robot_index, "action": row["action"], "fallback": row["fallback"], "geometry_labels": collision["geometry_labels"]}
                    collision_robots.add(robot_index)
                    collision_types.update(collision["geometry_labels"] or ["unknown"])
                unknown_ticks += int(collision["official_unclassified"])
                for field, target in (("observed_human_distance", observed_human_distance), ("robot_distance", robot_distance), ("static_center_distance", static_distance)):
                    if collision[field] is not None:
                        target.append(collision[field])
                distances = [collision[field] for field in ("observed_human_distance", "robot_distance") if collision[field] is not None]
                near_miss_ticks += int(any(0.4 <= distance < 0.65 for distance in distances))
                strata = ["all", "halt" if action[robot_index] else "go"]
                if abs(row["turn_rate"][robot_index]) > 0.2:
                    strata.append("turning")
                if row["speed_derivative"][robot_index] < -0.1:
                    strata.append("braking")
                local_target = np.asarray(observation["local_targets"])[robot_index, :2]
                heading = observation["robot_poses"][robot_index][2]
                goal_delta = goals[robot_index] - positions[robot_index]
                waypoint_difference = np.arctan2(local_target[1], local_target[0]) + heading - np.arctan2(goal_delta[1], goal_delta[0])
                waypoint_angle = abs(float(np.arctan2(np.sin(waypoint_difference), np.cos(waypoint_difference))))
                if np.linalg.norm(local_target) > 1e-6 and waypoint_angle > 0.35:
                    strata.append("waypoint_bends_from_goal")
                if collision["official_collision"]:
                    strata.append("official_collision")
                if row["fallback"]:
                    strata.append("fallback")
                for stratum in strata:
                    for metric in ("position_error", "velocity_error", "heading_error"):
                        value = row[metric][robot_index]
                        if value is not None:
                            overall_errors[stratum][metric].append(value)
        metrics = final[0]["metrics"]
        initial_collision = any(item["official_collision"] for item in initial[0]["observation"]["collisions"])
        if bool(metrics["initial_collision"]) != initial_collision or bool(metrics["collision"]) != bool(collision_robots):
            raise ValueError("tick collision evidence disagrees with navigation metrics")
        seen_evidence = set()
        short_windows = 0
        evidence_rows = 0
        reentries = 0
        new_ids = 0
        for window, update in zip(windows, updates):
            if window["step"] != update["step"]:
                raise ValueError("decision and update steps do not align")
            before = {track["track_id"]: track for track in window["tracks"]}
            for track in update["tracks"]:
                new_ids += int(track["track_id"] not in before)
                reentries += int(track["observed"] and track["track_id"] in before and before[track["track_id"]]["missed"] > 0)
            short_windows += int(not update["complete_model_step"])
            for row in update["evidence_rows"]:
                identifier = (row["step"], row["track_id"])
                if identifier in seen_evidence or row["step"] != update["step"] or row["track_id"] not in before:
                    raise ValueError("duplicate or unaligned actual evidence identity")
                seen_evidence.add(identifier)
                evidence_rows += 1
        episode_results.append({
            "policy": key[0], "seed": key[1], "scenario": key[2], "episode": key[3],
            "initial_state_hash": initial[0]["initial_state_hash"], "metrics": metrics,
            "ticks": len(ticks), "robot_ticks": len(ticks) * len(stall_durations),
            "first_post_reset_collision": first_collision,
            "official_collision_robots": sorted(collision_robots),
            "collision_geometry_robot_tick_counts_nonexclusive": dict(collision_types),
            "unclassified_official_robot_ticks": unknown_ticks,
            "observed_near_miss_robot_ticks_04_to_065m": near_miss_ticks,
            "robot_halt_fraction": robot_halting / max(len(ticks) * len(stall_durations), 1),
            "all_halt_fraction": all_halting / max(len(ticks), 1),
            "fallback_tick_fraction": fallback_ticks / max(len(ticks), 1),
            "maximum_stationary_unfinished_seconds_by_robot": maximum_stall.tolist(),
            "robots_stalled_at_least_2s": int(np.sum(maximum_stall >= 2.0 - 1e-7)),
            "budget_exhausted_before_team_success": not bool(metrics["team_success"]) and not bool(metrics["terminal_reached"]),
            "short_windows": short_windows, "evidence_rows": evidence_rows,
            "new_track_ids": new_ids, "observed_identity_reentries": reentries,
            "observed_human_speed_descriptive": _summary(human_speed),
            "minimum_observed_human_distance": min(observed_human_distance, default=None),
            "minimum_robot_distance": min(robot_distance, default=None),
            "minimum_static_center_distance": min(static_distance, default=None),
        })
    return {
        "status": "COMPLETED", "input": str(directory.resolve()),
        "stream_sha256": manifest["stream_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "waypoint_frame_audit": "Reconstructed from robot-local graph-navigation target and world robot heading, including v1 streams whose stored waypoint angle was incorrect.",
        "episodes": episode_results,
        "robot_tick_error_strata_descriptive": {stratum: {metric: _summary(values) for metric, values in metrics.items()} for stratum, metrics in overall_errors.items()},
        "boundary": "Physical diagnostic associations, not causal attribution or independent tick samples. Geometry labels are nonexclusive and observation limited. Human speed is not a counterfactual interference estimate; identity switches require independent identities not available here.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.input)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "episodes": len(result["episodes"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
