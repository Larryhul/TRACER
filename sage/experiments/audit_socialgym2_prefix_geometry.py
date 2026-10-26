"""Compare observable prefix hypotheses and pre-action geometric warnings."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from sage.experiments.socialgym2_execution_diagnostics import load_vector_map
from sage.experiments.socialgym2_joint_geometry import constant_twist_prefix, joint_prefix_clearances


def audit(directory: Path):
    manifest = json.loads((directory / "execution_diagnostics_manifest.json").read_text())
    content = (directory / "execution_diagnostics.jsonl").read_bytes()
    if manifest["status"] != "COMPLETED" or hashlib.sha256(content).hexdigest() != manifest["stream_sha256"]:
        raise ValueError("prefix audit requires a complete hash-bound stream")
    map_path = Path(manifest["map_path"])
    if hashlib.sha256(map_path.read_bytes()).hexdigest() != manifest["map_sha256"]:
        raise ValueError("prefix audit map differs from the executed map")
    segments = load_vector_map(map_path)
    windows = []
    episode_errors = defaultdict(lambda: defaultdict(list))
    previous = None
    current = None
    for line in content.splitlines():
        row = json.loads(line)
        if row["kind"] == "initial":
            previous = row["observation"]
        elif row["kind"] == "window":
            poses = np.asarray(previous["robot_poses"])
            body = np.asarray(previous["robot_body_velocities"])
            action_count = len(poses)
            actions = np.asarray(list(itertools.product((0, 1), repeat=action_count)))
            tick_count = round(manifest["model_dt"] / manifest["sim_dt"])
            predictions, _ = constant_twist_prefix(poses, body, actions, tick_count, manifest["sim_dt"])
            clearances = joint_prefix_clearances(poses[:, :2], predictions, segments)
            selected = int(np.flatnonzero(np.all(actions == row["action"], axis=1))[0])
            current = {
                "episode": row["episode"], "scenario": row["scenario"], "step": row["step"],
                "selected_action": row["action"],
                "initial_official_collision": any(item["official_collision"] for item in previous["collisions"]),
                "predicted_robot_conflict": bool(clearances["robot_clearance"][selected] < 0),
                "predicted_static_conflict": bool(clearances["static_clearance"][selected] < 0),
                "selected_constant_twist_feasible": bool(clearances["feasible"][selected]),
                "feasible_joint_actions": actions[clearances["feasible"]].tolist(),
                "all_halt_feasible": bool(clearances["feasible"][-1]),
                "actual_robot_collision": False, "actual_static_collision": False,
                "actual_human_collision": False, "actual_unknown_collision": False,
                "post_initial_first_collision_time": None,
                "selected_prefix": predictions[selected], "initial_pose": poses,
                "initial_world_velocity": np.asarray(previous["robot_world_velocities"]),
                "constant_twist_position_errors": [], "constant_velocity_position_errors": [],
                "original_position_errors": [],
            }
        elif row["kind"] == "tick":
            if current is None or row["step"] != current["step"]:
                raise ValueError("physical tick lacks its decision-time motion hypothesis")
            tick_index = round(row["elapsed_window_time"] / manifest["sim_dt"]) - 1
            actual = np.asarray(row["observation"]["robot_poses"])[:, :2]
            twist_error = np.linalg.norm(actual - current["selected_prefix"][:, tick_index], axis=-1)
            coast = current["initial_pose"][:, :2] + current["initial_world_velocity"] * row["elapsed_window_time"] * (1 - np.asarray(row["action"]))[:, None]
            coast_error = np.linalg.norm(actual - coast, axis=-1)
            current["constant_twist_position_errors"].extend(twist_error.tolist())
            current["constant_velocity_position_errors"].extend(coast_error.tolist())
            current["original_position_errors"].extend(row["position_error"])
            for collision in row["observation"]["collisions"]:
                if not collision["official_collision"]:
                    continue
                if current["post_initial_first_collision_time"] is None:
                    current["post_initial_first_collision_time"] = row["time"]
                labels = collision["geometry_labels"]
                current["actual_robot_collision"] |= "robot_robot" in labels
                current["actual_static_collision"] |= "robot_static_front_segment" in labels
                current["actual_human_collision"] |= "robot_human_observed" in labels
                current["actual_unknown_collision"] |= not labels
            previous = row["observation"]
        elif row["kind"] == "update":
            if current is None:
                raise ValueError("evidence update has no observed window")
            for field in ("selected_prefix", "initial_pose", "initial_world_velocity"):
                current.pop(field)
            for name in ("original", "constant_velocity", "constant_twist"):
                values = current.pop(f"{name}_position_errors")
                current[f"{name}_mean_position_error"] = float(np.mean(values))
                episode_errors[current["episode"]][name].extend(values)
            windows.append(current)
            current = None
    if current is not None:
        raise ValueError("prefix audit ended in an incomplete execution window")
    clean = [window for window in windows if not window["initial_official_collision"]]
    robot_confusion = {
        "true_positive": sum(window["predicted_robot_conflict"] and window["actual_robot_collision"] for window in clean),
        "false_positive": sum(window["predicted_robot_conflict"] and not window["actual_robot_collision"] for window in clean),
        "false_negative": sum(not window["predicted_robot_conflict"] and window["actual_robot_collision"] for window in clean),
        "true_negative": sum(not window["predicted_robot_conflict"] and not window["actual_robot_collision"] for window in clean),
    }
    episode_summary = {
        str(episode): {name: {"mean": float(np.mean(values)), "p95": float(np.quantile(values, 0.95))} for name, values in methods.items()}
        for episode, methods in episode_errors.items()
    }
    return {
        "status": "COMPLETED", "input": str(directory), "stream_sha256": manifest["stream_sha256"],
        "episode_errors": episode_summary, "windows": windows,
        "robot_confusion_clean_decision_windows": robot_confusion,
        "selected_geometrically_infeasible_windows": sum(not window["selected_constant_twist_feasible"] for window in windows),
        "no_feasible_action_windows": sum(not window["feasible_joint_actions"] for window in windows),
        "boundary": "Frozen observed-state replay, not repaired-policy navigation. Constant twist is a motion hypothesis, not a certified future bound. Resting GoAlone and Halt may coincide; this alone cannot replace a task-directed planner. Window counts are not independent experimental samples.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.input)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "episodes": len(result["episode_errors"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
