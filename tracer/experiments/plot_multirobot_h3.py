from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Circle
import numpy as np
from scipy.optimize import linear_sum_assignment

from tracer.experiments import plot_reference_trajectories as common
from tracer.experiments.plot_refined_trajectories import COLORS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "experiments/h3_multirobot_dense8_qualitative_20260908/run/execution_diagnostics.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/paper_trajectories_refined_20260908"
POLICIES = ("tracer-persistent", "tracer-no-pair", "tracer-reset", "official-goalone")
NAMES = ("TRACER", "TRACER w/o pair", "TRACER reset belief", "GoAlone")
CONTACT_DISTANCE = 0.4


def unique_humans(observation, expected=None):
    grouped = {}
    for item in observation.get("human_observations", []):
        grouped.setdefault(item["source_robot"], []).append(item["position"])
    if not grouped:
        raise ValueError("episode contains no human observations")
    if expected is not None:
        complete = [values for _, values in sorted(grouped.items()) if len(values) == expected]
        if not complete:
            raise ValueError(f"no complete observation of {expected} humans; refuse to invent missing tracks")
        return np.asarray(complete[0], dtype=float)
    values = []
    for positions in grouped.values():
        for position in positions:
            if not any(np.linalg.norm(np.asarray(position) - previous) < 1e-4 for previous in values):
                values.append(position)
    return np.asarray(values, dtype=float)


def align_humans(observations):
    first = np.asarray(observations[0], dtype=float)
    order = np.lexsort((first[:, 1], first[:, 0]))
    aligned = [first[order]]
    for values in observations[1:]:
        current = np.asarray(values, dtype=float)
        if current.shape != aligned[-1].shape:
            raise ValueError("human count changes inside episode")
        predicted = aligned[-1] if len(aligned) == 1 else 2 * aligned[-1] - aligned[-2]
        costs = np.linalg.norm(predicted[:, None] - current[None], axis=-1)
        _, assignment = linear_sum_assignment(costs)
        aligned.append(current[assignment])
    return np.asarray(aligned)


def load_episodes(path, expected_humans=8):
    episodes = {}
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["kind"] not in {"initial", "tick", "episode"}:
                continue
            key = (record["policy"], int(record["episode"]))
            if record["kind"] == "initial":
                if key in episodes:
                    raise ValueError(f"duplicate episode {key}")
                episodes[key] = {"robot": [], "humans": [], "time": [], "terminal": None,
                                 "initial": {name: record[name] for name in (
                                     "policy", "episode", "episode_seed", "initial_state_hash", "scenario")}}
            if key not in episodes:
                raise ValueError(f"missing initial state for {key}")
            episode = episodes[key]
            if record["kind"] == "episode":
                episode["terminal"] = record["metrics"]
                continue
            observation = record["observation"]
            episode["robot"].append(np.asarray(observation["robot_poses"], dtype=float)[:, :2])
            episode["humans"].append(unique_humans(observation, expected_humans))
            episode["time"].append(float(record["time"]))
            if record["kind"] == "initial":
                episode["goals"] = np.asarray(observation["goals"], dtype=float)[:, :2]
    for key, episode in episodes.items():
        if episode["terminal"] is None:
            raise ValueError(f"incomplete episode {key}")
        episode["robot"] = np.asarray(episode["robot"])
        episode["humans"] = align_humans(episode["humans"])
        episode["time"] = np.asarray(episode["time"])
        if len(episode["time"]) < 2 or episode["time"][0] != 0 or np.any(np.diff(episode["time"]) <= 0):
            raise ValueError(f"invalid physical timestamps for {key}")
        if not all(np.isfinite(episode[name]).all() for name in ("robot", "humans", "goals")):
            raise ValueError(f"nonfinite positions for {key}")
        gaps, contact = sampled_gap(episode)
        episode["min_gap"] = float(gaps.min())
        episode["contact"] = contact
    return episodes


def sampled_gap(episode):
    robot = episode["robot"]
    humans = episode["humans"]
    robot_human = np.linalg.norm(robot[:, :, None] - humans[:, None], axis=-1) - CONTACT_DISTANCE
    human_index = np.unravel_index(robot_human.argmin(), robot_human.shape)
    robot_robot = np.linalg.norm(robot[:, :, None] - robot[:, None], axis=-1) - CONTACT_DISTANCE
    robot_robot[:, np.arange(robot.shape[1]), np.arange(robot.shape[1])] = np.inf
    robot_index = np.unravel_index(robot_robot.argmin(), robot_robot.shape)
    if robot_human[human_index] <= robot_robot[robot_index]:
        step, robot_id, human_id = human_index
        contact = (step, robot[step, robot_id], humans[step, human_id], COLORS[human_id % len(COLORS)])
    else:
        step, first_robot, second_robot = robot_index
        contact = (step, robot[step, first_robot], robot[step, second_robot], common.ORANGE)
    return np.minimum(robot_human.min(axis=(1, 2)), robot_robot.min(axis=(1, 2))), contact


def same_scene(episodes):
    reference = episodes[0]
    for episode in episodes[1:]:
        if episode["initial"]["episode_seed"] != reference["initial"]["episode_seed"]:
            raise ValueError("episode seed mismatch between policies")
        for name in ("robot", "humans", "goals"):
            expected = reference[name] if name == "goals" else reference[name][0]
            actual = episode[name] if name == "goals" else episode[name][0]
            if expected.shape != actual.shape or not np.allclose(expected, actual, rtol=0, atol=1e-5):
                raise ValueError(f"initial scene mismatch: {name}")


def select_episode(episodes, policies=POLICIES):
    candidates = []
    rows = []
    for episode_id in sorted({episode for _, episode in episodes}):
        if not all((policy, episode_id) in episodes for policy in policies):
            raise ValueError(f"missing paired policy for episode {episode_id}")
        loaded = [episodes[(policy, episode_id)] for policy in policies]
        same_scene(loaded)
        for policy, episode in zip(policies, loaded):
            metrics = episode["terminal"]
            rows.append({"episode": episode_id, "policy": policy, "humans": episode["humans"].shape[1],
                         "robots": episode["robot"].shape[1], "team_success": metrics["team_success"],
                         "collision_free_team_success": metrics["collision_free_team_success"],
                         "initial_collision": metrics["initial_collision"], "collision": metrics["collision"],
                         "elapsed_time_s": float(episode["time"][-1]), "min_sampled_gap_m": episode["min_gap"]})
        tracer = loaded[0]
        if any(episode["terminal"]["initial_collision"] for episode in loaded):
            continue
        robot = tracer["robot"]
        displacement = np.linalg.norm(robot[-1] - robot[0], axis=-1)
        distances = np.linalg.norm(robot[:, :, None] - tracer["humans"][:, None], axis=-1)
        encounters = int((distances.min(axis=0) < 1.2).sum())
        safe = bool(tracer["terminal"]["collision_free_team_success"] and tracer["min_gap"] >= 0)
        candidates.append({"episode": episode_id, "safe_team_success": safe,
                           "robot_human_encounters_within_1p2m": encounters,
                           "min_robot_displacement_m": float(displacement.min()),
                           "mean_robot_displacement_m": float(displacement.mean())})
    if not candidates:
        raise ValueError("no complete paired episode without initial collisions")
    selected = max(candidates, key=lambda row: (row["safe_team_success"],
                   row["min_robot_displacement_m"] >= 3, row["robot_human_encounters_within_1p2m"],
                   row["mean_robot_displacement_m"], -row["episode"]))
    return {"selected": selected, "candidates": candidates, "rows": rows}


def marker_indices(times, interval=0.25):
    times = np.asarray(times)
    return np.unique(np.r_[np.searchsorted(times, np.arange(times[0], times[-1], interval)), len(times) - 1])


def draw_panel(axis, episode, limits):
    sampled = marker_indices(episode["time"])
    for human in range(episode["humans"].shape[1]):
        common.path_markers(axis, episode["humans"][sampled, human],
                            COLORS[human % len(COLORS)], size=5.0, zorder=3)
    for robot in range(episode["robot"].shape[1]):
        common.path_markers(axis, episode["robot"][sampled, robot], common.ORANGE,
                            solid=True, size=7.5, zorder=5)
    for goal in episode["goals"]:
        axis.scatter(*goal, marker="*", color="red", s=52, zorder=7)
    common.trajectory_axis(axis, limits)
    common.local_legend(axis)
    metrics = episode["terminal"]
    status = "Collision" if metrics["collision"] else ("Success" if metrics["team_success"] else "Timeout")
    color = "red" if status == "Collision" else ("green" if status == "Success" else "#CE9000")
    box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4}
    axis.text(0.50, 0.967, status, transform=axis.transAxes, fontsize=7, color=color,
              va="top", ha="center", bbox=box)
    axis.text(0.50, 0.88, f"Nav Time: {episode['time'][-1]:.2f}s", transform=axis.transAxes,
              fontsize=7, color=color, va="top", ha="center", bbox=box)
    axis.text(0.025, 0.025, f"Min sampled gap: {episode['min_gap']:.2f} m", transform=axis.transAxes,
              fontsize=6.7, bbox=box)
    if metrics["collision"] and episode["min_gap"] < 0:
        step, first, second, second_color = episode["contact"]
        for position, edgecolor in ((first, common.ORANGE), (second, second_color)):
            axis.add_patch(Circle(position, CONTACT_DISTANCE / 2, facecolor="none",
                                  edgecolor=edgecolor, linewidth=1.0, zorder=8))
        axis.annotate(f"Contact at {episode['time'][step]:.2f}s", (first + second) / 2,
                      xytext=(24, -22), textcoords="offset points", color="red", fontsize=6.5,
                      arrowprops={"arrowstyle": "->", "color": "red", "lw": 0.65}, bbox=box, zorder=9)


def plot_matched_scenes(sources, episode_ids, output):
    if len(sources) != 2 or len(episode_ids) != 2:
        raise ValueError("paired-scene figure requires two sources and two episode IDs")
    panels = []
    provenance = []
    all_rows = []
    for source, episode_id in zip(sources, episode_ids):
        episodes = load_episodes(source, 8)
        selection = select_episode(episodes)
        paired = [episodes[(policy, episode_id)] for policy in (POLICIES[0], POLICIES[-1])]
        same_scene(paired)
        panels.extend(paired)
        run_protocol = source.parent.parent / "protocol.json"
        provenance.append({"source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "episode": episode_id, "initial": paired[0]["initial"],
                           "run_protocol": json.loads(run_protocol.read_text(encoding="utf-8")) if run_protocol.is_file() else None,
                           "selection": selection})
        all_rows.extend({"source": str(source), **row} for row in selection["rows"])
    points = np.concatenate([np.concatenate([episode["robot"].reshape(-1, 2),
                             episode["humans"].reshape(-1, 2), episode["goals"]]) for episode in panels])
    low, high = points.min(axis=0), points.max(axis=0)
    limits = common.square_limits((low[0], high[0], low[1], high[1]), padding=0.5)
    common.style()
    figure, axes = common.compact_grid()
    for index, (axis, episode) in enumerate(zip(axes.flat, panels)):
        draw_panel(axis, episode, limits)
        name = "TRACER" if index % 2 == 0 else "GoAlone"
        case = "A" if index < 2 else "B"
        common.panel_caption(axis, f"({chr(97 + index)}) {name} (Case {case})")
    common.save(figure, output, "fig_h3_multirobot_dense8_matched")
    (output / "h3_multirobot_matched_provenance.json").write_text(json.dumps({
        "sources": provenance, "shared_limits_m": list(map(float, limits)),
        "robot_count": 3, "human_count": 8, "layout": "Each row is one matched initial scene; left TRACER, right official GoAlone.",
        "selection_boundary": "Outcome-selected illustrations from exploratory density transfer. Case A shows collision-free completion versus a collision; Case B shows clearance/time tradeoff. Neither is claimed to demonstrate pair or belief superiority; all original four-policy results are retained.",
        "metric_boundary": "Full-episode team time and all-tick dynamic center separation minus 0.4m. No claim about static clearance. All actual positions retained, no spatial warping, common physical limits.",
    }, indent=2) + "\n")
    with (output / "h3_multirobot_matched_screening.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--expected-humans", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--name", default="fig_h3_multirobot_dense8")
    parser.add_argument("--second-source", type=Path)
    parser.add_argument("--second-episode", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(args.source, args.expected_humans)
    selection = select_episode(episodes)
    episode_id = selection["selected"]["episode"] if args.episode is None else args.episode
    loaded = [episodes[(policy, episode_id)] for policy in POLICIES]
    same_scene(loaded)
    points = np.concatenate([np.concatenate([episode["robot"].reshape(-1, 2),
                             episode["humans"].reshape(-1, 2), episode["goals"]]) for episode in loaded])
    low, high = points.min(axis=0), points.max(axis=0)
    limits = common.square_limits((low[0], high[0], low[1], high[1]), padding=0.5)
    common.style()
    figure, axes = common.compact_grid()
    for index, (axis, episode) in enumerate(zip(axes.flat, loaded)):
        draw_panel(axis, episode, limits)
        common.panel_caption(axis, f"({chr(97 + index)}) {NAMES[index]}")
    common.save(figure, args.output_dir, args.name)
    selection.update({
        "plotted_episode": episode_id, "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "robot_count": loaded[0]["robot"].shape[1], "human_count": args.expected_humans,
        "shared_limits_m": list(map(float, limits)), "style_reference": "fig_h3_complex_selected",
        "sampling": "Markers every 0.25 physical seconds plus terminal state; metrics use all physical ticks.",
        "selection_rule": "Exclude initial-collision cases. Prefer collision-free TRACER team success, then nontrivial robot displacement and more distinct robot-human encounters within 1.2m. All paired outcomes are retained.",
        "gap_definition": "Minimum simultaneous robot-human or robot-robot center distance minus 0.4m, using the SocialGym2 diagnostic contact threshold; excludes static walls. Contact time is the minimum-gap instant, not first-contact time.",
        "time_definition": "Team episode elapsed time; success requires all robots. Timeout/collision panels display elapsed duration, not a successful navigation time.",
        "human_identity": "One complete robot observation, then velocity-predicted minimum-cost assignment. Reconstructed visualization identities, not exposed persistent simulator IDs.",
        "claim_boundary": "Exploratory three-robot/eight-human full-observation density transfer, not the frozen two-human H3 aggregate. Transferred likelihood scale is not certified at this density. These are TRACER ablations and an official GoAlone control, not SICNav or MPC-CVMM. Outcome-selected illustration, not typical-performance evidence.",
    })
    protocol_path = args.source.parent.parent / "protocol.json"
    if protocol_path.is_file():
        selection["run_protocol"] = json.loads(protocol_path.read_text(encoding="utf-8"))
    selection["same_scene_max_position_error_m"] = {
        policy: max(float(np.abs(episode[name][0] - loaded[0][name][0]).max()) for name in ("robot", "humans"))
        for policy, episode in zip(POLICIES, loaded)
    }
    selection["robot_trajectory_difference_from_TRACER_m"] = {
        policy: (float(np.max(np.linalg.norm(episode["robot"] - loaded[0]["robot"], axis=-1)))
                 if episode["robot"].shape == loaded[0]["robot"].shape
                 and np.allclose(episode["time"], loaded[0]["time"]) else None)
        for policy, episode in zip(POLICIES, loaded)
    }
    with (args.output_dir / "h3_multirobot_screening.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selection["rows"][0]))
        writer.writeheader()
        writer.writerows(selection["rows"])
    (args.output_dir / "h3_multirobot_dense_provenance.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps({"episode": episode_id, "selected": selection["selected"],
                      "rows": [row for row in selection["rows"] if row["episode"] == episode_id]}, indent=2))
    if args.second_source is not None:
        plot_matched_scenes([args.source, args.second_source], [episode_id, args.second_episode], args.output_dir)


if __name__ == "__main__":
    main()
