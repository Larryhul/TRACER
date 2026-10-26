from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports/paper_trajectories_20260908"
H1_SOURCE = ROOT / "reports/paper_visualizations_20260906/fig_h1_joint_counterfactual_trajectories_data.npz"
ORANGE = "#FFA500"
ROBOT_COLORS = (ORANGE, "#D97B00")
HUMAN_COLORS = ("#0000FF", "#008000", "#B000B0", "#FFB6C1", "#858585", "#CEAD00")
INK = "#20252C"
GRAY = "#7A8089"
JOINT = "#008000"
ADDITIVE = "#B000B0"


def style():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.edgecolor": "#777777", "axes.linewidth": 0.5,
        "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save(figure, output, name):
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=350, facecolor="white",
                       bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


def path_markers(axis, points, color, solid=False, size=12, alpha=1.0, line=None, zorder=3):
    points = np.asarray(points)
    colors = np.tile(to_rgba(color), (len(points), 1))
    colors[:, 3] = np.linspace(0.3, 1.0, len(points)) * alpha
    if line is not None:
        axis.plot(points[:, 0], points[:, 1], color=color, linestyle=line,
                  linewidth=0.65, alpha=alpha * 0.55, zorder=zorder - 1)
    axis.scatter(points[:, 0], points[:, 1], s=size, marker="o",
                 facecolors=colors if solid else "none", edgecolors=colors,
                 linewidths=0.6, zorder=zorder)


def legend_handle(label, color, solid=False, marker="o", line="None"):
    return Line2D([], [], color=color, linestyle=line, linewidth=0.8,
                  marker=marker, markerfacecolor=color if solid else "white",
                  markeredgewidth=0.7, markersize=4.5, label=label)


def compact_grid():
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 7.4))
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.066, top=0.992,
                           wspace=0.10, hspace=0.16)
    for axis in axes.flat:
        axis.set_box_aspect(1)
        axis.tick_params(labelsize=5.4, length=1.7, width=0.4, pad=1.5)
    return figure, axes


def panel_caption(axis, text):
    axis.text(0.5, -0.078, text, transform=axis.transAxes, ha="center", va="top",
              fontsize=13, fontfamily="STIXGeneral", color="black")


def square_limits(limits, padding=0.0):
    left, right, bottom, top = limits
    span = max(right - left, top - bottom) + padding * 2
    center_x, center_y = (left + right) / 2, (bottom + top) / 2
    return (center_x - span / 2, center_x + span / 2,
            center_y - span / 2, center_y + span / 2)


def local_legend(axis, handles=None):
    if handles is None:
        handles = [legend_handle("Robot", ORANGE, True),
                   legend_handle("Goal", "red", True, "*"),
                   legend_handle("Human", "black")]
    legend = axis.legend(handles=handles, loc="upper right", fontsize=6.2,
                         frameon=True, fancybox=False, borderpad=0.25,
                         labelspacing=0.18, handlelength=1.1, handletextpad=0.4,
                         borderaxespad=0.4, framealpha=0.96)
    legend.get_frame().set_linewidth(0.4)
    legend.get_frame().set_edgecolor("#BBBBBB")


def trajectory_axis(axis, limits):
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(limits[:2])
    axis.set_ylim(limits[2:])
    axis.xaxis.set_major_locator(MaxNLocator(5))
    axis.yaxis.set_major_locator(MaxNLocator(5))
    axis.tick_params(labelsize=5.4, length=1.7, width=0.4, pad=1.5)
    for spine in axis.spines.values():
        spine.set_visible(True)


def scale_bar(axis, length=1.0):
    left, right = axis.get_xlim()
    bottom, top = axis.get_ylim()
    start = left + 0.07 * (right - left)
    vertical = bottom + 0.045 * (top - bottom)
    axis.plot([start, start + length], [vertical, vertical], color=INK, linewidth=1.2)
    axis.text(start + length / 2, vertical + 0.025 * (top - bottom), f"{length:g} m",
              ha="center", fontsize=6.5)


def integrate(current, velocity, dt):
    return np.concatenate([current[None], current[None] + np.cumsum(velocity * dt, axis=0)])


def h1_figure(output):
    arrays = load_npz(H1_SOURCE)
    provenance_path = H1_SOURCE.with_name("fig_h1_joint_counterfactual_trajectories_provenance.json")
    provenance = json.loads(provenance_path.read_text())
    entity = provenance["entity_index"]
    current = arrays["agent_current"]
    dt = float(arrays["dt"])
    branch_names = ["Both yield", "R1 advances", "R2 advances", "Both advance"]
    figure, axes = compact_grid()
    primary_paths = [arrays["target_history"]]
    for branch in range(4):
        primary_paths.extend(integrate(current[robot], arrays["robot_branch_velocity"][branch, robot], dt)
                             for robot in range(2))
        primary_paths.extend(integrate(current[2 + entity], arrays[key][branch, entity], dt)
                             for key in ["target_velocity", "joint_velocity", "matched_additive_velocity"])
    primary_points = np.concatenate(primary_paths)
    low, high = primary_points.min(axis=0) - 0.55, primary_points.max(axis=0) + 0.55
    limits = square_limits((low[0], high[0], low[1], high[1]), padding=0.7)
    errors = []
    for branch, axis in enumerate(axes.flat):
        for other in range(arrays["target_velocity"].shape[1]):
            if other == entity:
                continue
            path = integrate(current[2 + other], arrays["target_velocity"][branch, other], dt)
            path_markers(axis, path, HUMAN_COLORS[other % len(HUMAN_COLORS)],
                         size=4, alpha=0.28, zorder=1)
        for robot in range(2):
            path = integrate(current[robot], arrays["robot_branch_velocity"][branch, robot], dt)
            path_markers(axis, path, ROBOT_COLORS[robot], solid=True, size=8, zorder=5)
            axis.annotate(f"R{robot + 1}", path[0], xytext=(-7, 7), textcoords="offset points",
                          color=ROBOT_COLORS[robot], fontsize=7, weight="bold")
        path_markers(axis, arrays["target_history"], GRAY, size=7, alpha=0.6)
        truth = integrate(current[2 + entity], arrays["target_velocity"][branch, entity], dt)
        joint = integrate(current[2 + entity], arrays["joint_velocity"][branch, entity], dt)
        additive = integrate(current[2 + entity], arrays["matched_additive_velocity"][branch, entity], dt)
        path_markers(axis, truth, HUMAN_COLORS[0], size=8, line="-", zorder=6)
        path_markers(axis, joint, JOINT, size=7, line="--", zorder=7)
        path_markers(axis, additive, ADDITIVE, size=7, line=":", zorder=8)
        joint_ade = float(np.linalg.norm(joint[1:] - truth[1:], axis=-1).mean())
        additive_ade = float(np.linalg.norm(additive[1:] - truth[1:], axis=-1).mean())
        errors.append({"branch": branch_names[branch], "joint_ADE_m": joint_ade,
                       "matched_additive_ADE_m": additive_ade})
        trajectory_axis(axis, limits)
        panel_caption(axis, f"({chr(97 + branch)}) {branch_names[branch]}")
        axis.text(0.03, 0.035, f"ADE (m): Joint {joint_ade:.2f}  /  Additive {additive_ade:.2f}",
                  transform=axis.transAxes, ha="left", va="bottom", fontsize=6.5)
    handles = [legend_handle("Robot 1", ROBOT_COLORS[0], True),
               legend_handle("Robot 2", ROBOT_COLORS[1], True),
               legend_handle("Observed history", GRAY),
               legend_handle("True response", HUMAN_COLORS[0], line="-"),
               legend_handle("Joint prediction", JOINT, line="--"),
               legend_handle("Additive prediction", ADDITIVE, line=":")]
    for axis in axes.flat:
        local_legend(axis, handles)
    save(figure, output, "fig_h1_counterfactual_reference")
    return {"source": str(H1_SOURCE), "source_sha256": hashlib.sha256(H1_SOURCE.read_bytes()).hexdigest(),
            "selection": provenance, "branch_metrics": errors,
            "display": "Orange filled robots; hollow entities; pale context is simulator truth, not model predictions. Shared viewport includes every primary trajectory; peripheral entities can leave it. No goal stars: recorded prediction endpoints are not navigation goals."}


def load_episode(output, case, policy):
    stem = output / "data" / f"case{case}_{policy}"
    return load_npz(stem.with_suffix(".npz")), json.loads(stem.with_suffix(".json").read_text())


def scene_limits(episodes):
    points = np.concatenate([np.concatenate([episode["robot"], episode["humans"].reshape(-1, 2),
                                            episode["goal"][None]]) for episode in episodes])
    return (min(-1.2, float(points[:, 0].min()) - 0.3), max(1.2, float(points[:, 0].max()) + 0.3),
            min(-2.05, float(points[:, 1].min()) - 0.3),
            max(2.05, float(points[:, 1].max()) + 0.3))


def draw_scene(axis, episode, limits, label_times=False):
    for segment in episode["obstacles"]:
        axis.plot(segment[:, 0], segment[:, 1], color="#747983", linewidth=1.8,
                  solid_capstyle="round", zorder=1)
    for human in range(episode["humans"].shape[1]):
        path_markers(axis, episode["humans"][:, human], HUMAN_COLORS[human % len(HUMAN_COLORS)],
                     size=7, zorder=3)
    path_markers(axis, episode["robot"], ORANGE, solid=True, size=9, zorder=5)
    axis.scatter(*episode["goal"], marker="*", s=55, facecolor="red", edgecolor="white",
                 linewidth=0.5, zorder=7)
    trajectory_axis(axis, limits)
    if label_times:
        for moment in [0.0, 2.0, 4.0]:
            index = int(np.argmin(np.abs(episode["time"] - moment)))
            if abs(float(episode["time"][index]) - moment) < 1e-6:
                axis.annotate(f"{moment:g}s", episode["robot"][index], xytext=(8, 0),
                              textcoords="offset points", fontsize=6.5, va="center", color="#A65A00",
                              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.3})


def scene_legend(figure, top=0.995):
    figure.legend(handles=[legend_handle("Robot", ORANGE, True),
                           legend_handle("Goal", "red", True, "*"),
                           *[legend_handle(f"Human {index + 1}", HUMAN_COLORS[index]) for index in range(3)]],
                  loc="upper center", bbox_to_anchor=(0.5, top), ncol=5, frameon=False,
                  columnspacing=1.8)


def external_figure(output):
    policies = ["tracer-persistent", "sicnav-np", "sicnav-p", "official-orca"]
    names = ["TRACER", "SICNav-NP", "SICNav-P", "ORCA"]
    records = []
    all_loaded = {case: [load_episode(output, case, policy) for policy in policies] for case in [88, 96]}
    limits = square_limits(scene_limits([episode for loaded in all_loaded.values() for episode, _ in loaded]), padding=0.3)
    for case in [88, 96]:
        figure, axes = compact_grid()
        loaded = all_loaded[case]
        hashes = {metadata["metrics"]["initial_scene_hash"] for _, metadata in loaded}
        if len(hashes) != 1:
            raise ValueError(f"Initial scene mismatch for case {case}")
        for column, (episode, metadata) in enumerate(loaded):
            axis = list(axes.flat)[column]
            draw_scene(axis, episode, limits)
            metrics = metadata["metrics"]
            panel_caption(axis, f"({chr(97 + column)}) {names[column]}")
            if metrics["collision"] or metrics["wall_collision"]:
                status, status_color = "Collision", "#E11919"
            elif metrics["success"]:
                status, status_color = "Success", "#008000"
            else:
                status, status_color = "Timeout", GRAY
            axis.text(0.17, 0.96, status, transform=axis.transAxes,
                      ha="center", va="top", color=status_color, fontsize=7)
            axis.text(0.17, 0.87, f"Nav Time: {episode['time'][-1]:.2f}s", transform=axis.transAxes,
                      ha="center", va="top", color=status_color, fontsize=7)
            local_legend(axis)
            records.append({"case": case, "policy": policies[column], "metrics": metrics})
        name = "fig_h3_external_reference" if case == 96 else "fig_h3_external_reference_case88"
        save(figure, output, name)
    return {"records": records, "main_figure_case": 96, "additional_figure_case": 88,
            "layout": "Two compact 2x2 figures, one case per figure; all eight panels share square coordinate limits.",
            "claim_boundary": "Illustrative archived-controller replay, not an aggregate or latest-H3 result."}


def mechanism_figure(output):
    persistent, persistent_meta = load_episode(output, 96, "tracer-persistent")
    reset, reset_meta = load_episode(output, 96, "tracer-reset")
    if persistent_meta["metrics"]["initial_scene_hash"] != reset_meta["metrics"]["initial_scene_hash"]:
        raise ValueError("Persistent and Reset must share their initial scene")
    figure, axes = compact_grid()
    persistent_axis, reset_axis, belief_axis, action_axis = axes.flat
    limits = square_limits(scene_limits([persistent, reset]), padding=0.3)
    for axis, episode, title in [(persistent_axis, persistent, "(a) Persistent"),
                                  (reset_axis, reset, "(b) Reset")]:
        draw_scene(axis, episode, limits)
        panel_caption(axis, title)
        axis.text(0.17, 0.96, "Success", color="green", fontsize=7,
                  transform=axis.transAxes, ha="center", va="top")
        axis.text(0.17, 0.87, f"Nav Time: {episode['time'][-1]:.2f}s", color="green", fontsize=7,
                  transform=axis.transAxes, ha="center", va="top")
        local_legend(axis)
    belief_times = persistent["time"][:-1]
    reset_times = reset["time"][:-1]
    nearest_distance = np.linalg.norm(persistent["humans"] - persistent["robot"][:, None], axis=-1)
    focal = int(np.argmin(nearest_distance.min(axis=0)))
    mode_colors = ["#0000FF", "#008000", "#B000B0"]
    for mode, color in enumerate(mode_colors):
        belief_axis.plot(belief_times, persistent["belief"][:, focal, mode], color=color, linewidth=1.3)
        belief_axis.plot(reset_times, reset["belief"][:, focal, mode], color=color,
                         linewidth=1.0, linestyle="--", alpha=0.65)
    panel_caption(belief_axis, f"(c) Belief for human {focal + 1}")
    belief_axis.set_ylim(0, 1)
    belief_axis.text(0.03, 0.96, "Mode probability", transform=belief_axis.transAxes, fontsize=7, va="top")
    belief_axis.text(0.97, 0.035, "Time (s)", transform=belief_axis.transAxes, fontsize=7, ha="right")
    local_legend(belief_axis, [Line2D([], [], color=color, label=f"Mode {mode + 1}")
                              for mode, color in enumerate(mode_colors)])
    belief_axis.text(0.02, 0.03, "Solid: Persistent\nDashed: Reset", transform=belief_axis.transAxes,
                     fontsize=6.5, color=GRAY)
    delta = np.linalg.norm(persistent["selected_path"][:, 0, 2:] - persistent["uniform_path"][:, 0, 2:], axis=-1)
    action_axis.plot(belief_times, delta, color="#0000FF", linewidth=1.2,
                     marker="o", markersize=2.5, markerfacecolor="white")
    action_axis.set_ylim(-0.025, max(0.15, float(delta.max()) * 1.3))
    panel_caption(action_axis, "(d) Same-state action probe")
    action_axis.text(0.03, 0.96, r"$\|v_{\rm belief}-v_{\rm uniform}\|$ (m/s)",
                     transform=action_axis.transAxes, fontsize=7, va="top")
    action_axis.text(0.97, 0.035, "Time (s)", transform=action_axis.transAxes, fontsize=7, ha="right")
    changed = int(np.sum(delta > 1e-6))
    action_axis.text(0.03, 0.87, f"Changed velocity: {changed}/{len(delta)} windows", transform=action_axis.transAxes,
                     va="top", fontsize=6.5, color=GRAY)
    common = min(len(persistent["robot"]), len(reset["robot"]))
    separation = float(np.linalg.norm(persistent["robot"][:common] - reset["robot"][:common], axis=-1).max())
    save(figure, output, "fig_h3_belief_decision_reference")
    return {"case": 96, "focal_human": focal, "focal_selection": "Smallest realized center distance under Persistent",
            "max_persistent_reset_position_difference_m": separation,
            "same_state_velocity_change_windows": changed, "same_state_total_windows": len(delta),
            "max_same_state_velocity_difference_mps": float(delta.max()),
            "mode_semantics": persistent_meta["mode_semantics"],
            "claim_boundary": "No benefit claim from this case; candidate index changes are not necessarily action changes. Mode labels are speed-bin proxies, not human-intention ground truth."}


def validate_replays(output):
    records = []
    initial_by_case = {}
    for path in sorted((output / "data").glob("*.npz")):
        episode = load_npz(path)
        metrics = json.loads(path.with_suffix(".json").read_text())["metrics"]
        assert all(np.isfinite(values).all() for values in episode.values()), path
        assert episode["robot"].shape == (len(episode["time"]), 2), path
        assert len(episode["humans"]) == len(episode["robot"]), path
        assert len(episode["robot"]) == int(metrics["steps"]) + 1, path
        np.testing.assert_allclose(np.diff(episode["time"]), metrics["environment_dt"], atol=1e-12)
        np.testing.assert_allclose(episode["time"][-1], metrics["total_time"], atol=1e-8)
        case = int(metrics["case"])
        if case in initial_by_case:
            initial, scene_hash = initial_by_case[case]
            assert scene_hash == metrics["initial_scene_hash"], path
            np.testing.assert_allclose(episode["robot"][0], initial["robot"][0], atol=1e-8)
            np.testing.assert_allclose(episode["humans"][0], initial["humans"][0], atol=1e-8)
            np.testing.assert_allclose(episode["goal"], initial["goal"], atol=1e-8)
            np.testing.assert_allclose(episode["obstacles"], initial["obstacles"], atol=1e-8)
        else:
            initial_by_case[case] = (episode, metrics["initial_scene_hash"])
        if "belief" in episode:
            assert len(episode["belief"]) == len(episode["robot"]) - 1, path
            np.testing.assert_allclose(episode["belief"].sum(axis=-1), 1.0, atol=1e-6)
            np.testing.assert_allclose(episode["decision_robot"], episode["robot"][:-1], atol=1e-6)
            np.testing.assert_allclose(episode["decision_humans"], episode["humans"][:-1], atol=1e-6)
        records.append({"file": path.name, "frames": len(episode["time"]),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    artifacts = {}
    for pattern in ["*.png", "*.pdf", "*.svg", "data/*.json", "figure_manifest.json", "replay_provenance.json"]:
        for path in sorted(output.glob(pattern)):
            artifacts[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"status": "passed", "episodes": records, "artifacts": artifacts,
            "checks": ["finite arrays", "frame counts and 0.25s clock", "episode duration",
                       "same-case initial scene hashes and actual positions", "normalized beliefs",
                       "decision/trajectory temporal alignment"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", choices=["all", "h1", "mechanism", "external"], default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    results = {}
    if args.figure in {"all", "h1"}:
        results["h1"] = h1_figure(args.output_dir)
    if args.figure in {"all", "mechanism"}:
        results["mechanism"] = mechanism_figure(args.output_dir)
    if args.figure in {"all", "external"}:
        results["external"] = external_figure(args.output_dir)
    manifest_path = args.output_dir / "figure_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous.update(results)
    manifest_path.write_text(json.dumps(previous, indent=2, allow_nan=False) + "\n")
    if args.figure == "all":
        validation = validate_replays(args.output_dir)
        (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"rendered": list(results), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
