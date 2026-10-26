from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np

from sage.experiments import plot_reference_trajectories as common


OUTPUT = common.ROOT / "reports/paper_trajectories_refined_20260908"
POLICIES = ["tracer-persistent", "sicnav-np", "official-mpc-cvmm", "official-orca"]
NAMES = ["TRACER", "SICNav-NP", "MPC-CVMM", "ORCA"]
COLORS = ["#0000FF", "#008000", "#B000B0", "#FFB6C1", "#858585", "#CEC000", "#00A6A6", "#A65A2A"]


def write_json(path, value):
    def normalize(item):
        if isinstance(item, dict):
            return {key: normalize(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        if isinstance(item, np.ndarray):
            return normalize(item.tolist())
        if isinstance(item, np.generic):
            return normalize(item.item())
        return item

    Path(path).write_text(json.dumps(normalize(value), indent=2, allow_nan=False) + "\n")


def interaction(velocities):
    return velocities[3] - velocities[1] - velocities[2] + velocities[0]


def plot_h1(output):
    arrays = common.load_npz(output / "h1_selected_data.npz")
    selection = json.loads((output / "h1_selection.json").read_text())
    entity = selection["entity_index"]
    current = arrays["agent_current"]
    dt = float(arrays["dt"])
    paths = []
    for branch in range(4):
        paths.append({
            "Truth": common.integrate(current[entity + 2], arrays["target_velocity"][branch, entity], dt),
            "Joint": common.integrate(current[entity + 2], arrays["joint_velocity"][branch, entity], dt),
            "Additive": common.integrate(current[entity + 2], arrays["matched_additive_velocity"][branch, entity], dt),
            "R1": common.integrate(current[0], arrays["robot_branch_velocity"][branch, 0], dt),
            "R2": common.integrate(current[1], arrays["robot_branch_velocity"][branch, 1], dt),
        })
    points = np.concatenate([path for branch in paths for path in branch.values()] + [arrays["target_history"]])
    low, high = points.min(axis=0), points.max(axis=0)
    limits = common.square_limits((low[0], high[0], low[1], high[1]), padding=0.65)
    early = np.concatenate([path[:5] for branch in paths for name, path in branch.items() if name in {"Truth", "Joint", "Additive"}])
    low, high = early.min(axis=0) - 0.12, early.max(axis=0) + 0.12
    zoom_limits = common.square_limits((low[0], high[0], low[1], high[1]))
    figure = plt.figure(figsize=(7.16, 9.1))
    grid = figure.add_gridspec(3, 2, height_ratios=[1, 1, 0.49], left=0.057, right=0.99,
                              top=0.965, bottom=0.066, wspace=0.11, hspace=0.25)
    labels = ["Both yield", "R1 advances", "R2 advances", "Both advance"]
    styles = {"Truth": (COLORS[0], "-"), "Joint": (COLORS[1], "--"), "Additive": (COLORS[2], ":")}
    handles = [common.legend_handle("Robot 1", common.ROBOT_COLORS[0], True),
               common.legend_handle("Robot 2", common.ROBOT_COLORS[1], True),
               *[common.legend_handle(name, color, line=line) for name, (color, line) in styles.items()]]
    branch_metrics = []
    for branch in range(4):
        axis = figure.add_subplot(grid[branch // 2, branch % 2])
        axis.set_box_aspect(1)
        common.trajectory_axis(axis, limits)
        for other in range(arrays["target_velocity"].shape[1]):
            if other != entity:
                path = common.integrate(current[other + 2], arrays["target_velocity"][branch, other], dt)
                common.path_markers(axis, path, "#939BA4", size=3, alpha=0.28, zorder=1)
        common.path_markers(axis, arrays["target_history"], "#6A7078", size=5, alpha=0.65)
        for robot in range(2):
            path = paths[branch][f"R{robot + 1}"]
            common.path_markers(axis, path, common.ROBOT_COLORS[robot], solid=True, size=8, zorder=4)
            axis.annotate(f"R{robot + 1}", path[0], xytext=(4, 5), textcoords="offset points",
                          fontsize=6.8, color=common.ROBOT_COLORS[robot])
        for name, (color, line) in styles.items():
            common.path_markers(axis, paths[branch][name], color, size=7, line=line, zorder=5)
        common.local_legend(axis, handles)
        common.panel_caption(axis, f"({chr(97 + branch)}) {labels[branch]}")
        inset = axis.inset_axes([0.635, 0.055, 0.345, 0.345])
        for name, (color, line) in styles.items():
            common.path_markers(inset, paths[branch][name], color, size=9, line=line, zorder=5)
        common.trajectory_axis(inset, zoom_limits)
        inset.tick_params(labelsize=4.6, length=1, pad=0.6)
        inset.set_title("Interaction detail", fontsize=6.3, pad=2)
        left, right, bottom, top = zoom_limits
        axis.add_patch(Rectangle((left, bottom), right - left, top - bottom,
                                 fill=False, edgecolor="#797F87", linewidth=0.6, linestyle="--"))
        figure.add_artist(ConnectionPatch(xyA=(right, bottom), coordsA=axis.transData,
                                          xyB=(0, 1), coordsB=inset.transAxes,
                                          color="#A2A6AD", linewidth=0.5, zorder=2))
        errors = {name: float(np.linalg.norm(paths[branch][name][1:] - paths[branch]["Truth"][1:], axis=-1).mean())
                  for name in ["Joint", "Additive"]}
        branch_metrics.append(errors)
        axis.text(0.035, 0.025, f"ADE (m)\nJoint {errors['Joint']:.3f}\nAdditive {errors['Additive']:.3f}",
                  transform=axis.transAxes, fontsize=6.3, va="bottom",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.0})
    residual_axis = figure.add_subplot(grid[2, 0])
    error_axis = figure.add_subplot(grid[2, 1])
    time = np.arange(1, arrays["target_velocity"].shape[-2] + 1) * dt
    residual_velocities = {name: interaction(arrays[key][:, entity]) for name, key in
                           [("Truth", "target_velocity"), ("Joint", "joint_velocity"), ("Additive", "matched_additive_velocity")]}
    residual_positions = {name: np.cumsum(velocity * dt, axis=0) for name, velocity in residual_velocities.items()}
    for name, (color, line) in styles.items():
        residual_axis.plot(time, np.linalg.norm(residual_positions[name], axis=-1),
                           color=color, linestyle=line, linewidth=1.1, label=name)
    for name in ["Joint", "Additive"]:
        color, line = styles[name]
        error_axis.plot(time, np.linalg.norm(residual_positions[name] - residual_positions["Truth"], axis=-1),
                        color=color, linestyle=line, linewidth=1.1, label=name)
        seed_values = arrays["joint_per_seed" if name == "Joint" else "additive_per_seed"][:, :, entity]
        seed_effect = seed_values[:, 3] - seed_values[:, 1] - seed_values[:, 2] + seed_values[:, 0]
        seed_positions = np.cumsum(seed_effect * dt, axis=-2)
        seed_error = np.linalg.norm(seed_positions - residual_positions["Truth"][None], axis=-1)
        error_axis.fill_between(time, seed_error.min(axis=0), seed_error.max(axis=0),
                                color=color, alpha=0.12, linewidth=0)
    residual_axis.set_ylabel(r"$\|\Delta_{12}p(t)\|$ (m)", fontsize=7)
    error_axis.set_ylabel("Residual error (m)", fontsize=7)
    for axis, caption in [(residual_axis, "(e) Non-additive response"), (error_axis, "(f) Joint-effect reconstruction")]:
        axis.set_xlabel("Prediction time (s)", fontsize=7, labelpad=1)
        axis.tick_params(labelsize=6, length=2, pad=1)
        axis.set_ylim(bottom=-0.03)
        axis.legend(frameon=False, fontsize=6.2, loc="upper left")
        axis.text(0.5, -0.33, caption, transform=axis.transAxes, ha="center", va="top", fontsize=10)
    gain = selection["selected"]["relative_residual_gain"] * 100
    error_axis.text(0.97, 0.20, f"Velocity-effect RMSE: {gain:.1f}% lower\nBands: five-seed range (not a CI)",
                    transform=error_axis.transAxes, ha="right", va="bottom", fontsize=6.0,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1})
    figure.text(0.5, 0.991, r"Selected contribution example: $\Delta_{12}p=p_{11}-p_{10}-p_{01}+p_{00}$",
                ha="center", va="top", fontsize=8.2)
    common.save(figure, output, "fig_h1_contribution_zoom_residual")
    result = {"selection": selection, "branch_ADE_m": branch_metrics, "zoom_rule": "Shared square enclosing the first five recorded/predicted focal-entity positions in all four interventions.",
              "zoom_limits_m": list(zoom_limits), "residual_formula": "p11-p10-p01+p00; displacement norms are not ordinary ADE",
              "additive_max_residual_velocity": float(np.abs(residual_velocities["Additive"]).max()),
              "truth_max_residual_displacement_m": float(np.linalg.norm(residual_positions["Truth"], axis=-1).max()),
              "joint_residual_velocity_rmse": float(np.sqrt(np.mean((residual_velocities["Joint"] - residual_velocities["Truth"]) ** 2)))}
    write_json(output / "h1_figure_provenance.json", result)
    return result


def load_h3(cohort, case, policy):
    stem = cohort / "h3_data" / f"case{case}_{policy}"
    return common.load_npz(stem.with_suffix(".npz")), json.loads(stem.with_suffix(".json").read_text())


def select_h3(cohort):
    protocol = json.loads((cohort / "h3_screening_protocol.json").read_text())
    rows, candidates = [], []
    for case in protocol["screening_cases"]:
        loaded = [load_h3(cohort, case, policy) for policy in POLICIES]
        hashes = {record["metrics"]["initial_scene_hash"] for _, record in loaded}
        if len(hashes) != 1:
            raise ValueError(f"Case {case}: scene mismatch")
        metrics = [record["metrics"] for _, record in loaded]
        for policy, (arrays, record) in zip(POLICIES, loaded):
            metric = record["metrics"]
            rows.append({"case": case, "policy": policy, "success": metric["success"],
                         "collision": metric["collision"], "wall_collision": metric["wall_collision"],
                         "collision_free_success": metric["collision_free_success"],
                         "total_time": metric["total_time"], "humans": arrays["humans"].shape[1],
                         "min_sampled_surface_gap": record["minimum_sampled_surface_gap_m"]})
        if metrics[0]["collision_free_success"]:
            safety_gain = metrics[0]["collision_free_success"] - metrics[1]["collision_free_success"]
            other_failures = sum(1 - metric["collision_free_success"] for metric in metrics[2:])
            time_gain = metrics[1]["total_time"] - metrics[0]["total_time"]
            candidates.append({"case": case, "sicnav_safety_gain": safety_gain,
                               "other_failures": other_failures, "sicnav_time_gain_s": time_gain})
    with (cohort / "screening_results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not candidates:
        result = {"status": "no_qualifying_safe_TRACER_case", "rows": rows}
        write_json(cohort / "h3_case_selection.json", result)
        return result
    selected = max(candidates, key=lambda item: (item["sicnav_safety_gain"], item["other_failures"],
                                                item["sicnav_time_gain_s"], -item["case"]))
    selected["has_sicnav_advantage"] = bool(selected["sicnav_safety_gain"] > 0 or selected["sicnav_time_gain_s"] > 0)
    result = {"status": "selected", "selected": selected, "rows": rows,
              "boundary": "Outcome-selected advantage illustration; full screen retained; not an estimate of typical performance."}
    write_json(cohort / "h3_case_selection.json", result)
    return result


def plot_h3(output, cohort):
    result = select_h3(cohort)
    if result["status"] != "selected":
        return result
    case = result["selected"]["case"]
    loaded = [load_h3(cohort, case, policy) for policy in POLICIES]
    points = np.concatenate([np.concatenate([episode["robot"], episode["humans"].reshape(-1, 2), episode["goal"][None]])
                             for episode, _ in loaded])
    low, high = points.min(axis=0), points.max(axis=0)
    limits = common.square_limits((low[0], high[0], low[1], high[1]), padding=0.5)
    figure, axes = common.compact_grid()
    for index, (axis, (episode, record)) in enumerate(zip(axes.flat, loaded)):
        for segment in episode["obstacles"]:
            axis.plot(segment[:, 0], segment[:, 1], color="#747983", linewidth=1.5, zorder=1)
        for human in range(episode["humans"].shape[1]):
            common.path_markers(axis, episode["humans"][:, human], COLORS[human % len(COLORS)], size=5.0, zorder=3)
        common.path_markers(axis, episode["robot"], common.ORANGE, solid=True, size=7.5, zorder=5)
        axis.scatter(*episode["goal"], marker="*", color="red", s=52, zorder=7)
        common.trajectory_axis(axis, limits)
        common.local_legend(axis)
        common.panel_caption(axis, f"({chr(97 + index)}) {NAMES[index]}")
        metrics = record["metrics"]
        status = "Collision" if metrics["collision"] or metrics["wall_collision"] else ("Success" if metrics["success"] else "Timeout")
        color = "red" if status == "Collision" else ("green" if status == "Success" else "#CE9000")
        axis.text(0.50, 0.967, status, transform=axis.transAxes, fontsize=7, color=color,
                  va="top", ha="center", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4})
        axis.text(0.50, 0.88, f"Nav Time: {metrics['total_time']:.2f}s", transform=axis.transAxes,
                  fontsize=7, color=color, va="top", ha="center",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4})
        gap = record["minimum_sampled_surface_gap_m"]
        axis.text(0.025, 0.025, f"Min sampled gap: {gap:.2f} m", transform=axis.transAxes,
                  fontsize=6.7, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4})
        if metrics["collision"] and gap < 0:
            gaps = np.linalg.norm(episode["humans"] - episode["robot"][:, None], axis=-1)
            gaps -= episode["robot_radius"] + episode["human_radii"][None]
            step, human = np.unravel_index(np.argmin(gaps), gaps.shape)
            for position, radius, color in [
                (episode["robot"][step], float(episode["robot_radius"]), common.ORANGE),
                (episode["humans"][step, human], float(episode["human_radii"][human]), COLORS[human % len(COLORS)]),
            ]:
                axis.add_patch(Circle(position, radius, facecolor="none", edgecolor=color,
                                      linewidth=1.0, zorder=8))
            contact = (episode["robot"][step] + episode["humans"][step, human]) * 0.5
            axis.annotate(f"Contact at {episode['time'][step]:.2f}s", contact, xytext=(24, -22),
                          textcoords="offset points", color="red", fontsize=6.5,
                          arrowprops={"arrowstyle": "->", "color": "red", "lw": 0.65},
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
                          zorder=9)
    common.save(figure, output, "fig_h3_complex_selected")
    result["cohort"] = str(cohort)
    result["shared_limits_m"] = list(limits)
    write_json(output / "h3_figure_provenance.json", result)
    return result


def write_results_summary(output, result):
    selected = result["selected"]
    case = selected["case"]
    h1 = json.loads((output / "h1_selection.json").read_text())
    rows = result["rows"]
    lines = ["# Refined trajectory results", "", "## H1 selected contribution example", "",
             f"Scene {h1['base_scene_index']}, entity {h1['entity_index']}; true interaction percentile {100 * h1['selected_interaction_percentile']:.2f}%.",
             "Outcome-selected from 3,072 pairs; all four branches retained; all five fitted seeds favor Joint on interaction RMSE.", "",
             "| Quantity | Joint | Matched additive |", "|---|---:|---:|",
             f"| Interaction velocity RMSE | {h1['selected']['joint_interaction_rmse']:.5f} | {h1['selected']['additive_interaction_rmse']:.5f} |",
             f"| Mean branch ADE (m) | {h1['selected']['joint_mean_ADE_m']:.5f} | {h1['selected']['additive_mean_ADE_m']:.5f} |", "",
             f"## H3 selected eight-human case {case}", "",
             "Post-hoc advantage illustration from a complete eight-case screen. No retraining or controller tuning. Not the original hallway benchmark and not a new blind/confirmatory result.", "",
             "| Method | Collision-free success | Human collision | Time (s) | Min sampled surface gap (m) |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["case"] == case:
            lines.append(f"| {row['policy']} | {row['collision_free_success']:.0f} | {row['collision']:.0f} | {row['total_time']:.2f} | {row['min_sampled_surface_gap']:.5f} |")
    lines.extend(["", "## Entire exploratory eight-human screen", "",
                  "Counts below include unfavorable TRACER cases. They describe eight cases at one seed, not statistical superiority.", "",
                  "| Method | Collision-free successes / 8 | Human-collision cases | Mean duration (s) |",
                  "|---|---:|---:|---:|"])
    for policy in POLICIES:
        group = [row for row in rows if row["policy"] == policy]
        lines.append(f"| {policy} | {sum(int(row['collision_free_success']) for row in group)} / {len(group)} | {sum(int(row['collision']) for row in group)} | {np.mean([row['total_time'] for row in group]):.5f} |")
    lines.extend(["", "## Claim limits", "",
                  "- A one-step navigation-time gain is 0.25 s; do not call it a large effect.",
                  "- Safer than ORCA in a selected case does not mean safer than SICNav or MPC-CVMM.",
                  "- Minimum gap is sampled, not a swept-volume safety certificate. Displayed contact circles use actual body radii at an overlapping recorded frame.",
                  "- A collision-marked policy may still reach the goal; collision-free success and endpoint success are distinct.",
                  "- The preceding six-human screen and the non-reproduced hallway case-90 safety audit remain archived; neither is hidden or substituted into this image.", ""])
    (output / "results_summary.md").write_text("\n".join(lines))


def validate_artifacts(output):
    checks = []
    for cohort in [output, output / "dense8", output / "safety_case90"]:
        if not (cohort / "h3_screening_protocol.json").exists():
            continue
        protocol = json.loads((cohort / "h3_screening_protocol.json").read_text())
        for case in protocol["screening_cases"]:
            baseline = None
            for policy in POLICIES:
                arrays, record = load_h3(cohort, case, policy)
                metrics = record["metrics"]
                assert all(np.isfinite(value).all() for value in arrays.values())
                assert len(arrays["robot"]) == metrics["steps"] + 1
                assert arrays["humans"].shape[0] == len(arrays["robot"])
                np.testing.assert_allclose(np.diff(arrays["time"]), 0.25, atol=1e-12)
                np.testing.assert_allclose(arrays["time"][-1], metrics["total_time"], atol=1e-12)
                expected = metrics["success"] * (1 - metrics["collision"]) * (1 - metrics["wall_collision"])
                assert expected == metrics["collision_free_success"]
                if baseline is not None:
                    other, other_metrics = baseline
                    assert metrics["initial_scene_hash"] == other_metrics["initial_scene_hash"]
                    for key in ["robot", "humans"]:
                        np.testing.assert_allclose(arrays[key][0], other[key][0], atol=1e-9)
                    for key in ["goal", "obstacles"]:
                        np.testing.assert_allclose(arrays[key], other[key], atol=1e-9)
                else:
                    baseline = arrays, metrics
                checks.append({"cohort": cohort.name, "case": case, "policy": policy, "status": "passed"})
    h1 = common.load_npz(output / "h1_selected_data.npz")
    selection = json.loads((output / "h1_selection.json").read_text())
    entity = selection["entity_index"]
    truth = interaction(h1["target_velocity"][:, entity])
    joint = interaction(h1["joint_velocity"][:, entity])
    additive = interaction(h1["matched_additive_velocity"][:, entity])
    np.testing.assert_allclose(np.sqrt(np.mean((joint - truth) ** 2)), selection["selected"]["joint_interaction_rmse"], atol=1e-6)
    np.testing.assert_allclose(np.sqrt(np.mean((additive - truth) ** 2)), selection["selected"]["additive_interaction_rmse"], atol=1e-6)
    artifacts = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "validation.json"}
    result = {"status": "passed", "checked_episodes": len(checks), "episode_checks": checks,
              "h1_checks": "Recomputed joint/additive interaction residual RMSE matches selection record.",
              "artifacts": artifacts}
    write_json(output / "validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--figure", choices=["h1", "h3", "all", "validate"], default="all")
    parser.add_argument("--h3-cohort", type=Path, default=OUTPUT / "dense8")
    args = parser.parse_args()
    common.style()
    if args.figure in {"h1", "all"}:
        plot_h1(args.output_dir)
    if args.figure in {"h3", "all"}:
        result = plot_h3(args.output_dir, args.h3_cohort)
        print(json.dumps(result.get("selected", {"status": result["status"]}), indent=2))
        if result["status"] == "selected":
            write_results_summary(args.output_dir, result)
    if args.figure == "validate":
        result = validate_artifacts(args.output_dir)
        print(json.dumps({"status": result["status"], "checked_episodes": result["checked_episodes"]}))


if __name__ == "__main__":
    main()
