from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch, Patch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports/paper_supervisor_revision_20260909"
WIDE = 7.00
COLUMN = 3.40
TRACER = "#008C88"
ORANGE = "#FFA500"
RESET = "#C68128"
GRAY = "#7B8794"
BLUE = "#315FBC"
TRUTH = "#0000FF"
ADDITIVE = "#B000B0"
INK = "#243447"
HUMANS = (TRUTH, "#008000", ADDITIVE, "#FFB6C1", "#858585", "#CEC000", "#00A6A6", "#A65A2A")
METHODS = ("persistent", "reset", "shuffled-id")
WINDOWS = tuple(range(1, 8))
NAMES = {"persistent": r"$\mathsf{TRACER}$", "reset": r"$\mathsf{Reset}$", "shuffled-id": r"$\mathsf{ShuffledID}$"}


def style():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "font.style": "normal", "mathtext.fontset": "stix",
        "mathtext.default": "regular", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#818A92", "axes.linewidth": 1.0,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "lines.solid_capstyle": "round", "hatch.linewidth": 0.5,
    })


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_data(destination):
    from tracer.experiments import plot_multirobot_h3 as multi
    from tracer.experiments import plot_selected_supplement as selected

    destination.mkdir(parents=True, exist_ok=True)
    original = ROOT / "reports/paper_trajectories_refined_20260908"
    sources = [original / "h1_selected_data.npz", original / "h1_selection.json",
               original / "h3_multirobot_matched_provenance.json", selected.H1_PATH, selected.H3_PATH]
    for name in ("h1_selected_data.npz", "h1_selection.json"):
        shutil.copy2(original / name, destination / name)
    paired = selected.h1_data()
    paired = paired[paired.effect_mode.eq("nonlinear")].copy()
    paired.to_csv(destination / "fig3_h1_paired_records.csv", index=False)
    selected.h3_data().to_csv(destination / "fig3_h3_relative_changes.csv", index=False)
    h2 = selected.load_h2()
    h2["concentration"] = np.log(3.0) - h2.posterior_entropy
    if not h2.window.isin(WINDOWS).all() or not np.isfinite(h2.concentration).all():
        raise ValueError("unexpected H2 window or nonfinite entropy")
    totals = h2.groupby(["split", "episode_id", "method", "window"]).concentration.agg(["sum", "count"]).reset_index()
    totals.to_csv(destination / "fig4_h2_episode_totals.csv", index=False)
    sources.extend(sorted(selected.H2_ROOT.glob("seed*/raw_metrics.csv")))
    provenance = json.loads((original / "h3_multirobot_matched_provenance.json").read_text(encoding="utf-8"))
    metadata, arrays = [], {}
    for case_index, case in enumerate(provenance["sources"]):
        source = Path(case["source"])
        if sha256(source) != case["source_sha256"]:
            raise ValueError("selected H3 source changed since original figure")
        sources.append(source)
        episodes = multi.load_episodes(source, expected_humans=8)
        pair = [episodes[(policy, case["episode"])] for policy in ("tracer-persistent", "official-goalone")]
        multi.same_scene(pair)
        for policy_index, episode in enumerate(pair):
            panel_index = case_index * 2 + policy_index
            for key in ("robot", "humans", "goals", "time"):
                arrays[f"panel{panel_index}_{key}"] = episode[key]
            contact_step, first, second, second_color = episode["contact"]
            metadata.append({
                "case": "AB"[case_index], "method": "TRACER" if policy_index == 0 else "GoAlone",
                "initial": episode["initial"], "terminal": episode["terminal"],
                "elapsed_time_s": float(episode["time"][-1]), "min_gap_m": episode["min_gap"],
                "contact_time_s": float(episode["time"][contact_step]),
                "contact_positions": [first.tolist(), second.tolist()], "contact_color": second_color,
            })
    np.savez_compressed(destination / "fig5_selected_executions.npz", **arrays)
    write_json(destination / "fig5_selected_executions.json", {
        "panels": metadata, "shared_limits_m": provenance["shared_limits_m"],
        "robot_count": 3, "human_count": 8, "selection_boundary": provenance["selection_boundary"],
        "metric_boundary": provenance["metric_boundary"],
    })
    write_json(destination / "source_manifest.json", {
        "date": "2026-09-09", "h2_model_seeds": sorted(h2.model_seed.unique().tolist()),
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in sources],
        "h2_window_semantics": "Post-update posterior at decision window t; the retained posterior informs the next window. Original scored indices 1–7 are unchanged; window 0 is excluded.",
    })


def cluster_bands(totals, repeats=4000, seed=20260909):
    if repeats < 2 or totals.duplicated(["split", "episode_id", "method", "window"]).any():
        raise ValueError("invalid bootstrap repeats or duplicate episode totals")
    if not np.isfinite(totals[["sum", "count"]]).all().all() or (totals["count"] <= 0).any():
        raise ValueError("invalid episode sums or counts")
    columns = pd.MultiIndex.from_product([METHODS, WINDOWS], names=["method", "window"])
    results = []
    for split_index, split in enumerate(("id", "ood")):
        part = totals[totals.split.eq(split)]
        sums = part.pivot(index="episode_id", columns=["method", "window"], values="sum").reindex(columns=columns).fillna(0)
        counts = part.pivot(index="episode_id", columns=["method", "window"], values="count").reindex(index=sums.index, columns=columns).fillna(0)
        if len(sums) < 2 or (counts.sum(axis=0) <= 0).any():
            raise ValueError("insufficient episode coverage for all six H2 curves")
        rng = np.random.default_rng(seed + split_index)
        weights = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=repeats)
        resampled_counts = weights @ counts.to_numpy()
        if (resampled_counts == 0).any():
            raise ValueError("bootstrap replicate has an unobserved window")
        resampled = (weights @ sums.to_numpy()) / resampled_counts
        low, high = np.quantile(resampled, [0.025, 0.975], axis=0)
        means = sums.sum(axis=0).to_numpy() / counts.sum(axis=0).to_numpy()
        for column_index, (method, window) in enumerate(columns):
            results.append({
                "split": split, "method": method, "window": window,
                "mean": means[column_index], "lower": low[column_index], "upper": high[column_index],
                "active_records": int(counts.iloc[:, column_index].sum()),
                "active_episode_clusters": int((counts.iloc[:, column_index] > 0).sum()),
                "total_episode_clusters": len(sums), "bootstrap_repeats": repeats,
            })
    return pd.DataFrame(results)


def frame(axis, grid=False):
    axis.set_axisbelow(True)
    axis.tick_params(length=3, width=0.8, pad=3)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#818A92")
    if grid:
        axis.grid(axis="y", color="#E7ECEF", linewidth=0.65)


def legend_handle(label, color, marker="o", solid=False, line="none"):
    return Line2D([], [], color=color, linestyle=line, linewidth=1.3, marker=marker,
                  markerfacecolor=color if solid else "white", markeredgewidth=0.9,
                  markersize=4.8, label=label)


def local_legend(axis, handles, fontsize=7.6, loc="upper right", ncol=1):
    legend = axis.legend(handles=handles, loc=loc, ncol=ncol, frameon=True, fancybox=False,
                         framealpha=0.95, fontsize=fontsize, borderpad=0.35,
                         labelspacing=0.25, handlelength=1.35, handletextpad=0.4,
                         borderaxespad=0.45, columnspacing=0.75)
    legend.get_frame().set_linewidth(0.35)
    legend.get_frame().set_edgecolor("#B6BDC3")
    return legend


def path_markers(axis, points, color, solid=False, size=12, alpha=1.0, line=None, zorder=3):
    colors = np.tile(to_rgba(color), (len(points), 1))
    colors[:, 3] = np.linspace(0.32, 1.0, len(points)) * alpha
    small_axis = axis.get_position().width * axis.figure.get_figwidth() < 1.5
    if line is not None:
        axis.plot(points[:, 0], points[:, 1], color=color, linestyle=line,
                  linewidth=0.85 if small_axis else 1.15, alpha=0.82 * alpha, zorder=zorder - 1)
    axis.scatter(points[:, 0], points[:, 1], s=size, marker="o", facecolors=colors if solid else "none",
                 edgecolors=colors, linewidths=0.6 if small_axis else 0.8, zorder=zorder)


def integrate(current, velocity, interval):
    return np.concatenate([current[None], current[None] + np.cumsum(velocity * interval, axis=0)])


def square_limits(points, padding=0.0):
    low, high = points.min(axis=0), points.max(axis=0)
    span = float((high - low).max()) + 2 * padding
    center = (low + high) / 2
    return (center[0] - span / 2, center[0] + span / 2, center[1] - span / 2, center[1] + span / 2)


def trajectory_grid(width, legend_rows=1):
    narrow = width < 5
    left, right, gap = (0.35, 0.05, 0.25) if narrow else (0.40, 0.07, 0.34)
    span = (width - left - right - gap) / 2
    bottom, row_gap, top = 0.45, 0.45, 0.12 + 0.08 * (legend_rows - 1)
    height = 2 * span + bottom + row_gap + top
    figure = plt.figure(figsize=(width, height))
    axes = []
    for row_index in range(2):
        for column_index in range(2):
            origin_x = left + column_index * (span + gap)
            origin_y = bottom + (1 - row_index) * (span + row_gap)
            axis = figure.add_axes([origin_x / width, origin_y / height, span / width, span / height])
            axis.set_aspect("equal", adjustable="box")
            frame(axis)
            axis.tick_params(labelsize=7.5 if narrow else 8.5, length=2.5, pad=2.3)
            axes.append(axis)
    return figure, axes


def panel_caption(figure, axis, label, narrow=False):
    position = axis.get_position()
    height = figure.get_figheight()
    figure.text(position.x0 + position.width / 2, position.y0 - 0.25 / height,
                label, ha="center", va="top", fontsize=9 if narrow else 11)


def build_fig2(data_dir, width=WIDE):
    with np.load(data_dir / "h1_selected_data.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    selection = json.loads((data_dir / "h1_selection.json").read_text(encoding="utf-8"))
    entity, current, interval = selection["entity_index"], arrays["agent_current"], float(arrays["dt"])
    paths = [{
        "truth": integrate(current[entity + 2], arrays["target_velocity"][branch, entity], interval),
        "joint": integrate(current[entity + 2], arrays["joint_velocity"][branch, entity], interval),
        "additive": integrate(current[entity + 2], arrays["matched_additive_velocity"][branch, entity], interval),
        "robot1": integrate(current[0], arrays["robot_branch_velocity"][branch, 0], interval),
        "robot2": integrate(current[1], arrays["robot_branch_velocity"][branch, 1], interval),
    } for branch in range(4)]
    limits = square_limits(np.concatenate([path for branch in paths for path in branch.values()] + [arrays["target_history"]]), 0.65)
    zoom = square_limits(np.concatenate([branch[name][:5] for branch in paths for name in ("truth", "joint", "additive")]), 0.12)
    narrow = width < 5
    figure, axes = trajectory_grid(width, legend_rows=2 if narrow else 1)
    handles = [legend_handle("Robot 1", ORANGE, solid=True), legend_handle("Robot 2", "#D97B00", solid=True),
               legend_handle("Truth", TRUTH, line="-"), legend_handle("Joint", TRACER, line="--"),
               legend_handle("Additive", ADDITIVE, line=":")]
    errors = []
    branches = ("Both yield", "R1 advances", "R2 advances", "Both advance")
    for branch, (axis, records) in enumerate(zip(axes, paths)):
        axis.set_xlim(limits[:2])
        axis.set_ylim(limits[2:])
        axis.set_xticks([-5, 0, 5] if narrow else [-5, -2.5, 0, 2.5, 5])
        axis.set_yticks([-2.5, 0, 2.5, 5])
        for other in range(arrays["target_velocity"].shape[1]):
            if other != entity:
                background = integrate(current[other + 2], arrays["target_velocity"][branch, other], interval)
                path_markers(axis, background, "#A6ADB3", size=1.8 if narrow else 5, alpha=0.24, zorder=1)
        path_markers(axis, arrays["target_history"], "#6A7078", size=3.5 if narrow else 9, alpha=0.65)
        for robot, color in ((1, ORANGE), (2, "#D97B00")):
            path_markers(axis, records[f"robot{robot}"], color, solid=True, size=4.2 if narrow else 15, zorder=4)
            if not narrow:
                axis.annotate(f"R{robot}", records[f"robot{robot}"][0], xytext=(3, 8), textcoords="offset points",
                              fontsize=9, color=color)
        inset = axis.inset_axes([0.595, 0.035 if narrow else 0.075, 0.38, 0.38])
        for name, color, line in (("truth", TRUTH, "-"), ("joint", TRACER, "--"), ("additive", ADDITIVE, ":")):
            path_markers(axis, records[name], color, size=4 if narrow else 17, line=line, zorder=5)
            path_markers(inset, records[name], color, size=4 if narrow else 19, line=line, zorder=5)
        local_legend(axis, handles, fontsize=6.8 if narrow else 7.6)
        frame(inset)
        inset.set_xlim(zoom[:2])
        inset.set_ylim(zoom[2:])
        inset.set_aspect("equal", adjustable="box")
        if narrow:
            inset.set_xticks([])
            inset.set_yticks([])
        else:
            inset.xaxis.set_major_locator(MaxNLocator(2))
            inset.yaxis.set_major_locator(MaxNLocator(2))
            inset.tick_params(labelsize=8, length=2, pad=1)
        left, right, bottom, top = zoom
        axis.add_patch(Rectangle((left, bottom), right - left, top - bottom, fill=False,
                                 edgecolor=GRAY, linewidth=0.85, linestyle="--", zorder=2))
        axis.add_artist(ConnectionPatch(xyA=(right, bottom), coordsA=axis.transData,
                                       xyB=(0, 1), coordsB=inset.transAxes, color=GRAY, linewidth=0.7, zorder=2))
        error = {name: float(np.linalg.norm(records[name][1:] - records["truth"][1:], axis=-1).mean()) for name in ("joint", "additive")}
        errors.append(error)
        metric_text = (f"ADE (m)\n{error['joint']:.3f} / {error['additive']:.3f}" if narrow else
                       "ADE (m)\n" + rf"$\mathsf{{TRACER}}$ {error['joint']:.3f}" + "\n" + rf"$\mathsf{{Additive}}$ {error['additive']:.3f}")
        axis.text(0.035, 0.975 if narrow else 0.035, metric_text,
                  transform=axis.transAxes, va="top" if narrow else "bottom", fontsize=7.2 if narrow else 9,
                  linespacing=1.3, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 1})
        panel_caption(figure, axis, f"({chr(97 + branch)}) {branches[branch]}", narrow)
    return figure, {"branch_ADE_m": errors, "limits_m": list(map(float, limits)),
                    "zoom_limits_m": list(map(float, zoom)), "scene": selection["selected"]["scene"]}


def full_pair_limit(paired):
    values = paired[["x", "y"]].to_numpy()
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("paired RMSE must be finite and nonnegative")
    return max(0.1, float(np.ceil(values.max() * 1.025 * 10) / 10))


def build_fig3(data_dir):
    paired = pd.read_csv(data_dir / "fig3_h1_paired_records.csv")
    changes = pd.read_csv(data_dir / "fig3_h3_relative_changes.csv")
    span = 2.89
    figure = plt.figure(figsize=(WIDE, 4.00))
    left = figure.add_axes([0.52 / WIDE, 0.90 / 4.00, span / WIDE, span / 4.00])
    right = figure.add_axes([4.01 / WIDE, 0.90 / 4.00, span / WIDE, span / 4.00])
    frame(left, grid=True)
    frame(right, grid=True)
    for axis in (left, right):
        axis.set_box_aspect(1)
    limit = full_pair_limit(paired)
    lower = paired.y < paired.x
    left.plot([0, limit], [0, limit], linestyle=(0, (4, 3)), color=GRAY, linewidth=1.25)
    left.scatter(paired.loc[~lower, "x"], paired.loc[~lower, "y"], s=13, color=GRAY, alpha=0.5, edgecolors="none")
    left.scatter(paired.loc[lower, "x"], paired.loc[lower, "y"], s=11, color=TRACER, alpha=0.27, edgecolors="none")
    left.set_xlim(0, limit)
    left.set_ylim(0, limit)
    left.set_xticks(np.arange(0, limit + 0.01, 0.2))
    left.set_yticks(np.arange(0, limit + 0.01, 0.2))
    left.set_xlabel(r"$\mathsf{MatchedAdditive}$ pair RMSE", labelpad=6)
    left.set_ylabel(r"$\mathsf{TRACER}$ (Joint) pair RMSE", labelpad=5)
    left.text(0.055, 0.93, f"Lower RMSE: {100 * lower.mean():.1f}%", color=TRACER,
              transform=left.transAxes, fontsize=11, va="top")
    left.text(0.055, 0.845, f"{lower.sum():,} / {len(paired):,} paired records", transform=left.transAxes,
              fontsize=9, color=TRACER, va="top")
    left_legend = left.legend(handles=[legend_handle("TRACER lower", TRACER, solid=True),
                                       legend_handle("TRACER not lower", GRAY, solid=True),
                                       Line2D([], [], color=GRAY, linestyle=(0, (4, 3)), linewidth=1.25, label="y = x")],
                              frameon=True, fancybox=False, framealpha=0.95, loc="lower right",
                              fontsize=8.2, handlelength=1.5, labelspacing=0.45, borderpad=0.35)
    left_legend.get_frame().set_linewidth(0.35)
    left_legend.get_frame().set_edgecolor("#B6BDC3")
    policies = (("reset-joint", RESET, r"$\mathsf{Reset}$"),
                ("persistent-joint", TRACER, r"$\mathsf{TRACER}$"),
                ("oracle-joint", BLUE, r"$\mathsf{Oracle}$"))
    metrics = ("task", "safety", "human", "collision")
    bar_width = 0.118
    centers = np.arange(4)
    for split_index, split in enumerate(("id", "ood")):
        for policy_index, (policy, color, _) in enumerate(policies):
            selected = changes[changes.split.eq(split) & changes.policy.eq(policy)]
            if len(selected) != 1:
                raise ValueError("missing or repeated H3 component condition")
            values = selected.iloc[0][list(metrics)].to_numpy(dtype=float)
            offset = (split_index * 3 + policy_index - 2.5) * bar_width
            right.bar(centers + offset, values, width=bar_width, color=color if split == "id" else "white",
                      edgecolor=color, linewidth=0.55, hatch=None if split == "id" else "////", zorder=3)
    right.axhline(0, linewidth=1.0, color="#596773", zorder=4)
    right.set_xlim(-0.58, 3.58)
    right.set_ylim(-30, 108)
    right.set_yticks([-20, 0, 20, 40, 60, 80, 100])
    right.set_xticks(centers, ["Task\nprogress", "Safety\ncost", "Human\ncost", "Window\ncollision"])
    right.tick_params(axis="x", pad=5)
    right.set_ylabel(r"Change vs. $\mathsf{TaskOnly}$ (%)", labelpad=5)
    h3_handles = []
    for _, color, label in policies:
        h3_handles.extend([Patch(facecolor=color, edgecolor=color, label=f"{label} · ID"),
                           Patch(facecolor="white", edgecolor=color, hatch="////", label=f"{label} · OOD")])
    h3_legend = right.legend(handles=h3_handles, ncol=2, loc="upper left", frameon=True,
                             fancybox=False, framealpha=0.95, fontsize=7.1, handlelength=1.15,
                             borderpad=0.35, labelspacing=0.24, columnspacing=0.65)
    h3_legend.get_frame().set_linewidth(0.35)
    h3_legend.get_frame().set_edgecolor("#B6BDC3")
    figure.text(0.535, 0.972, "Mechanism-level response modeling and response-aware closed-loop planning",
                ha="center", va="top", fontsize=8.5, color=INK)
    for axis, label in ((left, "(a) Joint-response prediction"), (right, "(b) Response-aware cost trade-offs")):
        position = axis.get_position()
        figure.text(position.x0 + position.width / 2, 0.042, label, ha="center", va="bottom", fontsize=11)
    return figure, {"records": len(paired), "lower_records": int(lower.sum()), "axis_max": limit,
                    "hidden_records": int((paired[["x", "y"]].to_numpy() > limit).any(axis=1).sum()),
                    "panel_width_in": span, "panel_height_in": span}


def build_fig4(summary):
    width, height = 7.16, 2.55
    figure = plt.figure(figsize=(width, height))
    axis = figure.add_axes([0.105, 0.27, 0.885, 0.52])
    frame(axis, grid=True)
    axis.tick_params(labelsize=8.3, length=2.7, width=0.85, pad=2.5)
    colors = {"persistent": TRACER, "reset": RESET, "shuffled-id": GRAY}
    markers = {"persistent": "o", "reset": "s", "shuffled-id": "^"}
    for method in METHODS:
        for split in ("id", "ood"):
            curve = summary[summary.method.eq(method) & summary.split.eq(split)].sort_values("window")
            if tuple(curve.window) != WINDOWS:
                raise ValueError("incomplete H2 curve")
            axis.fill_between(curve.window, curve.lower, curve.upper, color=colors[method],
                              alpha=0.10 if method == "persistent" else 0.055, linewidth=0)
            axis.plot(curve.window, curve["mean"], color=colors[method], linestyle="-" if split == "id" else (0, (3, 2)),
                      linewidth=1.8 if method == "persistent" else 1.25, marker=markers[method],
                      markersize=4.1 if method == "persistent" else 3.4,
                      markerfacecolor=colors[method] if split == "id" else "white", markeredgewidth=0.85,
                      label=f"{method}/{split}", zorder=5 if method == "persistent" else 3)
    axis.set_xlim(0.8, 7.2)
    axis.set_ylim(-0.004, max(0.135, summary.upper.max() * 1.06))
    axis.set_xticks(WINDOWS)
    axis.set_yticks([0, 0.04, 0.08, 0.12])
    axis.set_xlabel(r"Replanning window $\mathit{t}$", fontsize=9.6, labelpad=5)
    axis.set_ylabel("Posterior concentration\n(log 3 − posterior entropy)", fontsize=9.6, labelpad=0)
    h2_handles = []
    for method in METHODS:
        for split in ("id", "ood"):
            h2_handles.append(Line2D([], [], color=colors[method],
                                     linestyle="-" if split == "id" else (0, (3, 2)),
                                     linewidth=1.8 if method == "persistent" else 1.25,
                                     marker=markers[method], markersize=4.4 if method == "persistent" else 3.7,
                                     markerfacecolor=colors[method] if split == "id" else "white",
                                     markeredgewidth=0.8, label=f"{NAMES[method]} · {split.upper()}"))
    h2_legend = axis.legend(handles=h2_handles, loc="upper left", ncol=3, frameon=True,
                            fancybox=False, framealpha=0.95, fontsize=6.8, borderpad=0.30,
                            labelspacing=0.25, columnspacing=0.7, handlelength=1.45)
    h2_legend.get_frame().set_linewidth(0.35)
    h2_legend.get_frame().set_edgecolor("#B6BDC3")
    axis.set_title("Persistent identity evidence survives into later replanning windows", pad=7, fontsize=10.0)
    return figure, {"curves": 6, "windows": list(WINDOWS), "quantity": "ln(3) - H(posterior), nat",
                    "x_axis": "Replanning window t; posterior is evaluated after the execute-observe-update step and informs the next decision window.",
                    "uncertainty": "Pointwise 95% episode-cluster bootstrap, 4000 draws, conditional on the five fitted seeds. Pooled active-record means unchanged; shared cluster draws preserve all methods, entities, windows and model seeds within each episode."}


def marker_indices(times, interval=0.25):
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("invalid execution timestamps")
    return np.unique(np.r_[np.searchsorted(times, np.arange(times[0], times[-1], interval)), len(times) - 1])


def execution_status(terminal, elapsed):
    if terminal["collision"]:
        return "Collision", f"Elapsed: {elapsed:.2f} s", "red"
    if terminal["team_success"]:
        return "Success", f"Team time: {elapsed:.2f} s", "green"
    return "Timeout", f"Elapsed: {elapsed:.2f} s", "#AF7D00"


def build_fig5(data_dir, width=WIDE):
    metadata = json.loads((data_dir / "fig5_selected_executions.json").read_text(encoding="utf-8"))
    with np.load(data_dir / "fig5_selected_executions.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    narrow = width < 5
    figure, axes = trajectory_grid(width)
    handles = [legend_handle("Robot", ORANGE, solid=True),
               legend_handle("Goal", "red", marker="*", solid=True),
               legend_handle("Human", "black")]
    limits = metadata["shared_limits_m"]
    box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.91, "pad": 0.8}
    for panel_index, (axis, record) in enumerate(zip(axes, metadata["panels"])):
        prefix = f"panel{panel_index}_"
        times, robot, humans, goals = (arrays[prefix + key] for key in ("time", "robot", "humans", "goals"))
        sampled = marker_indices(times, interval=0.4 if narrow else 0.25)
        for human in range(humans.shape[1]):
            path_markers(axis, humans[sampled, human], HUMANS[human], size=2.8 if narrow else 9, zorder=3)
        for robot_index in range(robot.shape[1]):
            path_markers(axis, robot[sampled, robot_index], ORANGE, solid=True, size=4 if narrow else 12, zorder=5)
        axis.scatter(goals[:, 0], goals[:, 1], marker="*", color="red", s=24 if narrow else 76, zorder=7)
        local_legend(axis, handles, fontsize=6.8 if narrow else 7.6)
        axis.set_xlim(limits[:2])
        axis.set_ylim(limits[2:])
        axis.set_xticks([-6, 0, 6] if narrow else [-6, -3, 0, 3, 6])
        axis.set_yticks([-6, 0, 6] if narrow else [-6, -3, 0, 3, 6])
        status, time_label, color = execution_status(record["terminal"], record["elapsed_time_s"])
        axis.text(0.5, 0.965, status, transform=axis.transAxes, ha="center", va="top",
                  color=color, fontsize=8.5 if narrow else 10, bbox=box, zorder=10)
        axis.text(0.5, 0.85 if narrow else 0.89, time_label, transform=axis.transAxes, ha="center", va="top",
                  color=color, fontsize=7.3 if narrow else 9.5, bbox=box, zorder=10)
        gap_label = f"Gap: {record['min_gap_m']:.2f} m" if narrow else f"Min sampled gap: {record['min_gap_m']:.2f} m"
        axis.text(0.025, 0.025, gap_label, transform=axis.transAxes, fontsize=7.3 if narrow else 9,
                  bbox=box, zorder=10)
        if record["terminal"]["collision"] and record["min_gap_m"] < 0:
            first, second = np.asarray(record["contact_positions"])
            for position, edge in ((first, ORANGE), (second, record["contact_color"])):
                axis.add_patch(Circle(position, 0.2, facecolor="none", edgecolor=edge, linewidth=1.2, zorder=8))
            axis.annotate(f"Contact\n{record['contact_time_s']:.2f} s", (first + second) / 2,
                          xytext=(0.83 if narrow else 0.72, 0.075), textcoords="axes fraction", ha="center", va="bottom",
                          fontsize=7.2 if narrow else 9, color="red", bbox=box,
                          arrowprops={"arrowstyle": "->", "color": "red", "lw": 0.95}, zorder=11)
        panel_caption(figure, axis, f"({chr(97 + panel_index)}) {record['method']} (Case {record['case']})", narrow)
    return figure, metadata


def save_figure(figure, output, name):
    output.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(output / f"{name}.{suffix}", dpi=450, facecolor="white", bbox_inches=None)
    size = list(map(float, figure.get_size_inches()))
    plt.close(figure)
    return size


def render(data_dir, output, single_column=True):
    style()
    totals = pd.read_csv(data_dir / "fig4_h2_episode_totals.csv")
    summary = cluster_bands(totals)
    summary.to_csv(data_dir / "fig4_h2_cluster_summary.csv", index=False)
    report = {}
    builders = [("fig2_h1_counterfactual", lambda: build_fig2(data_dir)),
                ("fig3_h1_h3_compact", lambda: build_fig3(data_dir)),
                ("fig4_h2_posterior", lambda: build_fig4(summary)),
                ("fig5_h3_multirobot", lambda: build_fig5(data_dir))]
    for name, builder in builders:
        figure, details = builder()
        details["size_in"] = save_figure(figure, output, name)
        report[name] = details
        if name == "fig4_h2_posterior":
            for suffix in ("pdf", "svg", "png"):
                shutil.copy2(output / f"{name}.{suffix}", output / f"fig_h2_identity_evidence_long.{suffix}")
    if single_column:
        for name, builder in (("fig2_h1_counterfactual", build_fig2), ("fig5_h3_multirobot", build_fig5)):
            figure, _ = builder(data_dir, width=COLUMN)
            report[name]["single_column_size_in"] = save_figure(figure, output / "single_column", name)
    write_json(output / "figure_audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--no-single-column", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve() if args.data_dir else output / "plot_data"
    if args.data_dir is None:
        collect_data(data_dir)
    report = render(data_dir, output, single_column=not args.no_single_column)
    print(json.dumps({name: details["size_in"] for name, details in report.items()}, indent=2))


if __name__ == "__main__":
    main()
