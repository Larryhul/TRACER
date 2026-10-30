from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from tracer.experiments import plot_reference_trajectories as reference


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/paper_supplement_20260908"
H1_PATH = ROOT / "experiments/formal_v3_enhanced_h1/evaluation_corrected/paired_records.csv"
H2_ROOT = ROOT / "experiments/formal_v2/clean_chain_submission_20260904/h2"
H3_PATH = ROOT / "experiments/pilot/h3_candidate_rescue_replication_20260819/episode_metrics.csv"

ORANGE = reference.ORANGE
TEAL = "#008C88"
BLUE = "#315FBC"
GREEN = "#278B55"
GRAY = "#7B8794"
INK = "#243447"


def style():
    reference.style()
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.8,
        "axes.labelpad": 4.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "lines.solid_capstyle": "round",
    })


def frame(axis, grid=True):
    axis.set_axisbelow(True)
    axis.tick_params(length=2.1, width=0.5, pad=2.5)
    if grid:
        axis.grid(axis="y", color="#E3E8EC", linewidth=0.42, linestyle=(0, (1.5, 2.5)))
    for spine in axis.spines.values():
        spine.set_color("#9099A2")
        spine.set_linewidth(0.6)


def bootstrap(values, seed, repeats=2000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), *np.quantile(samples, [0.025, 0.975])


def save(figure, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference.save(figure, OUTPUT, name)
    plt.close(figure)


def h1_data():
    data = pd.read_csv(H1_PATH)
    selected = data[data.method.isin(["matched-additive", "joint"])].copy()
    keys = ["effect_mode", "model_seed", "data_seed", "split", "base_scene_id"]
    paired = selected.pivot(index=keys, columns="method", values="interaction_rmse").dropna()
    return paired.rename(columns={"matched-additive": "x", "joint": "y"}).reset_index()


def h3_data():
    data = pd.read_csv(H3_PATH)
    policies = ["task-only", "reset-joint", "persistent-joint", "oracle-joint"]
    summary = data[data.policy.isin(policies)].groupby(["split", "policy"], as_index=False).agg(
        task_progress=("task_progress", "mean"), safety_cost=("safety_cost", "mean"),
        human_cost=("human_cost", "mean"), collision_rate=("collision_rate", "mean"))
    base = summary[summary.policy.eq("task-only")].set_index("split")
    rows = []
    for _, row in summary[summary.policy.isin(policies[1:])].iterrows():
        reference_row = base.loc[row.split]
        rows.append({
            "split": row.split,
            "policy": row.policy,
            "task": 100 * (row.task_progress - reference_row.task_progress) / reference_row.task_progress,
            "safety": 100 * (reference_row.safety_cost - row.safety_cost) / reference_row.safety_cost,
            "human": 100 * (reference_row.human_cost - row.human_cost) / reference_row.human_cost,
            "collision": 100 * (reference_row.collision_rate - row.collision_rate) / reference_row.collision_rate,
        })
    return pd.DataFrame(rows)


def draw_h1(axis):
    data = h1_data()
    data = data[data.effect_mode.eq("nonlinear")].copy()
    data["gain"] = data.x - data.y
    limit = float(np.nanquantile(np.r_[data.x, data.y], 0.998) * 1.05)
    frame(axis, grid=False)
    axis.plot([0, limit], [0, limit], color="#778594", linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    axis.scatter(data.x, data.y, s=8, color=ORANGE, alpha=0.23, edgecolors="none", zorder=2)
    lower = data.y < data.x
    axis.text(0.05, 0.94, f"Joint lower: {lower.sum():,}/{len(data):,} ({100 * lower.mean():.1f}%)",
              transform=axis.transAxes, fontsize=6.9, color=ORANGE, va="top")
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Matched-Additive pair RMSE")
    axis.set_ylabel("Joint pair RMSE")
    axis.set_title("Nonlinear response", pad=5)
    axis.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=ORANGE,
                                markersize=3.8, label="Paired case"),
                         Line2D([], [], color="#778594", linestyle=(0, (3, 2)),
                                linewidth=0.9, label="y = x")],
                loc="lower right", frameon=True, fancybox=False, framealpha=0.94,
                borderpad=0.25, labelspacing=0.18, handlelength=1.4)


def draw_h3(axis):
    data = h3_data()
    policies = ["reset-joint", "persistent-joint", "oracle-joint"]
    labels = {"reset-joint": "Reset-Joint", "persistent-joint": "Persistent-Joint", "oracle-joint": "Oracle-Joint"}
    colors = {"reset-joint": GREEN, "persistent-joint": ORANGE, "oracle-joint": BLUE}
    metric_keys = ["task", "safety", "human", "collision"]
    metric_labels = ["Task\nprogress", "Safety\ncost", "Human\ncost", "Collision\nrate"]
    frame(axis)
    x = np.arange(4)
    width = 0.115
    for split_id, split in enumerate(("id", "ood")):
        for policy_id, policy in enumerate(policies):
            row = data[(data.split.eq(split)) & data.policy.eq(policy)].iloc[0]
            values = [row[key] for key in metric_keys]
            axis.bar(x + (split_id * 3 + policy_id - 1) * width, values, width=width,
                     color=colors[policy], alpha=0.92 if split == "id" else 0.48,
                     edgecolor="white", linewidth=0.25,
                     label=f"{labels[policy]} · {split.upper()}")
    axis.axhline(0, color="#778594", linewidth=0.7)
    axis.set_xticks(x, metric_labels)
    axis.set_ylabel("Relative change vs. Task-only (%)")
    axis.set_title("Response-aware cost composition", pad=5)
    axis.legend(fontsize=5.9, ncol=2, loc="upper left", frameon=True, fancybox=False,
                framealpha=0.94, borderpad=0.25, labelspacing=0.14, columnspacing=0.35)


def plot_row():
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.10), gridspec_kw={"width_ratios": [1, 1]})
    figure.subplots_adjust(left=0.095, right=0.99, top=0.84, bottom=0.20, wspace=0.28)
    draw_h1(axes[0])
    draw_h3(axes[1])
    axes[0].text(0.5, -0.20, "(a) H1 · Joint response advantage", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=11.3, fontfamily="STIXGeneral")
    axes[1].text(0.5, -0.20, "(b) H3 · Closed-loop component gain", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=11.3, fontfamily="STIXGeneral")
    figure.text(0.535, 0.925, "Mechanism-level response modeling and response-aware closed-loop planning",
                ha="center", fontsize=8.0, color=INK)
    save(figure, "fig_h1_h3_compact_row")


def load_h2():
    columns = ["split", "episode_id", "entity_id", "window", "method", "active", "main_evaluation",
               "posterior_entropy"]
    frames = []
    for path in sorted(H2_ROOT.glob("seed*/raw_metrics.csv")):
        data = pd.read_csv(path, usecols=columns)
        data = data[data.active.eq(1) & data.main_evaluation.eq(1) &
                    data.method.isin(["persistent", "reset", "shuffled-id"])].copy()
        data["model_seed"] = int(path.parent.name[4:])
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def plot_h2():
    data = load_h2()
    methods = {"persistent": (TEAL, "Persistent", "-", "o"),
               "reset": (ORANGE, "Reset", "--", "s"),
               "shuffled-id": (GRAY, "Shuffled ID", ":", "^")}
    figure, axis = plt.subplots(figsize=(7.16, 2.55))
    figure.subplots_adjust(left=0.105, right=0.99, top=0.79, bottom=0.27)
    frame(axis)
    initial_entropy = np.log(3.0)
    for method, (color, label, line, marker) in methods.items():
        subset = data[data.method.eq(method)]
        for split, alpha in (("id", 1.0), ("ood", 0.58)):
            points = []
            for window in sorted(subset.window.unique()):
                values = subset[(subset.split.eq(split)) & (subset.window.eq(window))]
                if values.empty:
                    continue
                gain = initial_entropy - values.posterior_entropy.to_numpy()
                mean, low, high = bootstrap(gain, 20260908 + window)
                points.append((window, mean, low, high))
            points = np.asarray(points)
            axis.plot(points[:, 0], points[:, 1], color=color, linestyle=line,
                      linewidth=1.75 if method == "persistent" else 0.95, alpha=alpha,
                      marker=marker, markersize=3.3 if method == "persistent" else 2.7,
                      markerfacecolor=color if method == "persistent" else "white",
                      markeredgewidth=0.6, label=f"{label} · {split.upper()}" if method != "shuffled-id" or split == "id" else None)
            axis.fill_between(points[:, 0], points[:, 2], points[:, 3], color=color,
                              alpha=0.10 if method == "persistent" else 0.04, linewidth=0)
    axis.set_xlim(0.7, 7.3)
    axis.set_xticks(range(1, 8))
    axis.set_xlabel("Observation window")
    axis.set_ylabel("Identity evidence gain\n(log 3 − posterior entropy) ↑")
    axis.set_title("Persistent identity evidence survives into later replanning windows", pad=5)
    axis.legend(loc="upper left", ncol=3, frameon=True, fancybox=False, framealpha=0.94,
                borderpad=0.25, labelspacing=0.16, columnspacing=0.55, handlelength=1.5)
    figure.text(0.535, 0.92, "H2 · Identity-bound evidence accumulation",
                ha="center", fontsize=8.0, color=INK)
    figure.text(0.535, 0.055, "Persistent retains cross-window evidence; Reset and Shuffled ID lose most of it after early observations.",
                ha="center", fontsize=6.9, color="#5D6874")
    save(figure, "fig_h2_identity_evidence_long")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=("row", "h2", "all"), default="all")
    args = parser.parse_args()
    style()
    if args.figure in ("row", "all"):
        plot_row()
    if args.figure in ("h2", "all"):
        plot_h2()


if __name__ == "__main__":
    main()
