from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib import patheffects

ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / "reports" / "paper_visualizations_20260906"

INK = "#1F2933"
NAVY = "#193B5A"
SLATE = "#607080"
GRID = "#DCE4EA"
PALE = "#F4F7F9"
TEAL = "#1F9D8B"
TEAL_DARK = "#14796C"
ORANGE = "#E98A5B"
GOLD = "#E4B84A"
PURPLE = "#7161D9"
RED = "#C94B50"
BLUE = "#3478B8"
WHITE = "#FFFFFF"

METHOD_COLORS = {
    "additive": GOLD,
    "joint-without-pair": SLATE,
    "matched-additive": ORANGE,
    "joint": TEAL,
    "shuffled-joint": RED,
}
METHOD_LABELS = {
    "additive": "Additive",
    "joint-without-pair": "Joint w/o pair",
    "matched-additive": "Matched additive",
    "joint": "Joint",
    "shuffled-joint": "Shuffled joint",
}
POLICY_COLORS = {
    "task-only": "#A8B2BB",
    "reset-joint": GOLD,
    "persistent-joint": TEAL,
    "recurrent-joint": PURPLE,
    "oracle-joint": NAVY,
}
POLICY_LABELS = {
    "task-only": "Task-only",
    "reset-joint": "Reset",
    "persistent-joint": "Persistent",
    "recurrent-joint": "Recurrent",
    "oracle-joint": "Oracle",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), dpi=600)
    fig.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(fig)


def panel(ax: plt.Axes, letter: str, title: str, subtitle: str | None = None) -> None:
    ax.text(-0.12, 1.12, letter, transform=ax.transAxes, fontsize=9.1, fontweight="bold", color=NAVY, va="top")
    ax.text(0.0, 1.12, title, transform=ax.transAxes, fontsize=8.0, fontweight="bold", color=INK, va="top")
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=5.9, color=SLATE, va="top")


def grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)


def ci(values: Iterable[float], seed: int = 1, n: int = 1000) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return tuple(np.quantile(boot, [0.025, 0.975]))


def h1(root: Path, out: Path) -> None:
    path = root / "experiments/formal_v3_enhanced_h1/evaluation_corrected/paired_records.csv"
    data = pd.read_csv(path)
    mechanisms = ["none", "additive", "nonlinear"]
    methods = ["additive", "joint-without-pair", "matched-additive", "joint", "shuffled-joint"]
    labels = {"none": "No interaction", "additive": "Additive", "nonlinear": "Nonlinear"}
    table = {
        metric: data.pivot_table(index="effect_mode", columns="method", values=metric, aggfunc="mean").reindex(index=mechanisms, columns=methods)
        for metric in ["nll", "interaction_rmse"]
    }
    fig = plt.figure(figsize=(7.16, 4.08))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.92], width_ratios=[1, 1, 1, 1, 1, 1], wspace=1.0, hspace=0.80)

    ax = fig.add_subplot(gs[0, :4])
    values = table["interaction_rmse"].to_numpy()
    vmax = np.nanmax(values)
    norm = Normalize(0, max(vmax, 1e-6))
    cmap = LinearSegmentedColormap.from_list("error", ["#E6F2EF", "#86C2B4", "#F0C765", "#C94B50"])
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(methods)), [METHOD_LABELS[x] for x in methods], rotation=28, ha="right")
    ax.set_yticks(range(len(mechanisms)), [labels[x] for x in mechanisms])
    ax.tick_params(length=0)
    ax.set_xlabel("Response model")
    ax.set_ylabel("Paired mechanism")
    panel(ax, "A", "Response deformation atlas", "Interaction RMSE · lower is better")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.5, color=WHITE if value > vmax * 0.55 else INK, fontweight="bold" if methods[j] == "joint" else "normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.020, pad=0.040)
    cbar.ax.tick_params(labelsize=6, length=0)
    cbar.set_label("RMSE", fontsize=6.6, labelpad=3)

    ax = fig.add_subplot(gs[0, 4:])
    x = np.arange(len(mechanisms))
    for method, offset in [("matched-additive", -0.12), ("joint", 0.12)]:
        vals = table["nll"][method].to_numpy()
        ax.plot(x + offset, vals, marker="o", markersize=3.3, linewidth=1.5, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        for xi, yi in zip(x + offset, vals):
            ax.text(xi, yi + 0.08 if yi < 0 else yi - 0.08, f"{yi:.2f}", ha="center", va="bottom" if yi < 0 else "top", fontsize=5.8, rotation=90, color=SLATE)
    ax.set_xticks(x, [labels[m] for m in mechanisms], rotation=25, ha="right")
    ax.set_ylabel("NLL")
    ax.axhline(0, color=GRID, linewidth=0.7)
    ax.legend(frameon=False, loc="lower left", handlelength=1.1, borderpad=0)
    grid(ax)
    panel(ax, "B", "Matched comparison", "Joint and matched additive")

    ax = fig.add_subplot(gs[1, :3])
    rows = []
    for mechanism in mechanisms:
        sub = data[data.effect_mode == mechanism]
        for metric in ["nll", "interaction_rmse"]:
            a = sub[sub.method == "matched-additive"].set_index(["model_seed", "base_scene_id"])[metric]
            b = sub[sub.method == "joint"].set_index(["model_seed", "base_scene_id"])[metric]
            joined = a.to_frame("a").join(b.to_frame("b"), how="inner")
            delta = (joined.a - joined.b).to_numpy()
            rows.append({"mechanism": mechanism, "metric": metric, "delta": float(np.mean(delta)), "low": ci(delta, 11 + len(rows))[0], "high": ci(delta, 11 + len(rows))[1]})
    contrast = pd.DataFrame(rows)
    y = np.arange(len(mechanisms))
    for metric, color, shift, name in [("nll", ORANGE, -0.10, "NLL"), ("interaction_rmse", TEAL, 0.10, "Interaction RMSE")]:
        sub = contrast[contrast.metric == metric].set_index("mechanism").reindex(mechanisms)
        ax.errorbar(sub.delta, y + shift, xerr=[sub.delta - sub.low, sub.high - sub.delta], fmt="o", color=color, markersize=3.8, linewidth=1.4, capsize=2, label=name)
    ax.axvline(0, color=INK, linewidth=0.7)
    ax.set_yticks(y, [labels[m] for m in mechanisms])
    ax.set_xlabel("Matched additive − joint")
    ax.set_xlim(min(contrast.low.min(), -0.01), max(contrast.high.max(), 0.02))
    grid(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", handlelength=1.1)
    panel(ax, "C", "Paired effect sizes", "Positive values favour Joint")

    ax = fig.add_subplot(gs[1, 3:])
    mode_data = data[data.effect_mode.isin(["none", "additive", "nonlinear"])].groupby("method")["interaction_rmse"].mean().reindex(methods)
    ax.barh(np.arange(len(methods)), mode_data, color=[METHOD_COLORS[m] for m in methods], height=0.62)
    ax.set_yticks(np.arange(len(methods)), [METHOD_LABELS[m] for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("Interaction RMSE")
    ax.set_xlim(0, max(mode_data.max() * 1.15, 0.1))
    grid(ax)
    for yi, value in enumerate(mode_data):
        ax.text(value + 0.01, yi, f"{value:.2f}", va="center", fontsize=6.1)
    panel(ax, "D", "Action permutation control", "Shuffled joint breaks the response relation")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.84)
    save(fig, out)


def h2(root: Path, out: Path) -> None:
    files = sorted((root / "experiments/h2_bayesian_hazard_audit_20260905").glob("seed*/raw_metrics.csv"))
    raw = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    raw = raw[(raw.active == 1) & raw.method.isin(["reset", "persistent", "recurrent", "shuffled-id", "oracle"])]
    methods = ["reset", "persistent", "recurrent"]
    fig = plt.figure(figsize=(7.16, 4.32))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.88], wspace=0.95, hspace=0.86)

    ax = fig.add_subplot(gs[0, :4])
    agg = raw.groupby(["split", "method", "window"], as_index=False).agg(m0=("posterior_true_probability", "mean"), entropy=("posterior_entropy", "mean"), correct=("posterior_mode_correct", "mean"))
    for split, col, ls in [("id", TEAL, "-"), ("ood", ORANGE, "--")]:
        sub = agg[(agg.split == split) & (agg.method == "persistent")]
        ax.plot(sub.window, sub.entropy, color=col, linewidth=1.8, linestyle=ls, marker="o", markersize=2.7, label=split.upper())
    reset = agg[agg.method == "reset"].groupby(["split", "window"], as_index=False).entropy.mean()
    for split, col in [("id", GOLD), ("ood", "#D89A42")]:
        sub = reset[reset.split == split]
        ax.plot(sub.window, sub.entropy, color=col, linewidth=0.9, linestyle=":" if split == "ood" else "-", alpha=0.8)
    ax.set_xlabel("Planning window")
    ax.set_ylabel("Posterior entropy")
    ax.set_ylim(bottom=0)
    grid(ax)
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.6)
    panel(ax, "A", "Belief lifecycle", "Persistent solid; OOD dashed")

    ax = fig.add_subplot(gs[0, 4:])
    sub = raw[(raw.split == "id") & (raw.method == "persistent")]
    matrix = pd.crosstab(sub.decision_predicted_mode, sub.posterior_predicted_mode, normalize="all").reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    im = ax.imshow(matrix, vmin=0, vmax=matrix.to_numpy().max(), cmap=LinearSegmentedColormap.from_list("flow", ["#EAF2F4", TEAL]))
    ax.set_xticks([0, 1, 2], ["0", "1", "2"])
    ax.set_yticks([0, 1, 2], ["0", "1", "2"])
    ax.set_xlabel("Posterior mode")
    ax.set_ylabel("Decision mode")
    ax.tick_params(length=0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix.iloc[i, j]:.0%}", ha="center", va="center", fontsize=6.4, color=INK if matrix.iloc[i, j] < matrix.to_numpy().max() * 0.55 else WHITE)
    panel(ax, "B", "Mode transition map", "Persistent · ID")

    ax = fig.add_subplot(gs[1, :3])
    cal = raw[raw.method.isin(methods)].copy()
    bins = np.linspace(1 / 3, 1.0, 8)
    cal["bin"] = pd.cut(cal.posterior_confidence, bins=bins, include_lowest=True)
    for method in methods:
        sub = cal[cal.method == method].groupby("bin", observed=True).agg(conf=("posterior_confidence", "mean"), acc=("posterior_mode_correct", "mean"))
        ax.plot(sub.conf, sub.acc, marker="o", markersize=2.8, linewidth=1.5 if method == "persistent" else 0.95, color={"reset": GOLD, "persistent": TEAL, "recurrent": PURPLE}[method], label=method.title())
    ax.plot([1 / 3, 1], [1 / 3, 1], color=SLATE, linewidth=0.7, linestyle=":")
    ax.set_xlim(1 / 3, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    grid(ax)
    ax.legend(frameon=False, loc="lower right", handlelength=1.2)
    panel(ax, "C", "Probability quality", "Reliability on posterior mode")

    ax = fig.add_subplot(gs[1, 3:])
    diff = agg[agg.method.isin(["reset", "persistent"])].pivot_table(index=["split", "window"], columns="method", values="correct").dropna()
    for split, color in [("id", TEAL), ("ood", ORANGE)]:
        sub = diff.loc[split]
        ax.plot(sub.index, sub.persistent - sub.reset, marker="o", markersize=2.8, linewidth=1.5, color=color, label=split.upper())
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.set_xlabel("Planning window")
    ax.set_ylabel("Persistent − reset")
    grid(ax)
    ax.legend(frameon=False, loc="lower left", handlelength=1.2)
    panel(ax, "D", "Memory gain over time", "Posterior mode accuracy difference")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.84)
    save(fig, out)


def load_h3_windows(root: Path, max_seeds: int = 5) -> pd.DataFrame:
    path = root / "experiments/b3_task_goal_ablation_full_recurrent_20260905/runs.json"
    with path.open() as f:
        obj = json.load(f)
    rows = []
    for seed in sorted(obj, key=lambda x: int(x))[:max_seeds]:
        for split in ["id", "ood"]:
            for policy in ["reset-joint", "persistent-joint"]:
                payload = obj[seed][split][policy]
                for records in payload.get("window_records", []):
                    for rec in records:
                        row = {"seed": int(seed), "split": split, "policy": policy}
                        row.update({k: rec.get(k) for k in ["window", "selected_candidate", "uniform_selected_candidate", "oracle_candidate", "candidate_rank_spearman", "oracle_regret", "belief_selection_flip", "selected_violation_probability", "safety_filter_rejections"]})
                        row["candidate_spread"] = float(np.ptp(np.asarray(rec.get("ground_truth_candidate_utility", []), dtype=float))) if rec.get("ground_truth_candidate_utility") else np.nan
                        rows.append(row)
    return pd.DataFrame(rows)


def h3(root: Path, out: Path) -> None:
    windows = load_h3_windows(root)
    episodes = pd.read_csv(root / "experiments/b3_task_goal_ablation_full_recurrent_20260905/episode_metrics.csv")
    keep = ["task-only", "reset-joint", "persistent-joint", "recurrent-joint", "oracle-joint"]
    episodes = episodes[episodes.policy.isin(keep)]
    fig = plt.figure(figsize=(7.16, 4.35))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.9], wspace=1.0, hspace=0.90)

    ax = fig.add_subplot(gs[0, :3])
    sub = windows[windows.policy == "persistent-joint"]
    heat = sub.pivot_table(index="seed", columns="window", values="candidate_rank_spearman", aggfunc="mean")
    im = ax.imshow(heat, aspect="auto", cmap=LinearSegmentedColormap.from_list("rank", [RED, "#F3F5F6", TEAL_DARK]), vmin=-1, vmax=1)
    ax.set_xticks(range(len(heat.columns)), heat.columns)
    ax.set_yticks(range(len(heat.index)), [str(x) for x in heat.index])
    ax.set_xlabel("Planning window")
    ax.set_ylabel("Seed")
    ax.tick_params(length=0)
    panel(ax, "A", "Online ranking fidelity", "Spearman ρ between predicted and realized candidate utility")
    ax.text(0.985, 0.05, "low ρ", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color=SLATE)
    ax.text(0.985, 0.95, "high ρ", transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=SLATE)

    ax = fig.add_subplot(gs[0, 3:])
    summary = episodes.groupby(["split", "policy"], as_index=False).agg(cost=("realized_total_cost", "mean"), progress=("task_progress", "mean"), regret=("oracle_regret", "mean"), rank=("candidate_rank_spearman", "mean"))
    for split, ls in [("id", "-"), ("ood", "--")]:
        sub = summary[summary.split == split].set_index("policy").reindex(keep)
        x = np.arange(len(keep))
        ax.plot(x, sub.cost, color=TEAL if split == "id" else ORANGE, linestyle=ls, marker="o", markersize=3, linewidth=1.4, label=split.upper())
    ax.set_xticks(range(len(keep)), [POLICY_LABELS[x] for x in keep], rotation=28, ha="right")
    ax.set_ylabel("Realized cost")
    ax.legend(frameon=False, loc="lower right", handlelength=1.4)
    grid(ax)
    panel(ax, "B", "Closed-loop transfer", "Policy profile from ID to OOD")

    ax = fig.add_subplot(gs[1, :3])
    pair = episodes[episodes.policy.isin(["reset-joint", "persistent-joint"])].pivot_table(index=["seed", "split"], columns="policy", values=["realized_total_cost", "oracle_regret", "candidate_rank_spearman"])
    metrics = [("realized_total_cost", "Cost ↓"), ("oracle_regret", "Regret ↓"), ("candidate_rank_spearman", "Rank ρ ↑")]
    x = np.arange(len(metrics))
    for idx, (metric, label) in enumerate(metrics):
        if (metric, "reset-joint") not in pair or (metric, "persistent-joint") not in pair:
            continue
        reset = pair[(metric, "reset-joint")].to_numpy(float)
        persistent = pair[(metric, "persistent-joint")].to_numpy(float)
        for a, b in zip(reset, persistent):
            ax.plot([idx - 0.12, idx + 0.12], [a, b], color=GRID, linewidth=0.8)
        ax.scatter(np.full_like(reset, idx - 0.12), reset, color=GOLD, s=17, zorder=3)
        ax.scatter(np.full_like(persistent, idx + 0.12), persistent, color=TEAL, s=17, zorder=3)
        ax.errorbar([idx - 0.12, idx + 0.12], [reset.mean(), persistent.mean()], yerr=[reset.std(), persistent.std()], fmt="none", color=INK, capsize=2, linewidth=1)
    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylabel("Paired seed aggregate")
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.axhline(0, color=GRID, linewidth=0.6)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=GOLD, label="Reset", markersize=4), Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, label="Persistent", markersize=4)], frameon=False, loc="upper right", ncol=2, handletextpad=0.3)
    panel(ax, "C", "Paired policy shift", "Keep mixed and near-neutral outcomes visible")

    ax = fig.add_subplot(gs[1, 3:])
    sub = windows[windows.policy == "persistent-joint"].copy()
    sub = sub[np.isfinite(sub.candidate_spread)]
    scatter = ax.scatter(sub.candidate_spread, sub.oracle_regret, c=sub.belief_selection_flip, s=10 + 35 * sub.belief_selection_flip, cmap=LinearSegmentedColormap.from_list("flip", ["#DCEDE9", TEAL, RED]), alpha=0.72, edgecolors=WHITE, linewidth=0.25)
    ax.set_xlabel("Candidate response spread")
    ax.set_ylabel("Oracle regret")
    grid(ax)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.045)
    cbar.ax.tick_params(labelsize=6, length=0)
    cbar.set_label("Selection flip", fontsize=6.5)
    panel(ax, "D", "Where belief matters", "Spread × regret × flip rate")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.84)
    save(fig, out)


def external(root: Path, out: Path) -> None:
    path = root / "experiments/external_dynamic_samecase_holdout_20260904_orca_fallback_sicnav_v2/episode_metrics.csv"
    data = pd.read_csv(path)
    selected = ["official-dwa", "official-orca", "official-mpc-cvmm", "official-sicnav-np", "tracer-persistent", "tracer-reset"]
    data = data[data.policy.isin(selected)]
    summary = data.groupby("policy", as_index=False).agg(progress=("task_progress", "mean"), collision=("collision", "mean"), min_distance=("min_distance", "mean"), success=("success", "mean"))
    fig = plt.figure(figsize=(7.16, 3.72))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1], wspace=1.0)
    ax = fig.add_subplot(gs[:, :4])
    for _, row in summary.iterrows():
        method = row.policy
        color = NAVY if method.startswith("official") else TEAL
        size = 35 + 140 * row.success
        ax.scatter(row.progress, row.collision, s=size, color=color, alpha=0.86, edgecolor=WHITE, linewidth=0.9, zorder=3)
        if method in ["official-orca", "official-dwa", "official-mpc-cvmm", "official-sicnav-np", "tracer-persistent"]:
            label = method.replace("official-", "").replace("tracer-", "TRACER ")
            ax.annotate(label, (row.progress, row.collision), xytext=(5, 4), textcoords="offset points", fontsize=6.2)
    ax.set_xlabel("Task progress ↑")
    ax.set_ylabel("Collision rate ↓")
    ax.set_xlim(summary.progress.min() - 0.03, summary.progress.max() + 0.03)
    ax.set_ylim(-0.02, summary.collision.max() + 0.08)
    grid(ax)
    panel(ax, "A", "Same-case operating points", "SICNav-compatible dynamic holdout")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, label="TRACER", markersize=5), Line2D([0], [0], marker="o", color="none", markerfacecolor=NAVY, label="Official controls", markersize=5)], frameon=False, loc="upper left", handletextpad=0.4)

    ax = fig.add_subplot(gs[:, 4:])
    values = summary.set_index("policy")["min_distance"].sort_values()
    y = np.arange(len(values))
    ax.barh(y, values, color=[TEAL if x.startswith("tracer") else NAVY for x in values.index], height=0.55)
    ax.set_yticks(y, [x.replace("official-", "").replace("tracer-", "TRACER ") for x in values.index])
    ax.set_xlabel("Minimum distance ↑")
    ax.set_xlim(0, values.max() * 1.20)
    grid(ax)
    for yi, val in enumerate(values):
        ax.text(val + 0.005, yi, f"{val:.2f}", va="center", fontsize=6)
    panel(ax, "B", "Safety margin", "Same protocol, same cases")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.82)
    save(fig, out)


def simplex_xy(p: np.ndarray) -> np.ndarray:
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    return p @ vertices


def draw_simplex(ax: plt.Axes) -> None:
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.fill(tri[:-1, 0], tri[:-1, 1], color=PALE, zorder=0)
    ax.plot(tri[:, 0], tri[:, 1], color=NAVY, linewidth=0.9)
    ax.text(-0.02, -0.04, "Mode 0", ha="right", va="top", fontsize=6.2, color=SLATE)
    ax.text(1.02, -0.04, "Mode 1", ha="left", va="top", fontsize=6.2, color=SLATE)
    ax.text(0.5, np.sqrt(3) / 2 + 0.04, "Mode 2", ha="center", va="bottom", fontsize=6.2, color=SLATE)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, np.sqrt(3) / 2 + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")


def belief_trace(root: Path, out: Path) -> None:
    files = sorted((root / "experiments/socialgym2_tracer_20260904_twoagent_onehuman").glob("seed*/belief_trace.csv"))
    data = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    data = data[data.policy.isin(["tracer-persistent", "tracer-reset"])]
    grouped = data.groupby(["seed", "policy", "step"], as_index=False).agg(m0=("belief_mode_0", "mean"), m1=("belief_mode_1", "mean"), m2=("belief_mode_2", "mean"), entropy=("belief_entropy", "mean"), fallback=("fallback", "mean"))
    variation = grouped[grouped.policy == "tracer-persistent"].groupby("seed").agg(rng=("entropy", lambda x: x.max() - x.min()), fallback=("fallback", "mean"))
    seeds = variation.sort_values(["rng", "fallback"], ascending=False).head(3).index.tolist()
    fig = plt.figure(figsize=(7.16, 3.86))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.85], height_ratios=[1, 0.78], wspace=0.55, hspace=0.72)
    for i, seed in enumerate(seeds):
        ax = fig.add_subplot(gs[0, i])
        draw_simplex(ax)
        for policy, color in [("tracer-persistent", TEAL), ("tracer-reset", GOLD)]:
            sub = grouped[(grouped.seed == seed) & (grouped.policy == policy)].sort_values("step")
            p = sub[["m0", "m1", "m2"]].to_numpy(float)
            p /= np.clip(p.sum(axis=1, keepdims=True), 1e-8, None)
            xy = simplex_xy(p)
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.2, alpha=0.76)
            ax.scatter(xy[:, 0], xy[:, 1], c=sub.step, cmap="viridis", s=8, edgecolors=WHITE, linewidth=0.2, alpha=0.82)
            ax.scatter(*xy[0], facecolor=WHITE, edgecolor=color, s=30, linewidth=1.2, zorder=5)
            ax.scatter(*xy[-1], marker="X", color=color, edgecolor=WHITE, s=38, linewidth=0.5, zorder=5)
        ax.set_title(f"seed {seed}", fontsize=8.2, fontweight="bold", loc="left", pad=2)
        if i == 0:
            ax.legend(handles=[Line2D([0], [0], color=TEAL, label="Persistent", linewidth=1.5), Line2D([0], [0], color=GOLD, label="Reset", linewidth=1.5)], frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, handlelength=1.3)
        panel(ax, chr(ord("A") + i), "", None)

    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    ax.text(0.0, 0.92, "Belief path", fontsize=8.5, fontweight="bold", color=NAVY)
    ax.text(0.0, 0.72, "○ initial\n× final\n\npoint colour = time\nline colour = policy", fontsize=7.2, color=INK, linespacing=1.45)
    ax.add_patch(FancyBboxPatch((0.0, 0.08), 0.96, 0.32, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor=PALE, edgecolor="none"))
    ax.text(0.07, 0.34, "The simplex shows how\nidentity belief moves, not\nonly whether accuracy rises.", fontsize=6.8, color=INK, va="top", linespacing=1.35)
    panel(ax, "D", "", None)

    ax = fig.add_subplot(gs[1, :3])
    seed = seeds[0]
    for policy, color in [("tracer-persistent", TEAL), ("tracer-reset", GOLD)]:
        sub = grouped[(grouped.seed == seed) & (grouped.policy == policy)].sort_values("step")
        ax.plot(sub.step, sub.entropy, color=color, linewidth=1.55 if policy == "tracer-persistent" else 1.0, label=policy.replace("tracer-", "").title())
        fail = sub[sub.fallback > 0.5]
        ax.scatter(fail.step, fail.entropy, marker="|", s=35, color=RED, zorder=4)
    ax.set_xlabel("Execution step")
    ax.set_ylabel("Belief entropy")
    ax.set_title(f"Representative temporal trace · seed {seed}", fontsize=8.4, fontweight="bold", loc="left", pad=2)
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.3)
    grid(ax)
    panel(ax, "E", "", None)

    ax = fig.add_subplot(gs[1, 3])
    var = variation.loc[seeds]
    y = np.arange(len(var))
    ax.barh(y, var.rng, height=0.32, color=PURPLE, label="Entropy range")
    ax.barh(y + 0.34, var.fallback, height=0.32, color=RED, label="Fallback rate")
    ax.set_yticks(y + 0.17, [str(x) for x in var.index])
    ax.set_xlabel("Range / rate")
    ax.set_title("Case spread", fontsize=8.4, fontweight="bold", loc="left", pad=2)
    ax.legend(frameon=False, loc="lower right", handlelength=1.0)
    grid(ax, "x")
    panel(ax, "F", "", None)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.82)
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    style()
    root = args.root.resolve()
    out = args.out.resolve()
    outputs = {
        "h1": out / "fig_h1_mechanism_atlas",
        "h2": out / "fig_h2_belief_lifecycle",
        "h3": out / "fig_h3_closed_loop",
        "external": out / "fig_external_boundary",
        "qualitative": out / "fig_qualitative_belief_simplex",
    }
    h1(root, outputs["h1"])
    h2(root, outputs["h2"])
    h3(root, outputs["h3"])
    external(root, outputs["external"])
    belief_trace(root, outputs["qualitative"])
    manifest = {key: str(value) for key, value in outputs.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for key, value in outputs.items():
        print(f"{key}: {value}.pdf")
        print(f"{key}: {value}.png")


if __name__ == "__main__":
    main()
