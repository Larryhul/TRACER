from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "reports" / "paper_visualizations_20260905"

COLORS = {
    "navy": "#143B5D",
    "blue": "#2878B5",
    "teal": "#2A9D8F",
    "green": "#4E9F70",
    "orange": "#E76F51",
    "gold": "#E9C46A",
    "violet": "#6C5CE7",
    "red": "#C94C4C",
    "ink": "#22313F",
    "muted": "#6B7785",
    "light": "#EEF3F6",
    "grid": "#DCE5EA",
    "white": "#FFFFFF",
}

METHOD_COLORS = {
    "history": "#9AA5B1",
    "nearest": "#D38B5D",
    "independent": "#8C9AA7",
    "additive": "#E9C46A",
    "matched-additive": "#F4A261",
    "joint": "#2A9D8F",
    "shuffled-joint": "#C94C4C",
    "no-belief": "#9AA5B1",
    "reset": "#E9C46A",
    "persistent": "#2A9D8F",
    "shuffled-id": "#C94C4C",
    "recurrent": "#6C5CE7",
    "oracle": "#143B5D",
    "bayesian-hazard": "#E76F51",
}

POLICY_COLORS = {
    "task-only": "#9AA5B1",
    "no-belief-joint": "#D38B5D",
    "reset-additive": "#E9C46A",
    "reset-joint": "#F4A261",
    "persistent-additive": "#7BB7A4",
    "persistent-joint": "#2A9D8F",
    "recurrent-joint": "#6C5CE7",
    "oracle-joint": "#143B5D",
}

FAMILY_LABELS = {
    "lateral": "Lateral",
    "pincer": "Pincer",
    "speed": "Speed",
    "threshold": "Threshold",
}

METHOD_LABELS = {
    "history": "History",
    "nearest": "Nearest",
    "independent": "Independent",
    "additive": "Additive",
    "matched-additive": "Matched additive",
    "joint": "Joint",
    "joint-without-pair": "Joint w/o pair",
    "shuffled-joint": "Shuffled joint",
    "no-belief": "No belief",
    "reset": "Reset",
    "persistent": "Persistent",
    "shuffled-id": "Shuffled ID",
    "recurrent": "Recurrent",
    "oracle": "Oracle",
    "bayesian-hazard": "Bayesian hazard",
}

POLICY_LABELS = {
    "task-only": "Task-only",
    "no-belief-joint": "No-belief joint",
    "reset-additive": "Reset additive",
    "reset-joint": "Reset joint",
    "persistent-additive": "Persistent additive",
    "persistent-joint": "Persistent joint",
    "recurrent-joint": "Recurrent joint",
    "oracle-joint": "Oracle joint",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "axes.facecolor": COLORS["white"],
            "figure.facecolor": COLORS["white"],
            "savefig.facecolor": COLORS["white"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", color=COLORS["navy"], va="top")


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def bootstrap_ci(values: Iterable[float], seed: int = 0, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def normalized_matrix(df: pd.DataFrame, row: str, column: str, value: str) -> pd.DataFrame:
    table = df.pivot_table(index=row, columns=column, values=value, aggfunc="mean")
    return table


def plot_h1_atlas(h1_path: Path, contrast_path: Path, out: Path) -> None:
    records = read_csv(h1_path)
    if contrast_path.suffix.lower() == ".csv" and contrast_path.exists():
        contrasts = read_csv(contrast_path)
    else:
        contrasts = pd.DataFrame()
    enhanced = "effect_mode" in records.columns and "target_family" not in records.columns
    if enhanced:
        methods = ["additive", "joint-without-pair", "matched-additive", "joint", "shuffled-joint"]
        families = ["none", "additive", "nonlinear"]
        contrast_rows = []
        pair_keys = [key for key in ["effect_mode", "model_seed", "data_seed", "split", "base_scene_id"] if key in records.columns]
        grouped = records.groupby(pair_keys + ["method"], as_index=False)[["nll", "interaction_rmse"]].mean()
        for mode in families:
            mode_data = grouped[grouped["effect_mode"] == mode]
            for method_a, method_b, name in [
                ("matched-additive", "joint", "matched_additive_minus_joint_nll"),
                ("matched-additive", "joint", "matched_additive_minus_joint_interaction_rmse"),
                ("shuffled-joint", "joint", "shuffled_minus_joint_nll"),
            ]:
                left = mode_data[mode_data["method"] == method_a].set_index(pair_keys)
                right = mode_data[mode_data["method"] == method_b].set_index(pair_keys)
                metric = "interaction_rmse" if "interaction_rmse" in name else "nll"
                joined = left[[metric]].join(right[[metric]], lsuffix="_left", rsuffix="_right", how="inner")
                values = (joined[f"{metric}_left"] - joined[f"{metric}_right"]).to_numpy()
                low, high = bootstrap_ci(values, seed=17 + len(contrast_rows))
                contrast_rows.append({"target_family": mode, "contrast": name, "n_scenes": len(values), "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "ci95_low": low, "ci95_high": high})
        contrasts = pd.DataFrame(contrast_rows)
        family_labels = {"none": "No interaction", "additive": "Additive", "nonlinear": "Nonlinear"}
    else:
        methods = ["history", "nearest", "independent", "additive", "matched-additive", "joint", "shuffled-joint"]
        families = ["lateral", "pincer", "speed", "threshold"]
        family_labels = FAMILY_LABELS
    fig = plt.figure(figsize=(11.2, 7.1), constrained_layout=False)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.38, hspace=0.48)

    ax = fig.add_subplot(grid[0, :2])
    family_key = "effect_mode" if enhanced else "target_family"
    table = normalized_matrix(records[records["method"].isin(methods)], family_key, "method", "interaction_rmse")
    table = table.reindex(index=families, columns=methods)
    row_min = table.min(axis=1)
    row_max = table.max(axis=1)
    rel = table.sub(row_min, axis=0).div((row_max - row_min).replace(0, 1), axis=0)
    cmap = LinearSegmentedColormap.from_list("h1risk", [COLORS["teal"], COLORS["gold"], COLORS["red"]])
    im = ax.imshow(rel.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=28, ha="right")
    ax.set_yticks(range(len(families)), [family_labels[f] for f in families])
    ax.set_title("Interaction error landscape across response mechanisms", loc="left", pad=10, fontweight="bold")
    ax.set_xlabel("Response model")
    ax.set_ylabel("Mechanism" if enhanced else "Intervention family")
    for i, family in enumerate(families):
        for j, method in enumerate(methods):
            val = table.loc[family, method]
            if pd.notna(val):
                color = COLORS["white"] if rel.loc[family, method] > 0.60 else COLORS["ink"]
                precision = 3 if enhanced else 2
                ax.text(j, i, f"{val:.{precision}f}", ha="center", va="center", fontsize=7.2, color=color, fontweight="bold" if method == "joint" else "normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Within-family relative error", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    add_panel_label(ax, "A")

    ax = fig.add_subplot(grid[0, 2])
    pivot = records[records["method"].isin(["matched-additive", "joint"])].pivot_table(index=family_key, columns="method", values="nll", aggfunc="mean").reindex(families)
    x = np.arange(len(families))
    for method, offset in [("matched-additive", -0.13), ("joint", 0.13)]:
        vals = pivot[method].to_numpy()
        bars = ax.bar(x + offset, vals, width=0.25, label=METHOD_LABELS[method], color=METHOD_COLORS[method], alpha=0.92)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=6.8, rotation=90)
    ax.set_xticks(x, [family_labels[f] for f in families], rotation=28, ha="right")
    ax.set_ylabel("NLL ↓")
    ax.set_title("Joint vs matched additive", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right", handlelength=1.2)
    clean_axis(ax)
    add_panel_label(ax, "B")

    ax = fig.add_subplot(grid[1, :2])
    selected = contrasts[contrasts["contrast"].isin(["matched_additive_minus_joint_nll", "matched_additive_minus_joint_interaction_rmse", "shuffled_minus_joint_nll"])].copy()
    selected["metric"] = selected["contrast"].map({
        "matched_additive_minus_joint_nll": "NLL: matched additive − joint",
        "matched_additive_minus_joint_interaction_rmse": "Interaction RMSE: matched additive − joint",
        "shuffled_minus_joint_nll": "NLL: shuffled joint − joint",
    })
    order = ["NLL: shuffled joint − joint", "NLL: matched additive − joint", "Interaction RMSE: matched additive − joint"]
    y_positions = []
    for i, metric in enumerate(order):
        sub = selected[selected["metric"] == metric]
        for j, family in enumerate(families):
            row = sub[sub["target_family"] == family].iloc[0]
            y = i * 1.25 + j * 0.22
            y_positions.append((y, metric, family, row))
            color = COLORS["red"] if "shuffled" in metric else COLORS["teal"]
            ax.plot([row["ci95_low"], row["ci95_high"]], [y, y], color=color, linewidth=2.2, solid_capstyle="round")
            ax.scatter(row["mean"], y, s=34, color=color, edgecolor=COLORS["white"], linewidth=0.7, zorder=3)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_yticks([0.33, 1.58, 2.83], ["Shuffled-joint NLL", "Matched-additive NLL", "Interaction RMSE"])
    ax.set_xlabel("Paired contrast (95% CI)")
    ax.set_title("Paired intervention evidence, not universal dominance", loc="left", fontweight="bold")
    ax.set_ylim(-0.28, 3.25)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.grid(axis="y", visible=False)
    add_panel_label(ax, "C")

    ax = fig.add_subplot(grid[1, 2])
    metric_rows = records[records["method"].isin(["joint", "shuffled-joint"])].copy()
    metric_rows["method_label"] = metric_rows["method"].map(METHOD_LABELS)
    means = metric_rows.groupby("method_label")["interaction_rmse"].mean().reindex(["Joint", "Shuffled joint"])
    ax.barh(range(2), means, color=[COLORS["teal"], COLORS["red"]], height=0.48)
    ax.set_yticks(range(2), means.index)
    ax.invert_yaxis()
    ax.set_xlabel("Interaction RMSE ↓")
    ax.set_title("Action permutation negative control", loc="left", fontweight="bold")
    for i, value in enumerate(means):
        ax.text(value + 0.01, i, f"{value:.2f}", va="center", fontsize=8)
    clean_axis(ax)
    add_panel_label(ax, "D")
    fig.suptitle("H1 · Joint-action-conditioned response modeling", x=0.04, ha="left", y=0.975, fontsize=14, fontweight="bold", color=COLORS["navy"])
    fig.text(0.04, 0.935, "The visual emphasis is the mechanism test: changing the team action changes the predicted response, while shuffled actions break that relation.", fontsize=8.5, color=COLORS["muted"])
    fig.subplots_adjust(top=0.82, bottom=0.08)
    save_figure(fig, out)


def sankey(ax: plt.Axes, flows: dict[tuple[int, int], float], title: str, subtitle: str) -> None:
    modes = [0, 1, 2]
    mode_names = ["Mode 0", "Mode 1", "Mode 2"]
    left = np.array([sum(v for (a, _), v in flows.items() if a == m) for m in modes], dtype=float)
    right = np.array([sum(v for (_, b), v in flows.items() if b == m) for m in modes], dtype=float)
    total = max(left.sum(), 1.0)
    left_y = np.linspace(0.80, 0.20, 3)
    right_y = np.linspace(0.80, 0.20, 3)
    height_left = left / total * 0.64
    height_right = right / total * 0.64
    for y, h, mode in zip(left_y, height_left, modes):
        ax.add_patch(FancyBboxPatch((0.02, y - h / 2), 0.12, h, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=COLORS["light"], edgecolor=COLORS["navy"], linewidth=0.7))
        ax.text(0.08, y, f"{mode_names[mode]}\n{left[mode]/total:.0%}", ha="center", va="center", fontsize=6.8)
    for y, h, mode in zip(right_y, height_right, modes):
        ax.add_patch(FancyBboxPatch((0.86, y - h / 2), 0.12, h, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=COLORS["light"], edgecolor=COLORS["teal"], linewidth=0.7))
        ax.text(0.92, y, f"{mode_names[mode]}\n{right[mode]/total:.0%}", ha="center", va="center", fontsize=6.8)
    left_offsets = {m: left_y[m] - height_left[m] / 2 for m in modes}
    right_offsets = {m: right_y[m] - height_right[m] / 2 for m in modes}
    for (a, b), value in sorted(flows.items()):
        h = value / total * 0.64
        y0 = left_offsets[a] + h / 2
        y1 = right_offsets[b] + h / 2
        curve = 0.27
        verts = [(0.14, y0), (0.38, y0), (0.62, y1), (0.86, y1), (0.86, y1 + h), (0.62, y1 + h), (0.38, y0 + h), (0.14, y0 + h), (0.14, y0)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        patch = PathPatch(MplPath(verts, codes), facecolor=COLORS["teal"] if a == b else COLORS["gold"], edgecolor="none", alpha=0.20 if a != b else 0.35)
        ax.add_patch(patch)
        left_offsets[a] += h
        right_offsets[b] += h
    ax.text(0.08, 0.96, "Decision", ha="center", fontsize=7.5, fontweight="bold", color=COLORS["navy"])
    ax.text(0.92, 0.96, "Posterior", ha="center", fontsize=7.5, fontweight="bold", color=COLORS["teal"])
    ax.text(0.5, 0.05, subtitle, ha="center", fontsize=7, color=COLORS["muted"])
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def plot_h2_belief_atlas(raw_root: Path, summary_path: Path, out: Path) -> None:
    raw_paths = sorted(raw_root.glob("seed*/raw_metrics.csv"))
    if not raw_paths:
        raise FileNotFoundError(f"No H2 raw metrics under {raw_root}")
    raw = pd.concat([read_csv(p) for p in raw_paths], ignore_index=True)
    summary = read_csv(summary_path)
    methods = ["no-belief", "reset", "persistent", "shuffled-id", "recurrent", "oracle", "bayesian-hazard"]
    raw = raw[raw["method"].isin(methods) & (raw["active"] == 1)].copy()
    fig = plt.figure(figsize=(11.2, 7.3))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.18, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.37, hspace=0.46)

    ax = fig.add_subplot(grid[0, :2])
    agg = raw.groupby(["split", "method", "window"], as_index=False).agg(entropy=("posterior_entropy", "mean"), confidence=("posterior_confidence", "mean"), correct=("posterior_mode_correct", "mean"))
    for split, linestyle in [("id", "-"), ("ood", "--")]:
        for method in ["reset", "persistent", "shuffled-id", "recurrent", "oracle"]:
            sub = agg[(agg["split"] == split) & (agg["method"] == method)]
            if sub.empty:
                continue
            color = METHOD_COLORS[method]
            ax.plot(sub["window"], sub["entropy"], color=color, linestyle=linestyle, linewidth=2.0 if method == "persistent" else 1.0, alpha=0.95 if method == "persistent" else 0.65, label=f"{METHOD_LABELS[method]} ({split.upper()})" if method in ["reset", "persistent", "recurrent"] else None)
    ax.set_xlabel("Planning window")
    ax.set_ylabel("Posterior entropy ↓")
    ax.set_title("Belief lifecycle: uncertainty evolves with executed evidence", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper right", handlelength=2.0, columnspacing=1.0)
    clean_axis(ax)
    add_panel_label(ax, "A")

    ax = fig.add_subplot(grid[0, 2])
    split = "id"
    sub = raw[(raw["split"] == split) & (raw["method"] == "persistent")]
    flows = sub.groupby(["decision_predicted_mode", "posterior_predicted_mode"]).size().to_dict()
    flows = {(int(a), int(b)): float(v) for (a, b), v in flows.items()}
    sankey(ax, flows, "Evidence-induced mode shifts", "Persistent · ID · all windows")
    add_panel_label(ax, "B")

    ax = fig.add_subplot(grid[1, :2])
    pivot = raw.groupby(["split", "method", "window"], as_index=False)["posterior_mode_correct"].mean().pivot_table(index=["method", "split"], columns="window", values="posterior_mode_correct")
    pivot = pivot.reindex([(m, s) for s in ["id", "ood"] for m in ["no-belief", "reset", "persistent", "shuffled-id", "recurrent", "oracle"] if (m, s) in pivot.index])
    im = ax.imshow(pivot.to_numpy(), cmap=LinearSegmentedColormap.from_list("beliefacc", [COLORS["light"], COLORS["gold"], COLORS["teal"]]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{METHOD_LABELS[m]} · {s.upper()}" for m, s in pivot.index])
    ax.set_xlabel("Planning window")
    ax.set_title("Posterior mode accuracy over time", loc="left", fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.3, color=COLORS["ink"] if value < 0.65 else COLORS["white"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Mode accuracy", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    add_panel_label(ax, "C")

    ax = fig.add_subplot(grid[1, 2])
    cal = raw[raw["method"].isin(["reset", "persistent", "recurrent"])].copy()
    cal["confidence_bin"] = pd.cut(cal["posterior_confidence"], bins=np.linspace(1 / 3, 1.0, 8), include_lowest=True)
    cal_agg = cal.groupby(["method", "confidence_bin"], observed=True).agg(confidence=("posterior_confidence", "mean"), accuracy=("posterior_mode_correct", "mean"), n=("posterior_mode_correct", "size")).reset_index()
    for method in ["reset", "persistent", "recurrent"]:
        sub = cal_agg[cal_agg["method"] == method]
        ax.plot(sub["confidence"], sub["accuracy"], marker="o", linewidth=1.7 if method == "persistent" else 1.0, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ax.plot([1 / 3, 1], [1 / 3, 1], color=COLORS["muted"], linestyle=":", linewidth=1)
    ax.set_xlim(1 / 3, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Calibration of identity-bound belief", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color=COLORS["grid"], linewidth=0.6)
    add_panel_label(ax, "D")
    fig.suptitle("H2 · Persistent, identity-bound interaction belief", x=0.04, ha="left", y=0.975, fontsize=14, fontweight="bold", color=COLORS["navy"])
    fig.text(0.04, 0.935, "Use this figure to show persistence as a lifecycle: evidence changes the posterior, not merely the final aggregate score.", fontsize=8.5, color=COLORS["muted"])
    fig.subplots_adjust(top=0.82, bottom=0.08)
    save_figure(fig, out)


def add_parallel_coordinates(ax: plt.Axes, data: pd.DataFrame, metrics: list[str], labels: list[str]) -> None:
    x = np.arange(len(metrics))
    for _, row in data.iterrows():
        vals = row[metrics].to_numpy(dtype=float)
        ax.plot(x, vals, color=POLICY_COLORS.get(row["policy"], COLORS["muted"]), alpha=0.18, linewidth=0.8)
    means = data.groupby("policy")[metrics].mean()
    for policy, row in means.iterrows():
        vals = row.to_numpy(dtype=float)
        ax.plot(x, vals, color=POLICY_COLORS.get(policy, COLORS["muted"]), linewidth=2.4 if policy == "persistent-joint" else 1.2, marker="o", markersize=3.2, label=POLICY_LABELS.get(policy, policy))
    ax.set_xticks(x, labels)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def plot_h3_closed_loop_atlas(posthoc_path: Path, episode_path: Path, out: Path) -> None:
    posthoc = read_csv(posthoc_path)
    episodes = read_csv(episode_path)
    keep = ["task-only", "reset-joint", "persistent-joint", "recurrent-joint", "oracle-joint"]
    episodes = episodes[episodes["policy"].isin(keep)].copy()
    posthoc = posthoc[posthoc["policy"].isin(keep)].copy()
    fig = plt.figure(figsize=(11.2, 7.3))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.39, hspace=0.48)

    ax = fig.add_subplot(grid[0, :2])
    cols = ["realized_total_cost", "task_progress", "oracle_regret", "safety_fallback", "candidate_rank_spearman"]
    subset = episodes.copy()
    for col in cols:
        lo, hi = subset[col].quantile(0.02), subset[col].quantile(0.98)
        subset[col + "_norm"] = ((subset[col].clip(lo, hi) - lo) / max(hi - lo, 1e-8))
    add_parallel_coordinates(ax, subset, [c + "_norm" for c in cols], ["Cost", "Progress", "Regret", "Fallback", "Rank ρ"])
    ax.set_ylabel("Within-metric normalized value")
    ax.set_title("Closed-loop operating profiles across paired episodes", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08), handlelength=2.0, columnspacing=1.0)
    add_panel_label(ax, "A")

    ax = fig.add_subplot(grid[0, 2])
    pivot = posthoc.pivot_table(index="policy", columns="split", values="realized_total_cost", aggfunc="mean").reindex(keep)
    for policy in keep:
        if policy not in pivot.index:
            continue
        y = pivot.loc[policy]
        ax.plot([0, 1], y.to_numpy(), color=POLICY_COLORS[policy], linewidth=2.0 if policy == "persistent-joint" else 1.0, marker="o")
        ax.text(1.03, y.iloc[-1], POLICY_LABELS[policy], va="center", fontsize=6.7, color=POLICY_COLORS[policy])
    ax.set_xticks([0, 1], ["ID", "OOD"])
    ax.set_ylabel("Realized cost ↓")
    ax.set_title("ID → OOD transfer", loc="left", fontweight="bold")
    clean_axis(ax)
    add_panel_label(ax, "B")

    ax = fig.add_subplot(grid[1, :2])
    sub = posthoc[posthoc["policy"].isin(["reset-joint", "persistent-joint"])].copy()
    wide = sub.pivot_table(index=["seed", "split"], columns="policy", values=["belief_selection_true_cost_delta", "belief_selection_flip_rate", "window_oracle_regret"])
    pairs = []
    metrics = [
        ("belief_selection_true_cost_delta", "Selection cost delta ↓", True),
        ("belief_selection_flip_rate", "Selection flip rate", False),
        ("window_oracle_regret", "Oracle regret ↓", True),
    ]
    positions = np.arange(len(metrics))
    for idx, (metric, label, lower_better) in enumerate(metrics):
        if (metric, "reset-joint") not in wide or (metric, "persistent-joint") not in wide:
            continue
        reset = wide[(metric, "reset-joint")].to_numpy(dtype=float)
        persistent = wide[(metric, "persistent-joint")].to_numpy(dtype=float)
        for x0, a, b in zip(np.full(len(reset), idx - 0.08), reset, persistent):
            ax.plot([x0, idx + 0.08], [a, b], color=COLORS["grid"], linewidth=0.8, alpha=0.65)
        ax.scatter(np.full(len(reset), idx - 0.08), reset, color=COLORS["gold"], s=22, alpha=0.8, edgecolor=COLORS["white"], linewidth=0.4, label="Reset" if idx == 0 else None)
        ax.scatter(np.full(len(persistent), idx + 0.08), persistent, color=COLORS["teal"], s=22, alpha=0.8, edgecolor=COLORS["white"], linewidth=0.4, label="Persistent" if idx == 0 else None)
        ax.errorbar([idx - 0.08, idx + 0.08], [np.nanmean(reset), np.nanmean(persistent)], yerr=[np.nanstd(reset), np.nanstd(persistent)], fmt="none", ecolor=COLORS["ink"], capsize=3, linewidth=1.2)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_xticks(positions, [m[1] for m in metrics], rotation=18, ha="right")
    ax.set_ylabel("Per-seed paired aggregate")
    ax.set_title("Belief-policy contrasts: show direction and variability", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    add_panel_label(ax, "C")

    ax = fig.add_subplot(grid[1, 2])
    x = posthoc["belief_entropy"].to_numpy(dtype=float)
    y = posthoc["belief_selection_true_cost_delta"].to_numpy(dtype=float)
    c = (posthoc["split"] == "ood").astype(int)
    size = 24 + 450 * posthoc["belief_selection_flip_rate"].to_numpy(dtype=float)
    scatter = ax.scatter(x, y, c=c, s=size, cmap=LinearSegmentedColormap.from_list("split", [COLORS["teal"], COLORS["orange"]]), alpha=0.75, edgecolor=COLORS["white"], linewidth=0.5)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_xlabel("Belief entropy")
    ax.set_ylabel("True cost delta on selected flips ↓")
    ax.set_title("When belief changes selection", loc="left", fontweight="bold")
    ax.text(0.03, 0.96, "size = flip rate", transform=ax.transAxes, va="top", fontsize=7, color=COLORS["muted"])
    ax.text(0.03, 0.88, "teal ID · orange OOD", transform=ax.transAxes, va="top", fontsize=7, color=COLORS["muted"])
    ax.grid(color=COLORS["grid"], linewidth=0.6)
    add_panel_label(ax, "D")
    fig.suptitle("H3 · Belief-aware execute–observe–update–replan", x=0.04, ha="left", y=0.975, fontsize=14, fontweight="bold", color=COLORS["navy"])
    fig.text(0.04, 0.935, "The figure deliberately keeps mixed outcomes visible: it diagnoses where persistent belief changes decisions and where the task/safety objective dominates.", fontsize=8.5, color=COLORS["muted"])
    fig.subplots_adjust(top=0.80, bottom=0.08)
    save_figure(fig, out)


def _simplex_xy(probabilities: np.ndarray) -> np.ndarray:
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    return probabilities @ vertices


def _draw_simplex(ax: plt.Axes) -> None:
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0], [0.0, 0.0]])
    ax.plot(vertices[:, 0], vertices[:, 1], color=COLORS["navy"], linewidth=1.0)
    ax.fill(vertices[:-1, 0], vertices[:-1, 1], color=COLORS["light"], alpha=0.72)
    ax.text(-0.03, -0.05, "Mode 0", ha="right", va="top", fontsize=7.2, color=COLORS["muted"])
    ax.text(1.03, -0.05, "Mode 1", ha="left", va="top", fontsize=7.2, color=COLORS["muted"])
    ax.text(0.5, np.sqrt(3.0) / 2.0 + 0.05, "Mode 2", ha="center", va="bottom", fontsize=7.2, color=COLORS["muted"])
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.10, np.sqrt(3.0) / 2.0 + 0.10)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_belief_trace_atlas(trace_root: Path, out: Path) -> None:
    paths = sorted(trace_root.glob("seed*/belief_trace.csv"))
    if not paths:
        raise FileNotFoundError(f"No belief traces under {trace_root}")
    raw = pd.concat([read_csv(path) for path in paths], ignore_index=True)
    policies = ["tracer-persistent", "tracer-reset"]
    raw = raw[raw["policy"].isin(policies)].copy()
    if raw.empty:
        raise ValueError("Belief traces do not contain persistent/reset policies")
    grouped = raw.groupby(["policy", "seed", "step"], as_index=False).agg(
        belief_mode_0=("belief_mode_0", "mean"),
        belief_mode_1=("belief_mode_1", "mean"),
        belief_mode_2=("belief_mode_2", "mean"),
        belief_entropy=("belief_entropy", "mean"),
        fallback=("fallback", "mean"),
        observed=("observed", "mean"),
        min_distance=("min_distance", "mean"),
    )
    seeds = sorted(grouped["seed"].unique())
    variation = grouped[grouped["policy"] == "tracer-persistent"].groupby("seed").agg(
        entropy_range=("belief_entropy", lambda values: float(values.max() - values.min())),
        fallback_rate=("fallback", "mean"),
        final_entropy=("belief_entropy", "last"),
    )
    selected_seeds = variation.sort_values(["entropy_range", "fallback_rate"], ascending=False).head(3).index.tolist()
    selected_seeds = [int(seed) for seed in selected_seeds]
    while len(selected_seeds) < 3:
        selected_seeds.append(int(seeds[len(selected_seeds) % len(seeds)]))

    fig = plt.figure(figsize=(11.2, 7.0))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.95], height_ratios=[1.0, 0.92], wspace=0.35, hspace=0.46)
    for col, seed in enumerate(selected_seeds):
        ax = fig.add_subplot(grid[0, col])
        _draw_simplex(ax)
        for policy, color in [("tracer-persistent", COLORS["teal"]), ("tracer-reset", COLORS["gold"])]:
            sub = grouped[(grouped["seed"] == seed) & (grouped["policy"] == policy)].sort_values("step")
            if sub.empty:
                continue
            probabilities = sub[["belief_mode_0", "belief_mode_1", "belief_mode_2"]].to_numpy(dtype=float)
            probabilities = probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), 1e-8, None)
            xy = _simplex_xy(probabilities)
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.5, alpha=0.65, label=METHOD_LABELS.get(policy.replace("tracer-", ""), policy))
            points = ax.scatter(xy[:, 0], xy[:, 1], c=sub["step"], cmap="viridis", s=13, alpha=0.78, edgecolor=COLORS["white"], linewidth=0.25)
            ax.scatter(xy[0, 0], xy[0, 1], marker="o", s=35, color=COLORS["white"], edgecolor=color, linewidth=1.4, zorder=4)
            ax.scatter(xy[-1, 0], xy[-1, 1], marker="X", s=45, color=color, edgecolor=COLORS["white"], linewidth=0.6, zorder=4)
        ax.set_title(f"Seed {seed}", loc="left", fontweight="bold")
        if col == 0:
            ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, handlelength=1.5)
        add_panel_label(ax, chr(ord("A") + col))

    ax = fig.add_subplot(grid[0, 3])
    ax.axis("off")
    ax.text(0.02, 0.96, "How to read", transform=ax.transAxes, va="top", fontsize=9.5, fontweight="bold", color=COLORS["navy"])
    ax.text(0.02, 0.82, "Each point is one planning step.\n\n○ initial belief\n✕ final belief\n\nColour = time\nEdge colour = policy", transform=ax.transAxes, va="top", fontsize=8.0, color=COLORS["ink"], linespacing=1.55)
    ax.add_patch(Rectangle((0.02, 0.05), 0.88, 0.21, facecolor=COLORS["light"], edgecolor="none", transform=ax.transAxes))
    ax.text(0.06, 0.23, "The simplex makes belief movement\nvisible even when aggregate accuracy\nis mixed.", transform=ax.transAxes, va="top", fontsize=7.5, color=COLORS["ink"], linespacing=1.45)
    add_panel_label(ax, "D")

    ax = fig.add_subplot(grid[1, :3])
    representative_seed = selected_seeds[0]
    for col, seed in enumerate(selected_seeds):
        if seed != representative_seed:
            continue
        sub = grouped[grouped["seed"] == seed]
        for policy, color in [("tracer-persistent", COLORS["teal"]), ("tracer-reset", COLORS["gold"])]:
            policy_sub = sub[sub["policy"] == policy].sort_values("step")
            if policy_sub.empty:
                continue
            label = f"{policy.replace('tracer-', '').title()} · {seed}" if col == 0 else None
            ax.plot(policy_sub["step"], policy_sub["belief_entropy"], color=color, linewidth=2.0 if policy == "tracer-persistent" else 1.1, alpha=0.78, label=label)
            fallback_steps = policy_sub.loc[policy_sub["fallback"] > 0.5, "step"]
            ax.scatter(fallback_steps, policy_sub.set_index("step").loc[fallback_steps, "belief_entropy"], color=COLORS["red"], marker="|", s=42, alpha=0.65)
    ax.set_xlabel("Step")
    ax.set_ylabel("Belief entropy")
    ax.set_title(f"Representative temporal trace and fallback events · seed {representative_seed}", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(color=COLORS["grid"], linewidth=0.6)
    add_panel_label(ax, "E")

    ax = fig.add_subplot(grid[1, 3])
    profile = variation.loc[[seed for seed in selected_seeds if seed in variation.index]].copy()
    y = np.arange(len(profile))
    ax.barh(y - 0.17, profile["entropy_range"], height=0.30, color=COLORS["violet"], label="Entropy range")
    ax.barh(y + 0.17, profile["fallback_rate"], height=0.30, color=COLORS["red"], label="Fallback rate")
    ax.set_yticks(y, [str(seed) for seed in profile.index])
    ax.invert_yaxis()
    ax.set_xlabel("Rate / range")
    ax.set_title("Case selection", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    add_panel_label(ax, "F")
    fig.suptitle("Qualitative belief traces · Movement in the interaction simplex", x=0.04, ha="left", y=0.975, fontsize=14, fontweight="bold", color=COLORS["navy"])
    fig.text(0.04, 0.935, "Selected cases maximize trace variation rather than selecting only the best-performing episode.", fontsize=8.5, color=COLORS["muted"])
    fig.subplots_adjust(top=0.82, bottom=0.10)
    save_figure(fig, out)


def _extract_markdown_table(text: str, header_fragment: str) -> pd.DataFrame | None:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if header_fragment.lower() not in line.lower() or "|" not in line:
            continue
        block = []
        for candidate in lines[i:]:
            if "|" not in candidate:
                break
            block.append(candidate)
        if len(block) < 3:
            continue
        rows = []
        for row in block:
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            if all(set(cell) <= set("-: ") for cell in cells):
                continue
            rows.append(cells)
        if len(rows) >= 2:
            width = len(rows[0])
            rows = [r for r in rows if len(r) == width]
            return pd.DataFrame(rows[1:], columns=rows[0])
    return None


def load_external_summary(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = [
        (
            "SICNav same-case",
            root / "experiments" / "external_dynamic_samecase_holdout_20260904_orca_fallback_sicnav_v2" / "episode_metrics.csv",
        ),
        (
            "CrowdSimPlus holdout",
            root / "experiments" / "external_validation_dev" / "crowdsimplus_final_static010_safety4_margin08_holdout16_31_cuda_20260904" / "episode_metrics.csv",
        ),
        (
            "SICNav FRM native",
            root / "experiments" / "external_sicnav_domainfrm_native_multiseed_20260905" / "episode_metrics.csv",
        ),
        (
            "SICNav FRM hybrid",
            root / "experiments" / "external_sicnav_domainfrm_hybrid_multiseed_20260905" / "episode_metrics.csv",
        ),
    ]
    for source_name, path in candidates:
        if not path.exists():
            continue
        data = read_csv(path)
        policy_col = "policy" if "policy" in data.columns else None
        if policy_col is None:
            continue
        for policy, sub in data.groupby(policy_col):
            item = {"source": source_name, "policy": str(policy), "n": len(sub)}
            for source_col, target_col in [("success", "success"), ("collision", "collision"), ("task_progress", "progress"), ("min_distance", "min_distance"), ("fallback_rate", "fallback")]:
                if source_col in sub:
                    item[target_col] = float(sub[source_col].mean())
            if "collision_free_success" in sub:
                item["collision_free_success"] = float(sub["collision_free_success"].mean())
            rows.append(item)
    socialgym_root = root / "experiments" / "socialgym2_tracer_20260904_twoagent_onehuman"
    socialgym_paths = sorted(socialgym_root.glob("seed*/episode_metrics.csv"))
    if socialgym_paths:
        data = pd.concat([read_csv(path) for path in socialgym_paths], ignore_index=True)
        for policy, sub in data.groupby("policy"):
            rows.append(
                {
                    "source": "SocialGym2 2R/1H",
                    "policy": str(policy),
                    "n": len(sub),
                    "success": float(sub["success"].mean()),
                    "collision": float(sub["collision"].mean()),
                    "progress": float(sub["task_progress"].mean()),
                    "min_distance": float(sub["min_distance"].mean()),
                    "fallback": float(sub["fallback_rate"].mean()),
                }
            )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["source", "policy", "n", "success", "collision", "progress", "min_distance", "fallback"])


def plot_external_atlas(root: Path, out: Path) -> None:
    data = load_external_summary(root)
    if data.empty:
        raise FileNotFoundError("No external episode_metrics.csv files found")
    fig = plt.figure(figsize=(11.2, 6.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.36, hspace=0.45)

    ax = fig.add_subplot(grid[0, 0])
    plot_data = data[(data["source"] == "SICNav same-case") & data["progress"].notna() & data["collision"].notna()].copy()
    label_offsets = {
        "official-dwa": (6, 6),
        "official-orca": (6, 4),
        "official-mpc-cvmm": (6, -12),
        "official-sicnav-np": (6, 10),
        "official-sicnav-p": (6, -16),
        "official-linear": (6, -12),
        "tracer-persistent": (6, 10),
        "tracer-reset": (6, -14),
        "tracer-hazard": (6, 22),
        "tracer-forgetting": (6, -26),
        "tracer-change-point": (6, 34),
    }
    for _, row in plot_data.iterrows():
        label = row["policy"].replace("tracer-", "TRACER ").replace("official-", "Official ")
        color = COLORS["teal"] if "tracer" in row["policy"] else COLORS["navy"]
        ax.scatter(row["progress"], row["collision"], s=70 + 12 * math.sqrt(float(row.get("n", 1))), color=color, alpha=0.76, edgecolor=COLORS["white"], linewidth=0.8)
        if row["policy"] in {"official-dwa", "official-orca", "official-mpc-cvmm", "official-sicnav-np", "official-sicnav-p", "tracer-persistent"}:
            ax.annotate(label, (row["progress"], row["collision"]), xytext=label_offsets.get(row["policy"], (6, 4)), textcoords="offset points", fontsize=6.7)
    ax.set_xlabel("Task progress ↑")
    ax.set_ylabel("Collision rate ↓")
    ax.set_title("External operating points", loc="left", fontweight="bold")
    ax.grid(color=COLORS["grid"], linewidth=0.6)
    add_panel_label(ax, "A")

    ax = fig.add_subplot(grid[0, 1])
    representative = ["official-dwa", "official-orca", "official-mpc-cvmm", "official-sicnav-np", "tracer-persistent", "tracer-reset"]
    matrix_data = data[data["policy"].isin(representative)].copy()
    matrix_data["collision_free"] = 1.0 - matrix_data["collision"]
    matrix = matrix_data.pivot_table(index="source", columns="policy", values="collision_free", aggfunc="mean")
    if not matrix.empty:
        im = ax.imshow(matrix.to_numpy(), cmap=LinearSegmentedColormap.from_list("safety", [COLORS["red"], COLORS["gold"], COLORS["teal"]]), vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(matrix.columns)), [str(c).replace("tracer-", "TRACER ").replace("official-", "Official ")[:18] for c in matrix.columns], rotation=32, ha="right")
        ax.set_yticks(range(len(matrix.index)), [str(i)[:22] for i in matrix.index])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=COLORS["ink"])
        ax.set_title("Portability matrix: collision-free rate", loc="left", fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("1 − collision", fontsize=7.5)
    else:
        ax.text(0.5, 0.5, "No external matrix", ha="center", va="center")
        ax.axis("off")
    add_panel_label(ax, "B")

    ax = fig.add_subplot(grid[1, :])
    safety = plot_data[plot_data["policy"].isin(representative)].sort_values("min_distance", ascending=False).copy()
    if "min_distance" in safety:
        labels = [str(x)[:25] for x in safety["policy"]]
        y = np.arange(len(safety))
        ax.barh(y, safety["min_distance"], color=[COLORS["teal"] if "tracer" in p else COLORS["navy"] for p in safety["policy"]], alpha=0.85)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Minimum distance ↑")
        ax.set_title("Safety margin and execution footprint", loc="left", fontweight="bold")
        for yi, value in zip(y, safety["min_distance"]):
            ax.text(value + max(safety["min_distance"].max(), 1) * 0.01, yi, f"{value:.2f}", va="center", fontsize=7)
        clean_axis(ax)
    else:
        ax.text(0.5, 0.5, "Minimum-distance data unavailable", ha="center", va="center")
        ax.axis("off")
    add_panel_label(ax, "C")
    fig.suptitle("External validation · Report portability and operating boundaries", x=0.04, ha="left", y=0.975, fontsize=14, fontweight="bold", color=COLORS["navy"])
    fig.text(0.04, 0.935, "Panel A is a same-case comparison; panel B keeps protocol boundaries explicit and uses only rate-like quantities across datasets.", fontsize=8.5, color=COLORS["muted"])
    fig.subplots_adjust(top=0.82, bottom=0.10)
    save_figure(fig, out)


def draw_card(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], title: str, body: str, color: str, number: str) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor=COLORS["white"], edgecolor=color, linewidth=1.6))
    ax.add_patch(FancyBboxPatch((x + 0.018, y + h - 0.11), 0.075, 0.075, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor=color, edgecolor="none"))
    ax.text(x + 0.055, y + h - 0.072, number, ha="center", va="center", fontsize=9, color=COLORS["white"], fontweight="bold")
    ax.text(x + 0.115, y + h - 0.07, title, ha="left", va="center", fontsize=9.2, color=COLORS["navy"], fontweight="bold")
    ax.text(x + 0.025, y + h - 0.16, body, ha="left", va="top", fontsize=7.2, color=COLORS["ink"], linespacing=1.45)


def plot_experiment_storyboard(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "Experimental evidence map", fontsize=15, fontweight="bold", color=COLORS["navy"], va="top")
    ax.text(0.02, 0.875, "A layered visual grammar for the paper: mechanism → memory → closed-loop behavior → external boundary", fontsize=8.8, color=COLORS["muted"], va="top")
    cards = [
        ((0.02, 0.22), (0.215, 0.50), "H1 · Mechanism", "Paired intervention\n\nJoint action\n→ response shift\n\nEvidence: response error,\nshuffled-action control,\ninteraction contrast", COLORS["orange"], "1"),
        ((0.27, 0.22), (0.215, 0.50), "H2 · Memory", "Entity identity\n\nObserved prefix\n→ posterior update\n\nEvidence: entropy lifecycle,\nmode transition, calibration", COLORS["violet"], "2"),
        ((0.52, 0.22), (0.215, 0.50), "H3 · Closed loop", "Candidate pool\n→ execute prefix\n→ observe\n→ replan\n\nEvidence: ranking, regret,\npaired policy shifts", COLORS["teal"], "3"),
        ((0.77, 0.22), (0.215, 0.50), "External · Boundary", "CrowdSimPlus\nSICNav · CrowdNav\nSocialGym2\n\nEvidence: portability,\noperating point, failure mode", COLORS["navy"], "4"),
    ]
    for xy, wh, title, body, color, number in cards:
        draw_card(ax, xy, wh, title, body, color, number)
    for x0, x1, color in [(0.237, 0.27, COLORS["muted"]), (0.487, 0.52, COLORS["muted"]), (0.737, 0.77, COLORS["muted"])]:
        ax.add_patch(FancyArrowPatch((x0, 0.47), (x1, 0.47), arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color=color))
    ax.text(0.5, 0.095, "Do not collapse these layers into one leaderboard: each figure answers a different causal or operational question.", ha="center", va="center", fontsize=8.3, color=COLORS["ink"], bbox=dict(boxstyle="round,pad=0.35", facecolor=COLORS["light"], edgecolor="none"))
    save_figure(fig, out)


def write_manifest(out: Path, paths: dict[str, Path]) -> None:
    manifest = {name: str(path) for name, path in paths.items()}
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-oriented visualizations for the complete ICRA experiment section.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    configure_style()
    outputs: dict[str, Path] = {}
    story = out / "fig_experiment_storyboard"
    plot_experiment_storyboard(story)
    outputs["storyboard"] = story
    h1 = out / "fig_h1_interaction_atlas"
    enhanced_h1 = root / "experiments/formal_v3_enhanced_h1/evaluation_corrected/paired_records.csv"
    legacy_h1 = root / "reports/h1_unified_baselines_20260904/records.csv"
    h1_source = enhanced_h1 if enhanced_h1.exists() else legacy_h1
    h1_contrasts = root / "reports/h1_unified_baselines_20260904/contrasts.csv" if h1_source == legacy_h1 else root / "reports/h1_enhanced_formal_20260905.md"
    plot_h1_atlas(h1_source, h1_contrasts, h1)
    outputs["h1"] = h1
    h2 = out / "fig_h2_belief_lifecycle"
    plot_h2_belief_atlas(root / "experiments/h2_bayesian_hazard_audit_20260905", root / "reports/h2_bayesian_hazard_audit_20260905/summary.csv", h2)
    outputs["h2"] = h2
    h3 = out / "fig_h3_closed_loop_atlas"
    plot_h3_closed_loop_atlas(root / "reports/h3_response_diverse_full_20260905.csv", root / "experiments/b3_task_goal_ablation_full_recurrent_20260905/episode_metrics.csv", h3)
    outputs["h3"] = h3
    traces = out / "fig_belief_trace_simplex"
    plot_belief_trace_atlas(root / "experiments/socialgym2_tracer_20260904_twoagent_onehuman", traces)
    outputs["belief_trace"] = traces
    external = out / "fig_external_operating_boundary"
    plot_external_atlas(root, external)
    outputs["external"] = external
    load_external_summary(root).to_csv(out / "external_summary.csv", index=False)
    write_manifest(out, outputs)
    for name, path in outputs.items():
        print(f"{name}={path}.png")
        print(f"{name}={path}.pdf")


if __name__ == "__main__":
    main()
