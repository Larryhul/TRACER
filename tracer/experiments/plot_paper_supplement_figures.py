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
H2_EXTERNAL = Path("experiments/external_baseline_calibrated_20260908_v1/tracer/episode_metrics.csv")
H3_PATH = ROOT / "experiments/pilot/h3_candidate_rescue_replication_20260819/episode_metrics.csv"

ORANGE = reference.ORANGE
TEAL = "#008C88"
BLUE = "#315FBC"
GREEN = "#278B55"
PURPLE = "#8E55A5"
RED = "#C84D4D"
GRAY = "#7B8794"
PALE = "#DDE5EB"
INK = "#243447"


def style():
    reference.style()
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "axes.labelsize": 8.4,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.0,
        "axes.labelpad": 4.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "lines.solid_capstyle": "round",
    })


def frame(axis, grid=True):
    axis.set_axisbelow(True)
    axis.tick_params(length=2.3, width=0.5, pad=3)
    if grid:
        axis.grid(axis="y", color="#E3E8EC", linewidth=0.45, linestyle=(0, (1.5, 2.5)))
    for spine in axis.spines.values():
        spine.set_color("#9099A2")
        spine.set_linewidth(0.65)


def panel_label(axis, value):
    axis.text(-0.10, 1.08, value, transform=axis.transAxes, fontsize=12.5,
              fontweight="bold", color=INK, va="top")


def save(figure, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference.save(figure, OUTPUT, name)
    plt.close(figure)


def bootstrap(values, seed, repeats=2000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), *np.quantile(samples, [0.025, 0.975])


def h1_pairs():
    data = pd.read_csv(H1_PATH)
    methods = data[data.method.isin(["matched-additive", "joint"])].copy()
    keys = ["effect_mode", "model_seed", "data_seed", "split", "base_scene_id"]
    table = methods.pivot(index=keys, columns="method", values="interaction_rmse").dropna()
    table = table.rename(columns={"matched-additive": "x", "joint": "y"}).reset_index()
    return table


def plot_h1():
    data = h1_pairs()
    colors = {"none": GRAY, "additive": GREEN, "nonlinear": ORANGE}
    labels = {"none": "No interaction", "additive": "Additive", "nonlinear": "Nonlinear"}
    nonlinear = data[data.effect_mode.eq("nonlinear")].copy()
    nonlinear["gain"] = nonlinear.x - nonlinear.y
    data["gain"] = data.x - data.y
    limit = float(np.nanquantile(np.r_[nonlinear.x, nonlinear.y], 0.998) * 1.05)
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.85), gridspec_kw={"width_ratios": [1.08, 0.92]})
    figure.subplots_adjust(left=0.095, right=0.985, top=0.865, bottom=0.235, wspace=0.36)
    for axis in axes:
        frame(axis, grid=False)

    axis = axes[0]
    axis.plot([0, limit], [0, limit], color="#778594", linestyle=(0, (3, 2)), linewidth=1.0, zorder=1)
    axis.scatter(nonlinear.x, nonlinear.y, s=10, color=ORANGE, alpha=0.23, edgecolors="none", zorder=2)
    lower = nonlinear.y < nonlinear.x
    axis.text(0.05, 0.94, f"Joint lower: {lower.sum():,}/{len(nonlinear):,} ({100 * lower.mean():.1f}%)",
              transform=axis.transAxes, fontsize=7.2, color=ORANGE, va="top")
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Nonlinear response: Joint wins case by case", fontsize=8.8, pad=6)
    axis.set_xlabel("Matched-Additive pair RMSE")
    axis.set_ylabel("Joint pair RMSE")
    axis.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=ORANGE,
                                markersize=4.2, label="Nonlinear paired case"),
                         Line2D([], [], color="#778594", linestyle=(0, (3, 2)),
                                linewidth=1.0, label="y = x")],
                loc="lower right", frameon=True, fancybox=False, framealpha=0.94,
                borderpad=0.35, labelspacing=0.25)
    panel_label(axis, "a")

    axis = axes[1]
    rng = np.random.default_rng(20260908)
    positions = {"none": 0, "additive": 1, "nonlinear": 2}
    for mode in ("none", "additive", "nonlinear"):
        values = data.loc[data.effect_mode.eq(mode), "gain"].to_numpy()
        jitter = rng.normal(positions[mode], 0.055, len(values))
        axis.scatter(values, jitter, s=7 if mode != "nonlinear" else 8, color=colors[mode],
                     alpha=0.18 if mode != "nonlinear" else 0.22, edgecolors="none", zorder=2)
        mean, low, high = bootstrap(values, 20260908 + positions[mode])
        axis.errorbar(mean, positions[mode], xerr=[[mean - low], [high - mean]], fmt="none",
                      color=colors[mode], elinewidth=1.4, capsize=2.5, zorder=4)
        axis.scatter(mean, positions[mode], s=43, color=colors[mode], edgecolor="white",
                     linewidth=0.7, zorder=5)
        axis.text(mean + (0.004 if mean >= 0 else -0.004), positions[mode],
                  f"{mean:+.3f}", ha="left" if mean >= 0 else "right", va="center",
                  fontsize=7.1, color=colors[mode])
    axis.axvline(0, color="#778594", linewidth=0.8)
    axis.set_yticks(range(3), [labels[mode] for mode in ("none", "additive", "nonlinear")])
    axis.set_xlabel("Pair-RMSE gain: Matched Additive − Joint ↑")
    axis.set_ylabel("Response mechanism")
    axis.set_title("The gain is specific to non-additive response", fontsize=8.8, pad=6)
    axis.text(0.04, 0.94, "positive = Joint lower RMSE", transform=axis.transAxes,
              fontsize=7.1, color="#5D6874", va="top")
    panel_label(axis, "b")
    figure.text(0.535, 0.948, "H1 · Paired interaction-RMSE evidence · lower prediction error is better",
                ha="center", fontsize=8.1, color=INK)
    save(figure, "fig_h1_pair_rmse_supplement")
    data.to_csv(OUTPUT / "h1_pair_rmse_records.csv", index=False)


def load_h2():
    columns = ["split", "episode_id", "entity_id", "window", "method", "active", "main_evaluation",
               "posterior_entropy", "predictive_trajectory_nll"]
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
    external = pd.read_csv(H2_EXTERNAL) if H2_EXTERNAL.exists() else pd.DataFrame()
    methods = {"persistent": (TEAL, "Persistent", "-", "o"),
               "reset": (ORANGE, "Reset", "--", "s"),
               "shuffled-id": (GRAY, "Shuffled ID", ":", "^")}
    figure, axes = plt.subplots(2, 1, figsize=(7.16, 5.85), gridspec_kw={"height_ratios": [1.05, 0.95]})
    figure.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.15, hspace=0.55)
    axis = axes[0]
    frame(axis)
    initial_entropy = np.log(3.0)
    for method, (color, label, line, marker) in methods.items():
        subset = data[data.method.eq(method)]
        grouped = subset.groupby(["split", "window"])
        for split, linestyle_alpha in [("id", 1.0), ("ood", 0.65)]:
            points = []
            for window in sorted(subset.window.unique()):
                values = subset[(subset.split.eq(split)) & (subset.window.eq(window))]
                if values.empty:
                    continue
                # Evidence gain is defined relative to a uniform three-mode prior.
                gain = initial_entropy - values.posterior_entropy.to_numpy()
                mean, low, high = bootstrap(gain, 20260908 + window)
                points.append((window, mean, low, high))
            if not points:
                continue
            points = np.asarray(points)
            axis.plot(points[:, 0], points[:, 1], color=color, linestyle=line, linewidth=1.8 if method == "persistent" else 1.0,
                      alpha=linestyle_alpha, marker=marker, markersize=3.8 if method == "persistent" else 3.0,
                      markerfacecolor=color if method == "persistent" else "white", markeredgewidth=0.7,
                      label=f"{label} · {split.upper()}" if method != "shuffled-id" or split == "id" else None)
            axis.fill_between(points[:, 0], points[:, 2], points[:, 3], color=color,
                              alpha=0.10 if method == "persistent" else 0.045, linewidth=0)
    axis.set_xlim(0.7, 7.3)
    axis.set_xticks(range(1, 8))
    axis.set_xlabel("Observation window")
    axis.set_ylabel("Identity evidence gain\n(log 3 − posterior entropy) ↑")
    axis.set_title("Persistent belief retains evidence across executed windows", fontsize=9.0, pad=6)
    axis.legend(loc="upper left", ncol=3, frameon=True, fancybox=False, framealpha=0.94,
                borderpad=0.35, labelspacing=0.25, columnspacing=0.8)
    panel_label(axis, "a")

    axis = axes[1]
    frame(axis)
    if external.empty:
        axis.text(0.5, 0.5, "External replanning audit unavailable", ha="center", va="center", transform=axis.transAxes)
    else:
        # These are direct planner-side audit quantities, not navigation superiority claims.
        values = {
            "Prediction NLL gain": external.belief_prediction_nll_gain.to_numpy(float),
            "Score change": external.belief_selected_total_delta.to_numpy(float),
            "Selection change": external.belief_selection_changed.to_numpy(float),
        }
        y = np.arange(len(values))
        for position, (label, array) in enumerate(values.items()):
            array = array[np.isfinite(array)]
            mean, low, high = bootstrap(array, 20261000 + position)
            axis.errorbar(mean, position, xerr=[[mean - low], [high - mean]], fmt="none",
                          color=TEAL, elinewidth=1.5, capsize=2.8, zorder=3)
            axis.scatter(mean, position, s=45, color=TEAL, edgecolor="white", linewidth=0.7, zorder=4)
            if label == "Selection change":
                axis.text(mean + 0.0008, position, f"{100 * mean:.2f}%", va="center", fontsize=7.1, color=TEAL)
            else:
                axis.text(mean + max(abs(mean) * 0.03, 0.0001), position, f"{mean:.4f}", va="center", fontsize=7.1, color=TEAL)
        axis.axvline(0, color="#778594", linewidth=0.8)
        axis.set_yticks(y, list(values))
        axis.set_xlabel("Belief-induced planner change (mean ± 95% case bootstrap)")
        axis.set_title("Executed replanning: evidence reaches the planner", fontsize=9.0, pad=6)
        axis.text(0.02, 0.08, "Validation audit; direct planner attribution, not a universal navigation claim",
                  transform=axis.transAxes, fontsize=6.9, color="#5D6874")
    panel_label(axis, "b")
    figure.text(0.535, 0.956, "H2 · Identity-bound evidence and executed replanning",
                ha="center", fontsize=8.2, color=INK)
    save(figure, "fig_h2_evidence_replanning_supplement")


def plot_h3():
    data = pd.read_csv(H3_PATH)
    policies = ["task-only", "reset-joint", "persistent-joint", "oracle-joint"]
    response = ["reset-joint", "persistent-joint", "oracle-joint"]
    labels = {"task-only": "Task-only", "reset-joint": "Reset-Joint", "persistent-joint": "Persistent-Joint", "oracle-joint": "Oracle-Joint"}
    colors = {"task-only": GRAY, "reset-joint": GREEN, "persistent-joint": ORANGE, "oracle-joint": BLUE}
    summary = data[data.policy.isin(policies)].groupby(["split", "policy"], as_index=False).agg(
        task_progress=("task_progress", "mean"), safety_cost=("safety_cost", "mean"),
        human_cost=("human_cost", "mean"), collision_rate=("collision_rate", "mean"),
        realized_total_cost=("realized_total_cost", "mean"), final_mode_accuracy=("final_mode_accuracy", "mean"),
        oracle_regret=("oracle_regret", "mean"))
    index = summary.set_index(["split", "policy"])
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.95))
    figure.subplots_adjust(left=0.095, right=0.985, top=0.865, bottom=0.22, wspace=0.35)
    for axis in axes:
        frame(axis)
    axis = axes[0]
    for split, marker, y in (("id", "o", 2.0), ("ood", "s", 1.0)):
        base = index.loc[(split, "task-only")]
        for offset, policy in enumerate(response):
            current = index.loc[(split, policy)]
            joined = data[(data.split.eq(split)) & data.policy.isin(["task-only", policy])].pivot(
                index=["seed", "episode"], columns="policy", values="realized_total_cost").dropna()
            gains = (joined["task-only"] - joined[policy]).to_numpy()
            mean, low, high = bootstrap(gains, 20260908 + offset + (100 if split == "ood" else 0))
            position = y + (offset - 1) * 0.18
            axis.errorbar(mean, position, xerr=[[mean - low], [high - mean]], fmt="none",
                          color="#A9B4BE" if split == "ood" else colors[policy], elinewidth=1.2,
                          capsize=2.2, zorder=2)
            axis.scatter(mean, position, s=37, marker=marker, color=colors[policy],
                         edgecolor="white", linewidth=0.7, zorder=4)
    axis.axvline(0, color="#778594", linewidth=0.8)
    axis.set_yticks([1, 2], ["OOD", "ID"])
    axis.set_xlabel("Task-only realized cost − policy realized cost ↑")
    axis.set_ylabel("Split")
    axis.set_title("Response-aware policies lower paired closed-loop cost", fontsize=9.0, pad=6)
    axis.text(0.03, 0.94, "circle = ID · square = OOD", transform=axis.transAxes,
              fontsize=6.9, color="#5D6874", va="top")
    axis.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=colors[p], markersize=4.5, label=labels[p]) for p in response],
                loc="lower right", frameon=True, fancybox=False, framealpha=0.94,
                borderpad=0.3, labelspacing=0.2)
    panel_label(axis, "a")

    axis = axes[1]
    subset = summary[summary.policy.isin(response)].copy()
    base = summary[summary.policy.eq("task-only")].set_index("split")
    metrics = []
    for _, row in subset.iterrows():
        reference_row = base.loc[row.split]
        metrics.append({
            "split": row.split, "policy": row.policy,
            "task": 100 * (row.task_progress - reference_row.task_progress) / reference_row.task_progress,
            "safety": 100 * (reference_row.safety_cost - row.safety_cost) / reference_row.safety_cost,
            "human": 100 * (reference_row.human_cost - row.human_cost) / reference_row.human_cost,
            "collision": 100 * (reference_row.collision_rate - row.collision_rate) / reference_row.collision_rate,
        })
    change = pd.DataFrame(metrics)
    x = np.arange(4)
    width = 0.12
    metric_keys = ["task", "safety", "human", "collision"]
    metric_labels = ["Task\nprogress", "Safety\ncost", "Human\ncost", "Collision\nrate"]
    for split_id, split in enumerate(("id", "ood")):
        for policy_id, policy in enumerate(response):
            row = change[(change.split.eq(split)) & change.policy.eq(policy)].iloc[0]
            axis.bar(x + (split_id * 3 + policy_id - 1) * width,
                     [row[key] for key in metric_keys], width=width, color=colors[policy],
                     alpha=0.9 if split == "id" else 0.48,
                     label=f"{labels[policy]} · {split.upper()}")
    axis.axhline(0, color="#778594", linewidth=0.8)
    axis.set_xticks(x, metric_labels)
    axis.set_ylabel("Relative change vs. Task-only (%)")
    axis.set_title("The gain comes from safety and social costs", fontsize=9.0, pad=6)
    axis.legend(fontsize=6.1, ncol=2, loc="upper left", frameon=True, fancybox=False,
                framealpha=0.94, borderpad=0.3, labelspacing=0.2, columnspacing=0.5)
    panel_label(axis, "b")
    figure.text(0.535, 0.956, "H3 · Closed-loop response-aware operating point · paired episodes",
                ha="center", fontsize=8.2, color=INK)
    save(figure, "fig_h3_closed_loop_supplement")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=("all", "h1", "h2", "h3"), default="all")
    args = parser.parse_args()
    style()
    if args.figure in ("all", "h1"):
        plot_h1()
    if args.figure in ("all", "h2"):
        plot_h2()
    if args.figure in ("all", "h3"):
        plot_h3()


if __name__ == "__main__":
    main()
