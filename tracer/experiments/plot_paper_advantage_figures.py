from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from tracer.experiments import plot_paper_evidence as evidence
from tracer.experiments import plot_reference_trajectories as reference


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/paper_advantage_20260908"
H1_PATH = ROOT / "experiments/formal_v3_enhanced_h1/evaluation_corrected/paired_records.csv"
H2_ROOT = ROOT / "experiments/formal_v2/clean_chain_submission_20260904/h2"
H3_PATH = ROOT / "experiments/pilot/h3_candidate_rescue_replication_20260819/episode_metrics.csv"
ORANGE = reference.ORANGE
BLUE = "#2455C3"
GREEN = "#168A4A"
PURPLE = "#8E44AD"
RED = "#C43C39"
GRAY = "#858585"
LIGHT_GRAY = "#D9D9D9"


def figure_style():
    reference.style()
    plt.rcParams.update({
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "axes.labelpad": 4.5,
        "lines.solid_capstyle": "round",
    })


def frame(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.45, linestyle=(0, (1.5, 2.5)))
    axis.tick_params(length=2.3, width=0.5, pad=3)
    axis.set_box_aspect(1)


def caption(axis, text, y=-0.205):
    axis.text(0.5, y, text, transform=axis.transAxes, ha="center", va="top",
              fontsize=12.5, color="black", fontfamily="STIXGeneral")


def legend(axis, **kwargs):
    value = axis.legend(fontsize=7.0, frameon=True, fancybox=False, framealpha=0.96,
                        borderpad=0.35, labelspacing=0.28, handlelength=1.8,
                        handletextpad=0.5, borderaxespad=0.45, **kwargs)
    value.get_frame().set_edgecolor("#BBBBBB")
    value.get_frame().set_linewidth(0.4)
    return value


def bootstrap_interval(values, seed=20260908, resamples=2000):
    values = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(resamples, len(values)))
    samples = values[indices].mean(axis=1)
    return float(values.mean()), *np.quantile(samples, [0.025, 0.975])


def save(figure, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference.save(figure, OUTPUT, name)
    plt.close(figure)


def plot_h1():
    data = pd.read_csv(H1_PATH)
    selected = data[data.method.isin(["matched-additive", "joint"])].copy()
    keys = ["effect_mode", "data_seed", "split", "base_scene_id", "model_seed"]
    paired = selected.pivot(index=keys, columns="method", values="interaction_rmse").dropna()
    target = selected.groupby(keys).interaction_target_rms.first()
    paired["target"] = target
    paired["gain"] = paired["matched-additive"] - paired["joint"]
    paired = paired.reset_index()
    scene = paired.groupby(["effect_mode", "data_seed", "split", "base_scene_id"], as_index=False).agg(
        gain=("gain", "mean"), target=("target", "mean"))
    scene["relative_gain"] = scene.gain / scene.target.clip(lower=1e-6)
    order = ["none", "additive", "nonlinear"]
    colors = {"none": GRAY, "additive": GREEN, "nonlinear": ORANGE}
    labels = {"none": "None", "additive": "Additive", "nonlinear": "Nonlinear"}
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.8))
    figure.subplots_adjust(left=0.085, right=0.985, top=0.865, bottom=0.235, wspace=0.34)
    for axis in axes:
        frame(axis)

    rng = np.random.default_rng(20260908)
    for position, mode in enumerate(order, start=1):
        values = scene.loc[scene.effect_mode.eq(mode), "gain"].to_numpy()
        jitter = rng.normal(position, 0.055, len(values))
        axes[0].scatter(jitter, values, s=8 if mode != "nonlinear" else 11,
                        color=colors[mode], alpha=0.26 if mode != "nonlinear" else 0.42,
                        edgecolors="none", zorder=2)
        mean, low, high = bootstrap_interval(values, 20260908 + position)
        axes[0].errorbar(position, mean, yerr=[[mean - low], [high - mean]],
                         color=colors[mode], marker="o", markersize=4.8,
                         markerfacecolor=colors[mode], markeredgecolor="white",
                         markeredgewidth=0.7, capsize=2.4, linewidth=1.6, zorder=5)
        axes[0].text(position, high + 0.008, f"{mean:.3f}", ha="center", va="bottom",
                     fontsize=7.1, color=colors[mode])
    axes[0].axhline(0, color="#555555", linewidth=0.7)
    axes[0].set_xticks(range(1, 4), [labels[value] for value in order])
    axes[0].set_xlim(0.5, 3.5)
    axes[0].set_ylabel("Interaction-RMSE gain\nMatched Additive − Joint")
    axes[0].set_title("Joint advantage is concentrated in nonlinear response", fontsize=8.5, pad=6)
    axes[0].text(0.04, 0.95, "positive = Joint better", transform=axes[0].transAxes,
                 fontsize=7.0, color="#555555", va="top")

    nonlinear = scene[scene.effect_mode.eq("nonlinear")].sort_values("target")
    axes[1].scatter(nonlinear.target, nonlinear.relative_gain, s=11, color=ORANGE,
                    alpha=0.5, edgecolors="white", linewidth=0.25, label="Nonlinear scene")
    grouped = nonlinear.assign(bin=pd.qcut(nonlinear.target, 6, duplicates="drop")).groupby("bin", observed=True)
    centers = grouped.target.mean().to_numpy()
    means = grouped.relative_gain.mean().to_numpy()
    lows = grouped.relative_gain.quantile(0.025).to_numpy()
    highs = grouped.relative_gain.quantile(0.975).to_numpy()
    axes[1].plot(centers, means, color=ORANGE, linewidth=1.8, marker="o", markersize=3.8,
                 markerfacecolor=ORANGE, label="Binned mean")
    axes[1].fill_between(centers, lows, highs, color=ORANGE, alpha=0.13, linewidth=0)
    axes[1].axhline(0, color="#555555", linewidth=0.7)
    axes[1].set_xlabel("True nonlinear interaction strength")
    axes[1].set_ylabel("Relative interaction-RMSE reduction")
    axes[1].set_title("The gain grows with interaction strength", fontsize=8.5, pad=6)
    axes[1].set_ylim(bottom=min(-0.08, float(nonlinear.relative_gain.min()) - 0.03))
    legend(axes[1], loc="upper left")
    caption(axes[0], "(a) Mechanism-level paired gain")
    caption(axes[1], "(b) Nonlinear scenes")
    figure.text(0.535, 0.945, "Frozen H1 · 256 paired scenes per mechanism · five model seeds averaged",
                ha="center", fontsize=8.0)
    save(figure, "fig_h1_joint_advantage")
    scene.to_csv(OUTPUT / "h1_joint_advantage_scenes.csv", index=False)


def load_h2_advantage():
    columns = ["split", "episode_id", "entity_id", "window", "method", "active", "main_evaluation",
               "predictive_trajectory_nll"]
    frames = []
    for path in sorted(H2_ROOT.glob("seed*/raw_metrics.csv")):
        data = pd.read_csv(path, usecols=columns)
        data = data[data.active.eq(1) & data.main_evaluation.eq(1)].copy()
        data["model_seed"] = int(path.parent.name[4:])
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def h2_gain(data, split, method):
    subset = data[data.split.eq(split) & data.method.isin(["persistent", method])]
    index = ["model_seed", "episode_id", "entity_id", "window"]
    pivot = subset.pivot(index=index, columns="method", values="predictive_trajectory_nll").dropna()
    pivot["gain"] = pivot[method] - pivot["persistent"]
    episode_window = pivot.reset_index().groupby(["model_seed", "episode_id", "window"], as_index=False).gain.mean()
    return episode_window.groupby(["episode_id", "window"], as_index=False).gain.mean()


def plot_h2():
    data = load_h2_advantage()
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.8))
    figure.subplots_adjust(left=0.085, right=0.985, top=0.865, bottom=0.235, wspace=0.34)
    for axis in axes:
        frame(axis)
    colors = {"shuffled-id": ORANGE, "reset": GREEN, "recurrent": BLUE}
    labels = {"shuffled-id": "Shuffled-ID", "reset": "Reset", "recurrent": "Recurrent"}
    for column, split in enumerate(("id", "ood")):
        axis = axes[column]
        for method in ("shuffled-id", "reset", "recurrent"):
            values = h2_gain(data, split, method)
            summary = values.groupby("window").gain.apply(lambda series: bootstrap_interval(series.to_numpy(), 20260908 + column)[0:3])
            windows = np.array(summary.index, dtype=float)
            means = np.array([item[0] for item in summary])
            low = np.array([item[1] for item in summary])
            high = np.array([item[2] for item in summary])
            axis.plot(windows, means, color=colors[method], linewidth=1.7 if method == "shuffled-id" else 1.1,
                      linestyle="-" if method == "shuffled-id" else "--", marker="o" if method == "shuffled-id" else None,
                      markersize=3.4, label=labels[method])
            axis.fill_between(windows, low, high, color=colors[method], alpha=0.08 if method != "shuffled-id" else 0.13,
                              linewidth=0)
        axis.axhline(0, color="#555555", linewidth=0.7)
        axis.set_xlim(0.7, 7.3)
        axis.set_xticks(range(1, 8))
        axis.set_xlabel("Planning window")
        axis.set_ylabel("Persistent predictive NLL advantage\n(baseline NLL − Persistent NLL)")
        axis.set_title(f"{split.upper()}: identity evidence is reusable", fontsize=8.5, pad=6)
        if column == 0:
            legend(axis, loc="upper left")
        caption(axis, f"({chr(97 + column)}) {split.upper()} evidence accumulation")
    figure.text(0.535, 0.945, "Positive values favor Persistent · matched decision windows · five model seeds",
                ha="center", fontsize=8.0)
    save(figure, "fig_h2_identity_evidence")


def plot_h3():
    data = pd.read_csv(H3_PATH)
    policies = ["task-only", "reset-joint", "persistent-joint", "oracle-joint"]
    response_keys = ["reset-joint", "persistent-joint", "oracle-joint"]
    labels = {"task-only": "Task-only", "reset-joint": "Reset-Joint", "persistent-joint": "Persistent-Joint", "oracle-joint": "Oracle-Joint"}
    colors = {"task-only": GRAY, "reset-joint": GREEN, "persistent-joint": ORANGE, "oracle-joint": BLUE}
    split_colors = {"id": GREEN, "ood": ORANGE}
    summaries = data[data.policy.isin(policies)].groupby(["split", "policy"], as_index=False).agg(
        task_progress=("task_progress", "mean"), human_cost=("human_cost", "mean"),
        safety_cost=("safety_cost", "mean"), collision_rate=("collision_rate", "mean"),
        realized_total_cost=("realized_total_cost", "mean"), final_mode_accuracy=("final_mode_accuracy", "mean"),
        oracle_regret=("oracle_regret", "mean"))
    summary_index = summaries.set_index(["split", "policy"])
    derived_rows = []
    paired_rows = []
    for split in ("id", "ood"):
        baseline = summary_index.loc[(split, "task-only")]
        for policy in response_keys:
            current = summary_index.loc[(split, policy)]
            derived_rows.append({
                "split": split,
                "policy": policy,
                "progress_change_pct": 100 * (current.task_progress - baseline.task_progress) / baseline.task_progress,
                "safety_reduction_pct": 100 * (baseline.safety_cost - current.safety_cost) / baseline.safety_cost,
                "human_reduction_pct": 100 * (baseline.human_cost - current.human_cost) / baseline.human_cost,
                "collision_reduction_pct": 100 * (baseline.collision_rate - current.collision_rate) / baseline.collision_rate,
                "total_cost_gain": baseline.realized_total_cost - current.realized_total_cost,
            })
        paired = data[data.split.eq(split) & data.policy.isin(["task-only", "persistent-joint"])]
        pivot = paired.pivot(index=["seed", "episode"], columns="policy", values="realized_total_cost").dropna()
        paired_rows.extend({"split": split, "gain": value} for value in (pivot["task-only"] - pivot["persistent-joint"]).to_numpy())
    derived = pd.DataFrame(derived_rows)
    paired = pd.DataFrame(paired_rows)

    figure, axes = plt.subplots(2, 2, figsize=(7.16, 6.55))
    figure.subplots_adjust(left=0.10, right=0.975, top=0.91, bottom=0.145, wspace=0.32, hspace=0.48)
    for axis in axes.flat:
        frame(axis)

    axis = axes[0, 0]
    positions = {"id": np.array([1.0, 2.0, 3.0]) - 0.11, "ood": np.array([1.0, 2.0, 3.0]) + 0.11}
    for split in ("id", "ood"):
        for position, policy in zip(positions[split], response_keys):
            values = data[(data.split.eq(split)) & (data.policy.eq("task-only"))].merge(
                data[(data.split.eq(split)) & (data.policy.eq(policy))], on=["seed", "episode"], suffixes=("_task", "_method"))
            gains = (values.realized_total_cost_task - values.realized_total_cost_method).to_numpy()
            mean, low, high = evidence.bootstrap_weights(len(gains), 2000, 20260908 + int(position * 10)).shape[0], None, None
            bootstrap = np.random.default_rng(20260908 + int(position * 10)).choice(gains, (2000, len(gains)), replace=True).mean(axis=1)
            mean = float(gains.mean())
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            axis.errorbar(mean, position, xerr=[[mean - low], [high - mean]], fmt="none",
                          ecolor=split_colors[split], elinewidth=1.4, capsize=2.5, zorder=3)
            axis.scatter(mean, position, s=40, marker="o" if split == "id" else "s",
                         color=colors[policy], edgecolor="white", linewidth=0.7, zorder=4)
    axis.axvline(0, color="#555555", linewidth=0.7)
    axis.set_yticks([1, 2, 3], [labels[key] for key in response_keys])
    axis.set_xlabel("Task-only cost − response-aware cost")
    axis.set_ylabel("Policy")
    axis.set_title("Paired closed-loop cost gain", fontsize=8.5, pad=6)
    axis.text(0.04, 0.95, "positive = lower realized cost\ncircle = ID · square = OOD",
              transform=axis.transAxes, fontsize=6.9, color="#555555", va="top")
    caption(axis, "(a) Episode-paired total-cost gain")

    axis = axes[0, 1]
    metrics = ["progress_change_pct", "safety_reduction_pct", "human_reduction_pct", "collision_reduction_pct"]
    metric_labels = ["Task\nprogress", "Safety\ncost", "Human\ncost", "Collision\nrate"]
    x = np.arange(len(metrics))
    width = 0.12
    for offset, split in enumerate(("id", "ood")):
        for policy_id, policy in enumerate(response_keys):
            row = derived[(derived.split.eq(split)) & (derived.policy.eq(policy))].iloc[0]
            values = [row[key] for key in metrics]
            axis.bar(x + (offset * 3 + policy_id - 1) * width, values, width=width,
                     color=colors[policy], alpha=0.9 if split == "id" else 0.55,
                     edgecolor="white", linewidth=0.25, label=f"{labels[policy]} · {split.upper()}" if offset == 0 else None)
    axis.axhline(0, color="#555555", linewidth=0.7)
    axis.set_xticks(x, metric_labels)
    axis.set_ylabel("Relative change vs. Task-only (%)")
    axis.set_title("Where the closed-loop gain comes from", fontsize=8.5, pad=6)
    axis.legend(fontsize=6.4, frameon=True, fancybox=False, framealpha=0.95, ncol=1,
                loc="upper left", borderpad=0.3, labelspacing=0.2)
    caption(axis, "(b) Component-level response-aware changes")

    method_handles = [Line2D([], [], marker="o", linestyle="none", markersize=5.0,
                              markerfacecolor=colors[policy], markeredgecolor="white",
                              markeredgewidth=0.6, label=labels[policy]) for policy in response_keys]
    split_handles = [Line2D([], [], marker=marker, linestyle="none", markersize=5.0,
                             markerfacecolor="#555555", markeredgecolor="white",
                             markeredgewidth=0.6, label=split.upper())
                     for split, marker in (("id", "o"), ("ood", "s"))]

    axis = axes[1, 0]
    for split, marker in (("id", "o"), ("ood", "s")):
        subset = summaries[summaries.split.eq(split)]
        for _, row in subset[subset.policy.isin(response_keys)].iterrows():
            axis.scatter(row.final_mode_accuracy, row.oracle_regret, marker=marker,
                         s=48, color=colors[row.policy], edgecolor="white", linewidth=0.7,
                         zorder=4)
    axis.set_xlabel("Final response-mode accuracy ↑")
    axis.set_ylabel("Oracle regret ↓")
    axis.set_title("Belief quality entering the loop", fontsize=8.5, pad=6)
    axis.text(0.04, 0.95, "circle = ID · square = OOD", transform=axis.transAxes,
              fontsize=6.9, color="#555555", va="top")
    method_legend = axis.legend(handles=method_handles, loc="upper left", fontsize=6.6,
                                frameon=True, fancybox=False, framealpha=0.95,
                                borderpad=0.3, labelspacing=0.2)
    axis.add_artist(method_legend)
    axis.legend(handles=split_handles, loc="lower left", fontsize=6.6,
                frameon=True, fancybox=False, framealpha=0.95,
                borderpad=0.3, labelspacing=0.2)
    caption(axis, "(c) Belief quality entering the loop")

    axis = axes[1, 1]
    for split, marker in (("id", "o"), ("ood", "s")):
        subset = summaries[summaries.split.eq(split)]
        for _, row in subset[subset.policy.isin(policies)].iterrows():
            axis.scatter(row.human_cost, row.task_progress, marker=marker,
                         s=60 if row.policy != "task-only" else 82, color=colors[row.policy],
                         edgecolor="white", linewidth=0.7, zorder=4)
    axis.set_xlabel("Human-response cost ↓")
    axis.set_ylabel("Task progress ↑")
    axis.set_title("Safety/social operating point", fontsize=8.5, pad=6)
    axis.text(0.04, 0.95, "circle = ID · square = OOD", transform=axis.transAxes,
              fontsize=6.9, color="#555555", va="top")
    axis.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=5.0,
                                markerfacecolor=colors[policy], markeredgecolor="white",
                                markeredgewidth=0.6, label=labels[policy]) for policy in policies],
                loc="lower left", fontsize=6.5, ncol=2, frameon=True, fancybox=False,
                framealpha=0.95, borderpad=0.3, labelspacing=0.2, columnspacing=0.6)
    caption(axis, "(d) Task progress versus social cost")
    figure.text(0.535, 0.958, "H3 independent closed-loop replication · paired episodes · three robots",
                ha="center", fontsize=8.0)
    save(figure, "fig_h3_response_aware_tradeoff")
    summaries.to_csv(OUTPUT / "h3_response_aware_summary.csv", index=False)
    derived.to_csv(OUTPUT / "h3_response_aware_changes.csv", index=False)
    paired.to_csv(OUTPUT / "h3_paired_cost_gain.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=("all", "h1", "h2", "h3"), default="all")
    args = parser.parse_args()
    figure_style()
    if args.figure in ("all", "h1"):
        plot_h1()
    if args.figure in ("all", "h2"):
        plot_h2()
    if args.figure in ("all", "h3"):
        plot_h3()


if __name__ == "__main__":
    main()
