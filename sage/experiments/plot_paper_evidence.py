from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import yaml

from sage.experiments import plot_reference_trajectories as reference


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/paper_quantitative_20260908"
H3_ROOT = Path("experiments/tracer_final_closed_loop_confirmation_20260908_v7")
H1_ROOT = ROOT / "experiments/formal_v3_enhanced_h1/evaluation_corrected"
H2_ROOT = ROOT / "experiments/formal_v2/clean_chain_submission_20260904/h2"
ORANGE = reference.ORANGE
GREEN = "#008000"
BLUE = "#0000FF"
PURPLE = "#B000B0"
GRAY = "#858585"
METHODS = {
    "tracer-persistent": ("TRACER", ORANGE, "-", "o"),
    "tracer-no-pair": ("No pair", PURPLE, "--", "s"),
    "tracer-reset": ("Reset belief", GREEN, "-.", "^"),
    "tracer-uniform": ("Uniform belief", GRAY, ":", "D"),
    "official-goalone": ("GoAlone", BLUE, "-", "v"),
    "official-halt": ("Halt", "#404040", (0, (1, 3)), "x"),
    "persistent": ("Persistent", ORANGE, "-", "o"),
    "reset": ("Reset", GREEN, "--", "s"),
    "recurrent": ("Recurrent", BLUE, "-.", "^"),
}
H3_POLICIES = tuple(list(METHODS)[:6])
H2_METHODS = ("persistent", "reset", "recurrent")
SCENE_KEYS = ["seed", "scenario", "episode"]
H1_KEYS = ["effect_mode", "data_seed", "split", "base_scene_id", "model_seed"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def bootstrap_weights(count, resamples=2000, seed=20260908):
    if count < 1 or resamples < 1:
        raise ValueError("bootstrap requires positive cluster and resample counts")
    generator = np.random.default_rng(seed)
    return generator.multinomial(count, np.full(count, 1 / count), size=resamples).astype(float)


def mean_interval(values, weights):
    values = np.asarray(values, dtype=float)
    if values.shape[0] != weights.shape[1] or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite and match the cluster count")
    samples = (weights @ values) / weights.sum(axis=1).reshape((-1,) + (1,) * (values.ndim - 1))
    return values.mean(axis=0), np.quantile(samples, 0.025, axis=0), np.quantile(samples, 0.975, axis=0)


def style():
    reference.style()
    plt.rcParams.update({"axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.labelpad": 5, "lines.solid_capstyle": "round"})


def panel_grid(rows=1):
    style()
    figure, axes = plt.subplots(rows, 2, figsize=(7.16, 4.05 if rows == 1 else 7.15), squeeze=False)
    figure.subplots_adjust(left=0.079, right=0.99, top=0.921 if rows == 1 else 0.951,
                           bottom=0.205 if rows == 1 else 0.11, wspace=0.30, hspace=0.40)
    for axis in axes.flat:
        axis.set_box_aspect(1)
        axis.tick_params(length=2.3, width=0.5, pad=3)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.45, linestyle=(0, (1.5, 2.5)))
    return figure, axes


def panel_caption(axis, text):
    axis.text(0.5, -0.192, text, transform=axis.transAxes, fontsize=12.8,
              ha="center", va="top", color="black", fontfamily="STIXGeneral")


def small_legend(axis, handles=None, **kwargs):
    legend = axis.legend(handles=handles, fontsize=7.2, frameon=True, fancybox=False,
                         framealpha=0.96, borderpad=0.35, labelspacing=0.30, handlelength=2,
                         handletextpad=0.55, borderaxespad=0.5, **kwargs)
    legend.get_frame().set_edgecolor("#BBBBBB")
    legend.get_frame().set_linewidth(0.4)
    return legend


def method_handle(method, label=None):
    name, color, line, marker = METHODS[method]
    return Line2D([], [], color=color, linestyle=line, marker=marker, markersize=3.8,
                  markerfacecolor=color if method in ("persistent", "tracer-persistent") else "white",
                  markeredgewidth=0.8, linewidth=1.4, label=label or name)


def save_figure(figure, output, name):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = []
    for axis in figure.axes:
        if axis.get_xlabel():
            boxes.append(axis.xaxis.label.get_window_extent(renderer))
        for text in axis.texts:
            if text.get_text().startswith("("):
                caption = text.get_window_extent(renderer)
                if any(caption.overlaps(box) for box in boxes):
                    raise ValueError(f"panel caption overlaps axis label in {name}")
    reference.save(figure, output, name)


def cumulative_safe_completion(elapsed, safe, times):
    elapsed = np.asarray(elapsed, dtype=float)
    safe = np.asarray(safe, dtype=float)
    times = np.asarray(times, dtype=float)
    if elapsed.shape != safe.shape or not np.isfinite(elapsed).all() or np.any(elapsed < 0):
        raise ValueError("invalid episode times")
    if not np.isin(safe, (0, 1)).all() or np.any(np.diff(times) < 0):
        raise ValueError("invalid success flags or timeline")
    return ((elapsed[:, None] <= times[None] + 1e-9) & safe[:, None].astype(bool)).astype(float)


def validate_h3(data, policies=H3_POLICIES):
    if data.duplicated(SCENE_KEYS + ["policy"]).any():
        raise ValueError("duplicate H3 policy-scene rows")
    if set(data.policy.unique()) != set(policies):
        raise ValueError("H3 policy set differs from the frozen comparison")
    grouped = data.groupby(SCENE_KEYS)
    if not grouped.policy.nunique().eq(len(policies)).all():
        raise ValueError("incomplete H3 paired scene")
    for name in ("episode_seed", "initial_state_hash", "initial_collision"):
        if not grouped[name].nunique().eq(1).all():
            raise ValueError(f"H3 reset mismatch: {name}")
    safe = data.team_success.eq(1) & data.collision.eq(0)
    if not np.array_equal(safe.astype(int), data.collision_free_team_success):
        raise ValueError("collision-free team success is inconsistent with terminal metrics")


def plot_h3(output, resamples):
    protocol_path = H3_ROOT / "protocol.json"
    status = json.loads((H3_ROOT / "status.json").read_text())
    protocol = json.loads(protocol_path.read_text())
    if status["status"] != "COMPLETED" or status["completed_policy_jobs"] != status["expected_policy_jobs"]:
        raise ValueError("H3 confirmation is incomplete")
    path = H3_ROOT / "aggregate/episode_metrics.csv"
    data = pd.read_csv(path)
    validate_h3(data)
    sample_metadata_path = H3_ROOT / "intersection_seed3109/tracer-persistent/metadata.json"
    sample_metadata = json.loads(sample_metadata_path.read_text())
    sim_dt = float(sample_metadata["parameters"]["sim_dt"])
    horizon = protocol["maximum_planning_windows"] * float(sample_metadata["parameters"]["env_dt"])
    if len(data) != protocol["expected_policy_rows"] or data.num_agents.nunique() != 1:
        raise ValueError("H3 size does not match the frozen protocol")
    data["elapsed_time_s"] = data.executed_sim_ticks * sim_dt
    if not np.isfinite(data.elapsed_time_s).all() or data.elapsed_time_s.max() > horizon + 1e-8:
        raise ValueError("H3 episode exceeds the frozen time limit")
    figure, axes = panel_grid()
    all_curves = []
    endpoint_rows = []
    for column, scenario in enumerate(("envs/scenario/open", "envs/scenario/intersection")):
        axis = axes[0, column]
        subset = data[data.scenario.eq(scenario)]
        scene_order = subset[SCENE_KEYS].drop_duplicates().sort_values(SCENE_KEYS)
        index = pd.MultiIndex.from_frame(scene_order)
        times = np.unique(np.r_[0, subset.elapsed_time_s, horizon])
        weights = bootstrap_weights(len(index), resamples, 20260908 + column)
        handles = []
        for policy in reversed(H3_POLICIES):
            selected = subset[subset.policy.eq(policy)].set_index(SCENE_KEYS).reindex(index)
            values = cumulative_safe_completion(selected.elapsed_time_s, selected.collision_free_team_success, times)
            mean, low, high = mean_interval(values, weights)
            name, color, line, marker = METHODS[policy]
            is_main = policy == "tracer-persistent"
            if policy in ("tracer-persistent", "official-goalone"):
                axis.fill_between(times, 100 * low, 100 * high, step="post", color=color,
                                  alpha=0.09, linewidth=0, zorder=1)
            axis.step(times, 100 * mean, where="post", color=color, linestyle=line,
                      linewidth=1.8 if is_main else 1.05, zorder=8 if is_main else 3)
            sample_times = np.arange(4 + H3_POLICIES.index(policy) * 0.45, horizon, 4)
            sampled = np.searchsorted(times, sample_times, side="right") - 1
            axis.plot(sample_times, mean[sampled] * 100, linestyle="none", marker=marker,
                      color=color, markersize=3.4, markerfacecolor=color if is_main else "white",
                      markeredgewidth=0.65, zorder=9 if is_main else 4)
            handles.append(method_handle(policy, f"{name}  {100 * mean[-1]:.1f}%"))
            endpoint_rows.append({"scenario": scenario, "policy": policy, "scenes": len(index),
                                  "safe_completions": int(selected.collision_free_team_success.sum()),
                                  "rate": float(mean[-1]), "ci95_low": float(low[-1]), "ci95_high": float(high[-1])})
            all_curves.append(pd.DataFrame({"scenario": scenario, "policy": policy, "time_s": times,
                                           "mean": mean, "ci95_low": low, "ci95_high": high}))
        axis.set_xlim(0, horizon)
        axis.set_ylim(-1, 63)
        axis.set_xticks(np.arange(0, horizon + 1, 6))
        axis.set_yticks([0, 15, 30, 45, 60])
        axis.set_xlabel("Elapsed simulation time (s)")
        axis.set_ylabel("Collision-free team completion (%)")
        small_legend(axis, list(reversed(handles)), loc="upper left", ncol=2, columnspacing=0.9)
        panel_caption(axis, f"({chr(97 + column)}) {'Open' if column == 0 else 'Intersection'}")
    figure.text(0.535, 0.972, "3 robots · 2 humans · 160 paired scenes per environment", ha="center", fontsize=8.1)
    save_figure(figure, output, "fig_h3_completion_time")
    pd.concat(all_curves).to_csv(output / "h3_completion_curves.csv", index=False)
    pd.DataFrame(endpoint_rows).to_csv(output / "h3_completion_endpoints.csv", index=False)
    write_json(output / "h3_provenance.json", {
        "sources": [fingerprint(path), fingerprint(protocol_path), fingerprint(sample_metadata_path)],
        "records": len(data), "paired_scenes": len(data[SCENE_KEYS].drop_duplicates()),
        "physical_dt_s": sim_dt, "time_limit_s": horizon, "bootstrap_resamples": resamples,
        "estimand": "P(episode is collision-free team success and its recorded team terminal time <= t). Failures, timeouts and initial collisions remain in the denominator.",
        "bootstrap": "Unique seed/scenario/episode scene clusters, shared resampling across six policies within each environment. Pointwise intervals, not simultaneous confidence bands; zero-success bootstrap degeneracy is not a zero-risk guarantee.",
        "style": "Exact reference trajectory font, orange/blue/green/purple palette, white background and thin gray frames. Step curves at recorded events; marker timestamps sample a step's existing value without shifting it.",
        "boundary": "Final three-robot/two-human SocialGym2 confirmation, not the eight-human qualitative run and not the single-robot SICNav benchmark. Terminal time is recorded episode termination, not reconstructed first goal-crossing time.",
        "endpoints": endpoint_rows,
    })
    return {"figure": "fig_h3_completion_time", "endpoints": endpoint_rows}


def paired_h1(data):
    methods = ("matched-additive", "joint")
    selected = data[data.method.isin(methods)]
    if selected.duplicated(H1_KEYS + ["method"]).any():
        raise ValueError("duplicate H1 paired observation")
    paired = selected.pivot(index=H1_KEYS, columns="method", values="interaction_rmse")
    if set(paired.columns) != set(methods) or paired.isna().any().any():
        raise ValueError("incomplete H1 model pairing")
    targets = selected.groupby(H1_KEYS).interaction_target_rms
    if (targets.max() - targets.min()).max() > 1e-6:
        raise ValueError("H1 truth differs between models")
    paired["gain"] = paired["matched-additive"] - paired["joint"]
    paired["strength"] = targets.mean()
    paired = paired.reset_index()
    if not np.isfinite(paired[["gain", "strength"]]).all().all():
        raise ValueError("nonfinite H1 values")
    groups = [name for name in H1_KEYS if name != "model_seed"]
    expected_seeds = selected.model_seed.nunique()
    if not paired.groupby(groups).model_seed.nunique().eq(expected_seeds).all():
        raise ValueError("missing H1 model-seed block")
    if paired.groupby(groups).strength.agg(lambda values: values.max() - values.min()).max() > 1e-6:
        raise ValueError("H1 physical scene differs across model seeds")
    result = paired.groupby(groups, as_index=False)[["gain", "strength"]].mean()
    return result, expected_seeds


def strength_groups(scenes, groups=8):
    result = scenes.copy()
    edges = np.unique(np.quantile(result.strength, np.linspace(0, 1, groups + 1)))
    if len(edges) < 3:
        raise ValueError("insufficient distinct interaction strengths for stratification")
    result["strength_bin"] = np.minimum(np.searchsorted(edges[1:], result.strength, side="right"), len(edges) - 2)
    return result, edges


def plot_h1(output, resamples):
    path = H1_ROOT / "paired_records.csv"
    scenes, seeds = paired_h1(pd.read_csv(path))
    nonlinear, edges = strength_groups(scenes[scenes.effect_mode.eq("nonlinear")])
    figure, axes = panel_grid()
    axis = axes[0, 0]
    axis.scatter(nonlinear.strength, nonlinear.gain, s=6, color="#A0A0A0", alpha=0.29, linewidth=0, zorder=2)
    axis.axhline(0, color="#777777", linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    bins = []
    for bin_id, subset in nonlinear.groupby("strength_bin"):
        mean, low, high = mean_interval(subset.gain.to_numpy(), bootstrap_weights(len(subset), resamples, 20261908 + int(bin_id)))
        bins.append({"bin": int(bin_id), "scenes": len(subset), "strength": float(subset.strength.mean()),
                     "lower_edge": float(edges[bin_id]), "upper_edge": float(edges[bin_id + 1]),
                     "gain": float(mean), "ci95_low": float(low), "ci95_high": float(high)})
    grouped = pd.DataFrame(bins)
    axis.fill_between(grouped.strength, grouped.ci95_low, grouped.ci95_high, color=GREEN, alpha=0.10, linewidth=0)
    axis.errorbar(grouped.strength, grouped.gain, yerr=[grouped.gain - grouped.ci95_low, grouped.ci95_high - grouped.gain],
                  color=GREEN, linewidth=1.55, marker="o", markersize=4.5, capsize=2.3,
                  markeredgewidth=0.8, elinewidth=0.85, zorder=5)
    axis.set_xlabel("True interaction strength (m/s)")
    axis.set_ylabel("Interaction RMSE reduction (m/s)")
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(5))
    handles = [Line2D([], [], marker="o", color="#999999", markersize=3.2, linestyle="none", label="Individual scene"),
               Line2D([], [], marker="o", color=GREEN, linewidth=1.5, markersize=4, label="Strength-bin mean")]
    small_legend(axis, handles, loc="upper left")
    panel_caption(axis, "(a) Nonlinear interaction strength")
    effects = []
    axis = axes[0, 1]
    axis.grid(False)
    axis.axvline(0, color="#777777", linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    for row_id, mode in enumerate(("none", "additive", "nonlinear")):
        subset = scenes[scenes.effect_mode.eq(mode)]
        mean, low, high = mean_interval(subset.gain.to_numpy(), bootstrap_weights(len(subset), resamples, 20262908 + row_id))
        effects.append({"mechanism": mode, "scenes": len(subset), "gain": float(mean),
                        "ci95_low": float(low), "ci95_high": float(high)})
        color = GREEN if mean >= 0 else PURPLE
        axis.hlines(row_id, -0.025, 0.17, color="#EEEEEE", linewidth=0.5, zorder=0)
        axis.errorbar(mean, row_id, xerr=[[mean - low], [high - mean]], color=color, fmt="o", markersize=5.1,
                      capsize=3, elinewidth=1.2, markeredgewidth=0.9, zorder=3)
        axis.annotate(f"{mean:+.3f}", (mean, row_id), xytext=(0, -17), textcoords="offset points",
                      ha="center", va="center", color=color, fontsize=8.5)
    axis.set_yticks([0, 1, 2], ["No interaction", "Additive", "Nonlinear"])
    axis.set_ylim(2.55, -0.65)
    axis.set_xlim(-0.028, max(0.145, max(row["ci95_high"] for row in effects) + 0.02))
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.set_xlabel("Matched Additive − Joint RMSE (m/s)")
    axis.text(0.05, 0.965, "Positive: Joint better", transform=axis.transAxes, fontsize=7.8, color=GREEN, va="top")
    panel_caption(axis, "(b) Paired mechanism effects")
    figure.text(0.535, 0.972, f"256 paired scenes per mechanism · {seeds} model seeds averaged", ha="center", fontsize=8.1)
    save_figure(figure, output, "fig_h1_interaction_gain")
    scenes.to_csv(output / "h1_scene_gains.csv", index=False)
    grouped.to_csv(output / "h1_strength_bins.csv", index=False)
    pd.DataFrame(effects).to_csv(output / "h1_mechanism_effects.csv", index=False)
    write_json(output / "h1_provenance.json", {
        "sources": [fingerprint(path), fingerprint(H1_ROOT / "config_frozen.yaml")],
        "independent_unit": "base_scene; model seeds averaged within a paired scene, not counted as independent scenes",
        "model_seeds": seeds, "bootstrap_resamples": resamples, "strength_bins": bins,
        "estimand": "Scene-mean matched-additive interaction RMSE minus Joint interaction RMSE in velocity units m/s. This is not an ADE or relative-percent gain.",
        "grouping": "Eight empirical quantile groups of truth strength, using all 256 nonlinear scenes; bin membership is fixed for conditional within-bin bootstrap. No smoothing or outcome-dependent exclusion.",
        "mechanism_effects": effects,
        "boundary": "Formal synthetic factorial response evidence. Negative effects in the no-interaction/additive mechanisms remain visible; not universal model superiority or real-human causal evidence.",
    })
    return {"figure": "fig_h1_interaction_gain", "mechanism_effects": effects}


def load_h2(directory):
    columns = ["split", "episode_id", "entity_id", "window", "method", "active", "main_evaluation",
               "predictive_trajectory_nll", "decision_confidence", "decision_mode_correct"]
    frames = []
    paths = sorted(Path(directory).glob("seed*/raw_metrics.csv"))
    if not paths:
        raise ValueError("no frozen H2 raw metrics")
    for path in paths:
        raw = pd.read_csv(path, usecols=columns)
        selected = raw[raw.method.isin(H2_METHODS) & raw.active.eq(1) & raw.main_evaluation.eq(1)].copy()
        selected["model_seed"] = int(path.parent.name[4:])
        frames.append(selected)
    data = pd.concat(frames, ignore_index=True)
    keys = ["model_seed", "split", "episode_id", "entity_id", "window"]
    if data.duplicated(keys + ["method"]).any() or not data.groupby(keys).method.nunique().eq(len(H2_METHODS)).all():
        raise ValueError("unmatched or duplicate H2 decision observations")
    if not np.isfinite(data[["predictive_trajectory_nll", "decision_confidence", "decision_mode_correct"]]).all().all():
        raise ValueError("nonfinite H2 observations")
    if data.window.min() <= 0:
        raise ValueError("window-zero observation cannot enter the scored decision set")
    if not np.isin(data.decision_mode_correct, (0, 1)).all():
        raise ValueError("decision accuracy must be binary")
    return data, paths


def decision_gain_matrix(data, method):
    keys = ["episode_id", "model_seed", "entity_id", "window"]
    matched = data[data.method.isin(["reset", method])].pivot(index=keys, columns="method", values="predictive_trajectory_nll")
    if matched.isna().any().any():
        raise ValueError("missing H2 predictive baseline observation")
    matched["gain"] = matched["reset"] - matched[method]
    per_seed = matched.reset_index().groupby(["episode_id", "model_seed", "window"]).gain.mean()
    per_episode = per_seed.groupby(["episode_id", "window"]).mean().unstack("window").sort_index()
    if per_episode.isna().any().any():
        raise ValueError("incomplete H2 episode lifecycle")
    return per_episode


def reliability_statistics(data, weights, bins=10, minimum_count=100, minimum_episodes=10):
    confidence = data.decision_confidence.to_numpy(float)
    if np.any(confidence < 1 / 3 - 1e-6) or np.any(confidence > 1 + 1e-6):
        raise ValueError("invalid three-mode decision confidence")
    confidence = np.clip(confidence, 1 / 3, 1)
    edges = np.linspace(1 / 3, 1, bins + 1)
    bin_ids = np.minimum(np.searchsorted(edges[1:], confidence, side="right"), bins - 1)
    episodes, episode_ids = np.unique(data.episode_id.to_numpy(), return_inverse=True)
    if len(episodes) != weights.shape[1]:
        raise ValueError("reliability bootstrap does not match the episode clusters")
    counts = np.zeros((len(episodes), bins))
    correct = np.zeros_like(counts)
    confidences = np.zeros_like(counts)
    np.add.at(counts, (episode_ids, bin_ids), 1)
    np.add.at(correct, (episode_ids, bin_ids), data.decision_mode_correct.to_numpy())
    np.add.at(confidences, (episode_ids, bin_ids), confidence)
    sampled_counts = weights @ counts
    sampled_correct = weights @ correct
    rows = []
    for bin_id in range(bins):
        count = counts[:, bin_id].sum()
        active_episodes = int(np.count_nonzero(counts[:, bin_id]))
        visible = count >= minimum_count and active_episodes >= minimum_episodes
        accuracy = float(correct[:, bin_id].sum() / count) if count else None
        mean_confidence = float(confidences[:, bin_id].sum() / count) if count else None
        valid = sampled_counts[:, bin_id] > 0
        if visible:
            samples = sampled_correct[valid, bin_id] / sampled_counts[valid, bin_id]
            low, high = np.quantile(samples, [0.025, 0.975])
        else:
            low, high = None, None
        rows.append({"bin": bin_id, "lower_edge": float(edges[bin_id]), "upper_edge": float(edges[bin_id + 1]),
                     "count": int(count), "episodes": active_episodes, "confidence": mean_confidence,
                     "accuracy": accuracy, "ci95_low": None if low is None else float(low),
                     "ci95_high": None if high is None else float(high), "plotted": visible})
    ece = sum(row["count"] * abs(row["accuracy"] - row["confidence"]) for row in rows if row["count"]) / len(data)
    accuracy = data.decision_mode_correct.mean()
    return rows, float(ece), float(accuracy)


def plot_h2(output, resamples):
    data, paths = load_h2(H2_ROOT)
    figure, axes = panel_grid(2)
    gain_rows = []
    reliability_rows = []
    score_rows = []
    for row_id, split in enumerate(("id", "ood")):
        subset = data[data.split.eq(split)]
        weights = bootstrap_weights(subset.episode_id.nunique(), resamples, 20263908 + row_id)
        axis = axes[row_id, 0]
        for method in ("recurrent", "persistent"):
            matrix = decision_gain_matrix(subset, method)
            mean, low, high = mean_interval(matrix.to_numpy(), weights)
            name, color, line, marker = METHODS[method]
            axis.fill_between(matrix.columns, low, high, color=color, alpha=0.09, linewidth=0)
            axis.plot(matrix.columns, mean, color=color, linestyle=line, marker=marker, markersize=4,
                      markerfacecolor=color if method == "persistent" else "white", markeredgewidth=0.8,
                      linewidth=1.65 if method == "persistent" else 1.15, label=name)
            for window, value, lower, upper in zip(matrix.columns, mean, low, high):
                gain_rows.append({"split": split, "method": method, "window": int(window), "gain": float(value),
                                  "ci95_low": float(lower), "ci95_high": float(upper), "episodes": len(matrix)})
        axis.axhline(0, color=GREEN, linestyle="--", linewidth=0.95)
        axis.set_xlabel("Decision window")
        axis.set_ylabel("Predictive NLL reduction vs. Reset")
        axis.set_xticks(range(1, 8))
        axis.set_xlim(0.7, 7.3)
        axis.set_ylim(-0.003, 0.017)
        axis.yaxis.set_major_locator(MaxNLocator(5))
        small_legend(axis, [method_handle(method) for method in ("persistent", "recurrent", "reset")], loc="upper right")
        panel_caption(axis, f"({chr(97 + 2 * row_id)}) {split.upper()} prediction gain")
        axis = axes[row_id, 1]
        axis.plot([1 / 3, 1], [1 / 3, 1], color="#B0B0B0", linestyle=(0, (2, 3)), linewidth=0.9, zorder=1)
        handles = []
        for method in H2_METHODS:
            method_rows = subset[subset.method.eq(method)]
            rows, ece, accuracy = reliability_statistics(method_rows, weights)
            reliability_rows.extend({"split": split, "method": method, **row} for row in rows)
            visible = pd.DataFrame([row for row in rows if row["plotted"]])
            name, color, line, marker = METHODS[method]
            axis.plot(visible.confidence, visible.accuracy, color=color, linestyle=line, linewidth=1.15, zorder=3)
            axis.errorbar(visible.confidence, visible.accuracy,
                          yerr=[visible.accuracy - visible.ci95_low, visible.ci95_high - visible.accuracy],
                          linestyle="none", color=color, elinewidth=0.75, capsize=1.8, zorder=4)
            sizes = 7 + 29 * visible["count"] / len(method_rows)
            axis.scatter(visible.confidence, visible.accuracy, s=sizes, marker=marker,
                         facecolor=color if method == "persistent" else "white", edgecolor=color, linewidth=0.8, zorder=5)
            handles.append(method_handle(method, f"{name}  ({ece:.2f})"))
            score_rows.append({"split": split, "method": method, "decision_ece": ece, "decision_accuracy": accuracy,
                               "decision_observations": len(method_rows), "independent_episodes": method_rows.episode_id.nunique(),
                               "predictive_nll": float(method_rows.predictive_trajectory_nll.mean()),
                               "plotted_bin_observations": sum(row["count"] for row in rows if row["plotted"])})
        axis.set_xlim(0.30, 1.02)
        axis.set_ylim(0, 1.02)
        axis.set_xticks([1 / 3, 0.5, 0.75, 1], ["⅓", "0.5", "0.75", "1.0"])
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1.0"])
        axis.set_xlabel("Decision-time confidence")
        axis.set_ylabel("Empirical mode accuracy")
        small_legend(axis, handles, loc="lower right", title="Decision ECE ↓", title_fontsize=7.2)
        panel_caption(axis, f"({chr(98 + 2 * row_id)}) {split.upper()} calibration")
    figure.text(0.535, 0.983, "Frozen H2 · 256 episodes per split · 5 model seeds · Decision-time evidence only", ha="center", fontsize=8.1)
    save_figure(figure, output, "fig_h2_prediction_calibration")
    pd.DataFrame(gain_rows).to_csv(output / "h2_prediction_gain.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(output / "h2_reliability_bins.csv", index=False)
    pd.DataFrame(score_rows).to_csv(output / "h2_scores.csv", index=False)
    data.groupby(["split", "method", "window"], as_index=False).predictive_trajectory_nll.mean().to_csv(
        output / "h2_absolute_prediction_nll.csv", index=False)
    config_path = H2_ROOT / "config_frozen.yaml"
    config = yaml.safe_load(config_path.read_text())
    write_json(output / "h2_provenance.json", {
        "sources": [fingerprint(path) for path in paths] + [fingerprint(config_path)],
        "bootstrap_resamples": resamples, "model_seeds": int(data.model_seed.nunique()),
        "likelihood_scale": config["evaluation"]["likelihood_scale"],
        "prediction": "Reset minus method predictive trajectory NLL on matched active/main-evaluation decisions, windows 1..7; window zero excluded. Entities averaged within each episode/window/model seed, then model seeds averaged; paired episode-cluster bootstrap.",
        "calibration": "decision_confidence vs decision_mode_correct, not posterior confidence. Ten fixed equal-width confidence bins in [1/3,1]. Curves omit bins with fewer than 100 model-seed observations or fewer than 10 unique episodes; all bins retained in CSV and ECE. Larger point area means more observations, not more independent episodes.",
        "calibration_bootstrap": "Resample complete episodes, keeping all entity/time/model-seed rows together; recompute per-bin accuracy ratios. No independent-window or independent-model-seed bootstrap.",
        "boundary": "Frozen formal synthetic H2, not the later calibration audit or the SocialGym2 navigation benchmark. No tuning or scale selection; confidence errors and Recurrent advantages remain visible.",
        "scores": score_rows,
    })
    return {"figure": "fig_h2_prediction_calibration", "scores": score_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=("h3", "h1", "h2", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, plot in (("h3", plot_h3), ("h1", plot_h1), ("h2", plot_h2)):
        if args.figure in (name, "all"):
            result = plot(args.output_dir, args.bootstrap_resamples)
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
