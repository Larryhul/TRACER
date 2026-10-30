from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D

from tracer.data.synthetic_navigation import collate_synthetic_scene_batch
from tracer.response.causal_synthetic import make_causal_response_dataset_from_config
from tracer.response.data import build_entity_response_batch
from tracer.response.models import ResponseModelConfig, TeamResponsePredictor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT
    / "experiments"
    / "formal_v3_enhanced_h1"
    / "evaluation_corrected"
    / "config_frozen.yaml"
)
DEFAULT_CHECKPOINT_ROOT = (
    ROOT / "experiments" / "formal_v3_enhanced_h1" / "checkpoints" / "nonlinear"
)
DEFAULT_OUTPUT_STEM = (
    ROOT
    / "reports"
    / "paper_visualizations_20260906"
    / "fig_h1_joint_counterfactual_trajectories"
)

INK = "#20262E"
SLATE = "#66727D"
GRID = "#D9E0E5"
CONTEXT = "#C6CDD3"
ROBOT_COLORS = ("#0072B2", "#CC79A7")
JOINT = "#009E73"
MATCHED = "#E69F00"
WHITE = "#FFFFFF"

BRANCH_TITLES = (
    r"$U_{00}$  Both yield",
    r"$U_{10}$  R1 advances",
    r"$U_{01}$  R2 advances",
    r"$U_{11}$  Both advance",
)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8.2,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
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


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positions(current: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    return np.concatenate(
        [current[None], current[None] + np.cumsum(velocity * dt, axis=0)], axis=0
    )


def _factorial_second_difference(values: np.ndarray) -> np.ndarray:
    if values.shape[0] != 4:
        raise ValueError(f"expected four factorial branches, got {values.shape[0]}")
    return values[3] - values[1] - values[2] + values[0]


def _select_representative_case(
    dataset,
    quantile: float,
) -> tuple[int, int, dict[str, float]]:
    records: list[tuple[int, int, float]] = []
    for base_scene_index in range(dataset.num_base_scenes):
        first = base_scene_index * dataset.interventions_per_scene
        samples = [dataset[first + branch] for branch in range(4)]
        target_velocity = torch.stack(
            [sample["future"][dataset.num_robots :] for sample in samples], dim=0
        ).numpy()
        interaction = _factorial_second_difference(target_velocity)
        entity_rms = np.sqrt(np.mean(np.square(interaction), axis=(1, 2)))
        records.extend(
            (base_scene_index, entity_index, float(score))
            for entity_index, score in enumerate(entity_rms)
        )

    scores = np.asarray([record[2] for record in records], dtype=np.float64)
    target_score = float(np.quantile(scores, quantile))
    selected = min(
        records,
        key=lambda record: (abs(record[2] - target_score), record[0], record[1]),
    )
    percentile = float(np.mean(scores <= selected[2]))
    return selected[0], selected[1], {
        "selection_quantile": float(quantile),
        "target_interaction_rms": target_score,
        "selected_interaction_rms": selected[2],
        "selected_empirical_percentile": percentile,
        "candidate_count": int(scores.size),
    }


def _load_model(
    checkpoint_path: Path,
    variant: str,
    device: torch.device,
) -> TeamResponsePredictor:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TeamResponsePredictor(
        variant, ResponseModelConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _ensemble_predictions(
    checkpoint_root: Path,
    seeds: list[int],
    batch,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[Path]]:
    prediction_lists: dict[str, list[np.ndarray]] = {
        "joint": [],
        "matched-additive": [],
    }
    checkpoint_paths: list[Path] = []
    device_batch = batch.to(device)
    with torch.inference_mode():
        for seed in seeds:
            seed_dir = checkpoint_root / f"seed{seed}"
            for variant in prediction_lists:
                checkpoint_path = seed_dir / f"{variant}.pt"
                model = _load_model(checkpoint_path, variant, device)
                prediction_lists[variant].append(
                    model(device_batch)["mean"].detach().cpu().numpy()
                )
                checkpoint_paths.append(checkpoint_path)
                del model
    predictions = {
        variant: np.mean(np.stack(values, axis=0), axis=0)
        for variant, values in prediction_lists.items()
    }
    return predictions, checkpoint_paths


def _shared_limits(points: list[np.ndarray], padding: float = 0.45) -> tuple[float, ...]:
    combined = np.concatenate([np.asarray(point).reshape(-1, 2) for point in points], axis=0)
    low = combined.min(axis=0)
    high = combined.max(axis=0)
    center = (low + high) * 0.5
    span = max(float(np.max(high - low)), 1.0) + 2.0 * padding
    return (
        float(center[0] - span * 0.5),
        float(center[0] + span * 0.5),
        float(center[1] - span * 0.5),
        float(center[1] + span * 0.5),
    )


def _mark_direction(ax: plt.Axes, path: np.ndarray, color: str) -> None:
    if path.shape[0] < 3:
        return
    index = max(1, path.shape[0] // 2)
    ax.annotate(
        "",
        xy=path[index],
        xytext=path[index - 1],
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.15},
        zorder=8,
    )


def _plot_branch(
    ax: plt.Axes,
    branch: int,
    samples: list[dict[str, Any]],
    entity_index: int,
    predictions: dict[str, np.ndarray],
    target_velocity: np.ndarray,
    num_entities: int,
    dt: float,
    limits: tuple[float, ...],
) -> None:
    sample = samples[branch]
    target_agent_index = 2 + entity_index
    history = sample["history"].numpy()
    current = sample["current_pos"].numpy()
    robot_velocity = sample["future"][:2].numpy()

    for other_index in range(num_entities):
        agent_index = 2 + other_index
        if other_index == entity_index:
            continue
        other_future = _positions(
            current[agent_index], sample["future"][agent_index].numpy(), dt
        )
        ax.plot(
            history[agent_index, :, 0],
            history[agent_index, :, 1],
            color=CONTEXT,
            linewidth=0.45,
            linestyle=(0, (1.2, 1.6)),
            alpha=0.42,
            zorder=1,
        )
        ax.plot(
            other_future[:, 0],
            other_future[:, 1],
            color=CONTEXT,
            linewidth=0.55,
            alpha=0.58,
            zorder=1,
        )
        ax.scatter(
            other_future[:, 0],
            other_future[:, 1],
            s=5.0,
            color=CONTEXT,
            alpha=0.58,
            linewidths=0,
            zorder=2,
        )

    for robot_index, color in enumerate(ROBOT_COLORS):
        robot_path = _positions(current[robot_index], robot_velocity[robot_index], dt)
        ax.plot(
            history[robot_index, :, 0],
            history[robot_index, :, 1],
            color=color,
            linewidth=0.65,
            linestyle=(0, (1.2, 1.6)),
            alpha=0.65,
            zorder=4,
        )
        ax.plot(
            robot_path[:, 0],
            robot_path[:, 1],
            color=color,
            linewidth=0.85,
            zorder=5,
        )
        ax.scatter(
            robot_path[:, 0],
            robot_path[:, 1],
            marker="o",
            s=8.0,
            facecolor=color,
            edgecolor=color,
            linewidth=0.2,
            zorder=7,
        )
        ax.scatter(
            robot_path[-1, 0],
            robot_path[-1, 1],
            marker="*",
            s=34,
            facecolor=color,
            edgecolor=WHITE,
            linewidth=0.45,
            zorder=8,
        )

    target_current = current[target_agent_index]
    ground_truth = _positions(target_current, target_velocity[branch, entity_index], dt)
    joint = _positions(target_current, predictions["joint"][branch, entity_index], dt)
    matched = _positions(
        target_current, predictions["matched-additive"][branch, entity_index], dt
    )
    ax.plot(
        history[target_agent_index, :, 0],
        history[target_agent_index, :, 1],
        color=SLATE,
        linewidth=0.55,
        linestyle=(0, (1.2, 1.6)),
        alpha=0.85,
        zorder=5,
    )
    ax.scatter(
        history[target_agent_index, :, 0],
        history[target_agent_index, :, 1],
        s=8.0,
        facecolor=WHITE,
        edgecolor=SLATE,
        linewidth=0.45,
        zorder=7,
    )
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], color=INK, linewidth=1.15, zorder=8)
    ax.plot(joint[:, 0], joint[:, 1], color=JOINT, linewidth=0.95, zorder=9)
    ax.plot(
        matched[:, 0],
        matched[:, 1],
        color=MATCHED,
        linewidth=0.95,
        linestyle=(0, (2.2, 1.5)),
        zorder=8,
    )
    marker_steps = np.arange(0, ground_truth.shape[0])
    for path, color, marker in (
        (ground_truth, INK, "o"),
        (joint, JOINT, "s"),
        (matched, MATCHED, "D"),
    ):
        ax.scatter(
            path[marker_steps, 0],
            path[marker_steps, 1],
            s=8.0,
            marker=marker,
            facecolor=color,
            edgecolor=WHITE if marker != "o" else color,
            linewidth=0.25,
            zorder=9,
        )

    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    ax.text(
        0.5,
        -0.16,
        f"{panel_labels[branch]}  {BRANCH_TITLES[branch].replace('  ', ': ')}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.0,
        fontfamily="STIXGeneral",
        fontweight="normal",
        color=INK,
    )
    ax.tick_params(length=2.2, width=0.55, pad=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color(SLATE)


def _plot_interaction(
    ax: plt.Axes,
    entity_index: int,
    predictions: dict[str, np.ndarray],
    target_velocity: np.ndarray,
    dt: float,
) -> dict[str, float]:
    ground_truth_velocity = _factorial_second_difference(target_velocity)[entity_index]
    joint_velocity = _factorial_second_difference(predictions["joint"])[entity_index]
    matched_velocity = _factorial_second_difference(
        predictions["matched-additive"]
    )[entity_index]
    trajectories = {
        "Ground truth": _positions(np.zeros(2, dtype=np.float32), ground_truth_velocity, dt),
        "Joint": _positions(np.zeros(2, dtype=np.float32), joint_velocity, dt),
        "Matched additive": _positions(
            np.zeros(2, dtype=np.float32), matched_velocity, dt
        ),
    }
    styles = {
        "Ground truth": (INK, "-", "o", 2.0),
        "Joint": (JOINT, "-", "s", 1.7),
        "Matched additive": (MATCHED, (0, (3, 1.7)), "D", 1.45),
    }
    for label, trajectory in trajectories.items():
        color, linestyle, marker, linewidth = styles[label]
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=4,
        )
        ax.scatter(
            trajectory[[4, 8, 12], 0],
            trajectory[[4, 8, 12], 1],
            color=color,
            marker=marker,
            s=13,
            edgecolor=WHITE,
            linewidth=0.35,
            zorder=6,
        )
        _mark_direction(ax, trajectory, color)

    joint_rmse = float(np.sqrt(np.mean(np.square(joint_velocity - ground_truth_velocity))))
    matched_rmse = float(
        np.sqrt(np.mean(np.square(matched_velocity - ground_truth_velocity)))
    )
    limits = _shared_limits(list(trajectories.values()), padding=0.08)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID, linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.axhline(0.0, color=GRID, linewidth=0.7, zorder=0)
    ax.axvline(0.0, color=GRID, linewidth=0.7, zorder=0)
    ax.set_xlabel(r"Second-order displacement, $\Delta_{12}x$ (m)")
    ax.set_ylabel(r"Second-order displacement, $\Delta_{12}y$ (m)")
    ax.set_title("Recovered joint response", pad=5.0, fontweight="bold")
    ax.text(
        -0.12,
        1.045,
        "E",
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        0.03,
        0.97,
        f"Interaction RMSE\nJoint  {joint_rmse:.3f}\nMatched  {matched_rmse:.3f}",
        transform=ax.transAxes,
        fontsize=6.4,
        color=SLATE,
        va="top",
        linespacing=1.35,
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.5, "alpha": 0.9},
    )
    return {"joint_interaction_rmse": joint_rmse, "matched_interaction_rmse": matched_rmse}


def _make_figure(
    samples: list[dict[str, Any]],
    entity_index: int,
    predictions: dict[str, np.ndarray],
    target_velocity: np.ndarray,
    num_entities: int,
    dt: float,
) -> tuple[plt.Figure, dict[str, float]]:
    _style()
    all_points: list[np.ndarray] = []
    for branch, sample in enumerate(samples):
        current = sample["current_pos"].numpy()
        history = sample["history"].numpy()
        all_points.extend([history[:2], current[2:]])
        for robot_index in range(2):
            all_points.append(
                _positions(
                    current[robot_index], sample["future"][robot_index].numpy(), dt
                )
            )
        target_current = current[2 + entity_index]
        all_points.extend(
            [
                _positions(target_current, target_velocity[branch, entity_index], dt),
                _positions(target_current, predictions["joint"][branch, entity_index], dt),
                _positions(
                    target_current,
                    predictions["matched-additive"][branch, entity_index],
                    dt,
                ),
            ]
        )
    limits = _shared_limits(all_points)

    figure = plt.figure(figsize=(7.16, 6.15))
    grid_spec = figure.add_gridspec(2, 2, wspace=0.12, hspace=0.31)
    branch_axes = (
        figure.add_subplot(grid_spec[0, 0]),
        figure.add_subplot(grid_spec[0, 1]),
        figure.add_subplot(grid_spec[1, 0]),
        figure.add_subplot(grid_spec[1, 1]),
    )
    for branch, axis in enumerate(branch_axes):
        _plot_branch(
            axis,
            branch,
            samples,
            entity_index,
            predictions,
            target_velocity,
            num_entities,
            dt,
            limits,
        )
    metrics = _interaction_metrics(entity_index, predictions, target_velocity)
    handles = [
        Line2D([0], [0], color=ROBOT_COLORS[0], lw=0.9, marker="o", markersize=3.0, label="Robot 1"),
        Line2D([0], [0], color=ROBOT_COLORS[1], lw=0.9, marker="o", markersize=3.0, label="Robot 2"),
        Line2D([0], [0], color=SLATE, lw=0.6, marker="o", markerfacecolor=WHITE, markersize=3.0, label="Observed"),
        Line2D([0], [0], color=INK, lw=1.1, marker="o", markersize=3.0, label="Ground truth"),
        Line2D([0], [0], color=JOINT, lw=0.95, marker="s", markersize=3.0, label="Joint"),
        Line2D([0], [0], color=MATCHED, lw=0.95, linestyle=(0, (2.2, 1.5)), marker="D", markersize=2.8, label="Matched additive"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=False,
        handlelength=1.25,
        labelspacing=0.28,
        columnspacing=1.0,
    )
    figure.subplots_adjust(left=0.085, right=0.98, top=0.91, bottom=0.075)
    return figure, metrics


def _interaction_metrics(
    entity_index: int,
    predictions: dict[str, np.ndarray],
    target_velocity: np.ndarray,
) -> dict[str, float]:
    ground_truth_velocity = _factorial_second_difference(target_velocity)[entity_index]
    joint_velocity = _factorial_second_difference(predictions["joint"])[entity_index]
    matched_velocity = _factorial_second_difference(predictions["matched-additive"])[entity_index]
    return {
        "joint_interaction_rmse": float(
            np.sqrt(np.mean(np.square(joint_velocity - ground_truth_velocity)))
        ),
        "matched_interaction_rmse": float(
            np.sqrt(np.mean(np.square(matched_velocity - ground_truth_velocity)))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a prediction-independent representative H1 factorial trajectory case."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--selection-quantile", type=float, default=0.875)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not 0.0 <= args.selection_quantile <= 1.0:
        raise ValueError("selection quantile must be in [0, 1]")
    raw = _load_yaml(args.config)
    formal = raw.get("formal_protocol", {})
    suite = raw.get("suite", {})
    data_seed = int(formal.get("formal_test_data_seed", 328731))
    seeds = [int(seed) for seed in suite.get("seeds", [2027, 2028, 2029, 2030, 2031])]
    dataset = make_causal_response_dataset_from_config(
        raw, split="test", effect_mode="nonlinear", seed=data_seed
    )
    base_scene_index, entity_index, selection = _select_representative_case(
        dataset, args.selection_quantile
    )
    first = base_scene_index * dataset.interventions_per_scene
    samples = [dataset[first + branch] for branch in range(4)]
    scene = collate_synthetic_scene_batch(samples)
    interaction_radius = float(raw.get("evaluation", {}).get("interaction_radius", 4.0))
    batch = build_entity_response_batch(
        scene,
        num_robots=dataset.num_robots,
        interaction_radius=interaction_radius,
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )
    flat_predictions, checkpoint_paths = _ensemble_predictions(
        args.checkpoint_root, seeds, batch, device
    )
    predictions = {
        variant: values.reshape(4, dataset.num_entities, dataset.horizon, 2)
        for variant, values in flat_predictions.items()
    }
    target_velocity = batch.target_future_vel.numpy().reshape(
        4, dataset.num_entities, dataset.horizon, 2
    )

    figure, metrics = _make_figure(
        samples,
        entity_index,
        predictions,
        target_velocity,
        dataset.num_entities,
        dataset.dt,
    )
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".pdf"), dpi=600)
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=600)
    plt.close(figure)

    np.savez_compressed(
        args.output_stem.with_name(args.output_stem.name + "_data.npz"),
        target_velocity=target_velocity,
        joint_velocity=predictions["joint"],
        matched_additive_velocity=predictions["matched-additive"],
        target_history=samples[0]["history"][2 + entity_index].numpy(),
        robot_history=samples[0]["history"][:2].numpy(),
        robot_branch_velocity=np.stack(
            [sample["future"][:2].numpy() for sample in samples], axis=0
        ),
        agent_current=samples[0]["current_pos"].numpy(),
        dt=np.asarray(dataset.dt),
    )
    provenance = {
        "figure": str(args.output_stem.with_suffix(".pdf").relative_to(ROOT)),
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": _sha256(args.config),
        "effect_mode": "nonlinear",
        "split": "test",
        "formal_test_data_seed": data_seed,
        "model_seeds": seeds,
        "ensemble": "arithmetic mean of predictive means",
        "base_scene_index": base_scene_index,
        "base_scene_id": int(samples[0]["base_scene_id"]),
        "entity_index": entity_index,
        "selection_rule": (
            "Closest entity-scene interaction RMS to the requested quantile over all "
            "formal nonlinear test entities; model predictions are not used for selection."
        ),
        **selection,
        **metrics,
        "checkpoint_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in checkpoint_paths
        },
    }
    provenance_path = args.output_stem.with_name(args.output_stem.name + "_provenance.json")
    with provenance_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
