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
from matplotlib.lines import Line2D

from tracer.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    stationary_reference_candidates,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "experiments" / "real_h1_navware_20260906" / "navware_test.npz"
DEFAULT_CHECKPOINT = Path("experiments/real_h1_navware_relative_20260906_seed2027/seed2027/response.pt")
DEFAULT_OUTPUT = ROOT / "reports" / "paper_visualizations_20260906" / "fig_navware_real_trajectories"

INK = "#20262E"
SLATE = "#66727D"
GRID = "#D9E0E5"
CONTEXT = "#C6CDD3"
ROBOT = "#0072B2"
JOINT = "#009E73"
STATIONARY = "#E69F00"
WHITE = "#FFFFFF"


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
            "axes.edgecolor": SLATE,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positions(current: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
    return np.concatenate(
        [current[None], current[None] + np.cumsum(velocity * dt, axis=0)], axis=0
    )


def _select_cases(arrays: dict[str, np.ndarray]) -> list[dict[str, int]]:
    history = arrays["entity_history"]
    current = arrays["robot_current"]
    candidates = arrays["robot_candidates"]
    active = arrays["active_mask"]
    near = arrays["high_interaction"]
    displacement = np.linalg.norm(
        candidates[..., 0, -1, :2] - current[..., 0, :2], axis=-1
    )
    eligible = np.argwhere(
        active
        & near
        & (displacement > 0.05)
        & np.isfinite(history).all(axis=(-1, -2))
        & np.isfinite(arrays["target_future_vel"]).all(axis=(-1, -2))
    )
    if not len(eligible):
        eligible = np.argwhere(
            active
            & (displacement > 0.05)
            & np.isfinite(history).all(axis=(-1, -2))
        )
    pair_near = arrays["high_interaction"].sum(axis=-1)
    pair_active = arrays["active_mask"].sum(axis=-1)
    pair_displacement = displacement.mean(axis=-1)
    pair_score = pair_near * 10.0 + pair_displacement
    pair_score[pair_active < 7] = -1.0
    episode_index, entity_slot = np.unravel_index(np.argmax(pair_score), pair_score.shape)
    available_windows = np.flatnonzero(arrays["active_mask"][episode_index, entity_slot])
    windows = available_windows[np.linspace(0, len(available_windows) - 1, 4).round().astype(int)]
    return [
        {
            "episode_index": int(episode_index),
            "entity_slot": int(entity_slot),
            "window": int(window),
        }
        for window in windows
    ]


def _load_model(path: Path, device: torch.device) -> BeliefConditionedTeamResponseModel:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = BeliefConditionedTeamResponseModel(
        PersistentTrajectoryModelConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def _case_arrays(
    arrays: dict[str, np.ndarray], cases: list[dict[str, int]], model, device: torch.device
) -> list[dict[str, Any]]:
    episodes = np.asarray([case["episode_index"] for case in cases], dtype=np.int64)
    slots = np.asarray([case["entity_slot"] for case in cases], dtype=np.int64)
    windows = np.asarray([case["window"] for case in cases], dtype=np.int64)
    history = arrays["entity_history"][episodes, slots]
    robot_current = arrays["robot_current"][episodes, slots]
    robot_candidates = arrays["robot_candidates"][episodes, slots]
    history_t = torch.from_numpy(history)
    current_t = torch.from_numpy(robot_current)
    candidate_t = torch.from_numpy(robot_candidates)
    stationary_t = stationary_reference_candidates(
        current_t, int(model.config.horizon)
    )
    with torch.inference_mode():
        action_output = model(history_t.to(device), current_t.to(device), candidate_t.to(device))
        stationary_output = model(
            history_t.to(device), current_t.to(device), stationary_t.to(device)
        )
    velocity_anchor = (history[..., -1, :] - history[..., -2, :]) / float(model.config.dt)
    action_velocity = action_output["mean"].mean(dim=2).cpu().numpy() + velocity_anchor[:, :, None, :]
    stationary_velocity = stationary_output["mean"].mean(dim=2).cpu().numpy() + velocity_anchor[:, :, None, :]
    result = []
    for case_index, case in enumerate(cases):
        episode = int(case["episode_index"])
        slot = int(case["entity_slot"])
        window = int(case["window"])
        entity_current = history[case_index, window, -1]
        robot_now = entity_current + robot_current[case_index, window, 0, :2]
        robot_future = entity_current[None] + robot_candidates[case_index, window, 0, :, :2]
        target_future = arrays["target_future_vel"][episode, slot, window]
        result.append(
            {
                **case,
                "episode_id": int(arrays["episode_id"][episode]),
                "reference_origin": history[0, 0, -1],
                "history": history[case_index, window],
                "entity_current": entity_current,
                "robot_path": np.concatenate([robot_now[None], robot_future], axis=0),
                "target_future": _positions(entity_current, target_future, model.config.dt),
                "action_prediction": _positions(
                    entity_current, action_velocity[case_index, window], model.config.dt
                ),
                "stationary_prediction": _positions(
                    entity_current, stationary_velocity[case_index, window], model.config.dt
                ),
                "other_history": arrays["entity_history"][episode, :, window],
                "other_future": np.stack(
                    [
                        _positions(
                            arrays["entity_history"][episode, other_slot, window, -1],
                            arrays["target_future_vel"][episode, other_slot, window],
                            model.config.dt,
                        )
                        for other_slot in range(arrays["entity_history"].shape[1])
                    ]
                ),
            }
        )
    return result


def _limits(cases: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    points = []
    for case in cases:
        origin = case["reference_origin"]
        points.extend(
            [
                case["history"] - origin,
                case["robot_path"] - origin,
                case["target_future"] - origin,
                case["action_prediction"] - origin,
                case["stationary_prediction"] - origin,
                case["other_future"].reshape(-1, 2) - origin,
            ]
        )
    values = np.concatenate([np.asarray(item).reshape(-1, 2) for item in points])
    low = values.min(axis=0)
    high = values.max(axis=0)
    center = (low + high) / 2.0
    span = max(float(np.max(high - low)) + 0.8, 1.0)
    return (
        float(center[0] - span / 2),
        float(center[0] + span / 2),
        float(center[1] - span / 2),
        float(center[1] + span / 2),
    )


def _plot_case(ax: plt.Axes, case: dict[str, Any], panel_index: int, limits: tuple[float, ...]) -> None:
    origin = case["reference_origin"]
    history = case["history"] - origin
    for slot, other_history in enumerate(case["other_history"]):
        if slot == case["entity_slot"]:
            continue
        other_history = other_history - origin
        other_future = case["other_future"][slot] - origin
        ax.plot(other_history[:, 0], other_history[:, 1], color=CONTEXT, lw=0.55, ls=(0, (1.2, 1.6)), alpha=0.75)
        ax.plot(other_future[:, 0], other_future[:, 1], color=CONTEXT, lw=0.65, alpha=0.75)
        ax.scatter(other_future[:, 0], other_future[:, 1], s=6, color=CONTEXT, alpha=0.65, linewidths=0)

    robot = case["robot_path"] - origin
    ax.plot(robot[:, 0], robot[:, 1], color=ROBOT, lw=0.9)
    ax.scatter(robot[:, 0], robot[:, 1], s=8, color=ROBOT, linewidths=0)
    ax.scatter(robot[-1, 0], robot[-1, 1], marker="*", s=35, color=ROBOT, edgecolor=WHITE, linewidth=0.4, zorder=8)

    ax.plot(history[:, 0], history[:, 1], color=SLATE, lw=0.55, ls=(0, (1.2, 1.6)))
    ax.scatter(history[:, 0], history[:, 1], s=8, facecolor=WHITE, edgecolor=SLATE, linewidth=0.45, zorder=7)
    for path, color, linestyle, marker in (
        (case["target_future"] - origin, INK, "-", "o"),
        (case["action_prediction"] - origin, JOINT, "-", "s"),
        (case["stationary_prediction"] - origin, STATIONARY, (0, (2.2, 1.5)), "D"),
    ):
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.0, ls=linestyle, zorder=6)
        ax.scatter(path[:, 0], path[:, 1], s=8, marker=marker, color=color, edgecolor=WHITE, linewidth=0.25, zorder=7)

    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.tick_params(length=2.2, width=0.55, pad=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color(SLATE)
    labels = ("(a)", "(b)", "(c)", "(d)")
    ax.text(
        0.5,
        -0.16,
        f"{labels[panel_index]}  Episode {case['episode_id']}, window {case['window'] + 1}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        fontfamily="STIXGeneral",
        color=INK,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real NavWareSet response trajectories.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    _style()
    with np.load(args.data) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    cases = _select_cases(arrays)
    device = torch.device(args.device)
    model = _load_model(args.checkpoint, device)
    plotted = _case_arrays(arrays, cases, model, device)
    limits = _limits(plotted)

    figure, axes = plt.subplots(2, 2, figsize=(7.16, 6.15), sharex=True, sharey=True)
    for index, axis in enumerate(axes.flat):
        _plot_case(axis, plotted[index], index, limits)
        if index >= 2:
            axis.set_xlabel("local x (m)")
        if index % 2 == 0:
            axis.set_ylabel("local y (m)")
    handles = [
        Line2D([0], [0], color=ROBOT, lw=0.9, marker="o", markersize=3.0, label="Robot"),
        Line2D([0], [0], color=SLATE, lw=0.6, marker="o", markerfacecolor=WHITE, markersize=3.0, label="Observed"),
        Line2D([0], [0], color=INK, lw=1.0, marker="o", markersize=3.0, label="Ground truth"),
        Line2D([0], [0], color=JOINT, lw=1.0, marker="s", markersize=3.0, label="Recorded action"),
        Line2D([0], [0], color=STATIONARY, lw=1.0, ls=(0, (2.2, 1.5)), marker="D", markersize=2.8, label="Stationary action"),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False, handlelength=1.25, columnspacing=1.0)
    figure.subplots_adjust(left=0.085, right=0.98, top=0.91, bottom=0.075, wspace=0.12, hspace=0.31)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".pdf"), dpi=600)
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=600)
    plt.close(figure)

    provenance = {
        "figure": str(args.output_stem.with_suffix(".pdf").relative_to(ROOT)),
        "data": str(args.data),
        "data_sha256": _sha256(args.data),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "protocol": "NavWareSet held-out observational action-response validation",
        "model": "single seed 2027 response checkpoint; arithmetic mean over three response modes",
        "selection_rule": "one test episode/entity with maximal near-interaction coverage and nontrivial recorded robot displacement; four evenly spaced active windows; model errors not used",
        "cases": cases,
        "dt": float(model.config.dt),
        "history": int(model.config.history),
        "horizon": int(model.config.horizon),
    }
    provenance_path = args.output_stem.with_name(args.output_stem.name + "_provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
