from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Patch
import numpy as np
import torch

from tracer.config import load_sage_config, sage_config_from_dict
from tracer.data.heterogeneous_scene import AgentRole, HeterogeneousSceneBatch
from tracer.data.socialgym_adapter import collate_socialgym_scene_batch, make_socialgym_dataset_from_config
from tracer.experiments.checkpoint_compat import apply_checkpoint_compatibility
from tracer.experiments.eval_real_sage import override_guidance_from_args
from tracer.experiments.eval_synthetic_sage import load_checkpoint
from tracer.models.sage_model import SAGE


ROLE_STYLE = {
    int(AgentRole.ROBOT): ("Robot", "#2563eb"),
    int(AgentRole.PHA): ("PHA", "#0f766e"),
    int(AgentRole.HNAA): ("HNAA", "#9333ea"),
    int(AgentRole.SSAA): ("SSAA", "#ea580c"),
    int(AgentRole.UNKNOWN_ENTITY): ("Entity", "#64748b"),
}

RISK_RADIUS = {
    int(AgentRole.ROBOT): 0.75,
    int(AgentRole.PHA): 1.20,
    int(AgentRole.HNAA): 1.35,
    int(AgentRole.SSAA): 1.00,
    int(AgentRole.UNKNOWN_ENTITY): 1.05,
}

PATH_EFFECT = [pe.Stroke(linewidth=5.0, foreground="white", alpha=0.86), pe.Normal()]


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _dt_value(dt: float | torch.Tensor) -> float:
    if torch.is_tensor(dt):
        return float(dt.detach().cpu().reshape(-1)[0].item())
    return float(dt)


def _comfort_for_role(role: int, comfort: dict[int, object]) -> tuple[float, float]:
    if not comfort:
        return 1.2, 0.7
    default = comfort.get(int(AgentRole.UNKNOWN_ENTITY)) or next(iter(comfort.values()))
    margin = comfort.get(int(role), default)
    return float(margin.sigma_long), float(margin.sigma_lat)


def _velocity_at(path: np.ndarray, current: np.ndarray, t: int, dt: float) -> np.ndarray:
    prev = current if t == 0 else path[t - 1]
    return (path[t] - prev) / max(float(dt), 1e-6)


def _svr_state(
    robot_pos: np.ndarray,
    other_pos: np.ndarray,
    robot_vel: np.ndarray,
    sigma_long: float,
    sigma_lat: float,
    front_only: bool = True,
) -> tuple[float, bool, float, float]:
    speed = float(np.linalg.norm(robot_vel))
    heading = 0.0 if speed < 1e-6 else float(np.arctan2(robot_vel[1], robot_vel[0]))
    c, s = float(np.cos(heading)), float(np.sin(heading))
    dx, dy = other_pos - robot_pos
    lon = c * dx + s * dy
    lat = -s * dx + c * dy
    ellipse = (lon / max(sigma_long, 1e-6)) ** 2 + (lat / max(sigma_lat, 1e-6)) ** 2
    violation = ellipse < 1.0 and (not front_only or lon >= 0.0)
    return heading, bool(violation), float(ellipse), speed


def _find_risk_annotation(
    roles: np.ndarray,
    agent_mask: np.ndarray,
    robot_mask: np.ndarray,
    entity_mask: np.ndarray,
    current: np.ndarray,
    unguided: np.ndarray,
    guided: np.ndarray,
    comfort: dict[int, object],
    dt: float,
    max_speed: float,
) -> dict[str, np.ndarray | float | int] | None:
    robots = np.where(robot_mask)[0]
    entities = np.where(entity_mask & agent_mask)[0]
    best = None
    best_score = -float("inf")
    for ri in robots:
        for oi in entities:
            sigma_long, sigma_lat = _comfort_for_role(int(roles[oi]), comfort)
            radius = float(RISK_RADIUS.get(int(roles[oi]), 1.05))
            for t in range(unguided.shape[1]):
                ru = unguided[ri, t]
                rg = guided[ri, t]
                ou = unguided[oi, t]
                og = guided[oi, t]
                vu = _velocity_at(unguided[ri], current[ri], t, dt)
                vg = _velocity_at(guided[ri], current[ri], t, dt)
                heading_u, svr_u, ellipse_u, speed_u = _svr_state(ru, ou, vu, sigma_long, sigma_lat)
                heading_g, svr_g, ellipse_g, speed_g = _svr_state(rg, og, vg, sigma_long, sigma_lat)
                du = float(np.linalg.norm(ru - ou))
                dg = float(np.linalg.norm(rg - og))
                correction = float(np.linalg.norm(rg - ru))
                risk = max(radius - du, 0.0)
                gain = dg - du
                score = (
                    7.0 * float(svr_u)
                    + 3.4 * float(svr_u and not svr_g)
                    + 2.0 * risk
                    + 1.7 * max(gain, 0.0)
                    + 1.0 * correction
                    + 0.5 * float(speed_u > max_speed)
                    - 1.0 * float(svr_g)
                    - 0.04 * du
                )
                if score > best_score:
                    best_score = score
                    best = {
                        "robot_idx": int(ri),
                        "other_idx": int(oi),
                        "robot_unguided": ru,
                        "robot_guided": rg,
                        "other_unguided": ou,
                        "other_guided": og,
                        "sigma_long": sigma_long,
                        "sigma_lat": sigma_lat,
                        "heading_u": heading_u,
                        "heading_g": heading_g,
                        "time": int(t),
                        "du": du,
                        "dg": dg,
                        "correction": correction,
                        "ellipse_u": ellipse_u,
                        "ellipse_g": ellipse_g,
                        "svr_u": float(svr_u),
                        "svr_g": float(svr_g),
                        "speed_u": speed_u,
                        "speed_g": speed_g,
                        "speed_violate_u": float(speed_u > max_speed),
                        "speed_violate_g": float(speed_g > max_speed),
                        "score": float(score),
                    }
    return best


def _selection_score(
    batch: HeterogeneousSceneBatch,
    pred_unguided_pos: torch.Tensor,
    pred_guided_pos: torch.Tensor,
    comfort: dict[int, object],
    max_speed: float,
) -> float:
    robot_mask = batch.robot_mask & batch.agent_mask
    if not robot_mask.any():
        return 0.0
    shift = torch.linalg.norm(pred_guided_pos - pred_unguided_pos, dim=-1)
    robot_shift = shift[robot_mask[:, :, None].expand_as(shift)]
    shift_score = float(robot_shift.max().item() + 0.35 * robot_shift.mean().item()) if robot_shift.numel() else 0.0
    risk = _find_risk_annotation(
        _np(batch.roles[0]).astype(int),
        _np(batch.agent_mask[0]).astype(bool),
        _np(robot_mask[0]).astype(bool),
        _np(batch.entity_mask[0]).astype(bool) & _np(batch.agent_mask[0]).astype(bool),
        _np(batch.current_pos[0]),
        _np(pred_unguided_pos[0]),
        _np(pred_guided_pos[0]),
        comfort,
        _dt_value(batch.dt),
        max_speed,
    )
    return shift_score + (0.0 if risk is None else float(risk["score"]))


def _draw_path(ax, path: np.ndarray, color: str, lw: float, alpha: float, ls: str = "-", zorder: int = 1) -> None:
    ax.plot(path[:, 0], path[:, 1], color=color, lw=lw, alpha=alpha, ls=ls, zorder=zorder)


def _draw_speed_violation_segments(ax, path: np.ndarray, current: np.ndarray, dt: float, max_speed: float) -> float:
    speeds = np.asarray([float(np.linalg.norm(_velocity_at(path, current, t, dt))) for t in range(path.shape[0])])
    violation = speeds > float(max_speed)
    for t in np.where(violation)[0]:
        p0 = current if t == 0 else path[t - 1]
        p1 = path[t]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#f97316", lw=6.0, alpha=0.70, solid_capstyle="round", zorder=12)
    return float(violation.mean()) if violation.size else 0.0


def _focus_limits(
    history: np.ndarray,
    current: np.ndarray,
    unguided: np.ndarray,
    guided: np.ndarray,
    risk: dict[str, np.ndarray | float | int] | None,
    focus_robot: int,
    focus_other: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    parts = [history[focus_robot], current[focus_robot].reshape(1, 2), unguided[focus_robot], guided[focus_robot]]
    if focus_other >= 0:
        parts.extend([
            history[focus_other],
            current[focus_other].reshape(1, 2),
            unguided[focus_other],
            guided[focus_other],
        ])
    if risk is not None:
        r = max(float(risk["sigma_long"]), float(risk["sigma_lat"]))
        for key in ("robot_unguided", "robot_guided", "other_unguided", "other_guided"):
            p = np.asarray(risk[key])
            parts.append(np.array([[p[0] - r, p[1] - r], [p[0] + r, p[1] + r]]))
    points = np.concatenate(parts, axis=0)
    finite = points[np.isfinite(points).all(axis=1)]
    mn = finite.min(axis=0)
    mx = finite.max(axis=0)
    center = 0.5 * (mn + mx)
    span = max(float((mx - mn).max()) * 1.04, 2.2)
    return (center[0] - span / 2, center[0] + span / 2), (center[1] - span / 2, center[1] + span / 2)


def _inside_box(p: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]) -> bool:
    return xlim[0] <= p[0] <= xlim[1] and ylim[0] <= p[1] <= ylim[1]



def _variant_state(
    risk: dict[str, np.ndarray | float | int],
    variant: str,
    guided: np.ndarray,
    unguided: np.ndarray,
    current: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float, bool, float]:
    focus_robot = int(risk["robot_idx"])
    focus_other = int(risk["other_idx"])
    if variant == "guided":
        robot_point = np.asarray(risk["robot_guided"])
        other_point = np.asarray(risk["other_guided"])
        vel = _velocity_at(guided[focus_robot], current[focus_robot], int(risk["time"]), dt)
    else:
        robot_point = np.asarray(risk["robot_unguided"])
        other_point = np.asarray(risk["other_unguided"])
        vel = _velocity_at(unguided[focus_robot], current[focus_robot], int(risk["time"]), dt)
    heading, svr, _, speed = _svr_state(robot_point, other_point, vel, float(risk["sigma_long"]), float(risk["sigma_lat"]))
    return robot_point, other_point, heading, svr, speed


def _draw_variant_panel(
    ax,
    batch: HeterogeneousSceneBatch,
    unguided_t: torch.Tensor,
    guided_t: torch.Tensor,
    risk: dict[str, np.ndarray | float | int],
    variant: str,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    max_speed: float,
) -> None:
    history = _np(batch.history[0, :, :, :2])
    current = _np(batch.current_pos[0])
    unguided = _np(unguided_t[0])
    guided = _np(guided_t[0])
    roles = _np(batch.roles[0]).astype(int)
    agent_mask = _np(batch.agent_mask[0]).astype(bool)
    robot_mask = _np(batch.robot_mask[0]).astype(bool) & agent_mask
    entity_mask = _np(batch.entity_mask[0]).astype(bool) & agent_mask
    dt = _dt_value(batch.dt)
    focus_robot = int(risk["robot_idx"])
    focus_other = int(risk["other_idx"])

    pred = guided if variant == "guided" else unguided
    main_color = "#15803d" if variant == "guided" else "#b91c1c"
    title_color = "#166534" if variant == "guided" else "#7f1d1d"
    line_style = "-" if variant == "guided" else (0, (5.5, 2.0))
    marker = "o" if variant == "guided" else "x"

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)

    _role_label, entity_color = ROLE_STYLE.get(int(roles[focus_other]), ROLE_STYLE[int(AgentRole.UNKNOWN_ENTITY)])

    robot_point, other_point, heading, svr, _speed = _variant_state(risk, variant, guided, unguided, current, dt)
    ax.add_patch(
        Ellipse(
            (float(robot_point[0]), float(robot_point[1])),
            width=2.0 * float(risk["sigma_long"]),
            height=2.0 * float(risk["sigma_lat"]),
            angle=np.degrees(float(heading)),
            facecolor="#ef4444" if svr else "#22c55e",
            edgecolor="#b91c1c" if svr else "#15803d",
            lw=1.9,
            alpha=0.25 if svr else 0.16,
            zorder=6,
        )
    )

    ax.scatter(current[focus_robot, 0], current[focus_robot, 1], s=78, marker="o", color="#2563eb", edgecolor="white", linewidth=0.9, zorder=11)
    speed_rate = _draw_speed_violation_segments(ax, pred[focus_robot], current[focus_robot], dt, max_speed)
    robot_path = np.vstack([current[focus_robot].reshape(1, 2), pred[focus_robot]])
    ax.plot(
        robot_path[:, 0],
        robot_path[:, 1],
        color=main_color,
        lw=3.9,
        ls=line_style,
        marker=marker,
        markersize=4.4,
        markevery=3,
        alpha=0.98,
        zorder=14,
        path_effects=PATH_EFFECT,
    )
    ax.scatter(robot_point[0], robot_point[1], s=84, facecolor=main_color, edgecolor="white", linewidth=1.0, zorder=16)
    ax.scatter(other_point[0], other_point[1], s=108, marker="D", facecolor=entity_color, edgecolor="white", linewidth=1.0, zorder=16)
    ax.plot([robot_point[0], other_point[0]], [robot_point[1], other_point[1]], color=main_color, lw=1.8, alpha=0.82, zorder=15)

    if variant == "guided":
        ru = np.asarray(risk["robot_unguided"])
        rg = np.asarray(risk["robot_guided"])
        if _inside_box(ru, xlim, ylim) and _inside_box(rg, xlim, ylim) and np.linalg.norm(rg - ru) > 0.03:
            ax.add_patch(FancyArrowPatch((ru[0], ru[1]), (rg[0], rg[1]), arrowstyle="-|>", mutation_scale=19, lw=5.4, color="white", alpha=0.90, zorder=18))
            ax.add_patch(FancyArrowPatch((ru[0], ru[1]), (rg[0], rg[1]), arrowstyle="-|>", mutation_scale=17, lw=3.1, color="#16a34a", alpha=0.98, zorder=19))
            ax.scatter(ru[0], ru[1], s=54, marker="x", color="#b91c1c", linewidth=1.7, zorder=18)

    distance = float(np.linalg.norm(robot_point - other_point))
    status = "SVR violation" if svr else "SVR cleared"
    if speed_rate > 0.0:
        badge = f"{status}\nd={distance:.2f}m  speed={speed_rate:.0%}"
    else:
        badge = f"{status}\nd={distance:.2f}m"
    ax.text(
        0.025,
        0.965,
        badge,
        transform=ax.transAxes,
        fontsize=7.4,
        color=title_color,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.26", facecolor="white", edgecolor=main_color, alpha=0.94, linewidth=0.8),
        zorder=22,
    )
    ax.set_title(title, fontsize=10.4, color=title_color, pad=4)


def _legend_handles() -> list[Line2D | Patch]:
    return [
        Line2D([0], [0], color="#2563eb", marker="o", lw=0, markersize=6.5, label="Robot start"),
        Line2D([0], [0], color="#7c3aed", marker="D", lw=0, markersize=6.2, label="Interacting entity"),
        Patch(facecolor="#ef4444", edgecolor="#b91c1c", alpha=0.26, label="SVR violation zone"),
        Patch(facecolor="#22c55e", edgecolor="#15803d", alpha=0.18, label="SVR cleared zone"),
        Line2D([0], [0], color="#b91c1c", lw=3.2, ls=(0, (5.5, 2.0)), marker="x", markersize=5.5, label="Unguided robot path"),
        Line2D([0], [0], color="#15803d", lw=3.2, marker="o", markersize=4.8, label="Guided robot path"),
        Line2D([0], [0], color="#16a34a", lw=2.8, marker=">", markersize=6.5, label="Guidance correction"),
        Line2D([0], [0], color="#f97316", lw=4.2, label="Speed-limit violation"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize paired SocialGym qualitative SAGE cases.")
    parser.add_argument("--config", default="configs/sage_socialgym.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--source", choices=["style", "trajectory", "rollout", "file", "offline"], default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--num-entities", type=int, default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--indices", default="0,1")
    parser.add_argument("--candidate-pool", type=int, default=0, help="Scan the first N cases and keep the clearest SVR-removal panels.")
    parser.add_argument("--top-k", type=int, default=2, help="Number of before/after case rows to keep.")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--w-collision", type=float, default=None)
    parser.add_argument("--w-social", type=float, default=None)
    parser.add_argument("--w-robot-robot-collision", type=float, default=None)
    parser.add_argument("--w-kinematic", type=float, default=None)
    parser.add_argument("--w-goal", type=float, default=None)
    parser.add_argument("--goal-tolerance", type=float, default=None)
    parser.add_argument("--comfort-profile", choices=["role", "shared", "isotropic"], default="role")
    parser.add_argument("--shared-sigma-long", type=float, default=None)
    parser.add_argument("--shared-sigma-lat", type=float, default=None)
    parser.add_argument("--collision-margin", type=float, default=None)
    parser.add_argument("--max-speed", type=float, default=None)
    parser.add_argument("--out", default="results/figures/socialgym_qualitative_overview.png")
    args = parser.parse_args()

    cfg_from_file, raw_cfg = load_sage_config(args.config)
    ckpt = load_checkpoint(args.checkpoint, args.device)
    cfg = sage_config_from_dict(ckpt.get("config", None)) if "config" in ckpt else cfg_from_file
    cfg = apply_checkpoint_compatibility(cfg, ckpt.get("model", {}))
    override_guidance_from_args(cfg, args)

    overrides = {}
    if args.source is not None:
        overrides["source"] = args.source
    if args.data_path is not None:
        overrides["data_path"] = args.data_path
    if args.num_robots is not None:
        overrides["num_robots"] = args.num_robots
    if args.num_entities is not None:
        overrides["num_entities"] = args.num_entities
    if args.scene is not None:
        overrides["scene"] = args.scene
    dataset = make_socialgym_dataset_from_config(raw_cfg, split=args.split, **overrides)
    cfg.diffusion.horizon = int(getattr(dataset, "horizon", raw_cfg.get("dataset", {}).get("horizon", cfg.diffusion.horizon)))

    model = SAGE(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.candidate_pool > 0:
        indices = list(range(min(int(args.candidate_pool), len(dataset))))
    else:
        indices = [int(x) for x in args.indices.split(",") if x.strip()]
    indices = [idx for idx in indices if 0 <= idx < len(dataset)]
    if not indices:
        raise ValueError("No valid case indices were requested.")

    payload = []
    for idx in indices:
        batch_cpu = collate_socialgym_scene_batch([dataset[idx]])
        batch = batch_cpu.to(args.device)
        best_pair = None
        best_score = -float("inf")
        for sample_no in range(max(1, args.samples)):
            seed = 777 + idx * 1000 + sample_no
            torch.manual_seed(seed)
            if args.device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                unguided = model.generate(batch, guided=False, step_stride=args.stride)["future_pos"].detach().cpu()
            torch.manual_seed(seed)
            if args.device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                guided = model.generate(batch, guided=True, step_stride=args.stride)["future_pos"].detach().cpu()
            score = _selection_score(batch_cpu, unguided, guided, model.guidance.config.comfort, model.guidance.config.max_speed)
            if score > best_score:
                best_score = score
                best_pair = (unguided, guided)
        if best_pair is not None:
            payload.append((idx, batch_cpu, best_pair[0], best_pair[1], best_score))

    if args.candidate_pool > 0:
        payload = sorted(payload, key=lambda item: item[-1], reverse=True)[: max(1, int(args.top_k))]
    else:
        payload = payload[: max(1, int(args.top_k))]
    if not payload:
        raise RuntimeError("No qualitative payload was generated.")

    rows = len(payload)
    fig, axes = plt.subplots(rows * 2, 1, figsize=(5.25, 6.95 * rows + 0.98), dpi=300)
    axes_arr = np.asarray(axes).reshape(rows, 2)
    for row, (idx, batch_cpu, unguided, guided, score) in enumerate(payload):
        history = _np(batch_cpu.history[0, :, :, :2])
        current = _np(batch_cpu.current_pos[0])
        roles = _np(batch_cpu.roles[0]).astype(int)
        agent_mask = _np(batch_cpu.agent_mask[0]).astype(bool)
        robot_mask = _np(batch_cpu.robot_mask[0]).astype(bool) & agent_mask
        entity_mask = _np(batch_cpu.entity_mask[0]).astype(bool) & agent_mask
        risk = _find_risk_annotation(
            roles,
            agent_mask,
            robot_mask,
            entity_mask,
            current,
            _np(unguided[0]),
            _np(guided[0]),
            model.guidance.config.comfort,
            _dt_value(batch_cpu.dt),
            model.guidance.config.max_speed,
        )
        if risk is None:
            for ax in axes_arr[row]:
                ax.axis("off")
            continue
        xlim, ylim = _focus_limits(history, current, _np(unguided[0]), _np(guided[0]), risk, int(risk["robot_idx"]), int(risk["other_idx"]))
        left_title = f"({chr(97 + row * 2)}) no guidance: unsafe sample"
        right_title = f"({chr(98 + row * 2)}) SAGE guidance: local correction"
        _draw_variant_panel(axes_arr[row, 0], batch_cpu, unguided, guided, risk, "unguided", left_title, xlim, ylim, model.guidance.config.max_speed)
        _draw_variant_panel(axes_arr[row, 1], batch_cpu, unguided, guided, risk, "guided", right_title, xlim, ylim, model.guidance.config.max_speed)

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=2, frameon=False, fontsize=8.4, bbox_to_anchor=(0.5, 0.010))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.965, bottom=0.158, hspace=0.125)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, pad_inches=0.03)
    plt.close(fig)
    print(f"saved={out}")
    print("selected_indices=" + ",".join(str(item[0]) for item in payload))


if __name__ == "__main__":
    main()
