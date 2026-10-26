from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
from PIL import Image
import torch

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:  # pragma: no cover - optional qualitative dependency.
    imageio = None

from sage.config import load_sage_config, sage_config_from_dict
from sage.experiments.checkpoint_compat import apply_checkpoint_compatibility
from sage.data.heterogeneous_scene import AgentRole, HeterogeneousSceneBatch, integrate_velocities
from sage.data.perception import perception_kwargs_from_config
from sage.data.sdd_adapter import (
    SDDOfficialSceneDataset,
    SDDRawSceneDataset,
    collate_sdd_scene_batch,
    find_sdd_annotation_file,
    normalize_label,
    parse_sdd_scene_id,
    role_from_label,
)
from sage.experiments.eval_sdd_sage import load_checkpoint, override_guidance_from_args
from sage.evaluation.metrics import social_compliance_metrics, _comfort_tensors
from sage.models.sage_model import SAGE


ROLE_STYLE = {
    int(AgentRole.ROBOT): ("Robot proxy", "#2563eb"),
    int(AgentRole.PHA): ("PHA", "#0f766e"),
    int(AgentRole.HNAA): ("HNAA", "#b45309"),
    int(AgentRole.SSAA): ("SSAA", "#7c3aed"),
    int(AgentRole.UNKNOWN_ENTITY): ("Unknown", "#64748b"),
}


@dataclass(frozen=True)
class CaseInfo:
    index: int
    scene_id: str
    track_id: str
    timestep: int
    score: float
    active_entities: int
    pha: int
    hnaa: int
    ssaa: int


@dataclass
class SelectionMetrics:
    selection_score: float = float("-inf")
    guided_ade: float = float("nan")
    guided_fde: float = float("nan")
    unguided_fde: float = float("nan")
    mean_shift: float = 0.0
    max_shift: float = 0.0
    mean_endpoint_shift: float = 0.0
    max_endpoint_shift: float = 0.0
    mean_path_length: float = 0.0
    max_path_length: float = 0.0
    energy_unguided: float = 0.0
    energy_guided: float = 0.0
    energy_delta: float = 0.0
    min_distance_unguided: float = 0.0
    min_distance_guided: float = 0.0
    pir_unguided: float = 0.0
    pir_guided: float = 0.0
    svr_unguided: float = 0.0
    svr_guided: float = 0.0
    local_robot: int = -1
    local_entity: int = -1
    local_role: str = ""
    local_time: int = -1
    local_svr_unguided: float = 0.0
    local_svr_guided: float = 0.0
    local_pir_unguided: float = 0.0
    local_pir_guided: float = 0.0
    local_min_distance_unguided: float = 0.0
    local_min_distance_guided: float = 0.0
    passes_visible_filter: bool = False
    used_fallback: bool = False


def _scene_reference_path(sdd_root: Path, scene_id: str) -> Path:
    scene_name, video_id = parse_sdd_scene_id(scene_id)
    candidates = [
        sdd_root / scene_name / f"video{video_id}" / "reference.jpg",
        sdd_root / "annotations" / scene_name / f"video{video_id}" / "reference.jpg",
        sdd_root / scene_name / video_id / "reference.jpg",
        sdd_root / "annotations" / scene_name / video_id / "reference.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No SDD reference.jpg found for scene_id={scene_id!r} under {sdd_root}")


def _scene_video_path(sdd_root: Path, scene_id: str) -> Path | None:
    scene_name, video_id = parse_sdd_scene_id(scene_id)
    video_names = [f"video{video_id}", str(video_id), f"video_{video_id}"]
    roots = [
        sdd_root / "video" / scene_name,
        sdd_root / "videos" / scene_name,
        sdd_root / scene_name,
    ]
    filenames = ["video.mp4", "video.mov", "video.avi", "video.mkv"]
    for root in roots:
        for video_name in video_names:
            video_dir = root / video_name
            for filename in filenames:
                candidate = video_dir / filename
                if candidate.exists():
                    return candidate
            if video_dir.exists():
                for candidate in sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.mov")):
                    return candidate
    return None


def _scene_offset_scaled(dataset: SDDRawSceneDataset, scene_id: str) -> np.ndarray:
    df = dataset.raw_df.copy()
    df["frame"] = df["frame"].astype(float).astype(int)
    if dataset.frame_stride > 1:
        df["frame"] = df["frame"] // dataset.frame_stride
    df = df.groupby(["sceneId", "trackId", "frame"], as_index=False).agg({"x": "mean", "y": "mean"})
    scene_df = df[df["sceneId"].astype(str) == str(scene_id)]
    if scene_df.empty or not dataset.center_scene:
        return np.zeros(2, dtype=np.float32)
    return scene_df[["x", "y"]].mean().to_numpy(dtype=np.float32) / float(dataset.coordinate_scale)


def _case_frame_bin(dataset: SDDRawSceneDataset, scene_id: str, timestep: int) -> tuple[int, int, int]:
    df = dataset.raw_df.copy()
    df["frame"] = df["frame"].astype(float).astype(int)
    scene_df = df[df["sceneId"].astype(str) == str(scene_id)]
    if scene_df.empty:
        raw_frame_ds = int(timestep)
    else:
        frame_ds = scene_df["frame"] // int(dataset.frame_stride)
        raw_frame_ds = int(timestep) + int(frame_ds.min())
    start = int(raw_frame_ds * int(dataset.frame_stride))
    end = int((raw_frame_ds + 1) * int(dataset.frame_stride) - 1)
    return raw_frame_ds, start, end


def _load_scene_background(
    dataset: SDDRawSceneDataset,
    sdd_root: Path,
    case: CaseInfo,
    *,
    use_video: bool,
) -> tuple[Image.Image, str, int | None]:
    reference = Image.open(_scene_reference_path(sdd_root, case.scene_id)).convert("RGB")
    _, frame_start, frame_end = _case_frame_bin(dataset, case.scene_id, case.timestep)
    video_frame = int(round(0.5 * (frame_start + frame_end)))
    base_note = f"annotation bin t={case.timestep} | raw {frame_start}-{frame_end}"
    if use_video and imageio is not None:
        video_path = _scene_video_path(sdd_root, case.scene_id)
        if video_path is not None:
            try:
                reader = imageio.get_reader(str(video_path), "ffmpeg")
                try:
                    frame = np.asarray(reader.get_data(max(0, video_frame))).astype(np.uint8)
                finally:
                    reader.close()
                image = Image.fromarray(frame).convert("RGB")
                if image.size != reference.size:
                    image = image.resize(reference.size, Image.Resampling.BILINEAR)
                return image, f"{base_note} | video frame {video_frame}", video_frame
            except Exception as exc:
                return reference, f"{base_note} | reference bg, video read failed: {type(exc).__name__}", None
    if use_video and imageio is None:
        return reference, f"{base_note} | reference bg, imageio missing", None
    return reference, f"{base_note} | reference bg", None


def _annotation_boxes_for_case(
    dataset: SDDRawSceneDataset,
    sdd_root: Path,
    case: CaseInfo,
) -> list[dict]:
    raw_frame_ds, _, _ = _case_frame_bin(dataset, case.scene_id, case.timestep)
    annotation_path = find_sdd_annotation_file(sdd_root, case.scene_id)
    if annotation_path is None:
        return []

    grouped: dict[str, dict] = {}
    with annotation_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            track_id, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label = parts[:10]
            try:
                frame_i = int(float(frame))
                if frame_i // int(dataset.frame_stride) != raw_frame_ds:
                    continue
                xmin_f, ymin_f, xmax_f, ymax_f = float(xmin), float(ymin), float(xmax), float(ymax)
                lost_i = int(float(lost))
                occluded_i = int(float(occluded))
                generated_i = int(float(generated))
            except ValueError:
                continue
            label_n = normalize_label(label)
            item = grouped.setdefault(
                str(track_id),
                {
                    "track_id": str(track_id),
                    "label": label_n,
                    "role": role_from_label(label_n),
                    "xmin": [],
                    "ymin": [],
                    "xmax": [],
                    "ymax": [],
                    "lost": [],
                    "occluded": [],
                    "generated": [],
                },
            )
            item["xmin"].append(xmin_f)
            item["ymin"].append(ymin_f)
            item["xmax"].append(xmax_f)
            item["ymax"].append(ymax_f)
            item["lost"].append(lost_i)
            item["occluded"].append(occluded_i)
            item["generated"].append(generated_i)

    boxes: list[dict] = []
    for item in grouped.values():
        xmin = float(np.mean(item["xmin"]))
        ymin = float(np.mean(item["ymin"]))
        xmax = float(np.mean(item["xmax"]))
        ymax = float(np.mean(item["ymax"]))
        boxes.append(
            {
                "track_id": item["track_id"],
                "label": item["label"],
                "role": item["role"],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "x": 0.5 * (xmin + xmax),
                "y": 0.5 * (ymin + ymax),
                "lost": float(np.mean(item["lost"])),
                "occluded": float(np.mean(item["occluded"])),
                "generated": float(np.mean(item["generated"])),
            }
        )
    return boxes


def _shift_boxes(boxes: list[dict], crop_box: tuple[int, int, int, int]) -> list[dict]:
    left, top, _, _ = crop_box
    shifted = []
    for box in boxes:
        shifted.append({**box, "xmin": box["xmin"] - left, "xmax": box["xmax"] - left, "ymin": box["ymin"] - top, "ymax": box["ymax"] - top, "x": box["x"] - left, "y": box["y"] - top})
    return shifted


def _box_overlaps_panel(box: dict, width: int, height: int, margin: float = 6.0) -> bool:
    return not (box["xmax"] < -margin or box["xmin"] > width + margin or box["ymax"] < -margin or box["ymin"] > height + margin)


def _plot_path_with_arrow(
    ax: plt.Axes,
    points: np.ndarray,
    *,
    color: str,
    lw: float,
    alpha: float,
    zorder: int,
    linestyle: str = "-",
) -> None:
    if points.shape[0] < 2:
        return
    ax.plot(points[:, 0], points[:, 1], color=color, lw=lw, alpha=alpha, ls=linestyle, solid_capstyle="round", zorder=zorder)
    p0, p1 = points[-2], points[-1]
    if np.isfinite(p0).all() and np.isfinite(p1).all() and float(np.linalg.norm(p1 - p0)) > 1e-3:
        ax.annotate(
            "",
            xy=(float(p1[0]), float(p1[1])),
            xytext=(float(p0[0]), float(p0[1])),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=max(0.8, lw * 0.75), alpha=min(0.95, alpha + 0.1), shrinkA=0, shrinkB=0),
            zorder=zorder + 1,
        )


def _draw_agent_footprint(
    ax: plt.Axes,
    center: np.ndarray,
    geometry: np.ndarray | None,
    *,
    color: str,
    is_robot: bool,
    zorder: int,
) -> None:
    if geometry is None or not np.isfinite(geometry).all() or float(np.max(geometry)) <= 0:
        width, height = (30.0, 30.0) if is_robot else (20.0, 20.0)
    else:
        width = max(18.0, float(geometry[0]))
        height = max(18.0, float(geometry[1]))
    alpha = 0.20 if is_robot else 0.16
    ax.add_patch(Ellipse((float(center[0]), float(center[1])), width=width, height=height, facecolor=color, edgecolor=color, lw=1.2, alpha=alpha, zorder=zorder))
    ax.scatter(center[0], center[1], s=58 if is_robot else 26, color=color, edgecolor="white", linewidth=0.8 if is_robot else 0.55, alpha=0.98, zorder=zorder + 2)


def _draw_annotation_boxes(ax: plt.Axes, boxes: list[dict], crop_box: tuple[int, int, int, int], panel_w: int, panel_h: int) -> None:
    for box in _shift_boxes(boxes, crop_box):
        if not _box_overlaps_panel(box, panel_w, panel_h):
            continue
        role_id = int(box.get("role", int(AgentRole.UNKNOWN_ENTITY)))
        _, role_color = ROLE_STYLE.get(role_id, ROLE_STYLE[int(AgentRole.UNKNOWN_ENTITY)])
        lost_or_occ = box.get("lost", 0.0) >= 0.5 or box.get("occluded", 0.0) >= 0.5
        alpha = 0.46 if not lost_or_occ else 0.25
        linestyle = "--" if lost_or_occ else "-"
        width = max(3.0, float(box["xmax"] - box["xmin"]))
        height = max(3.0, float(box["ymax"] - box["ymin"]))
        ax.add_patch(
            Rectangle(
                (float(box["xmin"]), float(box["ymin"])),
                width,
                height,
                fill=False,
                edgecolor=role_color,
                lw=1.15,
                linestyle=linestyle,
                alpha=alpha,
                zorder=2,
            )
        )


def _to_pixel(points: np.ndarray, offset_scaled: np.ndarray, coordinate_scale: float) -> np.ndarray:
    return (points + offset_scaled[None, :]) * float(coordinate_scale)


def _tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _load_dataset(args: argparse.Namespace, raw_cfg: dict) -> SDDRawSceneDataset:
    dataset_cfg = raw_cfg.get("dataset", {})
    sdd_cfg = dataset_cfg.get("sdd", {})
    role_cfg = dataset_cfg.get("pseudo_roles", {})
    num_robot_proxies_cfg = int(dataset_cfg.get("num_robot_proxies", 1))
    perception_mode_cfg = dataset_cfg.get("perception", {}).get("mode", "nearest")
    perception_kwargs = perception_kwargs_from_config(dataset_cfg)

    data_path = Path(args.data_dir) / f"{args.split}_trajnet.pkl"
    source = args.source or sdd_cfg.get("source", "official")
    sdd_root = args.sdd_root or sdd_cfg.get("sdd_root")
    role_strategy = args.role_strategy or sdd_cfg.get("role_strategy", "semantic_or_speed")
    dataset_kwargs = dict(
        history=int(dataset_cfg.get("history", 8)),
        horizon=int(dataset_cfg.get("horizon", 12)),
        num_robot_proxies=num_robot_proxies_cfg,
        max_entities=int(args.max_entities if args.max_entities is not None else dataset_cfg.get("max_entities", 12)),
        dt=float(dataset_cfg.get("dt", 0.4)),
        frame_stride=int(args.frame_stride if args.frame_stride is not None else sdd_cfg.get("frame_stride", 12)),
        coordinate_scale=float(
            args.coordinate_scale if args.coordinate_scale is not None else sdd_cfg.get("coordinate_scale", 50.0)
        ),
        center_scene=bool(sdd_cfg.get("center_scene", True)),
        role_strategy=role_strategy,
        role_slow_speed=float(role_cfg.get("slow_speed", 0.15)),
        role_fast_speed=float(role_cfg.get("fast_speed", 1.8)),
        role_slow_quantile=float(role_cfg.get("slow_quantile", 0.33)),
        role_fast_quantile=float(role_cfg.get("fast_quantile", 0.67)),
        limit_scenes=args.limit_scenes,
        **perception_kwargs,
    )
    if source == "official":
        if not sdd_root:
            raise ValueError("--sdd-root is required for official SDD visualization")
        return SDDOfficialSceneDataset(
            sdd_root,
            split_reference_pkl=data_path,
            coordinate=args.coordinate or sdd_cfg.get("coordinate", "center"),
            **dataset_kwargs,
        )
    return SDDRawSceneDataset(data_path, label_map_path=args.label_map or sdd_cfg.get("label_map"), **dataset_kwargs)


def _score_sample(sample: dict) -> tuple[float, int, int, int, int]:
    roles = sample["roles"]
    entity_mask = sample["entity_mask"]
    active_entities = int(entity_mask.sum().item())
    pha = int(((roles == int(AgentRole.PHA)) & entity_mask).sum().item())
    hnaa = int(((roles == int(AgentRole.HNAA)) & entity_mask).sum().item())
    ssaa = int(((roles == int(AgentRole.SSAA)) & entity_mask).sum().item())
    robot_future = sample["future"][0]
    progress = float(torch.linalg.norm(robot_future.sum(dim=0)).item())
    diversity_bonus = 3.0 * int(hnaa > 0) + 4.0 * int(ssaa > 0) + 1.5 * int(pha > 0)
    score = active_entities + diversity_bonus + 0.1 * progress
    return score, active_entities, pha, hnaa, ssaa


def _select_cases(dataset: SDDRawSceneDataset, args: argparse.Namespace) -> list[CaseInfo]:
    if args.indices:
        chosen = [int(x) for x in args.indices.split(",") if x.strip()]
        cases = []
        for raw_idx in chosen:
            idx = int(raw_idx)
            sample = dataset[idx]
            item = dataset.indices[idx]
            scene = dataset.scenes[item.scene_idx]
            target = scene.tracks[item.track_idx]
            score, active, pha, hnaa, ssaa = _score_sample(sample)
            cases.append(
                CaseInfo(idx, str(scene.scene_id), str(target.track_id), int(item.timestep), score, active, pha, hnaa, ssaa)
            )
        return cases

    candidates: list[CaseInfo] = []
    scan_budget = min(len(dataset), max(1, int(args.max_scan)))
    if scan_budget >= len(dataset):
        scan_indices = np.arange(0, len(dataset), max(1, int(args.scan_stride)), dtype=int)
    else:
        scan_indices = np.unique(np.linspace(0, len(dataset) - 1, scan_budget, dtype=int))
    for idx in scan_indices:
        sample = dataset[int(idx)]
        score, active, pha, hnaa, ssaa = _score_sample(sample)
        if active < args.min_entities:
            continue
        item = dataset.indices[idx]
        scene = dataset.scenes[item.scene_idx]
        target = scene.tracks[item.track_idx]
        candidates.append(
            CaseInfo(idx, str(scene.scene_id), str(target.track_id), int(item.timestep), score, active, pha, hnaa, ssaa)
        )
    mode = getattr(args, "candidate_selection", "score")
    if mode == "random":
        rng = np.random.default_rng(int(getattr(args, "seed", 7)))
        candidates = list(candidates)
        rng.shuffle(candidates)
    elif mode == "spread":
        candidates.sort(key=lambda c: c.score, reverse=True)
        buckets: dict[str, list[CaseInfo]] = {}
        for case in candidates:
            buckets.setdefault(case.scene_id, []).append(case)
        spread: list[CaseInfo] = []
        scene_ids = sorted(buckets, key=lambda sid: buckets[sid][0].score, reverse=True)
        while len(spread) < len(candidates):
            added = False
            for scene_id in scene_ids:
                if buckets[scene_id]:
                    spread.append(buckets[scene_id].pop(0))
                    added = True
            if not added:
                break
        candidates = spread
    else:
        candidates.sort(key=lambda c: c.score, reverse=True)

    selected: list[CaseInfo] = []
    if args.unique_scenes:
        best_by_scene: dict[str, CaseInfo] = {}
        for case in candidates:
            if case.scene_id not in best_by_scene:
                best_by_scene[case.scene_id] = case
        selected = sorted(best_by_scene.values(), key=lambda c: c.score, reverse=True)[: args.num_cases]

    if len(selected) < args.num_cases:
        for case in candidates:
            if case not in selected:
                selected.append(case)
            if len(selected) >= args.num_cases:
                break
    return selected


def _square_crop_box(points: np.ndarray, image_width: int, image_height: int, pad: float = 130.0, min_span: float = 520.0) -> tuple[int, int, int, int]:
    if points.size == 0:
        cx, cy = image_width / 2.0, image_height / 2.0
        span = min(float(image_width), float(image_height), min_span)
    else:
        finite = points[np.isfinite(points).all(axis=1)]
        if finite.size == 0:
            cx, cy = image_width / 2.0, image_height / 2.0
            span = min(float(image_width), float(image_height), min_span)
        else:
            xmin, ymin = finite.min(axis=0)
            xmax, ymax = finite.max(axis=0)
            cx = 0.5 * float(xmin + xmax)
            cy = 0.5 * float(ymin + ymax)
            span = max(float(xmax - xmin), float(ymax - ymin)) + 2.0 * pad
            span = max(span, min_span)
            span = min(span, float(image_width), float(image_height))
    span = max(32.0, span)
    half = span / 2.0
    left = min(max(0.0, cx - half), max(0.0, float(image_width) - span))
    top = min(max(0.0, cy - half), max(0.0, float(image_height) - span))
    right = min(float(image_width), left + span)
    bottom = min(float(image_height), top + span)
    return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))


def _shift(points: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, _, _ = crop_box
    return points - np.array([left, top], dtype=np.float32)


def _legend_handles() -> list[Line2D | Rectangle]:
    handles: list[Line2D | Rectangle] = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="#0f766e", lw=1.1, alpha=0.55, label="Official annotations"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2563eb", markeredgecolor="white", markersize=6, label="Robot proxies"),
        Line2D([0], [0], color="#111827", lw=1.8, ls=(0, (2.2, 2.0)), label="GT future"),
        Line2D([0], [0], color="#dc2626", lw=2.0, ls="--", label="Unguided"),
        Line2D([0], [0], color="#16a34a", lw=2.3, label="SAGE guided"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#facc15", markeredgecolor="#111827", markersize=9, label="Waypoint"),
        Line2D([0], [0], color="#64748b", lw=1.5, alpha=0.75, label="Observed history"),
    ]
    for role_id in (int(AgentRole.PHA), int(AgentRole.HNAA), int(AgentRole.SSAA)):
        name, color = ROLE_STYLE[role_id]
        handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=5.5, label=name)
        )
    return handles


def _prediction_velocities_from_positions(pred_pos: torch.Tensor, batch_cpu: HeterogeneousSceneBatch) -> torch.Tensor:
    prev = torch.cat([batch_cpu.current_pos[:, :, None, :], pred_pos[:, :, :-1]], dim=2)
    return (pred_pos - prev) / float(batch_cpu.dt)


def _role_display_name(role_id: int) -> str:
    return ROLE_STYLE.get(int(role_id), ROLE_STYLE[int(AgentRole.UNKNOWN_ENTITY)])[0]


def _pairwise_social_state(
    pred_pos: torch.Tensor,
    pred_vel: torch.Tensor,
    batch_cpu: HeterogeneousSceneBatch,
    comfort: dict[int, object],
    *,
    personal_space_margin: float,
    front_only: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot_pos = pred_pos[:, :, None, :, :]
    entity_pos = pred_pos[:, None, :, :, :]
    rel = entity_pos - robot_pos
    dist = torch.sqrt((rel**2).sum(dim=-1) + eps)
    pair_mask = (
        batch_cpu.robot_mask[:, :, None]
        & batch_cpu.entity_mask[:, None, :]
        & batch_cpu.agent_mask[:, :, None]
        & batch_cpu.agent_mask[:, None, :]
    )
    pair_mask_t = pair_mask[:, :, :, None].expand_as(dist)

    robot_vel = pred_vel[:, :, None, :, :]
    heading = torch.atan2(robot_vel[..., 1], robot_vel[..., 0] + eps)
    cos_t = torch.cos(heading)
    sin_t = torch.sin(heading)
    dx = rel[..., 0]
    dy = rel[..., 1]
    long = cos_t * dx + sin_t * dy
    lat = -sin_t * dx + cos_t * dy
    sigma_long, sigma_lat = _comfort_tensors(batch_cpu.roles, comfort, eps=eps)
    sl = sigma_long[:, None, :, None]
    st = sigma_lat[:, None, :, None]
    ellipse = (long / sl) ** 2 + (lat / st) ** 2
    violation = ellipse < 1.0
    if front_only:
        violation = violation & (long >= 0)
    pir = dist < float(personal_space_margin)
    return dist, violation, pir, pair_mask, pair_mask_t


def _pair_stats(dist: torch.Tensor, violation: torch.Tensor, pir: torch.Tensor, ri: int, ei: int) -> tuple[float, float, float, int]:
    d = dist[0, ri, ei]
    v = violation[0, ri, ei]
    p = pir[0, ri, ei]
    min_idx = int(torch.argmin(d).item())
    return float(v.float().mean().item()), float(p.float().mean().item()), float(d.min().item()), min_idx


def _local_social_risk_metrics(
    pred_unguided: torch.Tensor,
    pred_guided: torch.Tensor,
    batch_cpu: HeterogeneousSceneBatch,
    comfort: dict[int, object],
    *,
    personal_space_margin: float,
    front_only: bool,
    eps: float,
) -> dict[str, object]:
    ung_vel = _prediction_velocities_from_positions(pred_unguided, batch_cpu)
    guided_vel = _prediction_velocities_from_positions(pred_guided, batch_cpu)
    ung_dist, ung_v, ung_pir, pair_mask, _ = _pairwise_social_state(
        pred_unguided,
        ung_vel,
        batch_cpu,
        comfort,
        personal_space_margin=personal_space_margin,
        front_only=front_only,
        eps=eps,
    )
    guided_dist, guided_v, guided_pir, _, _ = _pairwise_social_state(
        pred_guided,
        guided_vel,
        batch_cpu,
        comfort,
        personal_space_margin=personal_space_margin,
        front_only=front_only,
        eps=eps,
    )
    pair_indices = torch.nonzero(pair_mask[0], as_tuple=False)
    if pair_indices.numel() == 0:
        return {}

    best_score = -float("inf")
    best: dict[str, object] = {}
    for pair in pair_indices:
        ri = int(pair[0].item())
        ei = int(pair[1].item())
        u_svr, u_pir, u_md, u_t = _pair_stats(ung_dist, ung_v, ung_pir, ri, ei)
        g_svr, g_pir, g_md, _ = _pair_stats(guided_dist, guided_v, guided_pir, ri, ei)
        score = 5.0 * u_svr + 2.0 * u_pir + max(0.0, personal_space_margin - u_md)
        score += 2.0 * max(0.0, u_svr - g_svr) + 1.0 * max(0.0, u_pir - g_pir)
        if score > best_score:
            role_id = int(batch_cpu.roles[0, ei].item())
            best_score = score
            best = {
                "local_robot": ri,
                "local_entity": ei,
                "local_role": _role_display_name(role_id),
                "local_time": u_t,
                "local_svr_unguided": u_svr,
                "local_svr_guided": g_svr,
                "local_pir_unguided": u_pir,
                "local_pir_guided": g_pir,
                "local_min_distance_unguided": u_md,
                "local_min_distance_guided": g_md,
            }
    return best


def _build_selection_metrics(
    pred_unguided: torch.Tensor,
    pred_guided: torch.Tensor,
    batch_cpu: HeterogeneousSceneBatch,
    gt_pos: torch.Tensor,
    model: SAGE,
    out_unguided: dict | None = None,
    out_guided: dict | None = None,
) -> SelectionMetrics:
    ung_vel = _prediction_velocities_from_positions(pred_unguided, batch_cpu)
    guided_vel = _prediction_velocities_from_positions(pred_guided, batch_cpu)
    ung_social = social_compliance_metrics(
        ung_vel,
        batch_cpu,
        comfort=model.guidance.config.comfort,
        max_speed=model.guidance.config.max_speed,
        personal_space_margin=1.2,
        front_only=model.guidance.config.front_only,
        eps=model.guidance.config.eps,
    )
    guided_social = social_compliance_metrics(
        guided_vel,
        batch_cpu,
        comfort=model.guidance.config.comfort,
        max_speed=model.guidance.config.max_speed,
        personal_space_margin=1.2,
        front_only=model.guidance.config.front_only,
        eps=model.guidance.config.eps,
    )
    robot_mask = batch_cpu.robot_mask[0].bool()
    if bool(robot_mask.any()):
        shift = torch.linalg.norm(pred_guided[0, robot_mask] - pred_unguided[0, robot_mask], dim=-1)
        mean_shift = float(shift.mean().item())
        max_shift = float(shift.max().item())
        endpoint_shift = shift[:, -1]
        mean_endpoint_shift = float(endpoint_shift.mean().item())
        max_endpoint_shift = float(endpoint_shift.max().item())
        ung_prev = torch.cat([batch_cpu.current_pos[:, robot_mask, None, :], pred_unguided[:, robot_mask, :-1]], dim=2)
        guided_prev = torch.cat([batch_cpu.current_pos[:, robot_mask, None, :], pred_guided[:, robot_mask, :-1]], dim=2)
        ung_path = torch.linalg.norm(pred_unguided[:, robot_mask] - ung_prev, dim=-1).sum(dim=-1)
        guided_path = torch.linalg.norm(pred_guided[:, robot_mask] - guided_prev, dim=-1).sum(dim=-1)
        path_len = torch.maximum(ung_path, guided_path)
        mean_path_length = float(path_len.mean().item())
        max_path_length = float(path_len.max().item())
    else:
        mean_shift = 0.0
        max_shift = 0.0
        mean_endpoint_shift = 0.0
        max_endpoint_shift = 0.0
        mean_path_length = 0.0
        max_path_length = 0.0

    ung_diag = {} if out_unguided is None else out_unguided.get("diagnostics", {})
    guided_diag = {} if out_guided is None else out_guided.get("diagnostics", {})
    energy_unguided = float(ung_diag.get("energy", 0.0))
    energy_guided = float(guided_diag.get("energy", 0.0))
    local = _local_social_risk_metrics(
        pred_unguided,
        pred_guided,
        batch_cpu,
        model.guidance.config.comfort,
        personal_space_margin=1.2,
        front_only=model.guidance.config.front_only,
        eps=model.guidance.config.eps,
    )
    metrics = SelectionMetrics(
        guided_ade=_robot_prediction_score(pred_guided, gt_pos, batch_cpu.robot_mask, "robot_ade"),
        guided_fde=_robot_prediction_score(pred_guided, gt_pos, batch_cpu.robot_mask, "robot_fde"),
        unguided_fde=_robot_prediction_score(pred_unguided, gt_pos, batch_cpu.robot_mask, "robot_fde"),
        mean_shift=mean_shift,
        max_shift=max_shift,
        mean_endpoint_shift=mean_endpoint_shift,
        max_endpoint_shift=max_endpoint_shift,
        mean_path_length=mean_path_length,
        max_path_length=max_path_length,
        energy_unguided=energy_unguided,
        energy_guided=energy_guided,
        energy_delta=(energy_unguided - energy_guided) / max(energy_unguided, 1e-6),
        min_distance_unguided=float(ung_diag.get("min_distance", local.get("local_min_distance_unguided", 0.0))),
        min_distance_guided=float(guided_diag.get("min_distance", local.get("local_min_distance_guided", 0.0))),
        pir_unguided=float(ung_social.get("pir", 0.0)),
        pir_guided=float(guided_social.get("pir", 0.0)),
        svr_unguided=float(ung_social.get("svr", 0.0)),
        svr_guided=float(guided_social.get("svr", 0.0)),
        local_robot=int(local.get("local_robot", -1)),
        local_entity=int(local.get("local_entity", -1)),
        local_role=str(local.get("local_role", "")),
        local_time=int(local.get("local_time", -1)),
        local_svr_unguided=float(local.get("local_svr_unguided", 0.0)),
        local_svr_guided=float(local.get("local_svr_guided", 0.0)),
        local_pir_unguided=float(local.get("local_pir_unguided", 0.0)),
        local_pir_guided=float(local.get("local_pir_guided", 0.0)),
        local_min_distance_unguided=float(local.get("local_min_distance_unguided", 0.0)),
        local_min_distance_guided=float(local.get("local_min_distance_guided", 0.0)),
    )
    return metrics


def _draw_local_social_callout(
    ax: plt.Axes,
    metrics: SelectionMetrics | None,
    current_s: np.ndarray,
    unguided_s: np.ndarray,
    guided_s: np.ndarray,
    roles: np.ndarray,
) -> None:
    if metrics is None or metrics.local_robot < 0 or metrics.local_entity < 0:
        return
    ri = int(metrics.local_robot)
    ei = int(metrics.local_entity)
    t = int(max(0, min(metrics.local_time, unguided_s.shape[1] - 1)))
    if ri >= unguided_s.shape[0] or ei >= current_s.shape[0]:
        return
    entity_xy = current_s[ei]
    ung_xy = unguided_s[ri, t]
    guided_xy = guided_s[ri, t]
    if not (np.isfinite(entity_xy).all() and np.isfinite(ung_xy).all() and np.isfinite(guided_xy).all()):
        return

    role_name, role_color = ROLE_STYLE.get(int(roles[ei]), ROLE_STYLE[int(AgentRole.UNKNOWN_ENTITY)])
    ax.plot([entity_xy[0], ung_xy[0]], [entity_xy[1], ung_xy[1]], color="#ef4444", lw=1.15, alpha=0.78, zorder=21)
    ax.scatter(entity_xy[0], entity_xy[1], s=150, facecolors="none", edgecolors=role_color, linewidths=2.0, zorder=23)
    ax.scatter(ung_xy[0], ung_xy[1], s=112, facecolors="none", edgecolors="#dc2626", linewidths=2.0, zorder=23)
    ax.scatter(guided_xy[0], guided_xy[1], s=112, facecolors="none", edgecolors="#16a34a", linewidths=2.0, zorder=24)
    ax.annotate(
        "",
        xy=(float(guided_xy[0]), float(guided_xy[1])),
        xytext=(float(ung_xy[0]), float(ung_xy[1])),
        arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.55, alpha=0.92, shrinkA=2, shrinkB=2),
        zorder=25,
    )
    label = (
        f"local SVR {metrics.local_svr_unguided:.2f}->{metrics.local_svr_guided:.2f}\n"
        f"dmin {metrics.local_min_distance_unguided:.2f}->{metrics.local_min_distance_guided:.2f} m | {role_name}"
    )
    text_x = min(max(float(entity_xy[0]) + 12.0, 8.0), max(8.0, float(ax.get_xlim()[1]) - 145.0))
    text_y = min(max(float(entity_xy[1]) - 12.0, 18.0), max(18.0, float(ax.get_ylim()[0]) - 18.0))
    ax.text(
        text_x,
        text_y,
        label,
        fontsize=6.3,
        color="#111827",
        bbox=dict(facecolor="white", edgecolor=role_color, linewidth=0.8, alpha=0.86, pad=1.4),
        zorder=31,
    )


def _draw_case(
    ax: plt.Axes,
    image: Image.Image,
    batch_cpu: HeterogeneousSceneBatch,
    pred_unguided: torch.Tensor,
    pred_guided: torch.Tensor,
    case: CaseInfo,
    offset_scaled: np.ndarray,
    coordinate_scale: float,
    panel_label: str | None,
    annotation_boxes: list[dict] | None = None,
    frame_note: str | None = None,
    selection_metrics: SelectionMetrics | None = None,
    annotate_social_metrics: bool = False,
    crop_pad: float = 130.0,
    crop_min_span: float = 520.0,
) -> None:
    img = np.asarray(image)

    history = _to_pixel(_tensor_to_np(batch_cpu.history[0, :, :, :2]), offset_scaled, coordinate_scale)
    current = _to_pixel(_tensor_to_np(batch_cpu.current_pos[0]), offset_scaled, coordinate_scale)
    gt_pos = integrate_velocities(batch_cpu.future, batch_cpu.current_pos, batch_cpu.dt)[0]
    gt = _to_pixel(_tensor_to_np(gt_pos), offset_scaled, coordinate_scale)
    goals = _to_pixel(_tensor_to_np(batch_cpu.effective_goals()[0]), offset_scaled, coordinate_scale)
    unguided = _to_pixel(_tensor_to_np(pred_unguided[0]), offset_scaled, coordinate_scale)
    guided = _to_pixel(_tensor_to_np(pred_guided[0]), offset_scaled, coordinate_scale)
    roles = _tensor_to_np(batch_cpu.roles[0]).astype(int)
    agent_mask = _tensor_to_np(batch_cpu.agent_mask[0]).astype(bool)
    robot_mask = _tensor_to_np(batch_cpu.robot_mask[0]).astype(bool) & agent_mask
    entity_mask = _tensor_to_np(batch_cpu.entity_mask[0]).astype(bool) & agent_mask
    geometry = None if batch_cpu.geometry is None else _tensor_to_np(batch_cpu.geometry[0]) * float(coordinate_scale)

    robot_indices = np.where(robot_mask)[0]
    if robot_indices.size == 0:
        return

    crop_points = [
        current[agent_mask].reshape(-1, 2),
        history[agent_mask].reshape(-1, 2),
        gt[robot_indices].reshape(-1, 2),
        unguided[robot_indices].reshape(-1, 2),
        guided[robot_indices].reshape(-1, 2),
        goals[robot_indices].reshape(-1, 2),
    ]
    points = np.concatenate([arr for arr in crop_points if arr.size], axis=0)
    h, w = img.shape[:2]
    crop_box = _square_crop_box(points, w, h, pad=float(crop_pad), min_span=float(crop_min_span))
    left, top, right, bottom = crop_box
    crop_img = img[top:bottom, left:right]
    panel_h, panel_w = crop_img.shape[:2]
    ax.imshow(crop_img)

    if annotation_boxes:
        _draw_annotation_boxes(ax, annotation_boxes, crop_box, panel_w, panel_h)

    history_s = _shift(history, crop_box)
    current_s = _shift(current, crop_box)
    gt_s = _shift(gt, crop_box)
    goals_s = _shift(goals, crop_box)
    unguided_s = _shift(unguided, crop_box)
    guided_s = _shift(guided, crop_box)

    for i in np.where(entity_mask)[0]:
        _, color = ROLE_STYLE.get(int(roles[i]), ROLE_STYLE[int(AgentRole.UNKNOWN_ENTITY)])
        _plot_path_with_arrow(ax, history_s[i], color=color, lw=1.35, alpha=0.74, zorder=5)
        geom_i = None if geometry is None else geometry[i]
        _draw_agent_footprint(ax, current_s[i], geom_i, color=color, is_robot=False, zorder=7)

    for rank, i in enumerate(robot_indices):
        _plot_path_with_arrow(ax, history_s[i], color="#2563eb", lw=1.35, alpha=0.62, zorder=8)
        primary = rank == 0
        gt_lw = 2.1 if primary else 1.35
        pred_lw = 2.65 if primary else 1.65
        pred_alpha = 0.96 if primary else 0.72
        ax.plot(gt_s[i, :, 0], gt_s[i, :, 1], color="#111827", lw=gt_lw, alpha=0.82 if primary else 0.55, ls=(0, (2.2, 2.0)), zorder=10)
        ax.plot(unguided_s[i, :, 0], unguided_s[i, :, 1], color="#dc2626", lw=pred_lw, alpha=pred_alpha, ls="--", zorder=11)
        ax.plot(guided_s[i, :, 0], guided_s[i, :, 1], color="#16a34a", lw=pred_lw + 0.35, alpha=pred_alpha, zorder=12)
        geom_i = None if geometry is None else geometry[i]
        _draw_agent_footprint(ax, current_s[i], geom_i, color="#2563eb", is_robot=True, zorder=15)
        ax.scatter(goals_s[i, 0], goals_s[i, 1], marker="*", s=145 if primary else 95, color="#facc15", edgecolor="#111827", linewidth=0.65, zorder=16)
        ax.text(
            float(current_s[i, 0]) + 7.0,
            float(current_s[i, 1]) - 7.0,
            f"R{rank + 1}",
            fontsize=6.8,
            fontweight="bold",
            color="#1d4ed8",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8),
            zorder=18,
        )

    if annotate_social_metrics:
        _draw_local_social_callout(ax, selection_metrics, current_s, unguided_s, guided_s, roles)

    if panel_label:
        ax.text(
            0.025,
            0.965,
            panel_label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.2,
            fontweight="bold",
            color="#111827",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=1.6),
            zorder=30,
        )
    if frame_note:
        ax.text(
            0.025,
            0.035,
            frame_note,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=5.7,
            color="#334155",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.68, pad=1.2),
            zorder=30,
        )

    ax.set_xlim(0, panel_w)
    ax.set_ylim(panel_h, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def _save_case_figure(
    out_path: Path,
    image: Image.Image,
    batch_cpu: HeterogeneousSceneBatch,
    pred_unguided_pos: torch.Tensor,
    pred_guided_pos: torch.Tensor,
    case: CaseInfo,
    offset_scaled: np.ndarray,
    coordinate_scale: float,
    annotation_boxes: list[dict] | None = None,
    frame_note: str | None = None,
    selection_metrics: SelectionMetrics | None = None,
    annotate_social_metrics: bool = False,
    crop_pad: float = 130.0,
    crop_min_span: float = 520.0,
) -> None:
    fig, ax = plt.subplots(figsize=(4.05, 4.38), dpi=300)
    _draw_case(
        ax,
        image,
        batch_cpu,
        pred_unguided_pos,
        pred_guided_pos,
        case,
        offset_scaled,
        coordinate_scale,
        panel_label=f"{case.scene_id} | idx {case.index}",
        annotation_boxes=annotation_boxes,
        frame_note=frame_note,
        selection_metrics=selection_metrics,
        annotate_social_metrics=annotate_social_metrics,
        crop_pad=crop_pad,
        crop_min_span=crop_min_span,
    )
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4, frameon=False, fontsize=6.1, bbox_to_anchor=(0.5, 0.004))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.20)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, pad_inches=0.01)
    plt.close(fig)

def _save_overview(
    out_path: Path,
    images: list[Image.Image],
    batches: list[HeterogeneousSceneBatch],
    unguided: list[torch.Tensor],
    guided: list[torch.Tensor],
    cases: list[CaseInfo],
    offsets: list[np.ndarray],
    coordinate_scale: float,
    annotation_sets: list[list[dict]],
    frame_notes: list[str],
    selection_metrics_list: list[SelectionMetrics] | None = None,
    annotate_social_metrics: bool = False,
    crop_pad: float = 130.0,
    crop_min_span: float = 520.0,
) -> None:
    n = len(cases)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.05 * cols, 4.0 * rows + 0.55), dpi=320)
    axes_arr = np.atleast_1d(axes).reshape(-1)
    for i, ax in enumerate(axes_arr):
        if i >= n:
            ax.axis("off")
            continue
        label = f"({chr(ord('a') + i)}) {cases[i].scene_id} | idx {cases[i].index}"
        _draw_case(
            ax,
            images[i],
            batches[i],
            unguided[i],
            guided[i],
            cases[i],
            offsets[i],
            coordinate_scale,
            panel_label=label,
            annotation_boxes=annotation_sets[i],
            frame_note=frame_notes[i],
            selection_metrics=None if selection_metrics_list is None else selection_metrics_list[i],
            annotate_social_metrics=annotate_social_metrics,
            crop_pad=crop_pad,
            crop_min_span=crop_min_span,
        )
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=6, frameon=False, fontsize=6.9, bbox_to_anchor=(0.5, 0.010))
    fig.subplots_adjust(left=0.010, right=0.990, top=0.990, bottom=0.105, wspace=0.022, hspace=0.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, pad_inches=0.01)
    plt.close(fig)

def _write_metadata(path: Path, cases: Iterable[CaseInfo], metrics_list: Iterable[SelectionMetrics] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cases_list = list(cases)
    metrics_seq = list(metrics_list) if metrics_list is not None else [SelectionMetrics() for _ in cases_list]
    metric_fields = list(SelectionMetrics().__dict__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "scene_id", "track_id", "timestep", "score", "active_entities", "pha", "hnaa", "ssaa", *metric_fields],
        )
        writer.writeheader()
        for c, m in zip(cases_list, metrics_seq):
            row = dict(c.__dict__)
            row.update(m.__dict__)
            writer.writerow(row)


def _robot_prediction_score(pred_pos: torch.Tensor, gt_pos: torch.Tensor, robot_mask: torch.Tensor, mode: str) -> float:
    robots = robot_mask[0].bool()
    if not bool(robots.any()):
        return float("inf")
    diff = pred_pos[0, robots] - gt_pos[0, robots]
    dist = torch.linalg.norm(diff, dim=-1)
    if mode == "robot_ade":
        return float(dist.mean().item())
    return float(dist[:, -1].mean().item())


def _generate_representative(
    model: SAGE,
    batch: HeterogeneousSceneBatch,
    batch_cpu: HeterogeneousSceneBatch,
    args: argparse.Namespace,
    case_no: int,
    guided: bool,
) -> tuple[torch.Tensor, float]:
    gt_pos = integrate_velocities(batch_cpu.future, batch_cpu.current_pos, batch_cpu.dt)
    best_pos: torch.Tensor | None = None
    best_score = float("inf")
    mode = "robot_fde" if args.select_by in {"social_improvement", "visible_correction"} else args.select_by
    for sample_no in range(max(1, int(args.samples))):
        seed = int(args.seed) + case_no * 1000 + sample_no
        torch.manual_seed(seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            out = model.generate(batch, guided=guided, step_stride=args.stride)
        pred_pos = out["future_pos"].detach().cpu()
        score = _robot_prediction_score(pred_pos, gt_pos, batch_cpu.robot_mask, mode)
        if score < best_score:
            best_score = score
            best_pos = pred_pos
    if best_pos is None:
        raise RuntimeError("No prediction sample was generated.")
    return best_pos, best_score


def _generate_social_improvement_pair(
    model: SAGE,
    batch: HeterogeneousSceneBatch,
    batch_cpu: HeterogeneousSceneBatch,
    args: argparse.Namespace,
    case_no: int,
) -> tuple[torch.Tensor, torch.Tensor, SelectionMetrics]:
    gt_pos = integrate_velocities(batch_cpu.future, batch_cpu.current_pos, batch_cpu.dt)
    best_pair: tuple[torch.Tensor, torch.Tensor] | None = None
    best_metrics: SelectionMetrics | None = None
    best_score = -float("inf")
    fallback_pair: tuple[torch.Tensor, torch.Tensor] | None = None
    fallback_metrics: SelectionMetrics | None = None
    fallback_score = -float("inf")

    for sample_no in range(max(1, int(args.samples))):
        seed = int(args.seed) + case_no * 1000 + sample_no
        torch.manual_seed(seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            out_unguided = model.generate(batch, guided=False, step_stride=args.stride)
        torch.manual_seed(seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            out_guided = model.generate(batch, guided=True, step_stride=args.stride)

        pred_unguided = out_unguided["future_pos"].detach().cpu()
        pred_guided = out_guided["future_pos"].detach().cpu()
        metrics = _build_selection_metrics(pred_unguided, pred_guided, batch_cpu, gt_pos, model, out_unguided, out_guided)

        svr_gain = metrics.svr_unguided - metrics.svr_guided
        pir_gain = metrics.pir_unguided - metrics.pir_guided
        local_svr_gain = metrics.local_svr_unguided - metrics.local_svr_guided
        local_pir_gain = metrics.local_pir_unguided - metrics.local_pir_guided
        md_gain = metrics.local_min_distance_guided - metrics.local_min_distance_unguided
        score = 4.0 * metrics.mean_shift + 3.0 * metrics.max_shift
        score += 2.5 * metrics.mean_endpoint_shift + 2.0 * metrics.max_endpoint_shift
        score += 0.6 * metrics.mean_path_length + 0.4 * metrics.max_path_length
        score += 6.0 * max(0.0, svr_gain) + 3.0 * max(0.0, pir_gain)
        score += 5.0 * max(0.0, local_svr_gain) + 2.5 * max(0.0, local_pir_gain)
        score += 1.5 * max(0.0, md_gain)
        score += 2.0 * metrics.svr_unguided + 1.0 * metrics.pir_unguided
        score += 2.5 * metrics.local_svr_unguided + 1.2 * metrics.local_pir_unguided
        score += 0.35 * metrics.energy_delta
        score -= 0.25 * max(metrics.guided_fde - metrics.unguided_fde, 0.0)
        metrics.selection_score = float(score)

        if score > fallback_score:
            fallback_score = float(score)
            fallback_pair = (pred_unguided, pred_guided)
            fallback_metrics = metrics

        if metrics.guided_fde > float(args.max_guided_fde) or metrics.guided_ade > float(args.max_guided_ade):
            continue
        if args.select_by == "visible_correction":
            if metrics.mean_shift < float(args.min_visible_shift):
                continue
            if metrics.max_shift < float(args.min_max_shift):
                continue
            if metrics.max_endpoint_shift < float(args.min_endpoint_shift):
                continue
            if metrics.max_path_length < float(args.min_robot_path_length):
                continue
            social_gain = max(svr_gain, pir_gain, local_svr_gain, local_pir_gain)
            if social_gain < float(args.min_social_gain) and md_gain < float(args.min_dmin_gain):
                continue
            has_unguided_risk = (
                metrics.svr_unguided >= float(args.min_unguided_svr)
                or metrics.pir_unguided >= float(args.min_unguided_pir)
                or metrics.local_svr_unguided >= float(args.min_unguided_svr)
                or metrics.local_pir_unguided >= float(args.min_unguided_pir)
            )
            if not has_unguided_risk:
                continue
            improves = (
                svr_gain > 1e-4
                or pir_gain > 1e-4
                or local_svr_gain > 1e-4
                or local_pir_gain > 1e-4
                or md_gain > 0.03
            )
            if not args.allow_risk_worse and not improves:
                continue
        metrics.passes_visible_filter = True
        metrics.used_fallback = False
        if score > best_score:
            best_score = float(score)
            best_pair = (pred_unguided, pred_guided)
            best_metrics = metrics

    if best_pair is not None and best_metrics is not None:
        return best_pair[0], best_pair[1], best_metrics
    if fallback_pair is not None and fallback_metrics is not None:
        fallback_metrics.selection_score = fallback_score
        fallback_metrics.passes_visible_filter = False
        fallback_metrics.used_fallback = True
        return fallback_pair[0], fallback_pair[1], fallback_metrics

    pred_unguided, _ = _generate_representative(model, batch, batch_cpu, args, case_no, guided=False)
    pred_guided, fallback_fde = _generate_representative(model, batch, batch_cpu, args, case_no, guided=True)
    metrics = _build_selection_metrics(pred_unguided, pred_guided, batch_cpu, gt_pos, model)
    metrics.selection_score = -float(fallback_fde)
    metrics.passes_visible_filter = False
    metrics.used_fallback = True
    return pred_unguided, pred_guided, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SDD qualitative trajectory figures on real reference images.")
    parser.add_argument("--config", default="configs/sage_sdd.yaml")
    parser.add_argument("--data-dir", default="raw_data/stanford")
    parser.add_argument("--source", choices=["official", "trajnet"], default=None)
    parser.add_argument("--sdd-root", default=None)
    parser.add_argument("--coordinate", choices=["center", "bottom_center", "top_left"], default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--checkpoint", default="experiments/sage/sdd/goal/sdd_sage_epoch80.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument("--role-strategy", default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--coordinate-scale", type=float, default=None)
    parser.add_argument("--label-map", default=None)
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--candidate-pool", type=int, default=0, help="Evaluate this many automatically selected candidates, rank by visual/social correction, and save the top --num-cases cases. 0 means use --num-cases only.")
    parser.add_argument("--candidate-selection", choices=["score", "spread", "random"], default="score", help="How to build the automatic candidate pool before visual/social ranking.")
    parser.add_argument("--indices", default=None, help="Comma-separated dataset indices. Overrides automatic selection.")
    parser.add_argument("--max-scan", type=int, default=5000)
    parser.add_argument("--scan-stride", type=int, default=5)
    parser.add_argument("--min-entities", type=int, default=5)
    parser.add_argument("--unique-scenes", dest="unique_scenes", action="store_true", default=True)
    parser.add_argument("--allow-repeat-scenes", dest="unique_scenes", action="store_false")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=20, help="Number of diffusion samples used to select the representative trajectory for each panel.")
    parser.add_argument("--select-by", choices=["robot_fde", "robot_ade", "social_improvement", "visible_correction"], default="robot_fde")
    parser.add_argument("--min-visible-shift", type=float, default=0.0, help="Minimum mean guided-vs-unguided robot displacement for visible_correction selection, in scene units.")
    parser.add_argument("--min-max-shift", type=float, default=0.0, help="Minimum maximum guided-vs-unguided robot displacement for visible_correction selection, in scene units.")
    parser.add_argument("--min-endpoint-shift", type=float, default=0.0, help="Minimum maximum final-point guided-vs-unguided robot displacement for visible_correction selection.")
    parser.add_argument("--min-robot-path-length", type=float, default=0.0, help="Minimum maximum predicted robot path length for visible_correction selection.")
    parser.add_argument("--min-social-gain", type=float, default=0.0, help="Minimum SVR/PIR improvement for visible_correction selection unless --min-dmin-gain is satisfied.")
    parser.add_argument("--min-dmin-gain", type=float, default=0.0, help="Minimum local minimum-distance improvement for visible_correction selection unless --min-social-gain is satisfied.")
    parser.add_argument("--min-unguided-svr", type=float, default=0.0, help="Minimum unguided global or local SVR for visible_correction selection.")
    parser.add_argument("--min-unguided-pir", type=float, default=0.0, help="Minimum unguided global or local PIR for visible_correction selection.")
    parser.add_argument("--max-guided-ade", type=float, default=1.50, help="Reject visible-correction samples whose guided robot ADE is above this value.")
    parser.add_argument("--max-guided-fde", type=float, default=2.50, help="Reject visible-correction samples whose guided robot FDE is above this value.")
    parser.add_argument("--allow-risk-worse", action="store_true", help="Allow visible-correction candidates even when no social-risk metric improves.")
    parser.add_argument("--strict-visible-selection", action="store_true", help="When ranking candidates, only save cases that actually pass visible-correction filters; do not fill with fallback samples.")
    parser.add_argument("--annotate-social-metrics", action="store_true", help="Draw the highest-risk local robot-entity pair and its SVR/min-distance change.")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--w-collision", type=float, default=None)
    parser.add_argument("--w-social", type=float, default=None)
    parser.add_argument("--w-kinematic", type=float, default=None)
    parser.add_argument("--w-goal", type=float, default=None)
    parser.add_argument("--goal-tolerance", type=float, default=None)
    parser.add_argument("--comfort-profile", choices=["role", "shared", "isotropic"], default="role")
    parser.add_argument("--shared-sigma-long", type=float, default=None)
    parser.add_argument("--shared-sigma-lat", type=float, default=None)
    parser.add_argument("--collision-margin", type=float, default=None)
    parser.add_argument("--max-speed", type=float, default=None)
    parser.add_argument("--use-geometry-margin", action="store_true")
    parser.add_argument("--disable-geometry-margin", action="store_true")
    parser.add_argument("--out-dir", default="results/figures/sdd_qualitative")
    parser.add_argument("--paper-out", default="PAPER/figures/sdd_qualitative_overview.png")
    parser.add_argument("--hide-all-annotations", action="store_true", help="Do not draw faint boxes for all SDD annotations in the current frame bin.")
    parser.add_argument("--use-reference-bg", action="store_true", help="Use static reference.jpg instead of the timestamp-aligned video frame when available.")
    parser.add_argument("--crop-pad", type=float, default=130.0, help="Pixel padding around trajectories for each qualitative crop.")
    parser.add_argument("--crop-min-span", type=float, default=520.0, help="Minimum crop side length in pixels for each qualitative panel.")
    args = parser.parse_args()

    cfg_from_file, raw_cfg = load_sage_config(args.config)
    ckpt = load_checkpoint(args.checkpoint, args.device)
    cfg = sage_config_from_dict(ckpt.get("config", None)) if "config" in ckpt else cfg_from_file
    cfg = apply_checkpoint_compatibility(cfg, ckpt.get("model", {}))
    dataset_cfg = raw_cfg.get("dataset", {})
    cfg.diffusion.horizon = int(dataset_cfg.get("horizon", cfg.diffusion.horizon))
    override_guidance_from_args(cfg, args)

    dataset = _load_dataset(args, raw_cfg)
    requested_num_cases = int(args.num_cases)
    if int(args.candidate_pool) > requested_num_cases and not args.indices:
        args.num_cases = int(args.candidate_pool)
        cases = _select_cases(dataset, args)
        args.num_cases = requested_num_cases
    else:
        cases = _select_cases(dataset, args)
    if not cases:
        raise RuntimeError("No suitable SDD cases were found. Try --max-scan larger or --min-entities smaller.")

    model = SAGE(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(args.out_dir)
    sdd_cfg = raw_cfg.get("dataset", {}).get("sdd", {})
    sdd_root = Path(args.sdd_root or sdd_cfg.get("sdd_root"))

    candidates_payload: list[tuple[CaseInfo, Image.Image, HeterogeneousSceneBatch, torch.Tensor, torch.Tensor, np.ndarray, list[dict], str, SelectionMetrics]] = []

    for case_no, case in enumerate(cases):
        sample = dataset[case.index]
        batch_cpu = collate_sdd_scene_batch([sample])
        batch = batch_cpu.to(args.device)
        if args.select_by in {"social_improvement", "visible_correction"}:
            pred_unguided_pos, pred_guided_pos, selection_metrics = _generate_social_improvement_pair(
                model, batch, batch_cpu, args, case_no
            )
        else:
            pred_unguided_pos, unguided_score = _generate_representative(
                model, batch, batch_cpu, args, case_no, guided=False
            )
            pred_guided_pos, guided_score = _generate_representative(
                model, batch, batch_cpu, args, case_no, guided=True
            )
            gt_pos = integrate_velocities(batch_cpu.future, batch_cpu.current_pos, batch_cpu.dt)
            selection_metrics = _build_selection_metrics(pred_unguided_pos, pred_guided_pos, batch_cpu, gt_pos, model)
            selection_metrics.selection_score = -float(guided_score)
        image, frame_note, _ = _load_scene_background(
            dataset,
            sdd_root,
            case,
            use_video=not args.use_reference_bg,
        )
        offset = _scene_offset_scaled(dataset, case.scene_id)
        annotation_boxes = _annotation_boxes_for_case(dataset, sdd_root, case)
        if args.hide_all_annotations:
            annotation_boxes = []

        candidates_payload.append(
            (
                case,
                image,
                batch_cpu,
                pred_unguided_pos,
                pred_guided_pos,
                offset,
                annotation_boxes,
                frame_note,
                selection_metrics,
            )
        )

    if not candidates_payload:
        raise RuntimeError("No prediction candidates were generated.")
    candidates_payload.sort(key=lambda item: item[-1].selection_score, reverse=True)
    ranked_payload = candidates_payload
    if args.strict_visible_selection and args.select_by == "visible_correction":
        ranked_payload = [item for item in candidates_payload if item[-1].passes_visible_filter]
        if not ranked_payload:
            print("warning=no_candidates_passed_strict_visible_selection; saving best fallback candidates")
            ranked_payload = candidates_payload
    selected_payload = ranked_payload[:requested_num_cases]

    images: list[Image.Image] = []
    batches: list[HeterogeneousSceneBatch] = []
    unguided_positions: list[torch.Tensor] = []
    guided_positions: list[torch.Tensor] = []
    offsets: list[np.ndarray] = []
    annotation_sets: list[list[dict]] = []
    frame_notes: list[str] = []
    selection_metrics_list: list[SelectionMetrics] = []
    cases = []

    for case, image, batch_cpu, pred_unguided_pos, pred_guided_pos, offset, annotation_boxes, frame_note, selection_metrics in selected_payload:
        filename = f"sdd_{case.scene_id}_idx{case.index}_track{case.track_id}_t{case.timestep}.png"
        _save_case_figure(
            out_dir / filename,
            image,
            batch_cpu,
            pred_unguided_pos,
            pred_guided_pos,
            case,
            offset,
            dataset.coordinate_scale,
            annotation_boxes=annotation_boxes,
            frame_note=frame_note,
            selection_metrics=selection_metrics,
            annotate_social_metrics=args.annotate_social_metrics,
            crop_pad=args.crop_pad,
            crop_min_span=args.crop_min_span,
        )
        cases.append(case)
        images.append(image)
        batches.append(batch_cpu)
        unguided_positions.append(pred_unguided_pos)
        guided_positions.append(pred_guided_pos)
        offsets.append(offset)
        annotation_sets.append(annotation_boxes)
        frame_notes.append(frame_note)
        selection_metrics_list.append(selection_metrics)

    _write_metadata(out_dir / "candidate_cases_ranked.csv", [item[0] for item in candidates_payload], [item[-1] for item in candidates_payload])
    _save_overview(
        Path(args.paper_out),
        images,
        batches,
        unguided_positions,
        guided_positions,
        cases,
        offsets,
        dataset.coordinate_scale,
        annotation_sets,
        frame_notes,
        selection_metrics_list=selection_metrics_list,
        annotate_social_metrics=args.annotate_social_metrics,
        crop_pad=args.crop_pad,
        crop_min_span=args.crop_min_span,
    )
    _write_metadata(out_dir / "selected_cases.csv", cases, selection_metrics_list)

    print(f"saved_overview={args.paper_out}")
    print(f"saved_dir={out_dir}")
    for case, metrics in zip(cases, selection_metrics_list):
        print(
            f"case index={case.index} scene={case.scene_id} track={case.track_id} t={case.timestep} "
            f"entities={case.active_entities} pha={case.pha} hnaa={case.hnaa} ssaa={case.ssaa} score={case.score:.3f} "
            f"selection={metrics.selection_score:.3f} shift={metrics.mean_shift:.3f}/{metrics.max_shift:.3f} endpoint={metrics.mean_endpoint_shift:.3f}/{metrics.max_endpoint_shift:.3f} path={metrics.mean_path_length:.3f}/{metrics.max_path_length:.3f} "
            f"svr={metrics.svr_unguided:.3f}->{metrics.svr_guided:.3f} pir={metrics.pir_unguided:.3f}->{metrics.pir_guided:.3f} "
            f"local_svr={metrics.local_svr_unguided:.3f}->{metrics.local_svr_guided:.3f} local_dmin={metrics.local_min_distance_unguided:.3f}->{metrics.local_min_distance_guided:.3f} "
            f"pass={metrics.passes_visible_filter} fallback={metrics.used_fallback}"
        )
    print(f"representative_selection={args.select_by} samples={max(1, int(args.samples))} candidate_pool={len(candidates_payload)} saved_cases={len(cases)}")


if __name__ == "__main__":
    main()
