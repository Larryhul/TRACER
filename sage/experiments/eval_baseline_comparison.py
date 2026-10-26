from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sage.config import load_sage_config
from sage.data.mid_adapter import ROLE_STRATEGIES
from sage.data.mid_real_dataset import MIDRealSceneDataset, collate_sage_scene_batch
from sage.data.perception import perception_kwargs_from_config
from sage.data.sdd_adapter import SDDOfficialSceneDataset, SDDRawSceneDataset, collate_sdd_scene_batch
from sage.data.socialgym_adapter import collate_socialgym_scene_batch, make_socialgym_dataset_from_config
from sage.data.synthetic_navigation import collate_synthetic_scene_batch, make_synthetic_dataset_from_config
from sage.data.heterogeneous_scene import HeterogeneousSceneBatch
from sage.evaluation.metrics import (
    multi_robot_metrics,
    navigation_metrics,
    role_safety_metrics,
    social_compliance_metrics,
)
from sage.experiments.eval_real_sage import append_result_csv, apply_comfort_profile
from sage.guidance.safety_social import SafetySocialGuidance


REAL_DATASETS = {"eth", "hotel", "univ", "zara1", "zara2"}
SDD_ROLE_STRATEGIES = ("semantic_or_speed", *ROLE_STRATEGIES)


def constant_velocity_baseline(batch: HeterogeneousSceneBatch) -> torch.Tensor:
    """Repeat the last observed velocity over the prediction horizon."""

    xy = batch.history[..., :2]
    if xy.shape[2] >= 2:
        vel = (xy[:, :, -1] - xy[:, :, -2]) / float(batch.dt)
    else:
        vel = torch.zeros_like(batch.current_pos)
    vel = vel * batch.agent_mask.to(vel.dtype)[..., None]
    return vel[:, :, None, :].expand(-1, -1, batch.horizon, -1).contiguous()


def constant_acceleration_baseline(batch: HeterogeneousSceneBatch) -> torch.Tensor:
    """Extrapolate the latest finite-difference acceleration."""

    xy = batch.history[..., :2]
    if xy.shape[2] < 3:
        return constant_velocity_baseline(batch)
    latest_vel = (xy[:, :, -1] - xy[:, :, -2]) / float(batch.dt)
    previous_vel = (xy[:, :, -2] - xy[:, :, -3]) / float(batch.dt)
    acceleration = (latest_vel - previous_vel) / float(batch.dt)
    steps = torch.arange(
        1,
        batch.horizon + 1,
        device=xy.device,
        dtype=xy.dtype,
    ).view(1, 1, -1, 1)
    pred = latest_vel[:, :, None, :] + acceleration[:, :, None, :] * (steps * float(batch.dt))
    return pred * batch.agent_mask[:, :, None, None].to(pred.dtype)


def direct_goal_baseline(batch: HeterogeneousSceneBatch) -> torch.Tensor:
    """Send goal-conditioned robots straight to their waypoint; extrapolate other agents."""

    pred = constant_velocity_baseline(batch)
    if batch.goals is None:
        return pred
    goal_mask = batch.effective_goal_mask()
    goal_vel = (batch.goals - batch.current_pos) / (float(batch.dt) * batch.horizon)
    goal_pred = goal_vel[:, :, None, :].expand(-1, -1, batch.horizon, -1)
    return torch.where(goal_mask[:, :, None, None], goal_pred, pred)


BASELINES: dict[str, Callable[[HeterogeneousSceneBatch], torch.Tensor]] = {
    "constant_acceleration": constant_acceleration_baseline,
    "constant_velocity": constant_velocity_baseline,
    "direct_goal": direct_goal_baseline,
}


def override_metric_config(cfg, args: argparse.Namespace) -> tuple[object, object]:
    if args.goal_tolerance is not None:
        cfg.guidance.goal_tolerance = float(args.goal_tolerance)
    if args.collision_margin is not None:
        cfg.guidance.collision_margin = float(args.collision_margin)
    if args.max_speed is not None:
        cfg.guidance.max_speed = float(args.max_speed)
    if args.use_geometry_margin:
        cfg.guidance.use_geometry_margin = True
    if args.disable_geometry_margin:
        cfg.guidance.use_geometry_margin = False
    return apply_comfort_profile(
        cfg.guidance,
        args.comfort_profile,
        args.shared_sigma_long,
        args.shared_sigma_lat,
    )


def build_real_dataset(args: argparse.Namespace, raw_cfg: dict, cfg):
    dataset_cfg = raw_cfg.get("dataset", {})
    role_cfg = dataset_cfg.get("pseudo_roles", {})
    history = int(dataset_cfg.get("history", 8))
    horizon = int(dataset_cfg.get("horizon", cfg.diffusion.horizon))
    max_entities = int(args.max_entities if args.max_entities is not None else dataset_cfg.get("max_entities", 8))
    num_robot_proxies = int(
        args.num_robot_proxies if args.num_robot_proxies is not None else dataset_cfg.get("num_robot_proxies", 1)
    )
    role_strategy = args.role_strategy or role_cfg.get("strategy", dataset_cfg.get("role_strategy", "speed_threshold"))
    data_path = Path(args.data_dir) / f"{args.dataset}_{args.split}.pkl"
    dataset = MIDRealSceneDataset(
        data_path,
        history=history,
        horizon=horizon,
        max_entities=max_entities,
        num_robot_proxies=num_robot_proxies,
        role_strategy=role_strategy,
        role_slow_speed=float(role_cfg.get("slow_speed", 0.15)),
        role_fast_speed=float(role_cfg.get("fast_speed", 1.8)),
        role_slow_quantile=float(role_cfg.get("slow_quantile", 0.33)),
        role_fast_quantile=float(role_cfg.get("fast_quantile", 0.67)),
        **perception_kwargs_from_config(dataset_cfg),
    )
    meta = {
        "source": "mid_processed",
        "history": history,
        "horizon": horizon,
        "max_entities": max_entities,
        "num_robot_proxies": num_robot_proxies,
        "role_strategy": role_strategy,
        "perception_mode": dataset_cfg.get("perception", {}).get("mode", "nearest"),
        "role_source": "speed_proxy",
    }
    return dataset, collate_sage_scene_batch, meta


def build_sdd_dataset(args: argparse.Namespace, raw_cfg: dict, cfg):
    dataset_cfg = raw_cfg.get("dataset", {})
    sdd_cfg = dataset_cfg.get("sdd", {})
    role_cfg = dataset_cfg.get("pseudo_roles", {})
    history = int(dataset_cfg.get("history", 8))
    horizon = int(dataset_cfg.get("horizon", cfg.diffusion.horizon))
    max_entities = int(args.max_entities if args.max_entities is not None else dataset_cfg.get("max_entities", 12))
    num_robot_proxies = int(
        args.num_robot_proxies if args.num_robot_proxies is not None else dataset_cfg.get("num_robot_proxies", 1)
    )
    role_strategy = args.role_strategy or sdd_cfg.get("role_strategy", "semantic_or_speed")
    source = args.source or sdd_cfg.get("source", "trajnet")
    sdd_root = args.sdd_root or sdd_cfg.get("sdd_root", None)
    coordinate = args.coordinate or sdd_cfg.get("coordinate", "center")
    frame_stride = int(args.frame_stride if args.frame_stride is not None else sdd_cfg.get("frame_stride", 12))
    coordinate_scale = float(
        args.coordinate_scale if args.coordinate_scale is not None else sdd_cfg.get("coordinate_scale", 50.0)
    )
    label_map_path = args.label_map or sdd_cfg.get("label_map", None)
    data_path = Path(args.data_dir) / f"{args.split}_trajnet.pkl"
    dataset_kwargs = dict(
        history=history,
        horizon=horizon,
        num_robot_proxies=num_robot_proxies,
        max_entities=max_entities,
        dt=float(dataset_cfg.get("dt", 0.4)),
        frame_stride=frame_stride,
        coordinate_scale=coordinate_scale,
        center_scene=bool(sdd_cfg.get("center_scene", True)),
        role_strategy=role_strategy,
        role_slow_speed=float(role_cfg.get("slow_speed", 0.15)),
        role_fast_speed=float(role_cfg.get("fast_speed", 1.8)),
        role_slow_quantile=float(role_cfg.get("slow_quantile", 0.33)),
        role_fast_quantile=float(role_cfg.get("fast_quantile", 0.67)),
        limit_scenes=args.limit_scenes,
        **perception_kwargs_from_config(dataset_cfg),
    )
    if source == "official":
        if not sdd_root:
            raise ValueError("--sdd-root is required when --dataset sdd --source official")
        dataset = SDDOfficialSceneDataset(
            sdd_root,
            split_reference_pkl=data_path,
            coordinate=coordinate,
            **dataset_kwargs,
        )
    else:
        dataset = SDDRawSceneDataset(data_path, label_map_path=label_map_path, **dataset_kwargs)
    meta = {
        "source": source,
        "sdd_root": sdd_root or "",
        "history": history,
        "horizon": horizon,
        "max_entities": max_entities,
        "num_robot_proxies": num_robot_proxies,
        "role_strategy": role_strategy,
        "role_source": getattr(dataset, "role_source", ""),
        "label_column": getattr(dataset, "label_column", ""),
        "label_map": label_map_path or "",
        "perception_mode": dataset_cfg.get("perception", {}).get("mode", "nearest"),
        "frame_stride": frame_stride,
        "coordinate_scale": coordinate_scale,
    }
    return dataset, collate_sdd_scene_batch, meta


def build_socialgym_dataset(args: argparse.Namespace, raw_cfg: dict, cfg):
    dataset_cfg = raw_cfg.get("dataset", {})
    socialgym_cfg = dataset_cfg.get("socialgym", {})
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
    horizon = int(getattr(dataset, "horizon", dataset_cfg.get("horizon", cfg.diffusion.horizon)))
    meta = {
        "source": overrides.get("source", socialgym_cfg.get("source", "style")),
        "history": dataset_cfg.get("history", 8),
        "horizon": horizon,
        "max_entities": getattr(dataset, "num_entities", args.num_entities),
        "num_robot_proxies": getattr(dataset, "num_robots", args.num_robots),
        "role_strategy": "semantic",
        "role_source": "socialgym",
        "scene": overrides.get("scene", socialgym_cfg.get("scene", "mixed")),
        "num_robots": getattr(dataset, "num_robots", args.num_robots),
        "num_entities": getattr(dataset, "num_entities", args.num_entities),
    }
    return dataset, collate_socialgym_scene_batch, meta


def build_synthetic_dataset(args: argparse.Namespace, raw_cfg: dict, cfg):
    dataset_cfg = raw_cfg.get("dataset", {})
    overrides = {}
    if args.num_robots is not None:
        overrides["num_robots"] = args.num_robots
    if args.num_entities is not None:
        overrides["num_entities"] = args.num_entities
    if getattr(args, "layout", None) is not None:
        overrides["layout"] = args.layout
    if getattr(args, "dataset_seed", None) is not None:
        overrides["seed"] = args.dataset_seed
    elif getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    dataset = make_synthetic_dataset_from_config(raw_cfg, split=args.split, **overrides)
    meta = {
        "source": "controlled_synthetic",
        "history": dataset.history,
        "horizon": dataset.horizon,
        "max_entities": dataset.num_entities,
        "num_robot_proxies": dataset.num_robots,
        "role_strategy": "semantic",
        "role_source": "synthetic_ground_truth",
        "layout": dataset.layout,
        "scene": dataset.layout,
        "num_robots": dataset.num_robots,
        "num_entities": dataset.num_entities,
        "dataset_seed": dataset.seed,
        "dt": float(dataset_cfg.get("dt", dataset.dt)),
    }
    return dataset, collate_synthetic_scene_batch, meta


def build_dataset(args: argparse.Namespace, raw_cfg: dict, cfg):
    if args.dataset in REAL_DATASETS:
        return build_real_dataset(args, raw_cfg, cfg)
    if args.dataset == "sdd":
        return build_sdd_dataset(args, raw_cfg, cfg)
    if args.dataset == "socialgym":
        return build_socialgym_dataset(args, raw_cfg, cfg)
    if args.dataset == "synthetic":
        return build_synthetic_dataset(args, raw_cfg, cfg)
    raise ValueError(f"unsupported dataset={args.dataset}")


def evaluate(args: argparse.Namespace) -> None:
    cfg, raw_cfg = load_sage_config(args.config)
    eval_cfg = raw_cfg.get("evaluation", {})
    cfg.diffusion.horizon = int(raw_cfg.get("dataset", {}).get("horizon", cfg.diffusion.horizon))
    comfort_sigma_long, comfort_sigma_lat = override_metric_config(cfg, args)

    dataset, collate_fn, meta = build_dataset(args, raw_cfg, cfg)
    if args.limit_samples > 0:
        if hasattr(dataset, "indices"):
            dataset.indices = dataset.indices[: args.limit_samples]
        elif hasattr(dataset, "num_samples"):
            dataset.num_samples = min(int(dataset.num_samples), int(args.limit_samples))
        elif hasattr(dataset, "samples"):
            dataset.samples = dataset.samples[: args.limit_samples]

    batch_size = int(args.batch_size if args.batch_size is not None else eval_cfg.get("batch_size", 128))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device.startswith("cuda"),
    )
    baseline_fn = BASELINES[args.baseline]
    guidance_helper = SafetySocialGuidance(cfg.guidance)

    totals: dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for batch in tqdm(loader, ncols=90, desc=f"eval {args.dataset}-{args.baseline}"):
            batch = batch.to(args.device)
            pred_vel = baseline_fn(batch)
            metrics = navigation_metrics(
                pred_vel,
                batch,
                include_role_metrics=args.role_metrics,
                goal_tolerance=cfg.guidance.goal_tolerance,
            )
            metrics.update(guidance_helper.diagnostics(pred_vel, batch))
            metrics.update(
                social_compliance_metrics(
                    pred_vel,
                    batch,
                    comfort=cfg.guidance.comfort,
                    max_speed=cfg.guidance.max_speed,
                    personal_space_margin=args.pir_margin,
                    front_only=cfg.guidance.front_only,
                    eps=cfg.guidance.eps,
                    include_role_metrics=args.role_metrics,
                )
            )
            collision_margin = guidance_helper._collision_margins(batch, pred_vel.dtype)
            metrics.update(
                multi_robot_metrics(
                    pred_vel,
                    batch,
                    collision_margin=collision_margin,
                    goal_tolerance=cfg.guidance.goal_tolerance,
                    eps=cfg.guidance.eps,
                )
            )
            if args.role_metrics:
                metrics.update(
                    role_safety_metrics(
                        pred_vel,
                        batch,
                        collision_margin=cfg.guidance.collision_margin,
                        eps=cfg.guidance.eps,
                    )
                )
            bs = batch.batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * bs
            n += bs

    summary = {key: totals[key] / max(n, 1) for key in sorted(totals)}
    print(
        f"baseline={args.baseline} dataset={args.dataset} split={args.split} samples=1 "
        f"history={meta.get('history')} horizon={meta.get('horizon')} max_entities={meta.get('max_entities')} "
        f"num_robot_proxies={meta.get('num_robot_proxies')} role_strategy={meta.get('role_strategy')} "
        f"role_source={meta.get('role_source', '')} comfort_profile={args.comfort_profile} "
        f"use_geometry_margin={cfg.guidance.use_geometry_margin} goal_tolerance={cfg.guidance.goal_tolerance}"
    )
    for key, value in summary.items():
        print(f"{key}={value:.6f}")

    if args.save_csv:
        row = {
            "dataset": args.dataset,
            "source": meta.get("source", ""),
            "split": args.split,
            "checkpoint": "",
            "config": args.config,
            "guided": False,
            "samples": 1,
            "stride": 0,
            "history": meta.get("history", ""),
            "horizon": meta.get("horizon", ""),
            "max_entities": meta.get("max_entities", ""),
            "num_robot_proxies": meta.get("num_robot_proxies", ""),
            "batch_size": batch_size,
            "limit_samples": args.limit_samples,
            "role_strategy": meta.get("role_strategy", ""),
            "role_source": meta.get("role_source", ""),
            "label_column": meta.get("label_column", ""),
            "label_map": meta.get("label_map", ""),
            "role_metrics": args.role_metrics,
            "encoder_variant": "none",
            "generation_mode": "deterministic",
            "model_variant": args.baseline,
            "baseline": args.baseline,
            "history_encoder_type": "none",
            "feature_fusion": "none",
            "relation_granularity": "none",
            "guidance_update": "none",
            "use_graph_context_token": False,
            "use_robot_robot_edges": False,
            "guidance_scale": 0.0,
            "w_collision": cfg.guidance.w_collision,
            "w_robot_robot_collision": cfg.guidance.w_robot_robot_collision,
            "w_social": cfg.guidance.w_social,
            "w_kinematic": cfg.guidance.w_kinematic,
            "w_goal": cfg.guidance.w_goal,
            "goal_tolerance": cfg.guidance.goal_tolerance,
            "comfort_profile": args.comfort_profile,
            "comfort_sigma_long": comfort_sigma_long,
            "comfort_sigma_lat": comfort_sigma_lat,
            "collision_margin": cfg.guidance.collision_margin,
            "max_speed": cfg.guidance.max_speed,
            "pir_margin": args.pir_margin,
            "use_geometry_margin": cfg.guidance.use_geometry_margin,
            "frame_stride": meta.get("frame_stride", ""),
            "coordinate_scale": meta.get("coordinate_scale", ""),
            "device": args.device,
            "scene": meta.get("scene", ""),
            "num_robots": meta.get("num_robots", meta.get("num_robot_proxies", "")),
            "num_entities": meta.get("num_entities", meta.get("max_entities", "")),
            "dataset_seed": meta.get("dataset_seed", ""),
        }
        row.update(summary)
        append_result_csv(args.save_csv, row)
        print(f"saved_csv={args.save_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lightweight baselines under the SAGE protocol.")
    parser.add_argument("--baseline", choices=sorted(BASELINES), default="constant_velocity")
    parser.add_argument("--config", default="configs/sage_sdd.yaml")
    parser.add_argument("--dataset", choices=sorted(REAL_DATASETS | {"sdd", "socialgym", "synthetic"}), default="sdd")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--sdd-root", default=None)
    parser.add_argument("--coordinate", choices=["center", "bottom_center", "top_left"], default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--coordinate-scale", type=float, default=None)
    parser.add_argument("--label-map", default=None)
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--num-entities", type=int, default=None)
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument("--num-robot-proxies", type=int, default=None)
    parser.add_argument("--role-strategy", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--role-metrics", action="store_true")
    parser.add_argument("--goal-tolerance", type=float, default=None)
    parser.add_argument("--comfort-profile", choices=["role", "shared", "isotropic"], default="role")
    parser.add_argument("--shared-sigma-long", type=float, default=None)
    parser.add_argument("--shared-sigma-lat", type=float, default=None)
    parser.add_argument("--collision-margin", type=float, default=None)
    parser.add_argument("--max-speed", type=float, default=None)
    parser.add_argument("--pir-margin", type=float, default=1.2)
    parser.add_argument("--use-geometry-margin", action="store_true")
    parser.add_argument("--disable-geometry-margin", action="store_true")
    parser.add_argument("--save-csv", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Synthetic dataset seed.")
    parser.add_argument("--dataset-seed", type=int, default=None, help="Explicit synthetic dataset seed.")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = "raw_data/stanford" if args.dataset == "sdd" else "models/processed_data_noise"
    if args.dataset == "socialgym" and args.data_dir == "models/processed_data_noise":
        args.data_dir = ""
    if args.dataset == "sdd" and args.role_strategy is not None and args.role_strategy not in SDD_ROLE_STRATEGIES:
        raise ValueError(f"unsupported SDD role strategy: {args.role_strategy}")
    if args.dataset in REAL_DATASETS and args.role_strategy is not None and args.role_strategy not in ROLE_STRATEGIES:
        raise ValueError(f"unsupported real-data role strategy: {args.role_strategy}")
    evaluate(args)


if __name__ == "__main__":
    main()
