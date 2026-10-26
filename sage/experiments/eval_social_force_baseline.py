from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sage.config import load_sage_config, sage_config_from_dict
from sage.data.socialgym_adapter import collate_socialgym_scene_batch, make_socialgym_dataset_from_config
from sage.data.synthetic_navigation import collate_synthetic_scene_batch, make_synthetic_dataset_from_config
from sage.evaluation.metrics import (
    multi_robot_metrics,
    navigation_metrics,
    role_safety_metrics,
    social_compliance_metrics,
)
from sage.experiments.eval_real_sage import append_result_csv, override_guidance_from_args, seed_everything
from sage.guidance.safety_social import SafetySocialGuidance
from sage.models.social_force import SocialForceConfig, social_force_baseline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a deterministic Social Force baseline on Synthetic or SocialGym-style SAGE scenes.")
    parser.add_argument("--benchmark", choices=["synthetic", "socialgym"], default="synthetic")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--source", choices=["style", "trajectory", "rollout", "file", "offline"], default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--num-entities", type=int, default=None)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible scene generation.")
    parser.add_argument("--role-metrics", action="store_true")
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
    parser.add_argument("--pir-margin", type=float, default=1.2)
    parser.add_argument("--desired-speed", type=float, default=1.2)
    parser.add_argument("--relaxation-time", type=float, default=0.5)
    parser.add_argument("--repulsion-gain", type=float, default=1.2)
    parser.add_argument("--robot-robot-gain", type=float, default=1.6)
    parser.add_argument("--repulsion-sigma", type=float, default=0.8)
    parser.add_argument("--interaction-radius", type=float, default=4.0)
    parser.add_argument("--save-csv", default=None)
    return parser


def _load_dataset(args: argparse.Namespace, raw_cfg: dict):
    overrides = {}
    if args.num_robots is not None:
        overrides["num_robots"] = args.num_robots
    if args.num_entities is not None:
        overrides["num_entities"] = args.num_entities
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.benchmark == "synthetic":
        if args.layout is not None:
            overrides["layout"] = args.layout
        dataset = make_synthetic_dataset_from_config(raw_cfg, split=args.split, **overrides)
        collate_fn = collate_synthetic_scene_batch
        source = "controlled"
        scene = getattr(dataset, "layout", args.layout or "mixed")
    else:
        if args.source is not None:
            overrides["source"] = args.source
        if args.data_path is not None:
            overrides["data_path"] = args.data_path
        if args.scene is not None:
            overrides["scene"] = args.scene
        dataset = make_socialgym_dataset_from_config(raw_cfg, split=args.split, **overrides)
        collate_fn = collate_socialgym_scene_batch
        socialgym_cfg = raw_cfg.get("dataset", {}).get("socialgym", {})
        source = overrides.get("source", socialgym_cfg.get("source", "style"))
        scene = overrides.get("scene", socialgym_cfg.get("scene", "mixed"))
    if args.limit_samples > 0 and hasattr(dataset, "num_samples"):
        dataset.num_samples = min(int(dataset.num_samples), int(args.limit_samples))
    elif args.limit_samples > 0 and hasattr(dataset, "samples"):
        dataset.samples = dataset.samples[: int(args.limit_samples)]
    return dataset, collate_fn, source, scene


def main() -> None:
    args = _parser().parse_args()
    seed_everything(args.seed)
    config_path = args.config or ("configs/sage_socialgym.yaml" if args.benchmark == "socialgym" else "configs/sage_synthetic.yaml")
    cfg_from_file, raw_cfg = load_sage_config(config_path)
    cfg = sage_config_from_dict({
        "encoder": cfg_from_file.encoder.__dict__,
        "diffusion": cfg_from_file.diffusion.__dict__,
        "guidance": cfg_from_file.guidance.__dict__,
        "guidance_scale": cfg_from_file.guidance_scale,
        "generation_mode": cfg_from_file.generation_mode,
    })
    comfort_sigma_long, comfort_sigma_lat = override_guidance_from_args(cfg, args)
    guidance = SafetySocialGuidance(cfg.guidance)

    dataset, collate_fn, source, scene = _load_dataset(args, raw_cfg)
    eval_cfg = raw_cfg.get("evaluation", {})
    batch_size = int(args.batch_size if args.batch_size is not None else eval_cfg.get("batch_size", 128))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device.startswith("cuda"),
    )
    sf_cfg = SocialForceConfig(
        desired_speed=args.desired_speed,
        max_speed=float(cfg.guidance.max_speed),
        relaxation_time=args.relaxation_time,
        repulsion_gain=args.repulsion_gain,
        robot_robot_gain=args.robot_robot_gain,
        repulsion_sigma=args.repulsion_sigma,
        interaction_radius=args.interaction_radius,
    )

    totals: dict[str, float] = {}
    n = 0
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for batch in tqdm(loader, ncols=90, desc=f"eval {args.benchmark}-social_force"):
        batch = batch.to(args.device)
        pred_vel = social_force_baseline(batch, sf_cfg)
        metrics = navigation_metrics(
            pred_vel,
            batch,
            include_role_metrics=args.role_metrics,
            goal_tolerance=cfg.guidance.goal_tolerance,
        )
        metrics.update(guidance.diagnostics(pred_vel, batch))
        metrics.update(
            social_compliance_metrics(
                pred_vel,
                batch,
                comfort=guidance.config.comfort,
                max_speed=guidance.config.max_speed,
                personal_space_margin=args.pir_margin,
                front_only=guidance.config.front_only,
                eps=guidance.config.eps,
                include_role_metrics=args.role_metrics,
            )
        )
        collision_margin = guidance._collision_margins(batch, pred_vel.dtype)
        metrics.update(
            multi_robot_metrics(
                pred_vel,
                batch,
                collision_margin=collision_margin,
                goal_tolerance=guidance.config.goal_tolerance,
                eps=guidance.config.eps,
            )
        )
        if args.role_metrics:
            metrics.update(
                role_safety_metrics(
                    pred_vel,
                    batch,
                    collision_margin=guidance.config.collision_margin,
                    eps=guidance.config.eps,
                )
            )
        bs = batch.batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * bs
        n += bs

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start_time
    summary = {key: totals[key] / max(n, 1) for key in sorted(totals)}
    summary["runtime_ms_per_scene"] = 1000.0 * elapsed_s / max(n, 1)
    num_robots = getattr(dataset, "num_robots", args.num_robots)
    num_entities = getattr(dataset, "num_entities", args.num_entities)
    horizon = getattr(dataset, "horizon", raw_cfg.get("dataset", {}).get("horizon", cfg.diffusion.horizon))
    history = getattr(dataset, "history", raw_cfg.get("dataset", {}).get("history", 8))
    dataset_seed = getattr(dataset, "seed", "")
    print(
        f"baseline=social_force dataset={args.benchmark} source={source} split={args.split} "
        f"seed={args.seed} dataset_seed={dataset_seed} scene={scene} num_robots={num_robots} num_entities={num_entities} history={history} horizon={horizon} "
        f"max_speed={sf_cfg.max_speed} desired_speed={sf_cfg.desired_speed} repulsion_gain={sf_cfg.repulsion_gain} "
        f"robot_robot_gain={sf_cfg.robot_robot_gain} interaction_radius={sf_cfg.interaction_radius}"
    )
    for key, value in summary.items():
        print(f"{key}={value:.6f}")

    if args.save_csv:
        row = {
            "dataset": args.benchmark,
            "source": source,
            "split": args.split,
            "checkpoint": "none",
            "config": config_path,
            "guided": False,
            "samples": 1,
            "stride": 1,
            "history": history,
            "horizon": horizon,
            "max_entities": num_entities,
            "num_robot_proxies": num_robots,
            "batch_size": batch_size,
            "limit_samples": args.limit_samples,
            "eval_seed": args.seed,
            "train_seed": "",
            "dataset_seed": dataset_seed,
            "role_strategy": "semantic",
            "role_metrics": args.role_metrics,
            "encoder_variant": "none",
            "generation_mode": "social_force",
            "model_variant": "social_force",
            "history_encoder_type": "none",
            "feature_fusion": "none",
            "relation_granularity": "none",
            "guidance_update": "none",
            "use_graph_context_token": False,
            "use_robot_robot_edges": True,
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
            "device": args.device,
            "baseline": "social_force",
            "scene": scene,
            "layout": scene,
            "num_robots": num_robots,
            "num_entities": num_entities,
            "desired_speed": sf_cfg.desired_speed,
            "relaxation_time": sf_cfg.relaxation_time,
            "repulsion_gain": sf_cfg.repulsion_gain,
            "robot_robot_gain": sf_cfg.robot_robot_gain,
            "repulsion_sigma": sf_cfg.repulsion_sigma,
            "interaction_radius": sf_cfg.interaction_radius,
        }
        row.update(summary)
        append_result_csv(args.save_csv, row)
        print(f"saved_csv={args.save_csv}")


if __name__ == "__main__":
    main()
