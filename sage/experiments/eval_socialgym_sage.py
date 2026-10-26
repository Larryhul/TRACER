from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sage.config import load_sage_config, sage_config_from_dict
from sage.data.socialgym_adapter import collate_socialgym_scene_batch, make_socialgym_dataset_from_config
from sage.evaluation.metrics import (
    best_of_navigation_metrics,
    multi_robot_metrics,
    navigation_metrics,
    role_safety_metrics,
    social_compliance_metrics,
)
from sage.experiments.checkpoint_compat import apply_checkpoint_compatibility
from sage.experiments.eval_real_sage import append_result_csv, override_guidance_from_args, seed_everything
from sage.experiments.eval_synthetic_sage import load_checkpoint
from sage.models.sage_model import SAGE


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAGE on SocialGym/SocialGym-style navigation episodes.")
    parser.add_argument("--config", default="configs/sage_socialgym.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--source", choices=["style", "trajectory", "rollout", "file", "offline"], default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--num-entities", type=int, default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible scene generation and stochastic sampling.")
    parser.add_argument("--unguided", action="store_true")
    parser.add_argument("--role-metrics", action="store_true")
    parser.add_argument("--disable-robot-robot-edges", action="store_true")
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
    parser.add_argument("--save-csv", default=None)
    args = parser.parse_args()
    seed_everything(args.seed)

    cfg_from_file, raw_cfg = load_sage_config(args.config)
    eval_cfg = raw_cfg.get("evaluation", {})
    socialgym_cfg = raw_cfg.get("dataset", {}).get("socialgym", {})
    ckpt = load_checkpoint(args.checkpoint, args.device)
    cfg = sage_config_from_dict(ckpt.get("config", None)) if "config" in ckpt else cfg_from_file
    cfg = apply_checkpoint_compatibility(cfg, ckpt.get("model", {}))

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
    if args.seed is not None:
        overrides["seed"] = args.seed
    dataset = make_socialgym_dataset_from_config(raw_cfg, split=args.split, **overrides)
    if args.limit_samples > 0 and hasattr(dataset, "num_samples"):
        dataset.num_samples = min(int(dataset.num_samples), int(args.limit_samples))
    elif args.limit_samples > 0 and hasattr(dataset, "samples"):
        dataset.samples = dataset.samples[: int(args.limit_samples)]

    batch_size = int(args.batch_size if args.batch_size is not None else eval_cfg.get("batch_size", 128))
    stride = int(args.stride if args.stride is not None else eval_cfg.get("stride", 10))
    samples = int(args.samples if args.samples is not None else eval_cfg.get("samples", 1))
    guided = bool(eval_cfg.get("guided", True)) and not args.unguided
    cfg.diffusion.horizon = int(getattr(dataset, "horizon", raw_cfg.get("dataset", {}).get("horizon", cfg.diffusion.horizon)))
    if args.disable_robot_robot_edges:
        cfg.encoder.use_robot_robot_edges = False
    encoder_variant = "heterogeneous" if cfg.encoder.use_role_embedding and cfg.encoder.use_typed_relations else "homogeneous"
    generation_mode = str(cfg.generation_mode).lower()
    relation_granularity = str(getattr(cfg.encoder, "relation_granularity", "control")).lower()
    history_encoder_type = str(getattr(cfg.encoder, "history_encoder_type", "gru")).lower()
    feature_fusion = str(getattr(cfg.encoder, "feature_fusion", "sum")).lower()
    guidance_update = str(getattr(cfg.diffusion, "guidance_update", "x0")).lower()
    use_graph_context_token = bool(getattr(cfg.diffusion, "use_graph_context_token", False))
    model_variant = f"{generation_mode}_{encoder_variant}_{relation_granularity}"
    comfort_sigma_long, comfort_sigma_lat = override_guidance_from_args(cfg, args)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_socialgym_scene_batch,
        pin_memory=args.device.startswith("cuda"),
    )
    model = SAGE(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    totals = {}
    n = 0
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(loader, ncols=90, desc=f"eval socialgym-{args.split}"):
            batch = batch.to(args.device)
            pred_samples = []
            diag_totals = {}
            role_diag_totals = {}
            for _ in range(max(samples, 1)):
                out = model.generate(batch, guided=guided, step_stride=stride)
                pred_samples.append(out["future_vel"])
                for key, value in out["diagnostics"].items():
                    diag_totals[key] = diag_totals.get(key, 0.0) + float(value)
                for key, value in social_compliance_metrics(
                    out["future_vel"],
                    batch,
                    comfort=model.guidance.config.comfort,
                    max_speed=model.guidance.config.max_speed,
                    personal_space_margin=args.pir_margin,
                    front_only=model.guidance.config.front_only,
                    eps=model.guidance.config.eps,
                    include_role_metrics=args.role_metrics,
                ).items():
                    diag_totals[key] = diag_totals.get(key, 0.0) + float(value)
                collision_margin = model.guidance._collision_margins(batch, out["future_vel"].dtype)
                for key, value in multi_robot_metrics(
                    out["future_vel"],
                    batch,
                    collision_margin=collision_margin,
                    goal_tolerance=model.guidance.config.goal_tolerance,
                    eps=model.guidance.config.eps,
                ).items():
                    diag_totals[key] = diag_totals.get(key, 0.0) + float(value)
                if args.role_metrics:
                    for key, value in role_safety_metrics(
                        out["future_vel"],
                        batch,
                        collision_margin=model.guidance.config.collision_margin,
                        eps=model.guidance.config.eps,
                    ).items():
                        role_diag_totals[key] = role_diag_totals.get(key, 0.0) + float(value)

            if len(pred_samples) == 1:
                metrics = navigation_metrics(
                    pred_samples[0],
                    batch,
                    include_role_metrics=args.role_metrics,
                    goal_tolerance=model.guidance.config.goal_tolerance,
                )
            else:
                metrics = best_of_navigation_metrics(
                    torch.stack(pred_samples, dim=0),
                    batch,
                    include_role_metrics=args.role_metrics,
                    goal_tolerance=model.guidance.config.goal_tolerance,
                )
            metrics.update({key: value / len(pred_samples) for key, value in diag_totals.items()})
            if args.role_metrics:
                metrics.update({key: value / len(pred_samples) for key, value in role_diag_totals.items()})
            bs = batch.batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * bs
            n += bs

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start_time
    summary = {key: totals[key] / max(n, 1) for key in sorted(totals)}
    summary["runtime_ms_per_scene"] = 1000.0 * elapsed_s / max(n, 1)
    source = overrides.get("source", socialgym_cfg.get("source", "style"))
    scene = overrides.get("scene", socialgym_cfg.get("scene", "mixed"))
    num_robots = getattr(dataset, "num_robots", args.num_robots)
    num_entities = getattr(dataset, "num_entities", args.num_entities)
    dataset_seed = getattr(dataset, "seed", "")
    print(
        f"checkpoint={args.checkpoint} dataset=socialgym source={source} split={args.split} guided={guided} samples={samples} stride={stride} "
        f"seed={args.seed} dataset_seed={dataset_seed} scene={scene} num_robots={num_robots} num_entities={num_entities} encoder_variant={encoder_variant} "
        f"generation_mode={generation_mode} model_variant={model_variant} relation_granularity={relation_granularity} "
        f"history_encoder={history_encoder_type} feature_fusion={feature_fusion} guidance_update={guidance_update} "
        f"graph_context_token={use_graph_context_token} use_robot_robot_edges={cfg.encoder.use_robot_robot_edges} "
        f"guidance_scale={cfg.guidance_scale} weights=({cfg.guidance.w_collision},{cfg.guidance.w_robot_robot_collision},{cfg.guidance.w_social},{cfg.guidance.w_kinematic},{cfg.guidance.w_goal})"
    )
    for key, value in summary.items():
        print(f"{key}={value:.6f}")

    if args.save_csv:
        row = {
            "dataset": "socialgym",
            "source": source,
            "split": args.split,
            "checkpoint": args.checkpoint,
            "config": args.config,
            "guided": guided,
            "samples": samples,
            "stride": stride,
            "history": raw_cfg.get("dataset", {}).get("history", 8),
            "horizon": cfg.diffusion.horizon,
            "max_entities": num_entities,
            "num_robot_proxies": num_robots,
            "batch_size": batch_size,
            "limit_samples": args.limit_samples,
            "eval_seed": args.seed,
            "train_seed": ckpt.get("seed", ""),
            "dataset_seed": dataset_seed,
            "role_strategy": "semantic",
            "role_metrics": args.role_metrics,
            "encoder_variant": encoder_variant,
            "generation_mode": generation_mode,
            "model_variant": model_variant,
            "history_encoder_type": history_encoder_type,
            "feature_fusion": feature_fusion,
            "relation_granularity": relation_granularity,
            "guidance_update": guidance_update,
            "use_graph_context_token": use_graph_context_token,
            "use_robot_robot_edges": cfg.encoder.use_robot_robot_edges,
            "guidance_scale": cfg.guidance_scale,
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
            "scene": scene,
            "num_robots": num_robots,
            "num_entities": num_entities,
        }
        row.update(summary)
        append_result_csv(args.save_csv, row)
        print(f"saved_csv={args.save_csv}")


if __name__ == "__main__":
    main()
