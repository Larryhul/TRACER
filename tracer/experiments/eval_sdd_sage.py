from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tracer.config import load_sage_config, sage_config_from_dict
from tracer.data.mid_adapter import ROLE_STRATEGIES
from tracer.data.perception import perception_kwargs_from_config
from tracer.data.sdd_adapter import SDDOfficialSceneDataset, SDDRawSceneDataset, collate_sdd_scene_batch
from tracer.evaluation.metrics import (
    best_of_navigation_metrics,
    multi_robot_metrics,
    navigation_metrics,
    role_safety_metrics,
    social_compliance_metrics,
)
from tracer.experiments.eval_real_sage import append_result_csv, apply_comfort_profile, seed_everything
from tracer.experiments.checkpoint_compat import apply_checkpoint_compatibility
from tracer.models.sage_model import SAGE


def load_checkpoint(path: str | Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

SDD_ROLE_STRATEGIES = ("semantic_or_speed", *ROLE_STRATEGIES)


def override_guidance_from_args(cfg, args: argparse.Namespace) -> tuple[object, object]:
    if args.guidance_scale is not None:
        cfg.guidance_scale = float(args.guidance_scale)
    if args.w_collision is not None:
        cfg.guidance.w_collision = float(args.w_collision)
    if args.w_social is not None:
        cfg.guidance.w_social = float(args.w_social)
    if getattr(args, "w_robot_robot_collision", None) is not None:
        cfg.guidance.w_robot_robot_collision = float(args.w_robot_robot_collision)
    if args.w_kinematic is not None:
        cfg.guidance.w_kinematic = float(args.w_kinematic)
    if args.w_goal is not None:
        cfg.guidance.w_goal = float(args.w_goal)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAGE on Stanford Drone Dataset trajectories.")
    parser.add_argument("--config", default="configs/sage_sdd.yaml")
    parser.add_argument("--data-dir", default="raw_data/stanford")
    parser.add_argument("--source", choices=["trajnet", "official"], default=None)
    parser.add_argument("--sdd-root", default=None, help="Official SDD root containing annotations/*/video*/annotations.txt.")
    parser.add_argument("--coordinate", choices=["center", "bottom_center", "top_left"], default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument("--num-robot-proxies", type=int, default=None)
    parser.add_argument("--disable-robot-robot-edges", action="store_true")
    parser.add_argument("--role-strategy", choices=SDD_ROLE_STRATEGIES, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--coordinate-scale", type=float, default=None)
    parser.add_argument("--label-map", default=None, help="Optional CSV with sceneId,trackId,label to restore SDD semantic labels.")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--unguided", action="store_true")
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
    parser.add_argument("--use-geometry-margin", action="store_true", help="Use SDD length/width metadata in collision margins.")
    parser.add_argument("--disable-geometry-margin", action="store_true", help="Disable geometry margins even if config enables them.")
    parser.add_argument("--save-csv", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible stochastic sampling.")
    args = parser.parse_args()

    seed_everything(args.seed)

    cfg_from_file, raw_cfg = load_sage_config(args.config)
    dataset_cfg = raw_cfg.get("dataset", {})
    sdd_cfg = dataset_cfg.get("sdd", {})
    eval_cfg = raw_cfg.get("evaluation", {})

    ckpt = load_checkpoint(args.checkpoint, args.device)
    cfg = sage_config_from_dict(ckpt.get("config", None)) if "config" in ckpt else cfg_from_file
    cfg = apply_checkpoint_compatibility(cfg, ckpt.get("model", {}))

    history = int(dataset_cfg.get("history", 8))
    horizon = int(dataset_cfg.get("horizon", cfg.diffusion.horizon))
    max_entities = int(args.max_entities if args.max_entities is not None else dataset_cfg.get("max_entities", 12))
    role_strategy = args.role_strategy or sdd_cfg.get("role_strategy", "semantic_or_speed")
    source = args.source or sdd_cfg.get("source", "trajnet")
    sdd_root = args.sdd_root or sdd_cfg.get("sdd_root", None)
    coordinate = args.coordinate or sdd_cfg.get("coordinate", "center")
    frame_stride = int(args.frame_stride if args.frame_stride is not None else sdd_cfg.get("frame_stride", 12))
    coordinate_scale = float(args.coordinate_scale if args.coordinate_scale is not None else sdd_cfg.get("coordinate_scale", 50.0))
    label_map_path = args.label_map or sdd_cfg.get("label_map", None)
    role_cfg = dataset_cfg.get("pseudo_roles", {})
    num_robot_proxies_cfg = int(args.num_robot_proxies if args.num_robot_proxies is not None else dataset_cfg.get("num_robot_proxies", 1))
    perception_mode_cfg = dataset_cfg.get("perception", {}).get("mode", "nearest")
    perception_kwargs = perception_kwargs_from_config(dataset_cfg)
    batch_size = int(args.batch_size if args.batch_size is not None else eval_cfg.get("batch_size", 128))
    stride = int(args.stride if args.stride is not None else eval_cfg.get("stride", 10))
    samples = int(args.samples if args.samples is not None else eval_cfg.get("samples", 1))
    guided = bool(eval_cfg.get("guided", True)) and not args.unguided
    cfg.diffusion.horizon = horizon
    if args.disable_robot_robot_edges:
        cfg.encoder.use_robot_robot_edges = False
    encoder_variant = "heterogeneous" if cfg.encoder.use_role_embedding and cfg.encoder.use_typed_relations else "homogeneous"
    relation_granularity = str(getattr(cfg.encoder, "relation_granularity", "control")).lower()
    history_encoder_type = str(getattr(cfg.encoder, "history_encoder_type", "gru")).lower()
    feature_fusion = str(getattr(cfg.encoder, "feature_fusion", "sum")).lower()
    guidance_update = str(getattr(cfg.diffusion, "guidance_update", "x0")).lower()
    use_graph_context_token = bool(getattr(cfg.diffusion, "use_graph_context_token", False))
    generation_mode = str(cfg.generation_mode).lower()
    model_variant = f"{generation_mode}_{encoder_variant}_{relation_granularity}"
    comfort_sigma_long, comfort_sigma_lat = override_guidance_from_args(cfg, args)

    data_path = Path(args.data_dir) / f"{args.split}_trajnet.pkl"
    dataset_kwargs = dict(
        history=history,
        horizon=horizon,
        num_robot_proxies=num_robot_proxies_cfg,
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
        **perception_kwargs,
    )
    if source == "official":
        if not sdd_root:
            raise ValueError("--sdd-root is required when --source official")
        dataset = SDDOfficialSceneDataset(
            sdd_root,
            split_reference_pkl=data_path,
            coordinate=coordinate,
            **dataset_kwargs,
        )
    else:
        dataset = SDDRawSceneDataset(
            data_path,
            label_map_path=label_map_path,
            **dataset_kwargs,
        )
    if args.limit_samples > 0:
        dataset.indices = dataset.indices[: args.limit_samples]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sdd_scene_batch,
        pin_memory=args.device.startswith("cuda"),
    )

    model = SAGE(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    totals = {}
    n = 0
    with torch.no_grad():
        for batch in tqdm(loader, ncols=90, desc=f"eval sdd-{args.split}"):
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

    summary = {key: totals[key] / max(n, 1) for key in sorted(totals)}
    print(
        f"checkpoint={args.checkpoint} dataset=sdd source={source} split={args.split} guided={guided} samples={samples} "
        f"stride={stride} history={history} horizon={horizon} max_entities={max_entities} "
        f"role_strategy={role_strategy} role_source={dataset.role_source} guidance_scale={cfg.guidance_scale} "
        f"comfort_profile={args.comfort_profile} encoder_variant={encoder_variant} generation_mode={generation_mode} model_variant={model_variant} relation_granularity={relation_granularity} history_encoder={history_encoder_type} feature_fusion={feature_fusion} guidance_update={guidance_update} graph_context_token={use_graph_context_token} relation_attention={cfg.encoder.relation_attention_type} agent_attention={cfg.diffusion.use_agent_attention} use_robot_robot_edges={cfg.encoder.use_robot_robot_edges} num_robot_proxies={num_robot_proxies_cfg} perception_mode={perception_mode_cfg} use_geometry_margin={cfg.guidance.use_geometry_margin} "
        f"weights=({cfg.guidance.w_collision},{cfg.guidance.w_robot_robot_collision},{cfg.guidance.w_social},{cfg.guidance.w_kinematic},{cfg.guidance.w_goal}) "
        f"goal_tolerance={cfg.guidance.goal_tolerance}"
    )
    for key, value in summary.items():
        print(f"{key}={value:.6f}")

    if args.save_csv:
        row = {
            "dataset": "sdd",
            "source": source,
            "sdd_root": sdd_root,
            "split": args.split,
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "config": args.config,
            "guided": guided,
            "samples": samples,
            "stride": stride,
            "history": history,
            "horizon": horizon,
            "max_entities": max_entities,
            "num_robot_proxies": num_robot_proxies_cfg,
            "batch_size": batch_size,
            "limit_samples": args.limit_samples,
            "limit_scenes": args.limit_scenes,
            "role_strategy": role_strategy,
            "role_source": dataset.role_source,
            "label_column": dataset.label_column,
            "label_map": label_map_path,
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
            "use_geometry_margin": cfg.guidance.use_geometry_margin,
            "frame_stride": frame_stride,
            "coordinate_scale": coordinate_scale,
            "device": args.device,
        }
        row.update(summary)
        append_result_csv(args.save_csv, row)
        print(f"saved_csv={args.save_csv}")


if __name__ == "__main__":
    main()
