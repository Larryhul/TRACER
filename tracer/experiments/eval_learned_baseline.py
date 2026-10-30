from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tracer.config import load_sage_config, sage_config_from_dict
from tracer.evaluation.metrics import (
    best_of_navigation_metrics,
    multi_robot_metrics,
    navigation_metrics,
    role_safety_metrics,
    social_compliance_metrics,
)
from tracer.experiments.eval_baseline_comparison import (
    REAL_DATASETS,
    SDD_ROLE_STRATEGIES,
    build_dataset,
    override_metric_config,
)
from tracer.experiments.eval_real_sage import append_result_csv, load_checkpoint
from tracer.guidance.safety_social import SafetySocialGuidance
from tracer.models.baseline_models import LearnedBaselineConfig, create_learned_baseline


LEARNED_BASELINES = ("mid", "trajectron_plus_plus")


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _limit_dataset(dataset, limit_samples: int) -> None:
    if limit_samples <= 0:
        return
    if hasattr(dataset, "indices"):
        dataset.indices = dataset.indices[:limit_samples]
    elif hasattr(dataset, "num_samples"):
        dataset.num_samples = min(int(dataset.num_samples), int(limit_samples))
    elif hasattr(dataset, "samples"):
        dataset.samples = dataset.samples[:limit_samples]


def _baseline_config_from_checkpoint(ckpt: dict, args: argparse.Namespace) -> LearnedBaselineConfig:
    raw = dict(ckpt.get("baseline_config", {}) or {})
    raw.setdefault("baseline", args.baseline or ckpt.get("baseline", "trajectron_plus_plus"))
    raw["latent_dim"] = int(args.latent_dim if args.latent_dim is not None else raw.get("latent_dim", 32))
    raw["kl_weight"] = float(args.kl_weight if args.kl_weight is not None else raw.get("kl_weight", 0.01))
    raw["decoder_hidden_dim"] = int(
        args.decoder_hidden_dim if args.decoder_hidden_dim is not None else raw.get("decoder_hidden_dim", 256)
    )
    return LearnedBaselineConfig(**{key: raw[key] for key in LearnedBaselineConfig.__dataclass_fields__ if key in raw})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MID/Trajectron++ learned baselines under the SAGE protocol.")
    parser.add_argument("--baseline", choices=LEARNED_BASELINES, default=None)
    parser.add_argument("--config", default="configs/tracer.yaml")
    parser.add_argument("--dataset", choices=sorted(REAL_DATASETS | {"sdd", "socialgym", "synthetic"}), default="eth")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--role-metrics", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--decoder-hidden-dim", type=int, default=None)
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible stochastic sampling.")
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        help="Synthetic dataset seed; kept separate from the evaluation/model seed.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.data_dir is None:
        args.data_dir = "raw_data/stanford" if args.dataset == "sdd" else "models/processed_data_noise"
    if args.dataset == "socialgym" and args.data_dir == "models/processed_data_noise":
        args.data_dir = ""
    if args.dataset == "sdd" and args.role_strategy is not None and args.role_strategy not in SDD_ROLE_STRATEGIES:
        raise ValueError(f"unsupported SDD role strategy: {args.role_strategy}")

    ckpt = load_checkpoint(args.checkpoint, args.device)
    baseline = str(args.baseline or ckpt.get("baseline", "trajectron_plus_plus"))
    args.baseline = baseline

    file_cfg, raw_cfg = load_sage_config(args.config)
    cfg = sage_config_from_dict(ckpt.get("sage_config", None)) if "sage_config" in ckpt else file_cfg
    cfg.diffusion.horizon = int(raw_cfg.get("dataset", {}).get("horizon", cfg.diffusion.horizon))
    comfort_sigma_long, comfort_sigma_lat = override_metric_config(cfg, args)

    dataset, collate_fn, meta = build_dataset(args, raw_cfg, cfg)
    _limit_dataset(dataset, args.limit_samples)
    eval_cfg = raw_cfg.get("evaluation", {})
    batch_size = int(args.batch_size if args.batch_size is not None else eval_cfg.get("batch_size", 128))
    samples = int(args.samples if args.samples is not None else eval_cfg.get("samples", 20))
    stride = int(args.stride if args.stride is not None else eval_cfg.get("stride", 10))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=str(args.device).startswith("cuda"),
    )

    baseline_cfg = _baseline_config_from_checkpoint(ckpt, args)
    model = create_learned_baseline(baseline, cfg, baseline_cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    guidance_helper = SafetySocialGuidance(cfg.guidance)

    totals: dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for batch in tqdm(loader, ncols=96, desc=f"eval {args.dataset}-{baseline}"):
            batch = batch.to(args.device)
            out = model.generate(batch, samples=samples, step_stride=stride)
            pred_samples = out["future_vel"]
            if pred_samples.ndim != 5:
                raise ValueError(f"model.generate must return future_vel as (K,B,A,T,2), got {tuple(pred_samples.shape)}")

            if pred_samples.shape[0] == 1:
                metrics = navigation_metrics(
                    pred_samples[0],
                    batch,
                    include_role_metrics=args.role_metrics,
                    goal_tolerance=cfg.guidance.goal_tolerance,
                )
            else:
                metrics = best_of_navigation_metrics(
                    pred_samples,
                    batch,
                    include_role_metrics=args.role_metrics,
                    goal_tolerance=cfg.guidance.goal_tolerance,
                )

            diag_totals: dict[str, float] = {}
            for sample_idx in range(pred_samples.shape[0]):
                pred_vel = pred_samples[sample_idx]
                for key, value in guidance_helper.diagnostics(pred_vel, batch).items():
                    diag_totals[key] = diag_totals.get(key, 0.0) + float(value)
                for key, value in social_compliance_metrics(
                    pred_vel,
                    batch,
                    comfort=cfg.guidance.comfort,
                    max_speed=cfg.guidance.max_speed,
                    personal_space_margin=args.pir_margin,
                    front_only=cfg.guidance.front_only,
                    eps=cfg.guidance.eps,
                    include_role_metrics=args.role_metrics,
                ).items():
                    diag_totals[key] = diag_totals.get(key, 0.0) + float(value)
                collision_margin = guidance_helper._collision_margins(batch, pred_vel.dtype)
                for key, value in multi_robot_metrics(
                    pred_vel,
                    batch,
                    collision_margin=collision_margin,
                    goal_tolerance=cfg.guidance.goal_tolerance,
                    eps=cfg.guidance.eps,
                ).items():
                    diag_totals[key] = diag_totals.get(key, 0.0) + float(value)
                if args.role_metrics:
                    for key, value in role_safety_metrics(
                        pred_vel,
                        batch,
                        collision_margin=cfg.guidance.collision_margin,
                        eps=cfg.guidance.eps,
                    ).items():
                        diag_totals[key] = diag_totals.get(key, 0.0) + float(value)
            metrics.update({key: value / pred_samples.shape[0] for key, value in diag_totals.items()})

            bs = batch.batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * bs
            n += bs

    summary = {key: totals[key] / max(n, 1) for key in sorted(totals)}
    print(
        f"baseline={baseline} dataset={args.dataset} split={args.split} checkpoint={args.checkpoint} "
        f"samples={samples} stride={stride} seed={args.seed} dataset_seed={meta.get('dataset_seed', '')} "
        f"history={meta.get('history')} horizon={meta.get('horizon')} "
        f"max_entities={meta.get('max_entities')} num_robot_proxies={meta.get('num_robot_proxies')} "
        f"role_strategy={meta.get('role_strategy')} role_source={meta.get('role_source', '')}"
    )
    for key, value in summary.items():
        print(f"{key}={value:.6f}")

    if args.save_csv:
        row = {
            "dataset": args.dataset,
            "source": meta.get("source", ""),
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "config": args.config,
            "guided": False,
            "samples": samples,
            "stride": stride,
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
            "encoder_variant": "heterogeneous" if cfg.encoder.use_role_embedding and cfg.encoder.use_typed_relations else "homogeneous",
            "generation_mode": "diffusion" if baseline == "mid" else "cvae",
            "model_variant": baseline,
            "baseline": baseline,
            "history_encoder_type": str(getattr(cfg.encoder, "history_encoder_type", "")),
            "feature_fusion": str(getattr(cfg.encoder, "feature_fusion", "")),
            "relation_granularity": str(getattr(cfg.encoder, "relation_granularity", "")),
            "guidance_update": "none",
            "use_graph_context_token": bool(getattr(cfg.diffusion, "use_graph_context_token", False)),
            "use_robot_robot_edges": bool(getattr(cfg.encoder, "use_robot_robot_edges", False)),
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
            "epoch": ckpt.get("epoch", ""),
            "eval_seed": args.seed,
            "train_seed": ckpt.get("seed", ""),
        }
        row.update(summary)
        append_result_csv(args.save_csv, row)
        print(f"saved_csv={args.save_csv}")


if __name__ == "__main__":
    main()
