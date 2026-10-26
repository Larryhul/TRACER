from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sage.config import load_sage_config
from sage.data.mid_adapter import ROLE_STRATEGIES
from sage.data.perception import perception_kwargs_from_config
from sage.data.sdd_adapter import SDDOfficialSceneDataset, SDDRawSceneDataset, collate_sdd_scene_batch
from sage.experiments.eval_real_sage import seed_everything
from sage.models.sage_model import SAGE

SDD_ROLE_STRATEGIES = ("semantic_or_speed", *ROLE_STRATEGIES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAGE on Stanford Drone Dataset trajectories.")
    parser.add_argument("--config", default="configs/sage_sdd.yaml")
    parser.add_argument("--data-dir", default="raw_data/stanford")
    parser.add_argument("--source", choices=["trajnet", "official"], default=None)
    parser.add_argument("--sdd-root", default=None, help="Official SDD root containing annotations/*/video*/annotations.txt.")
    parser.add_argument("--coordinate", choices=["center", "bottom_center", "top_left"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
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
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible training and shuffling.")
    args = parser.parse_args()

    seed_everything(args.seed)

    cfg, raw_cfg = load_sage_config(args.config)
    dataset_cfg = raw_cfg.get("dataset", {})
    sdd_cfg = dataset_cfg.get("sdd", {})
    training_cfg = raw_cfg.get("training", {})

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
    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 50))
    batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 128))
    lr = float(args.lr if args.lr is not None else training_cfg.get("lr", 3e-4))
    save_dir = Path(args.save_dir or training_cfg.get("checkpoint_dir", "experiments/sage/sdd/goal"))
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

    data_path = Path(args.data_dir) / "train_trajnet.pkl"
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
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_sdd_scene_batch,
        pin_memory=args.device.startswith("cuda"),
    )

    model = SAGE(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"dataset=sdd source={source} samples={len(dataset)} scenes={len(dataset.scenes)} role_source={dataset.role_source} "
        f"device={args.device} seed={args.seed} history={history} horizon={horizon} max_entities={max_entities} "
        f"role_strategy={role_strategy} goal_conditioning={cfg.encoder.use_goal_conditioning} encoder_variant={encoder_variant} generation_mode={generation_mode} model_variant={model_variant} relation_granularity={relation_granularity} history_encoder={history_encoder_type} feature_fusion={feature_fusion} guidance_update={guidance_update} graph_context_token={use_graph_context_token} relation_attention={cfg.encoder.relation_attention_type} agent_attention={cfg.diffusion.use_agent_attention} use_robot_robot_edges={cfg.encoder.use_robot_robot_edges} num_robot_proxies={num_robot_proxies_cfg} perception_mode={perception_mode_cfg} w_goal={cfg.guidance.w_goal}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(loader, ncols=90, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = batch.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = model(batch)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=sum(losses) / len(losses))

        if epoch % args.save_every == 0 or epoch == epochs:
            ckpt_path = save_dir / f"sdd_sage_epoch{epoch}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "config_path": args.config,
                    "dataset": "sdd",
                    "role_source": dataset.role_source,
                    "label_column": dataset.label_column,
                    "label_map": label_map_path,
                    "source": source,
                    "sdd_root": sdd_root,
                    "seed": args.seed,
                },
                ckpt_path,
            )
            print(f"saved={ckpt_path}")


if __name__ == "__main__":
    main()
