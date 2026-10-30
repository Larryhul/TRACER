from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tracer.config import load_sage_config
from tracer.data.mid_adapter import ROLE_STRATEGIES
from tracer.data.perception import perception_kwargs_from_config
from tracer.data.mid_real_dataset import MIDRealSceneDataset, collate_sage_scene_batch
from tracer.models.sage_model import SAGE


def seed_everything(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAGE on MID processed real trajectory data.")
    parser.add_argument("--config", default="configs/tracer.yaml")
    parser.add_argument("--data-dir", default="models/processed_data_noise")
    parser.add_argument("--dataset", default="eth", choices=["eth", "hotel", "univ", "zara1", "zara2"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument("--num-robot-proxies", type=int, default=None)
    parser.add_argument("--disable-robot-robot-edges", action="store_true")
    parser.add_argument("--role-strategy", choices=ROLE_STRATEGIES, default=None)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible training and shuffling.")
    args = parser.parse_args()

    train_generator = seed_everything(args.seed)

    cfg, raw_cfg = load_sage_config(args.config)
    dataset_cfg = raw_cfg.get("dataset", {})
    training_cfg = raw_cfg.get("training", {})

    history = int(dataset_cfg.get("history", 8))
    horizon = int(dataset_cfg.get("horizon", cfg.diffusion.horizon))
    max_entities = int(args.max_entities if args.max_entities is not None else dataset_cfg.get("max_entities", 8))
    num_robot_proxies = int(args.num_robot_proxies if args.num_robot_proxies is not None else dataset_cfg.get("num_robot_proxies", 1))
    perception_mode_cfg = dataset_cfg.get("perception", {}).get("mode", "nearest")
    role_cfg = dataset_cfg.get("pseudo_roles", {})
    perception_kwargs = perception_kwargs_from_config(dataset_cfg)
    role_strategy = args.role_strategy or role_cfg.get("strategy", dataset_cfg.get("role_strategy", "speed_threshold"))
    role_slow_speed = float(role_cfg.get("slow_speed", 0.15))
    role_fast_speed = float(role_cfg.get("fast_speed", 1.8))
    role_slow_quantile = float(role_cfg.get("slow_quantile", 0.33))
    role_fast_quantile = float(role_cfg.get("fast_quantile", 0.67))
    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 50))
    batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 128))
    lr = float(args.lr if args.lr is not None else training_cfg.get("lr", 3e-4))
    save_dir = Path(args.save_dir or training_cfg.get("checkpoint_dir", "experiments/tracer/eth_ucy/goal"))

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

    train_path = Path(args.data_dir) / f"{args.dataset}_train.pkl"
    dataset = MIDRealSceneDataset(
        train_path,
        history=history,
        horizon=horizon,
        max_entities=max_entities,
        num_robot_proxies=num_robot_proxies,
        role_strategy=role_strategy,
        role_slow_speed=role_slow_speed,
        role_fast_speed=role_fast_speed,
        role_slow_quantile=role_slow_quantile,
        role_fast_quantile=role_fast_quantile,
        **perception_kwargs,
    )
    if args.limit_samples > 0:
        dataset.indices = dataset.indices[: args.limit_samples]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_sage_scene_batch,
        pin_memory=args.device.startswith("cuda"),
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=train_generator,
    )

    model = SAGE(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"dataset={args.dataset} samples={len(dataset)} device={args.device} "
        f"history={history} horizon={horizon} max_entities={max_entities} role_strategy={role_strategy} "
        f"seed={args.seed} goal_conditioning={cfg.encoder.use_goal_conditioning} encoder_variant={encoder_variant} generation_mode={generation_mode} model_variant={model_variant} relation_granularity={relation_granularity} history_encoder={history_encoder_type} feature_fusion={feature_fusion} guidance_update={guidance_update} graph_context_token={use_graph_context_token} relation_attention={cfg.encoder.relation_attention_type} agent_attention={cfg.diffusion.use_agent_attention} use_robot_robot_edges={cfg.encoder.use_robot_robot_edges} num_robot_proxies={num_robot_proxies} perception_mode={perception_mode_cfg} w_goal={cfg.guidance.w_goal}"
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
            ckpt_path = save_dir / f"{args.dataset}_sage_epoch{epoch}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "config_path": args.config,
                    "seed": args.seed,
                },
                ckpt_path,
            )
            print(f"saved={ckpt_path}")


if __name__ == "__main__":
    main()
