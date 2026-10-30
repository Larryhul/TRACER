from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tracer.config import load_sage_config
from tracer.data.synthetic_navigation import collate_synthetic_scene_batch, make_synthetic_dataset_from_config
from tracer.models.sage_model import SAGE


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAGE on controlled synthetic heterogeneous navigation episodes.")
    parser.add_argument("--config", default="configs/sage_synthetic.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--num-entities", type=int, default=None)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset-seed", type=int, default=None, help="Synthetic dataset seed, separate from model seed.")
    parser.add_argument("--resume-checkpoint", default=None)
    args = parser.parse_args()

    cfg, raw_cfg = load_sage_config(args.config)
    dataset_cfg = raw_cfg.get("dataset", {})
    training_cfg = raw_cfg.get("training", {})
    synthetic_cfg = dataset_cfg.get("synthetic", {})
    model_seed = int(
        args.seed if args.seed is not None else synthetic_cfg.get("seed", 2027)
    )
    dataset_seed = int(
        args.dataset_seed if args.dataset_seed is not None else synthetic_cfg.get("seed", 2027)
    )
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    overrides = {}
    if args.num_robots is not None:
        overrides["num_robots"] = args.num_robots
    if args.num_entities is not None:
        overrides["num_entities"] = args.num_entities
    if args.layout is not None:
        overrides["layout"] = args.layout
    overrides["seed"] = dataset_seed
    dataset = make_synthetic_dataset_from_config(raw_cfg, split="train", **overrides)
    if args.limit_samples > 0:
        dataset.num_samples = min(dataset.num_samples, args.limit_samples)

    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 50))
    batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 128))
    lr = float(args.lr if args.lr is not None else training_cfg.get("lr", 3e-4))
    save_dir = Path(args.save_dir or training_cfg.get("checkpoint_dir", "experiments/tracer/synthetic/default"))
    cfg.diffusion.horizon = dataset.horizon

    train_generator = torch.Generator().manual_seed(model_seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_synthetic_scene_batch,
        pin_memory=args.device.startswith("cuda"),
        generator=train_generator,
    )
    model = SAGE(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    start_epoch = 1
    if args.resume_checkpoint:
        checkpoint = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer"])
            for state in opt.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(args.device)
        if "train_generator_state" in checkpoint:
            train_generator.set_state(checkpoint["train_generator_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        start_epoch = int(checkpoint["epoch"]) + 1
    save_dir.mkdir(parents=True, exist_ok=True)

    encoder_variant = "heterogeneous" if cfg.encoder.use_role_embedding and cfg.encoder.use_typed_relations else "homogeneous"
    print(
        f"dataset=synthetic samples={len(dataset)} split=train layout={dataset.layout} "
        f"num_robots={dataset.num_robots} num_entities={dataset.num_entities} device={args.device} "
        f"history={dataset.history} horizon={dataset.horizon} encoder_variant={encoder_variant} "
        f"generation_mode={cfg.generation_mode} use_robot_robot_edges={cfg.encoder.use_robot_robot_edges} "
        f"guidance_scale={cfg.guidance_scale} w_goal={cfg.guidance.w_goal} "
        f"model_seed={model_seed} dataset_seed={dataset_seed} start_epoch={start_epoch}"
    )

    for epoch in range(start_epoch, epochs + 1):
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
            ckpt_path = save_dir / f"synthetic_sage_epoch{epoch}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "train_generator_state": train_generator.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "config_path": args.config,
                    "dataset": "synthetic",
                    "layout": dataset.layout,
                    "num_robots": dataset.num_robots,
                    "num_entities": dataset.num_entities,
                    "seed": model_seed,
                    "dataset_seed": dataset_seed,
                },
                ckpt_path,
            )
            print(f"saved={ckpt_path}")


if __name__ == "__main__":
    main()
