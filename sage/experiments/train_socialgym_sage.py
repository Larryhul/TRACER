from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sage.config import load_sage_config
from sage.data.socialgym_adapter import collate_socialgym_scene_batch, make_socialgym_dataset_from_config
from sage.models.sage_model import SAGE


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAGE on SocialGym/SocialGym-style navigation episodes.")
    parser.add_argument("--config", default="configs/sage_socialgym.yaml")
    parser.add_argument("--source", choices=["style", "trajectory", "rollout", "file", "offline"], default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--num-entities", type=int, default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    cfg, raw_cfg = load_sage_config(args.config)
    training_cfg = raw_cfg.get("training", {})
    socialgym_cfg = raw_cfg.get("dataset", {}).get("socialgym", {})

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
    dataset = make_socialgym_dataset_from_config(raw_cfg, split="train", **overrides)
    if args.limit_samples > 0 and hasattr(dataset, "num_samples"):
        dataset.num_samples = min(int(dataset.num_samples), int(args.limit_samples))
    elif args.limit_samples > 0 and hasattr(dataset, "samples"):
        dataset.samples = dataset.samples[: int(args.limit_samples)]

    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 60))
    batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 128))
    lr = float(args.lr if args.lr is not None else training_cfg.get("lr", 3e-4))
    save_dir = Path(args.save_dir or training_cfg.get("checkpoint_dir", "experiments/sage/socialgym/default"))
    cfg.diffusion.horizon = int(getattr(dataset, "horizon", raw_cfg.get("dataset", {}).get("horizon", cfg.diffusion.horizon)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_socialgym_scene_batch,
        pin_memory=args.device.startswith("cuda"),
    )
    model = SAGE(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    save_dir.mkdir(parents=True, exist_ok=True)

    encoder_variant = "heterogeneous" if cfg.encoder.use_role_embedding and cfg.encoder.use_typed_relations else "homogeneous"
    print(
        f"dataset=socialgym samples={len(dataset)} split=train source={overrides.get('source', socialgym_cfg.get('source', 'style'))} "
        f"scene={overrides.get('scene', socialgym_cfg.get('scene', 'mixed'))} device={args.device} "
        f"history={raw_cfg.get('dataset', {}).get('history', 8)} horizon={cfg.diffusion.horizon} "
        f"encoder_variant={encoder_variant} generation_mode={cfg.generation_mode} "
        f"use_robot_robot_edges={cfg.encoder.use_robot_robot_edges} guidance_scale={cfg.guidance_scale}"
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
            ckpt_path = save_dir / f"socialgym_sage_epoch{epoch}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "config_path": args.config,
                    "dataset": "socialgym",
                    "source": overrides.get("source", socialgym_cfg.get("source", "style")),
                    "scene": overrides.get("scene", socialgym_cfg.get("scene", "mixed")),
                },
                ckpt_path,
            )
            print(f"saved={ckpt_path}")


if __name__ == "__main__":
    main()
