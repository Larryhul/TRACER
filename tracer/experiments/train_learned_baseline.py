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
from tracer.experiments.eval_baseline_comparison import (
    REAL_DATASETS,
    SDD_ROLE_STRATEGIES,
    build_dataset,
)
from tracer.models.baseline_models import LearnedBaselineConfig, create_learned_baseline


LEARNED_BASELINES = ("mid", "trajectron_plus_plus")


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


def _limit_dataset(dataset, limit_samples: int) -> None:
    if limit_samples <= 0:
        return
    if hasattr(dataset, "indices"):
        dataset.indices = dataset.indices[:limit_samples]
    elif hasattr(dataset, "num_samples"):
        dataset.num_samples = min(int(dataset.num_samples), int(limit_samples))
    elif hasattr(dataset, "samples"):
        dataset.samples = dataset.samples[:limit_samples]


def _default_save_dir(baseline: str, dataset: str) -> str:
    suffix = "sdd" if dataset == "sdd" else dataset
    dataset_group = "sdd" if suffix == "sdd" else "eth_ucy"
    return f"experiments/baselines/{baseline}/{dataset_group}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MID/Trajectron++ learned baselines under the SAGE protocol.")
    parser.add_argument("--baseline", choices=LEARNED_BASELINES, required=True)
    parser.add_argument("--config", default="configs/tracer.yaml")
    parser.add_argument("--dataset", choices=sorted(REAL_DATASETS | {"sdd", "socialgym", "synthetic"}), default="eth")
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible training and shuffling.")
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        help="Synthetic dataset seed; kept separate from the model/training seed.",
    )
    args = parser.parse_args()

    train_generator = seed_everything(args.seed)

    if args.data_dir is None:
        args.data_dir = "raw_data/stanford" if args.dataset == "sdd" else "models/processed_data_noise"
    if args.dataset == "socialgym" and args.data_dir == "models/processed_data_noise":
        args.data_dir = ""
    if args.dataset == "sdd" and args.role_strategy is not None and args.role_strategy not in SDD_ROLE_STRATEGIES:
        raise ValueError(f"unsupported SDD role strategy: {args.role_strategy}")

    cfg, raw_cfg = load_sage_config(args.config)
    dataset_cfg = raw_cfg.get("dataset", {})
    training_cfg = raw_cfg.get("training", {})
    cfg.diffusion.horizon = int(dataset_cfg.get("horizon", cfg.diffusion.horizon))
    args.split = "train"
    dataset, collate_fn, meta = build_dataset(args, raw_cfg, cfg)
    _limit_dataset(dataset, args.limit_samples)

    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 60))
    batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 128))
    lr = float(args.lr if args.lr is not None else training_cfg.get("lr", 3e-4))
    save_dir = Path(args.save_dir or _default_save_dir(args.baseline, args.dataset))
    save_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=str(args.device).startswith("cuda"),
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=train_generator,
    )

    baseline_cfg = LearnedBaselineConfig(
        baseline=args.baseline,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
        decoder_hidden_dim=args.decoder_hidden_dim,
    )
    model = create_learned_baseline(args.baseline, cfg, baseline_cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    print(
        f"baseline={args.baseline} dataset={args.dataset} split=train samples={len(dataset)} "
        f"device={args.device} epochs={epochs} batch_size={batch_size} lr={lr} seed={args.seed} "
        f"history={meta.get('history')} horizon={meta.get('horizon')} max_entities={meta.get('max_entities')} "
        f"num_robot_proxies={meta.get('num_robot_proxies')} role_strategy={meta.get('role_strategy')} "
        f"role_source={meta.get('role_source', '')} source={meta.get('source', '')}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        count = 0
        pbar = tqdm(loader, ncols=96, desc=f"{args.baseline} epoch {epoch}")
        for batch in pbar:
            batch = batch.to(args.device)
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            count += 1
            for key, value in out.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item())
            pbar.set_postfix({key: totals[key] / count for key in sorted(totals) if key.endswith("loss")})

        if epoch % args.save_every == 0 or epoch == epochs:
            ckpt_path = save_dir / f"{args.dataset}_{args.baseline}_epoch{epoch}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "sage_config": asdict(cfg),
                    "baseline_config": asdict(baseline_cfg),
                    "baseline": args.baseline,
                    "epoch": epoch,
                    "seed": args.seed,
                    "config_path": args.config,
                    "dataset": args.dataset,
                    "meta": meta,
                },
                ckpt_path,
            )
            print(f"saved={ckpt_path}")


if __name__ == "__main__":
    main()
