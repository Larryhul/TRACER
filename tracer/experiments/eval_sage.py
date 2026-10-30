from __future__ import annotations

import argparse

import torch

from tracer.config import sage_config_from_dict
from tracer.experiments.checkpoint_compat import apply_checkpoint_compatibility
from tracer.evaluation.metrics import navigation_metrics
from tracer.experiments.smoke_test import make_synthetic_batch
from tracer.models.sage_model import SAGE, SAGEConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SAGE evaluation entry point.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    batch = make_synthetic_batch().to(args.device)
    cfg = SAGEConfig()
    cfg.diffusion.horizon = batch.horizon
    ckpt = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        cfg = sage_config_from_dict(ckpt.get("config", None)) if "config" in ckpt else cfg
        cfg = apply_checkpoint_compatibility(cfg, ckpt.get("model", {}))
        cfg.diffusion.horizon = batch.horizon
    model = SAGE(cfg).to(args.device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    model.eval()
    sample = model.generate(batch, guided=True, step_stride=args.stride)
    metrics = navigation_metrics(sample["future_vel"], batch)
    metrics.update(sample["diagnostics"])
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
