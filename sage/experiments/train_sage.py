from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from sage.experiments.smoke_test import make_synthetic_batch
from sage.models.sage_model import SAGE, SAGEConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SAGE training entry point.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", default="experiments/sage/smoke/sage_synthetic.pt")
    args = parser.parse_args()

    batch = make_synthetic_batch().to(args.device)
    cfg = SAGEConfig()
    cfg.diffusion.horizon = batch.horizon
    model = SAGE(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss = model(batch)["loss"]
        loss.backward()
        opt.step()
        print(f"epoch={epoch} loss={loss.item():.6f}")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, save_path)
    print(f"saved={save_path}")


if __name__ == "__main__":
    main()
