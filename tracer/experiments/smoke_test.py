from __future__ import annotations

import argparse

import torch

from tracer.data.mid_adapter import build_scene_batch_from_tensors
from tracer.models.sage_model import SAGE, SAGEConfig


def make_synthetic_batch(batch_size: int = 4, num_entities: int = 3, history: int = 8, horizon: int = 12):
    robot_history = torch.zeros(batch_size, history, 2)
    robot_history[..., 0] = torch.linspace(-1.0, 0.0, history)
    robot_future = torch.zeros(batch_size, horizon, 2)
    robot_future[..., 0] = 0.4

    entity_history = torch.randn(batch_size, num_entities, history, 2) * 0.1
    entity_history[..., 0] += torch.linspace(1.0, 2.0, num_entities).view(1, num_entities, 1)
    entity_history[..., 1] += torch.linspace(-0.6, 0.6, num_entities).view(1, num_entities, 1)
    entity_future = torch.zeros(batch_size, num_entities, horizon, 2)
    entity_future[..., 0] = -0.1
    entity_mask = torch.ones(batch_size, num_entities, dtype=torch.bool)
    return build_scene_batch_from_tensors(
        robot_history=robot_history,
        robot_future=robot_future,
        neighbor_history=entity_history,
        neighbor_future=entity_future,
        neighbor_mask=entity_mask,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(7)
    batch = make_synthetic_batch().to(args.device)
    cfg = SAGEConfig()
    cfg.encoder.use_goal_conditioning = True
    cfg.guidance.w_goal = 0.3
    cfg.diffusion.horizon = batch.horizon
    model = SAGE(cfg).to(args.device)
    out = model(batch)
    out["loss"].backward()
    sample = model.generate(batch, guided=True, step_stride=args.stride)
    assert sample["future_vel"].shape == batch.future.shape
    assert torch.isfinite(sample["future_vel"]).all()
    assert torch.isfinite(out["loss"]).all()
    print(f"loss={out['loss'].item():.6f}")
    print(sample["diagnostics"])


if __name__ == "__main__":
    main()
