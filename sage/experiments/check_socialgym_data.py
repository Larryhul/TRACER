from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from sage.config import load_sage_config
from sage.data.heterogeneous_scene import AgentRole
from sage.data.socialgym_adapter import collate_socialgym_scene_batch, make_socialgym_dataset_from_config


def _role_count(batch, role: AgentRole) -> int:
    return int(((batch.roles == int(role)) & batch.agent_mask).sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SocialGym/SocialGym-style SAGE navigation data.")
    parser.add_argument("--config", default="configs/sage_socialgym.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--source", choices=["style", "trajectory", "rollout", "file", "offline"], default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--num-entities", type=int, default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    _, raw_cfg = load_sage_config(args.config)
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
    dataset = make_socialgym_dataset_from_config(raw_cfg, split=args.split, **overrides)
    if args.limit_samples > 0 and hasattr(dataset, "num_samples"):
        dataset.num_samples = min(int(dataset.num_samples), int(args.limit_samples))
    elif args.limit_samples > 0 and hasattr(dataset, "samples"):
        dataset.samples = dataset.samples[: int(args.limit_samples)]

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_socialgym_scene_batch)
    batch = next(iter(loader))
    batch.validate()

    print(f"dataset=socialgym")
    print(f"config={args.config}")
    print(f"split={args.split}")
    print(f"source={overrides.get('source', raw_cfg.get('dataset', {}).get('socialgym', {}).get('source', 'style'))}")
    print(f"samples={len(dataset)}")
    print(f"history={batch.history.shape[2]}")
    print(f"horizon={batch.future.shape[2]}")
    print(f"dt={batch.dt}")
    print(f"batch_history_shape={tuple(batch.history.shape)}")
    print(f"batch_future_shape={tuple(batch.future.shape)}")
    print(f"geometry_shape={None if batch.geometry is None else tuple(batch.geometry.shape)}")
    print(f"active_agents={int(batch.agent_mask.sum().item())}")
    print(f"robots={int((batch.robot_mask & batch.agent_mask).sum().item())}")
    print(f"entities={int((batch.entity_mask & batch.agent_mask).sum().item())}")
    print(f"role_robot={_role_count(batch, AgentRole.ROBOT)}")
    print(f"role_pha={_role_count(batch, AgentRole.PHA)}")
    print(f"role_hnaa={_role_count(batch, AgentRole.HNAA)}")
    print(f"role_ssaa={_role_count(batch, AgentRole.SSAA)}")
    if batch.geometry is not None:
        active_geom = batch.geometry[batch.agent_mask]
        print(f"geometry_mean={active_geom.float().mean(dim=0).tolist()}")
        print(f"geometry_max={active_geom.float().max(dim=0).values.tolist()}")
    if batch.goals is not None:
        goal_disp = batch.goal_displacement()[batch.effective_goal_mask()]
        if goal_disp.numel():
            print(f"goal_distance_mean={torch.linalg.norm(goal_disp, dim=-1).mean().item():.6f}")


if __name__ == "__main__":
    main()
