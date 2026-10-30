from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tracer.config import load_sage_config
from tracer.data.heterogeneous_scene import ID_TO_ROLE
from tracer.data.mid_adapter import ROLE_STRATEGIES
from tracer.data.perception import perception_kwargs_from_config
from tracer.data.mid_real_dataset import MIDRealSceneDataset, collate_sage_scene_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SAGE real-data loading without training.")
    parser.add_argument("--config", default="configs/tracer.yaml")
    parser.add_argument("--data-dir", default="models/processed_data_noise")
    parser.add_argument("--dataset", default="eth", choices=["eth", "hotel", "univ", "zara1", "zara2"])
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument("--num-robot-proxies", type=int, default=None)
    parser.add_argument("--role-strategy", choices=ROLE_STRATEGIES, default=None)
    parser.add_argument("--history", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=32)
    args = parser.parse_args()

    _, raw_cfg = load_sage_config(args.config)
    dataset_cfg = raw_cfg.get("dataset", {})
    history = int(args.history if args.history is not None else dataset_cfg.get("history", 8))
    horizon = int(args.horizon if args.horizon is not None else dataset_cfg.get("horizon", 12))
    max_entities = int(args.max_entities if args.max_entities is not None else dataset_cfg.get("max_entities", 8))
    num_robot_proxies = int(args.num_robot_proxies if args.num_robot_proxies is not None else dataset_cfg.get("num_robot_proxies", 1))
    role_cfg = dataset_cfg.get("pseudo_roles", {})
    num_robot_proxies_cfg = num_robot_proxies
    perception_mode_cfg = dataset_cfg.get("perception", {}).get("mode", "nearest")
    perception_kwargs = perception_kwargs_from_config(dataset_cfg)
    role_strategy = args.role_strategy or role_cfg.get("strategy", dataset_cfg.get("role_strategy", "speed_threshold"))
    role_slow_speed = float(role_cfg.get("slow_speed", 0.15))
    role_fast_speed = float(role_cfg.get("fast_speed", 1.8))
    role_slow_quantile = float(role_cfg.get("slow_quantile", 0.33))
    role_fast_quantile = float(role_cfg.get("fast_quantile", 0.67))

    data_path = Path(args.data_dir) / f"{args.dataset}_{args.split}.pkl"
    dataset = MIDRealSceneDataset(
        data_path,
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sage_scene_batch,
    )
    batch = next(iter(loader))

    print(f"data_path={data_path}")
    print(f"config={args.config}")
    print(f"samples={len(dataset)}")
    print(f"history={history}")
    print(f"horizon={horizon}")
    print(f"max_entities={max_entities}")
    print(f"num_robot_proxies={num_robot_proxies_cfg}")
    print(f"perception_mode={perception_mode_cfg}")
    print(f"role_strategy={role_strategy}")
    print(f"history_shape={tuple(batch.history.shape)}")
    print(f"future_shape={tuple(batch.future.shape)}")
    print(f"active_agents={int(batch.agent_mask.sum().item())}")
    print(f"supervised_agents={int(batch.supervision_mask().sum().item())}")
    print(f"robots={int(batch.robot_mask.sum().item())}")
    print(f"entities={int(batch.entity_mask.sum().item())}")
    active_roles = batch.roles[batch.agent_mask]
    for role_id, name in sorted(ID_TO_ROLE.items()):
        count = int((active_roles == role_id).sum().item())
        if count:
            print(f"role_{name}={count}")
    print(f"goal_shape={tuple(batch.effective_goals().shape)}")
    print(f"goal_active={int(batch.effective_goal_mask().sum().item())}")
    goal_delta = batch.goal_displacement()
    goal_mask = batch.effective_goal_mask()
    if goal_mask.any():
        goal_distance = torch.linalg.norm(goal_delta[goal_mask], dim=-1)
        print(f"goal_distance_mean={float(goal_distance.mean().item()):.6f}")
        print(f"goal_distance_max={float(goal_distance.max().item()):.6f}")
    print(f"dt={batch.dt}")


if __name__ == "__main__":
    main()
