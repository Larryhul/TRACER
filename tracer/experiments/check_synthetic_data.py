from __future__ import annotations

import argparse
from collections import Counter

from torch.utils.data import DataLoader

from tracer.config import load_sage_config
from tracer.data.heterogeneous_scene import ID_TO_ROLE
from tracer.data.synthetic_navigation import collate_synthetic_scene_batch, make_synthetic_dataset_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect controlled synthetic SAGE navigation data.")
    parser.add_argument("--config", default="configs/sage_synthetic.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--num-entities", type=int, default=None)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    _, raw_cfg = load_sage_config(args.config)
    overrides = {}
    if args.num_robots is not None:
        overrides["num_robots"] = args.num_robots
    if args.num_entities is not None:
        overrides["num_entities"] = args.num_entities
    if args.layout is not None:
        overrides["layout"] = args.layout
    dataset = make_synthetic_dataset_from_config(raw_cfg, split=args.split, **overrides)
    if args.samples > 0:
        dataset.num_samples = min(dataset.num_samples, args.samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_synthetic_scene_batch)
    batch = next(iter(loader))

    roles = batch.roles[batch.agent_mask].tolist()
    role_counts = Counter(ID_TO_ROLE.get(int(role), str(int(role))) for role in roles)
    print(f"config={args.config}")
    print(f"split={args.split}")
    print(f"samples={len(dataset)}")
    print(f"layout={dataset.layout}")
    print(f"history={dataset.history}")
    print(f"horizon={dataset.horizon}")
    print(f"num_robots={dataset.num_robots}")
    print(f"num_entities={dataset.num_entities}")
    print(f"num_agents={dataset.num_agents}")
    print(f"dt={dataset.dt}")
    print(f"history_shape={tuple(batch.history.shape)}")
    print(f"future_shape={tuple(batch.future.shape)}")
    print(f"geometry_shape={tuple(batch.geometry.shape) if batch.geometry is not None else None}")
    print(f"active_agents={int(batch.agent_mask.sum().item())}")
    print(f"robots={int(batch.robot_mask.sum().item())}")
    print(f"entities={int(batch.entity_mask.sum().item())}")
    print("role_counts=" + ",".join(f"{key}:{value}" for key, value in sorted(role_counts.items())))
    print(f"goal_shape={tuple(batch.goals.shape) if batch.goals is not None else None}")
    print(f"goal_mask={int(batch.effective_goal_mask().sum().item())}")
    print(f"mean_robot_goal_delta={batch.goal_displacement()[batch.robot_mask].norm(dim=-1).mean().item():.4f}")


if __name__ == "__main__":
    main()
