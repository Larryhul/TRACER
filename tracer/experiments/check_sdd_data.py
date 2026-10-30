from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tracer.config import load_sage_config
from tracer.data.heterogeneous_scene import ID_TO_ROLE
from tracer.data.mid_adapter import ROLE_STRATEGIES
from tracer.data.perception import perception_kwargs_from_config
from tracer.data.sdd_adapter import SDDOfficialSceneDataset, SDDRawSceneDataset, collate_sdd_scene_batch

SDD_ROLE_STRATEGIES = ("semantic_or_speed", *ROLE_STRATEGIES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SAGE SDD heterogeneous data loading without training.")
    parser.add_argument("--config", default="configs/sage_sdd.yaml")
    parser.add_argument("--data-dir", default="raw_data/stanford")
    parser.add_argument("--source", choices=["trajnet", "official"], default=None)
    parser.add_argument("--sdd-root", default=None, help="Official SDD root containing annotations/*/video*/annotations.txt.")
    parser.add_argument("--coordinate", choices=["center", "bottom_center", "top_left"], default=None)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument("--num-robot-proxies", type=int, default=None)
    parser.add_argument("--role-strategy", choices=SDD_ROLE_STRATEGIES, default=None)
    parser.add_argument("--history", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--coordinate-scale", type=float, default=None)
    parser.add_argument("--label-map", default=None, help="Optional CSV with sceneId,trackId,label to restore SDD semantic labels.")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=32)
    args = parser.parse_args()

    _, raw_cfg = load_sage_config(args.config)
    dataset_cfg = raw_cfg.get("dataset", {})
    sdd_cfg = dataset_cfg.get("sdd", {})
    history = int(args.history if args.history is not None else dataset_cfg.get("history", 8))
    horizon = int(args.horizon if args.horizon is not None else dataset_cfg.get("horizon", 12))
    max_entities = int(args.max_entities if args.max_entities is not None else dataset_cfg.get("max_entities", 12))
    role_strategy = args.role_strategy or sdd_cfg.get("role_strategy", "semantic_or_speed")
    source = args.source or sdd_cfg.get("source", "trajnet")
    sdd_root = args.sdd_root or sdd_cfg.get("sdd_root", None)
    coordinate = args.coordinate or sdd_cfg.get("coordinate", "center")
    frame_stride = int(args.frame_stride if args.frame_stride is not None else sdd_cfg.get("frame_stride", 12))
    coordinate_scale = float(args.coordinate_scale if args.coordinate_scale is not None else sdd_cfg.get("coordinate_scale", 50.0))
    label_map_path = args.label_map or sdd_cfg.get("label_map", None)

    role_cfg = dataset_cfg.get("pseudo_roles", {})
    num_robot_proxies_cfg = int(args.num_robot_proxies if args.num_robot_proxies is not None else dataset_cfg.get("num_robot_proxies", 1))
    perception_mode_cfg = dataset_cfg.get("perception", {}).get("mode", "nearest")
    perception_kwargs = perception_kwargs_from_config(dataset_cfg)
    data_path = Path(args.data_dir) / f"{args.split}_trajnet.pkl"
    dataset_kwargs = dict(
        history=history,
        horizon=horizon,
        num_robot_proxies=num_robot_proxies_cfg,
        max_entities=max_entities,
        dt=float(dataset_cfg.get("dt", 0.4)),
        frame_stride=frame_stride,
        coordinate_scale=coordinate_scale,
        center_scene=bool(sdd_cfg.get("center_scene", True)),
        role_strategy=role_strategy,
        role_slow_speed=float(role_cfg.get("slow_speed", 0.15)),
        role_fast_speed=float(role_cfg.get("fast_speed", 1.8)),
        role_slow_quantile=float(role_cfg.get("slow_quantile", 0.33)),
        role_fast_quantile=float(role_cfg.get("fast_quantile", 0.67)),
        limit_scenes=args.limit_scenes,
        **perception_kwargs,
    )
    if source == "official":
        if not sdd_root:
            raise ValueError("--sdd-root is required when --source official")
        dataset = SDDOfficialSceneDataset(
            sdd_root,
            split_reference_pkl=data_path,
            coordinate=coordinate,
            **dataset_kwargs,
        )
    else:
        dataset = SDDRawSceneDataset(
            data_path,
            label_map_path=label_map_path,
            **dataset_kwargs,
        )
    if args.limit_samples > 0:
        dataset.indices = dataset.indices[: args.limit_samples]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sdd_scene_batch,
    )
    batch = next(iter(loader))

    print(f"data_path={data_path}")
    print(f"source={source}")
    print(f"sdd_root={sdd_root}")
    print(f"config={args.config}")
    print(f"scenes={len(dataset.scenes)}")
    print(f"samples={len(dataset)}")
    print(f"role_source={dataset.role_source}")
    print(f"label_column={dataset.label_column}")
    print(f"label_map={label_map_path}")
    if hasattr(dataset, "missing_scene_ids"):
        print(f"missing_scene_ids={len(dataset.missing_scene_ids)}")
    print(f"history={history}")
    print(f"horizon={horizon}")
    print(f"max_entities={max_entities}")
    print(f"num_robot_proxies={num_robot_proxies_cfg}")
    print(f"perception_mode={perception_mode_cfg}")
    print(f"role_strategy={role_strategy}")
    print(f"frame_stride={frame_stride}")
    print(f"coordinate_scale={coordinate_scale}")
    print(f"history_shape={tuple(batch.history.shape)}")
    print(f"future_shape={tuple(batch.future.shape)}")
    print(f"geometry_shape={tuple(batch.geometry.shape) if batch.geometry is not None else None}")
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
    if batch.geometry is not None:
        active_geometry = batch.geometry[batch.agent_mask]
        print(f"geometry_mean={active_geometry.mean(dim=0).tolist()}")
        print(f"geometry_max={active_geometry.max(dim=0).values.tolist()}")
    print(f"dt={batch.dt}")


if __name__ == "__main__":
    main()
