from __future__ import annotations

from argparse import Namespace

import torch

from tracer.config import load_sage_config
from tracer.data.synthetic_navigation import collate_synthetic_scene_batch, make_synthetic_dataset_from_config
from tracer.experiments.eval_baseline_comparison import (
    build_dataset,
    constant_acceleration_baseline,
    constant_velocity_baseline,
    direct_goal_baseline,
)


def _synthetic_batch():
    _, raw_cfg = load_sage_config("configs/sage_synthetic.yaml")
    dataset = make_synthetic_dataset_from_config(
        raw_cfg,
        split="test",
        num_samples=2,
        num_robots=2,
        num_entities=3,
        seed=17,
    )
    return collate_synthetic_scene_batch([dataset[0], dataset[1]])


def test_kinematic_baselines_return_protocol_shape() -> None:
    batch = _synthetic_batch()

    cv = constant_velocity_baseline(batch)
    ca = constant_acceleration_baseline(batch)

    assert cv.shape == batch.future.shape
    assert ca.shape == batch.future.shape
    assert torch.isfinite(cv).all()
    assert torch.isfinite(ca).all()


def test_direct_goal_reaches_robot_waypoint_and_keeps_entity_cv() -> None:
    batch = _synthetic_batch()
    pred = direct_goal_baseline(batch)
    cv = constant_velocity_baseline(batch)
    endpoint = batch.current_pos + pred.sum(dim=2) * batch.dt

    goal_mask = batch.effective_goal_mask()
    assert torch.allclose(endpoint[goal_mask], batch.goals[goal_mask], atol=1e-5)
    assert torch.allclose(pred[batch.entity_mask], cv[batch.entity_mask])


def test_build_dataset_supports_controlled_synthetic_protocol() -> None:
    cfg, raw_cfg = load_sage_config("configs/sage_synthetic.yaml")
    args = Namespace(
        dataset="synthetic",
        split="test",
        num_robots=2,
        num_entities=4,
        layout="corridor",
        seed=31,
    )

    dataset, collate_fn, meta = build_dataset(args, raw_cfg, cfg)
    batch = collate_fn([dataset[0]])

    assert batch.num_agents == 6
    assert meta["source"] == "controlled_synthetic"
    assert meta["layout"] == "corridor"
    assert meta["dataset_seed"] == 200_031
