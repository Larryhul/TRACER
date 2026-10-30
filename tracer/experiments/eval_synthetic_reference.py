from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tracer.config import load_sage_config
from tracer.data.synthetic_navigation import (
    collate_synthetic_scene_batch,
    make_synthetic_dataset_from_config,
)
from tracer.evaluation.metrics import multi_robot_metrics, navigation_metrics
from tracer.guidance.safety_social import SafetySocialGuidance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the synthetic expert trajectories as a metric reference."
    )
    parser.add_argument("--config", default="configs/sage_synthetic.yaml")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--num-entities", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg, raw = load_sage_config(args.config)
    overrides = {}
    if args.num_robots is not None:
        overrides["num_robots"] = args.num_robots
    if args.num_entities is not None:
        overrides["num_entities"] = args.num_entities
    if args.seed is not None:
        overrides["seed"] = args.seed
    dataset = make_synthetic_dataset_from_config(raw, split=args.split, **overrides)
    if args.limit_samples > 0:
        dataset.num_samples = min(dataset.num_samples, args.limit_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_synthetic_scene_batch,
    )
    guidance = SafetySocialGuidance(cfg.guidance)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        reference_vel = batch.future
        metrics = navigation_metrics(
            reference_vel,
            batch,
            goal_tolerance=cfg.guidance.goal_tolerance,
        )
        collision_margin = guidance._collision_margins(batch, reference_vel.dtype)
        metrics.update(
            multi_robot_metrics(
                reference_vel,
                batch,
                collision_margin=collision_margin,
                goal_tolerance=cfg.guidance.goal_tolerance,
            )
        )
        batch_size = batch.batch_size
        for key, value in metrics.items():
            if torch.isfinite(torch.tensor(value)):
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        count += batch_size

    summary = {key: value / max(count, 1) for key, value in sorted(totals.items())}
    payload = {
        "reference": "synthetic_generator_expert_future",
        "config": args.config,
        "split": args.split,
        "seed": args.seed,
        "dataset_seed": dataset.seed,
        "num_samples": count,
        "num_robots": dataset.num_robots,
        "num_entities": dataset.num_entities,
        "metrics": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
