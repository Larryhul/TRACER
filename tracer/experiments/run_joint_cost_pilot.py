"""Pilot a planner-aligned permutation-invariant joint cost head.

This experiment is separate from the frozen TRACER result.  It tests whether
the closed-loop bottleneck is the response-to-cost interface by training a
shared candidate scorer on counterfactual realized costs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tracer.experiments.run_continuous_ranking_pilot import (
    CounterfactualRankingDataset,
    _set_seed,
)
from tracer.experiments.run_independent_multientity_suite import (
    _build_spec,
    _multi_entity_true_costs,
)
from tracer.response.closed_loop import PlanningCostConfig
from tracer.response.independent_social_force import (
    MultiEntityIndependentSocialForceSimulator,
)
from tracer.response.joint_cost import (
    JointCandidateCostConfig,
    JointCandidateCostModel,
)


def _rank_loss(
    predicted: torch.Tensor, truth: torch.Tensor, gap_power: float = 1.0
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for first in range(predicted.shape[-1]):
        for second in range(first + 1, predicted.shape[-1]):
            delta = truth[:, first] - truth[:, second]
            mask = delta.abs() > 1e-5
            if bool(mask.any()):
                signed = torch.where(delta < 0.0, predicted[:, first] - predicted[:, second],
                                     predicted[:, second] - predicted[:, first])
                weights = delta.abs()[mask].pow(float(gap_power))
                terms.append(
                    (F.softplus(signed[mask]) * weights).sum()
                    / weights.sum().clamp_min(1e-6)
                )
    return torch.stack(terms).mean() if terms else predicted.new_zeros(())


def _make_model(args: argparse.Namespace) -> JointCandidateCostModel:
    return JointCandidateCostModel(
        JointCandidateCostConfig(
            history=args.history,
            horizon=args.horizon,
            num_robots=args.num_robots,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            dt=args.dt,
            coordinate_min=args.coordinate_min,
            coordinate_max=args.coordinate_max,
        )
    ).to(args.device)


def _train(
    model: JointCandidateCostModel,
    loader: DataLoader,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        totals = {"loss": 0.0, "regression": 0.0, "ranking": 0.0, "parameter": 0.0}
        batches = 0
        for raw in loader:
            # Dataset layout is (B, windows, candidates, entities, ...).
            batch_size, windows, candidate_count = raw["true_cost"].shape
            history_tensor = raw["history"].to(args.device).permute(0, 1, 2, 3, 4)
            current = raw["current"].to(args.device)
            candidates = raw["candidates"].to(args.device)
            active = raw["active"].to(args.device)
            evidence_candidates = raw["executed_candidates"].to(args.device).permute(
                0, 2, 1, 3, 4, 5
            )
            evidence_responses = raw["executed_responses"].to(args.device).permute(
                0, 2, 1, 3, 4
            )
            evidence_mask = raw["executed_mask"].to(args.device)
            history_flat = history_tensor.reshape(
                batch_size * windows, args.num_entities, args.history, 2
            )
            current_flat = current.reshape(
                batch_size * windows, candidate_count, args.num_entities,
                args.num_robots, 4
            )
            candidates_flat = candidates.reshape(
                batch_size * windows, candidate_count, args.num_entities,
                args.num_robots, args.horizon, 4
            )
            active_flat = active.reshape(batch_size * windows, args.num_entities)
            evidence_candidates_flat = evidence_candidates[:, None].expand(
                batch_size, windows, args.num_entities, windows, args.num_robots,
                args.horizon, 4
            ).reshape(
                batch_size * windows, args.num_entities, windows, args.num_robots,
                args.horizon, 4
            )
            evidence_responses_flat = evidence_responses[:, None].expand(
                batch_size, windows, args.num_entities, windows, args.horizon, 2
            ).reshape(
                batch_size * windows, args.num_entities, windows, args.horizon, 2
            )
            evidence_mask_flat = evidence_mask.reshape(batch_size * windows, windows)
            truth = raw["true_cost"].to(args.device).reshape(
                batch_size * windows, candidate_count
            )
            predicted = model(
                history_flat,
                current_flat,
                candidates_flat,
                active_flat,
                evidence_candidates_flat,
                evidence_responses_flat,
                evidence_mask_flat,
            )
            regression = F.smooth_l1_loss(predicted, truth)
            ranking = _rank_loss(predicted, truth, args.rank_gap_power)
            parameter_target = raw["coordinate"].to(args.device)[:, None].expand(
                batch_size, windows, args.num_entities
            ).reshape(batch_size * windows, args.num_entities)
            parameter_prediction = model.predict_parameters(
                history_flat,
                evidence_candidates_flat,
                evidence_responses_flat,
                evidence_mask_flat,
            )
            parameter_loss = F.smooth_l1_loss(parameter_prediction, parameter_target)
            loss = (
                args.regression_weight * regression
                + args.ranking_weight * ranking
                + args.parameter_weight * parameter_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["regression"] += float(regression.detach())
            totals["ranking"] += float(ranking.detach())
            totals["parameter"] += float(parameter_loss.detach())
            batches += 1
        record = {key: value / max(batches, 1) for key, value in totals.items()}
        history.append(record)
        print(
            f"joint-cost epoch={epoch + 1}/{args.epochs} "
            f"loss={record['loss']:.5f} regression={record['regression']:.5f} "
            f"ranking={record['ranking']:.5f} parameter={record['parameter']:.5f}",
            flush=True,
        )
    return history


def _planning(args: argparse.Namespace) -> PlanningCostConfig:
    return PlanningCostConfig(
        task_weight=1.0,
        safety_weight=2.0,
        human_weight=0.6,
        comfort_scale=1.0,
        collision_margin=args.collision_margin,
        task_steps=args.execute_steps,
        violation_probability_alpha=args.risk_alpha,
        risk_samples=args.risk_samples,
    )


@torch.no_grad()
def _evaluate(
    model: JointCandidateCostModel,
    args: argparse.Namespace,
    selector: str,
) -> dict[str, float]:
    model.eval()
    planning = _planning(args)
    costs: list[float] = []
    regrets: list[float] = []
    hits: list[float] = []
    for episode in range(args.eval_episodes):
        spec = _build_spec(args, "ood", args.num_robots)
        simulator = MultiEntityIndependentSocialForceSimulator(
            spec,
            episode,
            args.num_entities,
            args.active_entities,
            continuous_parameters=True,
            entity_interaction_strength=args.entity_interaction_strength,
            entity_interaction_decay=args.entity_interaction_decay,
            entity_interaction_radius=args.entity_interaction_radius,
        )
        episode_cost = 0.0
        episode_regret = 0.0
        episode_hits = 0.0
        rng = np.random.default_rng(args.data_seed + episode * 9176 + 991)
        evidence_candidates = np.zeros(
            (args.num_entities, args.windows, args.num_robots, args.horizon, 4),
            dtype=np.float32,
        )
        evidence_responses = np.zeros(
            (args.num_entities, args.windows, args.horizon, 2), dtype=np.float32
        )
        for window in range(args.windows):
            pool = simulator.candidate_pool()
            active = simulator.active_mask()
            true_costs = _multi_entity_true_costs(
                simulator.counterfactuals(pool, planning), active, spec, planning
            )
            if selector == "random":
                selected = int(rng.integers(args.num_candidates))
            elif selector == "oracle":
                selected = int(true_costs.argmin())
            else:
                history = torch.from_numpy(simulator.histories[None]).to(args.device)
                current = pool.robot_current.to(args.device)[None]
                candidates = pool.robot_candidates.to(args.device)[None]
                active_tensor = torch.from_numpy(active[None]).to(args.device)
                predicted = model(
                    history,
                    current,
                    candidates,
                    active_tensor,
                    torch.from_numpy(evidence_candidates[None]).to(args.device),
                    torch.from_numpy(evidence_responses[None]).to(args.device),
                    torch.from_numpy(
                        (np.arange(args.windows) < window)[None]
                    ).to(args.device),
                )[0]
                selected = int(predicted.argmin())
            selected_cost = float(true_costs[selected])
            episode_cost += selected_cost
            episode_regret += selected_cost - float(true_costs.min())
            episode_hits += float(selected == int(true_costs.argmin()))
            realized = simulator.step(pool, selected, planning)
            evidence_candidates[:, window] = pool.robot_candidates[selected].numpy()
            evidence_responses[:, window, : args.execute_steps] = realized.response.numpy()[
                :, : args.execute_steps
            ]
        costs.append(episode_cost / args.windows)
        regrets.append(episode_regret / args.windows)
        hits.append(episode_hits / args.windows)
    return {
        "cost": float(statistics.fmean(costs)),
        "oracle_regret": float(statistics.fmean(regrets)),
        "oracle_hit_rate": float(statistics.fmean(hits)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/joint_cost_pilot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--data-seed", type=int, default=618731)
    parser.add_argument("--train-split", choices=["train", "ood", "mixed"], default="train")
    parser.add_argument("--train-episodes", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--num-entities", type=int, default=6)
    parser.add_argument("--active-entities", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.3)
    parser.add_argument("--response-noise", type=float, default=0.08)
    parser.add_argument("--entity-max-speed", type=float, default=2.2)
    parser.add_argument("--entity-interaction-strength", type=float, default=0.8)
    parser.add_argument("--entity-interaction-decay", type=float, default=0.72)
    parser.add_argument("--entity-interaction-radius", type=float, default=0.85)
    parser.add_argument("--scene-family", default="crossing")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--rank-gap-power", type=float, default=1.0)
    parser.add_argument("--parameter-weight", type=float, default=1.0)
    parser.add_argument("--coordinate-min", type=float, default=0.0)
    parser.add_argument("--coordinate-max", type=float, default=2.0)
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument("--risk-alpha", type=float, default=0.3)
    parser.add_argument("--risk-samples", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.num_threads))
    _set_seed(args.seed)
    dataset = CounterfactualRankingDataset(args, args.train_split, args.train_episodes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = _make_model(args)
    train_history = _train(model, loader, args)
    results = {
        selector: _evaluate(model, args, selector)
        for selector in ("joint-cost", "random", "oracle")
    }
    payload = {
        "config": vars(args),
        "train": train_history,
        **results,
        "joint_minus_random": results["joint-cost"]["cost"] - results["random"]["cost"],
        "joint_minus_oracle": results["joint-cost"]["cost"] - results["oracle"]["cost"],
        "model_config": vars(model.config),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": vars(model.config)}, output_dir / "model.pt")
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
