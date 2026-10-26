"""Train a continuous TRACER response model with a closed-loop ranking loss.

The pilot adds a differentiable pairwise loss between predicted candidate
costs and resettable counterfactual costs.  It is deliberately separate from
the frozen discrete experiment and is intended to test whether candidate
ranking, rather than response NLL alone, is the closed-loop bottleneck.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sage.experiments.run_continuous_tracer_pilot import (
    _closed_loop,
    _neighbor_batch,
    _set_seed,
)
from sage.experiments.run_independent_multientity_suite import (
    _build_spec,
    _multi_entity_predicted_costs,
    _multi_entity_true_costs,
)
from sage.response.closed_loop import PlanningCostConfig
from sage.response.independent_social_force import (
    MultiEntityIndependentSocialForceSimulator,
)
from sage.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    continuous_coordinate_nll,
    continuous_trajectory_nll,
    stationary_reference_candidates,
)
from sage.response.independent_social_force import MultiEntitySocialForceDatasetSpec


class CounterfactualRankingDataset(Dataset):
    """Fixed-size episodes containing every candidate's realized response/cost."""

    def __init__(self, args: argparse.Namespace, split: str, episodes: int) -> None:
        self.args = args
        self.split = split
        self.episodes = int(episodes)
        self.rollout_model: BeliefConditionedTeamResponseModel | None = None

    def __len__(self) -> int:
        return self.episodes

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        simulator_split = (
            "ood"
            if self.split == "ood" or (self.split == "mixed" and index % 2 == 1)
            else "id"
        )
        spec = _build_spec(self.args, simulator_split, self.args.num_robots)
        simulator = MultiEntityIndependentSocialForceSimulator(
            spec,
            index,
            self.args.num_entities,
            self.args.active_entities,
            continuous_parameters=True,
            entity_interaction_strength=self.args.entity_interaction_strength,
            entity_interaction_decay=self.args.entity_interaction_decay,
            entity_interaction_radius=self.args.entity_interaction_radius,
        )
        planning = PlanningCostConfig(
            task_weight=1.0,
            safety_weight=2.0,
            human_weight=0.6,
            comfort_scale=1.0,
            collision_margin=self.args.collision_margin,
            task_steps=self.args.execute_steps,
            violation_probability_alpha=self.args.risk_alpha,
            risk_samples=self.args.risk_samples,
        )
        histories: list[np.ndarray] = []
        currents: list[np.ndarray] = []
        candidates: list[np.ndarray] = []
        responses: list[np.ndarray] = []
        active_masks: list[np.ndarray] = []
        presence_masks: list[np.ndarray] = []
        true_costs: list[np.ndarray] = []
        executed_candidates = np.zeros(
            (self.args.windows, self.args.num_entities, self.args.num_robots,
             self.args.horizon, 4), dtype=np.float32
        )
        executed_responses = np.zeros(
            (self.args.windows, self.args.num_entities, self.args.horizon, 2),
            dtype=np.float32,
        )
        rng = np.random.default_rng(self.args.data_seed + index * 9176)
        for window in range(self.args.windows):
            pool = simulator.candidate_pool()
            active = simulator.active_mask()
            outcomes = simulator.counterfactuals(pool, planning)
            histories.append(simulator.histories.copy())
            currents.append(pool.robot_current.numpy().copy())
            candidates.append(pool.robot_candidates.numpy().copy())
            responses.append(np.stack([item.response.numpy() for item in outcomes]))
            active_masks.append(active.copy())
            presence_masks.append(simulator.presence[:, simulator.window].copy())
            true_costs.append(_multi_entity_true_costs(outcomes, active, spec, planning))
            selected = int(rng.integers(0, self.args.num_candidates))
            if self.rollout_model is not None and rng.random() < getattr(self.args, "on_policy_fraction", 0.0):
                selected = _on_policy_candidate(self.rollout_model, simulator, pool, self.args, planning)
            realized = simulator.step(pool, selected, planning)
            executed_candidates[window] = pool.robot_candidates[selected].numpy()
            executed_responses[window, :, : self.args.execute_steps] = realized.response.numpy()[:, : self.args.execute_steps]
        return {
            "history": torch.from_numpy(np.stack(histories).transpose(1, 0, 2, 3).astype(np.float32)),
            "current": torch.from_numpy(np.stack(currents).astype(np.float32)),
            "candidates": torch.from_numpy(np.stack(candidates).astype(np.float32)),
            "responses": torch.from_numpy(np.stack(responses).astype(np.float32)),
            "active": torch.from_numpy(np.stack(active_masks).astype(np.bool_)),
            "presence": torch.from_numpy(np.stack(presence_masks).astype(np.bool_)),
            "true_cost": torch.from_numpy(np.stack(true_costs).astype(np.float32)),
            "coordinate": torch.from_numpy(simulator.parameter_coordinates.copy()),
            "executed_candidates": torch.from_numpy(executed_candidates),
            "executed_responses": torch.from_numpy(executed_responses),
            "executed_mask": torch.tril(
                torch.ones(self.args.windows, self.args.windows, dtype=torch.bool),
                diagonal=-1,
            ),
        }


@torch.no_grad()
def _on_policy_candidate(model, simulator, pool, args, planning) -> int:
    """Roll out the learned response prior without reading simulator mode labels."""

    was_training = model.training
    model.eval()
    try:
        entities = args.num_entities
        count = len(pool.robot_candidates)
        history = torch.as_tensor(simulator.histories, device=args.device, dtype=torch.float32)[None]
        presence = torch.as_tensor(simulator.presence[:, simulator.window], device=args.device, dtype=torch.bool)[None]
        neighbor, neighbor_mask = _neighbor_batch(history, presence, count)
        current = pool.robot_current.to(args.device)
        candidates = pool.robot_candidates.to(args.device)
        flat_current = current.permute(1, 0, 2, 3).reshape(entities * count, 1, args.num_robots, 4)
        flat_candidates = candidates.permute(1, 0, 2, 3, 4).reshape(entities * count, 1, args.num_robots, args.horizon, 4)
        flat_history = history[0, :, None].expand(entities, count, args.history, 2).reshape(entities * count, 1, args.history, 2)
        grid = torch.linspace(args.coordinate_min, args.coordinate_max, args.grid_size, device=args.device)
        coordinates = grid[None, None].expand(entities * count, 1, -1)
        kwargs = dict(neighbor_history=neighbor, neighbor_mask=neighbor_mask, condition_coordinates=coordinates)
        output = model(flat_history, flat_current, flat_candidates, **kwargs)
        reference = model(flat_history, flat_current, stationary_reference_candidates(flat_current, args.horizon), **kwargs)
        prior = output["coordinate_prior"][:, 0].reshape(entities, count, 2)[:, 0]
        logits = -0.5 * (grid[None] - prior[:, :1]).square() * torch.exp(-2.0 * prior[:, 1:])
        shape = (entities, count, args.grid_size, args.horizon, 2)
        cost = _multi_entity_predicted_costs(
            output["mean"].reshape(shape), output["log_std"].reshape(shape), reference["mean"].reshape(shape),
            current.permute(1, 0, 2, 3), candidates.permute(1, 0, 2, 3, 4),
            logits.softmax(dim=-1), simulator.active_mask(), args, planning,
        )
        scores = cost["total"]
        if not torch.isfinite(scores).any():
            scores = cost["violation_probability"]
        return int(scores.argmin())
    finally:
        model.train(was_training)


def _ranking_loss(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Pairwise logistic ranking loss, ignoring numerically tied outcomes."""

    count = predicted.shape[-1]
    terms: list[torch.Tensor] = []
    for first in range(count):
        for second in range(first + 1, count):
            delta = truth[..., first] - truth[..., second]
            pred_delta = predicted[..., first] - predicted[..., second]
            mask = delta.abs() > 1e-5
            if bool(mask.any()):
                desired = torch.where(delta < 0.0, pred_delta, -pred_delta)
                terms.append(F.softplus(desired[mask]).mean())
    if not terms:
        return predicted.new_zeros(())
    return torch.stack(terms).mean()


def _make_model(args: argparse.Namespace) -> BeliefConditionedTeamResponseModel:
    return BeliefConditionedTeamResponseModel(
        PersistentTrajectoryModelConfig(
            history=args.history,
            horizon=args.horizon,
            num_robots=args.num_robots,
            num_modes=1,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            dt=args.dt,
            architecture="tracer",
            num_heads=4,
            graph_layers=2,
            prefix_loss_weight=args.prefix_loss_weight,
            conditioning="continuous",
            coordinate_min=args.coordinate_min,
            coordinate_max=args.coordinate_max,
            evidence_conditioning=getattr(args, "evidence_conditioning", False),
            geometry_features=getattr(args, "enhanced_response", False),
            temporal_pooling="attention" if getattr(args, "enhanced_response", False) else "mean",
            candidate_uncertainty=getattr(args, "enhanced_response", False),
            set_interactions=getattr(args, "enhanced_response", False),
            evidence_pooling="attention" if getattr(args, "enhanced_response", False) else "mean",
            evidence_prefix_steps=args.execute_steps,
        )
    ).to(args.device)


def _train(
    model: BeliefConditionedTeamResponseModel,
    loader: DataLoader,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history: list[dict[str, float]] = []
    planning = PlanningCostConfig(
        task_weight=1.0,
        safety_weight=2.0,
        human_weight=0.6,
        comfort_scale=1.0,
        collision_margin=args.collision_margin,
        task_steps=args.execute_steps,
        violation_probability_alpha=args.risk_alpha,
        risk_samples=args.risk_samples,
    )
    ranking_planning = PlanningCostConfig(
        task_weight=planning.task_weight,
        safety_weight=planning.safety_weight,
        human_weight=planning.human_weight,
        comfort_scale=planning.comfort_scale,
        collision_margin=planning.collision_margin,
        task_steps=planning.task_steps,
        violation_probability_alpha=None,
        risk_samples=planning.risk_samples,
    )
    for epoch in range(args.epochs):
        model.train()
        totals = {"loss": 0.0, "nll": 0.0, "rank": 0.0, "prior": 0.0}
        batches = 0
        for raw in loader:
            history_tensor = raw["history"].to(args.device)
            current = raw["current"].to(args.device)
            candidates = raw["candidates"].to(args.device)
            responses = raw["responses"].to(args.device)
            active = raw["active"].to(args.device)
            presence = raw["presence"].to(args.device).permute(0, 2, 1)
            true_cost = raw["true_cost"].to(args.device)
            coordinates = raw["coordinate"].to(args.device)
            evidence_candidates_base = raw["executed_candidates"].to(args.device).permute(
                0, 2, 1, 3, 4, 5
            )
            evidence_responses_base = raw["executed_responses"].to(args.device).permute(
                0, 2, 1, 3, 4
            )
            evidence_mask_base = raw["executed_mask"].to(args.device)
            batch_size, entities, windows = history_tensor.shape[:3]
            candidate_count = candidates.shape[2]
            flat_history = history_tensor[:, :, None].expand(
                batch_size, entities, candidate_count, windows, args.history, 2
            ).reshape(batch_size * entities * candidate_count, windows, args.history, 2)
            flat_current = current.permute(0, 3, 2, 1, 4, 5).reshape(
                batch_size * entities * candidate_count, windows, args.num_robots, 4
            )
            flat_candidates = candidates.permute(0, 3, 2, 1, 4, 5, 6).reshape(
                batch_size * entities * candidate_count,
                windows,
                args.num_robots,
                args.horizon,
                4,
            )
            neighbor_history = history_tensor[:, :, None].expand(
                batch_size, entities, entities, windows, args.history, 2
            )
            neighbor_history = neighbor_history[:, :, None].expand(
                batch_size, entities, candidate_count, entities, windows, args.history, 2
            ).permute(0, 1, 2, 4, 3, 5, 6).reshape(
                batch_size * entities * candidate_count,
                windows,
                entities,
                args.history,
                2,
            )
            neighbor_mask = presence[:, :, None].expand(
                batch_size, entities, entities, windows
            )
            neighbor_mask = neighbor_mask[:, :, None].expand(
                batch_size, entities, candidate_count, entities, windows
            ).permute(0, 1, 2, 4, 3).reshape(
                batch_size * entities * candidate_count, windows, entities
            )
            target_coordinates = coordinates[:, :, None, None].expand(
                batch_size, entities, candidate_count, windows
            ).reshape(batch_size * entities * candidate_count, windows, 1)
            evidence_candidates = evidence_candidates_base[:, :, None].expand(
                batch_size, entities, candidate_count, windows, args.num_robots,
                args.horizon, 4
            ).reshape(
                batch_size * entities * candidate_count, windows, args.num_robots,
                args.horizon, 4
            )
            evidence_candidates = evidence_candidates[:, :, None].expand(
                batch_size * entities * candidate_count, windows, windows,
                args.num_robots, args.horizon, 4
            )
            evidence_responses = evidence_responses_base[:, :, None].expand(
                batch_size, entities, candidate_count, windows, args.horizon, 2
            ).reshape(
                batch_size * entities * candidate_count, windows, args.horizon, 2
            )
            evidence_responses = evidence_responses[:, :, None].expand(
                batch_size * entities * candidate_count, windows, windows,
                args.horizon, 2
            )
            evidence_mask = evidence_mask_base[:, None, None].expand(
                batch_size, entities, candidate_count, windows, windows
            ).reshape(batch_size * entities * candidate_count, windows, windows)
            model_evidence = (
                bool(getattr(args, "evidence_conditioning", False))
            )
            output = model(
                flat_history,
                flat_current,
                flat_candidates,
                neighbor_history=neighbor_history,
                neighbor_mask=neighbor_mask,
                condition_coordinates=target_coordinates,
                evidence_candidates=evidence_candidates if model_evidence else None,
                evidence_responses=evidence_responses if model_evidence else None,
                evidence_mask=evidence_mask if model_evidence else None,
            )
            reference_candidates = stationary_reference_candidates(
                flat_current, args.horizon
            )
            reference_output = model(
                flat_history,
                flat_current,
                reference_candidates,
                neighbor_history=neighbor_history,
                neighbor_mask=neighbor_mask,
                condition_coordinates=target_coordinates,
                evidence_candidates=evidence_candidates if model_evidence else None,
                evidence_responses=evidence_responses if model_evidence else None,
                evidence_mask=evidence_mask if model_evidence else None,
            )
            mean = output["mean"].reshape(
                batch_size, entities, candidate_count, windows, 1, args.horizon, 2
            )
            log_std = output["log_std"].squeeze(-3).reshape_as(mean)
            reference_mean = reference_output["mean"].reshape(
                batch_size, entities, candidate_count, windows, 1, args.horizon, 2
            )
            nll = continuous_trajectory_nll(
                output["mean"],
                output["log_std"],
                responses.permute(0, 3, 2, 1, 4, 5).reshape(
                    batch_size * entities * candidate_count,
                    windows,
                    args.horizon,
                    2,
                ),
                active.permute(0, 2, 1)[:, :, None].expand(
                    batch_size, entities, candidate_count, windows
                ).reshape(batch_size * entities * candidate_count, windows),
            )
            prior_target = coordinates[:, :, None].expand(batch_size, entities, windows)
            prior_nll = continuous_coordinate_nll(
                output["coordinate_prior"].reshape(batch_size, entities, candidate_count, windows, 2)[:, :, 0],
                prior_target,
                active.permute(0, 2, 1),
            )
            predicted_by_window: list[torch.Tensor] = []
            for batch_index in range(batch_size):
                per_window: list[torch.Tensor] = []
                for window in range(windows):
                    beliefs = torch.ones(
                        entities, 1, device=args.device, dtype=mean.dtype
                    )
                    costs = _multi_entity_predicted_costs(
                        mean[batch_index, :, :, window],
                        log_std[batch_index, :, :, window],
                        reference_mean[batch_index, :, :, window],
                        current[batch_index, window].permute(1, 0, 2, 3),
                        candidates[batch_index, window].permute(1, 0, 2, 3, 4),
                        beliefs,
                        active[batch_index, window].detach().cpu().numpy(),
                        args,
                        ranking_planning,
                    )
                    per_window.append(costs["total"])
                predicted_by_window.append(torch.stack(per_window))
            predicted = torch.stack(predicted_by_window)
            rank = _ranking_loss(predicted, true_cost)
            loss = (
                args.response_loss_weight * nll
                + args.coordinate_prior_weight * prior_nll
                + args.ranking_weight * rank
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["nll"] += float(nll.detach())
            totals["rank"] += float(rank.detach())
            totals["prior"] += float(prior_nll.detach())
            batches += 1
        record = {key: value / max(batches, 1) for key, value in totals.items()}
        history.append(record)
        print(
            f"ranking epoch={epoch + 1}/{args.epochs} "
            f"loss={record['loss']:.5f} nll={record['nll']:.5f} "
            f"rank={record['rank']:.5f} prior={record['prior']:.5f}",
            flush=True,
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/continuous_ranking_pilot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--data-seed", type=int, default=618731)
    parser.add_argument("--train-episodes", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
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
    parser.add_argument("--prefix-loss-weight", type=float, default=1.0)
    parser.add_argument("--coordinate-prior-weight", type=float, default=0.2)
    parser.add_argument("--response-loss-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--coordinate-min", type=float, default=0.0)
    parser.add_argument("--coordinate-max", type=float, default=2.0)
    parser.add_argument("--evidence-conditioning", action="store_true")
    parser.add_argument("--enhanced-response", action="store_true")
    parser.add_argument("--on-policy-fraction", type=float, default=0.0)
    parser.add_argument("--likelihood-scale", type=float, default=0.5)
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument("--risk-alpha", type=float, default=0.3)
    parser.add_argument("--risk-samples", type=int, default=32)
    parser.add_argument("--risk-aggregation", choices=["union", "max", "mean"], default="union")
    parser.add_argument("--fallback-risk-weight", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args()
    if not 0.0 <= args.on_policy_fraction <= 1.0:
        parser.error("--on-policy-fraction must lie in [0, 1]")
    torch.set_num_threads(max(1, args.num_threads))
    _set_seed(args.seed)
    dataset = CounterfactualRankingDataset(args, "train", args.train_episodes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = _make_model(args)
    dataset.rollout_model = model
    train_history = _train(model, loader, args)
    uncertainty_scale = 1.0
    persistent = _closed_loop(model, args, persistent=True, uncertainty_scale=uncertainty_scale)
    reset = _closed_loop(model, args, persistent=False, uncertainty_scale=uncertainty_scale)
    payload: dict[str, Any] = {
        "config": vars(args),
        "train": train_history,
        "persistent": persistent,
        "reset": reset,
        "reset_minus_persistent": reset["cost"] - persistent["cost"],
        "model_config": vars(model.config),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "model_config": vars(model.config)},
        output_dir / "model.pt",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
