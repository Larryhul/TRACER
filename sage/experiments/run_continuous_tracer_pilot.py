"""Pilot a continuous-parameter TRACER response model in closed loop.

This is intentionally separate from the frozen discrete suite.  The model is
conditioned on the Social-Force coordinate ``z`` and the planner approximates
the continuous posterior with a fixed quadrature grid.  The output is a
diagnostic pilot, not a replacement for the submission-scale experiment.
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
from torch.utils.data import DataLoader

from sage.belief.filter import categorical_bayes_update, uniform_log_belief
from sage.experiments.run_independent_multientity_suite import (
    _build_spec,
    _multi_entity_predicted_costs,
    _multi_entity_true_costs,
)
from sage.experiments.train_persistent_trajectory_response import (
    _flatten_entities,
    _move,
)
from sage.response.closed_loop import PlanningCostConfig
from sage.response.independent_social_force import (
    MultiEntityIndependentSocialForceDataset,
    MultiEntityIndependentSocialForceSimulator,
    MultiEntitySocialForceDatasetSpec,
)
from sage.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    continuous_coordinate_nll,
    continuous_trajectory_nll,
    stationary_reference_candidates,
    trajectory_mode_log_likelihood,
)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train(
    model: BeliefConditionedTeamResponseModel,
    loader: DataLoader,
    args: argparse.Namespace,
) -> list[float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for raw in loader:
            batch = _flatten_entities(_move(raw, args.device))
            coordinates = batch["parameter_coordinate"][:, None, None].expand(
                -1, batch["entity_history"].shape[1], 1
            )
            output = model(
                batch["entity_history"],
                batch["robot_current"],
                batch["robot_candidates"],
                robot_mask=None,
                neighbor_history=batch.get("neighbor_history"),
                neighbor_mask=batch.get("neighbor_mask"),
                condition_coordinates=coordinates,
            )
            nll = continuous_trajectory_nll(
                output["mean"],
                output["log_std"],
                batch["target_future_vel"],
                batch["active_mask"],
            )
            coordinate_target = batch["parameter_coordinate"][:, None].expand(
                -1, batch["entity_history"].shape[1]
            )
            prior_nll = continuous_coordinate_nll(
                output["coordinate_prior"],
                coordinate_target,
                batch["active_mask"],
            )
            steps = int(torch.randint(1, model.config.horizon + 1, ()).item())
            prefix_nll = continuous_trajectory_nll(
                output["mean"][..., :steps, :],
                output["log_std"][..., :steps, :],
                batch["target_future_vel"][..., :steps, :],
                batch["active_mask"],
            )
            loss = (
                nll
                + args.prefix_loss_weight * prefix_nll
                + args.coordinate_prior_weight * prior_nll
            )
            if args.coordinate_mse_weight > 0.0:
                loss = loss + args.coordinate_mse_weight * F.smooth_l1_loss(
                    output["coordinate_prior"][..., 0],
                    coordinate_target,
                    reduction="none",
                ).masked_select(batch["active_mask"]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            observations = int(batch["active_mask"].sum())
            total += float(loss.detach()) * observations
            count += observations
        value = total / max(count, 1)
        history.append(value)
        print(f"continuous response epoch={epoch + 1}/{args.epochs} loss={value:.6f}", flush=True)
    return history


def _neighbor_batch(
    histories: torch.Tensor,
    presence: torch.Tensor,
    candidate_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, entities, history, _ = histories.shape
    neighbor = histories[:, :, None].expand(batch, entities, entities, history, 2)
    mask = presence[:, :, None].expand(batch, entities, entities).clone()
    mask &= ~torch.eye(entities, device=histories.device, dtype=torch.bool)[None]
    neighbor = neighbor[:, :, None].expand(
        batch, entities, candidate_count, entities, history, 2
    ).reshape(batch * entities * candidate_count, 1, entities, history, 2)
    mask = mask[:, :, None].expand(
        batch, entities, candidate_count, entities
    ).reshape(batch * entities * candidate_count, 1, entities)
    return neighbor, mask


@torch.no_grad()
def _calibrate_uncertainty(
    model: BeliefConditionedTeamResponseModel,
    args: argparse.Namespace,
) -> float:
    """Estimate one multiplicative variance calibration factor on validation."""

    validation = MultiEntityIndependentSocialForceDataset(
        MultiEntitySocialForceDatasetSpec(
            split="val",
            num_episodes=max(16, args.eval_episodes),
            windows=args.windows,
            history=args.history,
            horizon=args.horizon,
            num_robots=args.num_robots,
            num_entities=args.num_entities,
            active_entities=args.active_entities,
            num_candidates=args.num_candidates,
            execute_steps=args.execute_steps,
            dt=args.dt,
            response_noise=args.response_noise,
            entity_max_speed=args.entity_max_speed,
            entity_interaction_strength=args.entity_interaction_strength,
            entity_interaction_decay=args.entity_interaction_decay,
            entity_interaction_radius=args.entity_interaction_radius,
            scene_family=args.scene_family,
            seed=args.data_seed + 101,
            cache_samples=True,
            parameter_mode="continuous",
        )
    )
    loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    numerator = 0.0
    denominator = 0.0
    model.eval()
    for raw in loader:
        batch = _flatten_entities(_move(raw, args.device))
        coordinates = batch["parameter_coordinate"][:, None, None].expand(
            -1, batch["entity_history"].shape[1], 1
        )
        output = model(
            batch["entity_history"],
            batch["robot_current"],
            batch["robot_candidates"],
            neighbor_history=batch.get("neighbor_history"),
            neighbor_mask=batch.get("neighbor_mask"),
            condition_coordinates=coordinates,
        )
        mean = output["mean"].squeeze(-3)
        log_std = output["log_std"].squeeze(-3)
        residual = batch["target_future_vel"] - mean
        mask = batch["active_mask"][..., None, None].to(residual.dtype)
        numerator += float((residual.square() * mask).sum())
        denominator += float((torch.exp(2.0 * log_std) * mask).sum())
    if denominator <= 0.0:
        return 1.0
    return float(np.clip(np.sqrt(numerator / denominator), 0.5, 3.0))


@torch.no_grad()
def _closed_loop(
    model: BeliefConditionedTeamResponseModel,
    args: argparse.Namespace,
    persistent: bool,
    uncertainty_scale: float,
    joint_cost_model: torch.nn.Module | None = None,
    joint_cost_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    grid = torch.linspace(
        args.coordinate_min,
        args.coordinate_max,
        args.grid_size,
        device=args.device,
    )
    planning = PlanningCostConfig(
        task_weight=1.0,
        safety_weight=2.0,
        human_weight=float(getattr(args, "planner_human_weight", 0.6)),
        comfort_scale=1.0,
        collision_margin=args.collision_margin,
        task_steps=args.execute_steps,
        violation_probability_alpha=args.risk_alpha,
        risk_samples=args.risk_samples,
    )
    evaluation_planning = PlanningCostConfig(
        task_weight=1.0,
        safety_weight=2.0,
        human_weight=0.6,
        comfort_scale=1.0,
        collision_margin=args.collision_margin,
        task_steps=args.execute_steps,
        violation_probability_alpha=args.risk_alpha,
        risk_samples=args.risk_samples,
    )
    costs: list[float] = []
    flips: list[float] = []
    fallbacks: list[float] = []
    regrets: list[float] = []
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
        belief = uniform_log_belief(args.grid_size, device=args.device)[None].expand(
            args.num_entities, args.grid_size
        ).clone()
        episode_cost = 0.0
        episode_flip = 0.0
        episode_fallback = 0.0
        episode_regret = 0.0
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
            candidates = pool.robot_candidates.to(args.device)
            current = pool.robot_current.to(args.device)
            candidate_count = int(candidates.shape[0])
            histories = torch.from_numpy(simulator.histories[None]).to(args.device)
            presence = torch.from_numpy(
                simulator.presence[:, simulator.window][None]
            ).to(args.device)
            expanded_history = histories[:, :, None].expand(
                1, args.num_entities, candidate_count, args.history, 2
            ).reshape(1 * args.num_entities * candidate_count, 1, args.history, 2)
            flat_current = current.permute(1, 0, 2, 3).reshape(
                args.num_entities * candidate_count, args.num_robots, 4
            )
            flat_candidates = candidates.permute(1, 0, 2, 3, 4).reshape(
                args.num_entities * candidate_count,
                args.num_robots,
                args.horizon,
                4,
            )
            neighbor_history, neighbor_mask = _neighbor_batch(
                histories, presence, candidate_count
            )
            coordinates = grid[None, None].expand(
                args.num_entities * candidate_count, 1, args.grid_size
            )
            evidence_enabled = bool(getattr(model.config, "evidence_conditioning", False))
            model_evidence_candidates = torch.from_numpy(
                evidence_candidates
            ).to(args.device)
            model_evidence_candidates = model_evidence_candidates[:, None].expand(
                args.num_entities, candidate_count, args.windows, args.num_robots,
                args.horizon, 4
            ).reshape(
                args.num_entities * candidate_count, args.windows,
                args.num_robots, args.horizon, 4
            )[:, None]
            model_evidence_responses = torch.from_numpy(
                evidence_responses
            ).to(args.device)
            model_evidence_responses = model_evidence_responses[:, None].expand(
                args.num_entities, candidate_count, args.windows, args.horizon, 2
            ).reshape(
                args.num_entities * candidate_count, args.windows, args.horizon, 2
            )[:, None]
            model_evidence_mask = torch.from_numpy(
                (np.arange(args.windows) < window)[None, None]
            ).to(args.device).expand(
                args.num_entities * candidate_count, 1, args.windows
            )
            output = model(
                expanded_history,
                flat_current[:, None],
                flat_candidates[:, None],
                neighbor_history=neighbor_history,
                neighbor_mask=neighbor_mask,
                condition_coordinates=coordinates,
                evidence_candidates=model_evidence_candidates if evidence_enabled else None,
                evidence_responses=model_evidence_responses if evidence_enabled else None,
                evidence_mask=model_evidence_mask if evidence_enabled else None,
            )
            prior = output["coordinate_prior"][:, 0].reshape(
                args.num_entities, candidate_count, 2
            )[:, 0]
            if not persistent or window == 0:
                prior_mean = prior[:, 0, None]
                prior_log_std = prior[:, 1, None]
                prior_log_prob = (
                    -0.5
                    * (grid[None] - prior_mean).square()
                    * torch.exp(-2.0 * prior_log_std)
                    - prior_log_std
                )
                belief = torch.log_softmax(prior_log_prob, dim=-1)
            reference = stationary_reference_candidates(
                flat_current[:, None], args.horizon
            )
            reference_output = model(
                expanded_history,
                flat_current[:, None],
                reference,
                neighbor_history=neighbor_history,
                neighbor_mask=neighbor_mask,
                condition_coordinates=coordinates,
                evidence_candidates=model_evidence_candidates if evidence_enabled else None,
                evidence_responses=model_evidence_responses if evidence_enabled else None,
                evidence_mask=model_evidence_mask if evidence_enabled else None,
            )
            means = output["mean"][:, 0].reshape(
                args.num_entities,
                candidate_count,
                args.grid_size,
                args.horizon,
                2,
            )
            log_stds = output["log_std"][:, 0].reshape_as(means)
            log_stds = log_stds + float(np.log(max(uncertainty_scale, 1e-6)))
            reference_means = reference_output["mean"][:, 0].reshape_as(means)
            costs_pred = _multi_entity_predicted_costs(
                means,
                log_stds,
                reference_means,
                current.permute(1, 0, 2, 3),
                candidates.permute(1, 0, 2, 3, 4),
                belief.exp(),
                active,
                args,
                planning,
            )
            scores = costs_pred["total"].detach().cpu().numpy().astype(np.float64)
            if joint_cost_model is not None and joint_cost_weight > 0.0:
                joint_cost_model.eval()
                joint_scores = joint_cost_model(
                    histories,
                    current[None],
                    candidates[None],
                    torch.from_numpy(active[None]).to(args.device),
                )[0].detach().cpu().numpy().astype(np.float64)
                finite = np.isfinite(scores)
                if finite.any():
                    base = scores[finite]
                    joint = joint_scores[finite]
                    base_scale = max(float(base.std()), 1e-6)
                    joint_scale = max(float(joint.std()), 1e-6)
                    scores[finite] = (1.0 - joint_cost_weight) * (
                        (base - float(base.mean())) / base_scale
                    ) + joint_cost_weight * (
                        (joint - float(joint.mean())) / joint_scale
                    )
                else:
                    scores = joint_scores
            hard_safe = np.ones(candidate_count, dtype=bool)
            safe = costs_pred["safe_mask"].detach().cpu().numpy().astype(bool)
            scores[~safe] = np.inf
            fallback = not np.isfinite(scores).any()
            if fallback and args.fallback_risk_weight > 0.0:
                fallback_scores = costs_pred["total"].detach().cpu().numpy().astype(np.float64)
                fallback_scores += args.fallback_risk_weight * costs_pred[
                    "violation_probability"
                ].detach().cpu().numpy().astype(np.float64)
                selected = int(np.argmin(fallback_scores))
            else:
                selected = int(
                    costs_pred["violation_probability"].argmin()
                    if fallback
                    else np.argmin(scores)
                )
            uniform_costs = _multi_entity_predicted_costs(
                means,
                log_stds,
                reference_means,
                current.permute(1, 0, 2, 3),
                candidates.permute(1, 0, 2, 3, 4),
                torch.full_like(belief.exp(), 1.0 / args.grid_size),
                active,
                args,
                planning,
            )
            uniform_scores = uniform_costs["total"].detach().cpu().numpy().astype(np.float64)
            uniform_scores[~uniform_costs["safe_mask"].cpu().numpy().astype(bool)] = np.inf
            uniform_selected = int(
                uniform_costs["violation_probability"].argmin()
                if not np.isfinite(uniform_scores).any()
                else np.argmin(uniform_scores)
            )
            outcomes = simulator.counterfactuals(pool, evaluation_planning)
            true_costs = _multi_entity_true_costs(
                outcomes, active, spec, evaluation_planning
            )
            realized = simulator.step(pool, selected, evaluation_planning)
            episode_cost += float(true_costs[selected])
            episode_regret += float(true_costs[selected] - true_costs.min())
            episode_flip += float(selected != uniform_selected)
            episode_fallback += float(fallback)
            evidence_candidates[:, window] = pool.robot_candidates[selected].numpy()
            evidence_responses[:, window, : args.execute_steps] = realized.response.numpy()[
                :, : args.execute_steps
            ]
            if persistent:
                for entity in np.flatnonzero(active):
                    likelihood = trajectory_mode_log_likelihood(
                        means[entity, selected][None],
                        log_stds[entity, selected][None],
                        realized.response[entity].to(args.device)[None],
                    )[0]
                    belief[entity] = categorical_bayes_update(
                        belief[entity], args.likelihood_scale * likelihood
                    )
        costs.append(episode_cost / args.windows)
        regrets.append(episode_regret / args.windows)
        flips.append(episode_flip / args.windows)
        fallbacks.append(episode_fallback / args.windows)
    return {
        "cost": float(statistics.fmean(costs)),
        "oracle_regret": float(statistics.fmean(regrets)),
        "selection_flip": float(statistics.fmean(flips)),
        "fallback": float(statistics.fmean(fallbacks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/continuous_tracer_pilot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--data-seed", type=int, default=618731)
    parser.add_argument("--train-episodes", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--windows", type=int, default=8)
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
    parser.add_argument("--coordinate-mse-weight", type=float, default=0.0)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--coordinate-min", type=float, default=0.0)
    parser.add_argument("--coordinate-max", type=float, default=2.0)
    parser.add_argument("--likelihood-scale", type=float, default=0.05)
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument("--risk-alpha", type=float, default=0.1)
    parser.add_argument("--risk-samples", type=int, default=32)
    parser.add_argument("--risk-aggregation", choices=["union", "max", "mean"], default="union")
    parser.add_argument("--fallback-risk-weight", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.num_threads))
    _set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = MultiEntityIndependentSocialForceDataset(
        MultiEntitySocialForceDatasetSpec(
            split="train",
            num_episodes=args.train_episodes,
            windows=args.windows,
            history=args.history,
            horizon=args.horizon,
            num_robots=args.num_robots,
            num_entities=args.num_entities,
            active_entities=args.active_entities,
            num_candidates=args.num_candidates,
            execute_steps=args.execute_steps,
            dt=args.dt,
            response_noise=args.response_noise,
            entity_max_speed=args.entity_max_speed,
            entity_interaction_strength=args.entity_interaction_strength,
            entity_interaction_decay=args.entity_interaction_decay,
            entity_interaction_radius=args.entity_interaction_radius,
            scene_family=args.scene_family,
            seed=args.data_seed,
            cache_samples=True,
            parameter_mode="continuous",
        )
    )
    model = BeliefConditionedTeamResponseModel(
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
        )
    ).to(args.device)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    losses = _train(model, loader, args)
    uncertainty_scale = _calibrate_uncertainty(model, args)
    persistent = _closed_loop(
        model, args, persistent=True, uncertainty_scale=uncertainty_scale
    )
    reset = _closed_loop(
        model, args, persistent=False, uncertainty_scale=uncertainty_scale
    )
    payload: dict[str, Any] = {
        "config": vars(args),
        "train_loss": losses,
        "persistent": persistent,
        "reset": reset,
        "reset_minus_persistent": reset["cost"] - persistent["cost"],
        "validation_uncertainty_scale": uncertainty_scale,
        "model_config": vars(model.config),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
