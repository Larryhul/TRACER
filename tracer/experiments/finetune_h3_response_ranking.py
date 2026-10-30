"""Fine-tune a persistent response model against closed-loop candidate ranking."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from tracer.experiments.eval_persistent_closed_loop import _realized_total_cost
from tracer.response.closed_loop import (
    ClosedLoopSpec,
    PlanningCostConfig,
    SyntheticClosedLoopSimulator,
    expected_candidate_costs,
)
from tracer.response.candidate_ranking import candidate_ranking_metrics, pairwise_ranking_loss
from tracer.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    stationary_reference_candidates,
    trajectory_mode_log_likelihood,
)
from tracer.response.sage_candidate_adapter import (
    SAGECandidateAdapter,
    SAGECandidateAdapterConfig,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rank_loss(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Return pairwise logistic loss for lower-is-better candidate costs."""

    if predicted.shape != truth.shape or predicted.ndim != 3:
        raise ValueError("predicted and truth must have shape (batch, windows, candidates)")
    return pairwise_ranking_loss(
        predicted.reshape(-1, predicted.shape[-1]),
        truth.reshape(-1, truth.shape[-1]),
        tie_tolerance=1e-5,
        gap_weight_power=0.5,
        margin=0.02,
    )


def _ranking_metrics(predicted: torch.Tensor, truth: torch.Tensor) -> dict[str, float | int]:
    return candidate_ranking_metrics(
        predicted.detach().reshape(-1, predicted.shape[-1]).cpu(),
        truth.detach().reshape(-1, truth.shape[-1]).cpu(),
        tie_tolerance=1e-5,
    )


def _adapter_from_config(raw: dict[str, Any], device: str) -> SAGECandidateAdapter:
    simulation = raw["simulation"]
    candidate = raw["candidate_source"]
    config = SAGECandidateAdapterConfig(
        num_candidates=int(simulation["num_candidates"]),
        guided=bool(candidate.get("guided", True)),
        step_stride=int(candidate.get("step_stride", 10)),
        goal_speed=float(candidate.get("goal_speed", 1.2)),
        oversample_factor=int(candidate.get("oversample_factor", 1)),
        task_prefilter=bool(candidate.get("task_prefilter", False)),
        task_prefix_steps=int(candidate.get("task_prefix_steps", 3)),
        goal_direction=str(candidate.get("goal_direction", "velocity")),
        include_task_anchor=bool(candidate.get("include_task_anchor", False)),
        task_anchor_speed_scales=tuple(
            float(value) for value in candidate.get("task_anchor_speed_scales", [1.0])
        ),
        cache_candidate_pools=False,
        batch_size=int(candidate.get("batch_size", 4)),
        guidance_step_clip_norm=(
            None
            if candidate.get("guidance_step_clip_norm") is None
            else float(candidate["guidance_step_clip_norm"])
        ),
        diversity_weight=float(candidate.get("diversity_weight", 0.0)),
        goal_distance_prefilter=bool(candidate.get("goal_distance_prefilter", False)),
        stop_task_anchors_at_goal=bool(candidate.get("stop_task_anchors_at_goal", False)),
    )
    adapter = SAGECandidateAdapter.from_checkpoint(
        candidate["checkpoint"], device=device, config=config
    )
    if candidate.get("joint_speed_completion"):
        from tracer.response.coordinated_candidates import CoordinatedCandidateAdapter

        adapter = CoordinatedCandidateAdapter(adapter, candidate["joint_speed_completion"])
    return adapter


def _make_spec(raw: dict[str, Any], split: str) -> ClosedLoopSpec:
    simulation = raw["simulation"]
    fields = {
        key: value
        for key, value in simulation.items()
        if key not in {"episodes", "splits", "data_seed"}
    }
    fields["seed"] = int(simulation.get("data_seed", 2027))
    fields["split"] = split
    return ClosedLoopSpec(**fields)


def _planning_from_config(raw: dict[str, Any]) -> PlanningCostConfig:
    planning = dict(raw["planning"])
    simulation = raw["simulation"]
    planning["task_steps"] = int(simulation["execute_steps"])
    planning["response_steps"] = int(simulation["execute_steps"])
    return PlanningCostConfig(**planning)


def _collect_rollout_batch(
    raw: dict[str, Any],
    adapter: SAGECandidateAdapter,
    episodes: int,
    start_episode: int,
    split: str,
) -> dict[str, torch.Tensor]:
    simulation = raw["simulation"]
    planning = _planning_from_config(raw)
    spec = _make_spec(raw, split)
    simulators = [
        SyntheticClosedLoopSimulator(spec, start_episode + episode)
        for episode in range(episodes)
    ]
    histories: list[list[np.ndarray]] = [[] for _ in range(episodes)]
    currents: list[list[np.ndarray]] = [[] for _ in range(episodes)]
    candidates: list[list[np.ndarray]] = [[] for _ in range(episodes)]
    responses: list[list[np.ndarray]] = [[] for _ in range(episodes)]
    true_costs: list[list[np.ndarray]] = [[] for _ in range(episodes)]
    task_directions: list[list[np.ndarray]] = [[] for _ in range(episodes)]
    rngs = [
        np.random.default_rng(
            int(simulation.get("data_seed", 2027))
            + (start_episode + episode) * 104729
            + 991
        )
        for episode in range(episodes)
    ]
    windows = int(simulation["windows"])
    split_offset = 0 if split == "id" else 7_000_003
    for window in range(windows):
        sample_seeds = [
            spec.seed
            + split_offset
            + (start_episode + episode) * 104729
            + window * 1009
            for episode in range(episodes)
        ]
        scenes = [
            (
                simulator.history,
                simulator.robot_positions,
                simulator.robot_velocities,
                spec.dt,
                simulator.robot_goals,
            )
            for simulator in simulators
        ]
        pools = adapter.candidate_pools(scenes, sample_seeds)
        for episode, (simulator, pool) in enumerate(zip(simulators, pools)):
            outcomes = simulator.counterfactuals(pool, planning)
            histories[episode].append(simulator.history.copy())
            currents[episode].append(pool.robot_current.numpy().copy())
            candidates[episode].append(pool.robot_candidates.numpy().copy())
            responses[episode].append(
                np.stack([outcome.response.numpy() for outcome in outcomes])
            )
            true_costs[episode].append(
                np.asarray(
                    [
                        _realized_total_cost(outcome, spec, planning)
                        for outcome in outcomes
                    ],
                    dtype=np.float32,
                )
            )
            task_directions[episode].append(
                (simulator.robot_goals - simulator.robot_positions).astype(np.float32)
            )
            selected = int(rngs[episode].integers(0, len(outcomes)))
            simulator.step(pool, selected, planning)
    return {
        "history": torch.from_numpy(np.asarray(histories, dtype=np.float32)),
        "current": torch.from_numpy(np.asarray(currents, dtype=np.float32)),
        "candidates": torch.from_numpy(np.asarray(candidates, dtype=np.float32)),
        "responses": torch.from_numpy(np.asarray(responses, dtype=np.float32)),
        "true_cost": torch.from_numpy(np.asarray(true_costs, dtype=np.float32)),
        "task_direction": torch.from_numpy(
            np.asarray(task_directions, dtype=np.float32)
        ),
        "mode": torch.tensor([simulator.mode for simulator in simulators], dtype=torch.long),
    }


def _model_costs(
    model: BeliefConditionedTeamResponseModel,
    batch: dict[str, torch.Tensor],
    device: str,
    planning: PlanningCostConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    history = batch["history"].to(device)
    current = batch["current"].to(device)
    candidates = batch["candidates"].to(device)
    responses = batch["responses"].to(device)
    modes = batch["mode"].to(device)
    task_direction = batch["task_direction"].to(device)
    batch_size, windows, candidate_count = candidates.shape[:3]
    flat_history = history[:, :, None].expand(
        batch_size, windows, candidate_count, history.shape[-2], history.shape[-1]
    ).reshape(batch_size * windows * candidate_count, 1, history.shape[-2], history.shape[-1])
    flat_current = current.reshape(
        batch_size * windows * candidate_count, current.shape[-2], current.shape[-1]
    )
    flat_candidates = candidates.reshape(
        batch_size * windows * candidate_count,
        candidates.shape[-3],
        candidates.shape[-2],
        candidates.shape[-1],
    )
    flat_target = responses.reshape(
        batch_size * windows * candidate_count, responses.shape[-2], responses.shape[-1]
    )
    output = model(
        flat_history,
        flat_current[:, None],
        flat_candidates[:, None],
        include_pair_interactions=True,
    )
    reference = model(
        flat_history,
        flat_current[:, None],
        stationary_reference_candidates(flat_current, model.config.horizon)[:, None],
        include_pair_interactions=True,
    )
    likelihood = trajectory_mode_log_likelihood(
        output["mean"], output["log_std"], flat_target[:, None]
    )
    mode_targets = modes[:, None, None].expand(batch_size, windows, candidate_count)
    mode_targets = mode_targets.reshape(-1, 1, 1)
    true_likelihood = likelihood.gather(-1, mode_targets).squeeze(-1)
    nll = -true_likelihood.mean() / (model.config.horizon * 2)

    mode_means = output["mean"][:, 0]
    mode_log_stds = output["log_std"][:, 0]
    reference_means = reference["mean"][:, 0]
    mode_belief = F.one_hot(
        modes[:, None, None].expand(batch_size, windows, candidate_count).reshape(-1),
        num_classes=model.config.num_modes,
    ).to(device=device, dtype=mode_means.dtype)
    predicted_groups: list[torch.Tensor] = []
    for episode in range(batch_size):
        for window in range(windows):
            start = (episode * windows + window) * candidate_count
            stop = start + candidate_count
            predicted_groups.append(
                expected_candidate_costs(
                    mode_means[start:stop],
                    mode_belief[start:stop][0],
                    current[episode, window].to(device),
                    candidates[episode, window].to(device),
                    reference_means[start:stop],
                    model.config.dt,
                    planning,
                    mode_log_stds=mode_log_stds[start:stop],
                    task_direction=task_direction[episode, window],
                )["unconstrained_total"]
            )
    predicted = torch.stack(predicted_groups).reshape(batch_size, windows, candidate_count)
    return nll, predicted


def _load_response(path: Path, device: str) -> BeliefConditionedTeamResponseModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = PersistentTrajectoryModelConfig(**checkpoint["model_config"])
    model = BeliefConditionedTeamResponseModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--response-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--episodes-per-epoch", type=int, default=4)
    parser.add_argument("--ranking-weight", type=float, default=0.10)
    parser.add_argument("--response-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    if args.epochs < 1 or args.episodes_per_epoch < 1:
        raise ValueError("epochs and episodes-per-epoch must be positive")
    if args.ranking_weight < 0.0 or args.response_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    _set_seed(args.seed)
    model = _load_response(Path(args.response_checkpoint), args.device)
    adapter = _adapter_from_config(raw, args.device)
    planning = _planning_from_config(raw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    fixed_evaluation_batch = _collect_rollout_batch(
        raw, adapter, args.episodes_per_epoch, 10000, "ood"
    )
    with torch.no_grad():
        _, initial_predicted = _model_costs(model, fixed_evaluation_batch, args.device, planning)
    initial_truth = fixed_evaluation_batch["true_cost"].to(args.device)
    initial_metrics = _ranking_metrics(initial_predicted, initial_truth)
    history: list[dict[str, float]] = []
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_regret = float(initial_metrics["selected_candidate_regret"])
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        batch = _collect_rollout_batch(
            raw,
            adapter,
            args.episodes_per_epoch,
            epoch * args.episodes_per_epoch,
            "id",
        )
        nll, predicted = _model_costs(model, batch, args.device, planning)
        truth = batch["true_cost"].to(args.device)
        ranking = _rank_loss(predicted, truth)
        loss = args.response_weight * nll + args.ranking_weight * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_metrics = _ranking_metrics(predicted, truth)
        model.eval()
        with torch.no_grad():
            _, evaluation_predicted = _model_costs(model, fixed_evaluation_batch, args.device, planning)
        evaluation_truth = fixed_evaluation_batch["true_cost"].to(args.device)
        evaluation_metrics = _ranking_metrics(evaluation_predicted, evaluation_truth)
        record = {
            "loss": float(loss.detach()),
            "response_nll": float(nll.detach()),
            "ranking": float(ranking.detach()),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"ood_{key}": value for key, value in evaluation_metrics.items()},
        }
        history.append(record)
        current_regret = float(evaluation_metrics["selected_candidate_regret"])
        if current_regret < best_regret:
            best_regret = current_regret
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(
            f"h3-ranking epoch={epoch + 1}/{args.epochs} "
            f"loss={record['loss']:.6f} response_nll={record['response_nll']:.6f} "
            f"ranking={record['ranking']:.6f}",
            flush=True,
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_checkpoint = torch.load(
        Path(args.response_checkpoint), map_location="cpu", weights_only=False
    )
    model.load_state_dict(best_state)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": vars(model.config),
            "seed": args.seed,
            "source_response_checkpoint": str(args.response_checkpoint),
            "ranking_weight": args.ranking_weight,
            "response_weight": args.response_weight,
            "training_history": history,
            "initial_ood_ranking": initial_metrics,
            "selected_epoch": best_epoch,
            "source_metadata": {
                key: value
                for key, value in source_checkpoint.items()
                if key not in {"model", "model_config"}
            },
        },
        output_dir / "response.pt",
    )
    (output_dir / "fine_tune_config.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "response_checkpoint": args.response_checkpoint,
                "seed": args.seed,
                "epochs": args.epochs,
                "episodes_per_epoch": args.episodes_per_epoch,
                "ranking_weight": args.ranking_weight,
                "response_weight": args.response_weight,
                "lr": args.lr,
                "history": history,
                "initial_ood_ranking": initial_metrics,
                "selected_epoch": best_epoch,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
