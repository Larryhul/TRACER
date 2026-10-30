from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from tracer.belief.filter import (
    hazard_belief_update,
    lagged_reset_beliefs,
    run_categorical_filter,
    uniform_log_belief,
)
from tracer.response.metrics import trajectory_errors
from tracer.response.h1_transfer import initialize_h1_encoder
from tracer.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    TrajectoryRecurrentBelief,
    TrajectoryRecurrentConfig,
    belief_weighted_mean,
    executed_prefix_mode_log_likelihood,
    mode_conditioned_trajectory_loss,
    trajectory_mode_log_likelihood,
)
from tracer.response.persistent_trajectory import (
    MultiEntityPersistentTrajectoryEpisodeDataset,
    MultiEntityPersistentTrajectorySpec,
    PersistentTrajectoryEpisodeDataset,
    PersistentTrajectorySpec,
    TRAJECTORY_MODE_NAMES,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_dataset(
    raw: dict[str, Any],
    split: str,
    seed: int,
    action_policy: str = "iid",
) -> PersistentTrajectoryEpisodeDataset | MultiEntityPersistentTrajectoryEpisodeDataset:
    cfg = raw["dataset"]
    common = {
        "split": split,
        "num_episodes": int(cfg[f"{split}_episodes"]),
        "windows": int(cfg.get("windows", 8)),
        "history": int(cfg.get("history", 8)),
        "horizon": int(cfg.get("horizon", 12)),
        "num_robots": int(cfg.get("num_robots", 2)),
        "dt": float(cfg.get("dt", 0.4)),
        "seed": seed,
        "action_policy": action_policy,
        "response_noise": float(cfg.get("response_noise", 0.12)),
        "influence_radius": float(cfg.get("influence_radius", 4.0)),
        "execute_steps": int(cfg.get("execute_steps", 3)),
        "entity_max_speed": float(cfg.get("entity_max_speed", 3.0)),
        "scene_family": str(cfg.get("scene_family", "radial")),
        "response_family": str(cfg.get("response_family", "pincer")),
        "cache_samples": bool(cfg.get("cache_samples", False)),
        "include_interaction_residual": bool(cfg.get("include_interaction_residual", False)),
    }
    num_entities = int(cfg.get("num_entities", 1))
    if num_entities > 1:
        return MultiEntityPersistentTrajectoryEpisodeDataset(
            MultiEntityPersistentTrajectorySpec(
                **common,
                num_entities=num_entities,
                active_entities=int(cfg.get("active_entities", 3)),
                shared_team_actions=bool(cfg.get("shared_team_actions", False)),
            )
        )
    return PersistentTrajectoryEpisodeDataset(PersistentTrajectorySpec(**common))


def _move(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def constant_velocity_target(
    entity_history: torch.Tensor,
    horizon: int,
    dt: float,
) -> torch.Tensor:
    if entity_history.ndim != 4 or entity_history.shape[-1] != 2:
        raise ValueError("entity_history must have shape (batch, windows, history, 2)")
    if entity_history.shape[-2] < 2 or horizon < 1 or dt <= 0:
        raise ValueError("history, horizon, and dt must be positive")
    velocity = (entity_history[..., -1, :] - entity_history[..., -2, :]) / dt
    return velocity.unsqueeze(-2).expand(
        *velocity.shape[:-1], horizon, velocity.shape[-1]
    )


def mixture_mean_trajectory_loss(
    mode_means: torch.Tensor,
    target: torch.Tensor,
    observation_mask: torch.Tensor,
    dt: float,
    endpoint_weight: float = 0.25,
    mode_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode_means.ndim != 5 or target.ndim != 4:
        raise ValueError("expected mode means (batch, windows, modes, horizon, 2) and target (batch, windows, horizon, 2)")
    if mode_means.shape[:2] != target.shape[:2] or mode_means.shape[-2:] != target.shape[-2:]:
        raise ValueError("mode means and target must align")
    if target.shape[-1] != 2 or target.shape[-2] < 1 or mode_means.shape[2] < 1:
        raise ValueError("nonempty modes and horizon with two coordinates required")
    if observation_mask.shape != target.shape[:2] or not observation_mask.any():
        raise ValueError("observation mask must align and select an observation")
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(endpoint_weight) or endpoint_weight < 0:
        raise ValueError("dt must be positive and endpoint weight nonnegative and finite")
    if mode_weights is None:
        prediction = mode_means.mean(dim=2)
    else:
        if mode_weights.shape != mode_means.shape[:3]:
            raise ValueError("mode weights must align with batch, windows, and modes")
        if not torch.isfinite(mode_weights).all() or (mode_weights < 0).any():
            raise ValueError("mode weights must be finite and nonnegative")
        if not torch.allclose(mode_weights.sum(-1), torch.ones_like(mode_weights[..., 0]), atol=1e-5, rtol=1e-5):
            raise ValueError("mode weights must sum to one")
        prediction = belief_weighted_mean(mode_means, mode_weights)
    velocity_error = prediction - target
    displacement_error = velocity_error.cumsum(dim=-2) * dt
    distances = torch.linalg.vector_norm(displacement_error, dim=-1)
    loss = distances.mean(dim=-1) + endpoint_weight * distances[..., -1]
    return loss[observation_mask.bool()].mean() / (dt * target.shape[-2])


def causal_training_beliefs(mode_means, log_stds, target, observation_mask, execute_steps, likelihood_scale):
    """Return detached decision priors from earlier executed prefixes only."""
    if not math.isfinite(likelihood_scale) or likelihood_scale <= 0:
        raise ValueError("training likelihood scale must be positive and finite")
    with torch.no_grad():
        likelihoods = executed_prefix_mode_log_likelihood(mode_means, log_stds, target, execute_steps)
        before, _ = run_categorical_filter(likelihoods * likelihood_scale, observation_mask=observation_mask)
    return before.exp()


def _flatten_entities(batch: dict[str, Any]) -> dict[str, Any]:
    history = batch["entity_history"]
    if history.ndim == 4:
        batch_size = history.shape[0]
        if "active_mask" not in batch:
            batch["active_mask"] = torch.ones(
                history.shape[:2], device=history.device, dtype=torch.bool
            )
        if "presence_mask" not in batch:
            batch["presence_mask"] = batch["active_mask"].clone()
        if "episode_id" not in batch:
            batch["episode_id"] = batch.get(
                "entity_id", torch.arange(batch_size, device=history.device)
            )
        batch["episode_index"] = torch.arange(batch_size, device=history.device)
        batch["entity_slot"] = torch.zeros(
            batch_size, device=history.device, dtype=torch.long
        )
        return batch
    if history.ndim != 5:
        raise ValueError("entity_history must be single- or multi-entity batched data")
    batch_size, entities = history.shape[:2]
    # Keep the other tracked entities as optional graph neighbors.  The target
    # row is still flattened exactly as before, so legacy checkpoints and all
    # existing metric code retain their original tensor contract.
    presence = batch.get("presence_mask")
    if isinstance(presence, torch.Tensor):
        neighbor_history = history[:, None].expand(
            batch_size, entities, entities, *history.shape[2:]
        )
        neighbor_mask = presence[:, None].expand(
            batch_size, entities, entities, *presence.shape[2:]
        ).clone()
        identity = torch.eye(entities, device=history.device, dtype=torch.bool)
        neighbor_mask = neighbor_mask & ~identity[None, :, :, None]
        batch["neighbor_history"] = neighbor_history.permute(0, 1, 3, 2, 4, 5).reshape(
            batch_size * entities, history.shape[2], entities, history.shape[3], history.shape[4]
        )
        batch["neighbor_mask"] = neighbor_mask.permute(0, 1, 3, 2).reshape(
            batch_size * entities, presence.shape[2], entities
        )
    flattened: dict[str, Any] = {}
    for key, value in batch.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and tuple(value.shape[:2]) == (batch_size, entities)
        ):
            flattened[key] = value.reshape(batch_size * entities, *value.shape[2:])
        else:
            flattened[key] = value
    flattened["episode_index"] = torch.arange(
        batch_size, device=history.device
    ).repeat_interleave(entities)
    flattened["entity_slot"] = torch.arange(
        entities, device=history.device
    ).repeat(batch_size)
    episode_id = batch.get("episode_id")
    if isinstance(episode_id, torch.Tensor) and tuple(episode_id.shape) == (batch_size,):
        flattened["episode_id"] = episode_id.repeat_interleave(entities)
    return flattened


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.mean()) if selected.numel() else float("nan")


def _masked_by_window(values: torch.Tensor, mask: torch.Tensor) -> list[float]:
    return [_masked_mean(values[:, window], mask[:, window]) for window in range(values.shape[1])]


def _masked_belief_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    active_mask: torch.Tensor,
    confidence_threshold: float,
) -> dict[str, float | list[float]]:
    probabilities = probabilities.clamp_min(1e-9)
    windows = probabilities.shape[1]
    true_probability = probabilities.gather(
        dim=-1,
        index=targets[:, None, None].expand(-1, windows, 1),
    ).squeeze(-1)
    correct = probabilities.argmax(dim=-1).eq(targets[:, None]).float()
    one_hot = F.one_hot(targets, num_classes=probabilities.shape[-1]).float()
    brier = (probabilities - one_hot[:, None, :]).square().sum(dim=-1)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    main_mask = active_mask.clone()
    main_mask[:, 0] = False

    valid_tracks = active_mask.any(dim=1)
    reversed_active = active_mask.flip(dims=(1,)).float().argmax(dim=1)
    last_active = windows - 1 - reversed_active
    track_indices = torch.arange(probabilities.shape[0])
    final_probability = true_probability[track_indices, last_active][valid_tracks]
    final_correct = correct[track_indices, last_active][valid_tracks]
    final_brier = brier[track_indices, last_active][valid_tracks]
    final_entropy = entropy[track_indices, last_active][valid_tracks]

    observation_number = active_mask.long().cumsum(dim=1)
    confident = active_mask & true_probability.ge(confidence_threshold)
    sentinel = active_mask.sum(dim=1) + 1
    first_confident = torch.where(
        confident,
        observation_number,
        sentinel[:, None],
    ).min(dim=1).values[valid_tracks]
    window0 = active_mask[:, 0]
    return {
        "mode_nll": _masked_mean(-true_probability.log(), main_mask),
        "mode_accuracy": _masked_mean(correct, main_mask),
        "mode_brier": _masked_mean(brier, main_mask),
        "posterior_entropy": _masked_mean(entropy, main_mask),
        "final_mode_nll": float(-final_probability.log().mean()),
        "final_mode_accuracy": float(final_correct.mean()),
        "final_mode_brier": float(final_brier.mean()),
        "final_posterior_entropy": float(final_entropy.mean()),
        "windows_to_confidence": float(first_confident.float().mean()),
        "accuracy_by_window": _masked_by_window(correct, active_mask),
        "true_probability_by_window": _masked_by_window(true_probability, active_mask),
        "entropy_by_window": _masked_by_window(entropy, active_mask),
        "window0_mode_nll": _masked_mean(-true_probability[:, 0].log(), window0),
        "window0_mode_accuracy": _masked_mean(correct[:, 0], window0),
    }


def _shuffled_identity_filter(
    log_likelihoods: torch.Tensor,
    observation_mask: torch.Tensor,
    episode_ids: torch.Tensor,
    entity_slots: torch.Tensor,
    prior_log_belief: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate entity-to-store keys across windows to corrupt identity continuity."""

    before = torch.empty_like(log_likelihoods)
    after = torch.empty_like(log_likelihoods)
    for group_index, episode_id in enumerate(torch.unique(episode_ids, sorted=True)):
        indices = torch.where(episode_ids.eq(episode_id))[0]
        indices = indices[entity_slots[indices].argsort()]
        entity_count = int(indices.numel())
        if entity_count == 1:
            group_before, group_after = run_categorical_filter(
                log_likelihoods[indices],
                prior_log_belief=prior_log_belief[indices],
                observation_mask=observation_mask[indices],
            )
            before[indices], after[indices] = group_before, group_after
            continue
        storage = prior_log_belief[indices].clone()
        entity_order = torch.arange(entity_count)
        for window in range(log_likelihoods.shape[1]):
            shift = (window + group_index) % entity_count
            corrupted_key = torch.roll(entity_order, shifts=shift)
            current = storage[corrupted_key]
            before[indices, window] = current
            updated = torch.log_softmax(
                current + log_likelihoods[indices, window], dim=-1
            )
            current_after = torch.where(
                observation_mask[indices, window, None], updated, current
            )
            after[indices, window] = current_after
            next_storage = storage.clone()
            next_storage[corrupted_key] = current_after
            storage = next_storage
    return before, after


def _train_response(
    model: BeliefConditionedTeamResponseModel,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    mode_discrimination_weight: float,
    mode_temperature: float,
    prefix_loss_weight: float = 0.0,
    interaction_residual_weight: float = 0.0,
    velocity_residual: bool = False,
    mean_trajectory_weight: float = 0.0,
    mean_endpoint_weight: float = 0.25,
    mean_trajectory_conditioning: str = "uniform",
    mean_belief_execute_steps: int = 3,
    mean_belief_likelihood_scale: float = 0.05,
) -> None:
    if not math.isfinite(mean_trajectory_weight) or mean_trajectory_weight < 0:
        raise ValueError("mean trajectory weight must be nonnegative and finite")
    if mean_trajectory_conditioning not in {"uniform", "causal-belief"}:
        raise ValueError("mean trajectory conditioning must be uniform or causal-belief")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    coordinates = model.config.horizon * 2
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        total_nll = 0.0
        total_discrimination = 0.0
        total_prefix = 0.0
        total_interaction = 0.0
        total_mean_trajectory = 0.0
        count = 0
        for raw_batch in loader:
            batch = _flatten_entities(_move(raw_batch, device))
            output = model(
                batch["entity_history"],
                batch["robot_current"],
                batch["robot_candidates"],
                include_pair_interactions=True,
                neighbor_history=batch.get("neighbor_history"),
                neighbor_mask=batch.get("neighbor_mask"),
            )
            target = batch["target_future_vel"]
            target_offset = torch.zeros_like(target)
            if velocity_residual:
                target_offset = constant_velocity_target(
                    batch["entity_history"], model.config.horizon, model.config.dt
                )
            model_target = target - target_offset
            interaction_loss = torch.zeros((), device=output["mean"].device)
            if interaction_residual_weight > 0.0:
                target_residual = batch.get("interaction_residual")
                if not isinstance(target_residual, torch.Tensor):
                    raise ValueError(
                        "interaction_residual_weight requires dataset.include_interaction_residual=true"
                    )
                additive_output = model(
                    batch["entity_history"],
                    batch["robot_current"],
                    batch["robot_candidates"],
                    include_pair_interactions=False,
                    neighbor_history=batch.get("neighbor_history"),
                    neighbor_mask=batch.get("neighbor_mask"),
                )
                predicted_residual = output["mean"] - additive_output["mean"]
                mode_index = batch["mode"].long()
                if mode_index.ndim != 1:
                    raise ValueError("mode must be one-dimensional after entity flattening")
                gather_index = mode_index[:, None, None, None, None].expand(
                    -1,
                    target_residual.shape[1],
                    1,
                    target_residual.shape[3],
                    target_residual.shape[4],
                )
                predicted_residual = predicted_residual.gather(2, gather_index).squeeze(2)
                target_residual = target_residual.gather(2, gather_index).squeeze(2)
                residual_error = (predicted_residual - target_residual).square().mean(dim=(-1, -2))
                active = batch["active_mask"].bool()
                interaction_loss = residual_error.masked_select(active).mean()
            log_likelihood = trajectory_mode_log_likelihood(
                output["mean"], output["log_std"], model_target
            )
            losses = mode_conditioned_trajectory_loss(
                log_likelihood,
                batch["mode"],
                coordinates,
                discrimination_weight=mode_discrimination_weight,
                temperature=mode_temperature,
                observation_mask=batch["active_mask"],
            )
            prefix_steps = int(torch.randint(1, model.config.horizon + 1, ()).item())
            prefix_log_likelihood = executed_prefix_mode_log_likelihood(
                output["mean"], output["log_std"], model_target, prefix_steps
            )
            prefix_losses = mode_conditioned_trajectory_loss(
                prefix_log_likelihood,
                batch["mode"],
                prefix_steps * batch["target_future_vel"].shape[-1],
                discrimination_weight=mode_discrimination_weight,
                temperature=mode_temperature,
                observation_mask=batch["active_mask"],
            )
            mean_trajectory_loss = output["mean"].new_zeros(())
            if mean_trajectory_weight > 0:
                mode_weights = None
                if mean_trajectory_conditioning == "causal-belief":
                    mode_weights = causal_training_beliefs(
                        output["mean"], output["log_std"], model_target, batch["active_mask"],
                        mean_belief_execute_steps, mean_belief_likelihood_scale,
                    )
                mean_trajectory_loss = mixture_mean_trajectory_loss(
                    output["mean"], model_target, batch["active_mask"],
                    model.config.dt, mean_endpoint_weight, mode_weights,
                )
            loss = (
                losses["loss"]
                + prefix_loss_weight * prefix_losses["loss"]
                + interaction_residual_weight * interaction_loss
                + mean_trajectory_weight * mean_trajectory_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            observations = int(batch["active_mask"].sum())
            total += float(loss.detach()) * observations
            total_nll += float(losses["trajectory_nll"].detach()) * observations
            total_discrimination += (
                float(losses["mode_discrimination"].detach()) * observations
            )
            total_prefix += float(prefix_losses["trajectory_nll"].detach()) * observations
            total_interaction += float(interaction_loss.detach()) * observations
            total_mean_trajectory += float(mean_trajectory_loss.detach()) * observations
            count += observations
        print(
            f"response epoch={epoch}/{epochs} train_loss={total / count:.6f} "
            f"train_nll={total_nll / count:.6f} "
            f"train_mode_ce={total_discrimination / count:.6f} "
            f"train_prefix_nll={total_prefix / count:.6f} "
            f"train_interaction_mse={total_interaction / count:.6f} "
            f"train_mean_trajectory={total_mean_trajectory / count:.6f}",
            flush=True,
        )


def _train_recurrent(
    model: TrajectoryRecurrentBelief,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for raw_batch in loader:
            batch = _flatten_entities(_move(raw_batch, device))
            log_probabilities = model(
                batch["robot_current"],
                batch["robot_candidates"],
                batch["target_future_vel"],
                observation_mask=batch["active_mask"],
                entity_history=batch["entity_history"],
            )
            targets = batch["mode"][:, None].expand(-1, log_probabilities.shape[1])
            active = batch["active_mask"].bool()
            loss = F.nll_loss(
                log_probabilities[active],
                targets[active],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            observations = int(active.sum())
            total += float(loss.detach()) * observations
            count += observations
        print(f"recurrent epoch={epoch}/{epochs} train_mode_nll={total / count:.6f}", flush=True)


@torch.no_grad()
def _evaluate(
    response_model: BeliefConditionedTeamResponseModel,
    recurrent_model: TrajectoryRecurrentBelief,
    loader: DataLoader,
    device: str,
    dt: float,
    likelihood_scale: float,
    confidence_threshold: float,
    execute_steps: int,
    split_name: str,
    bayesian_hazard_rate: float | None = None,
    velocity_residual: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    response_model.eval()
    recurrent_model.eval()
    means_all = []
    log_stds_all = []
    likelihoods_all = []
    prefix_likelihoods_all = []
    targets_all = []
    modes_all = []
    recurrent_all = []
    high_all = []
    active_all = []
    episode_ids_all = []
    entity_ids_all = []
    entity_slots_all = []
    for raw_batch in loader:
        batch = _flatten_entities(_move(raw_batch, device))
        target = batch["target_future_vel"]
        target_offset = torch.zeros_like(target)
        if velocity_residual:
            target_offset = constant_velocity_target(
                batch["entity_history"], response_model.config.horizon, response_model.config.dt
            )
        model_target = target - target_offset
        output = response_model(
            batch["entity_history"],
            batch["robot_current"],
            batch["robot_candidates"],
            neighbor_history=batch.get("neighbor_history"),
            neighbor_mask=batch.get("neighbor_mask"),
        )
        likelihoods_all.append(
            trajectory_mode_log_likelihood(
                output["mean"], output["log_std"], model_target
            ).cpu()
        )
        prefix_likelihoods_all.append(
            executed_prefix_mode_log_likelihood(
                output["mean"],
                output["log_std"],
                model_target,
                execute_steps,
            ).cpu()
        )
        means_all.append((output["mean"] + target_offset.unsqueeze(-3)).cpu())
        log_stds_all.append(output["log_std"].cpu())
        targets_all.append(batch["target_future_vel"].cpu())
        modes_all.append(batch["mode"].cpu())
        recurrent_all.append(
            recurrent_model(
                batch["robot_current"],
                batch["robot_candidates"],
                batch["target_future_vel"],
                observation_mask=batch["active_mask"],
                entity_history=batch["entity_history"],
            ).exp().cpu()
        )
        high_all.append(batch["high_interaction"].cpu())
        active_all.append(batch["active_mask"].cpu())
        episode_ids_all.append(batch["episode_id"].cpu())
        entity_ids_all.append(batch["entity_id"].cpu())
        entity_slots_all.append(batch["entity_slot"].cpu())

    means = torch.cat(means_all)
    log_stds = torch.cat(log_stds_all)
    raw_likelihoods = torch.cat(likelihoods_all)
    prefix_likelihoods = torch.cat(prefix_likelihoods_all)
    targets = torch.cat(targets_all)
    modes = torch.cat(modes_all)
    recurrent_after = torch.cat(recurrent_all)
    high = torch.cat(high_all).bool()
    active = torch.cat(active_all).bool()
    episode_ids = torch.cat(episode_ids_all).long()
    entity_ids = torch.cat(entity_ids_all).long()
    entity_slots = torch.cat(entity_slots_all).long()
    tracks, windows, num_modes = raw_likelihoods.shape
    mode_targets = modes[:, None].expand(-1, windows)
    true_likelihoods = prefix_likelihoods.gather(
        dim=-1, index=mode_targets[..., None]
    ).squeeze(-1)
    wrong_mask = F.one_hot(mode_targets, num_classes=num_modes).bool()
    best_wrong_likelihoods = prefix_likelihoods.masked_fill(wrong_mask, -torch.inf).max(dim=-1).values
    likelihood_margin = (true_likelihoods - best_wrong_likelihoods) / (
        execute_steps * targets.shape[-1]
    )
    likelihood_mode_correct = prefix_likelihoods.argmax(dim=-1).eq(mode_targets).float()
    prior_log = uniform_log_belief(num_modes, tracks)
    prior = prior_log.exp()
    filter_likelihoods = prefix_likelihoods * likelihood_scale
    persistent_before_log, persistent_after_log = run_categorical_filter(
        filter_likelihoods,
        prior_log_belief=prior_log,
        observation_mask=active,
    )
    bayesian_before_log = bayesian_after_log = None
    if bayesian_hazard_rate is not None:
        if not 0.0 <= float(bayesian_hazard_rate) <= 1.0:
            raise ValueError("bayesian_hazard_rate must lie in [0, 1]")
        current = prior_log
        before_values = []
        after_values = []
        for window in range(windows):
            before_values.append(current)
            updated = hazard_belief_update(
                current,
                filter_likelihoods[:, window],
                float(bayesian_hazard_rate),
            )
            current = torch.where(
                active[:, window, None], updated, current
            )
            after_values.append(current)
        bayesian_before_log = torch.stack(before_values, dim=1)
        bayesian_after_log = torch.stack(after_values, dim=1)
    reset_before_log, reset_after_log = lagged_reset_beliefs(
        filter_likelihoods,
        prior_log_belief=prior_log,
        observation_mask=active,
    )
    shuffled_before_log, shuffled_after_log = _shuffled_identity_filter(
        filter_likelihoods,
        active,
        episode_ids,
        entity_slots,
        prior_log,
    )
    uniform = prior[:, None, :].expand(-1, windows, -1)
    has_observation = active.long().cumsum(dim=1).gt(0)
    recurrent_after = torch.where(
        has_observation[..., None], recurrent_after, uniform
    )
    recurrent_before = torch.cat([prior[:, None, :], recurrent_after[:, :-1]], dim=1)
    oracle = F.one_hot(modes, num_classes=num_modes).float()[:, None, :].expand(-1, windows, -1)
    beliefs = {
        "no-belief": (uniform, uniform),
        "reset": (reset_before_log.exp(), reset_after_log.exp()),
        "persistent": (persistent_before_log.exp(), persistent_after_log.exp()),
        "shuffled-id": (shuffled_before_log.exp(), shuffled_after_log.exp()),
        "recurrent": (recurrent_before, recurrent_after),
        "oracle": (oracle, oracle),
    }
    if bayesian_before_log is not None and bayesian_after_log is not None:
        beliefs["bayesian-hazard"] = (
            bayesian_before_log.exp(),
            bayesian_after_log.exp(),
        )

    coordinates = targets.shape[-2] * targets.shape[-1]
    main_mask = active.clone()
    main_mask[:, 0] = False
    window0_mask = active[:, 0]
    results: dict[str, dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []
    for name, (before, after) in beliefs.items():
        belief_metrics = _masked_belief_metrics(
            after, modes, active, confidence_threshold
        )
        decision_metrics = _masked_belief_metrics(
            before, modes, active, confidence_threshold
        )
        predictive_nll_values = -torch.logsumexp(
            before.clamp_min(1e-9).log() + raw_likelihoods,
            dim=-1,
        ) / coordinates
        predicted_mean = belief_weighted_mean(means, before)
        ade, fde = trajectory_errors(
            predicted_mean.reshape(-1, targets.shape[-2], 2),
            targets.reshape(-1, targets.shape[-2], 2),
            dt,
        )
        ade = ade.reshape(tracks, windows)
        fde = fde.reshape(tracks, windows)
        high_mask = main_mask & high
        result = {
            **belief_metrics,
            "decision_mode_nll": decision_metrics["mode_nll"],
            "decision_mode_accuracy": decision_metrics["mode_accuracy"],
            "predictive_trajectory_nll": _masked_mean(predictive_nll_values, main_mask),
            "predictive_trajectory_nll_by_window": _masked_by_window(
                predictive_nll_values, active
            ),
            "trajectory_ade": _masked_mean(ade, main_mask),
            "trajectory_fde": _masked_mean(fde, main_mask),
            "trajectory_ade_by_window": _masked_by_window(ade, active),
            "trajectory_fde_by_window": _masked_by_window(fde, active),
            "high_trajectory_ade": _masked_mean(ade, high_mask),
            "high_trajectory_fde": _masked_mean(fde, high_mask),
            "window0_predictive_trajectory_nll": _masked_mean(
                predictive_nll_values[:, 0], window0_mask
            ),
            "window0_trajectory_ade": _masked_mean(ade[:, 0], window0_mask),
            "window0_trajectory_fde": _masked_mean(fde[:, 0], window0_mask),
            "likelihood_mode_accuracy": _masked_mean(
                likelihood_mode_correct, main_mask
            ),
            "likelihood_mode_accuracy_by_window": _masked_by_window(
                likelihood_mode_correct, active
            ),
            "likelihood_margin_per_coordinate": _masked_mean(
                likelihood_margin, main_mask
            ),
            "likelihood_margin_per_coordinate_by_window": _masked_by_window(
                likelihood_margin, active
            ),
            "likelihood_mode_accuracy_by_true_mode": {
                TRAJECTORY_MODE_NAMES[mode]: _masked_mean(
                    likelihood_mode_correct,
                    main_mask & modes[:, None].eq(mode),
                )
                for mode in range(num_modes)
            },
            "likelihood_margin_by_true_mode": {
                TRAJECTORY_MODE_NAMES[mode]: _masked_mean(
                    likelihood_margin,
                    main_mask & modes[:, None].eq(mode),
                )
                for mode in range(num_modes)
            },
            "main_evaluation_count": int(main_mask.sum()),
            "active_observation_count": int(active.sum()),
        }
        results[name] = result
        decision_true_probability = before.gather(
            dim=-1, index=mode_targets[..., None]
        ).squeeze(-1)
        posterior_true_probability = after.gather(
            dim=-1, index=mode_targets[..., None]
        ).squeeze(-1)
        target_probabilities = F.one_hot(mode_targets, num_classes=num_modes).to(before)
        decision_brier = (before - target_probabilities).square().sum(dim=-1)
        posterior_brier = (after - target_probabilities).square().sum(dim=-1)
        posterior_entropy = -(after.clamp_min(1e-9) * after.clamp_min(1e-9).log()).sum(
            dim=-1
        )
        decision_correct = before.argmax(dim=-1).eq(mode_targets)
        posterior_correct = after.argmax(dim=-1).eq(mode_targets)
        decision_predicted_mode = before.argmax(dim=-1)
        posterior_predicted_mode = after.argmax(dim=-1)
        component_variance = log_stds.mul(2.0).exp()
        predictive_second_moment = belief_weighted_mean(
            component_variance + means.square(), before
        )
        predictive_variance = (
            predictive_second_moment - predicted_mean.square()
        ).clamp_min(1e-9)
        predictive_std = predictive_variance.sqrt()
        target_error = targets - predicted_mean
        interval_1sigma = (
            target_error.abs() <= predictive_std
        ).float().mean(dim=(-2, -1))
        interval_2sigma = (
            target_error.abs() <= 2.0 * predictive_std
        ).float().mean(dim=(-2, -1))
        for track in range(tracks):
            for window in range(windows):
                raw_rows.append(
                    {
                        "split": split_name,
                        "episode_id": int(episode_ids[track]),
                        "entity_id": int(entity_ids[track]),
                        "entity_slot": int(entity_slots[track]),
                        "window": window,
                        "method": name,
                        "mode": int(modes[track]),
                        "active": int(active[track, window]),
                        "main_evaluation": int(main_mask[track, window]),
                        "high_interaction": int(high[track, window]),
                        "predictive_trajectory_nll": float(
                            predictive_nll_values[track, window]
                        ),
                        "trajectory_ade": float(ade[track, window]),
                        "trajectory_fde": float(fde[track, window]),
                        "decision_true_probability": float(
                            decision_true_probability[track, window]
                        ),
                        "decision_brier": float(decision_brier[track, window]),
                        "posterior_brier": float(posterior_brier[track, window]),
                        "decision_confidence": float(
                            before[track, window].max()
                        ),
                        "decision_predicted_mode": int(
                            decision_predicted_mode[track, window]
                        ),
                        "posterior_true_probability": float(
                            posterior_true_probability[track, window]
                        ),
                        "posterior_confidence": float(
                            after[track, window].max()
                        ),
                        "posterior_predicted_mode": int(
                            posterior_predicted_mode[track, window]
                        ),
                        "decision_mode_correct": int(
                            decision_correct[track, window]
                        ),
                        "posterior_mode_correct": int(
                            posterior_correct[track, window]
                        ),
                        "posterior_entropy": float(posterior_entropy[track, window]),
                        "likelihood_mode_correct": int(
                            likelihood_mode_correct[track, window]
                        ),
                        "likelihood_predicted_mode": int(
                            prefix_likelihoods[track, window].argmax()
                        ),
                        "likelihood_margin_per_coordinate": float(
                            likelihood_margin[track, window]
                        ),
                        "predictive_1sigma_coordinate_coverage": float(
                            interval_1sigma[track, window]
                        ),
                        "predictive_2sigma_coordinate_coverage": float(
                            interval_2sigma[track, window]
                        ),
                    }
                )
    return results, raw_rows


def _write_raw_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("raw evaluation rows must not be empty")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train integrated trajectory response and persistent belief.")
    parser.add_argument("--config", default="configs/persistent_trajectory_response.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--response-epochs", type=int, default=None)
    parser.add_argument("--recurrent-epochs", type=int, default=None)
    parser.add_argument("--limit-train-episodes", type=int, default=0)
    parser.add_argument("--limit-val-episodes", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode-discrimination-weight", type=float, default=None)
    parser.add_argument("--mode-temperature", type=float, default=None)
    parser.add_argument("--interaction-residual-weight", type=float, default=None)
    parser.add_argument("--disable-response-enhancements", action="store_true")
    parser.add_argument("--h1-encoder-checkpoint", default=None)
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    dataset_cfg = raw["dataset"]
    model_cfg = raw.get("model", {})
    training_cfg = raw.get("training", {})
    output_dir = Path(args.output_dir or raw.get("output_dir", "experiments/response/persistent_trajectory"))
    if any((output_dir / name).exists() for name in ("response.pt", "recurrent.pt", "metrics.json")):
        raise ValueError("training output already contains results; use a new output directory")
    evaluation_cfg = raw.get("evaluation", {})
    seed = int(args.seed if args.seed is not None else dataset_cfg.get("seed", 2027))
    data_seed = int(dataset_cfg.get("data_seed", seed))
    id_split = str(evaluation_cfg.get("id_split", "val")).lower()
    if id_split not in {"val", "test"}:
        raise ValueError("evaluation.id_split must be 'val' or 'test'")
    _set_seed(seed)
    torch.set_num_threads(int(training_cfg.get("num_threads", 8)))

    if args.limit_train_episodes > 0:
        raw["dataset"]["train_episodes"] = args.limit_train_episodes
    if args.limit_val_episodes > 0:
        raw["dataset"]["val_episodes"] = args.limit_val_episodes
        if "test_episodes" in raw["dataset"]:
            raw["dataset"]["test_episodes"] = args.limit_val_episodes
        raw["dataset"]["ood_episodes"] = args.limit_val_episodes
    train_dataset = _make_dataset(raw, "train", data_seed)
    id_dataset = _make_dataset(raw, id_split, data_seed)
    ood_dataset = _make_dataset(
        raw,
        "ood",
        data_seed,
        action_policy=str(dataset_cfg.get("ood_action_policy", "correlated")),
    )
    loader_kwargs = {
        "batch_size": int(training_cfg.get("batch_size", 32)),
        "num_workers": int(training_cfg.get("num_workers", 0)),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    id_loader = DataLoader(id_dataset, shuffle=False, **loader_kwargs)
    ood_loader = DataLoader(ood_dataset, shuffle=False, **loader_kwargs)
    response_config = PersistentTrajectoryModelConfig(
        history=int(dataset_cfg.get("history", 8)),
        horizon=int(dataset_cfg.get("horizon", 12)),
        num_robots=int(dataset_cfg.get("num_robots", 2)),
        num_modes=len(TRAJECTORY_MODE_NAMES),
        hidden_dim=int(model_cfg.get("hidden_dim", 96)),
        dropout=float(model_cfg.get("dropout", 0.05)),
        dt=float(dataset_cfg.get("dt", 0.4)),
        architecture=str(model_cfg.get("architecture", "legacy")),
        num_heads=int(model_cfg.get("num_heads", 4)),
        graph_layers=int(model_cfg.get("graph_layers", 2)),
        prefix_loss_weight=float(training_cfg.get("prefix_loss_weight", 1.0)),
        geometry_features=not args.disable_response_enhancements and bool(model_cfg.get("geometry_features", False)),
        dynamic_interaction_features=not args.disable_response_enhancements and bool(model_cfg.get("dynamic_interaction_features", False)),
        temporal_pooling="mean" if args.disable_response_enhancements else str(model_cfg.get("temporal_pooling", "mean")),
        candidate_uncertainty=not args.disable_response_enhancements and bool(model_cfg.get("candidate_uncertainty", False)),
        set_interactions=not args.disable_response_enhancements and bool(model_cfg.get("set_interactions", False)),
        zero_init_interaction_residual=not args.disable_response_enhancements and bool(
            model_cfg.get("zero_init_interaction_residual", False)
        ),
    )
    recurrent_config = TrajectoryRecurrentConfig(
        history=response_config.history,
        horizon=response_config.horizon,
        num_robots=response_config.num_robots,
        num_modes=response_config.num_modes,
        hidden_dim=int(model_cfg.get("recurrent_hidden_dim", 64)),
        execute_steps=int(dataset_cfg.get("execute_steps", 3)),
        use_entity_history=bool(model_cfg.get("recurrent_use_entity_history", False)),
        dt=float(dataset_cfg.get("dt", 0.4)),
    )
    response_model = BeliefConditionedTeamResponseModel(response_config).to(args.device)
    transfer_metadata = None
    h1_checkpoint = args.h1_encoder_checkpoint or training_cfg.get("h1_encoder_checkpoint")
    if h1_checkpoint is not None:
        transfer_metadata = initialize_h1_encoder(response_model, Path(str(h1_checkpoint).format(seed=seed)))
        print(f"h1_encoder_transfer={json.dumps(transfer_metadata, sort_keys=True)}", flush=True)
    recurrent_model = TrajectoryRecurrentBelief(recurrent_config).to(args.device)
    default_epochs = int(args.epochs or training_cfg.get("epochs", 25))
    response_epochs = int(
        args.response_epochs
        or args.epochs
        or training_cfg.get("response_epochs", default_epochs)
    )
    recurrent_epochs = int(
        args.recurrent_epochs
        or args.epochs
        or training_cfg.get("recurrent_epochs", default_epochs)
    )
    mode_discrimination_weight = float(
        args.mode_discrimination_weight
        if args.mode_discrimination_weight is not None
        else training_cfg.get("mode_discrimination_weight", 0.0)
    )
    mode_temperature = float(
        args.mode_temperature
        if args.mode_temperature is not None
        else training_cfg.get("mode_temperature", 0.2)
    )
    interaction_residual_weight = float(
        args.interaction_residual_weight
        if args.interaction_residual_weight is not None
        else training_cfg.get("interaction_residual_weight", 0.0)
    )
    if interaction_residual_weight < 0.0:
        raise ValueError("interaction_residual_weight must be non-negative")
    print(
        f"train_episodes={len(train_dataset)} id_episodes={len(id_dataset)} "
        f"id_split={id_split} model_seed={seed} data_seed={data_seed} "
        f"windows={train_dataset.windows} modes={response_config.num_modes} "
        f"response_parameters={sum(p.numel() for p in response_model.parameters())} "
        f"recurrent_parameters={sum(p.numel() for p in recurrent_model.parameters())} "
        f"response_epochs={response_epochs} recurrent_epochs={recurrent_epochs} "
        f"interaction_residual_weight={interaction_residual_weight} "
        f"device={args.device}",
        flush=True,
    )
    _train_response(
        response_model,
        train_loader,
        args.device,
        response_epochs,
        float(training_cfg.get("lr", 3e-4)),
        float(training_cfg.get("weight_decay", 1e-4)),
        mode_discrimination_weight,
        mode_temperature,
        response_config.prefix_loss_weight,
        interaction_residual_weight,
    )
    _train_recurrent(
        recurrent_model,
        train_loader,
        args.device,
        recurrent_epochs,
        float(training_cfg.get("recurrent_lr", 1e-3)),
        float(training_cfg.get("weight_decay", 1e-4)),
    )
    evaluation_kwargs = {
        "device": args.device,
        "dt": float(dataset_cfg.get("dt", 0.4)),
        "likelihood_scale": float(evaluation_cfg.get("likelihood_scale", 0.2)),
        "confidence_threshold": float(evaluation_cfg.get("confidence_threshold", 0.8)),
        "execute_steps": int(dataset_cfg.get("execute_steps", 3)),
    }
    results, id_raw_rows = _evaluate(
        response_model,
        recurrent_model,
        id_loader,
        split_name="id",
        **evaluation_kwargs,
    )
    ood_results, ood_raw_rows = _evaluate(
        response_model,
        recurrent_model,
        ood_loader,
        split_name="ood",
        **evaluation_kwargs,
    )
    output_dir = Path(
        args.output_dir or raw.get("output_dir", "experiments/response/persistent_trajectory")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "ood_metrics.json").write_text(
        json.dumps(ood_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_raw_rows(output_dir / "raw_metrics.csv", id_raw_rows + ood_raw_rows)
    torch.save(
        {
            "model": response_model.state_dict(),
            "model_config": vars(response_config),
            "seed": seed,
            "data_seed": data_seed,
            "id_split": id_split,
            "mode_discrimination_weight": mode_discrimination_weight,
            "mode_temperature": mode_temperature,
            "interaction_residual_weight": interaction_residual_weight,
            "h1_encoder_transfer": transfer_metadata,
        },
        output_dir / "response.pt",
    )
    torch.save(
        {
            "model": recurrent_model.state_dict(),
            "model_config": vars(recurrent_config),
            "seed": seed,
            "data_seed": data_seed,
            "id_split": id_split,
        },
        output_dir / "recurrent.pt",
    )
    (output_dir / "training_config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    (output_dir / "training_metadata.json").write_text(
        json.dumps({"response_model_config": vars(response_config), "h1_encoder_transfer": transfer_metadata,
                    "model_seed": seed, "data_seed": data_seed, "response_epochs": response_epochs,
                    "recurrent_epochs": recurrent_epochs, "id_split": id_split,
                    "mode_discrimination_weight": mode_discrimination_weight, "mode_temperature": mode_temperature,
                    "interaction_residual_weight": interaction_residual_weight},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    for split_name, split_results in (("id", results), ("ood", ood_results)):
        for name, metrics in split_results.items():
            print(
                f"split={split_name} baseline={name} "
                f"traj_nll={metrics['predictive_trajectory_nll']:.5f} "
                f"ade={metrics['trajectory_ade']:.5f} fde={metrics['trajectory_fde']:.5f} "
                f"final_mode_accuracy={metrics['final_mode_accuracy']:.5f} "
                f"windows_to_80={metrics['windows_to_confidence']:.5f}",
                flush=True,
            )
    print(f"saved_results={output_dir}", flush=True)


if __name__ == "__main__":
    main()
