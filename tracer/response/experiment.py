from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch
from torch.nn.utils import clip_grad_norm_

from .data import build_entity_response_batch, shuffle_robot_candidates
from .factorial import factorial_second_difference
from .metrics import gaussian_nll_per_sample, response_metrics
from .models import TeamResponsePredictor


def train_epoch(
    model: TeamResponsePredictor,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    num_robots: int,
    interaction_radius: float,
    max_batches: int = 0,
    trainable_module_names: tuple[str, ...] = (),
    interaction_loss_weight: float = 0.0,
    interventions_per_scene: int = 1,
) -> float:
    # A nested residual stage must keep the fitted base path deterministic.
    # Dropout remains active only in the residual modules being optimized.
    if trainable_module_names:
        model.eval()
        for module_name in trainable_module_names:
            getattr(model, module_name).train()
    else:
        model.train()
    if interaction_loss_weight < 0.0:
        raise ValueError("interaction_loss_weight must be non-negative")
    if interaction_loss_weight > 0.0 and interventions_per_scene != 4:
        raise ValueError("interaction auxiliary loss requires four interventions per scene")
    if interaction_loss_weight > 0.0:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
    loss_sum = 0.0
    sample_count = 0
    for batch_idx, scene in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        response = build_entity_response_batch(
            scene.to(device),
            num_robots=num_robots,
            interaction_radius=interaction_radius,
            robot_actions=scene.robot_actions,
        )
        prediction = model(response)
        per_sample = gaussian_nll_per_sample(
            prediction["mean"], prediction["log_std"], response.target_future_vel
        )
        loss = per_sample.mean()
        if interaction_loss_weight > 0.0:
            if scene.batch_size % interventions_per_scene != 0:
                raise ValueError(
                    "factorial batches must contain complete intervention groups"
                )
            if response.num_samples % scene.batch_size != 0:
                raise ValueError(
                    "factorial response rows must be evenly divisible by scene count"
                )
            groups = scene.batch_size // interventions_per_scene
            entities_per_intervention = response.num_samples // scene.batch_size
            structured_shape = (
                groups,
                interventions_per_scene,
                entities_per_intervention,
                response.target_future_vel.shape[1],
                response.target_future_vel.shape[2],
            )
            predicted_interaction = factorial_second_difference(
                prediction["mean"].reshape(structured_shape), dim=1
            )
            target_interaction = factorial_second_difference(
                response.target_future_vel.reshape(structured_shape), dim=1
            )
            loss = loss + interaction_loss_weight * torch.nn.functional.mse_loss(
                predicted_interaction, target_interaction
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += float(per_sample.detach().sum())
        sample_count += response.num_samples
    if sample_count == 0:
        raise RuntimeError("training loader produced no entity-response samples")
    return loss_sum / sample_count


@torch.no_grad()
def evaluate_model(
    model: TeamResponsePredictor,
    loader: Iterable,
    device: torch.device | str,
    num_robots: int,
    interaction_radius: float,
    shuffle_candidates: bool = False,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    high_count = 0
    for batch_idx, scene in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        response = build_entity_response_batch(
            scene.to(device),
            num_robots=num_robots,
            interaction_radius=interaction_radius,
            robot_actions=scene.robot_actions,
        )
        if shuffle_candidates:
            response = shuffle_robot_candidates(response)
        prediction = model(response)
        metrics = response_metrics(
            prediction["mean"],
            prediction["log_std"],
            response.target_future_vel,
            response.dt,
        )
        for name, values in metrics.items():
            totals[name] += float(values.sum())
            if response.high_interaction.any():
                totals[f"high_{name}"] += float(values[response.high_interaction].sum())
        count += response.num_samples
        high_count += int(response.high_interaction.sum())
    if count == 0:
        raise RuntimeError("evaluation loader produced no entity-response samples")
    result = {name: totals[name] / count for name in ("nll", "ade", "fde")}
    for name in ("nll", "ade", "fde"):
        result[f"high_{name}"] = totals[f"high_{name}"] / high_count if high_count else float("nan")
    result["num_samples"] = float(count)
    result["num_high_interaction"] = float(high_count)
    return result
