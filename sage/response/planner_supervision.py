from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch

from sage.response.candidate_ranking import pairwise_ranking_loss
from sage.response.closed_loop import expected_candidate_costs
from sage.response.planning_support import aggregate_entity_costs


def add_action_conditioned_uncertainty(model):
    from sage.response.persistent_models import BeliefConditionedTeamResponseModel

    if model.config.candidate_uncertainty or model.config.joint_uncertainty_decoder:
        raise ValueError("uncertainty upgrade requires the original action-independent variance decoder")
    upgraded = BeliefConditionedTeamResponseModel(replace(model.config, candidate_uncertainty=True))
    result = upgraded.load_state_dict(model.state_dict(), strict=False)
    if result.unexpected_keys or any(not name.startswith("candidate_std_decoder.") for name in result.missing_keys):
        raise ValueError("unexpected parameter changes in uncertainty upgrade")
    torch.nn.init.zeros_(upgraded.candidate_std_decoder[-1].weight)
    torch.nn.init.zeros_(upgraded.candidate_std_decoder[-1].bias)
    return upgraded.to(next(model.parameters()).device).eval()


def configure_planner_finetuning(model, preserve_factual_prefix=False):
    if preserve_factual_prefix and (
        model.config.architecture != "tracer" or not model.config.factual_prefix_dynamics
        or model.config.mode_response_gate or model.config.response_delay_steps < 1
    ):
        raise ValueError("prefix-preserving tuning requires the separate factual prefix dynamics without mode gating")
    allowed = ("tracer_encoder.candidate_encoder.", "base_decoder.", "single_decoder.", "log_std_decoder.", "candidate_std_decoder.")
    names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not preserve_factual_prefix or name.startswith(allowed))
        if parameter.requires_grad:
            names.append(name)
    if not names:
        raise ValueError("planner fine-tuning has no trainable parameters")
    return names


def prefix_equivalence(snapshots, execute_steps):
    if execute_steps < 1 or execute_steps > snapshots.shape[1]:
        raise ValueError("executed prefix must lie inside the recorded horizon")
    prefixes = np.asarray(snapshots)[:, :execute_steps, 0, :5].reshape(len(snapshots), -1)
    return np.isclose(prefixes[:, None], prefixes[None], atol=1e-6, rtol=0).all(-1)


def planner_scores(prediction, group, planning, dt, weights, risk_gradient_temperature=None, return_components=False):
    means, log_stds, references, current, candidates = prediction
    direction = torch.as_tensor(group["direction"], device=means.device, dtype=means.dtype)
    active = torch.as_tensor(group["response_valid_steps"], device=means.device, dtype=torch.bool)
    costs = [expected_candidate_costs(
        means[human], weights[human], current[human], candidates[human], references[human], dt,
        planning, log_stds[human], task_direction=direction, response_valid_steps=active,
        risk_gradient_temperature=risk_gradient_temperature,
    ) for human in range(len(means))]
    total, _ = aggregate_entity_costs(
        costs, planning.task_weight, planning.safety_weight, planning.human_weight,
        planning.smoothness_weight, planning.risk_weight, planning.crowd_aggregation,
    )
    total = total + torch.as_tensor(group["penalty"], device=means.device, dtype=means.dtype)
    return (total, costs) if return_components else total


def entity_risk_calibration_loss(costs, factual_entity_risk, eligible):
    predicted = torch.stack([cost["violation_probability"] for cost in costs])
    target = torch.as_tensor(factual_entity_risk, device=predicted.device, dtype=predicted.dtype)
    if target.shape != predicted.shape or not torch.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("risk supervision must align with humans and candidates in [0, 1]")
    if eligible.shape != predicted.shape[1:] or not eligible.any():
        raise ValueError("risk supervision requires supported candidates")
    return (predicted[:, eligible] - target[:, eligible]).square().mean()


def top_prefix_ranking_loss(scores, factual, eligible, same_prefix, temperature=.05, top_candidates=8):
    if top_candidates < 1:
        raise ValueError("top_candidates must be positive")
    support = eligible & torch.isfinite(scores) & torch.isfinite(factual)
    if not support.any():
        raise ValueError("ranking requires finite supported candidates")
    valid_indices = torch.where(support)[0]
    count = min(top_candidates, len(valid_indices))
    oracle_top = valid_indices[torch.topk(factual[valid_indices], count, largest=False).indices]
    predicted_top = valid_indices[torch.topk(scores[valid_indices].detach(), count, largest=False).indices]
    relevant = torch.zeros_like(support)
    relevant[oracle_top] = True
    relevant[predicted_top] = True
    pair_weights = (relevant[:, None] & relevant[None] & ~same_prefix).to(scores.dtype)
    return pairwise_ranking_loss(
        scores[None], factual[None], valid_mask=support[None], pair_weights=pair_weights[None],
        gap_weight_power=1.0, temperature=temperature,
    )


def planner_gap_calibration_loss(scores, factual, eligible, same_prefix, top_candidates=8):
    if top_candidates < 1:
        raise ValueError("top_candidates must be positive")
    support = eligible & torch.isfinite(scores) & torch.isfinite(factual)
    if not support.any():
        raise ValueError("gap calibration requires finite supported candidates")
    valid = torch.where(support)[0]
    count = min(top_candidates, len(valid))
    relevant = torch.zeros_like(support)
    relevant[valid[torch.topk(scores[valid].detach(), count, largest=False).indices]] = True
    relevant[valid[torch.topk(factual[valid], count, largest=False).indices]] = True
    pairs = torch.triu(relevant[:, None] & relevant[None] & ~same_prefix, diagonal=1)
    if not pairs.any():
        return scores[support].sum() * 0.0
    predicted_gap = scores[:, None] - scores[None]
    factual_gap = factual[:, None] - factual[None]
    return torch.nn.functional.smooth_l1_loss(predicted_gap[pairs], factual_gap[pairs], beta=.1)


def expected_prefix_regret_loss(scores, factual, eligible, same_prefix, temperature=.05):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("prefix regret temperature must be finite and positive")
    support = eligible & torch.isfinite(scores) & torch.isfinite(factual)
    if not support.any():
        raise ValueError("prefix regret requires supported candidates")
    remaining = support.clone()
    model_costs, factual_costs = [], []
    while remaining.any():
        index = int(torch.where(remaining)[0][0])
        members = support & same_prefix[index]
        if not members[index]:
            raise ValueError("prefix equivalence must be reflexive")
        model_costs.append(scores[members].min())
        factual_costs.append(factual[members].min())
        remaining = remaining & ~members
    model_costs = torch.stack(model_costs)
    factual_costs = torch.stack(factual_costs)
    regret = factual_costs - factual_costs.min()
    return (torch.softmax(-model_costs / temperature, dim=0) * regret).sum()


def response_supervision(prediction, group):
    means, log_stds, references, _, _ = prediction
    device = means.device
    labels = torch.as_tensor(group["labels"], device=device, dtype=torch.long)
    target = torch.as_tensor(group["target"], device=device, dtype=means.dtype)
    reference_target = torch.as_tensor(group["reference_target"], device=device, dtype=means.dtype)
    active = torch.as_tensor(group["target_active"], device=device, dtype=torch.bool)
    human_ids = torch.arange(len(means), device=device)
    conditional_mean = means[human_ids, :, labels]
    conditional_std = log_stds[human_ids, :, labels]
    conditional_reference = references[human_ids, :, labels]
    valid = active[None, :, :, None].expand_as(target)
    error = conditional_mean - target
    nll = (.5 * error.square() * (-2 * conditional_std).exp() + conditional_std + .5 * math.log(2 * math.pi))[valid].mean()
    reference_error = conditional_reference - reference_target[:, None]
    reference_active = torch.as_tensor(group["reference_active"], device=device, dtype=torch.bool)
    reference_valid = reference_active[None, None, :, None].expand_as(reference_error)
    reference_mse = reference_error.square()[reference_valid].mean()
    effect_error = error - reference_error
    effect_valid = valid & reference_valid
    effect_mse = effect_error.square()[effect_valid].mean()
    ade = (error * float(group["dt"])).cumsum(-2).norm(dim=-1)[valid[..., 0]].mean()
    return {"nll": nll, "reference_mse": reference_mse, "effect_mse": effect_mse, "ade": ade}


def group_regret(scores, factual, eligible, same_prefix):
    support = eligible & torch.isfinite(scores) & torch.isfinite(factual)
    if not support.any():
        raise ValueError("regret requires finite supported candidates")
    selected = int(torch.where(support, scores, torch.inf).argmin())
    oracle = factual[support].min()
    same = support & same_prefix[selected]
    return {
        "regret": float(factual[selected] - oracle),
        "prefix_regret": float(factual[same].min() - oracle),
        "selected": selected,
    }
