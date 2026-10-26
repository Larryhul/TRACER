from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sage.belief.filter import run_categorical_filter
from sage.experiments.calibrate_persistent_belief import calibration_scores
from sage.experiments.run_sicnav_tracer_loop import _new_episode
from sage.experiments.train_rolling_external_response import (
    batch_data,
    prediction_loss,
    response_group_batches,
)
from sage.response.candidate_ranking import candidate_ranking_metrics, pairwise_ranking_loss
from sage.response.h1_transfer import initialize_h1_encoder, initialize_single_robot_response
from sage.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    mode_conditioned_trajectory_loss,
    trajectory_mode_log_likelihood,
)


MODE_SEMANTICS = (
    "training-quantile bins of persistent simulator preferred speed; "
    "hidden labels used for supervised training and validation only, never planning inputs; "
    "behavioral-parameter proxy, not human-intention ground truth"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def speed_thresholds(data, speeds, num_modes=3):
    identities = sorted(set(zip(data["case"].tolist(), data["human"].tolist())))
    values = np.asarray([speeds[identity] for identity in identities])
    if num_modes < 2:
        raise ValueError("speed bins require at least two modes")
    thresholds = np.quantile(values, np.arange(1, num_modes) / num_modes)
    if not np.isfinite(thresholds).all() or np.any(np.diff(thresholds) <= 0):
        raise ValueError("training identities must support distinct speed bins")
    return thresholds


def attach_modes(data, speeds, thresholds):
    labels = [int(np.searchsorted(thresholds, speeds[int(case), int(human)], side="right"))
              for case, human in zip(data["case"], data["human"])]
    preferred_speed = [speeds[int(case), int(human)] for case, human in zip(data["case"], data["human"])]
    return {**data, "mode": np.asarray(labels, dtype=np.int64), "preferred_speed": np.asarray(preferred_speed, dtype=np.float32)}


def validate_mode_probe_data(data, manifest, native_train_hash, train_cases, val_cases, speed_values):
    if manifest.get("phase") != "train" or manifest.get("source_native_train_sha256") != native_train_hash:
        raise ValueError("mode interventions must come from the matching native training data")
    cases = set(map(int, data["case"]))
    if not cases or not cases <= set(train_cases) or cases & set(val_cases):
        raise ValueError("mode interventions cannot include validation or test cases")
    if not np.allclose(manifest["mode_speed_values"], speed_values, rtol=0, atol=1e-6):
        raise ValueError("mode intervention centers must match train-only model mode coordinates")
    labels = data["mode"]
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= len(speed_values)):
        raise ValueError("mode interventions contain invalid mode labels")
    if not np.allclose(data["preferred_speed"], np.asarray(speed_values)[labels], rtol=0, atol=1e-6):
        raise ValueError("mode labels must match the intervened preferred speeds")


def matched_mode_probe_groups(data, num_modes):
    groups = {}
    for row, identity in enumerate(zip(data["case"], data["window"], data["human"], data["branch"])):
        groups.setdefault(tuple(identity), []).append(row)
    result = []
    for rows in groups.values():
        ordered = np.asarray(rows)[np.argsort(data["mode"][rows])]
        if not np.array_equal(data["mode"][ordered], np.arange(num_modes)):
            raise ValueError("matched mode groups must contain exactly one response per mode")
        for name in ("history", "current", "candidates"):
            if not np.allclose(data[name][ordered], data[name][ordered[:1]], atol=2e-5, rtol=0):
                raise ValueError("matched mode interventions require identical observed state and robot action")
        result.append(ordered)
    return np.stack(result)


def matched_mode_effect_loss(predicted, target, num_modes):
    if predicted.shape != target.shape or predicted.ndim != 3 or len(target) % num_modes:
        raise ValueError("mode effect supervision requires complete mode groups and matching trajectories")
    predicted = predicted.reshape(-1, num_modes, *predicted.shape[1:])
    target = target.reshape_as(predicted)
    difference = predicted - predicted.mean(1, keepdim=True) - target + target.mean(1, keepdim=True)
    return difference.square().mean() + difference[..., 0, :].square().mean()


def recover_training_labels(datasets, external_root):
    """Read hidden parameters only from the disjoint training/calibration resets."""
    speeds = {}
    for case in sorted(set(np.concatenate([data["case"] for data in datasets.values()]).tolist())):
        env, *_ = _new_episode(external_root, case, external_root / "sicnav/configs/env.config",
                              None, "sicnav-unicycle")
        for human_index, human in enumerate(env.humans):
            matching = [data for data in datasets.values() if case in data["case"]]
            for data in matching:
                rows = np.flatnonzero((data["case"] == case) & (data["human"] == human_index) & (data["window"] == 0))
                if not len(rows):
                    raise ValueError("hidden-label recovery requires an observed initial state for every identity")
                if not np.allclose(data["history"][rows, -1], human.get_position(), atol=1e-6):
                    raise ValueError("label recovery reset does not match recorded observed positions")
            speeds[case, human_index] = float(human.v_pref)
    return speeds


def predict(model, data, indices, device, *, true_speed=False):
    batch = batch_data(data, indices, device)
    context = {}
    if model.config.static_geometry_features:
        context["static_segments"] = torch.as_tensor(data["static_segments"][indices, None], device=device).float()
    condition_coordinates = None
    if str(model.config.conditioning).lower() == "continuous" and true_speed:
        condition_coordinates = torch.as_tensor(
            data["preferred_speed"][indices], device=device, dtype=batch.entity_history.dtype
        )[:, None, None]
    output = model(batch.entity_history[:, None], batch.robot_current[:, None],
                   batch.robot_candidates[:, None], robot_mask=batch.robot_mask[:, None],
                   neighbor_history=batch.neighbor_history[:, None],
                   neighbor_mask=batch.neighbor_mask[:, None], include_pair_interactions=False,
                   condition_coordinates=condition_coordinates, **context)
    prediction = {name: output[name][:, 0] for name in ("mean", "log_std")}
    if output.get("preferred_speed_mean") is not None:
        prediction["preferred_speed_mean"] = output["preferred_speed_mean"][:, 0]
    if output.get("mode_gate_logits") is not None:
        prediction["mode_gate_logits"] = output["mode_gate_logits"][:, 0]
    return prediction, batch.target_future_vel


def predict_all_modes(model, data, indices, device):
    batch = batch_data(data, indices, device)
    context = {}
    if model.config.static_geometry_features:
        context["static_segments"] = torch.as_tensor(data["static_segments"][indices, None], device=device).float()
    output = model(batch.entity_history[:, None], batch.robot_current[:, None],
                   batch.robot_candidates[:, None], robot_mask=batch.robot_mask[:, None],
                   neighbor_history=batch.neighbor_history[:, None],
                   neighbor_mask=batch.neighbor_mask[:, None], include_pair_interactions=False, **context)
    prediction = {name: output[name][:, 0] for name in ("mean", "log_std")}
    if output.get("preferred_speed_mean") is not None:
        prediction["preferred_speed_mean"] = output["preferred_speed_mean"][:, 0]
    if output.get("mode_gate_logits") is not None:
        prediction["mode_gate_logits"] = output["mode_gate_logits"][:, 0]
    return prediction, batch.target_future_vel


def response_induced_cost(
    response_mean: torch.Tensor,
    reference_mean: torch.Tensor,
) -> torch.Tensor:
    if response_mean.ndim not in {3, 4} or response_mean.shape[-1] != 2:
        raise ValueError("response_mean must have shape (batch, horizon, 2) or (batch, modes, horizon, 2)")
    try:
        reference_mean = torch.broadcast_to(reference_mean, response_mean.shape)
    except RuntimeError as error:
        raise ValueError("reference_mean must broadcast to response_mean") from error
    return (response_mean - reference_mean).square().sum(dim=-1).mean(dim=-1)


def grouped_candidate_costs(
    data,
    indices,
    prediction,
    target,
    device,
    dt: float,
    comfort_scale: float = 1.0,
    mode_weights: torch.Tensor | None = None,
):
    rows = np.asarray(indices, dtype=np.int64)
    candidates = torch.as_tensor(data["candidates"][rows, 0], device=device, dtype=target.dtype)
    actions = candidates[..., 2:]
    if mode_weights is None:
        mode_weights = torch.ones_like(prediction["mean"][..., 0, 0])
    if mode_weights.shape != prediction["mean"].shape[:2]:
        raise ValueError("mode weights must align with candidate rows and modes")
    mode_weights = mode_weights.to(prediction["mean"])
    if not torch.isfinite(mode_weights).all() or (mode_weights < 0).any() or (mode_weights.sum(-1) <= 0).any():
        raise ValueError("mode weights must be finite, nonnegative, and have positive row sums")
    mode_weights = mode_weights / mode_weights.sum(-1, keepdim=True)
    groups = {}
    for position, row in enumerate(rows):
        key = (int(data["case"][row]), int(data["window"][row]), int(data["human"][row]))
        groups.setdefault(key, []).append(position)
    if not groups:
        raise ValueError("candidate ranking requires at least one response group")
    max_candidates = max(map(len, groups.values()))
    predicted_rows = []
    oracle_rows = []
    masks = []
    for positions in groups.values():
        positions_tensor = torch.as_tensor(positions, device=device, dtype=torch.long)
        count = len(positions)
        padding = max_candidates - count
        if "reference_branch" in data:
            reference_branch = int(data["reference_branch"][rows[positions[0]]])
            reference_mask = torch.as_tensor(
                data["branch"][rows[positions]], device=device, dtype=torch.long
            ) == reference_branch
        elif "branch" in data:
            branch_values = np.asarray(data["branch"][rows[positions]], dtype=np.int64)
            reference_mask = torch.as_tensor(
                branch_values == int(branch_values.max()), device=device, dtype=torch.bool
            )
        else:
            stationary = torch.linalg.vector_norm(actions[positions_tensor], dim=-1).amax(dim=-1) <= 1e-5
            reference_mask = stationary
        if int(reference_mask.sum()) != 1:
            raise ValueError("every response group must contain exactly one stationary reference branch")
        reference_position = positions_tensor[reference_mask][0]
        if torch.linalg.vector_norm(actions[reference_position], dim=-1).max() > 1e-5:
            raise ValueError("reference branch must contain an executed stationary action")
        weights = mode_weights[positions_tensor]
        if not torch.allclose(weights, weights[:1].expand_as(weights)):
            raise ValueError("same-state candidates must use the same decision-time mode weights")
        predicted_means = prediction["mean"][positions_tensor]
        predicted_reference = prediction["mean"][reference_position]
        oracle_means = target[positions_tensor]
        oracle_reference = target[reference_position]
        predicted = (response_induced_cost(predicted_means, predicted_reference) * weights).sum(-1)
        oracle = response_induced_cost(oracle_means, oracle_reference)
        valid_positions = torch.arange(count, device=device) != int(torch.nonzero(reference_mask, as_tuple=False)[0])
        if int(valid_positions.sum()) < 2:
            raise ValueError("candidate ranking requires at least two non-reference branches")
        predicted = predicted.masked_fill(~valid_positions, float("nan"))
        oracle = oracle.masked_fill(~valid_positions, float("nan"))
        predicted_rows.append(torch.cat([predicted, predicted.new_full((padding,), float("nan"))]))
        oracle_rows.append(torch.cat([oracle, oracle.new_full((padding,), float("nan"))]))
        masks.append(torch.cat([
            valid_positions,
            torch.zeros(padding, device=device, dtype=torch.bool),
        ]))
    return torch.stack(predicted_rows), torch.stack(oracle_rows), torch.stack(masks)


def candidate_effect_loss(data, indices, predicted_mean, target):
    rows = np.asarray(indices, dtype=np.int64)
    if predicted_mean.ndim != 3 or target.ndim != 3 or predicted_mean.shape != target.shape:
        raise ValueError("predicted_mean and target must have shape (batch, horizon, 2)")
    groups = {}
    for position, row in enumerate(rows):
        key = (int(data["case"][row]), int(data["window"][row]), int(data["human"][row]))
        groups.setdefault(key, []).append(position)
    losses = []
    for positions in groups.values():
        if len(positions) < 2:
            continue
        position_tensor = torch.as_tensor(positions, device=predicted_mean.device, dtype=torch.long)
        predicted_delta = predicted_mean[position_tensor][:, None] - predicted_mean[position_tensor][None]
        target_delta = target[position_tensor][:, None] - target[position_tensor][None]
        upper = torch.triu(
            torch.ones(len(positions), len(positions), device=predicted_mean.device, dtype=torch.bool),
            diagonal=1,
        )
        losses.append((predicted_delta - target_delta).square()[upper].mean())
    if not losses:
        return predicted_mean.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_candidate_ranking(model, data, batch_size, device, dt, comfort_scale=1.0):
    model.eval()
    rows = np.arange(len(data["target"]), dtype=np.int64)
    predictions = []
    targets = []
    for start in range(0, len(rows), batch_size):
        prediction, target = predict_all_modes(model, data, rows[start:start + batch_size], device)
        predictions.append({name: value for name, value in prediction.items() if name in {"mean", "log_std"}})
        targets.append(target)
    prediction = {name: torch.cat([item[name] for item in predictions], dim=0) for name in ("mean", "log_std")}
    target = torch.cat(targets, dim=0)
    predicted_costs, oracle_costs, valid = grouped_candidate_costs(
        data, rows, prediction, target, device, dt, comfort_scale
    )
    metrics = candidate_ranking_metrics(predicted_costs, oracle_costs, valid_mask=valid)
    labels = torch.as_tensor(data["mode"], device=device, dtype=torch.long)
    conditional_weights = torch.nn.functional.one_hot(labels, model.config.num_modes).to(target)
    conditional_costs, _, _ = grouped_candidate_costs(
        data, rows, prediction, target, device, dt, comfort_scale, mode_weights=conditional_weights
    )
    metrics.update({f"conditional_{name}": value for name, value in candidate_ranking_metrics(
        conditional_costs, oracle_costs, valid_mask=valid
    ).items()})
    return metrics


def conditional_prediction(prediction, labels):
    rows = torch.arange(len(labels), device=labels.device)
    return {name: prediction[name][rows, labels] for name in ("mean", "log_std")}


def prefix_mode_discrimination_loss(prediction, target, labels, temperature, start_step=0, end_step=None):
    start_step = int(start_step)
    end_step = target.shape[1] if end_step is None else int(end_step)
    if not 0 <= start_step < end_step <= target.shape[1]:
        raise ValueError("mode evidence slice must lie inside the response horizon")
    steps = end_step - start_step
    likelihood = trajectory_mode_log_likelihood(
        prediction["mean"][..., start_step:end_step, :],
        prediction["log_std"][..., start_step:end_step, :],
        target[:, start_step:end_step],
    )
    return mode_conditioned_trajectory_loss(
        likelihood[:, None, :], labels, coordinates=2 * steps, temperature=temperature
    )["mode_discrimination"]


def first_prefix_rows(data):
    """Use one observed first step per state, never multiple counterfactual updates."""
    groups = {}
    for index, key in enumerate(zip(data["case"], data["human"], data["window"])):
        groups.setdefault(tuple(map(int, key)), []).append(index)
    rows = []
    for key in sorted(groups):
        indices = groups[key]
        if not np.allclose(data["target"][indices, 0], data["target"][indices[0], 0], atol=1e-6):
            raise ValueError("first-step responses differ across counterfactual branches")
        if "mode" in data and not np.all(data["mode"][indices] == data["mode"][indices[0]]):
            raise ValueError("mode labels must be constant within an observed state")
        factual = [index for index in indices if "selected_branch" in data and data["selected_branch"][index]]
        if "selected_branch" in data and len(factual) != 1:
            raise ValueError("every observed state requires exactly one factual selected branch")
        rows.append(factual[0] if factual else indices[0])
    return np.asarray(rows, dtype=np.int64)


def prefix_likelihood(prediction, target):
    residual = target[:, None, 0] - prediction["mean"][:, :, 0]
    log_std = prediction["log_std"][:, :, 0]
    return -(.5 * residual.square() * (-2 * log_std).exp() + log_std + .5 * np.log(2 * np.pi)).sum(-1)


@torch.no_grad()
def evaluate_factual_modes(model, data, batch_size, device):
    model.eval()
    rows = first_prefix_rows(data)
    confusion = torch.zeros(model.config.num_modes, model.config.num_modes, dtype=torch.long)
    nll, spread, signal_to_noise = 0., 0., 0.
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        prediction, target = predict_all_modes(model, data, selected, device)
        labels = torch.as_tensor(data["mode"][selected], device=device)
        likelihood = prefix_likelihood(prediction, target)
        nll += float(torch.nn.functional.cross_entropy(likelihood, labels, reduction="sum"))
        counts = torch.bincount((labels * model.config.num_modes + likelihood.argmax(-1)).cpu(),
                                minlength=model.config.num_modes ** 2)
        confusion += counts.reshape_as(confusion)
        deviation = prediction["mean"][:, :, 0].std(dim=1, unbiased=False)
        spread += float(deviation.mean(-1).sum())
        noise = prediction["log_std"][:, :, 0].exp().mean(1).norm(dim=-1)
        signal_to_noise += float((deviation.norm(dim=-1) / noise.clamp_min(1e-6)).sum())
    present = confusion.sum(-1) > 0
    return {"factual_mode_nll": nll / len(rows),
            "factual_mode_accuracy": float(confusion.diag().sum() / confusion.sum()),
            "factual_mode_balanced_accuracy": float((confusion.diag()[present] / confusion.sum(-1)[present]).mean()),
            "factual_mean_spread": spread / len(rows),
            "factual_signal_to_noise": signal_to_noise / len(rows),
            "factual_observation_count": len(rows)}


def track_tensors(data, rows, likelihoods):
    tracks = {}
    for position, row in enumerate(rows):
        tracks.setdefault((int(data["case"][row]), int(data["human"][row])), []).append(position)
    longest = max(map(len, tracks.values()))
    values = torch.zeros(len(tracks), longest, likelihoods.shape[-1])
    active = torch.zeros(len(tracks), longest, dtype=torch.bool)
    modes = torch.zeros(len(tracks), dtype=torch.long)
    for track_index, positions in enumerate(tracks.values()):
        ordered = sorted(positions, key=lambda position: int(data["window"][rows[position]]))
        labels = data["mode"][rows[ordered]]
        if not np.all(labels == labels[0]):
            raise ValueError("persistent labels must be constant within a track")
        values[track_index, :len(ordered)] = likelihoods[ordered]
        active[track_index, :len(ordered)] = True
        modes[track_index] = int(labels[0])
    return values, modes, active


@torch.no_grad()
def belief_response_scores(model, data, batch_size, device, scale):
    model.eval()
    rows = first_prefix_rows(data)
    likelihoods = []
    for start in range(0, len(rows), batch_size):
        prediction, target = predict(model, data, rows[start:start + batch_size], device)
        likelihoods.append(prefix_likelihood(prediction, target).cpu())
    values, _, active = track_tensors(data, rows, torch.cat(likelihoods))
    persistent, _ = run_categorical_filter(values * scale, observation_mask=active)
    uniform = torch.full_like(values, -np.log(model.config.num_modes))
    reset = uniform.clone()
    reset[:, 1:] = (values[:, :-1] * scale).log_softmax(-1)
    lookup = {}
    identities = {}
    for row in rows:
        identity = int(data["case"][row]), int(data["human"][row])
        track = identities.setdefault(identity, len(identities))
        window = sum(key[:2] == identity for key in lookup)
        lookup[(*identity, int(data["window"][row]))] = track, window
    totals = {policy: {"mixture_nll": 0., "mixture_ade": 0.} for policy in ("persistent", "reset", "uniform")}
    beliefs = {"persistent": persistent, "reset": reset, "uniform": uniform}
    for start in range(0, len(data["target"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(data["target"])))
        prediction, target = predict(model, data, indices, device)
        residual = target[:, None] - prediction["mean"]
        log_std = prediction["log_std"]
        mode_likelihood = -(.5 * residual.square() * (-2 * log_std).exp() + log_std + .5 * np.log(2 * np.pi)).sum((-2, -1))
        states = [lookup[int(data["case"][row]), int(data["human"][row]), int(data["window"][row])] for row in indices]
        for policy, before in beliefs.items():
            weights = torch.stack([before[track, window] for track, window in states]).to(device)
            nll = -torch.logsumexp(weights + mode_likelihood, -1) / (2 * model.config.horizon)
            mean = (weights.exp()[..., None, None] * prediction["mean"]).sum(1)
            ade = ((mean - target) * model.config.dt).cumsum(1).norm(dim=-1).mean(1)
            totals[policy]["mixture_nll"] += float(nll.sum())
            totals[policy]["mixture_ade"] += float(ade.sum())
    return {policy: {name: value / len(data["target"]) for name, value in metrics.items()}
            for policy, metrics in totals.items()}


@torch.no_grad()
def evaluate(model, data, batch_size, device, mode_evidence_start=0):
    model.eval()
    totals = {name: 0.0 for name in (
        "conditional_nll", "conditional_ade", "prefix_mse", "mode_mean_spread",
        "prefix_mode_nll", "prefix_mode_accuracy", "prefix_mean_spread", "prefix_signal_to_noise",
    )}
    confusion = torch.zeros(model.config.num_modes, model.config.num_modes, dtype=torch.long)
    speed_squared_error = 0.0
    for start in range(0, len(data["target"]), batch_size):
        rows = np.arange(start, min(start + batch_size, len(data["target"])))
        prediction, target = predict_all_modes(model, data, rows, device)
        labels = torch.as_tensor(data["mode"][rows], device=device)
        conditional = conditional_prediction(prediction, labels)
        mode_evidence_start = min(max(int(mode_evidence_start), 0), target.shape[1] - 1)
        prefix_likelihood = trajectory_mode_log_likelihood(
            prediction["mean"][..., mode_evidence_start:, :], prediction["log_std"][..., mode_evidence_start:, :], target[:, mode_evidence_start:]
        )
        prefix_logits = prefix_likelihood / float(2 * (target.shape[1] - mode_evidence_start))
        metrics = {
            "conditional_nll": prediction_loss(conditional, target),
            "conditional_ade": ((conditional["mean"] - target) * model.config.dt).cumsum(1).norm(dim=-1).mean(),
            "prefix_mse": (conditional["mean"][:, 0] - target[:, 0]).square().mean(),
            "mode_mean_spread": prediction["mean"].std(dim=1).mean(),
            "prefix_mode_nll": torch.nn.functional.cross_entropy(prefix_logits, labels),
            "prefix_mode_accuracy": (prefix_logits.argmax(dim=-1) == labels).float().mean(),
            "prefix_mean_spread": prediction["mean"][:, :, mode_evidence_start:].std(dim=1, unbiased=False).mean(),
            "prefix_signal_to_noise": (
                prediction["mean"][:, :, mode_evidence_start:].std(dim=1, unbiased=False).norm(dim=-1).mean(dim=-1)
                / prediction["log_std"][:, :, mode_evidence_start:].exp().mean(dim=(1, 2)).norm(dim=-1).clamp_min(1e-6)
            ).mean(),
        }
        counts = torch.bincount((labels * model.config.num_modes + prefix_logits.argmax(-1)).cpu(), minlength=model.config.num_modes ** 2)
        confusion += counts.reshape(model.config.num_modes, model.config.num_modes)
        if "preferred_speed_mean" in prediction:
            speed_target = torch.as_tensor(data["preferred_speed"][rows], device=device)
            speed_squared_error += float((prediction["preferred_speed_mean"] - speed_target).square().sum())
        for name, value in metrics.items():
            totals[name] += float(value) * len(rows)
    result = {name: value / len(data["target"]) for name, value in totals.items()}
    present = confusion.sum(-1) > 0
    result["prefix_mode_balanced_accuracy"] = float((confusion.diag()[present] / confusion.sum(-1)[present]).mean())
    if model.config.factual_prefix_dynamics:
        result["preferred_speed_rmse"] = (speed_squared_error / len(data["target"])) ** 0.5
    result.update(evaluate_factual_modes(model, data, batch_size, device))
    return result


@torch.no_grad()
def calibrate(model, data, batch_size, device, scales, observation_stride):
    model.eval()
    rows = first_prefix_rows(data)
    likelihoods = []
    for start in range(0, len(rows), batch_size):
        prediction, target = predict_all_modes(model, data, rows[start:start + batch_size], device)
        likelihoods.append(prefix_likelihood(prediction, target).cpu())
    values, modes, active = track_tensors(data, rows, torch.cat(likelihoods))
    records = calibration_scores(values, modes, active, scales)
    selected = min(records, key=lambda row: (row["decision_mode_nll"], row["likelihood_scale"]))
    reset_before = torch.full_like(values, -np.log(model.config.num_modes))
    reset_before[:, 1:] = (values[:, :-1] * selected["likelihood_scale"]).log_softmax(-1)
    mask = active.clone()
    mask[:, 0] = False
    labels = modes[:, None, None].expand(-1, values.shape[1], 1)
    reset = {"decision_mode_nll": float(-reset_before.gather(-1, labels).squeeze(-1)[mask].mean()),
             "decision_mode_accuracy": float((reset_before.argmax(-1) == modes[:, None])[mask].float().mean())}
    return {"phase": "validation", "selection_metric": "before_observation_decision_mode_nll",
            "selected": selected, "records": records, "reset_at_selected_scale": reset,
            "uniform_mode_nll": float(np.log(model.config.num_modes)), "tracks": len(modes),
            "unique_observations": len(rows), "observation_stride": int(observation_stride),
            "boundary": "factual first-step observations at disclosed stride; no repeated counterfactual evidence; validation, not final test"}


def main():
    parser = argparse.ArgumentParser(description="Train a genuine persistent external response mixture on disjoint rolling cases.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder-checkpoint-dir")
    parser.add_argument("--external-root", default="raw_data/external/source_repos/sicnav_official_filtered")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-modes", type=int, choices=(2, 3), default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--static-geometry-features", action="store_true")
    parser.add_argument("--observed-velocity-anchor", action="store_true")
    parser.add_argument("--factual-prefix-dynamics", action="store_true")
    parser.add_argument("--mode-speed-conditioning", action="store_true")
    parser.add_argument("--robot-positions-target-relative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preferred-speed-weight", type=float, default=0.0)
    parser.add_argument("--selection-mode-weight", type=float, default=0.0)
    parser.add_argument("--mode-evidence-start", type=int, default=1)
    parser.add_argument("--continuous-preferred-speed", action="store_true")
    parser.add_argument("--full-response-transfer", action="store_true")
    parser.add_argument("--belief-discrimination-weight", type=float, default=0.0)
    parser.add_argument("--factual-mode-discrimination-weight", type=float, default=0.0)
    parser.add_argument("--factual-selection-weight", type=float, default=0.0)
    parser.add_argument("--belief-discrimination-temperature", type=float, default=1.0)
    parser.add_argument("--candidate-ranking-weight", type=float, default=0.0)
    parser.add_argument("--candidate-ranking-gap-power", type=float, default=0.0)
    parser.add_argument("--candidate-ranking-margin", type=float, default=0.0)
    parser.add_argument("--candidate-ranking-temperature", type=float, default=1.0)
    parser.add_argument("--ranking-selection-weight", type=float, default=0.0)
    parser.add_argument("--candidate-effect-weight", type=float, default=0.0)
    parser.add_argument("--checkpoint-selection", choices=("weighted", "candidate-regret"), default="weighted")
    parser.add_argument("--mode-probe-dir")
    parser.add_argument("--mode-probe-weight", type=float, default=0.0)
    parser.add_argument("--mode-probe-effect-weight", type=float, default=0.0)
    parser.add_argument("--mode-probe-informative-fraction", type=float, default=0.0)
    parser.add_argument("--mode-response-gate", action="store_true")
    parser.add_argument("--mode-gate-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--scales", nargs="+", type=float, default=[.01, .02, .05, .1, .2, .5, 1.0])
    args = parser.parse_args()
    if args.belief_discrimination_weight < 0.0 or args.belief_discrimination_temperature <= 0.0:
        raise ValueError("belief discrimination weight must be nonnegative and temperature positive")
    if any(not np.isfinite(value) or value < 0 for value in
           (args.factual_mode_discrimination_weight, args.factual_selection_weight)):
        raise ValueError("factual mode weights must be finite and nonnegative")
    if (
        args.candidate_ranking_weight < 0.0
        or args.candidate_ranking_gap_power < 0.0
        or args.candidate_ranking_margin < 0.0
        or args.ranking_selection_weight < 0.0
        or args.candidate_effect_weight < 0.0
    ):
        raise ValueError("candidate ranking weights, gap power, margin, and selection weight must be nonnegative")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not np.isfinite(args.mode_probe_weight) or args.mode_probe_weight < 0:
        raise ValueError("mode probe weight must be finite and nonnegative")
    if bool(args.mode_probe_dir) != bool(args.mode_probe_weight):
        raise ValueError("mode probes require both a training directory and positive loss weight")
    if not np.isfinite(args.mode_probe_effect_weight) or args.mode_probe_effect_weight < 0:
        raise ValueError("mode probe effect weight must be finite and nonnegative")
    if not 0 <= args.mode_probe_informative_fraction <= 1:
        raise ValueError("informative mode sampling fraction must lie in [0, 1]")
    if (args.mode_probe_effect_weight or args.mode_probe_informative_fraction) and not args.mode_probe_dir:
        raise ValueError("matched mode losses require training mode probe data")
    if args.mode_response_gate and (not args.mode_probe_dir or args.continuous_preferred_speed):
        raise ValueError("mode response gating requires discrete matched training interventions")
    if not np.isfinite(args.mode_gate_weight) or args.mode_gate_weight <= 0:
        raise ValueError("mode gate supervision weight must be positive and finite")
    if args.checkpoint_selection == "candidate-regret" and not args.candidate_ranking_weight:
        raise ValueError("candidate-regret checkpoint selection requires candidate ranking evaluation")
    if not np.isfinite(args.candidate_ranking_temperature) or args.candidate_ranking_temperature <= 0:
        raise ValueError("candidate ranking temperature must be finite and positive")
    if args.preferred_speed_weight < 0 or args.selection_mode_weight < 0:
        raise ValueError("speed and selection weights must be nonnegative")
    if args.preferred_speed_weight and not args.factual_prefix_dynamics:
        raise ValueError("preferred speed supervision requires factual prefix dynamics")
    if args.full_response_transfer and (not args.encoder_checkpoint_dir or args.static_geometry_features):
        raise ValueError("full response transfer requires a source checkpoint and no extra static encoder")
    torch.set_num_threads(2)
    data_dir, output = Path(args.data_dir), Path(args.output_dir)
    if output.exists():
        raise ValueError("output already exists; use a fresh experiment directory")
    source = json.loads((data_dir / "data_manifest.json").read_text())
    if args.mode_evidence_start < 0 or args.mode_evidence_start >= source["parameters"]["horizon"]:
        raise ValueError("mode evidence start must lie inside the response horizon")
    if source["version"] not in {"rolling-prefix-aligned-v2", "tracer-live-prefix-aligned-v1", "tracer-live-prefix-aligned-v2"}:
        raise ValueError("external belief training requires aligned rolling histories")
    if source["version"].startswith("tracer-live-prefix-aligned-") and source["parameters"]["dt"] != .25:
        raise ValueError("live response data clock must match the external belief model")
    collection_stride = int(source["parameters"]["collection_stride"])
    if collection_stride < 1:
        raise ValueError("calibration protocol requires positive observation spacing")
    datasets = {}
    for split in ("train", "val"):
        with np.load(data_dir / f"{split}.npz") as loaded:
            datasets[split] = dict(loaded)
    if set(datasets["train"]["case"]) & set(datasets["val"]["case"]):
        raise ValueError("training and calibration cases must be disjoint")
    speeds = recover_training_labels(datasets, Path(args.external_root).resolve())
    thresholds = speed_thresholds(datasets["train"], speeds, args.num_modes)
    datasets = {split: attach_modes(data, speeds, thresholds) for split, data in datasets.items()}
    train_identities = sorted(set(zip(datasets["train"]["case"], datasets["train"]["human"])))
    train_speeds = np.asarray([speeds[int(case), int(human)] for case, human in train_identities])
    speed_values = tuple(float(np.median(train_speeds[np.searchsorted(thresholds, train_speeds) == mode])) for mode in range(args.num_modes))
    mode_probes, mode_probe_provenance = None, None
    mode_probe_groups, informative_mode_groups = None, None
    if args.mode_probe_dir:
        probe_dir = Path(args.mode_probe_dir)
        mode_probe_provenance = json.loads((probe_dir / "manifest.json").read_text())
        with np.load(probe_dir / "train.npz") as loaded:
            mode_probes = dict(loaded)
        native_train_hash = source.get("static_context", {}).get("source_data_sha256", {}).get("train", sha256(data_dir / "train.npz"))
        validate_mode_probe_data(mode_probes, mode_probe_provenance, native_train_hash,
                                 set(datasets["train"]["case"]), set(datasets["val"]["case"]), speed_values)
        if mode_probes["target"].shape[1:] != datasets["train"]["target"].shape[1:]:
            raise ValueError("mode probe response horizon must match native training data")
        mode_probe_provenance = {**mode_probe_provenance, "data_sha256": sha256(probe_dir / "train.npz")}
        if args.mode_probe_effect_weight or args.mode_probe_informative_fraction or args.mode_response_gate:
            mode_probe_groups = matched_mode_probe_groups(mode_probes, args.num_modes)
            deviations = mode_probes["target"][mode_probe_groups].std(axis=1)
            informative_mode_groups = np.flatnonzero(np.linalg.norm(deviations[:, 0], axis=-1) > .02)
            if args.mode_probe_informative_fraction and not len(informative_mode_groups):
                raise ValueError("mode probe data has no informative factual response groups")
    for data in datasets.values():
        first_prefix_rows(data)
    output.mkdir(parents=True)
    provenance = {"parameters": vars(args), "source_manifest": source,
                  "source_data_sha256": {split: sha256(data_dir / f"{split}.npz") for split in datasets},
                  "source_code_sha256": {name: sha256(Path(__file__).resolve().parents[1] / name)
                      for name in ("experiments/train_rolling_external_belief.py", "experiments/train_rolling_external_response.py",
                                   "response/persistent_models.py", "response/tracer.py", "response/candidate_ranking.py")},
                  "mode_semantics": MODE_SEMANTICS, "speed_thresholds_train_only": thresholds.tolist(),
                  "training_mode_interventions": mode_probe_provenance,
                  "case_splits": {split: sorted(set(map(int, data["case"]))) for split, data in datasets.items()}}
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    for seed in args.seeds:
        torch.manual_seed(seed)
        uncertainty = {}
        if args.full_response_transfer:
            source_checkpoint = torch.load(Path(args.encoder_checkpoint_dir) / f"joint_seed{seed}.pt", weights_only=False, map_location="cpu")
            uncertainty = {name: source_checkpoint["model_config"][name] for name in ("min_log_std", "max_log_std")}
        config = PersistentTrajectoryModelConfig(
            architecture="tracer", history=source["parameters"]["history"], horizon=source["parameters"]["horizon"],
            num_robots=1, num_modes=args.num_modes, hidden_dim=64, dt=.25, geometry_features=True,
            dynamic_interaction_features=True, temporal_pooling="attention", response_delay_steps=1,
            robot_positions_target_relative=args.robot_positions_target_relative,
            mode_response_gate=args.mode_response_gate,
            conditioning="continuous" if args.continuous_preferred_speed else "discrete",
            static_geometry_features=args.static_geometry_features,
            observed_velocity_anchor=args.observed_velocity_anchor,
            factual_prefix_dynamics=args.factual_prefix_dynamics,
            mode_speed_values=speed_values if (args.mode_speed_conditioning or args.continuous_preferred_speed) else (),
            joint_uncertainty_decoder=args.full_response_transfer, **uncertainty)
        model = BeliefConditionedTeamResponseModel(config).to(args.device)
        transfer = None
        if args.encoder_checkpoint_dir:
            initializer = initialize_single_robot_response if args.full_response_transfer else initialize_h1_encoder
            transfer = initializer(model, Path(args.encoder_checkpoint_dir) / f"joint_seed{seed}.pt")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        generator = np.random.default_rng(seed)
        probe_generator = np.random.default_rng(seed + 104729)
        best, history = float("inf"), []
        checkpoint_path = output / f"seed{seed}" / "response.pt"
        checkpoint_path.parent.mkdir()
        for epoch in range(args.epochs):
            model.train()
            order = generator.permutation(len(datasets["train"]["target"]))
            if args.candidate_ranking_weight or args.candidate_effect_weight:
                batches = response_group_batches(datasets["train"], args.batch_size, generator)
            else:
                batches = (order[start:start + args.batch_size] for start in range(0, len(order), args.batch_size))
            ranking_losses = []
            for rows in batches:
                labels = torch.as_tensor(datasets["train"]["mode"][rows], device=args.device)
                if args.continuous_preferred_speed:
                    conditional, target = predict(model, datasets["train"], rows, args.device, true_speed=True)
                    prediction, _ = predict_all_modes(model, datasets["train"], rows, args.device)
                    conditional = {name: value[:, 0] for name, value in conditional.items() if name in {"mean", "log_std"}}
                else:
                    prediction, target = predict(model, datasets["train"], rows, args.device)
                    conditional = conditional_prediction(prediction, labels)
                prefix = {name: value[:, :1] for name, value in conditional.items()}
                loss = prediction_loss(conditional, target) + prediction_loss(prefix, target[:, :1])
                loss = loss + args.belief_discrimination_weight * prefix_mode_discrimination_loss(
                    prediction, target, labels, args.belief_discrimination_temperature,
                    start_step=args.mode_evidence_start,
                )
                if args.factual_mode_discrimination_weight:
                    loss = loss + args.factual_mode_discrimination_weight * prefix_mode_discrimination_loss(
                        prediction, target, labels, args.belief_discrimination_temperature,
                        start_step=0, end_step=1,
                    )
                if args.preferred_speed_weight:
                    speed_target = torch.as_tensor(datasets["train"]["preferred_speed"][rows], device=args.device)
                    loss = loss + args.preferred_speed_weight * (prediction["preferred_speed_mean"] - speed_target).square().mean()
                if args.candidate_effect_weight:
                    labels_for_effect = torch.as_tensor(datasets["train"]["mode"][rows], device=args.device)
                    effect_mean = prediction["mean"][torch.arange(len(rows), device=args.device), labels_for_effect]
                    loss = loss + args.candidate_effect_weight * candidate_effect_loss(
                        datasets["train"], rows, effect_mean, target
                    )
                if args.candidate_ranking_weight:
                    ranking_weights = torch.nn.functional.one_hot(labels, args.num_modes).to(target)
                    predicted_costs, oracle_costs, valid = grouped_candidate_costs(
                        datasets["train"], rows, prediction, target, args.device, .25, mode_weights=ranking_weights
                    )
                    ranking_loss = pairwise_ranking_loss(
                        predicted_costs,
                        oracle_costs,
                        valid_mask=valid,
                        gap_weight_power=args.candidate_ranking_gap_power,
                        margin=args.candidate_ranking_margin,
                        temperature=args.candidate_ranking_temperature,
                    )
                    loss = loss + args.candidate_ranking_weight * ranking_loss
                    ranking_losses.append(float(ranking_loss.detach()))
                if mode_probes is not None:
                    if mode_probe_groups is None:
                        probe_rows = probe_generator.choice(len(mode_probes["target"]), len(rows), replace=True)
                    else:
                        group_count = max(1, len(rows) // args.num_modes)
                        informative_count = int(group_count * args.mode_probe_informative_fraction)
                        group_indices = probe_generator.integers(len(mode_probe_groups), size=group_count)
                        if informative_count:
                            group_indices[:informative_count] = probe_generator.choice(informative_mode_groups, informative_count)
                        probe_rows = mode_probe_groups[group_indices].reshape(-1)
                    if mode_probe_groups is None:
                        probe_prediction, probe_target = predict_all_modes(model, mode_probes, probe_rows, args.device)
                    else:
                        representative_rows = probe_rows.reshape(-1, args.num_modes)[:, 0]
                        shared_prediction, _ = predict_all_modes(model, mode_probes, representative_rows, args.device)
                        probe_prediction = {name: value.repeat_interleave(args.num_modes, dim=0)
                                            for name, value in shared_prediction.items()}
                        probe_target = torch.as_tensor(mode_probes["target"][probe_rows], device=args.device).float()
                    probe_labels = torch.as_tensor(mode_probes["mode"][probe_rows], device=args.device)
                    probe_conditional = conditional_prediction(probe_prediction, probe_labels)
                    probe_prefix = {name: value[:, :1] for name, value in probe_conditional.items()}
                    probe_loss = prediction_loss(probe_conditional, probe_target) + prediction_loss(probe_prefix, probe_target[:, :1])
                    if mode_probe_groups is None:
                        factual_informative = torch.ones(len(probe_rows), device=args.device, dtype=torch.bool)
                        future_informative = factual_informative
                    else:
                        target_deviation = probe_target.reshape(-1, args.num_modes, *probe_target.shape[1:]).std(1, unbiased=False)
                        if args.mode_response_gate:
                            gate_target = (target_deviation.norm(dim=-1) > .02).float().repeat_interleave(args.num_modes, dim=0)
                            probe_loss = probe_loss + args.mode_gate_weight * torch.nn.functional.binary_cross_entropy_with_logits(
                                probe_prediction["mode_gate_logits"], gate_target,
                            )
                        factual_informative = (target_deviation[:, 0].norm(dim=-1) > .02).repeat_interleave(args.num_modes)
                        future_informative = (target_deviation[:, args.mode_evidence_start:].norm(dim=-1).mean(-1) > .02).repeat_interleave(args.num_modes)
                        probe_loss = probe_loss + args.mode_probe_effect_weight * matched_mode_effect_loss(
                            probe_conditional["mean"], probe_target, args.num_modes
                        )
                    for informative, weight, start_step, end_step in (
                        (factual_informative, args.factual_mode_discrimination_weight, 0, 1),
                        (future_informative, args.belief_discrimination_weight, args.mode_evidence_start, None),
                    ):
                        if weight and informative.any():
                            informative_prediction = {name: probe_prediction[name][informative] for name in ("mean", "log_std")}
                            probe_loss = probe_loss + weight * prefix_mode_discrimination_loss(
                                informative_prediction, probe_target[informative], probe_labels[informative],
                                args.belief_discrimination_temperature, start_step=start_step, end_step=end_step,
                            )
                    loss = loss + args.mode_probe_weight * probe_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            metrics = evaluate(model, datasets["val"], args.batch_size, args.device, args.mode_evidence_start)
            if args.candidate_ranking_weight:
                ranking_metrics = evaluate_candidate_ranking(model, datasets["val"], args.batch_size, args.device, .25)
                metrics.update({f"candidate_{name}": value for name, value in ranking_metrics.items()})
                metrics["candidate_ranking_loss"] = float(np.mean(ranking_losses)) if ranking_losses else 0.0
            metrics["selection_score"] = (
                metrics["conditional_nll"]
                + args.selection_mode_weight * metrics["prefix_mode_nll"]
                + args.ranking_selection_weight * metrics.get("candidate_selected_candidate_regret", 0.0)
                + args.factual_selection_weight * metrics["factual_mode_nll"]
            )
            if args.checkpoint_selection == "candidate-regret":
                metrics["selection_score"] = metrics["candidate_selected_candidate_regret"]
            history.append(metrics)
            print(f"seed={seed} epoch={epoch + 1} {json.dumps(metrics)}", flush=True)
            if metrics["selection_score"] < best:
                best = metrics["selection_score"]
                torch.save({"model": model.state_dict(), "model_config": vars(config), "seed": seed,
                            "selection_split": "val", "best_epoch": epoch + 1, "best_metrics": metrics,
                            "mode_semantics": MODE_SEMANTICS, "speed_thresholds_train_only": thresholds.tolist(),
                            "encoder_transfer": transfer, "provenance": provenance}, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=args.device)
        model.load_state_dict(checkpoint["model"])
        calibration = calibrate(
            model, datasets["val"], args.batch_size, args.device, args.scales, collection_stride
        )
        mixture_scores = belief_response_scores(model, datasets["val"], args.batch_size, args.device,
                                                calibration["selected"]["likelihood_scale"])
        (checkpoint_path.parent / "response_mixture_audit.json").write_text(json.dumps({
            "phase": "validation", "before_observation": True,
            "observation_stride": collection_stride,
            "metrics": mixture_scores,
        }, indent=2) + "\n")
        checkpoint["belief_calibration"] = calibration
        torch.save(checkpoint, checkpoint_path)
        (checkpoint_path.parent / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
        (checkpoint_path.parent / "training.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"seed={seed} calibration={json.dumps(calibration['selected'])}", flush=True)


if __name__ == "__main__":
    main()
