"""Fit mode-dependent candidate ordering without altering frozen emissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sage.experiments.closed_loop_protocol import prepare_manifest
from sage.experiments.eval_component_factorial import _local_sources
from sage.experiments.eval_persistent_closed_loop import _load_model
from sage.experiments.formal_manifest import sha256_file
from sage.experiments.train_component_residual import freeze_noninteraction_parameters
from sage.response.closed_loop import PlanningCostConfig, expected_candidate_costs
from sage.response.persistent_models import stationary_reference_candidates


def mode_candidate_contrast(costs):
    if costs.ndim != 3:
        raise ValueError("costs must have shape (groups, modes, candidates)")
    candidate_centered = costs - costs.mean(dim=-1, keepdim=True)
    return candidate_centered - candidate_centered.mean(dim=-2, keepdim=True)


def grouped_prefix_costs(means, references, current, candidates, planning, dt, prefix):
    displacement = means[..., :prefix, :].cumsum(dim=-2) * dt
    relative = candidates[..., :prefix, :2]
    distances = torch.linalg.vector_norm(relative[:, :, None] - displacement[:, :, :, None], dim=-1)
    safety = torch.exp(-distances.amin(dim=(-1, -2)) / planning["comfort_scale"])
    human = (means[..., :prefix, :] - references[..., :prefix, :]).square().sum(dim=-1).mean(dim=-1)
    direction = torch.nn.functional.normalize(-current[..., :2], dim=-1)
    progress = ((candidates[..., prefix - 1, :2] - current[..., :2]) * direction).sum(dim=-1).mean(dim=-1)
    task = -progress / (prefix * dt * planning.get("task_speed_scale", 1.7))
    return (planning["task_weight"] * task[..., None] + planning["safety_weight"] * safety + planning["human_weight"] * human).transpose(1, 2)


def executable_candidate_regret(predicted, truth, support, temperature):
    if predicted.shape != truth.shape or predicted.ndim != 3 or support.shape != predicted.shape[::2]:
        raise ValueError("candidate regret requires aligned group/mode/candidate costs and support")
    if not np.isfinite(temperature) or temperature <= 0 or not support.any(-1).all():
        raise ValueError("positive temperature and nonempty executable support are required")
    predicted = torch.cat([predicted, predicted.mean(1, keepdim=True)], dim=1)
    truth = torch.cat([truth, truth.mean(1, keepdim=True)], dim=1)
    support = support[:, None].expand_as(predicted)
    weights = torch.softmax((-predicted / temperature).masked_fill(~support, -torch.inf), dim=-1)
    optimum = truth.masked_fill(~support, torch.inf).amin(-1, keepdim=True)
    return (weights * (truth - optimum)).sum(-1).mean()


@torch.no_grad()
def frozen_executable_support(model, data, device, planning):
    masks = []
    for start in range(0, len(data["history"]), 8):
        history = data["history"][start:start + 8].to(device)
        current = data["current"][start:start + 8].to(device)
        candidates = data["candidates"][start:start + 8].to(device)
        groups, count = candidates.shape[:2]
        flat_history = history[:, None].expand(-1, count, -1, -1).reshape(-1, 1, *history.shape[-2:])
        flat_current = current.reshape(-1, 1, *current.shape[-2:])
        flat_candidates = candidates.reshape(-1, 1, *candidates.shape[-3:])
        output = model(flat_history, flat_current, flat_candidates)
        additive = model(flat_history, flat_current, flat_candidates, include_pair_interactions=False)
        reference = model(flat_history, flat_current, stationary_reference_candidates(flat_current, model.config.horizon))
        costs = expected_candidate_costs(
            output["mean"][:, 0], torch.full((model.config.num_modes,), 1. / model.config.num_modes, device=device),
            flat_current[:, 0], flat_candidates[:, 0], reference_mode_means=reference["mean"][:, 0],
            dt=model.config.dt, config=PlanningCostConfig(**planning),
            mode_log_stds=torch.maximum(output["log_std"], additive["log_std"])[:, 0],
        )
        mask = costs["safe_mask"].reshape(groups, count)
        missing = ~mask.any(-1)
        fallback = costs["violation_probability"].reshape(groups, count).argmin(-1)
        mask[missing, fallback[missing]] = True
        masks.append(mask.cpu())
    return torch.cat(masks)


def grouped_loss(model, batch, planning, prefix, contrast_weight, regret_weight=0., regret_temperature=.05):
    history, current, candidates = batch["history"], batch["current"], batch["candidates"]
    groups, count = candidates.shape[:2]
    expanded_history = history[:, None].expand(-1, count, -1, -1).reshape(-1, 1, *history.shape[-2:])
    flat_current = current.reshape(-1, 1, *current.shape[-2:])
    flat_candidates = candidates.reshape(-1, 1, *candidates.shape[-3:])
    output = model(expanded_history, flat_current, flat_candidates)
    with torch.no_grad():
        reference = model(expanded_history, flat_current, stationary_reference_candidates(flat_current, model.config.horizon))
    means = output["mean"][:, 0].reshape(groups, count, model.config.num_modes, model.config.horizon, 2)
    references = reference["mean"][:, 0].reshape_as(means)
    target = batch["all_mode_responses"].transpose(1, 2)
    target_reference = batch["all_mode_references"].transpose(1, 2)
    deviations = output["log_std"][:, 0].reshape_as(means).exp().detach().clamp_min(.05)
    normalized = ((means - target) / deviations).square()
    fit = normalized.mean() + normalized[..., :prefix, :].mean()
    effects = ((means - references)[..., :prefix, :] - (target - target_reference)[..., :prefix, :]).square().mean()
    predicted = grouped_prefix_costs(means, references, current, candidates, planning, model.config.dt, prefix)
    truth = batch["all_mode_costs"]
    cost_error = (predicted - truth).square().mean()
    contrast = (mode_candidate_contrast(predicted) - mode_candidate_contrast(truth)).square().mean()
    regret = predicted.new_zeros(())
    if regret_weight > 0:
        regret = executable_candidate_regret(predicted, truth, batch["admissible_mask"], regret_temperature)
    total = fit + 8. * effects + 64. * cost_error + contrast_weight * contrast + regret_weight * regret
    return total, {"response_loss": float(fit.detach()), "cost_mse": float(cost_error.detach()), "mode_contrast_mse": float(contrast.detach()), "soft_candidate_regret": float(regret.detach())}


def load_groups(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    names = ("history", "current", "candidates", "all_mode_responses", "all_mode_references", "all_mode_costs")
    if any(name not in raw for name in names):
        raise ValueError("all-mode grouped supervision is required")
    return {name: raw[name].flatten(0, 1) for name in names}


def grouped_training_inputs(source, raw):
    source = Path(source).resolve()
    files = [source / "train_rollouts.pt", source / "validation_rollouts.pt"]
    manifest_path = source / "run_manifest.json"
    if manifest_path.exists():
        origin = json.loads(manifest_path.read_text())["config"]
        if not origin["arguments"].get("all_mode_supervision") or origin["config"] != raw:
            raise ValueError("grouped rollout protocol mismatch")
        return [*files, manifest_path], origin["arguments"]
    manifest_path = source.parent / "run_manifest.json"
    summary_path = source.parent / "summary.json"
    origin = json.loads(manifest_path.read_text())["config"]
    summary = json.loads(summary_path.read_text())
    if origin.get("schema") == "component-candidate-relabel-v1":
        if origin["config"] != raw or not origin.get("all_mode_supervision") or source.name not in {"original", "completed"}:
            raise ValueError("candidate relabeling training protocol mismatch")
        if summary.get("schema") != origin["schema"] or summary.get("status") != "COMPLETED" or summary.get("changed_inputs"):
            raise ValueError("completed unchanged candidate relabeling required")
        for path in files:
            if summary["trajectory_sha256"].get(str(path)) != sha256_file(path):
                raise ValueError("candidate relabeling trajectory hash mismatch")
        if sha256_file(files[1]) != summary["shared_validation_sha256"]:
            raise ValueError("candidate relabeling controls require shared validation labels")
        return [*files, manifest_path, summary_path], {
            **origin["source_arguments"], "collection_profile": source.name, "all_mode_supervision": True,
            "state_scope": origin["state_scope"], "label_scope": origin["label_scope"],
            "validation_context_repetitions": summary["validation_context_repetitions"],
            "validation_independent_scenes": summary["validation_independent_scenes"],
        }
    if origin.get("schema") != "component-policy-coverage-v1" or not origin.get("all_mode_supervision") or origin["config"] != raw:
        raise ValueError("grouped coverage protocol mismatch")
    if source.name not in {"uniform", "factorial"} or summary.get("status") != "COMPLETED" or summary.get("changed_inputs"):
        raise ValueError("completed policy coverage collection required")
    for path in files:
        if summary["trajectory_sha256"].get(str(path)) != sha256_file(path):
            raise ValueError("grouped coverage trajectory hash mismatch")
    if sha256_file(files[1]) != summary["shared_validation_sha256"]:
        raise ValueError("coverage variants must share validation labels")
    return [*files, manifest_path, summary_path], {**origin["arguments"], "collection_profile": source.name, "all_mode_supervision": True}


@torch.no_grad()
def evaluate(model, data, device, planning, prefix, contrast_weight, regret_weight=0., regret_temperature=.05):
    scores = []
    for start in range(0, len(data["history"]), 8):
        batch = {name: values[start:start + 8].to(device) for name, values in data.items()}
        loss, metrics = grouped_loss(model, batch, planning, prefix, contrast_weight, regret_weight, regret_temperature)
        scores.append({"loss": float(loss), **metrics})
    return {name: float(np.mean([row[name] for row in scores])) for name in scores[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--contrast-weight", type=float, default=512.)
    parser.add_argument("--regret-weight", type=float, default=0.)
    parser.add_argument("--regret-temperature", type=float, default=.05)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if any(not np.isfinite(value) or value < 0 for value in [args.contrast_weight, args.regret_weight]) or not np.isfinite(args.regret_temperature) or args.regret_temperature <= 0:
        raise ValueError("finite nonnegative loss weights and positive regret temperature are required")
    torch.set_num_threads(4)
    torch.manual_seed(2027)
    raw = yaml.safe_load(Path(args.config).read_text())
    planning = raw["planning"]
    if planning.get("risk_weight", 0.) or planning.get("smoothness_weight", 0.) or raw["simulation"].get("task_type", "entity_approach") != "entity_approach":
        raise ValueError("grouped objective currently supports the fixed entity-approach cost only")
    checkpoint = Path(args.checkpoint).resolve()
    model = _load_model(checkpoint, args.device)
    if model.config.dt != raw["simulation"]["dt"]:
        raise ValueError("checkpoint timestep mismatch")
    trainable = freeze_noninteraction_parameters(model)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    source = Path(args.rollout_dir)
    files, source_arguments = grouped_training_inputs(source, raw)
    output = Path(args.output_dir)
    manifest = prepare_manifest(output, {"arguments": vars(args), "config": raw, "source_arguments": source_arguments, "trainable": trainable,
        "selection": "validation grouped loss, epoch zero eligible; no test outcomes"}, [checkpoint, *files], _local_sources(), False)
    train, validation = load_groups(files[0]), load_groups(files[1])
    if args.regret_weight:
        for data in [train, validation]:
            data["admissible_mask"] = frozen_executable_support(model, data, args.device, planning)
    prefix = raw["simulation"]["execute_steps"]
    first = evaluate(model, validation, args.device, planning, prefix, args.contrast_weight, args.regret_weight, args.regret_temperature)
    best_loss, best_epoch, best_state = first["loss"], 0, initial
    history = [{"epoch": 0, **first}]
    print(history[-1], flush=True)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        for indices in torch.randperm(len(train["history"])).split(8):
            batch = {name: values[indices].to(args.device) for name, values in train.items()}
            loss, _ = grouped_loss(model, batch, planning, prefix, args.contrast_weight, args.regret_weight, args.regret_temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.)
            optimizer.step()
        metrics = evaluate(model, validation, args.device, planning, prefix, args.contrast_weight, args.regret_weight, args.regret_temperature)
        history.append({"epoch": epoch, **metrics})
        if metrics["loss"] < best_loss:
            best_loss, best_epoch = metrics["loss"], epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(history[-1], flush=True)
    changed_parameters = [name for name in initial if not torch.equal(initial[name], best_state[name])]
    if not set(changed_parameters).issubset(trainable):
        raise ValueError("grouped training changed frozen parameters")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.update({"model": best_state, "mode_contrast_selected_epoch": best_epoch, "mode_contrast_source": str(checkpoint)})
    torch.save(payload, output / "response.pt")
    changed = [path for group in ("sources", "checkpoints") for path, digest in manifest[group].items() if sha256_file(Path(path)) != digest]
    result = {"status": "COMPLETED" if not changed else "INVALIDATED", "changed_inputs": changed, "selected_epoch": best_epoch,
              "changed_parameters": changed_parameters, "history": history}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if changed:
        raise RuntimeError("grouped training inputs changed")


if __name__ == "__main__":
    main()
