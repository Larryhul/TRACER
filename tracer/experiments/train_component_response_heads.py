import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from tracer.experiments.closed_loop_protocol import prepare_manifest
from tracer.experiments.eval_component_factorial import _local_sources
from tracer.experiments.eval_persistent_closed_loop import _load_model
from tracer.experiments.formal_manifest import sha256_file
from tracer.experiments.train_component_mode_contrast import grouped_prefix_costs, grouped_training_inputs, load_groups
from tracer.response.persistent_models import stationary_reference_candidates


def select_response_head_parameters(model, scope):
    if scope not in {"pair", "mean-heads"}:
        raise ValueError("response training scope must be pair or mean-heads")
    prefixes = ("pair_encoder.", "pair_decoder.", "set_encoder.", "set_decoder.")
    if scope == "mean-heads":
        prefixes += ("base_decoder.", "single_decoder.")
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
        if parameter.requires_grad:
            selected.append(name)
    if not selected:
        raise ValueError("response model has no selected training parameters")
    model.eval()
    return selected


def mean_reference_effect_loss(means, reference, target, reference_target, prefix):
    if means.shape != reference.shape or means.shape != target.shape or means.shape != reference_target.shape or not 1 <= prefix <= means.shape[-2]:
        raise ValueError("candidate/reference/all-mode response shapes and prefix must align")
    errors = {"candidate": means - target, "reference": reference - reference_target,
              "effect": (means - reference) - (target - reference_target)}
    losses = {name: .5 * (error.square().mean() + error[..., :prefix, :].square().mean()) for name, error in errors.items()}
    contrast = means - reference
    truth_contrast = target - reference_target
    metrics = {f"{name}_prefix_mse": float(error[..., :prefix, :].square().mean().detach()) for name, error in errors.items()}
    metrics["effect_mode_averaged_prefix_mse"] = float((contrast.mean(1, keepdim=True)[..., :prefix, :] - truth_contrast[..., :prefix, :]).square().mean().detach())
    return sum(losses.values()), metrics


def response_head_loss(model, batch, prefix, planning=None, cost_weight=0.):
    groups, count = batch["candidates"].shape[:2]
    history = batch["history"][:, None].expand(-1, count, -1, -1).reshape(-1, 1, *batch["history"].shape[-2:])
    current = batch["current"].reshape(-1, 1, *batch["current"].shape[-2:])
    candidates = batch["candidates"].reshape(-1, 1, *batch["candidates"].shape[-3:])
    prediction = model(history, current, candidates)["mean"][:, 0]
    reference = model(history, current, stationary_reference_candidates(current, model.config.horizon))["mean"][:, 0]
    targets = batch["all_mode_responses"].transpose(1, 2).reshape_as(prediction)
    reference_targets = batch["all_mode_references"].transpose(1, 2).reshape_as(reference)
    loss, metrics = mean_reference_effect_loss(prediction, reference, targets, reference_targets, prefix)
    if not np.isfinite(cost_weight) or cost_weight < 0 or (cost_weight and planning is None):
        raise ValueError("finite nonnegative cost weight and matching planning costs required")
    if planning is not None:
        means = prediction.reshape(groups, count, model.config.num_modes, model.config.horizon, 2)
        references = reference.reshape_as(means)
        predicted_costs = grouped_prefix_costs(means, references, batch["current"], batch["candidates"], planning, model.config.dt, prefix)
        cost_mse = (predicted_costs - batch["all_mode_costs"]).square().mean()
        metrics["cost_mse"] = float(cost_mse.detach())
        loss = loss + cost_weight * cost_mse
    return loss, metrics


@torch.no_grad()
def evaluate_response_heads(model, data, prefix, device, planning=None, cost_weight=0.):
    records = []
    for start in range(0, len(data["history"]), 8):
        batch = {name: values[start:start + 8].to(device) for name, values in data.items()}
        loss, metrics = response_head_loss(model, batch, prefix, planning, cost_weight)
        records.append({"loss": float(loss), **metrics, "count": len(batch["history"])})
    total = sum(row["count"] for row in records)
    return {name: sum(row[name] * row["count"] for row in records) / total for name in records[0] if name != "count"}


def main():
    parser = argparse.ArgumentParser(description="Repair all-mode reference/effect consistency with an equal-budget pair-only control")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=["pair", "mean-heads"], required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cost-weight", type=float, default=0.)
    args = parser.parse_args()
    if args.epochs < 1 or not np.isfinite(args.cost_weight) or args.cost_weight < 0:
        raise ValueError("positive epoch budget and finite nonnegative cost weight required")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.cost_weight and (raw["simulation"].get("task_type", "entity_approach") != "entity_approach" or raw["planning"].get("risk_weight", 0.)):
        raise ValueError("cost-aligned response heads currently require entity-approach with zero indicator risk weight")
    checkpoint = Path(args.checkpoint).resolve()
    inputs, origin = grouped_training_inputs(args.rollout_dir, raw)
    model = _load_model(checkpoint, args.device, expected_dt=raw["simulation"]["dt"]).eval()
    trainable = select_response_head_parameters(model, args.scope)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    output = Path(args.output_dir)
    manifest = prepare_manifest(output, {"arguments": vars(args), "config": raw, "data_origin": origin,
        "trainable_parameters": trainable, "objective": "equal sum of candidate/reference/interaction-effect mean squared velocity errors, each half full-horizon and half executed-prefix; optional explicitly weighted all-mode executed-prefix cost regression",
        "selection": "minimum validation objective, epoch zero eligible; no navigation outcome or component contrast selection",
        "scope": "Offline all-mode response labels only. Frozen encoders, mode embeddings and uncertainty; matched data, steps and objective for pair-only versus unlocked mean heads. New checkpoints are not automatically deployed or promoted."},
        [checkpoint, Path(args.config).resolve(), *inputs], _local_sources(), False)
    train = load_groups(Path(args.rollout_dir) / "train_rollouts.pt")
    validation = load_groups(Path(args.rollout_dir) / "validation_rollouts.pt")
    prefix = raw["simulation"]["execute_steps"]
    planning = raw["planning"] if args.cost_weight else None
    baseline = evaluate_response_heads(model, validation, prefix, args.device, planning, args.cost_weight)
    best_loss, best_epoch, best_state = baseline["loss"], 0, initial
    history = [{"epoch": 0, **baseline}]
    print(history[-1], flush=True)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        for indices in torch.randperm(len(train["history"])).split(8):
            batch = {name: values[indices].to(args.device) for name, values in train.items()}
            loss, _ = response_head_loss(model, batch, prefix, planning, args.cost_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.)
            optimizer.step()
        metrics = evaluate_response_heads(model, validation, prefix, args.device, planning, args.cost_weight)
        history.append({"epoch": epoch, **metrics})
        print(history[-1], flush=True)
        if metrics["loss"] < best_loss:
            best_loss, best_epoch = metrics["loss"], epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    changed_parameters = [name for name in initial if not torch.equal(initial[name], best_state[name])]
    if not set(changed_parameters).issubset(trainable):
        raise ValueError("mean-head training changed a frozen parameter")
    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    payload.update({"model": best_state, "response_head_selected_epoch": best_epoch, "response_head_scope": args.scope, "response_head_parent_sha256": sha256_file(checkpoint)})
    torch.save(payload, output / "response.pt")
    changed = [path for group in ("sources", "checkpoints") for path, digest in manifest[group].items() if sha256_file(Path(path)) != digest]
    summary = {"status": "INVALIDATED" if changed else "COMPLETED", "changed_inputs": changed, "changed_parameters": changed_parameters,
               "selected_epoch": best_epoch, "history": history, "checkpoint_sha256": sha256_file(output / "response.pt")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if changed:
        raise RuntimeError("response-head training inputs changed")


if __name__ == "__main__":
    main()
