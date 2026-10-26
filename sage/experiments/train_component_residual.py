"""Repair only the joint residual on independent simulator training rollouts."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from sage.experiments.closed_loop_protocol import prepare_manifest
from sage.experiments.eval_component_factorial import FACTORIAL_POLICIES, _local_sources
from sage.experiments.eval_persistent_closed_loop import _evaluate_policy, _load_model, _realized_total_cost
from sage.experiments.finetune_h3_response_ranking import _adapter_from_config, _collect_rollout_batch, _make_spec
from sage.experiments.formal_manifest import sha256_file
from sage.response.closed_loop import PlanningCostConfig, expected_candidate_costs
from sage.response.persistent_models import stationary_reference_candidates


def freeze_noninteraction_parameters(model) -> list[str]:
    prefixes = ("pair_encoder.", "pair_decoder.", "set_encoder.", "set_decoder.")
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
        if parameter.requires_grad:
            selected.append(name)
    if not selected:
        raise ValueError("model has no trainable joint residual parameters")
    model.eval()
    return selected


def flatten_rollouts(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    episodes, windows, count = batch["candidates"].shape[:3]
    if "all_mode_responses" in batch:
        modes = batch["all_mode_responses"].shape[2]
        history = batch["history"][:, :, None, None].expand(-1, -1, modes, count, -1, -1)
        current = batch["current"][:, :, None].expand(-1, -1, modes, -1, -1, -1)
        candidates = batch["candidates"][:, :, None].expand(-1, -1, modes, -1, -1, -1, -1)
        return {
            "history": history.reshape(-1, 1, *history.shape[-2:]),
            "current": current.reshape(-1, 1, *current.shape[-2:]),
            "candidates": candidates.reshape(-1, 1, *candidates.shape[-3:]),
            "target": batch["all_mode_responses"].reshape(-1, *batch["all_mode_responses"].shape[-2:]),
            "reference_target": batch["all_mode_references"].reshape(-1, *batch["all_mode_references"].shape[-2:]),
            "true_cost": batch["all_mode_costs"].reshape(-1),
            "mode": torch.arange(modes)[None, None, :, None].expand(episodes, windows, -1, count).reshape(-1),
        }
    history = batch["history"][:, :, None].expand(-1, -1, count, -1, -1)
    result = {
        "history": history.reshape(-1, 1, *history.shape[-2:]),
        "current": batch["current"].reshape(-1, 1, *batch["current"].shape[-2:]),
        "candidates": batch["candidates"].reshape(-1, 1, *batch["candidates"].shape[-3:]),
        "target": batch["responses"].reshape(-1, *batch["responses"].shape[-2:]),
        "mode": batch["mode"][:, None, None].expand(episodes, windows, count).reshape(-1),
    }
    if "reference_responses" in batch:
        result["reference_target"] = batch["reference_responses"].reshape(-1, *batch["reference_responses"].shape[-2:])
    if "true_cost" in batch:
        result["true_cost"] = batch["true_cost"].reshape(-1)
    return result


@torch.no_grad()
def collect_uniform_rollouts(raw, adapter, model, episodes: int, device: str, all_mode_supervision: bool = False):
    spec = _make_spec(raw, "id")
    planning = PlanningCostConfig(**raw["planning"])
    trace = []
    _evaluate_policy(
        model, FACTORIAL_POLICIES[0], spec, episodes, device, planning, 0.05,
        candidate_adapter=adapter, captured_reference_trace=trace,
        evaluation_mode="same_state", response_gate_min=1.0, response_gate_max=1.0,
        prefix_objective=True,
        retain_previous_plan=bool(raw.get("candidate_source", {}).get("retain_previous_plan", False)),
    )
    return trace_to_rollouts(trace, spec, planning, model.config.num_modes, all_mode_supervision)


@torch.no_grad()
def trace_to_rollouts(trace, spec, planning, num_modes, all_mode_supervision=False):
    if len(trace) != spec.windows or not trace or not trace[0]:
        raise ValueError("factual rollout trace must cover nonempty windows and episodes")
    episodes = len(trace[0])
    if any(len(window) != episodes or any(record is None for record in window) for window in trace):
        raise ValueError("factual rollout trace episode slots must align")
    names = ("history", "current", "candidates", "responses", "reference_responses", "true_cost")
    if all_mode_supervision:
        names += ("all_mode_responses", "all_mode_references", "all_mode_costs")
    records = {name: [[] for _ in range(episodes)] for name in names}
    for window in trace:
        for episode, record in enumerate(window):
            simulator, pool = record.simulator, record.pool
            current, candidates = pool.robot_current.numpy(), pool.robot_candidates.numpy()
            rng_before = copy.deepcopy(simulator.rng.bit_generator.state)
            noise = simulator.rng.normal(0.0, spec.response_noise, size=(spec.horizon, 2)).astype(np.float32)
            simulator.rng.bit_generator.state = rng_before
            reference_candidates = stationary_reference_candidates(pool.robot_current, spec.horizon).numpy()
            responses = simulator._response_batch(current, candidates, noise=noise)
            references = simulator._response_batch(current, reference_candidates, noise=noise)
            outcomes = simulator.counterfactuals(pool, planning)
            torch.testing.assert_close(torch.from_numpy(responses[record.executed_candidate]), outcomes[record.executed_candidate].response, rtol=0, atol=0)
            values = {
                "history": simulator.history.copy(), "current": current.copy(),
                "candidates": candidates.copy(), "responses": responses,
                "reference_responses": references,
                "true_cost": np.asarray([_realized_total_cost(outcome, spec, planning) for outcome in outcomes], dtype=np.float32),
            }
            if all_mode_supervision:
                mode_responses, mode_references, mode_costs = [], [], []
                factual_mode = simulator.mode
                try:
                    for mode in range(num_modes):
                        simulator.mode = mode
                        mode_responses.append(simulator._response_batch(current, candidates, noise=noise))
                        mode_references.append(simulator._response_batch(current, reference_candidates, noise=noise))
                        mode_outcomes = simulator.counterfactuals(pool, planning)
                        mode_costs.append([_realized_total_cost(outcome, spec, planning) for outcome in mode_outcomes])
                finally:
                    simulator.mode = factual_mode
                values.update({
                    "all_mode_responses": np.asarray(mode_responses, dtype=np.float32),
                    "all_mode_references": np.asarray(mode_references, dtype=np.float32),
                    "all_mode_costs": np.asarray(mode_costs, dtype=np.float32),
                })
                if not np.array_equal(values["all_mode_responses"][factual_mode], responses):
                    raise ValueError("all-mode supervision changed factual response")
            for name, value in values.items():
                records[name][episode].append(value)
    batch = {name: torch.from_numpy(np.asarray(values, dtype=np.float32)) for name, values in records.items()}
    batch["mode"] = torch.tensor([record.simulator.mode for record in trace[0]], dtype=torch.long)
    return batch


def residual_training_loss(output, target, mode, dt: float, prefix: int):
    batch = torch.arange(len(mode), device=mode.device)
    predicted = output["mean"][batch, 0, mode]
    standard_deviation = output["log_std"][batch, 0, mode].exp().detach()
    error = predicted - target
    displacement = error.cumsum(dim=-2) * dt
    normalized_error = (error / standard_deviation.clamp_min(0.05)).square()
    return normalized_error.mean() + normalized_error[:, :prefix].mean() + displacement[:, :prefix].square().mean()


def action_effect_loss(output, reference_output, target, reference_target, mode, prefix):
    batch = torch.arange(len(mode), device=mode.device)
    predicted = output["mean"][batch, 0, mode, :prefix] - reference_output["mean"][batch, 0, mode, :prefix]
    actual = target[:, :prefix] - reference_target[:, :prefix]
    return (predicted - actual).square().mean()


def prefix_cost_loss(output, reference_output, values, planning, dt, prefix):
    indices = torch.arange(len(values["mode"]), device=values["mode"].device)
    means = output["mean"][indices, 0, values["mode"]][:, None]
    references = reference_output["mean"][indices, 0, values["mode"]][:, None]
    configuration = replace(planning, task_steps=prefix, response_steps=prefix, violation_probability_alpha=None)
    if configuration.risk_weight:
        raise ValueError("component cost regression currently requires a zero collision-indicator weight")
    costs = expected_candidate_costs(
        means, torch.ones(1, device=means.device), values["current"][:, 0],
        values["candidates"][:, 0], references, dt, configuration,
    )
    return (costs["unconstrained_total"] - values["true_cost"]).square().mean()


@torch.no_grad()
def evaluate_residual(model, data, device: str, prefix: int, batch_size: int, effect_weight: float = 0.0, cost_weight: float = 0.0, planning=None) -> dict:
    metrics = {f"{response}_{metric}": [] for response in ("joint", "additive") for metric in ("loss", "prefix_ade", "prefix_mse")}
    for start in range(0, len(data["mode"]), batch_size):
        values = {name: tensor[start:start + batch_size].to(device) for name, tensor in data.items()}
        batch = torch.arange(len(values["mode"]), device=device)
        for response in ("joint", "additive"):
            output = model(values["history"], values["current"], values["candidates"], include_pair_interactions=response == "joint")
            error = output["mean"][batch, 0, values["mode"], :prefix] - values["target"][:, :prefix]
            loss = residual_training_loss(output, values["target"], values["mode"], model.config.dt, prefix)
            if effect_weight or cost_weight:
                reference_output = model(values["history"], values["current"], stationary_reference_candidates(values["current"], model.config.horizon), include_pair_interactions=response == "joint")
            if effect_weight:
                loss = loss + effect_weight * action_effect_loss(output, reference_output, values["target"], values["reference_target"], values["mode"], prefix)
            if cost_weight:
                loss = loss + cost_weight * prefix_cost_loss(output, reference_output, values, planning, model.config.dt, prefix)
            metrics[f"{response}_loss"].extend([float(loss)] * len(batch))
            metrics[f"{response}_prefix_mse"].extend(error.square().mean(dim=(-1, -2)).cpu().tolist())
            metrics[f"{response}_prefix_ade"].extend(torch.linalg.vector_norm(error.cumsum(dim=1) * model.config.dt, dim=-1).mean(dim=-1).cpu().tolist())
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--response-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--train-data-seed", type=int, default=924601)
    parser.add_argument("--validation-data-seed", type=int, default=924607)
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--validation-episodes", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--behavior", choices=["random", "uniform"], default="random")
    parser.add_argument("--effect-weight", type=float, default=0.0)
    parser.add_argument("--cost-weight", type=float, default=0.0)
    parser.add_argument("--reuse-rollout-dir")
    parser.add_argument("--all-mode-supervision", action="store_true")
    args = parser.parse_args()
    if args.train_data_seed == args.validation_data_seed:
        raise ValueError("training and validation scene seeds must be disjoint")
    if min(args.train_episodes, args.validation_episodes, args.epochs, args.batch_size, args.lr) <= 0:
        raise ValueError("positive training sizes, epochs and learning rate are required")
    if min(args.effect_weight, args.cost_weight) < 0 or (args.effect_weight and args.behavior != "uniform"):
        raise ValueError("effect supervision requires uniform factual rollout collection")
    if args.all_mode_supervision and args.behavior != "uniform":
        raise ValueError("all-mode supervision requires a factual Uniform trace")
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    raw = yaml.safe_load(Path(args.config).read_text())
    planning = PlanningCostConfig(**raw["planning"])
    if args.cost_weight and raw["simulation"].get("task_type", "entity_approach") != "entity_approach":
        raise ValueError("component cost regression requires explicit support before fixed-goal training")
    checkpoint = Path(args.response_checkpoint).resolve()
    model = _load_model(checkpoint, args.device, expected_dt=raw["simulation"]["dt"])
    if model.training_dt != float(raw["simulation"]["dt"]):
        raise ValueError("residual training must preserve the checkpoint timestep")
    trainable = freeze_noninteraction_parameters(model)
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    adapter = _adapter_from_config(raw, args.device)
    output_dir = Path(args.output_dir)
    protocol = {
        "arguments": vars(args), "config": raw, "trainable_parameters": trainable,
        "collection_policy": args.behavior + "; no oracle decisions",
        "supervision": "simulator mode and candidate responses, offline training only",
        "selection": "minimum validation full/prefix response loss; epoch zero eligible",
        "frozen_components": "backbone, base, single, mode embedding, uncertainty",
        "claim_boundary": "synthetic residual repair; closed-loop benefit requires separate evaluation",
    }
    inputs = [checkpoint, Path(raw["candidate_source"]["checkpoint"]).resolve()]
    if args.reuse_rollout_dir:
        collection_manifest = Path(args.reuse_rollout_dir) / "run_manifest.json"
        collection = json.loads(collection_manifest.read_text())["config"]
        for name in ("train_data_seed", "validation_data_seed", "train_episodes", "validation_episodes", "behavior"):
            if collection["arguments"][name] != getattr(args, name):
                raise ValueError(f"reused rollout {name} mismatch")
        if collection["arguments"].get("all_mode_supervision", False) != args.all_mode_supervision:
            raise ValueError("reused rollout all-mode supervision mismatch")
        if collection["config"] != raw:
            raise ValueError("reused rollout simulator/candidate configuration mismatch")
        inputs += [collection_manifest] + [Path(args.reuse_rollout_dir) / f"{split}_rollouts.pt" for split in ("train", "validation")]
        protocol["reused_factual_behavior_checkpoint"] = collection["arguments"]["response_checkpoint"]
    manifest = prepare_manifest(output_dir, protocol, inputs, _local_sources(), False)
    datasets = {}
    for split, data_seed, episodes in (
        ("train", args.train_data_seed, args.train_episodes),
        ("validation", args.validation_data_seed, args.validation_episodes),
    ):
        configuration = copy.deepcopy(raw)
        configuration["simulation"]["data_seed"] = data_seed
        if args.reuse_rollout_dir:
            batch = torch.load(Path(args.reuse_rollout_dir) / f"{split}_rollouts.pt", map_location="cpu", weights_only=False)
        else:
            batch = (
                collect_uniform_rollouts(configuration, adapter, model, episodes, args.device, args.all_mode_supervision)
                if args.behavior == "uniform" else _collect_rollout_batch(configuration, adapter, episodes, 0, "id")
            )
        datasets[split] = flatten_rollouts(batch)
        torch.save(batch, output_dir / f"{split}_rollouts.pt")
        print(f"collected {split}: {episodes} scenes, {len(datasets[split]['mode'])} candidate responses", flush=True)
    prefix = int(raw["simulation"]["execute_steps"])
    initial = evaluate_residual(model, datasets["validation"], args.device, prefix, args.batch_size, args.effect_weight, args.cost_weight, planning)
    best_loss = initial["joint_loss"]
    best_epoch = 0
    best_state = initial_state
    history = [{"epoch": 0, **initial}]
    print(f"epoch=0 {initial}", flush=True)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr, weight_decay=1e-4)
    training = datasets["train"]
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(training["mode"]))
        for indices in permutation.split(args.batch_size):
            values = {name: tensor[indices].to(args.device) for name, tensor in training.items()}
            output = model(values["history"], values["current"], values["candidates"], include_pair_interactions=True)
            loss = residual_training_loss(output, values["target"], values["mode"], model.config.dt, prefix)
            if args.effect_weight or args.cost_weight:
                with torch.no_grad():
                    reference_output = model(values["history"], values["current"], stationary_reference_candidates(values["current"], model.config.horizon))
            if args.effect_weight:
                loss = loss + args.effect_weight * action_effect_loss(output, reference_output, values["target"], values["reference_target"], values["mode"], prefix)
            if args.cost_weight:
                loss = loss + args.cost_weight * prefix_cost_loss(output, reference_output, values, planning, model.config.dt, prefix)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
            optimizer.step()
        metrics = evaluate_residual(model, datasets["validation"], args.device, prefix, args.batch_size, args.effect_weight, args.cost_weight, planning)
        history.append({"epoch": epoch, **metrics})
        if metrics["joint_loss"] < best_loss:
            best_loss = metrics["joint_loss"]
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(f"epoch={epoch} {metrics}", flush=True)
    changed_parameters = [name for name in initial_state if not torch.equal(initial_state[name], best_state[name])]
    if not set(changed_parameters).issubset(trainable):
        raise RuntimeError("residual training changed a frozen component")
    source_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_checkpoint.update({"model": best_state, "source_response_checkpoint": str(checkpoint), "residual_selected_epoch": best_epoch, "residual_training_protocol": protocol})
    torch.save(source_checkpoint, output_dir / "response.pt")
    changed_sources = [path for path, digest in manifest["sources"].items() if sha256_file(Path(path)) != digest]
    changed_inputs = [path for path, digest in manifest["checkpoints"].items() if sha256_file(Path(path)) != digest]
    result = {
        "status": "COMPLETED" if not changed_sources and not changed_inputs else "INVALIDATED",
        "selected_epoch": best_epoch, "history": history, "changed_parameters": changed_parameters,
        "changed_sources": changed_sources, "changed_inputs": changed_inputs,
        "source_checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(output_dir / "response.pt"),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if result["status"] != "COMPLETED":
        raise RuntimeError("training inputs changed during execution")


if __name__ == "__main__":
    main()
