from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sage.experiments.run_sicnav_tracer_loop import (
    _action_velocity, _new_episode, _velocity_action,
)
from sage.experiments.run_crowdsimplus_paired_response import _state_hash
from sage.response.data import EntityResponseBatch
from sage.response.models import ResponseModelConfig, TeamResponsePredictor
from sage.response.external_checkpoint_adapter import delayed_response_prediction
from sage.response.planning_support import swept_surface_clearance
from sage.response.static_route import static_route
from sage.response.unicycle_candidates import UnicycleLimits, project_unicycle_candidates


def aligned_history(env, agent, history):
    index = None if agent is env.robot else env.humans.index(agent)
    observed = [np.asarray((state[0] if index is None else state[1][index]).position, dtype=np.float32)
                for state in env.states]
    current = np.asarray(agent.get_position(), dtype=np.float32)
    if len(observed) > 1:
        velocity = (observed[-1] - observed[-2]) / env.time_step
    else:
        velocity = np.asarray(agent.get_velocity(), dtype=np.float32)
    if not observed or not np.allclose(observed[-1], current, atol=1e-6):
        observed.append(current)
    while len(observed) < history:
        observed.insert(0, observed[0] - velocity * env.time_step)
    result = np.asarray(observed[-history:])
    result[-1] = current
    return result


def branch_response(env, velocities, headings, action_type, preferred_speeds=None):
    clone = copy.deepcopy(env)
    for human_index, speed in (preferred_speeds or {}).items():
        if not 0 <= human_index < len(clone.humans) or not np.isfinite(speed) or speed <= 0:
            raise ValueError("training mode interventions require valid human indices and positive finite speeds")
        clone.humans[human_index].v_pref = float(speed)
    robot = clone.robot
    start = np.asarray(robot.get_position(), dtype=np.float32)
    positions, actions, responses = [], [], []
    random_state = np.random.get_state()
    try:
        for velocity, heading in zip(velocities, headings):
            raw = _velocity_action(robot, action_type, velocity, heading)
            action = clone.constrain_agent_action_exact(robot, raw)
            actions.append(_action_velocity(robot, action))
            clone.step(action)
            positions.append(np.asarray(robot.get_position(), dtype=np.float32))
            responses.append(np.array([human.get_velocity() for human in clone.humans], dtype=np.float32))
    finally:
        np.random.set_state(random_state)
    actions = np.asarray(actions)
    positions = np.asarray(positions)
    if not np.allclose(positions, start + np.cumsum(actions * env.time_step, axis=0), atol=2e-5):
        raise ValueError("rolling response branch violated executable trajectory contract")
    return positions, actions, np.asarray(responses)


def collect(args, cases, *, state_callback=None, collect_native_responses=True):
    root = Path(args.external_root).resolve()
    configuration = root / "sicnav/configs/env.config"
    limits = UnicycleLimits.from_sicnav_config(root / "sicnav/configs/policy.config")
    data = {name: [] for name in ("history", "current", "candidates", "neighbors", "neighbor_mask", "target", "case", "window", "human", "branch", "reference_branch")}
    generator = np.random.default_rng(args.data_seed)
    for case in cases:
        env, robot, _, obstacles, action_type, _, _ = _new_episode(root, case, configuration, None, "sicnav-unicycle")
        route = static_route(obstacles, robot.get_goal_position(), robot.radius)
        for window in range(args.windows):
            position, velocity = np.array(robot.get_position()), np.array(robot.get_velocity())
            humans = list(env.humans)
            human_positions = np.array([human.get_position() for human in humans])
            routed, route_headings = route.unicycle_candidates(position, velocity, robot.theta, args.horizon, .25, [.35, .65, .9], limits)
            offsets = np.linspace(-np.pi, np.pi, 8, endpoint=False)
            direction = np.arctan2(*(np.array(robot.get_goal_position()) - position)[::-1])
            sampled = np.zeros((9, args.horizon, 4), dtype=np.float32)
            sampled[:8, :, 2:] = .65 * np.stack([np.cos(direction + offsets), np.sin(direction + offsets)], -1)[:, None]
            projected, headings = project_unicycle_candidates(sampled, position, velocity, robot.theta, .25, limits, return_headings=True)
            candidates = np.concatenate([routed, projected])
            headings = np.concatenate([route_headings, headings])
            probe_start = len(candidates)
            probe_directions = []
            probe_count = int(getattr(args, "interaction_probes", 0))
            if probe_count:
                if probe_count < 1 or probe_count > 4:
                    raise ValueError("interaction-probes must lie in [1, 4]")
                for human_position in human_positions:
                    angle = np.arctan2(human_position[1] - position[1], human_position[0] - position[0])
                    probe_offsets = (0.0, np.pi / 2.0, -np.pi / 2.0, np.pi)
                    probe_directions.extend(angle + offset for offset in probe_offsets[:probe_count])
                probe_samples = np.zeros((len(probe_directions), args.horizon, 4), dtype=np.float32)
                probe_samples[:, :, 2:] = .9 * np.stack(
                    [np.cos(probe_directions), np.sin(probe_directions)], -1
                )[:, None]
                probe_candidates, probe_headings = project_unicycle_candidates(
                    probe_samples, position, velocity, robot.theta, .25, limits, return_headings=True
                )
                candidates = np.concatenate([candidates, probe_candidates])
                headings = np.concatenate([headings, probe_headings])
            stationary_index = len(candidates)
            stationary = np.zeros((1, args.horizon, 4), dtype=np.float32)
            stationary[0, :, :2] = position
            candidates = np.concatenate([candidates, stationary])
            headings = np.concatenate([headings, np.full((1, args.horizon), robot.theta, dtype=np.float32)])
            human_velocities = np.array([human.get_velocity() for human in humans])
            clearance = swept_surface_clearance(candidates, position, human_positions, human_velocities,
                                                 np.array([robot.radius + human.radius for human in humans]), .25, 1).min(axis=1)
            task = route.distance_at(position) - route.distance_at(candidates[:, -1, :2])
            score = -task + 2.0 * np.exp(-np.clip(clearance, -.6, 5.) / .3)
            score += 5.0 * (~route.free_positions(candidates[..., :2])).mean(axis=1)
            selected = int(score.argmin())
            safe_exploration = np.flatnonzero((clearance >= .05) & route.free_positions(candidates[:, 0, :2]))
            if len(safe_exploration) and generator.random() < .35:
                selected = int(generator.choice(safe_exploration))
            if window % args.collection_stride == 0:
                histories = np.stack([aligned_history(env, human, args.history) for human in humans])
                selected_branches = [0, 1, 2, selected, len(candidates) - 1]
                if probe_count:
                    selected_branches.extend(range(probe_start, len(candidates)))
                selected_branches.append(stationary_index)
                selected_branches.append(int(generator.integers(3, len(candidates) - 1)))
                selected_branches = sorted(set(selected_branches))
                state_before = _state_hash(robot, humans, obstacles)
                if state_callback is not None:
                    state_callback(env, candidates, headings, histories, selected, stationary_index,
                                   action_type, case, window, obstacles)
                for branch in selected_branches if collect_native_responses else ():
                    positions, actions, targets = branch_response(env, candidates[branch, :, 2:], headings[branch], action_type)
                    for human_index, human in enumerate(humans):
                        origin = np.asarray(human.get_position(), dtype=np.float32)
                        data["history"].append(histories[human_index])
                        data["current"].append(np.concatenate([position - origin, velocity])[None])
                        data["candidates"].append(np.concatenate([positions - origin, actions], -1)[None])
                        data["neighbors"].append(histories)
                        data["neighbor_mask"].append(np.arange(len(humans)) != human_index)
                        data["target"].append(targets[:, human_index])
                        data["case"].append(case)
                        data["window"].append(window)
                        data["human"].append(human_index)
                        data["branch"].append(branch)
                        data["reference_branch"].append(stationary_index)
                if _state_hash(robot, humans, obstacles) != state_before:
                    raise ValueError("counterfactual collection mutated the live scene")
            raw = _velocity_action(robot, action_type, candidates[selected, 0, 2:], headings[selected, 0])
            _, _, done, info = env.step(env.constrain_agent_action_exact(robot, raw))
            if done:
                break
        print(f"collected case={case} windows={window + 1} samples={len(data['target'])}", flush=True)
    return {name: np.asarray(values) for name, values in data.items()}


def batch_data(data, indices, device):
    tensors = {name: torch.from_numpy(value[indices]).to(device) for name, value in data.items()}
    count = len(indices)
    return EntityResponseBatch(
        entity_history=tensors["history"].float(), entity_current=tensors["history"][:, -1].float(),
        robot_current=tensors["current"].float(), robot_candidates=tensors["candidates"].float(),
        robot_mask=torch.ones(count, 1, dtype=torch.bool, device=device),
        target_future_vel=tensors["target"].float(), high_interaction=torch.ones(count, dtype=torch.bool, device=device),
        scene_index=tensors["case"].long() * 1000 + tensors["window"].long(), entity_index=tensors["human"].long(),
        neighbor_history=tensors["neighbors"].float(), neighbor_mask=tensors["neighbor_mask"].bool(), dt=.25,
    )


def prediction_loss(prediction, target):
    residual = target - prediction["mean"]
    return (.5 * residual.square() * (-2 * prediction["log_std"]).exp() + prediction["log_std"] + .5 * np.log(2 * np.pi)).mean()


def paired_effect_loss(prediction, batch):
    same_group = batch.scene_index[:, None].eq(batch.scene_index[None]) & batch.entity_index[:, None].eq(batch.entity_index[None])
    pairs = torch.triu(same_group, diagonal=1)
    first, second = torch.where(pairs)
    if not len(first):
        return prediction["mean"].sum() * 0.0
    error = prediction["mean"][first] - prediction["mean"][second] - batch.target_future_vel[first] + batch.target_future_vel[second]
    return error.square().mean()


def response_group_batches(data, batch_size, generator):
    groups = {}
    for row, identity in enumerate(zip(data["case"], data["window"], data["human"])):
        groups.setdefault(tuple(identity), []).append(row)
    ordered = list(groups.values())
    if batch_size < max(map(len, ordered)):
        raise ValueError("batch size must fit a complete counterfactual response group")
    pending = []
    for index in generator.permutation(len(ordered)):
        if len(pending) + len(ordered[index]) > batch_size:
            yield np.asarray(pending, dtype=np.int64)
            pending = []
        pending.extend(ordered[index])
    if pending:
        yield np.asarray(pending, dtype=np.int64)


@torch.no_grad()
def evaluate(model, data, args):
    model.eval()
    losses, errors, prefix = [], [], []
    for start in range(0, len(data["target"]), args.batch_size):
        batch = batch_data(data, np.arange(start, min(start + args.batch_size, len(data["target"]))), args.device)
        prediction = delayed_response_prediction(model, batch, getattr(args, "response_delay_steps", 0))
        losses.append(float(prediction_loss(prediction, batch.target_future_vel)) * batch.num_samples)
        errors.append(float(((prediction["mean"] - batch.target_future_vel) * .25).cumsum(1).norm(dim=-1).mean()) * batch.num_samples)
        prefix.append(float((prediction["mean"][:, 0] - batch.target_future_vel[:, 0]).square().mean()) * batch.num_samples)
    return {"nll": sum(losses) / len(data["target"]), "ade": sum(errors) / len(data["target"]),
            "prefix_mse": sum(prefix) / len(data["target"])}


def main():
    parser = argparse.ArgumentParser(description="Train external FRM on independent rolling mixed-policy response branches.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--external-root", default="raw_data/external/source_repos/sicnav_official_filtered")
    parser.add_argument("--train-start", type=int, default=112)
    parser.add_argument("--train-cases", type=int, default=24)
    parser.add_argument("--val-cases", type=int, default=8)
    parser.add_argument("--windows", type=int, default=60)
    parser.add_argument("--collection-stride", type=int, default=6)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--data-seed", type=int, default=978731)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2031, 2039])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--response-delay-steps", type=int, default=0)
    parser.add_argument("--effect-weight", type=float, default=0.0)
    parser.add_argument("--grouped-response-batches", action="store_true")
    parser.add_argument("--interaction-probes", type=int, default=0,
                        help="Add per-human approach/lateral/away counterfactual probes to expose informative response changes.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "rolling-prefix-aligned-v2", "parameters": {key: value for key, value in vars(args).items() if key not in {"reuse_data", "device", "output_dir"}}}
    manifest_path = output / "data_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("rolling data collection/training configuration does not match existing manifest")
    elif any(output.iterdir()):
        raise ValueError("unversioned rolling data must not be reused")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    datasets = {}
    source_dir = output if args.data_dir is None else Path(args.data_dir)
    if args.data_dir is not None:
        source_manifest = json.loads((source_dir / "data_manifest.json").read_text())
        if source_manifest["version"] not in {"rolling-prefix-aligned-v2", "tracer-live-prefix-aligned-v1", "tracer-live-prefix-aligned-v2"}:
            raise ValueError("rolling source data lacks aligned current-state histories")
        for name in ("train_start", "train_cases", "val_cases", "windows", "collection_stride", "history", "horizon", "data_seed"):
            if source_manifest["parameters"][name] != getattr(args, name):
                raise ValueError(f"rolling source data configuration mismatch: {name}")
        if source_manifest["version"].startswith("tracer-live-prefix-aligned-") and source_manifest["parameters"]["dt"] != .25:
            raise ValueError("live response data clock must match the external model")
        (output / "source_data.json").write_text(json.dumps({"directory": str(source_dir),
            "hashes": {name: hashlib.sha256((source_dir / f"{name}.npz").read_bytes()).hexdigest() for name in ("train", "val")}}, indent=2) + "\n")
    for split, cases in (("train", range(args.train_start, args.train_start + args.train_cases)),
                         ("val", range(args.train_start + args.train_cases, args.train_start + args.train_cases + args.val_cases))):
        path = source_dir / f"{split}.npz"
        if path.exists():
            if not args.reuse_data and args.data_dir is None:
                raise ValueError("rolling data already exists; use --reuse-data explicitly")
            with np.load(path) as loaded:
                datasets[split] = dict(loaded)
        else:
            datasets[split] = collect(args, cases)
            np.savez_compressed(path, **datasets[split])
    if set(datasets["train"]["case"]) & set(datasets["val"]["case"]):
        raise ValueError("rolling training and validation cases overlap")
    results = {}
    for seed in args.seeds:
        path = output / "checkpoints" / f"joint_seed{seed}.pt"
        if path.exists():
            raise ValueError("rolling response checkpoint already exists")
        torch.manual_seed(seed)
        config = ResponseModelConfig(history=args.history, horizon=args.horizon, num_robots=1, hidden_dim=64,
                                     architecture="tracer", dt=.25, geometry_features=True,
                                     dynamic_interaction_features=True, temporal_pooling="attention")
        model = TeamResponsePredictor("joint", config).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        generator = np.random.default_rng(seed)
        best = float("inf")
        history = []
        for epoch in range(args.epochs):
            model.train()
            order = generator.permutation(len(datasets["train"]["target"]))
            batches = (response_group_batches(datasets["train"], args.batch_size, generator)
                       if args.grouped_response_batches else
                       (order[start:start + args.batch_size] for start in range(0, len(order), args.batch_size)))
            for indices in batches:
                batch = batch_data(datasets["train"], indices, args.device)
                prediction = delayed_response_prediction(model, batch, args.response_delay_steps)
                prefix_prediction = {name: value[:, :1] for name, value in prediction.items() if name in {"mean", "log_std"}}
                loss = prediction_loss(prediction, batch.target_future_vel) + prediction_loss(prefix_prediction, batch.target_future_vel[:, :1])
                loss = loss + args.effect_weight * paired_effect_loss(prediction, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            metrics = evaluate(model, datasets["val"], args)
            history.append(metrics)
            print(f"seed={seed} epoch={epoch + 1} {json.dumps(metrics)}", flush=True)
            if metrics["nll"] < best:
                best = metrics["nll"]
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "model_config": vars(config), "variant": "joint", "seed": seed,
                            "selection_split": "val", "best_epoch": epoch + 1, "rolling_collection": vars(args), "best_metrics": metrics,
                            "response_delay_steps": args.response_delay_steps, "effect_weight": args.effect_weight}, path)
        results[str(seed)] = history
    (output / "training_summary.json").write_text(json.dumps({"parameters": vars(args), "metrics": results,
         "claim_boundary": "single-mode rolling mixed-policy response model; not an external latent-mode belief model"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
