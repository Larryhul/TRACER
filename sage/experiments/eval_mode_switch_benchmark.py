"""Evaluate persistent-belief adaptation after a latent response-mode switch.

This is a frozen-checkpoint diagnostic. Training remains on stable-mode
episodes, while evaluation changes the simulator's mode after a registered
window. It tests whether the executed-prefix filter recovers from behavior
change rather than only accumulating evidence for a stationary mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sage.belief.filter import lagged_reset_beliefs, run_categorical_filter, uniform_log_belief
from sage.response.metrics import trajectory_errors
from sage.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    TrajectoryRecurrentBelief,
    TrajectoryRecurrentConfig,
    belief_weighted_mean,
    executed_prefix_mode_log_likelihood,
    trajectory_mode_log_likelihood,
)
from sage.response.persistent_trajectory import (
    PersistentTrajectorySpec,
    PersistentTrajectoryEpisodeDataset,
    TRAJECTORY_MODE_NAMES,
)


def _switch_sample(
    spec: PersistentTrajectorySpec,
    episode_index: int,
    switch_window: int,
) -> dict[str, torch.Tensor | int | str]:
    """Generate one single-entity episode with one unannounced mode switch."""

    if not 1 <= switch_window < spec.windows:
        raise ValueError("switch_window must lie inside the episode")
    generator = PersistentTrajectoryEpisodeDataset(
        PersistentTrajectorySpec(
            split=spec.split,
            num_episodes=1,
            windows=spec.windows,
            history=spec.history,
            horizon=spec.horizon,
            num_robots=spec.num_robots,
            dt=spec.dt,
            seed=spec.seed + int(episode_index) * 104729,
            action_policy=spec.action_policy,
            response_noise=spec.response_noise,
            influence_radius=spec.influence_radius,
            execute_steps=spec.execute_steps,
            entity_max_speed=spec.entity_max_speed,
            scene_family=spec.scene_family,
            response_family=spec.response_family,
            cache_samples=False,
        )
    )
    rng = np.random.default_rng(generator.seed)
    initial_mode = int(rng.integers(0, len(TRAJECTORY_MODE_NAMES)))
    post_mode = (initial_mode + 1 + int(rng.integers(0, 2))) % len(TRAJECTORY_MODE_NAMES)
    entity_current = rng.uniform(-0.6, 0.6, size=2).astype(np.float32)
    preferred_angle = rng.uniform(-np.pi, np.pi)
    preferred_speed = rng.uniform(0.55, 1.05)
    preferred_velocity = (
        np.asarray([np.cos(preferred_angle), np.sin(preferred_angle)]) * preferred_speed
    ).astype(np.float32)
    history = np.stack(
        [
            entity_current + preferred_velocity * ((step - spec.history + 1) * spec.dt)
            for step in range(spec.history)
        ],
        axis=0,
    ).astype(np.float32)
    histories = np.zeros((spec.windows, spec.history, 2), dtype=np.float32)
    robot_current_all = np.zeros((spec.windows, spec.num_robots, 4), dtype=np.float32)
    candidates_all = np.zeros(
        (spec.windows, spec.num_robots, spec.horizon, 4), dtype=np.float32
    )
    responses = np.zeros((spec.windows, spec.horizon, 2), dtype=np.float32)
    modes = np.zeros(spec.windows, dtype=np.int64)
    high_interaction = np.zeros(spec.windows, dtype=bool)
    for window in range(spec.windows):
        mode = initial_mode if window < switch_window else post_mode
        histories[window] = history
        current, candidates, _ = generator._robot_candidates(rng, entity_current)
        response = generator._response(
            rng, mode, entity_current, preferred_velocity, current, candidates
        )
        robot_current_all[window] = current
        candidates_all[window] = candidates
        responses[window] = response
        modes[window] = mode
        high_interaction[window] = bool(
            (np.linalg.norm(current[:, :2], axis=-1) < spec.influence_radius).all()
        )
        for step in range(spec.execute_steps):
            entity_current = entity_current + response[step] * spec.dt
            history = np.concatenate([history[1:], entity_current[None]], axis=0)
    return {
        "entity_id": generator.split_offset + int(episode_index),
        "entity_history": torch.from_numpy(histories),
        "robot_current": torch.from_numpy(robot_current_all),
        "robot_candidates": torch.from_numpy(candidates_all),
        "target_future_vel": torch.from_numpy(responses),
        "mode": torch.from_numpy(modes),
        "high_interaction": torch.from_numpy(high_interaction),
        "switch_window": switch_window,
        "initial_mode": initial_mode,
        "post_mode": post_mode,
        "dt": spec.dt,
    }


def _load_models(checkpoint_dir: Path, device: str) -> tuple[
    BeliefConditionedTeamResponseModel, TrajectoryRecurrentBelief
]:
    response_checkpoint = torch.load(
        checkpoint_dir / "response.pt", map_location=device, weights_only=True
    )
    response_config = PersistentTrajectoryModelConfig(
        **response_checkpoint["model_config"]
    )
    response_model = BeliefConditionedTeamResponseModel(response_config).to(device)
    response_model.load_state_dict(response_checkpoint["model"])
    response_model.eval()

    recurrent_checkpoint = torch.load(
        checkpoint_dir / "recurrent.pt", map_location=device, weights_only=True
    )
    recurrent_config = TrajectoryRecurrentConfig(
        **recurrent_checkpoint["model_config"]
    )
    recurrent_model = TrajectoryRecurrentBelief(recurrent_config).to(device)
    recurrent_model.load_state_dict(recurrent_checkpoint["model"])
    return response_model, recurrent_model.eval()


def _mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    return float(finite.mean()) if finite.numel() else float("nan")


def _adaptation_delay(correct: torch.Tensor, switch_window: int) -> torch.Tensor:
    """Number of post-switch windows until mode argmax is correct; inf if absent."""

    post = correct[:, switch_window:]
    found = post.float().argmax(dim=1)
    has = post.any(dim=1)
    return torch.where(has, found + 1, torch.full_like(found, post.shape[1] + 1))


@torch.no_grad()
def _evaluate_split(
    checkpoint_dir: Path,
    spec: PersistentTrajectorySpec,
    episodes: int,
    switch_window: int,
    device: str,
    likelihood_scale: float,
    episode_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    response_model, recurrent_model = _load_models(checkpoint_dir, device)
    samples = [_switch_sample(spec, episode, switch_window) for episode in range(episodes)]
    history = torch.stack([sample["entity_history"] for sample in samples]).to(device)
    current = torch.stack([sample["robot_current"] for sample in samples]).to(device)
    candidates = torch.stack([sample["robot_candidates"] for sample in samples]).to(device)
    target = torch.stack([sample["target_future_vel"] for sample in samples]).to(device)
    modes = torch.stack([sample["mode"] for sample in samples]).to(device)
    active = torch.ones(episodes, spec.windows, device=device, dtype=torch.bool)
    output = response_model(history, current, candidates)
    log_likelihood = trajectory_mode_log_likelihood(
        output["mean"], output["log_std"], target
    )
    prefix_likelihood = executed_prefix_mode_log_likelihood(
        output["mean"], output["log_std"], target, spec.execute_steps
    )
    mode_count = response_model.config.num_modes
    prior = uniform_log_belief(mode_count, episodes, device=device)
    persistent_before, persistent_after = run_categorical_filter(
        prefix_likelihood * likelihood_scale,
        prior_log_belief=prior,
        observation_mask=active,
    )
    reset_before, reset_after = lagged_reset_beliefs(
        prefix_likelihood * likelihood_scale,
        prior_log_belief=prior,
        observation_mask=active,
    )
    recurrent_after = recurrent_model(
        current,
        candidates,
        target,
        observation_mask=active,
        entity_history=history,
    ).exp()
    recurrent_before = torch.cat([prior.exp()[:, None], recurrent_after[:, :-1]], dim=1)
    uniform = prior.exp()[:, None].expand(-1, spec.windows, -1)
    beliefs = {
        "no-belief": (uniform, uniform),
        "reset": (reset_before.exp(), reset_after.exp()),
        "persistent": (persistent_before.exp(), persistent_after.exp()),
        "recurrent": (recurrent_before, recurrent_after),
        "oracle": (
            F.one_hot(modes, num_classes=mode_count).float(),
            F.one_hot(modes, num_classes=mode_count).float(),
        ),
    }
    coordinates = target.shape[-2] * target.shape[-1]
    rows: list[dict[str, Any]] = []
    for name, (before, after) in beliefs.items():
        before_probability = before.gather(-1, modes[..., None]).squeeze(-1)
        after_probability = after.gather(-1, modes[..., None]).squeeze(-1)
        before_correct = before.argmax(-1).eq(modes)
        after_correct = after.argmax(-1).eq(modes)
        predicted_mean = belief_weighted_mean(output["mean"], before)
        ade, fde = trajectory_errors(
            predicted_mean.reshape(-1, spec.horizon, 2),
            target.reshape(-1, spec.horizon, 2),
            spec.dt,
        )
        post_correct = after_correct[:, switch_window:]
        delay = _adaptation_delay(after_correct, switch_window)
        if episode_records is not None:
            episode_ade = ade.reshape(episodes, spec.windows)
            episode_fde = fde.reshape(episodes, spec.windows)
            episode_nll = -torch.logsumexp(before.log().clamp_min(-30.0) + log_likelihood, dim=-1) / coordinates
            labels = F.one_hot(modes, num_classes=mode_count).to(before)
            for episode in range(episodes):
                for phase, bounds in (("all", slice(None)), ("pre_switch", slice(None, switch_window)), ("post_switch", slice(switch_window, None))):
                    episode_records.append({
                        "episode": episode, "method": name, "phase": phase,
                        "trajectory_ade": _mean(episode_ade[episode, bounds]),
                        "trajectory_fde": _mean(episode_fde[episode, bounds]),
                        "predictive_nll": _mean(episode_nll[episode, bounds]),
                        "decision_brier": _mean((before[episode, bounds] - labels[episode, bounds]).square().sum(-1)),
                        "posterior_brier": _mean((after[episode, bounds] - labels[episode, bounds]).square().sum(-1)),
                        "decision_mode_accuracy": _mean(before_correct[episode, bounds].float()),
                        "posterior_mode_accuracy": _mean(after_correct[episode, bounds].float()),
                        "adaptation_delay_windows": int(delay[episode]),
                    })
        rows.append(
            {
                "method": name,
                "episodes": episodes,
                "split": spec.split,
                "switch_window": switch_window,
                "pre_switch_mode_accuracy": _mean(after_correct[:, :switch_window].float()),
                "post_switch_first_window_accuracy": _mean(after_correct[:, switch_window].float()),
                "post_switch_mode_accuracy": _mean(post_correct.float()),
                "post_switch_mean_probability": _mean(after_probability[:, switch_window:]),
                "adaptation_delay_windows": float(delay.float().mean()),
                "adaptation_success_rate": float((delay <= post_correct.shape[1]).float().mean()),
                "predictive_nll": _mean(
                    -torch.logsumexp(before.log().clamp_min(-30.0) + log_likelihood, dim=-1)
                    / coordinates
                ),
                "trajectory_ade": _mean(ade),
                "trajectory_fde": _mean(fde),
                "mean_before_confidence": _mean(before.max(-1).values),
                "mean_after_entropy": _mean(
                    -(after.clamp_min(1e-9) * after.clamp_min(1e-9).log()).sum(-1)
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen persistent-belief mode-switch benchmark.")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--switch-window", type=int, default=4)
    parser.add_argument("--seed", type=int, default=418731)
    parser.add_argument("--likelihood-scale", type=float, default=0.2)
    parser.add_argument("--scene-family", default="radial")
    parser.add_argument("--response-family", default="pincer")
    parser.add_argument("--action-policy", choices=["iid", "correlated"], default="iid")
    parser.add_argument("--split", choices=["id", "ood"], default="id")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_seed in args.seeds:
        checkpoint = Path(args.checkpoint_root) / f"seed{model_seed}"
        spec = PersistentTrajectorySpec(
            split=args.split,
            num_episodes=args.episodes,
            windows=args.windows,
            num_robots=3,
            seed=args.seed,
            action_policy=args.action_policy,
            scene_family=args.scene_family,
            response_family=args.response_family,
            cache_samples=False,
        )
        print(
            f"mode_switch split={args.split} seed={model_seed} device={args.device}",
            flush=True,
        )
        for row in _evaluate_split(
            checkpoint,
            spec,
            args.episodes,
            args.switch_window,
            args.device,
            args.likelihood_scale,
        ):
            row["model_seed"] = model_seed
            rows.append(row)
    with (output / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    summary = {}
    for method, values in grouped.items():
        summary[method] = {
            key: statistics.fmean(float(value[key]) for value in values)
            for key in values[0]
            if key in {
                "pre_switch_mode_accuracy",
                "post_switch_first_window_accuracy",
                "post_switch_mode_accuracy",
                "post_switch_mean_probability",
                "adaptation_delay_windows",
                "adaptation_success_rate",
                "predictive_nll",
                "trajectory_ade",
                "trajectory_fde",
                "mean_before_confidence",
                "mean_after_entropy",
            }
        }
        summary[method]["model_seed_count"] = len(values)
    payload = {
        "protocol": {
            "status": "completed",
            "independent_unit": "episode",
            "mode_switch_window": args.switch_window,
            "episodes_per_seed": args.episodes,
            "model_seeds": args.seeds,
            "split": args.split,
            "action_policy": args.action_policy,
            "device": args.device,
            "likelihood_scale": args.likelihood_scale,
            "checkpoint_root": args.checkpoint_root,
        },
        "summary": summary,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
