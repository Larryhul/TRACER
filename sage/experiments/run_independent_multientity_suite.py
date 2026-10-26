"""Run the missing independent-environment robustness experiments.

This suite keeps the TRACER response model and belief update unchanged while
changing only the data-generating environment.  It covers multi-entity
identity persistence, continuous Social-Force parameter mismatch, simple
response baselines, and a fixed-budget R=6 scalability stress test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sage.experiments.eval_persistent_closed_loop import (
    POLICIES,
    _kendall_tau_b,
    _load_model,
    _spearman,
)
from sage.experiments.run_independent_interaction_benchmark import (
    _loader,
    _set_seed,
    _train_recurrent,
    _train_response,
)
from sage.experiments.train_persistent_trajectory_response import (
    _evaluate,
    _flatten_entities,
    _move,
)
from sage.response.closed_loop import (
    ClosedLoopSpec,
    PlanningCostConfig,
    expected_candidate_costs,
)
from sage.response.independent_social_force import (
    MultiEntityIndependentSocialForceSimulator,
    MultiEntityIndependentSocialForceDataset,
    MultiEntitySocialForceDatasetSpec,
)
from sage.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    TrajectoryRecurrentBelief,
    TrajectoryRecurrentConfig,
    stationary_reference_candidates,
    trajectory_mode_log_likelihood,
)
from sage.belief.filter import categorical_bayes_update, uniform_log_belief


DEFAULT_SEEDS = (2027, 2031, 2039, 2053, 2063)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": float(statistics.fmean(values)) if values else float("nan"),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def _bootstrap_ci(
    values: list[float], seed: int, samples: int = 5000
) -> dict[str, float | int]:
    if not values:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "count": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "count": int(len(array)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired_episode_cost_deltas(
    path: Path,
    split: str,
    first_policy: str,
    second_policy: str,
) -> list[float]:
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split or row.get("policy") not in {
                first_policy,
                second_policy,
            }:
                continue
            key = (row.get("seed", ""), row.get("episode", ""))
            grouped.setdefault(key, {})[row["policy"]] = float(
                row["realized_total_cost"]
            )
    return [
        pair[second_policy] - pair[first_policy]
        for pair in grouped.values()
        if first_policy in pair and second_policy in pair
    ]


def refresh_completed_output(output_dir: Path) -> None:
    """Backfill post-run paired CIs without rerunning the frozen experiment."""

    summary_path = output_dir / "summary.json"
    episode_path = output_dir / "h3_multientity_episode.csv"
    if not summary_path.exists() or not episode_path.exists():
        raise FileNotFoundError("completed suite requires summary.json and h3_multientity_episode.csv")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for split, seed_offset in (("id", 0), ("ood", 1)):
        reset_delta = _paired_episode_cost_deltas(
            episode_path, split, "persistent-joint", "reset-joint"
        )
        additive_delta = _paired_episode_cost_deltas(
            episode_path, split, "persistent-joint", "persistent-additive"
        )
        contrasts = payload["h3"]["summary"][split]["contrasts"]
        contrasts["reset_minus_persistent_total_cost"] = float(np.mean(reset_delta))
        contrasts["reset_minus_persistent_total_cost_ci95"] = _bootstrap_ci(
            reset_delta, payload["metadata"]["data_seed"] + seed_offset
        )
        contrasts["additive_minus_joint_total_cost"] = float(np.mean(additive_delta))
        contrasts["additive_minus_joint_total_cost_ci95"] = _bootstrap_ci(
            additive_delta, payload["metadata"]["data_seed"] + 2 + seed_offset
        )
    payload["postprocessing"] = {
        "protocol": "paired-episode-ci-refresh-v1",
        "source": "h3_multientity_episode.csv",
        "git_commit": _git_commit(),
    }
    summary_path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")

    report_path = output_dir / "report.md"
    if report_path.exists():
        report_lines = report_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(report_lines):
            for split in ("id", "ood"):
                if not line.startswith(f"- `{split}` Persistent-joint cost"):
                    continue
                contrasts = payload["h3"]["summary"][split]["contrasts"]
                prefix, remainder = line.split("reset-minus-persistent `", 1)
                suffix = remainder.split("`; selection changes", 1)[1]
                ci = contrasts["reset_minus_persistent_total_cost_ci95"]
                report_lines[index] = (
                    f"{prefix}reset-minus-persistent `"
                    f"{contrasts['reset_minus_persistent_total_cost']:.4f}` "
                    f"CI `[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]`; selection changes"
                    f"{suffix}"
                )
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("artifacts", {})["summary.json"] = _sha256(summary_path)
        if report_path.exists():
            manifest["artifacts"]["report.md"] = _sha256(report_path)
        manifest["postprocessing"] = payload["postprocessing"]
        manifest_path.write_text(
            json.dumps(_jsonable(manifest), indent=2) + "\n", encoding="utf-8"
        )


def _build_spec(
    args: argparse.Namespace,
    split: str,
    num_robots: int,
    num_candidates: int | None = None,
) -> ClosedLoopSpec:
    return ClosedLoopSpec(
        windows=args.windows,
        history=args.history,
        horizon=args.horizon,
        num_robots=num_robots,
        num_candidates=int(num_candidates or args.num_candidates),
        execute_steps=args.execute_steps,
        dt=args.dt,
        response_noise=args.response_noise,
        influence_radius=4.0,
        entity_max_speed=args.entity_max_speed,
        scene_family=args.scene_family,
        response_family="pincer",
        seed=args.data_seed,
        split=split,
    )


def _train_models(args: argparse.Namespace, output_dir: Path) -> dict[str, Path]:
    train = MultiEntityIndependentSocialForceDataset(
        MultiEntitySocialForceDatasetSpec(
            split="train", num_episodes=args.train_episodes,
            windows=args.windows, history=args.history, horizon=args.horizon,
            num_robots=args.num_robots, num_entities=args.num_entities,
            active_entities=args.active_entities, num_candidates=args.num_candidates,
            execute_steps=args.execute_steps, dt=args.dt,
            response_noise=args.response_noise, entity_max_speed=args.entity_max_speed,
            entity_interaction_strength=args.entity_interaction_strength,
            entity_interaction_decay=args.entity_interaction_decay,
            entity_interaction_radius=args.entity_interaction_radius,
            scene_family=args.scene_family, seed=args.data_seed, cache_samples=True,
            parameter_mode="discrete",
        )
    )
    checkpoints: dict[str, Path] = {}
    for seed in args.seeds:
        _set_seed(seed)
        response_config = PersistentTrajectoryModelConfig(
            history=args.history, horizon=args.horizon, num_robots=args.num_robots,
            num_modes=3, hidden_dim=args.hidden_dim, dropout=args.dropout,
            dt=args.dt, architecture="tracer", num_heads=4, graph_layers=2,
            prefix_loss_weight=args.prefix_loss_weight,
        )
        recurrent_config = TrajectoryRecurrentConfig(
            history=args.history, horizon=args.horizon, num_robots=args.num_robots,
            num_modes=3, hidden_dim=args.recurrent_hidden_dim,
            execute_steps=args.execute_steps, use_entity_history=True, dt=args.dt,
        )
        response_model = BeliefConditionedTeamResponseModel(response_config).to(args.device)
        recurrent_model = TrajectoryRecurrentBelief(recurrent_config).to(args.device)
        _train_response(
            response_model, _loader(train, args.batch_size, True, seed), args.device,
            args.response_epochs, args.lr, args.weight_decay,
            args.mode_discrimination_weight, args.mode_temperature,
            args.prefix_loss_weight,
        )
        _train_recurrent(
            recurrent_model, _loader(train, args.batch_size, True, seed + 1),
            args.device, args.recurrent_epochs, args.recurrent_lr, args.weight_decay,
        )
        seed_dir = output_dir / "checkpoints" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": response_model.state_dict(), "model_config": vars(response_config),
             "seed": seed, "protocol": "independent-social-force-multientity-v1"},
            seed_dir / "response.pt",
        )
        torch.save(
            {"model": recurrent_model.state_dict(), "model_config": vars(recurrent_config),
             "seed": seed, "protocol": "independent-social-force-multientity-v1"},
            seed_dir / "recurrent.pt",
        )
        checkpoints[str(seed)] = seed_dir
    return checkpoints


def _h2_multientity(args: argparse.Namespace, checkpoints: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    results_by_seed: dict[str, Any] = {}
    for split in ("val", "test"):
        dataset = MultiEntityIndependentSocialForceDataset(
            MultiEntitySocialForceDatasetSpec(
                split=split, num_episodes=args.eval_episodes,
                windows=args.windows, history=args.history, horizon=args.horizon,
                num_robots=args.num_robots, num_entities=args.num_entities,
                active_entities=args.active_entities, num_candidates=args.num_candidates,
                execute_steps=args.execute_steps, dt=args.dt,
                response_noise=args.response_noise, entity_max_speed=args.entity_max_speed,
                entity_interaction_strength=args.entity_interaction_strength,
                entity_interaction_decay=args.entity_interaction_decay,
                entity_interaction_radius=args.entity_interaction_radius,
                scene_family=args.scene_family, seed=args.data_seed,
                cache_samples=True, parameter_mode="discrete",
            )
        )
        results_by_seed.setdefault(split, {})
        for seed, seed_dir in checkpoints.items():
            response = _load_model(seed_dir / "response.pt", args.device)
            recurrent_checkpoint = torch.load(seed_dir / "recurrent.pt", map_location=args.device, weights_only=False)
            recurrent_config = TrajectoryRecurrentConfig(**recurrent_checkpoint["model_config"])
            recurrent = TrajectoryRecurrentBelief(recurrent_config).to(args.device)
            recurrent.load_state_dict(recurrent_checkpoint["model"])
            metrics, rows = _evaluate(
                response, recurrent,
                DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0),
                device=args.device, dt=args.dt, likelihood_scale=args.likelihood_scale,
                confidence_threshold=args.confidence_threshold,
                execute_steps=args.execute_steps, split_name=split,
            )
            episode_values: dict[int, dict[str, dict[str, list[float]]]] = {}
            for row in rows:
                if not row["main_evaluation"]:
                    continue
                episode = int(row["episode_id"])
                method = str(row["method"])
                episode_values.setdefault(episode, {}).setdefault(method, {
                    "predictive_trajectory_nll": [],
                    "trajectory_ade": [],
                    "trajectory_fde": [],
                })
                for metric in episode_values[episode][method]:
                    episode_values[episode][method][metric].append(float(row[metric]))
            episode_contrasts = []
            for episode, methods_for_episode in sorted(episode_values.items()):
                if not {"persistent", "reset", "shuffled-id"}.issubset(methods_for_episode):
                    continue
                episode_record = {"episode": episode}
                for metric in ("predictive_trajectory_nll", "trajectory_ade", "trajectory_fde"):
                    means = {
                        method: float(statistics.fmean(methods_for_episode[method][metric]))
                        for method in ("persistent", "reset", "shuffled-id")
                    }
                    episode_record[f"reset_minus_persistent_{metric}"] = (
                        means["reset"] - means["persistent"]
                    )
                    episode_record[f"shuffled_minus_persistent_{metric}"] = (
                        means["shuffled-id"] - means["persistent"]
                    )
                episode_contrasts.append(episode_record)
            results_by_seed[split][seed] = {
                "metrics": metrics,
                "episode_contrasts": episode_contrasts,
            }
            row_path = output_dir / "h2" / f"seed{seed}_{split}_rows.csv"
            if rows:
                with row_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    summary: dict[str, Any] = {}
    for split, seed_results in results_by_seed.items():
        summary[split] = {}
        methods = next(iter(seed_results.values()))["metrics"]
        for method in methods:
            summary[split][method] = {}
            for metric, value in next(iter(seed_results.values()))["metrics"][method].items():
                if isinstance(value, (int, float)):
                    summary[split][method][metric] = _mean_std(
                        [float(seed_results[seed]["metrics"][method][metric]) for seed in seed_results]
                    )
                else:
                    summary[split][method][metric] = value
        for contrast in (
            "reset_minus_persistent_predictive_trajectory_nll",
            "reset_minus_persistent_trajectory_ade",
            "reset_minus_persistent_trajectory_fde",
            "shuffled_minus_persistent_predictive_trajectory_nll",
            "shuffled_minus_persistent_trajectory_ade",
            "shuffled_minus_persistent_trajectory_fde",
        ):
            values = [
                float(record[contrast])
                for seed in seed_results
                for record in seed_results[seed]["episode_contrasts"]
            ]
            contrast_summary = summary[split].setdefault("episode_contrasts", {})
            contrast_summary[contrast] = _bootstrap_ci(
                values, args.bootstrap_seed + len(contrast_summary)
            )
    return {"per_seed": results_by_seed, "summary": summary}


@torch.no_grad()
def _h1_continuous_mismatch(
    args: argparse.Namespace, checkpoints: dict[str, Path], output_dir: Path
) -> dict[str, Any]:
    test = MultiEntityIndependentSocialForceDataset(
        MultiEntitySocialForceDatasetSpec(
            split="test", num_episodes=args.eval_episodes,
            windows=args.windows, history=args.history, horizon=args.horizon,
            num_robots=args.num_robots, num_entities=args.num_entities,
            active_entities=args.active_entities, num_candidates=args.num_candidates,
            execute_steps=args.execute_steps, dt=args.dt,
            response_noise=args.response_noise, entity_max_speed=args.entity_max_speed,
            entity_interaction_strength=args.entity_interaction_strength,
            entity_interaction_decay=args.entity_interaction_decay,
            entity_interaction_radius=args.entity_interaction_radius,
            scene_family=args.scene_family, seed=args.data_seed,
            cache_samples=True, parameter_mode="continuous",
        )
    )
    values_by_seed: dict[str, dict[str, list[list[float]]]] = {
        str(seed): {"joint": {"ade": [], "fde": [], "nll": []}, "additive": {"ade": [], "fde": [], "nll": []}}
        for seed in checkpoints
    }
    for seed, seed_dir in checkpoints.items():
        model = _load_model(seed_dir / "response.pt", args.device)
        model.eval()
        for raw in DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0):
            batch = _flatten_entities(_move(raw, args.device))
            history = batch["entity_history"]
            current = batch["robot_current"]
            candidates = batch["robot_candidates"]
            target = batch["target_future_vel"]
            active = batch["active_mask"]
            mode = batch["mode"]
            episode_ids = batch["episode_id"].detach().cpu().numpy()
            unique_episodes = np.unique(episode_ids)
            for name, include_pair in (("joint", True), ("additive", False)):
                output = model(
                    history, current, candidates, include_pair_interactions=include_pair
                )
                selected_mean = output["mean"].gather(
                    2, mode[:, None, None, None, None].expand(
                        -1, history.shape[1], 1, model.config.horizon, 2
                    )
                ).squeeze(2)
                prediction = torch.cumsum(selected_mean * args.dt, dim=-2)
                target_position = torch.cumsum(target * args.dt, dim=-2)
                error = torch.linalg.vector_norm(prediction - target_position, dim=-1)
                likelihood = trajectory_mode_log_likelihood(
                    output["mean"], output["log_std"], target
                )
                nll = -likelihood.gather(
                    -1, mode[:, None, None].expand(-1, history.shape[1], 1)
                ).squeeze(-1)
                active_cpu = active.detach().cpu().numpy()
                error_cpu = error.detach().cpu().numpy()
                fde_cpu = error[..., -1].detach().cpu().numpy()
                nll_cpu = nll.detach().cpu().numpy()
                episode_ade = []
                episode_fde = []
                episode_nll = []
                for episode_id in unique_episodes:
                    track_mask = episode_ids == episode_id
                    valid = active_cpu[track_mask]
                    episode_ade.append(float(error_cpu[track_mask][valid].mean()))
                    episode_fde.append(float(fde_cpu[track_mask][valid].mean()))
                    episode_nll.append(float(nll_cpu[track_mask][valid].mean()))
                values_by_seed[seed][name]["ade"].extend(episode_ade)
                values_by_seed[seed][name]["fde"].extend(episode_fde)
                values_by_seed[seed][name]["nll"].extend(episode_nll)
    values: dict[str, dict[str, list[float]]] = {
        name: {
            metric: [
                value
                for seed in checkpoints
                for value in values_by_seed[seed][name][metric]
            ]
            for metric in ("ade", "fde", "nll")
        }
        for name in ("joint", "additive")
    }
    aggregate: dict[str, Any] = {}
    for name, metrics in values.items():
        aggregate[name] = {
            metric: _bootstrap_ci(items, args.bootstrap_seed + index)
            for index, (metric, items) in enumerate(metrics.items())
        }
    aggregate["additive_minus_joint"] = {
        metric: _bootstrap_ci(
            [
                additive - joint
                for additive, joint in zip(
                    values["additive"][metric], values["joint"][metric]
                )
            ],
            args.bootstrap_seed + 100 + metric_index,
        )
        for metric_index, metric in enumerate(("ade", "fde", "nll"))
    }
    payload = {
        "protocol": "independent-social-force-continuous-parameter-mismatch-v1",
        "backend": MultiEntityIndependentSocialForceSimulator.backend_name,
        "training_parameter_mode": "discrete",
        "test_parameter_mode": "continuous_interpolation_0_to_2",
        "seeds": list(args.seeds),
        "aggregate": aggregate,
        "note": "Continuous parameter coordinates are generated by the independent Social-Force backend; rounded modes are labels only.",
    }
    return payload


def _multi_entity_predicted_costs(
    means: torch.Tensor,
    log_stds: torch.Tensor,
    reference_means: torch.Tensor,
    current: torch.Tensor,
    candidates: torch.Tensor,
    beliefs: torch.Tensor,
    active: np.ndarray,
    args: argparse.Namespace,
    planning: PlanningCostConfig,
) -> dict[str, torch.Tensor]:
    """Aggregate per-entity TRACER candidate costs for a shared team action."""

    candidate_count = means.shape[1]
    total = torch.zeros(candidate_count, device=means.device, dtype=means.dtype)
    task = torch.zeros_like(total)
    safety = torch.zeros_like(total)
    human = torch.zeros_like(total)
    violation = torch.zeros_like(total)
    entity_violations: list[torch.Tensor] = []
    safe_mask = torch.ones(candidate_count, device=means.device, dtype=torch.bool)
    active_indices = np.flatnonzero(active)
    for entity in active_indices:
        entity_costs = expected_candidate_costs(
            means[entity],
            beliefs[entity],
            current[entity],
            candidates[entity],
            reference_means[entity],
            args.dt,
            planning,
            mode_log_stds=log_stds[entity],
        )
        total = total + (
            planning.task_weight * entity_costs["task"]
            + planning.safety_weight * entity_costs["safety"]
            + planning.human_weight * entity_costs["human"]
        )
        task = task + entity_costs["task"]
        safety = safety + entity_costs["safety"]
        human = human + entity_costs["human"]
        violation = 1.0 - (1.0 - violation) * (
            1.0 - entity_costs["violation_probability"]
        )
        entity_violations.append(entity_costs["violation_probability"])
        safe_mask = safe_mask & entity_costs["safe_mask"]
    denominator = float(max(len(active_indices), 1))
    risk_aggregation = str(getattr(args, "risk_aggregation", "union")).lower()
    if risk_aggregation == "mean" and entity_violations:
        violation = torch.stack(entity_violations, dim=0).mean(dim=0)
        if planning.violation_probability_alpha is not None:
            safe_mask = violation.le(float(planning.violation_probability_alpha))
    elif risk_aggregation == "max" and entity_violations:
        violation = torch.stack(entity_violations, dim=0).amax(dim=0)
        if planning.violation_probability_alpha is not None:
            safe_mask = violation.le(float(planning.violation_probability_alpha))
    elif risk_aggregation != "union":
        raise ValueError("risk_aggregation must be one of union, max, mean")
    return {
        "total": total / denominator,
        "task": task / denominator,
        "safety": safety / denominator,
        "human": human / denominator,
        "violation_probability": violation,
        "safe_mask": safe_mask,
    }


def _multi_entity_true_costs(
    outcomes: tuple[Any, ...],
    active: np.ndarray,
    spec: ClosedLoopSpec,
    planning: PlanningCostConfig,
) -> np.ndarray:
    active_indices = np.flatnonzero(active)
    values = []
    for outcome in outcomes:
        task = -float(outcome.task_progress[active_indices].mean()) / (
            spec.execute_steps * spec.dt * 1.7
        )
        safety = float(outcome.safety_cost[active_indices].mean())
        human = float(outcome.human_cost[active_indices].mean())
        values.append(
            planning.task_weight * task
            + planning.safety_weight * safety
            + planning.human_weight * human
        )
    return np.asarray(values, dtype=np.float64)


@torch.no_grad()
def _multi_entity_episode_policy(
    model: BeliefConditionedTeamResponseModel,
    policy: Any,
    spec: ClosedLoopSpec,
    episodes: int,
    args: argparse.Namespace,
    planning: PlanningCostConfig,
) -> dict[str, Any]:
    """Run one H3 policy on a genuinely multi-entity independent world."""

    simulators = [
        MultiEntityIndependentSocialForceSimulator(
            spec, episode, args.num_entities, args.active_entities,
            entity_interaction_strength=args.entity_interaction_strength,
            entity_interaction_decay=args.entity_interaction_decay,
            entity_interaction_radius=args.entity_interaction_radius,
        )
        for episode in range(episodes)
    ]
    mode_count = model.config.num_modes
    belief_store = [
        uniform_log_belief(mode_count, device=args.device)[None].expand(
            args.num_entities, mode_count
        ).clone()
        for _ in range(episodes)
    ]
    metric_names = (
        "realized_total_cost", "task_progress", "safety_cost", "human_cost",
        "min_distance", "collision_rate", "oracle_regret", "oracle_hit_rate",
        "candidate_rank_spearman", "candidate_rank_kendall",
        "task_feasible_candidate_count", "candidate_true_cost_range",
        "belief_selection_flip", "belief_selection_cost_delta",
        "safety_filter_rejections", "safety_fallback",
    )
    values = [{name: [] for name in metric_names} for _ in range(episodes)]
    start = time.perf_counter()
    model.eval()
    for window in range(spec.windows):
        pools = [simulator.candidate_pool() for simulator in simulators]
        active_masks = [simulator.active_mask() for simulator in simulators]
        candidate_count = int(pools[0].robot_candidates.shape[0])
        histories = torch.from_numpy(
            np.stack([simulator.histories for simulator in simulators])
        ).to(args.device)
        current = torch.stack([pool.robot_current for pool in pools]).to(args.device)
        candidates = torch.stack([pool.robot_candidates for pool in pools]).to(args.device)
        batch = len(simulators)
        entities = args.num_entities
        history = histories[:, :, None].expand(
            batch, entities, candidate_count, spec.history, 2
        ).reshape(batch * entities * candidate_count, 1, spec.history, 2)
        flat_current = current.permute(0, 2, 1, 3, 4).reshape(
            batch * entities * candidate_count, spec.num_robots, 4
        )
        flat_candidates = candidates.permute(0, 2, 1, 3, 4, 5).reshape(
            batch * entities * candidate_count,
            spec.num_robots,
            spec.horizon,
            4,
        )
        robot_mask = torch.ones(
            batch * entities * candidate_count, 1, spec.num_robots,
            device=args.device, dtype=torch.bool,
        )
        neighbor_history = histories[:, :, None].expand(
            batch, entities, entities, spec.history, 2
        )
        presence = torch.from_numpy(
            np.stack([simulator.presence[:, simulator.window] for simulator in simulators])
        ).to(args.device)
        neighbor_mask = presence[:, :, None].expand(
            batch, entities, entities
        ).clone()
        neighbor_mask = neighbor_mask & ~torch.eye(
            entities, device=args.device, dtype=torch.bool
        )[None]
        neighbor_history = neighbor_history[:, :, None].expand(
            batch, entities, candidate_count, entities, spec.history, 2
        ).permute(0, 1, 2, 3, 4, 5).reshape(
            batch * entities * candidate_count, 1, entities, spec.history, 2
        )
        neighbor_mask = neighbor_mask[:, :, None].expand(
            batch, entities, candidate_count, entities
        ).reshape(batch * entities * candidate_count, 1, entities)
        output = model(
            history, flat_current[:, None], flat_candidates[:, None],
            include_pair_interactions=policy.include_pair_interactions,
            robot_mask=robot_mask,
            neighbor_history=neighbor_history,
            neighbor_mask=neighbor_mask,
        )
        reference = stationary_reference_candidates(
            flat_current[:, None], spec.horizon
        )
        reference_output = model(
            history, flat_current[:, None], reference,
            include_pair_interactions=policy.include_pair_interactions,
            robot_mask=robot_mask,
            neighbor_history=neighbor_history,
            neighbor_mask=neighbor_mask,
        )
        means = output["mean"][:, 0].reshape(
            batch, entities, candidate_count, mode_count, spec.horizon, 2
        )
        log_stds = output["log_std"][:, 0].reshape_as(means)
        reference_means = reference_output["mean"][:, 0].reshape_as(means)
        for episode, simulator in enumerate(simulators):
            active = active_masks[episode]
            if policy.belief_type == "oracle":
                beliefs = torch.nn.functional.one_hot(
                    torch.from_numpy(simulator.modes).to(args.device),
                    num_classes=mode_count,
                ).float()
            elif policy.belief_type == "reset":
                beliefs = torch.full_like(belief_store[episode].exp(), 1.0 / mode_count)
            else:
                beliefs = belief_store[episode].exp()
            costs = _multi_entity_predicted_costs(
                means[episode], log_stds[episode], reference_means[episode],
                current[episode].permute(1, 0, 2, 3),
                candidates[episode].permute(1, 0, 2, 3, 4),
                beliefs, active, args, planning,
            )
            uniform_beliefs = torch.full_like(beliefs, 1.0 / mode_count)
            uniform_costs = _multi_entity_predicted_costs(
                means[episode], log_stds[episode], reference_means[episode],
                current[episode].permute(1, 0, 2, 3),
                candidates[episode].permute(1, 0, 2, 3, 4),
                uniform_beliefs, active, args, planning,
            )
            selection_scores = costs[policy.selection_objective]
            selection_scores = torch.where(
                costs["safe_mask"], selection_scores,
                torch.full_like(selection_scores, float("inf")),
            )
            fallback = not bool(costs["safe_mask"].any())
            selected = int(
                costs["violation_probability"].argmin()
                if fallback else selection_scores.argmin()
            )
            uniform_scores = uniform_costs[policy.selection_objective]
            uniform_scores = torch.where(
                uniform_costs["safe_mask"], uniform_scores,
                torch.full_like(uniform_scores, float("inf")),
            )
            uniform_selected = int(
                uniform_costs["violation_probability"].argmin()
                if not bool(uniform_costs["safe_mask"].any()) else uniform_scores.argmin()
            )
            counterfactuals = simulator.counterfactuals(pools[episode], planning)
            true_costs = _multi_entity_true_costs(
                counterfactuals, active, spec, planning
            )
            oracle = int(true_costs.argmin())
            oracle_cost = float(true_costs[oracle])
            selected_true = float(true_costs[selected])
            uniform_true = float(true_costs[uniform_selected])
            realized = simulator.step(pools[episode], selected, planning)
            torch.testing.assert_close(
                realized.response,
                counterfactuals[selected].response,
                rtol=0.0,
                atol=0.0,
            )
            active_indices = np.flatnonzero(active)
            realized_task = float(realized.task_progress[active_indices].mean())
            realized_safety = float(realized.safety_cost[active_indices].mean())
            realized_human = float(realized.human_cost[active_indices].mean())
            realized_min_distance = float(realized.min_distance[active_indices].min())
            realized_collision = float(realized.collision[active_indices].max())
            realized_total = (
                planning.task_weight * (-realized_task / (spec.execute_steps * spec.dt * 1.7))
                + planning.safety_weight * realized_safety
                + planning.human_weight * realized_human
            )
            values[episode]["realized_total_cost"].append(realized_total)
            values[episode]["task_progress"].append(realized_task)
            values[episode]["safety_cost"].append(realized_safety)
            values[episode]["human_cost"].append(realized_human)
            values[episode]["min_distance"].append(realized_min_distance)
            values[episode]["collision_rate"].append(realized_collision)
            values[episode]["oracle_regret"].append(max(selected_true - oracle_cost, 0.0))
            values[episode]["oracle_hit_rate"].append(float(selected == oracle))
            values[episode]["candidate_rank_spearman"].append(
                _spearman(costs["total"].detach().cpu().numpy(), true_costs)
            )
            values[episode]["candidate_rank_kendall"].append(
                _kendall_tau_b(costs["total"].detach().cpu().numpy(), true_costs)
            )
            values[episode]["task_feasible_candidate_count"].append(
                float(sum(outcome.task_progress[active_indices].mean() > 0.0 for outcome in counterfactuals))
            )
            values[episode]["candidate_true_cost_range"].append(
                float(true_costs.max() - true_costs.min())
            )
            values[episode]["belief_selection_flip"].append(
                float(selected != uniform_selected)
            )
            values[episode]["belief_selection_cost_delta"].append(
                uniform_true - selected_true
            )
            values[episode]["safety_filter_rejections"].append(
                float((~costs["safe_mask"]).sum())
            )
            values[episode]["safety_fallback"].append(float(fallback))
            if policy.belief_type in {"persistent", "reset"}:
                for entity in active_indices:
                    likelihood = trajectory_mode_log_likelihood(
                        means[episode, entity, selected, :, : spec.execute_steps][None, None],
                        log_stds[episode, entity, selected, :, : spec.execute_steps][None, None],
                        realized.response.to(args.device)[entity, : spec.execute_steps][None, None],
                    )[0, 0]
                    prior = (
                        belief_store[episode][entity]
                        if policy.belief_type == "persistent"
                        else uniform_log_belief(mode_count, device=args.device)
                    )
                    belief_store[episode][entity] = categorical_bayes_update(
                        prior,
                        args.likelihood_scale * likelihood,
                    )
    results: dict[str, Any] = {}
    episode_records = []
    for episode, simulator in enumerate(simulators):
        record = {name: float(statistics.fmean(items)) for name, items in values[episode].items()}
        if policy.belief_type == "oracle":
            final_belief = torch.nn.functional.one_hot(
                torch.from_numpy(simulator.modes).to(args.device),
                num_classes=mode_count,
            ).float()
        else:
            final_belief = belief_store[episode].exp()
        final_accuracy = float(
            (final_belief.argmax(dim=-1).cpu().numpy() == simulator.modes).mean()
        )
        entropy = float(
            (-(final_belief * final_belief.clamp_min(1e-9).log()).sum(dim=-1)).mean()
        )
        record["final_mode_accuracy"] = final_accuracy
        record["final_belief_entropy"] = entropy
        episode_records.append(record)
    all_metric_names = tuple(episode_records[0])
    for name in all_metric_names:
        results[name] = _mean_std([record[name] for record in episode_records])
    results["episode_records"] = episode_records
    results["runtime_ms_per_episode"] = 1000.0 * (time.perf_counter() - start) / max(episodes, 1)
    return results


def _h3_multientity(
    args: argparse.Namespace, checkpoints: dict[str, Path], output_dir: Path
) -> dict[str, Any]:
    planning = PlanningCostConfig(
        task_weight=1.0, safety_weight=2.0, human_weight=0.6,
        comfort_scale=1.0, collision_margin=args.collision_margin,
        task_steps=args.execute_steps, violation_probability_alpha=args.risk_alpha,
        risk_samples=args.risk_samples,
    )
    per_seed: dict[str, Any] = {}
    for seed, seed_dir in checkpoints.items():
        model = _load_model(seed_dir / "response.pt", args.device)
        per_seed[seed] = {}
        for split in ("id", "ood"):
            spec = _build_spec(args, split, args.num_robots)
            per_seed[seed][split] = {}
            for policy in POLICIES:
                print(f"multientity_loop seed={seed} split={split} policy={policy.name}", flush=True)
                per_seed[seed][split][policy.name] = _multi_entity_episode_policy(
                    model, policy, spec, args.eval_episodes, args, planning
                )
    summary: dict[str, Any] = {}
    for split in ("id", "ood"):
        summary[split] = {}
        for policy in POLICIES:
            name = policy.name
            metric_names = [
                key for key, value in per_seed[next(iter(per_seed))][split][name].items()
                if isinstance(value, dict) and "mean" in value
            ]
            summary[split][name] = {}
            for metric in metric_names:
                episode_values = [
                    record[metric]
                    for seed in per_seed
                    for record in per_seed[seed][split][name]["episode_records"]
                ]
                summary[split][name][metric] = _bootstrap_ci(
                    episode_values,
                    args.bootstrap_seed
                    + int.from_bytes(
                        hashlib.sha256(f"{split}:{name}:{metric}".encode()).digest()[:4],
                        "little",
                    ),
                )
        persistent = summary[split]["persistent-joint"]
        reset = summary[split]["reset-joint"]
        additive = summary[split]["persistent-additive"]
        persistent_records = [
            record
            for seed in per_seed
            for record in per_seed[seed][split]["persistent-joint"]["episode_records"]
        ]
        reset_records = [
            record
            for seed in per_seed
            for record in per_seed[seed][split]["reset-joint"]["episode_records"]
        ]
        additive_records = [
            record
            for seed in per_seed
            for record in per_seed[seed][split]["persistent-additive"]["episode_records"]
        ]
        reset_cost_delta = [
            reset_record["realized_total_cost"] - persistent_record["realized_total_cost"]
            for persistent_record, reset_record in zip(persistent_records, reset_records)
        ]
        additive_cost_delta = [
            additive_record["realized_total_cost"] - persistent_record["realized_total_cost"]
            for persistent_record, additive_record in zip(persistent_records, additive_records)
        ]
        summary[split]["contrasts"] = {
            "reset_minus_persistent_total_cost": float(np.mean(reset_cost_delta)),
            "reset_minus_persistent_total_cost_ci95": _bootstrap_ci(
                reset_cost_delta, args.bootstrap_seed + (0 if split == "id" else 1)
            ),
            "additive_minus_joint_total_cost": float(np.mean(additive_cost_delta)),
            "additive_minus_joint_total_cost_ci95": _bootstrap_ci(
                additive_cost_delta, args.bootstrap_seed + (2 if split == "id" else 3)
            ),
            "persistent_selection_change": persistent["belief_selection_flip"]["mean"],
        }
    with (output_dir / "h3_multientity_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        records = []
        for seed, splits in per_seed.items():
            for split, policies in splits.items():
                for policy, result in policies.items():
                    for episode, record in enumerate(result["episode_records"]):
                        records.append({"seed": seed, "split": split, "policy": policy, "episode": episode, **record})
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return {"per_seed": per_seed, "summary": summary}


def _baseline_and_scale(
    args: argparse.Namespace, output_dir: Path
) -> dict[str, Any]:
    baseline_values: dict[str, list[float]] = {"constant_velocity": [], "constant_acceleration": []}
    for index in range(args.eval_episodes):
        simulator = MultiEntityIndependentSocialForceSimulator(
            _build_spec(args, "id", args.num_robots), index,
            args.num_entities, args.active_entities,
            entity_interaction_strength=args.entity_interaction_strength,
            entity_interaction_decay=args.entity_interaction_decay,
            entity_interaction_radius=args.entity_interaction_radius,
        )
        pool = simulator.candidate_pool()
        outcomes = simulator.counterfactuals(pool, PlanningCostConfig())
        active = simulator.active_mask()
        target = outcomes[0].response.numpy()
        history_velocity = (
            simulator.histories[:, -1] - simulator.histories[:, -2]
        ) / args.dt
        velocity = history_velocity[:, None]
        constant_velocity = np.broadcast_to(velocity, target.shape)
        previous_velocity = (
            simulator.histories[:, -2] - simulator.histories[:, -3]
        ) / args.dt
        initial_acceleration = (history_velocity - previous_velocity) / args.dt
        step_index = np.arange(target.shape[1], dtype=np.float32)[None, :, None]
        constant_acceleration = velocity + step_index * initial_acceleration[:, None, :]
        for name, prediction in (("constant_velocity", constant_velocity), ("constant_acceleration", constant_acceleration)):
            error = np.linalg.norm(np.cumsum(prediction * args.dt, axis=1) - np.cumsum(target * args.dt, axis=1), axis=-1)
            baseline_values[name].append(float(error[active].mean()))
    baselines = {name: _bootstrap_ci(values, args.bootstrap_seed + index) for index, (name, values) in enumerate(baseline_values.items())}
    scale: dict[str, Any] = {}
    for robots in (args.num_robots, 6):
        if robots < 2:
            continue
        start = time.perf_counter()
        finite = 0
        spread = []
        for index in range(args.scale_episodes):
            simulator = MultiEntityIndependentSocialForceSimulator(
                _build_spec(
                    args,
                    "id",
                    robots,
                    num_candidates=max(args.num_candidates, robots + 3),
                ), index,
                args.num_entities, min(args.active_entities, args.num_entities),
                entity_interaction_strength=args.entity_interaction_strength,
                entity_interaction_decay=args.entity_interaction_decay,
                entity_interaction_radius=args.entity_interaction_radius,
            )
            pool = simulator.candidate_pool()
            outcomes = simulator.counterfactuals(pool, PlanningCostConfig())
            responses = np.stack([outcome.response.numpy() for outcome in outcomes])
            if np.isfinite(responses).all():
                finite += 1
            spread.append(float(responses.reshape(len(responses), -1).std(axis=0).mean()))
        elapsed = time.perf_counter() - start
        scale[f"R{robots}"] = {
            "episodes": args.scale_episodes,
            "candidates": args.num_candidates,
            "finite_rate": finite / max(args.scale_episodes, 1),
            "mean_branch_response_spread": float(np.mean(spread)),
            "wall_seconds": elapsed,
            "ms_per_counterfactual_episode": 1000.0 * elapsed / max(args.scale_episodes, 1),
        }
    return {"protocol": "independent-social-force-baseline-scale-v1", "baselines": baselines, "scale": scale}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "h2").mkdir(exist_ok=True)
    torch.set_num_threads(max(1, int(args.num_threads)))
    metadata = {
        "protocol": "independent-social-force-multientity-suite-v1",
        "backend": MultiEntityIndependentSocialForceSimulator.backend_name,
        "device": args.device,
        "seeds": list(args.seeds),
        "data_seed": args.data_seed,
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "num_robots": args.num_robots,
        "num_entities": args.num_entities,
        "active_entities": args.active_entities,
        "windows": args.windows,
        "execute_steps": args.execute_steps,
        "num_candidates": args.num_candidates,
        "risk_aggregation": args.risk_aggregation,
        "entity_interaction_strength": args.entity_interaction_strength,
        "entity_interaction_decay": args.entity_interaction_decay,
        "entity_interaction_radius": args.entity_interaction_radius,
        "num_threads": args.num_threads,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "claim_boundary": "independent-environment robustness; no real-human causal claim",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    checkpoints = _train_models(args, output_dir)
    h2 = _h2_multientity(args, checkpoints, output_dir)
    h3 = _h3_multientity(args, checkpoints, output_dir)
    mismatch = _h1_continuous_mismatch(args, checkpoints, output_dir)
    baseline_scale = _baseline_and_scale(args, output_dir)
    payload = {
        "metadata": metadata,
        "h2": h2,
        "h3": h3,
        "continuous_mismatch": mismatch,
        "baseline_scale": baseline_scale,
    }
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Independent Multi-Entity TRACER Suite", "", "## Scope", "",
        f"Backend: `{metadata['backend']}`; device: `{metadata['device']}`; seeds: `{metadata['seeds']}`.",
        f"World: `{metadata['num_entities']}` entity IDs, `{metadata['active_entities']}` active slots, `{metadata['num_robots']}` robots, `{metadata['num_candidates']}` candidates.",
        "", "## H2 Multi-Entity Persistence", "",
    ]
    for split in ("val", "test"):
        persistent = h2["summary"][split]["persistent"]
        reset = h2["summary"][split]["reset"]
        shuffled = h2["summary"][split]["shuffled-id"]
        lines.append(
            f"- `{split}` Persistent final mode accuracy `{persistent['final_mode_accuracy']['mean']:.4f}`; "
            f"Reset `{reset['final_mode_accuracy']['mean']:.4f}`; Shuffled-ID `{shuffled['final_mode_accuracy']['mean']:.4f}`."
        )
    lines += ["", "## H3 Multi-Entity Closed Loop", ""]
    for split in ("id", "ood"):
        persistent = h3["summary"][split]["persistent-joint"]
        reset = h3["summary"][split]["reset-joint"]
        lines.append(
            f"- `{split}` Persistent-joint cost `{persistent['realized_total_cost']['mean']:.4f}`; "
            f"Reset-joint `{reset['realized_total_cost']['mean']:.4f}`; "
            f"reset-minus-persistent `{h3['summary'][split]['contrasts']['reset_minus_persistent_total_cost']:.4f}` "
            f"CI `[{h3['summary'][split]['contrasts']['reset_minus_persistent_total_cost_ci95']['ci95_low']:.4f}, "
            f"{h3['summary'][split]['contrasts']['reset_minus_persistent_total_cost_ci95']['ci95_high']:.4f}]`; "
            f"selection changes `{persistent['belief_selection_flip']['mean']:.4f}`."
        )
        lines.append(
            f"  Persistent task progress `{persistent['task_progress']['mean']:.4f}`; "
            f"Reset task progress `{reset['task_progress']['mean']:.4f}`; "
            f"Persistent collision rate `{persistent['collision_rate']['mean']:.4f}`."
        )
    mismatch_aggregate = mismatch["aggregate"]
    lines += ["", "## Continuous Parameter Mismatch", "", f"- Joint ADE `{mismatch_aggregate['joint']['ade']['mean']:.4f}`; additive ADE `{mismatch_aggregate['additive']['ade']['mean']:.4f}`; additive-minus-joint `{mismatch_aggregate['additive_minus_joint']['ade']['mean']:.4f}`.", "", "## Baselines and R=6 Scale", ""]
    for name, result in baseline_scale["baselines"].items():
        lines.append(f"- `{name}` ADE `{result['mean']:.4f}` CI `[{result['ci95_low']:.4f}, {result['ci95_high']:.4f}]`.")
    for robots, result in baseline_scale["scale"].items():
        lines.append(f"- `{robots}`: finite rate `{result['finite_rate']:.3f}`, branch spread `{result['mean_branch_response_spread']:.4f}`, `{result['ms_per_counterfactual_episode']:.2f}` ms/episode.")
    lines += ["", "## Claim Boundary", "", "These are independent-environment response, identity-persistence, mismatch and scalability diagnostics. They do not establish real-human causality, universal planner superiority, or external-data counterfactual validity.", ""]
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    config_path = output_dir / "config_frozen.yaml"
    config_path.write_text(
        "# JSON is valid YAML and preserves the exact parsed command-line configuration.\n"
        + json.dumps(_jsonable(vars(args)), indent=2)
        + "\n",
        encoding="utf-8",
    )
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device_requested": args.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit_at_start": metadata["git_commit"],
    }
    environment_path = output_dir / "environment.json"
    environment_path.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    deviations_path = output_dir / "deviations.md"
    deviations_path.write_text(
        "# Protocol Deviations\n\nNone. All configured seeds and requested suite components completed.\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "COMPLETED",
        "protocol": metadata["protocol"],
        "backend": metadata["backend"],
        "claim_boundary": metadata["claim_boundary"],
        "seeds": list(args.seeds),
        "splits": ["val", "test", "id", "ood"],
        "components": {
            "h2_multientity": "h2/*_rows.csv",
            "h3_multientity": "h3_multientity_episode.csv",
            "continuous_parameter_mismatch": "summary.json:continuous_mismatch",
            "response_baselines": "summary.json:baseline_scale.baselines",
            "r6_scalability": "summary.json:baseline_scale.scale.R6",
        },
        "artifacts": {
            path.name: _sha256(path)
            for path in (report_path, config_path, environment_path, deviations_path)
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "completion.json").write_text(
        json.dumps({"status": "COMPLETED", "finished_unix": time.time()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/independent_multientity_suite_20260902")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--data-seed", type=int, default=618731)
    parser.add_argument("--train-episodes", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--scale-episodes", type=int, default=128)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--num-entities", type=int, default=6)
    parser.add_argument("--active-entities", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.3)
    parser.add_argument("--response-noise", type=float, default=0.08)
    parser.add_argument("--entity-max-speed", type=float, default=2.2)
    parser.add_argument("--entity-interaction-strength", type=float, default=0.0)
    parser.add_argument("--entity-interaction-decay", type=float, default=0.72)
    parser.add_argument("--entity-interaction-radius", type=float, default=0.85)
    parser.add_argument("--scene-family", default="crossing")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--response-epochs", type=int, default=12)
    parser.add_argument("--recurrent-epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--recurrent-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mode-discrimination-weight", type=float, default=0.1)
    parser.add_argument("--mode-temperature", type=float, default=0.2)
    parser.add_argument("--prefix-loss-weight", type=float, default=1.0)
    parser.add_argument("--likelihood-scale", type=float, default=0.05)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument("--risk-alpha", type=float, default=0.1)
    parser.add_argument("--risk-samples", type=int, default=32)
    parser.add_argument("--risk-aggregation", choices=["union", "max", "mean"], default="union")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--bootstrap-seed", type=int, default=618731)
    parser.add_argument("--refresh-output-dir", default=None)
    args = parser.parse_args()
    if args.refresh_output_dir:
        refresh_completed_output(Path(args.refresh_output_dir))
    else:
        run(args)


if __name__ == "__main__":
    main()
