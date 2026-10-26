"""Run the full TRACER software stack as off-policy external shadow replay.

THOR-MAGNI and HuRoN contain one logged robot action per scene.  The executed
action supplies observed response and belief updates.  Alternative candidates
are scored by TRACER but are never treated as observed counterfactual outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sage.belief.filter import categorical_bayes_update, uniform_log_belief
from sage.data.external_response import assign_response_modes
from sage.experiments.eval_persistent_closed_loop import (
    _candidate_and_reference_outputs,
    _load_model,
)
from sage.response.persistent_models import trajectory_mode_log_likelihood


CANDIDATE_NAMES = (
    "logged",
    "hold",
    "speed-0.50",
    "speed-0.75",
    "speed-1.25",
    "left-15deg",
    "right-15deg",
    "left-30deg",
    "right-30deg",
)
POLICIES = ("no-belief", "reset", "persistent", "persistent-no-filter", "oracle")


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _load_clustering(path: Path) -> dict[str, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: np.asarray(value, dtype=np.float32) for key, value in raw.items()}


def _candidate_velocities(logged: torch.Tensor) -> torch.Tensor:
    """Return a fixed action set from a logged velocity path (batch, horizon, 2)."""

    variants = [logged, torch.zeros_like(logged), 0.50 * logged, 0.75 * logged, 1.25 * logged]
    for degrees in (15.0, -15.0, 30.0, -30.0):
        angle = math.radians(degrees)
        rotation = logged.new_tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        variants.append(logged @ rotation.T)
    return torch.stack(variants, dim=1)


def _candidate_tensors(
    current: torch.Tensor,
    logged: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build entity-relative candidates with one shared robot velocity command."""

    batch, entities = current.shape[:2]
    logged_velocity = logged[:, 0, 0, :, 2:]
    velocities = _candidate_velocities(logged_velocity)
    count = velocities.shape[1]
    relative_position = current[:, :, 0, :2]
    positions = relative_position[:, :, None, None, :] + torch.cumsum(
        velocities[:, None] * dt, dim=3
    )
    velocity_all = velocities[:, None].expand(batch, entities, count, -1, -1)
    candidates = torch.cat([positions, velocity_all], dim=-1).unsqueeze(3)
    currents = current[:, :, None].expand(batch, entities, count, -1, -1)
    logged_positions = logged[:, 0, 0, :, :2]
    task_cost = torch.square(positions[:, 0] - logged_positions[:, None]).mean(
        dim=(-1, -2)
    )
    return currents, candidates, task_cost


def _normal_samples(count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    indices = torch.arange(count, device=device, dtype=dtype)
    u1 = ((indices + 0.5) / count).clamp(1e-5, 1.0 - 1e-5)
    u2 = ((indices * 0.61803398875 + 0.5 / count) % 1.0).clamp(1e-5, 1.0 - 1e-5)
    root_two = torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    return torch.stack(
        [root_two * torch.erfinv(2.0 * u1 - 1.0), root_two * torch.erfinv(2.0 * u2 - 1.0)],
        dim=-1,
    )


def _policy_costs(
    means: torch.Tensor,
    log_stds: torch.Tensor,
    reference_means: torch.Tensor,
    candidates: torch.Tensor,
    beliefs: torch.Tensor,
    active: torch.Tensor,
    task_cost: torch.Tensor,
    *,
    dt: float,
    collision_margin: float,
    risk_samples: int,
    task_weight: float,
    safety_weight: float,
    human_weight: float,
) -> dict[str, torch.Tensor]:
    """Aggregate per-entity predictions into one recorded-scene decision."""

    displacement = torch.cumsum(means * dt, dim=-2)
    robot_relative = candidates[..., 0, :, :2]
    distance = torch.linalg.vector_norm(
        robot_relative[:, :, :, None] - displacement, dim=-1
    )
    mode_min_distance = distance.min(dim=-1).values
    mode_safety = torch.exp(-mode_min_distance)
    safety = (mode_safety * beliefs[:, :, None]).sum(dim=-1)
    mode_human = torch.square(means - reference_means).sum(dim=-1).mean(dim=-1)
    human = (mode_human * beliefs[:, :, None]).sum(dim=-1)

    normals = _normal_samples(risk_samples, means.device, means.dtype)
    samples = means[..., None, :, :] + log_stds.exp()[..., None, :, :] * normals[
        None, None, None, None, :, None, :
    ]
    sample_positions = torch.cumsum(samples * dt, dim=-2)
    sample_distance = torch.linalg.vector_norm(
        robot_relative[:, :, :, None, None] - sample_positions, dim=-1
    )
    mode_violation = sample_distance.lt(collision_margin).any(dim=-1).float().mean(dim=-1)
    entity_violation = (mode_violation * beliefs[:, :, None]).sum(dim=-1)

    active_float = active.float()
    denominator = active_float.sum(dim=1).clamp_min(1.0)[:, None]
    safety_scene = (safety * active_float[:, :, None]).sum(dim=1) / denominator
    human_scene = (human * active_float[:, :, None]).sum(dim=1) / denominator
    active_violation = torch.where(
        active[:, :, None], entity_violation, torch.zeros_like(entity_violation)
    )
    union_violation = 1.0 - torch.prod(1.0 - active_violation.clamp(0.0, 1.0), dim=1)
    total = task_weight * task_cost + safety_weight * safety_scene + human_weight * human_scene
    return {
        "total": total,
        "task": task_cost,
        "safety": safety_scene,
        "human": human_scene,
        "violation_probability": union_violation,
    }


def _summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()) if len(array) else float("nan"),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "count": int(len(array)),
    }


@torch.no_grad()
def evaluate_seed(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    modes: np.ndarray,
    *,
    device: str,
    batch_size: int,
    dt: float,
    execute_steps: int,
    likelihood_scale: float,
    risk_alpha: float,
    collision_margin: float,
    risk_samples: int,
) -> tuple[dict[str, dict[str, float | int]], list[dict[str, Any]]]:
    model.eval()
    metrics = {
        policy: {
            name: []
            for name in (
                "response_ade", "response_fde", "mode_accuracy",
                "selection_change", "logged_candidate_rank", "predicted_cost_gain",
                "action_velocity_deviation", "selected_violation_probability",
                "safety_rejections", "safety_fallback",
            )
        }
        for policy in POLICIES
    }
    records: list[dict[str, Any]] = []
    total_episodes = len(arrays["episode_id"])
    num_modes = model.config.num_modes
    for batch_start in range(0, total_episodes, batch_size):
        batch_stop = min(batch_start + batch_size, total_episodes)
        entity_history = torch.from_numpy(
            arrays["entity_history"][batch_start:batch_stop].astype(np.float32)
        ).to(device)
        robot_current = torch.from_numpy(
            arrays["robot_current"][batch_start:batch_stop].astype(np.float32)
        ).to(device)
        logged_candidates = torch.from_numpy(
            arrays["robot_candidates"][batch_start:batch_stop].astype(np.float32)
        ).to(device)
        targets = torch.from_numpy(
            arrays["target_future_vel"][batch_start:batch_stop].astype(np.float32)
        ).to(device)
        active_mask = torch.from_numpy(
            arrays["active_mask"][batch_start:batch_stop]
        ).to(device)
        true_modes = torch.from_numpy(modes[batch_start:batch_stop]).long().to(device)
        batch, entities, windows = active_mask.shape
        uniform = uniform_log_belief(num_modes, device=device)
        log_beliefs = {
            policy: uniform[None, None].expand(batch, entities, -1).clone()
            for policy in POLICIES
        }
        for window in range(windows):
            active = active_mask[:, :, window]
            current = robot_current[:, :, window]
            logged = logged_candidates[:, :, window]
            current_pool, candidate_pool, task_cost = _candidate_tensors(current, logged, dt)
            candidate_count = candidate_pool.shape[2]
            flat_history = entity_history[:, :, window, None].expand(
                batch, entities, candidate_count, -1, -1
            ).reshape(batch * entities * candidate_count, 1, model.config.history, 2)
            flat_current = current_pool.reshape(
                batch * entities * candidate_count, 1, 1, 4
            )
            flat_candidates = candidate_pool.reshape(
                batch * entities * candidate_count, 1, 1, model.config.horizon, 4
            )
            robot_mask = torch.ones(
                flat_current.shape[:3], device=flat_current.device, dtype=torch.bool
            )
            output, reference = _candidate_and_reference_outputs(
                model,
                flat_history,
                flat_current[:, 0],
                flat_candidates[:, 0],
                robot_mask,
                True,
            )
            means = output["mean"][:, 0].reshape(
                batch, entities, candidate_count, num_modes, model.config.horizon, 2
            )
            log_stds = output["log_std"][:, 0].reshape_as(means)
            reference_means = reference["mean"][:, 0].reshape_as(means)
            target = targets[:, :, window]

            for policy in POLICIES:
                if policy == "oracle":
                    belief = torch.nn.functional.one_hot(
                        true_modes, num_classes=num_modes
                    ).float()
                else:
                    belief = log_beliefs[policy].exp()
                costs = _policy_costs(
                    means, log_stds, reference_means, candidate_pool, belief, active,
                    task_cost, dt=dt, collision_margin=collision_margin,
                    risk_samples=risk_samples, task_weight=1.0,
                    safety_weight=2.0, human_weight=0.6,
                )
                safe = costs["violation_probability"].le(risk_alpha)
                apply_filter = policy != "persistent-no-filter"
                scores = costs["total"]
                fallback = torch.zeros(batch, device=scores.device, dtype=torch.bool)
                if apply_filter:
                    masked = torch.where(safe, scores, torch.full_like(scores, float("inf")))
                    fallback = ~safe.any(dim=1)
                    selected = masked.argmin(dim=1)
                    selected[fallback] = costs["violation_probability"][fallback].argmin(dim=1)
                else:
                    selected = scores.argmin(dim=1)
                logged_score = scores[:, 0]
                selected_score = scores.gather(1, selected[:, None]).squeeze(1)
                rank = (scores < logged_score[:, None]).sum(dim=1).float() + 1.0
                velocity = candidate_pool[:, 0, :, 0, :, 2:]
                chosen_velocity = velocity[
                    torch.arange(batch, device=velocity.device), selected
                ]
                logged_velocity = velocity[:, 0]
                deviation = torch.linalg.vector_norm(
                    chosen_velocity - logged_velocity, dim=-1
                ).mean(dim=-1)
                chosen_risk = costs["violation_probability"].gather(
                    1, selected[:, None]
                ).squeeze(1)

                predicted_velocity = (
                    means[:, :, 0] * belief[:, :, :, None, None]
                ).sum(dim=2)
                predicted_position = torch.cumsum(predicted_velocity * dt, dim=-2)
                target_position = torch.cumsum(target * dt, dim=-2)
                error = torch.linalg.vector_norm(predicted_position - target_position, dim=-1)
                entity_selector = active
                metrics[policy]["response_ade"].extend(
                    error.mean(dim=-1)[entity_selector].cpu().tolist()
                )
                metrics[policy]["response_fde"].extend(
                    error[..., -1][entity_selector].cpu().tolist()
                )
                metrics[policy]["selection_change"].extend(selected.ne(0).float().cpu().tolist())
                metrics[policy]["logged_candidate_rank"].extend(rank.cpu().tolist())
                metrics[policy]["predicted_cost_gain"].extend(
                    (logged_score - selected_score).cpu().tolist()
                )
                metrics[policy]["action_velocity_deviation"].extend(deviation.cpu().tolist())
                metrics[policy]["selected_violation_probability"].extend(chosen_risk.cpu().tolist())
                metrics[policy]["safety_rejections"].extend(
                    ((~safe).sum(dim=1) if apply_filter else torch.zeros_like(rank)).float().cpu().tolist()
                )
                metrics[policy]["safety_fallback"].extend(fallback.float().cpu().tolist())
                for local_episode in range(batch):
                    records.append({
                        "episode": int(arrays["episode_id"][batch_start + local_episode]),
                        "window": window,
                        "policy": policy,
                        "selected_candidate": int(selected[local_episode]),
                        "selected_candidate_name": CANDIDATE_NAMES[int(selected[local_episode])],
                        "logged_candidate_rank": float(rank[local_episode]),
                        "predicted_cost_gain": float(logged_score[local_episode] - selected_score[local_episode]),
                        "selected_violation_probability": float(chosen_risk[local_episode]),
                        "safety_rejections": int((~safe[local_episode]).sum()) if apply_filter else 0,
                        "safety_fallback": int(fallback[local_episode]),
                    })

            observed = target[..., :execute_steps, :]
            logged_means = means[:, :, 0, :, :execute_steps]
            logged_stds = log_stds[:, :, 0, :, :execute_steps]
            likelihood = trajectory_mode_log_likelihood(
                logged_means.reshape(batch * entities, 1, num_modes, execute_steps, 2),
                logged_stds.reshape(batch * entities, 1, num_modes, execute_steps, 2),
                observed.reshape(batch * entities, 1, execute_steps, 2),
            ).reshape(batch, entities, num_modes)
            for policy in ("persistent", "persistent-no-filter", "reset"):
                prior = (
                    log_beliefs[policy]
                    if policy != "reset"
                    else uniform[None, None].expand(batch, entities, -1)
                )
                posterior = categorical_bayes_update(
                    prior, likelihood_scale * likelihood
                )
                log_beliefs[policy] = torch.where(
                    active[..., None], posterior, log_beliefs[policy]
                )

        final_active = active_mask.any(dim=-1)
        for policy in POLICIES:
            if policy == "oracle":
                accuracy = torch.ones_like(true_modes, dtype=torch.float32)
            else:
                accuracy = log_beliefs[policy].argmax(dim=-1).eq(true_modes).float()
            metrics[policy]["mode_accuracy"].extend(
                accuracy[final_active].cpu().tolist()
            )
    return (
        {
            policy: {name: _summarize(values) for name, values in policy_metrics.items()}
            for policy, policy_metrics in metrics.items()
        },
        records,
    )


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _aggregate(seed_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for policy in POLICIES:
        summary[policy] = {}
        for metric in next(iter(seed_results.values()))[policy]:
            values = [seed_results[seed][policy][metric]["mean"] for seed in seed_results]
            summary[policy][metric] = _summarize(values)
    return summary


def _report(output_dir: Path, results: dict[str, Any], metadata: dict[str, Any]) -> None:
    lines = [
        "# External TRACER Full-Stack Shadow Replay",
        "",
        "## Protocol",
        "",
        "The logged robot action is candidate 0. Eight deterministic perturbations "
        "form a common action set. TRACER runs response prediction, persistent-mode "
        "belief, probabilistic safety filtering and candidate selection. Only the "
        "logged action has an observed response, so all alternative selections remain "
        "off-policy shadow decisions.",
        "",
        f"- Datasets: `{metadata['datasets']}`; seeds: `{metadata['seeds']}`.",
        f"- Risk threshold: `{metadata['risk_alpha']}`; candidates: `{list(CANDIDATE_NAMES)}`.",
        "",
        "## Results",
        "",
    ]
    for dataset, summary in results.items():
        lines.append(f"### {dataset}")
        for policy in ("no-belief", "reset", "persistent", "persistent-no-filter", "oracle"):
            value = summary[policy]
            lines.append(
                f"- `{policy}`: ADE `{value['response_ade']['mean']:.4f}`, "
                f"final mode accuracy `{value['mode_accuracy']['mean']:.4f}`, "
                f"shadow action-change `{value['selection_change']['mean']:.4f}`, "
                f"predicted cost gain `{value['predicted_cost_gain']['mean']:.4f}`, "
                f"selected violation probability `{value['selected_violation_probability']['mean']:.4f}`."
            )
        lines.append("")
    lines += [
        "## Claim Boundary",
        "",
        "These results validate execution of the complete TRACER software stack on "
        "recorded real-world trajectories and evaluate logged-action response prediction. "
        "They do not measure the realized outcome of an unexecuted alternative action, "
        "multi-robot coordination, real-human causality, or long-term persistence.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}
    raw_dir = Path(args.data_dir)
    checkpoint_root = Path(args.checkpoint_dir)
    for dataset in args.datasets:
        arrays = _load_arrays(raw_dir / f"{dataset}_test.npz")
        dataset_results: dict[str, Any] = {}
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            seed_checkpoint = checkpoint_root / dataset / f"seed{seed}"
            model = _load_model(seed_checkpoint / "response.pt", args.device)
            clustering = _load_clustering(seed_checkpoint / "mode_clustering.json")
            modes = assign_response_modes(
                arrays["mode_features"], arrays["entity_id"], arrays["active_mask"], clustering
            )
            metrics, records = evaluate_seed(
                model, arrays, modes, device=args.device, batch_size=args.batch_size,
                dt=args.dt, execute_steps=args.execute_steps,
                likelihood_scale=args.likelihood_scale, risk_alpha=args.risk_alpha,
                collision_margin=args.collision_margin, risk_samples=args.risk_samples,
            )
            dataset_results[str(seed)] = metrics
            seed_dir = dataset_dir / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            _write_records(seed_dir / "shadow_decisions.csv", records)
            print(
                f"shadow dataset={dataset} seed={seed} "
                f"persistent_ade={metrics['persistent']['response_ade']['mean']:.5f} "
                f"change={metrics['persistent']['selection_change']['mean']:.5f}",
                flush=True,
            )
        all_results[dataset] = _aggregate(dataset_results)
        (dataset_dir / "summary.json").write_text(
            json.dumps(all_results[dataset], indent=2) + "\n", encoding="utf-8"
        )
    metadata = {
        "protocol": "external-tracer-full-stack-shadow-replay-v1",
        "datasets": args.datasets,
        "seeds": args.seeds,
        "device": args.device,
        "risk_alpha": args.risk_alpha,
        "collision_margin": args.collision_margin,
        "risk_samples": args.risk_samples,
        "candidate_names": CANDIDATE_NAMES,
        "observed_branch": "logged candidate 0 only",
        "claim_boundary": "real-recording full-stack shadow execution; no alternate-action outcomes",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(all_results, indent=2) + "\n", encoding="utf-8"
    )
    _report(output_dir, all_results, metadata)
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["thor_magni", "huron_sacson"], default=["thor_magni", "huron_sacson"])
    parser.add_argument("--data-dir", default="experiments/external_response_data")
    parser.add_argument("--checkpoint-dir", default="experiments/external_response_benchmark_formal_v2")
    parser.add_argument("--output-dir", default="experiments/external_tracer_shadow_replay")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2031, 2039, 2053, 2063])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--likelihood-scale", type=float, default=0.05)
    parser.add_argument("--risk-alpha", type=float, default=0.1)
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument("--risk-samples", type=int, default=16)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
