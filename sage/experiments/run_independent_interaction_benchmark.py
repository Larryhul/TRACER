"""Run TRACER on an independent resettable Social-Force environment.

This is the independent-interaction evidence track for H1/H2/H3.  It trains
the response model on force-environment episodes, audits paired replay, and
then evaluates the unchanged TRACER belief/risk/selection loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sage.experiments.eval_persistent_closed_loop import (
    POLICIES,
    _aggregate_runs,
    _evaluate_policy,
)
from sage.experiments.train_persistent_trajectory_response import (
    _evaluate,
    _train_recurrent,
    _train_response,
)
from sage.response.closed_loop import ClosedLoopSpec, PlanningCostConfig
from sage.response.independent_social_force import (
    IndependentSocialForceDataset,
    IndependentSocialForceDatasetSpec,
    IndependentSocialForceSimulator,
)
from sage.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    TrajectoryRecurrentBelief,
    TrajectoryRecurrentConfig,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset: IndependentSocialForceDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _h1_audit(spec: ClosedLoopSpec, episodes: int) -> dict[str, Any]:
    cost = PlanningCostConfig()
    replay_errors: list[float] = []
    branch_spreads: list[float] = []
    state_errors: list[float] = []
    branch_effective: list[float] = []
    for episode in range(episodes):
        first = IndependentSocialForceSimulator(spec, episode)
        second = IndependentSocialForceSimulator(spec, episode)
        first_pool = first.candidate_pool()
        second_pool = second.candidate_pool()
        first_outcomes = first.counterfactuals(first_pool, cost)
        second_outcomes = second.counterfactuals(second_pool, cost)
        response_stack = np.stack([item.response.numpy() for item in first_outcomes])
        replay_errors.append(
            float(
                max(
                    np.max(np.abs(left.response.numpy() - right.response.numpy()))
                    for left, right in zip(first_outcomes, second_outcomes)
                )
            )
        )
        branch_spreads.append(float(response_stack.reshape(len(response_stack), -1).std(axis=0).mean()))
        branch_effective.append(float(np.linalg.norm(response_stack - response_stack[0], axis=-1).mean()))
        before = first.history.copy()
        _ = first.counterfactuals(first_pool, cost)
        state_errors.append(float(np.max(np.abs(before - first.history))))
    return {
        "protocol": "paired-reset-replay-v1",
        "backend": IndependentSocialForceSimulator.backend_name,
        "episodes": episodes,
        "same_initial_state_same_seed_max_abs_error": max(replay_errors, default=float("nan")),
        "same_initial_state_same_seed_mean_abs_error": float(np.mean(replay_errors)) if replay_errors else float("nan"),
        "counterfactual_state_mutation_max_abs_error": max(state_errors, default=float("nan")),
        "mean_branch_response_spread": float(np.mean(branch_spreads)) if branch_spreads else float("nan"),
        "mean_branch_effective_response_delta": float(np.mean(branch_effective)) if branch_effective else float("nan"),
        "pass": bool(
            replay_errors
            and max(replay_errors) == 0.0
            and state_errors
            and max(state_errors) == 0.0
            and float(np.mean(branch_effective)) > 1e-4
        ),
        "interpretation": (
            "The same reset state and common process-noise path reproduce every branch; "
            "nonzero branch spread confirms action-dependent interaction dynamics."
        ),
    }


def _write_policy_rows(runs: dict[str, dict[str, dict[str, Any]]], path: Path) -> None:
    fields = [
        "seed", "split", "policy", "realized_total_cost", "task_progress",
        "safety_cost", "human_cost", "min_distance", "collision_rate",
        "oracle_regret", "oracle_hit_rate", "candidate_rank_spearman",
        "candidate_rank_kendall", "belief_selection_flip", "safety_filter_rejections",
        "safety_fallback", "final_belief_entropy", "final_mode_accuracy",
        "runtime_ms_per_window",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed, splits in sorted(runs.items()):
            for split, policies in splits.items():
                for policy, result in policies.items():
                    writer.writerow({
                        "seed": seed,
                        "split": split,
                        "policy": policy,
                        **{field: result[field]["mean"] for field in fields[3:]},
                    })


def _write_report(
    output_dir: Path,
    metadata: dict[str, Any],
    h1: dict[str, Any],
    h2: dict[str, Any],
    h3: dict[str, Any],
) -> None:
    lines = [
        "# Independent Interaction TRACER Benchmark",
        "",
        "## Protocol",
        "",
        "This experiment uses an independent resettable Social-Force backend "
        f"(`{IndependentSocialForceSimulator.backend_name}`) rather than the submission simulator. "
        "The response model and closed-loop evaluator are the TRACER pipeline.",
        "",
        f"- Seeds: `{metadata['seeds']}`; train/val/test episodes per seed: "
        f"`{metadata['train_episodes']}/{metadata['val_episodes']}/{metadata['test_episodes']}`.",
        f"- Device: `{metadata['device']}`; robot count: `{metadata['num_robots']}`; "
        f"candidate count: `{metadata['num_candidates']}`.",
        "- H1 uses paired reset, common-noise counterfactuals and exact replay checks.",
        "- H2 reports held-out trajectory and belief metrics; H3 reports full-loop "
        "selection, risk filter, realized cost and oracle regret.",
        "",
        "## H1 Paired Replay",
        "",
        f"- Gate: **{'PASS' if h1['pass'] else 'FAIL'}**.",
        f"- Same-state/seed maximum response error: `{h1['same_initial_state_same_seed_max_abs_error']:.8g}`.",
        f"- Counterfactual state mutation: `{h1['counterfactual_state_mutation_max_abs_error']:.8g}`.",
        f"- Mean branch response spread: `{h1['mean_branch_response_spread']:.6f}`; "
        f"mean effective branch delta: `{h1['mean_branch_effective_response_delta']:.6f}`.",
        "",
        "## H2 Held-Out Response",
        "",
    ]
    for split, methods in h2.items():
        lines.append(f"### {split}")
        for method, metrics in methods.items():
            lines.append(
                f"- `{method}`: ADE `{metrics['trajectory_ade']:.4f}`, "
                f"FDE `{metrics['trajectory_fde']:.4f}`, "
                f"NLL `{metrics['predictive_trajectory_nll']:.4f}`, "
                f"final mode accuracy `{metrics['final_mode_accuracy']:.4f}`."
            )
    lines += ["", "## H3 Full Closed Loop", ""]
    for split, policies in h3.items():
        lines.append(f"### {split}")
        for policy in ("persistent-joint", "reset-joint", "persistent-additive", "oracle-joint"):
            metrics = policies[policy]
            lines.append(
                f"- `{policy}`: realized cost `{metrics['realized_total_cost']['mean']:.4f}`, "
                f"oracle regret `{metrics['oracle_regret']['mean']:.4f}`, "
                f"collision `{metrics['collision_rate']['mean']:.4f}`, "
                f"final belief accuracy `{metrics['final_mode_accuracy']['mean']:.4f}`."
            )
        contrast = policies["contrasts"]
        lines.append(
            f"- Core contrasts: reset-persistent cost delta "
            f"`{contrast['reset_minus_persistent_total_cost']:.4f}`; "
            f"additive-joint cost delta `{contrast['additive_minus_joint_total_cost']:.4f}`."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "This is independent-environment evidence for resettable interaction, "
        "conditional response prediction, belief adaptation and policy selection. "
        "It does not establish causal or long-term human conclusions on THÖR, HuRoN "
        "or any real population.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_spec = ClosedLoopSpec(
        windows=args.windows,
        history=args.history,
        horizon=args.horizon,
        num_robots=args.num_robots,
        num_candidates=args.num_candidates,
        execute_steps=args.execute_steps,
        dt=args.dt,
        response_noise=args.response_noise,
        influence_radius=4.0,
        entity_max_speed=args.entity_max_speed,
        scene_family=args.scene_family,
        response_family="pincer",
        seed=args.data_seed,
        split="id",
    )
    h1 = _h1_audit(base_spec, args.h1_episodes)
    (output_dir / "h1_paired_replay.json").write_text(
        json.dumps(h1, indent=2) + "\n", encoding="utf-8"
    )
    train = IndependentSocialForceDataset(IndependentSocialForceDatasetSpec(
        split="train", num_episodes=args.train_episodes, windows=args.windows,
        history=args.history, horizon=args.horizon, num_robots=args.num_robots,
        num_candidates=args.num_candidates, execute_steps=args.execute_steps,
        dt=args.dt, response_noise=args.response_noise,
        entity_max_speed=args.entity_max_speed, scene_family=args.scene_family,
        seed=args.data_seed, cache_samples=True,
    ))
    val = IndependentSocialForceDataset(IndependentSocialForceDatasetSpec(
        split="val", num_episodes=args.val_episodes, windows=args.windows,
        history=args.history, horizon=args.horizon, num_robots=args.num_robots,
        num_candidates=args.num_candidates, execute_steps=args.execute_steps,
        dt=args.dt, response_noise=args.response_noise,
        entity_max_speed=args.entity_max_speed, scene_family=args.scene_family,
        seed=args.data_seed, cache_samples=True,
    ))
    test = IndependentSocialForceDataset(IndependentSocialForceDatasetSpec(
        split="test", num_episodes=args.test_episodes, windows=args.windows,
        history=args.history, horizon=args.horizon, num_robots=args.num_robots,
        num_candidates=args.num_candidates, execute_steps=args.execute_steps,
        dt=args.dt, response_noise=args.response_noise,
        entity_max_speed=args.entity_max_speed, scene_family=args.scene_family,
        seed=args.data_seed, cache_samples=True,
    ))
    h2_by_seed: dict[str, dict[str, dict[str, Any]]] = {}
    h3_runs: dict[str, dict[str, dict[str, Any]]] = {}
    planning = PlanningCostConfig(
        task_weight=1.0, safety_weight=2.0, human_weight=0.6,
        comfort_scale=1.0, collision_margin=args.collision_margin,
        task_steps=args.execute_steps, violation_probability_alpha=args.risk_alpha,
        risk_samples=args.risk_samples,
    )
    for model_seed in args.seeds:
        print(f"independent_seed={model_seed} device={device}", flush=True)
        _set_seed(model_seed)
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
        response_model = BeliefConditionedTeamResponseModel(response_config).to(device)
        recurrent_model = TrajectoryRecurrentBelief(recurrent_config).to(device)
        _train_response(
            response_model, _loader(train, args.batch_size, True, model_seed), device,
            args.response_epochs, args.lr, args.weight_decay,
            args.mode_discrimination_weight, args.mode_temperature,
            args.prefix_loss_weight,
        )
        _train_recurrent(
            recurrent_model, _loader(train, args.batch_size, True, model_seed + 1),
            device, args.recurrent_epochs, args.recurrent_lr, args.weight_decay,
        )
        seed_dir = output_dir / f"seed{model_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": response_model.state_dict(),
            "model_config": vars(response_config),
            "seed": model_seed,
            "protocol": "independent-social-force-tracer-v2",
        }, seed_dir / "response.pt")
        torch.save({
            "model": recurrent_model.state_dict(),
            "model_config": vars(recurrent_config),
            "seed": model_seed,
            "protocol": "independent-social-force-tracer-v2",
        }, seed_dir / "recurrent.pt")
        h2_by_seed[str(model_seed)] = {}
        for split, dataset in (("val", val), ("test", test)):
            results, _ = _evaluate(
                response_model, recurrent_model,
                _loader(dataset, args.batch_size, False, model_seed),
                device=device, dt=args.dt, likelihood_scale=args.likelihood_scale,
                confidence_threshold=args.confidence_threshold,
                execute_steps=args.execute_steps, split_name=split,
            )
            h2_by_seed[str(model_seed)][split] = results
            (seed_dir / f"{split}_metrics.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
        h3_runs[str(model_seed)] = {}
        for split in ("id", "ood"):
            spec = ClosedLoopSpec(
                windows=args.windows, history=args.history, horizon=args.horizon,
                num_robots=args.num_robots, num_candidates=args.num_candidates,
                execute_steps=args.execute_steps, dt=args.dt,
                response_noise=args.response_noise, influence_radius=4.0,
                entity_max_speed=args.entity_max_speed, scene_family=args.scene_family,
                response_family="pincer", seed=args.data_seed, split=split,
            )
            h3_runs[str(model_seed)][split] = {}
            for policy in POLICIES:
                print(f"independent_loop seed={model_seed} split={split} policy={policy.name}", flush=True)
                h3_runs[str(model_seed)][split][policy.name] = _evaluate_policy(
                    response_model, policy, spec, args.loop_episodes, device,
                    planning, args.likelihood_scale,
                    candidate_adapter=None, recurrent_model=recurrent_model,
                    simulator_factory=IndependentSocialForceSimulator,
                )
            (seed_dir / f"{split}_closed_loop.json").write_text(
                json.dumps(h3_runs[str(model_seed)][split], indent=2) + "\n",
                encoding="utf-8",
            )
    h2_summary: dict[str, dict[str, Any]] = {}
    for split in ("val", "test"):
        h2_summary[split] = {}
        for method in next(iter(h2_by_seed.values()))[split]:
            h2_summary[split][method] = {}
            for metric in h2_by_seed[next(iter(h2_by_seed))][split][method]:
                values = [h2_by_seed[seed][split][method][metric] for seed in h2_by_seed]
                if isinstance(values[0], (float, int)):
                    h2_summary[split][method][metric] = float(np.mean(values))
                else:
                    h2_summary[split][method][metric] = values[0]
    h3_summary = _aggregate_runs(h3_runs)
    metadata = {
        "protocol": "independent-social-force-tracer-v2",
        "backend": IndependentSocialForceSimulator.backend_name,
        "seeds": args.seeds, "data_seed": args.data_seed,
        "train_episodes": args.train_episodes, "val_episodes": args.val_episodes,
        "test_episodes": args.test_episodes, "loop_episodes": args.loop_episodes,
        "num_robots": args.num_robots, "num_candidates": args.num_candidates,
        "scene_family": args.scene_family, "device": device,
        "created_unix": time.time(),
        "claim_boundary": "independent interaction, conditional response and closed-loop selection only",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "h2_summary.json").write_text(
        json.dumps(h2_summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "h3_summary.json").write_text(
        json.dumps(h3_summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_policy_rows(h3_runs, output_dir / "h3_runs.csv")
    _write_report(output_dir, metadata, h1, h2_summary, h3_summary)
    return {"metadata": metadata, "h1": h1, "h2": h2_summary, "h3": h3_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/independent_social_force_tracer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029])
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--data-seed", type=int, default=418731)
    parser.add_argument("--train-episodes", type=int, default=2048)
    parser.add_argument("--val-episodes", type=int, default=256)
    parser.add_argument("--test-episodes", type=int, default=256)
    parser.add_argument("--loop-episodes", type=int, default=256)
    parser.add_argument("--h1-episodes", type=int, default=256)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.3)
    parser.add_argument("--response-noise", type=float, default=0.08)
    parser.add_argument("--entity-max-speed", type=float, default=2.2)
    parser.add_argument("--scene-family", choices=["radial", "corridor", "crossing", "bottleneck"], default="crossing")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--response-epochs", type=int, default=20)
    parser.add_argument("--recurrent-epochs", type=int, default=25)
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
    parser.add_argument("--num-threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
