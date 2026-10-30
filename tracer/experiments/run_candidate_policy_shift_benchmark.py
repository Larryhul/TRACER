"""Evaluate frozen response models under held-out candidate-policy families.

Training checkpoints use the standard factorial action generator. Evaluation
keeps each base scene, latent response, and noise draw fixed while replacing
the candidate generator with wider-lateral or crossing-style trajectories.
This is a response-interface generalization diagnostic, not a real-world or
closed-loop intervention claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tracer.data.synthetic_navigation import collate_synthetic_scene_batch
from tracer.response.causal_synthetic import CausalResponseSpec, CausalTeamResponseDataset
from tracer.response.data import build_entity_response_batch
from tracer.response.metrics import response_metrics
from tracer.response.models import ResponseModelConfig, TeamResponsePredictor


POLICIES = ("standard", "wide-lateral", "crossing")
MODES = ("none", "additive", "nonlinear")
VARIANTS = ("additive", "matched-additive", "joint", "team-concat")


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-6)


def _policy_actions(
    dataset: CausalTeamResponseDataset,
    base_scene_id: int,
    base: dict[str, np.ndarray],
    policy: str,
) -> np.ndarray:
    if policy == "standard":
        return dataset._candidate_actions(base_scene_id, base["robot_current"])
    rng = np.random.default_rng(dataset.seed + int(base_scene_id) * 104729 + 918273)
    robot_current = base["robot_current"]
    base_direction = _unit(-robot_current)
    base_angle = np.arctan2(base_direction[:, 1], base_direction[:, 0])
    active = np.asarray(
        [[(branch >> robot_idx) & 1 for robot_idx in range(dataset.num_robots)] for branch in range(4)],
        dtype=bool,
    )
    if policy == "wide-lateral":
        active_offsets = rng.normal(0.0, 0.28, size=dataset.num_robots)
        yielding_offsets = rng.choice([-1.0, 1.0], size=dataset.num_robots) * rng.uniform(
            1.20, 1.60, size=dataset.num_robots
        )
        active_speeds = rng.uniform(1.65, 2.10, size=dataset.num_robots)
        yielding_speeds = rng.uniform(0.35, 0.65, size=dataset.num_robots)
        turns = rng.uniform(-0.18, 0.18, size=dataset.num_robots)
    elif policy == "crossing":
        cross_sign = np.where(np.arange(dataset.num_robots) % 2 == 0, 1.0, -1.0)
        active_offsets = cross_sign * (np.pi / 2.0) + rng.normal(0.0, 0.12, size=dataset.num_robots)
        yielding_offsets = -cross_sign * (np.pi / 2.0) + rng.normal(0.0, 0.12, size=dataset.num_robots)
        active_speeds = rng.uniform(0.95, 1.35, size=dataset.num_robots)
        yielding_speeds = rng.uniform(0.55, 0.95, size=dataset.num_robots)
        turns = rng.uniform(-0.08, 0.08, size=dataset.num_robots)
    else:
        raise ValueError(f"unknown candidate policy {policy!r}")
    offsets = np.where(active, active_offsets[None], yielding_offsets[None])
    speeds = np.where(active, active_speeds[None], yielding_speeds[None])
    steps = np.arange(dataset.horizon, dtype=np.float32)[None, None, :]
    angles = base_angle[None, :, None] + offsets[:, :, None] + turns[None, :, None] * steps
    return np.stack(
        [np.cos(angles) * speeds[:, :, None], np.sin(angles) * speeds[:, :, None]], axis=-1
    ).astype(np.float32)


def _load_checkpoint(path: Path, device: str) -> TeamResponsePredictor:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = ResponseModelConfig(**checkpoint["model_config"])
    model = TeamResponsePredictor(str(checkpoint["variant"]), config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) < 1e-9 or np.std(second) < 1e-9:
        return float("nan")
    first_rank = np.argsort(np.argsort(first)).astype(np.float64)
    second_rank = np.argsort(np.argsort(second)).astype(np.float64)
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _finite_mean(value: torch.Tensor) -> float:
    finite = value[torch.isfinite(value)]
    return float(finite.mean()) if finite.numel() else float("nan")


@torch.inference_mode()
def _evaluate(
    dataset: CausalTeamResponseDataset,
    models: dict[str, TeamResponsePredictor],
    policy: str,
    mode: str,
    seed: int,
    device: str,
    scenes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_scene_id in range(scenes):
        base = dataset._base_scene(base_scene_id)
        actions = _policy_actions(dataset, base_scene_id, base, policy)
        targets_np = dataset._entity_responses(base, actions)
        samples = [
            dataset._make_sample(
                base_scene_id,
                branch,
                base,
                actions[branch],
                targets_np[branch],
            )
            for branch in range(actions.shape[0])
        ]
        scene = collate_synthetic_scene_batch(samples)
        batch = build_entity_response_batch(scene, num_robots=dataset.num_robots).to(device)
        target = batch.target_future_vel.reshape(4, dataset.num_entities, dataset.horizon, 2)
        for variant, model in models.items():
            output = model(batch)
            metrics = response_metrics(output["mean"], output["log_std"], batch.target_future_vel, batch.dt)
            prediction = output["mean"].reshape_as(target)
            target_delta = target - target[:1]
            prediction_delta = prediction - prediction[:1]
            true_score = target_delta.square().mean(dim=(1, 2, 3)).detach().cpu().numpy()
            pred_score = prediction_delta.square().mean(dim=(1, 2, 3)).detach().cpu().numpy()
            true_order = np.argsort(-true_score)
            pred_order = np.argsort(-pred_score)
            rows.append(
                {
                    "mode": mode,
                    "policy": policy,
                    "model_seed": seed,
                    "base_scene_id": base_scene_id,
                    "variant": variant,
                    "nll": _finite_mean(metrics["nll"]),
                    "ade": _finite_mean(metrics["ade"]),
                    "fde": _finite_mean(metrics["fde"]),
                    "effect_rmse": float(torch.sqrt((prediction_delta - target_delta).square().mean())),
                    "effect_rank_spearman": _spearman(true_score, pred_score),
                    "effect_top1": float(pred_order[0] == true_order[0]),
                    "effect_top3": float(len(set(pred_order[:3]) & set(true_order[:3])) / 3.0),
                }
            )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["mode"], row["policy"], row["variant"]), []).append(row)
    metrics = ("nll", "ade", "fde", "effect_rmse", "effect_rank_spearman", "effect_top1", "effect_top3")
    summary: list[dict[str, Any]] = []
    for (mode, policy, variant), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "mode": mode,
            "policy": policy,
            "variant": variant,
            "model_seed_count": len({int(value["model_seed"]) for value in values}),
            "n_scenes": len({int(value["base_scene_id"]) for value in values}),
        }
        for metric in metrics:
            finite = np.asarray([float(value[metric]) for value in values], dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            row[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[f"{metric}_sd"] = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
        summary.append(row)
    return summary


def run(
    output_dir: Path,
    checkpoint_root: Path,
    team_concat_root: Path,
    device: str = "cuda",
    modes: tuple[str, ...] = MODES,
    policies: tuple[str, ...] = POLICIES,
    seeds: tuple[int, ...] = (2027, 2028, 2029, 2030, 2031),
    scenes: int = 256,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for mode in modes:
        for seed in seeds:
            models: dict[str, TeamResponsePredictor] = {}
            for variant in VARIANTS[:-1]:
                models[variant] = _load_checkpoint(
                    checkpoint_root / mode / f"seed{seed}" / f"{variant}.pt", device
                )
            concat_path = team_concat_root / mode / f"seed{seed}" / "team-concat.pt"
            if concat_path.exists():
                models["team-concat"] = _load_checkpoint(concat_path, device)
            else:
                models.pop("team-concat", None)
            dataset = CausalTeamResponseDataset(
                CausalResponseSpec(
                    split="test",
                    num_base_scenes=scenes,
                    interventions_per_scene=4,
                    num_robots=2,
                    num_entities=12,
                    seed=328731,
                    effect_mode=mode,
                    cache_samples=False,
                )
            )
            for policy in policies:
                print(f"candidate_policy_shift mode={mode} policy={policy} seed={seed} device={device}", flush=True)
                rows.extend(_evaluate(dataset, models, policy, mode, seed, device, scenes))
    fields = list(rows[0])
    with (output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows)
    payload = {
        "status": "completed",
        "protocol": {
            "device": device,
            "policies": list(policies),
            "modes": list(modes),
            "model_seeds": list(seeds),
            "test_scenes": scenes,
            "branches_per_scene": 4,
            "training_policy": "standard factorial candidate generator",
            "candidate_policy_claim_boundary": "held-out response-interface generalization diagnostic only",
        },
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-root", default="experiments/pilot/h1_paired_rescue_stage1_v2/checkpoints")
    parser.add_argument("--team-concat-root", default="experiments/continuous_intervention_benchmark_20260824/team_concat_checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--scenes", type=int, default=256)
    args = parser.parse_args()
    run(
        Path(args.output_dir),
        Path(args.checkpoint_root),
        Path(args.team_concat_root),
        args.device,
        tuple(args.modes),
        tuple(args.policies),
        tuple(args.seeds),
        args.scenes,
    )


if __name__ == "__main__":
    main()
