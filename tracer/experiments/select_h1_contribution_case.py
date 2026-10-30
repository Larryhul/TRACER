from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tracer.experiments import plot_h1_counterfactual_trajectories as source


OUTPUT = source.ROOT / "reports/paper_trajectories_refined_20260908"


def second_difference(values):
    return values[:, 3] - values[:, 1] - values[:, 2] + values[:, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    raw = source._load_yaml(source.DEFAULT_CONFIG)
    data_seed = int(raw["formal_protocol"]["formal_test_data_seed"])
    seeds = [int(seed) for seed in raw["suite"]["seeds"]]
    dataset = source.make_causal_response_dataset_from_config(raw, split="test", effect_mode="nonlinear", seed=data_seed)
    device = torch.device(args.device)
    model_paths = {variant: [source.DEFAULT_CHECKPOINT_ROOT / f"seed{seed}" / f"{variant}.pt" for seed in seeds]
                   for variant in ["joint", "matched-additive"]}
    models = {variant: [source._load_model(path, variant, device) for path in paths]
              for variant, paths in model_paths.items()}
    truths, predictions = [], {variant: [] for variant in models}
    radius = float(raw.get("evaluation", {}).get("interaction_radius", 4.0))
    for start in range(0, dataset.num_base_scenes, 16):
        stop = min(start + 16, dataset.num_base_scenes)
        samples = [dataset[scene * dataset.interventions_per_scene + branch]
                   for scene in range(start, stop) for branch in range(4)]
        scene = source.collate_synthetic_scene_batch(samples)
        batch = source.build_entity_response_batch(scene, num_robots=dataset.num_robots,
                                                  interaction_radius=radius)
        shape = (stop - start, 4, dataset.num_entities, dataset.horizon, 2)
        truths.append(batch.target_future_vel.numpy().reshape(shape))
        with torch.inference_mode():
            moved = batch.to(device)
            for variant, ensemble in models.items():
                values = np.stack([model(moved)["mean"].detach().cpu().numpy().reshape(shape) for model in ensemble])
                predictions[variant].append(values)
        print(f"H1 screened scenes {stop}/{dataset.num_base_scenes}", flush=True)
    truth = np.concatenate(truths)
    per_seed = {variant: np.concatenate(values, axis=1) for variant, values in predictions.items()}
    prediction = {variant: values.mean(axis=0) for variant, values in per_seed.items()}
    true_interaction = second_difference(truth)
    magnitude = np.sqrt(np.mean(true_interaction ** 2, axis=(-1, -2)))
    residual_error = {variant: np.sqrt(np.mean((second_difference(values) - true_interaction) ** 2, axis=(-1, -2)))
                      for variant, values in prediction.items()}
    ade = {variant: np.linalg.norm(np.cumsum((values - truth) * dataset.dt, axis=-2), axis=-1).mean(axis=-1)
           for variant, values in prediction.items()}
    per_seed_error = {variant: np.sqrt(np.mean((values[:, :, 3] - values[:, :, 1] - values[:, :, 2]
                                              + values[:, :, 0] - true_interaction[None]) ** 2, axis=(-1, -2)))
                      for variant, values in per_seed.items()}
    true_positions = np.cumsum(truth * dataset.dt, axis=-2)
    joint_positions = np.cumsum(prediction["joint"] * dataset.dt, axis=-2)
    additive_positions = np.cumsum(prediction["matched-additive"] * dataset.dt, axis=-2)
    endpoint = true_positions[:, :, :, -1, :]
    branch_endpoint_spread = np.linalg.norm(
        endpoint[:, :, :, None, :] - endpoint[:, :, None, :, :], axis=-1
    ).mean(axis=(1, 3))
    bd_truth_separation = np.linalg.norm(
        true_positions[:, 1, :, -1] - true_positions[:, 3, :, -1], axis=-1
    )
    bd_model_gap = np.linalg.norm(
        joint_positions[:, 1, :, -1] - additive_positions[:, 1, :, -1], axis=-1
    ) + np.linalg.norm(
        joint_positions[:, 3, :, -1] - additive_positions[:, 3, :, -1], axis=-1
    )
    visual_low = float(np.quantile(bd_truth_separation, 0.65))
    model_gap_low = float(np.quantile(bd_model_gap, 0.65))
    low, high = np.quantile(magnitude, [0.75, 0.975])
    rows = []
    for scene in range(dataset.num_base_scenes):
        for entity in range(dataset.num_entities):
            joint_error = float(residual_error["joint"][scene, entity])
            additive_error = float(residual_error["matched-additive"][scene, entity])
            joint_ade = float(ade["joint"][scene, :, entity].mean())
            additive_ade = float(ade["matched-additive"][scene, :, entity].mean())
            seed_wins = int(np.sum(per_seed_error["joint"][:, scene, entity] < per_seed_error["matched-additive"][:, scene, entity]))
            eligible = bool(low <= magnitude[scene, entity] <= high and seed_wins >= 4
                            and joint_error < additive_error and joint_ade < additive_ade)
            visual_eligible = bool(bd_truth_separation[scene, entity] >= visual_low
                                   and bd_model_gap[scene, entity] >= model_gap_low)
            eligible = eligible and visual_eligible
            contribution_gain = (additive_error - joint_error) / max(additive_error, 1e-8)
            visual_score = float(bd_truth_separation[scene, entity] * bd_model_gap[scene, entity])
            score = contribution_gain * (1.0 + visual_score)
            rows.append({"scene": scene, "entity": entity, "true_interaction_rms": float(magnitude[scene, entity]),
                         "joint_interaction_rmse": joint_error, "additive_interaction_rmse": additive_error,
                         "joint_mean_ADE_m": joint_ade, "additive_mean_ADE_m": additive_ade,
                         "joint_better_branches": int(np.sum(ade["joint"][scene, :, entity] < ade["matched-additive"][scene, :, entity])),
                         "joint_better_seeds": seed_wins, "bd_truth_separation_m": float(bd_truth_separation[scene, entity]),
                         "bd_model_gap_m": float(bd_model_gap[scene, entity]), "branch_endpoint_spread_m": float(branch_endpoint_spread[scene, entity]),
                         "visual_eligible": visual_eligible, "eligible": eligible, "relative_residual_gain": contribution_gain,
                         "selection_score": score})
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("No contribution case meets the recorded selection rule")
    selected = max(eligible, key=lambda row: (row["selection_score"], row["relative_residual_gain"], -row["scene"], -row["entity"]))
    scene, entity = selected["scene"], selected["entity"]
    samples = [dataset[scene * dataset.interventions_per_scene + branch] for branch in range(4)]
    np.savez_compressed(args.output_dir / "h1_selected_data.npz",
                        target_velocity=truth[scene], joint_velocity=prediction["joint"][scene],
                        matched_additive_velocity=prediction["matched-additive"][scene],
                        target_history=samples[0]["history"][2 + entity].numpy(),
                        robot_history=samples[0]["history"][:2].numpy(),
                        robot_branch_velocity=np.stack([sample["future"][:2].numpy() for sample in samples]),
                        agent_current=samples[0]["current_pos"].numpy(), dt=np.asarray(dataset.dt),
                        joint_per_seed=per_seed["joint"][:, scene],
                        additive_per_seed=per_seed["matched-additive"][:, scene])
    with (args.output_dir / "h1_selection_pool.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "selected": selected, "entity_index": entity, "base_scene_index": scene,
        "base_scene_id": int(samples[0]["base_scene_id"]), "formal_test_data_seed": data_seed,
        "model_seeds": seeds, "total_candidates": len(rows), "eligible_candidates": len(eligible),
        "selected_interaction_percentile": float(np.mean(magnitude <= magnitude[scene, entity])),
        "selection_rule": f"Post-hoc contribution/visibility example. True interaction magnitude between the 75th and 97.5th percentile; Joint interaction RMSE better in >=4/5 seeds; ensemble mean ADE and interaction RMSE both improve; b-vs-d truth endpoint separation >= {visual_low:.5f}m and b/d Joint-vs-Additive endpoint gap >= {model_gap_low:.5f}m (both 65th-percentile screen thresholds). Rank eligible cases by residual gain times visual score, then stable order.",
        "boundary": "Outcome-informed illustrative selection from the formal nonlinear test split, not a typical case or unbiased estimate. All four interventions and both models retained; full screening CSV supplied. No universal Joint superiority claim.",
        "config": str(source.DEFAULT_CONFIG), "config_sha256": source._sha256(source.DEFAULT_CONFIG),
        "checkpoints": {str(path): source._sha256(path) for paths in model_paths.values() for path in paths},
        "code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                        [Path(__file__), Path(source.__file__), source.ROOT / "tracer/response/models.py", source.ROOT / "tracer/response/causal_synthetic.py"]},
    }
    (args.output_dir / "h1_selection.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["selected"], indent=2))


if __name__ == "__main__":
    main()
