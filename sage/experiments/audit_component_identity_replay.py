"""Replay repaired identity controls on frozen real-data response checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sage.data.external_response import ExternalResponseEpisodeDataset, assign_response_modes, stable_entity_id
from sage.experiments.closed_loop_protocol import prepare_manifest
from sage.experiments.eval_component_factorial import _local_sources
from sage.experiments.formal_manifest import sha256_file
from sage.experiments.run_external_response_benchmark import _load_arrays, _loader, _set_seed, _validate_protocol_arrays
from sage.experiments.train_persistent_trajectory_response import _evaluate
from sage.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig, TrajectoryRecurrentBelief, TrajectoryRecurrentConfig


METRICS = ("trajectory_ade", "trajectory_fde", "predictive_trajectory_nll", "decision_brier")
METHODS = ("no-belief", "reset", "persistent", "shuffled-id", "recurrent", "oracle")


def recording_scores(rows, entity_recordings):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not row["main_evaluation"] or row["method"] not in METHODS:
            continue
        recording = entity_recordings.get(row["entity_id"])
        if recording is None:
            raise ValueError("active entity is absent from the recording manifest")
        for metric in METRICS:
            grouped[row["method"]][metric, recording].append(row[metric])
    return {
        method: {metric: {recording: float(np.mean(values[metric, recording])) for current_metric, recording in values if current_metric == metric} for metric in METRICS}
        for method, values in grouped.items()
    }


def aggregate_contrasts(results, dataset):
    records = []
    for comparator in ("reset", "shuffled-id", "recurrent", "no-belief"):
        for metric in METRICS:
            per_seed = []
            units = None
            for result in results:
                scores = result["recording_scores"]
                grouped = defaultdict(list)
                for recording, value in scores["persistent"][metric].items():
                    unit = recording.rsplit("/", 1)[0] if dataset == "bi3" else recording
                    grouped[unit].append(value - scores[comparator][metric][recording])
                current_units = sorted(grouped)
                if units is not None and current_units != units:
                    raise ValueError("recording-cluster coverage differs between model seeds")
                units = current_units
                per_seed.append([np.mean(grouped[unit]) for unit in units])
            values = np.asarray(per_seed)
            means = values.mean(axis=0)
            bootstrap = np.random.default_rng(924623).integers(0, len(units), size=(10000, len(units)))
            records.append({
                "contrast": "persistent_minus_" + comparator, "metric": metric,
                "mean_difference": float(means.mean()),
                "descriptive_cluster_ci95": np.quantile(means[bootstrap].mean(axis=1), [.025, .975]).tolist(),
                "clusters": units, "fixed_seed_means": dict(zip([result["seed"] for result in results], values.mean(axis=1).tolist())),
            })
    return records


def evaluate_seed(baseline, split, device, batch_size):
    metadata = json.loads((baseline / "run_metadata.json").read_text())
    data_path = Path(metadata["data_dir"]) / f"{metadata['dataset']}_{split}.npz"
    if sha256_file(data_path) != metadata["input_sha256"][split]:
        raise ValueError("frozen data do not match the checkpoint metadata")
    arrays = _load_arrays(data_path)
    _validate_protocol_arrays(arrays, data_path, history=metadata["history"], horizon=metadata["horizon"], windows=metadata["windows"])
    clustering = {name: np.asarray(value, dtype=np.float32) for name, value in json.loads((baseline / "mode_clustering.json").read_text()).items()}
    modes = assign_response_modes(arrays["mode_features"], arrays["entity_id"], arrays["active_mask"], clustering)
    dataset = ExternalResponseEpisodeDataset(data_path, modes)
    split_manifest = json.loads(data_path.with_suffix(".json").read_text())
    entity_recordings = {stable_entity_id(recording, track): recording for recording in split_manifest["recording_ids"] for track in range(1, 64)}
    del arrays
    response_checkpoint = torch.load(baseline / "response.pt", map_location="cpu", weights_only=False)
    recurrent_checkpoint = torch.load(baseline / "recurrent.pt", map_location="cpu", weights_only=False)
    response = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(**response_checkpoint["model_config"])).to(device)
    response.load_state_dict(response_checkpoint["model"])
    recurrent = TrajectoryRecurrentBelief(TrajectoryRecurrentConfig(**recurrent_checkpoint["model_config"])).to(device)
    recurrent.load_state_dict(recurrent_checkpoint["model"])
    seed = metadata["model_seed"]
    _set_seed(seed)
    metrics, rows = _evaluate(
        response, recurrent, _loader(dataset, batch_size=batch_size, shuffle=False, seed=seed), device,
        response.config.dt, metadata["training_arguments"]["likelihood_scale"], metadata["training_arguments"]["confidence_threshold"],
        metadata["execute_steps"], split, velocity_residual=metadata["velocity_residual"],
    )
    original = json.loads((baseline / f"{split}_metrics.json").read_text())
    unchanged = {
        method: {metric: float(metrics[method][metric] - original[method][metric]) for metric in METRICS if metric in original[method]}
        for method in METHODS if method != "shuffled-id"
    }
    if any(abs(value) > 2e-6 for difference in unchanged.values() for value in difference.values()):
        raise ValueError("identity repair changed a non-identity-control prediction")
    return {"seed": seed, "metrics": metrics, "original_shuffled": original["shuffled-id"], "unchanged_method_differences": unchanged, "recording_scores": recording_scores(rows, entity_recordings)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-template", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2031, 2039])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("unique model seeds are required")
    torch.set_num_threads(4)
    baselines = [Path(args.baseline_template.format(seed=seed)).resolve() for seed in args.seeds]
    inputs = []
    dataset_name = None
    for seed, baseline in zip(args.seeds, baselines):
        metadata = json.loads((baseline / "run_metadata.json").read_text())
        if metadata["model_seed"] != seed or (dataset_name is not None and metadata["dataset"] != dataset_name):
            raise ValueError("baseline template mismatches the requested seeds or datasets")
        dataset_name = metadata["dataset"]
        inputs.extend(baseline / name for name in ["response.pt", "recurrent.pt", "mode_clustering.json", "run_metadata.json", f"{args.split}_metrics.json"])
        data_path = Path(metadata["data_dir"]) / f"{dataset_name}_{args.split}.npz"
        inputs.extend([data_path, data_path.with_suffix(".json")])
    output = Path(args.output_dir)
    manifest = prepare_manifest(output, {"arguments": vars(args), "protocol": "rotating_identity_keys_v2", "scope": "frozen real-data post-hoc identity-control repair; pseudo modes, no training or test selection"}, inputs, _local_sources(), False)
    results = []
    for baseline in baselines:
        result = evaluate_seed(baseline, args.split, args.device, args.batch_size)
        results.append(result)
        (output / f"seed{result['seed']}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print({"seed": result["seed"], "persistent_ade": result["metrics"]["persistent"]["trajectory_ade"], "old_shuffled_ade": result["original_shuffled"]["trajectory_ade"], "repaired_shuffled_ade": result["metrics"]["shuffled-id"]["trajectory_ade"]}, flush=True)
    changed = [filename for group in ("sources", "checkpoints") for filename, digest in manifest[group].items() if sha256_file(Path(filename)) != digest]
    summary = {"status": "COMPLETED" if not changed else "INVALIDATED", "dataset": dataset_name, "split": args.split, "changed_inputs": changed, "contrasts": aggregate_contrasts(results, dataset_name), "model_seeds": args.seeds, "scope": "descriptive independent recording/experiment clusters after averaging fixed seeds; not new heldout confirmation"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if changed:
        raise RuntimeError("identity replay inputs changed")


if __name__ == "__main__":
    main()
