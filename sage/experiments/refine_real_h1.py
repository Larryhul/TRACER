"""Matched continuation of real-data action and history-only predictors."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from sage.data.external_response import ExternalResponseEpisodeDataset, assign_response_modes
from sage.experiments.run_external_response_benchmark import (
    _HistoryOnlyDataset,
    _add_history_only_results,
    _load_arrays,
    _loader,
    _set_seed,
    _sha256,
    _validate_protocol_arrays,
)
from sage.experiments.train_persistent_trajectory_response import _evaluate, _train_response
from sage.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    TrajectoryRecurrentBelief,
    TrajectoryRecurrentConfig,
)


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.num_threads)
    baseline = Path(args.baseline)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    metadata = json.loads((baseline / "run_metadata.json").read_text())
    expected_hashes = metadata["input_sha256"].copy()
    if args.split not in expected_hashes:
        parent = Path(metadata["refinement"]["arguments"]["baseline"])
        expected_hashes.update(json.loads((parent / "run_metadata.json").read_text())["input_sha256"])
    original_arguments = metadata["training_arguments"]
    seed = metadata["model_seed"]
    data_dir = Path(args.data_dir or metadata["data_dir"]).resolve()
    clustering = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in json.loads((baseline / "mode_clustering.json").read_text()).items()
    }
    datasets = {}
    accessed_splits = (args.split,) if args.evaluate_only else ("train", args.split)
    for split in accessed_splits:
        path = data_dir / f"{metadata['dataset']}_{split}.npz"
        if _sha256(path) != expected_hashes[split]:
            raise ValueError(f"baseline data hash mismatch: {path}")
        arrays = _load_arrays(path)
        _validate_protocol_arrays(
            arrays, path, history=metadata["history"],
            horizon=metadata["horizon"], windows=metadata["windows"],
        )
        modes = assign_response_modes(
            arrays["mode_features"], arrays["entity_id"], arrays["active_mask"], clustering,
        )
        datasets[split] = ExternalResponseEpisodeDataset(path, modes)
        del arrays
    checkpoints = {
        name: torch.load(baseline / name, map_location="cpu", weights_only=False)
        for name in ("response.pt", "history_only_response.pt", "recurrent.pt")
    }
    if checkpoints["response.pt"]["model_config"] != checkpoints["history_only_response.pt"]["model_config"]:
        raise ValueError("paired baseline model configurations differ")
    for name in ("response.pt", "history_only_response.pt"):
        if checkpoints[name].get("velocity_residual", False) != metadata["velocity_residual"]:
            raise ValueError(f"residual representation mismatch: {name}")
    recurrent_checkpoint = checkpoints["recurrent.pt"]
    recurrent = TrajectoryRecurrentBelief(
        TrajectoryRecurrentConfig(**recurrent_checkpoint["model_config"])
    ).to(args.device)
    recurrent.load_state_dict(recurrent_checkpoint["model"])
    output.mkdir(parents=True)
    parent_refinement = metadata.get("refinement")
    metadata["refinement"] = {
        "arguments": vars(args),
        "initialization_sha256": {name: _sha256(baseline / name) for name in checkpoints},
        "recurrent_policy": "frozen baseline; not retrained for response continuation",
        "paired_training": "same loss, epochs, optimizer, RNG seed, and data order for both predictors",
        "parent_refinement": parent_refinement,
    }
    metadata["output_dir"] = str(output)
    metadata["data_dir"] = str(data_dir)
    metadata["evaluation_splits"] = args.split
    metadata["input_sha256"] = expected_hashes
    metadata["accessed_splits"] = list(accessed_splits)
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy2(baseline / "mode_clustering.json", output / "mode_clustering.json")
    shutil.copy2(baseline / "recurrent.pt", output / "recurrent.pt")
    results = {}
    for name in ("response.pt", "history_only_response.pt"):
        _set_seed(seed)
        checkpoint = checkpoints[name]
        config = PersistentTrajectoryModelConfig(**checkpoint["model_config"])
        model = BeliefConditionedTeamResponseModel(config).to(args.device)
        model.load_state_dict(checkpoint["model"])
        history_only = name == "history_only_response.pt"
        train_dataset = datasets.get("train")
        eval_dataset = datasets[args.split]
        if history_only:
            if train_dataset is not None:
                train_dataset = _HistoryOnlyDataset(train_dataset, config.horizon)
            eval_dataset = _HistoryOnlyDataset(eval_dataset, config.horizon)
        _set_seed(seed)
        if args.evaluate_only:
            shutil.copy2(baseline / name, output / name)
        else:
            print(f"training {name} weight={args.mean_trajectory_weight} seed={seed}", flush=True)
            _train_response(
                model,
                _loader(train_dataset, batch_size=args.batch_size, shuffle=True, seed=seed),
                args.device, args.epochs, args.lr, original_arguments["weight_decay"],
                original_arguments["mode_discrimination_weight"],
                original_arguments["mode_temperature"], original_arguments["prefix_loss_weight"],
                velocity_residual=metadata["velocity_residual"],
                mean_trajectory_weight=args.mean_trajectory_weight,
                mean_endpoint_weight=args.mean_endpoint_weight,
                mean_trajectory_conditioning=getattr(args, "mean_trajectory_conditioning", "uniform"),
                mean_belief_execute_steps=metadata["execute_steps"],
                mean_belief_likelihood_scale=original_arguments["likelihood_scale"],
            )
            checkpoint["model"] = model.state_dict()
            checkpoint["refinement"] = metadata["refinement"]
            torch.save(checkpoint, output / name)
        metrics, rows = _evaluate(
            model, recurrent,
            _loader(eval_dataset, batch_size=args.batch_size, shuffle=False, seed=seed),
            args.device, config.dt, original_arguments["likelihood_scale"],
            original_arguments["confidence_threshold"], metadata["execute_steps"], args.split,
            velocity_residual=metadata["velocity_residual"],
        )
        del rows
        if history_only:
            _add_history_only_results(results, [], metrics, [])
        else:
            results.update(metrics)
        (output / f"{args.split}_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({name: metrics["no-belief"]}), flush=True)
        del model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--mean-trajectory-weight", type=float, default=1.0)
    parser.add_argument("--mean-endpoint-weight", type=float, default=0.25)
    parser.add_argument("--mean-trajectory-conditioning", choices=["uniform", "causal-belief"], default="uniform")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
