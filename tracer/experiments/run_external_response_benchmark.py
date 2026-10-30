"""Run the current paper's response/belief benchmark on external recordings.

This is intentionally a single-robot, observational degeneration of the
paper's interface.  A recording supplies one executed robot trajectory and
the observed entity response; it does not supply counterfactual joint actions.
The output therefore supports held-out-recording response prediction and
recording-local identity continuity only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tracer.data.external_response import (
    ExternalResponseEpisodeDataset,
    assign_response_modes,
    fit_response_modes,
)
from tracer.experiments.train_persistent_trajectory_response import (
    _evaluate,
    _train_recurrent,
    _train_response,
)
from tracer.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    TrajectoryRecurrentBelief,
    TrajectoryRecurrentConfig,
    stationary_reference_candidates,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _slice_arrays(arrays: dict[str, np.ndarray], limit: int) -> dict[str, np.ndarray]:
    if limit <= 0:
        return arrays
    return {
        key: value[:limit] if value.ndim >= 1 and len(value) >= limit else value
        for key, value in arrays.items()
    }


def _validate_protocol_arrays(
    arrays: dict[str, np.ndarray],
    path: Path,
    *,
    history: int,
    horizon: int,
    windows: int,
) -> None:
    expected = {
        "entity_history": (windows, history, 2),
        "robot_current": (windows, 1, 4),
        "robot_candidates": (windows, 1, horizon, 4),
        "target_future_vel": (windows, horizon, 2),
        "active_mask": (windows,),
    }
    leading_shape = None
    for key, suffix in expected.items():
        if key not in arrays:
            raise ValueError(f"{path} is missing protocol field {key}")
        shape = tuple(arrays[key].shape[-len(suffix):])
        if arrays[key].ndim != len(suffix) + 2 or shape != suffix:
            raise ValueError(
                f"{path} has {key} trailing shape {shape}, expected {suffix}; "
                "match --history/--horizon/--windows to the conversion command"
            )
        if leading_shape is None:
            leading_shape = arrays[key].shape[:2]
        elif arrays[key].shape[:2] != leading_shape:
            raise ValueError(f"{path} has unaligned episode/entity axes in {key}")


def _dataset(
    path: Path,
    modes: np.ndarray,
    limit: int,
) -> ExternalResponseEpisodeDataset:
    if limit > 0:
        arrays = _load_arrays(path)
        arrays = _slice_arrays(arrays, limit)
        temporary = path.with_name(f".{path.stem}.limited.npz")
        np.savez_compressed(temporary, **arrays)
        dataset = ExternalResponseEpisodeDataset(temporary, modes[:limit])
        temporary.unlink()
        return dataset
    return ExternalResponseEpisodeDataset(path, modes)


class _HistoryOnlyDataset(Dataset):
    """Remove the executed future action while preserving entity history."""

    def __init__(self, source: Dataset, horizon: int) -> None:
        self.source = source
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.source[index]
        sample["robot_candidates"] = stationary_reference_candidates(
            sample["robot_current"], self.horizon
        )
        return sample


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _add_history_only_results(
    results: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    history_results: dict[str, dict[str, Any]],
    history_rows: list[dict[str, Any]],
) -> None:
    names = {
        "no-belief": "history-only",
        "oracle": "history-only-oracle",
    }
    for source, target in names.items():
        results[target] = history_results[source]
    for row in history_rows:
        target = names.get(str(row["method"]))
        if target is not None:
            copied = dict(row)
            copied["method"] = target
            rows.append(copied)


def _summary_config(
    dataset: str,
    data_dir: Path,
    output_dir: Path,
    seed: int,
    data_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "protocol": "observational-single-robot-response-belief-v1",
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "model_seed": seed,
        "data_seed": data_seed,
        "num_modes": 3,
        "evaluation_splits": args.evaluation_splits,
        "history_baseline_enabled": not args.skip_history_baseline,
        "mode_discrimination_weight": args.mode_discrimination_weight,
        "mode_temperature": args.mode_temperature,
        "prefix_loss_weight": args.prefix_loss_weight,
        "velocity_residual": args.velocity_residual,
        "geometry_features": args.geometry_features,
        "dynamic_interaction_features": args.dynamic_interaction_features,
        "temporal_pooling": args.temporal_pooling,
        "robot_positions_target_relative": True,
        "training_arguments": vars(args).copy(),
        "response_architecture": "tracer",
        "response_epochs": args.response_epochs,
        "recurrent_epochs": args.recurrent_epochs,
        "response_baseline": (
            "history-only uses the same architecture and initialization with "
            "the recorded future robot action replaced by a stationary reference"
        ),
        "history": args.history,
        "horizon": args.horizon,
        "windows": args.windows,
        "execute_steps": args.execute_steps,
        "max_entities": args.max_entities,
        "claim_boundary": (
            "held-out-recording action-conditioned response prediction and "
            "recording-local identity continuity; no multi-robot "
            "counterfactual, causal, or long-term human claim"
        ),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    data_dir = Path(args.data_dir)
    train_path = data_dir / f"{args.dataset}_train.npz"
    val_path = data_dir / f"{args.dataset}_val.npz"
    test_path = data_dir / f"{args.dataset}_test.npz"
    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(path)

    train_arrays = _slice_arrays(_load_arrays(train_path), args.limit_train_episodes)
    val_arrays = _slice_arrays(_load_arrays(val_path), args.limit_eval_episodes)
    test_arrays = _slice_arrays(_load_arrays(test_path), args.limit_eval_episodes)
    for path, arrays in (
        (train_path, train_arrays),
        (val_path, val_arrays),
        (test_path, test_arrays),
    ):
        _validate_protocol_arrays(
            arrays,
            path,
            history=args.history,
            horizon=args.horizon,
            windows=args.windows,
        )
    clustering = fit_response_modes(
        train_arrays["mode_features"],
        train_arrays["entity_id"],
        train_arrays["active_mask"],
        num_modes=3,
        seed=args.data_seed,
    )
    train_modes = assign_response_modes(
        train_arrays["mode_features"],
        train_arrays["entity_id"],
        train_arrays["active_mask"],
        clustering,
    )
    val_modes = assign_response_modes(
        val_arrays["mode_features"],
        val_arrays["entity_id"],
        val_arrays["active_mask"],
        clustering,
    )
    test_modes = assign_response_modes(
        test_arrays["mode_features"],
        test_arrays["entity_id"],
        test_arrays["active_mask"],
        clustering,
    )
    identity_shuffle_available = bool(
        max(
            int(train_arrays["active_mask"].any(axis=-1).sum(axis=-1).max()),
            int(val_arrays["active_mask"].any(axis=-1).sum(axis=-1).max()),
            int(test_arrays["active_mask"].any(axis=-1).sum(axis=-1).max()),
        )
        >= 2
    )

    output_dir = Path(args.output_dir) / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mode_clustering.json").write_text(
        json.dumps(
            {key: value.tolist() for key, value in clustering.items()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = _summary_config(
        args.dataset,
        data_dir,
        output_dir,
        args.seed,
        args.data_seed,
        args,
    )
    metadata.update(
        {
            "train_episodes": int(len(train_arrays["episode_id"])),
            "val_episodes": int(len(val_arrays["episode_id"])),
            "test_episodes": int(len(test_arrays["episode_id"])),
            "identity_shuffle_available": identity_shuffle_available,
            "mode_clustering_seed": args.data_seed,
            "git": _git_state(),
            "input_sha256": {
                "train": _sha256(train_path),
                "val": _sha256(val_path),
                "test": _sha256(test_path),
            },
            "mode_counts": {
                split: np.bincount(
                    modes[mask.any(axis=-1)], minlength=3
                ).tolist()
                for split, modes, mask in (
                    ("train", train_modes, train_arrays["active_mask"]),
                    ("val", val_modes, val_arrays["active_mask"]),
                    ("test", test_modes, test_arrays["active_mask"]),
                )
            },
        }
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    train_dataset = _dataset(
        train_path, train_modes, args.limit_train_episodes
    )
    val_dataset = _dataset(val_path, val_modes, args.limit_eval_episodes)
    test_dataset = _dataset(test_path, test_modes, args.limit_eval_episodes)
    train_loader = _loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    history_train_loader = None
    if not args.skip_history_baseline:
        history_train_loader = _loader(
            _HistoryOnlyDataset(train_dataset, args.horizon),
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
        )
    val_loader = _loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    history_val_loader = None
    if not args.skip_history_baseline:
        history_val_loader = _loader(
            _HistoryOnlyDataset(val_dataset, args.horizon),
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
        )
    test_loader = _loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    history_test_loader = None
    if not args.skip_history_baseline:
        history_test_loader = _loader(
            _HistoryOnlyDataset(test_dataset, args.horizon),
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
        )

    response_config = PersistentTrajectoryModelConfig(
        history=args.history,
        horizon=args.horizon,
        num_robots=1,
        num_modes=3,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        dt=args.dt,
        architecture="tracer",
        num_heads=4,
        graph_layers=2,
        prefix_loss_weight=args.prefix_loss_weight,
        geometry_features=args.geometry_features,
        dynamic_interaction_features=args.dynamic_interaction_features,
        temporal_pooling=args.temporal_pooling,
        robot_positions_target_relative=True,
    )
    recurrent_config = TrajectoryRecurrentConfig(
        history=args.history,
        horizon=args.horizon,
        num_robots=1,
        num_modes=3,
        hidden_dim=args.recurrent_hidden_dim,
        execute_steps=args.execute_steps,
        use_entity_history=True,
        dt=args.dt,
    )
    response_model = BeliefConditionedTeamResponseModel(response_config).to(args.device)
    history_model = None
    if not args.skip_history_baseline:
        history_model = BeliefConditionedTeamResponseModel(response_config).to(
            args.device
        )
        history_model.load_state_dict(response_model.state_dict())
    recurrent_model = TrajectoryRecurrentBelief(recurrent_config).to(args.device)
    print(
        f"dataset={args.dataset} seed={args.seed} device={args.device} "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
        f"response_parameters={sum(p.numel() for p in response_model.parameters())} "
        f"recurrent_parameters={sum(p.numel() for p in recurrent_model.parameters())}",
        flush=True,
    )
    _set_seed(args.seed)
    _train_response(
        response_model,
        train_loader,
        args.device,
        args.response_epochs,
        args.lr,
        args.weight_decay,
        args.mode_discrimination_weight,
        args.mode_temperature,
        args.prefix_loss_weight,
        velocity_residual=args.velocity_residual,
        mean_trajectory_weight=getattr(args, "mean_trajectory_weight", 0.0),
        mean_endpoint_weight=getattr(args, "mean_endpoint_weight", 0.25),
    )
    if history_model is not None and history_train_loader is not None:
        _set_seed(args.seed)
        _train_response(
            history_model,
            history_train_loader,
            args.device,
            args.response_epochs,
            args.lr,
            args.weight_decay,
            args.mode_discrimination_weight,
            args.mode_temperature,
            args.prefix_loss_weight,
            velocity_residual=args.velocity_residual,
            mean_trajectory_weight=getattr(args, "mean_trajectory_weight", 0.0),
            mean_endpoint_weight=getattr(args, "mean_endpoint_weight", 0.25),
        )
    _set_seed(args.seed + 1)
    _train_recurrent(
        recurrent_model,
        train_loader,
        args.device,
        args.recurrent_epochs,
        args.recurrent_lr,
        args.weight_decay,
    )

    evaluation_kwargs = {
        "device": args.device,
        "dt": args.dt,
        "likelihood_scale": args.likelihood_scale,
        "confidence_threshold": args.confidence_threshold,
        "execute_steps": args.execute_steps,
        "velocity_residual": args.velocity_residual,
    }
    selected_loaders = {
        "val": (val_loader, history_val_loader),
        "test": (test_loader, history_test_loader),
    }
    if args.evaluation_splits != "both":
        selected_loaders = {
            args.evaluation_splits: selected_loaders[args.evaluation_splits]
        }
    results_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, (loader, history_loader) in selected_loaders.items():
        results, rows = _evaluate(
            response_model,
            recurrent_model,
            loader,
            split_name=split,
            **evaluation_kwargs,
        )
        if history_model is not None and history_loader is not None:
            history_results, history_rows = _evaluate(
                history_model,
                recurrent_model,
                history_loader,
                split_name=split,
                **evaluation_kwargs,
            )
            _add_history_only_results(results, rows, history_results, history_rows)
        if not identity_shuffle_available:
            results.pop("shuffled-id", None)
            rows = [row for row in rows if row["method"] != "shuffled-id"]
        results_by_split[split] = results
        rows_by_split[split] = rows
        (output_dir / f"{split}_metrics.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    import csv

    all_rows = [
        row
        for split in selected_loaders
        for row in rows_by_split[split]
    ]
    with (output_dir / "raw_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    torch.save(
        {
            "model": response_model.state_dict(),
            "model_config": vars(response_config),
            "velocity_residual": args.velocity_residual,
            "output_representation": "constant_velocity_residual" if args.velocity_residual else "absolute_velocity",
            "seed": args.seed,
            "data_seed": args.data_seed,
            "protocol": metadata["protocol"],
        },
        output_dir / "response.pt",
    )
    if history_model is not None:
        torch.save(
            {
                "model": history_model.state_dict(),
                "model_config": vars(response_config),
                "velocity_residual": args.velocity_residual,
                "output_representation": "constant_velocity_residual" if args.velocity_residual else "absolute_velocity",
                "seed": args.seed,
                "data_seed": args.data_seed,
                "protocol": metadata["protocol"],
                "ablation": "history-only",
            },
            output_dir / "history_only_response.pt",
        )
    torch.save(
        {
            "model": recurrent_model.state_dict(),
            "model_config": vars(recurrent_config),
            "seed": args.seed,
            "data_seed": args.data_seed,
            "protocol": metadata["protocol"],
        },
        output_dir / "recurrent.pt",
    )
    for split, result in results_by_split.items():
        for method, metrics in result.items():
            print(
                f"split={split} method={method} "
                f"nll={metrics['predictive_trajectory_nll']:.6f} "
                f"ade={metrics['trajectory_ade']:.6f} "
                f"fde={metrics['trajectory_fde']:.6f} "
                f"mode_acc={metrics['final_mode_accuracy']:.6f} "
                f"confidence_windows={metrics['windows_to_confidence']:.4f}",
                flush=True,
            )
    return {"metadata": metadata, **results_by_split}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["thor_magni", "huron_sacson", "navware", "bi3"],
    )
    parser.add_argument("--data-dir", default="experiments/external_response_data")
    parser.add_argument("--output-dir", default="experiments/external_response_benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=20260901)
    parser.add_argument("--limit-train-episodes", type=int, default=0)
    parser.add_argument("--limit-eval-episodes", type=int, default=0)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--max-entities", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--response-epochs", type=int, default=10)
    parser.add_argument("--recurrent-epochs", type=int, default=10)
    parser.add_argument(
        "--evaluation-splits",
        choices=["val", "test", "both"],
        default="both",
    )
    parser.add_argument("--skip-history-baseline", action="store_true")
    parser.add_argument("--velocity-residual", action="store_true")
    parser.add_argument("--geometry-features", action="store_true")
    parser.add_argument("--dynamic-interaction-features", action="store_true")
    parser.add_argument("--temporal-pooling", choices=["mean", "attention"], default="mean")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--recurrent-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mode-discrimination-weight", type=float, default=0.1)
    parser.add_argument("--mode-temperature", type=float, default=0.2)
    parser.add_argument("--prefix-loss-weight", type=float, default=1.0)
    parser.add_argument("--mean-trajectory-weight", type=float, default=0.0)
    parser.add_argument("--mean-endpoint-weight", type=float, default=0.25)
    parser.add_argument("--likelihood-scale", type=float, default=0.05)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
