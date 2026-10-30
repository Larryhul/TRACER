"""Train and evaluate SAGE and protocol-aligned baselines on external windows.

This runner deliberately uses the same SAGE model, learned baseline models,
loss masks, and navigation metrics as the internal experiments.  It is kept
separate from the MID/ETH-UCY entrypoints because the external releases use a
different preprocessed file contract.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tracer.config import load_sage_config
from tracer.data.external_trajectory import UnifiedTrajectoryDataset, collate_unified_trajectory_batch
from tracer.evaluation.metrics import (
    best_of_navigation_metrics,
    multi_robot_metrics,
    navigation_metrics,
    social_compliance_metrics,
)
from tracer.experiments.eval_baseline_comparison import constant_acceleration_baseline, constant_velocity_baseline
from tracer.guidance.safety_social import SafetySocialGuidance
from tracer.models.baseline_models import LearnedBaselineConfig, create_learned_baseline
from tracer.models.sage_model import SAGE


METHODS = ("sage", "mid", "trajectron_plus_plus", "constant_velocity", "constant_acceleration")
LEARNED_METHODS = {"mid", "trajectron_plus_plus"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(path: Path, batch_size: int, shuffle: bool, workers: int) -> tuple[UnifiedTrajectoryDataset, DataLoader]:
    dataset = UnifiedTrajectoryDataset(path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_unified_trajectory_batch,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def _model_for(method: str, cfg, device: str):
    if method == "sage":
        return SAGE(cfg).to(device)
    return create_learned_baseline(
        method,
        cfg,
        LearnedBaselineConfig(baseline=method, latent_dim=32, kl_weight=0.01, decoder_hidden_dim=256),
    ).to(device)


def _metric_row(
    pred_vel: torch.Tensor,
    batch,
    guidance: SafetySocialGuidance,
    goal_tolerance: float = 0.5,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if pred_vel.ndim == 5:
        navigation = best_of_navigation_metrics(
            pred_vel, batch, goal_tolerance=goal_tolerance
        )
        samples = pred_vel.unbind(0)
    elif pred_vel.ndim == 4:
        navigation = navigation_metrics(
            pred_vel, batch, goal_tolerance=goal_tolerance
        )
        samples = (pred_vel,)
    else:
        raise ValueError(
            "pred_vel must have shape (B,A,T,2) or (K,B,A,T,2), "
            f"got {tuple(pred_vel.shape)}"
        )
    for key, value in navigation.items():
        metrics[key] = float(value)

    # Keep navigation metrics best-of-K, but report safety diagnostics as the
    # expectation over the same fixed stochastic sample pool.
    diagnostic_rows: list[dict[str, float]] = []
    for sample in samples:
        row: dict[str, float] = {}
        row.update(
            social_compliance_metrics(
                sample,
                batch,
                comfort=guidance.config.comfort,
                max_speed=guidance.config.max_speed,
                personal_space_margin=1.2,
                front_only=guidance.config.front_only,
                eps=guidance.config.eps,
            )
        )
        collision_margin = guidance._collision_margins(batch, sample.dtype)
        row.update(
            multi_robot_metrics(
                sample,
                batch,
                collision_margin=collision_margin,
                goal_tolerance=guidance.config.goal_tolerance,
                eps=guidance.config.eps,
            )
        )
        diagnostic_rows.append(row)
    for key in diagnostic_rows[0]:
        metrics[key] = float(
            sum(row[key] for row in diagnostic_rows) / len(diagnostic_rows)
        )
    return metrics


def train(
    method: str,
    train_path: Path,
    cfg,
    output_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    workers: int,
    seed: int,
) -> Path:
    dataset, loader = _loader(train_path, batch_size, True, workers)
    model = _model_for(method, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"train method={method} samples={len(dataset)} device={device} epochs={epochs} seed={seed}")
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        progress = tqdm(loader, ncols=96, desc=f"{method} epoch {epoch}")
        for batch in progress:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss for {method} at epoch {epoch}: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            progress.set_postfix(loss=sum(losses) / len(losses))
        print(f"method={method} epoch={epoch} loss={sum(losses) / max(1, len(losses)):.6f}")
    checkpoint = output_dir / f"{method}_seed{seed}_epoch{epochs}.pt"
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "sage_config": asdict(cfg),
        "config": asdict(cfg),
        "epoch": epochs,
        "seed": seed,
        "method": method,
        "dataset_path": str(train_path),
    }
    if method in LEARNED_METHODS:
        payload["baseline"] = method
        payload["baseline_config"] = asdict(model.baseline_config)
    torch.save(payload, checkpoint)
    print(f"saved={checkpoint}")
    return checkpoint


@torch.no_grad()
def evaluate(
    method: str,
    checkpoint: Path | None,
    test_path: Path,
    cfg,
    device: str,
    batch_size: int,
    workers: int,
    stride: int,
    guided: bool = True,
    samples: int = 1,
) -> dict[str, float]:
    dataset, loader = _loader(test_path, batch_size, False, workers)
    guidance = SafetySocialGuidance(cfg.guidance)
    if method in {"constant_velocity", "constant_acceleration"}:
        model = None
    else:
        if checkpoint is None:
            raise ValueError(f"checkpoint is required for {method}")
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = _model_for(method, cfg, device)
        model.load_state_dict(payload["model"])
        model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, ncols=96, desc=f"eval {method}"):
        batch = batch.to(device)
        if method == "constant_velocity":
            pred_vel = constant_velocity_baseline(batch)
        elif method == "constant_acceleration":
            pred_vel = constant_acceleration_baseline(batch)
        elif method == "sage":
            generated = [
                model.generate(  # type: ignore[union-attr]
                    batch,
                    guided=guided,
                    step_stride=stride,
                )["future_vel"]
                for _ in range(max(1, int(samples)))
            ]
            pred_vel = generated[0] if int(samples) <= 1 else torch.stack(generated, dim=0)
        else:
            pred_vel = model.generate(  # type: ignore[union-attr]
                batch,
                samples=max(1, int(samples)),
                step_stride=stride,
            )["future_vel"]
            if int(samples) <= 1:
                pred_vel = pred_vel[0]
        row = _metric_row(pred_vel, batch, guidance, guidance.config.goal_tolerance)
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + value * batch.batch_size
        count += batch.batch_size
    summary = {key: value / max(1, count) for key, value in sorted(totals.items())}
    summary["n_samples"] = float(count)
    print(f"evaluated method={method} split=test samples={count}")
    for key, value in summary.items():
        print(f"{method} {key}={value:.6f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate SAGE and baselines on unified external data.")
    parser.add_argument("--dataset", required=True, choices=["uf_retail", "thor_magni", "huron_sacson"])
    parser.add_argument("--data-dir", default="experiments/external_sage_data")
    parser.add_argument("--output-dir", default="experiments/external_sage_benchmark")
    parser.add_argument("--report-dir", default="reports/external_validation_20260825/external_sage_portability")
    parser.add_argument("--config", default="configs/tracer.yaml")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Fixed stochastic sample count; >1 uses best-of-K navigation metrics and mean safety diagnostics.",
    )
    parser.add_argument(
        "--unguided",
        action="store_true",
        help="Disable SAGE navigation guidance during inference (for validation-only selection audits).",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.dataset
    train_path = data_dir / f"{args.dataset}_train.npz"
    test_path = data_dir / f"{args.dataset}_test.npz"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"build adapter output first: {train_path} and {test_path}")
    cfg, raw_cfg = load_sage_config(args.config)
    with np.load(train_path, allow_pickle=False) as data:
        cfg.diffusion.horizon = int(data["future"].shape[2])
    records: dict[str, Any] = {
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "device": args.device,
        "seed": args.seed,
        "epochs": args.epochs,
        "samples": args.samples,
        "methods": {},
        "protocol": {
            "history": int(np.load(train_path, allow_pickle=False)["history"].shape[2]),
            "horizon": cfg.diffusion.horizon,
            "train_split": str(train_path),
            "test_split": str(test_path),
            "claim_boundary": "cross-dataset forecasting/behavior-response compatibility only; no real-human causal or persistent-effect conclusion",
        },
    }
    for method in args.methods:
        checkpoint = None
        if method not in {"constant_velocity", "constant_acceleration"}:
            checkpoint = output_dir / f"{method}_seed{args.seed}_epoch{args.epochs}.pt"
            if not args.skip_training:
                checkpoint = train(method, train_path, cfg, output_dir, args.device, args.epochs, args.batch_size, args.lr, args.num_workers, args.seed)
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
        metrics = evaluate(
            method,
            checkpoint,
            test_path,
            cfg,
            args.device,
            args.batch_size,
            args.num_workers,
            args.stride,
            guided=not args.unguided,
            samples=args.samples,
        )
        records["methods"][method] = {"checkpoint": str(checkpoint) if checkpoint else None, "metrics": metrics}
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{args.dataset}_sage_baseline.json"
    report_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
