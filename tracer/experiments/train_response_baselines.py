from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import BatchSampler, DataLoader

from tracer.data.socialgym_adapter import collate_socialgym_scene_batch, make_socialgym_dataset_from_config
from tracer.data.synthetic_navigation import collate_synthetic_scene_batch, make_synthetic_dataset_from_config
from tracer.response.causal_synthetic import make_causal_response_dataset_from_config
from tracer.response.experiment import evaluate_model, train_epoch
from tracer.response.models import ResponseModelConfig, TeamResponsePredictor, build_response_predictor
from tracer.response.external_baselines import external_method_metadata


SMOKE_ONLY_SOURCES = {"synthetic", "socialgym_style"}


class _FactorialBatchSampler(BatchSampler):
    """Shuffle complete factorial scenes without separating their branches."""

    def __init__(
        self,
        num_base_scenes: int,
        interventions_per_scene: int,
        base_batch_size: int,
    ) -> None:
        if interventions_per_scene != 4:
            raise ValueError("factorial training currently requires four interventions per scene")
        if base_batch_size < 1:
            raise ValueError("base_batch_size must be positive")
        self.num_base_scenes = int(num_base_scenes)
        self.interventions_per_scene = int(interventions_per_scene)
        self.base_batch_size = int(base_batch_size)

    def __iter__(self):
        order = torch.randperm(self.num_base_scenes).tolist()
        for offset in range(0, len(order), self.base_batch_size):
            batch = []
            for base_scene_id in order[offset : offset + self.base_batch_size]:
                start = base_scene_id * self.interventions_per_scene
                batch.extend(range(start, start + self.interventions_per_scene))
            yield batch

    def __len__(self) -> int:
        return (self.num_base_scenes + self.base_batch_size - 1) // self.base_batch_size


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _make_dataset(
    raw: dict[str, Any],
    split: str,
    effect_mode: str | None = None,
    seed: int | None = None,
):
    dataset_cfg = raw.get("dataset", {})
    source = str(dataset_cfg.get("source", "synthetic")).lower()
    if source == "synthetic":
        return make_synthetic_dataset_from_config(raw, split=split), collate_synthetic_scene_batch
    if source == "causal_response":
        dataset = make_causal_response_dataset_from_config(
            raw, split=split, effect_mode=effect_mode, seed=seed
        )
        return dataset, collate_synthetic_scene_batch
    if source in {"socialgym_style", "socialgym_trajectory"}:
        socialgym_source = "style" if source == "socialgym_style" else "trajectory"
        dataset = make_socialgym_dataset_from_config(raw, split=split, source=socialgym_source)
        return dataset, collate_socialgym_scene_batch
    raise ValueError(
        f"unsupported dataset.source={source!r}; expected causal_response, synthetic, "
        "socialgym_style, or socialgym_trajectory"
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_metrics(metrics: dict[str, float]) -> str:
    keys = ("nll", "ade", "fde", "high_nll", "num_samples", "num_high_interaction")
    return " ".join(f"{key}={metrics[key]:.5g}" for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train team-action-conditioned entity response baselines.")
    parser.add_argument("--config", default="configs/response_e0.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument(
        "--variants", nargs="+",
        choices=sorted((*TeamResponsePredictor.VALID_VARIANTS, "eiffert2019-style")),
    )
    parser.add_argument("--effect-mode", choices=["none", "additive", "nonlinear"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Fixed corpus seed. Keep this identical across model seeds for paired evaluation.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse completed checkpoints and resume an interrupted variant epoch.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    raw = _load_config(args.config)
    dataset_cfg = raw.get("dataset", {})
    model_cfg = raw.get("model", {})
    training_cfg = raw.get("training", {})
    evaluation_cfg = raw.get("evaluation", {})
    source = str(dataset_cfg.get("source", "synthetic")).lower()
    if source in SMOKE_ONLY_SOURCES:
        print(
            "WARNING: this source does not contain robot-caused entity responses. "
            "Use results only to validate code/training, not as response-effect evidence."
        )

    seed = int(args.seed if args.seed is not None else training_cfg.get("seed", 2027))
    causal_cfg = dataset_cfg.get("causal_response", {})
    data_seed = int(
        args.data_seed if args.data_seed is not None else causal_cfg.get("seed", 2027)
    )
    _set_seed(seed)
    train_dataset, collate_fn = _make_dataset(raw, "train", args.effect_mode, data_seed)
    val_dataset, val_collate_fn = _make_dataset(raw, "val", args.effect_mode, data_seed)
    batch_size = int(args.batch_size or training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": args.device.startswith("cuda"),
    }
    val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=val_collate_fn, **loader_kwargs)

    config = ResponseModelConfig(
        history=int(dataset_cfg.get("history", 8)),
        horizon=int(dataset_cfg.get("horizon", 12)),
        num_robots=int(model_cfg.get("num_robots", 2)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        min_log_std=float(model_cfg.get("min_log_std", -4.0)),
        max_log_std=float(model_cfg.get("max_log_std", 2.0)),
        architecture=str(model_cfg.get("architecture", "legacy")),
        graph_layers=int(model_cfg.get("graph_layers", 2)),
        dt=float(dataset_cfg.get("dt", 0.4)),
        geometry_features=bool(model_cfg.get("geometry_features", False)),
        temporal_pooling=str(model_cfg.get("temporal_pooling", "mean")),
        dynamic_interaction_features=bool(
            model_cfg.get("dynamic_interaction_features", False)
        ),
        temporal_pair_interactions=bool(
            model_cfg.get("temporal_pair_interactions", False)
        ),
        hybrid_temporal_pair_interactions=bool(
            model_cfg.get("hybrid_temporal_pair_interactions", False)
        ),
    )
    variants = args.variants or list(training_cfg.get("variants", ["history", "nearest", "additive", "joint"]))
    epochs = int(args.epochs or training_cfg.get("epochs", 20))
    lr = float(training_cfg.get("lr", 3e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    interaction_radius = float(evaluation_cfg.get("interaction_radius", 3.0))
    output_dir = Path(args.output_dir or training_cfg.get("output_dir", "experiments/response/e0"))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, float]] = {}
    nested_residual_training = bool(
        training_cfg.get(
            "nested_residual_training",
            training_cfg.get("joint_residual_training", False),
        )
    )
    interaction_loss_weight = float(training_cfg.get("interaction_loss_weight", 0.0))
    paired_groups_per_batch = int(
        training_cfg.get("paired_groups_per_batch", max(batch_size // 4, 1))
    )
    if interaction_loss_weight < 0.0:
        raise ValueError("interaction_loss_weight must be non-negative")
    if interaction_loss_weight > 0.0 and source != "causal_response":
        raise ValueError("interaction_loss_weight requires dataset.source=causal_response")
    checkpoint_rule = str(training_cfg.get("checkpoint_rule", "final_epoch"))
    best_epoch = 0
    additive_state: dict[str, torch.Tensor] | None = None

    print(
        f"source={source} train_scenes={len(train_dataset)} val_scenes={len(val_dataset)} "
        f"history={config.history} horizon={config.horizon} selected_robots={config.num_robots} "
        f"effect_mode={args.effect_mode or 'configured'} model_seed={seed} "
        f"data_seed={data_seed} device={args.device}"
    )
    for variant in variants:
        checkpoint_path = output_dir / f"{variant}.pt"
        progress_path = output_dir / f"{variant}.progress.pt"
        if args.resume_existing and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            expected = {
                "variant": variant,
                "effect_mode": args.effect_mode,
                "seed": seed,
                "data_seed": data_seed,
            }
            mismatches = {
                key: (checkpoint.get(key), value)
                for key, value in expected.items()
                if checkpoint.get(key) != value
            }
            if checkpoint.get("model_config") != vars(config):
                mismatches["model_config"] = (checkpoint.get("model_config"), vars(config))
            if mismatches:
                raise ValueError(
                    f"cannot resume {checkpoint_path}: checkpoint mismatch {mismatches}"
                )

            model = build_response_predictor(variant, config).to(args.device)
            model.load_state_dict(checkpoint["model"])
            best_epoch = int(checkpoint.get("best_epoch", 0))
            metrics = evaluate_model(
                model,
                val_loader,
                args.device,
                config.num_robots,
                interaction_radius,
                max_batches=args.limit_val_batches,
            )
            all_results[variant] = metrics
            print(f"variant={variant} resume=completed split=val {_format_metrics(metrics)}")
            if variant == "additive":
                additive_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            if variant == "joint":
                shuffled = evaluate_model(
                    model,
                    val_loader,
                    args.device,
                    config.num_robots,
                    interaction_radius,
                    shuffle_candidates=True,
                    max_batches=args.limit_val_batches,
                )
                all_results["shuffled-joint"] = shuffled
                print(
                    f"variant=shuffled-joint resume=completed split=val "
                    f"{_format_metrics(shuffled)}"
                )
            continue

        # Use the same initialization/data-order seed for a controlled baseline comparison.
        _set_seed(seed)
        train_generator = torch.Generator().manual_seed(seed + 32452843)
        if interaction_loss_weight > 0.0:
            causal_interventions = int(causal_cfg.get("interventions_per_scene", 4))
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=_FactorialBatchSampler(
                    len(train_dataset) // causal_interventions,
                    causal_interventions,
                    paired_groups_per_batch,
                ),
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=args.device.startswith("cuda"),
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=collate_fn,
                generator=train_generator,
                **loader_kwargs,
            )
        model = build_response_predictor(variant, config).to(args.device)
        residual_module_names: tuple[str, ...] = ()
        if variant in {"matched-additive", "joint"} and nested_residual_training:
            if additive_state is None:
                raise ValueError(
                    "nested_residual_training requires additive to appear before residual variants"
                )
            missing, unexpected = model.load_state_dict(additive_state, strict=False)
            if str(config.architecture).lower() == "tracer":
                if variant == "matched-additive":
                    expected_prefixes = ("tracer_capacity_",)
                elif config.hybrid_temporal_pair_interactions:
                    expected_prefixes = (
                        "tracer_pair_",
                        "tracer_temporal_pair_",
                    )
                else:
                    expected_prefixes = ("tracer_pair_",)
                residual_names = (
                    "tracer_capacity_decoder",
                    "tracer_pair_encoder",
                    "tracer_pair_decoder",
                    "tracer_temporal_pair_decoder",
                )
            else:
                expected_prefixes = {
                    "matched-additive": ("capacity_",),
                    "joint": ("pair_",),
                }[variant]
                residual_names = (
                    "capacity_encoder",
                    "capacity_delta_decoder",
                    "pair_encoder",
                    "pair_delta_decoder",
                )
            if unexpected or any(
                not any(name.startswith(prefix) for prefix in expected_prefixes) for name in missing
            ):
                raise RuntimeError(
                    f"failed to initialize nested {variant} model: missing={missing}, unexpected={unexpected}"
                )
            for name, parameter in model.named_parameters():
                parameter.requires_grad = any(name.startswith(prefix) for prefix in expected_prefixes)
            residual_module_names = tuple(
                name for name in residual_names
                if any(name.startswith(prefix) for prefix in expected_prefixes)
            )
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
        total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"variant={variant} trainable_parameters={trainable_parameter_count} "
            f"total_parameters={total_parameter_count}"
        )
        optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
        best_state = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        best_epoch = 0
        best_metric = float("inf")
        start_epoch = 1
        if checkpoint_rule == "best_val_nll":
            initial_metrics = evaluate_model(
                model,
                val_loader,
                args.device,
                config.num_robots,
                interaction_radius,
                max_batches=args.limit_val_batches,
            )
            best_metric = float(initial_metrics["nll"])
        if args.resume_existing and progress_path.exists():
            progress = torch.load(progress_path, map_location="cpu", weights_only=True)
            expected = {
                "variant": variant,
                "effect_mode": args.effect_mode,
                "seed": seed,
                "data_seed": data_seed,
                "epochs": epochs,
                "checkpoint_rule": checkpoint_rule,
            }
            mismatches = {
                key: (progress.get(key), value)
                for key, value in expected.items()
                if progress.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"cannot resume {progress_path}: progress mismatch {mismatches}"
                )
            model.load_state_dict(progress["model"])
            optimizer.load_state_dict(progress["optimizer"])
            best_state = progress["best_state"]
            best_epoch = int(progress["best_epoch"])
            best_metric = float(progress["best_metric"])
            train_generator.set_state(progress["train_generator_state"])
            start_epoch = int(progress["completed_epoch"]) + 1
            print(
                f"variant={variant} resume=epoch completed_epoch={start_epoch - 1} "
                f"best_epoch={best_epoch}"
            )
        for epoch in range(start_epoch, epochs + 1):
            train_nll = train_epoch(
                model,
                train_loader,
                optimizer,
                args.device,
                config.num_robots,
                interaction_radius,
                max_batches=args.limit_train_batches,
                trainable_module_names=residual_module_names,
                interaction_loss_weight=(
                    interaction_loss_weight
                    if interaction_loss_weight > 0.0
                    and variant in {"additive", "matched-additive", "joint"}
                    else 0.0
                ),
                interventions_per_scene=(
                    int(causal_cfg.get("interventions_per_scene", 4))
                    if interaction_loss_weight > 0.0
                    and variant in {"additive", "matched-additive", "joint"}
                    else 1
                ),
            )
            epoch_metrics = evaluate_model(
                model,
                val_loader,
                args.device,
                config.num_robots,
                interaction_radius,
                max_batches=args.limit_val_batches,
            ) if checkpoint_rule == "best_val_nll" else None
            if epoch_metrics is not None and float(epoch_metrics["nll"]) < best_metric:
                best_metric = float(epoch_metrics["nll"])
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
            print(
                f"variant={variant} epoch={epoch}/{epochs} train_nll={train_nll:.5g}"
                + (f" val_nll={epoch_metrics['nll']:.5g}" if epoch_metrics is not None else "")
            )
            torch.save(
                {
                    "variant": variant,
                    "effect_mode": args.effect_mode,
                    "seed": seed,
                    "data_seed": data_seed,
                    "epochs": epochs,
                    "checkpoint_rule": checkpoint_rule,
                    "completed_epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_state": best_state,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "train_generator_state": train_generator.get_state(),
                },
                progress_path,
            )
        if checkpoint_rule == "best_val_nll":
            model.load_state_dict(best_state)
        metrics = evaluate_model(
            model,
            val_loader,
            args.device,
            config.num_robots,
            interaction_radius,
            max_batches=args.limit_val_batches,
        )
        all_results[variant] = metrics
        print(f"variant={variant} split=val {_format_metrics(metrics)}")
        if variant == "additive":
            additive_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        torch.save(
            {
                "model": model.state_dict(),
                "variant": variant,
                "model_config": vars(config),
                "source": source,
                "effect_mode": args.effect_mode,
                "seed": seed,
                "data_seed": data_seed,
                "nested_residual_training": nested_residual_training,
                "trainable_parameters": trainable_parameter_count,
                "total_parameters": total_parameter_count,
                "method_metadata": external_method_metadata(model, variant),
                "best_epoch": best_epoch,
                "best_val_nll": best_metric if checkpoint_rule == "best_val_nll" else None,
            },
            output_dir / f"{variant}.pt",
        )
        if variant == "joint":
            shuffled = evaluate_model(
                model,
                val_loader,
                args.device,
                config.num_robots,
                interaction_radius,
                shuffle_candidates=True,
                max_batches=args.limit_val_batches,
            )
            all_results["shuffled-joint"] = shuffled
            print(f"variant=shuffled-joint split=val {_format_metrics(shuffled)}")

    result_path = output_dir / "metrics.json"
    result_path.write_text(json.dumps(all_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "effect_mode": args.effect_mode,
        "model_seed": seed,
        "data_seed": data_seed,
        "epochs": epochs,
        "checkpoint_rule": checkpoint_rule,
        "nested_residual_training": nested_residual_training,
        "best_epoch": best_epoch,
        "train_base_scenes": getattr(train_dataset, "num_base_scenes", None),
        "validation_base_scenes": getattr(val_dataset, "num_base_scenes", None),
        "variants": variants,
        "resume_existing": args.resume_existing,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved_results={result_path}")


if __name__ == "__main__":
    main()
