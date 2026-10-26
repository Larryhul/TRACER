from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from sage.data.synthetic_navigation import collate_synthetic_scene_batch
from sage.response.causal_synthetic import make_causal_response_dataset_from_config
from sage.response.data import build_entity_response_batch
from sage.response.experiment import evaluate_model
from sage.response.factorial import factorial_second_difference
from sage.response.metrics import gaussian_nll_per_sample
from sage.response.models import ResponseModelConfig, TeamResponsePredictor


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _initialize_nested_joint(
    additive_checkpoint_path: Path,
    device: str,
) -> tuple[TeamResponsePredictor, dict[str, Any]]:
    checkpoint = torch.load(additive_checkpoint_path, map_location=device, weights_only=True)
    model = TeamResponsePredictor(
        "joint",
        ResponseModelConfig(**checkpoint["model_config"]),
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected or any(not name.startswith("pair_") for name in missing):
        raise RuntimeError(
            f"failed to initialize nested joint model: missing={missing}, unexpected={unexpected}"
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("pair_")
    return model, checkpoint


def _train_epoch(
    model: TeamResponsePredictor,
    dataset,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_robots: int,
    interaction_radius: float,
    base_batch_size: int,
    pair_weight: float,
    epoch_seed: int,
) -> dict[str, float]:
    model.train()
    # Independent dropout masks across the four interventions create a fake
    # second difference in the frozen additive trunk. The factorial objective
    # must see the same deterministic main effects that are used at inference.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    generator = torch.Generator().manual_seed(epoch_seed)
    base_order = torch.randperm(dataset.num_base_scenes, generator=generator).tolist()
    nll_sum = 0.0
    pair_sum = 0.0
    num_entities = 0
    num_interactions = 0

    for offset in range(0, len(base_order), base_batch_size):
        base_ids = base_order[offset : offset + base_batch_size]
        samples = []
        for base_id in base_ids:
            first = base_id * dataset.interventions_per_scene
            samples.extend(dataset[first + intervention] for intervention in range(4))
        scene = collate_synthetic_scene_batch(samples).to(device)
        response = build_entity_response_batch(
            scene,
            num_robots=num_robots,
            interaction_radius=interaction_radius,
        )
        prediction = model(response)
        per_sample_nll = gaussian_nll_per_sample(
            prediction["mean"],
            prediction["log_std"],
            response.target_future_vel,
        )
        entities_per_intervention = response.num_samples // (len(base_ids) * 4)
        structured_shape = (
            len(base_ids),
            4,
            entities_per_intervention,
            dataset.horizon,
            2,
        )
        predicted_interaction = factorial_second_difference(
            prediction["mean"].reshape(structured_shape), dim=1
        )
        target_interaction = factorial_second_difference(
            response.target_future_vel.reshape(structured_shape), dim=1
        )
        pair_loss = F.mse_loss(predicted_interaction, target_interaction)
        loss = per_sample_nll.mean() + pair_weight * pair_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
        optimizer.step()

        nll_sum += float(per_sample_nll.detach().sum())
        pair_sum += float(pair_loss.detach()) * predicted_interaction.shape[0]
        num_entities += response.num_samples
        num_interactions += predicted_interaction.shape[0]

    return {
        "nll": nll_sum / num_entities,
        "pair_mse": pair_sum / num_interactions,
        "total": nll_sum / num_entities + pair_weight * pair_sum / num_interactions,
    }


def train_run(
    raw: dict[str, Any],
    additive_checkpoint_path: Path,
    run_dir: Path,
    effect_mode: str,
    seed: int,
    device: str,
    epochs: int,
    pair_weight: float,
) -> dict[str, Any]:
    _set_seed(seed)
    model_cfg = raw.get("model", {})
    training_cfg = raw.get("training", {})
    evaluation_cfg = raw.get("evaluation", {})
    train_dataset = make_causal_response_dataset_from_config(
        raw,
        split="train",
        effect_mode=effect_mode,
        seed=seed,
    )
    val_dataset = make_causal_response_dataset_from_config(
        raw,
        split="val",
        effect_mode=effect_mode,
        seed=seed,
    )
    if train_dataset.interventions_per_scene != 4:
        raise ValueError("factorial auxiliary training requires exactly four interventions")

    model, additive_checkpoint = _initialize_nested_joint(additive_checkpoint_path, device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training_cfg.get("lr", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    scene_batch_size = int(training_cfg.get("batch_size", 32))
    base_batch_size = max(1, scene_batch_size // 4)
    num_robots = int(model_cfg.get("num_robots", 2))
    interaction_radius = float(evaluation_cfg.get("interaction_radius", 4.0))

    history = []
    for epoch in range(1, epochs + 1):
        metrics = _train_epoch(
            model,
            train_dataset,
            optimizer,
            device,
            num_robots,
            interaction_radius,
            base_batch_size,
            pair_weight,
            seed + epoch * 104729,
        )
        history.append({"epoch": epoch, **metrics})
        print(
            f"mode={effect_mode} seed={seed} epoch={epoch}/{epochs} "
            f"nll={metrics['nll']:.6g} pair_mse={metrics['pair_mse']:.6g} "
            f"total={metrics['total']:.6g}",
            flush=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=scene_batch_size,
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=collate_synthetic_scene_batch,
        pin_memory=device.startswith("cuda"),
    )
    validation = evaluate_model(
        model,
        val_loader,
        device,
        num_robots,
        interaction_radius,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(additive_checkpoint_path, run_dir / "additive.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "variant": "joint",
            "model_config": additive_checkpoint["model_config"],
            "source": additive_checkpoint["source"],
            "effect_mode": effect_mode,
            "seed": seed,
            "joint_residual_training": True,
            "factorial_pair_weight": pair_weight,
        },
        run_dir / "joint.pt",
    )
    result = {
        "effect_mode": effect_mode,
        "seed": seed,
        "pair_weight": pair_weight,
        "epochs": epochs,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "final_training": history[-1],
        "validation": validation,
    }
    (run_dir / "aux_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"saved_run={run_dir} val_nll={validation['nll']:.6g} "
        f"val_ade={validation['ade']:.6g}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train joint response with a paired 2x2 auxiliary loss.")
    parser.add_argument("--config", default="configs/response_causal_suite.yaml")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modes", nargs="+", choices=["none", "additive", "nonlinear"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pair-weight", type=float, required=True)
    args = parser.parse_args()
    if args.pair_weight < 0.0:
        raise ValueError("pair-weight must be non-negative")

    raw = _load_config(args.config)
    suite_cfg = raw.get("suite", {})
    training_cfg = raw.get("training", {})
    modes = args.modes or list(suite_cfg.get("modes", ["none", "additive", "nonlinear"]))
    seeds = args.seeds or [int(seed) for seed in suite_cfg.get("seeds", [2027, 2028, 2029])]
    epochs = int(args.epochs or training_cfg.get("epochs", 30))
    checkpoint_root = Path(args.checkpoint_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict[str, Any]] = {}
    for mode in modes:
        for seed in seeds:
            run_key = f"{mode}/{seed}"
            run_dir = output_dir / mode / f"seed{seed}"
            print(
                f"aux_run mode={mode} seed={seed} pair_weight={args.pair_weight} "
                f"device={args.device}",
                flush=True,
            )
            runs[run_key] = train_run(
                raw,
                checkpoint_root / mode / f"seed{seed}" / "additive.pt",
                run_dir,
                mode,
                seed,
                args.device,
                epochs,
                args.pair_weight,
            )
            (output_dir / "runs.json").write_text(
                json.dumps(runs, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    print(f"saved_suite={output_dir}", flush=True)


if __name__ == "__main__":
    main()
