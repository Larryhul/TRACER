"""Re-evaluate frozen H2 checkpoints with calibration-ready raw outputs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from sage.experiments.train_persistent_trajectory_response import (
    _evaluate,
    _make_dataset,
    _set_seed,
)
from sage.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    TrajectoryRecurrentBelief,
    TrajectoryRecurrentConfig,
)


def _write_raw_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("calibration evaluation returned no rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_checkpoint_models(seed_dir: Path, device: str) -> tuple[Any, Any, dict[str, Any]]:
    response_payload = torch.load(seed_dir / "response.pt", map_location=device)
    recurrent_payload = torch.load(seed_dir / "recurrent.pt", map_location=device)
    response_config = PersistentTrajectoryModelConfig(**response_payload["model_config"])
    recurrent_config = TrajectoryRecurrentConfig(**recurrent_payload["model_config"])
    response_model = BeliefConditionedTeamResponseModel(response_config).to(device)
    recurrent_model = TrajectoryRecurrentBelief(recurrent_config).to(device)
    response_model.load_state_dict(response_payload["model"])
    recurrent_model.load_state_dict(recurrent_payload["model"])
    response_model.eval()
    recurrent_model.eval()
    metadata = {
        "model_seed": int(response_payload["seed"]),
        "data_seed": int(response_payload["data_seed"]),
        "id_split": str(response_payload["id_split"]),
        "response_checkpoint": str(seed_dir / "response.pt"),
        "recurrent_checkpoint": str(seed_dir / "recurrent.pt"),
    }
    return response_model, recurrent_model, metadata


def evaluate_seed(
    config: dict[str, Any],
    seed_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    limit_episodes: int = 0,
) -> None:
    config = copy.deepcopy(config)
    response_model, recurrent_model, metadata = _load_checkpoint_models(seed_dir, device)
    if limit_episodes > 0:
        config["dataset"][metadata["id_split"] + "_episodes"] = limit_episodes
        config["dataset"]["ood_episodes"] = limit_episodes
    raw_dataset = config["dataset"]
    evaluation_cfg = config.get("evaluation", {})
    data_seed = metadata["data_seed"]
    id_split = metadata["id_split"]
    id_dataset = _make_dataset(config, id_split, data_seed)
    ood_dataset = _make_dataset(
        config,
        "ood",
        data_seed,
        action_policy=str(raw_dataset.get("ood_action_policy", "correlated")),
    )
    loader_kwargs = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}
    id_results, id_rows = _evaluate(
        response_model,
        recurrent_model,
        DataLoader(id_dataset, **loader_kwargs),
        device=device,
        dt=float(raw_dataset.get("dt", 0.4)),
        likelihood_scale=float(evaluation_cfg.get("likelihood_scale", 0.05)),
        confidence_threshold=float(evaluation_cfg.get("confidence_threshold", 0.8)),
        execute_steps=int(raw_dataset.get("execute_steps", 3)),
        split_name="id",
        bayesian_hazard_rate=(
            None
            if evaluation_cfg.get("bayesian_hazard_rate") is None
            else float(evaluation_cfg["bayesian_hazard_rate"])
        ),
    )
    ood_results, ood_rows = _evaluate(
        response_model,
        recurrent_model,
        DataLoader(ood_dataset, **loader_kwargs),
        device=device,
        dt=float(raw_dataset.get("dt", 0.4)),
        likelihood_scale=float(evaluation_cfg.get("likelihood_scale", 0.05)),
        confidence_threshold=float(evaluation_cfg.get("confidence_threshold", 0.8)),
        execute_steps=int(raw_dataset.get("execute_steps", 3)),
        split_name="ood",
        bayesian_hazard_rate=(
            None
            if evaluation_cfg.get("bayesian_hazard_rate") is None
            else float(evaluation_cfg["bayesian_hazard_rate"])
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(id_results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "ood_metrics.json").write_text(
        json.dumps(ood_results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_raw_rows(output_dir / "raw_metrics.csv", id_rows + ood_rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                **metadata,
                "source_config": str(config.get("output_dir", "")),
                "device": device,
                "batch_size": batch_size,
                "calibration_fields": [
                    "decision_brier",
                    "posterior_brier",
                    "decision_confidence",
                    "posterior_confidence",
                    "decision_predicted_mode",
                    "posterior_predicted_mode",
                    "predictive_1sigma_coordinate_coverage",
                    "predictive_2sigma_coordinate_coverage",
                ],
                "identity_control_protocol": "rotating_identity_keys_v2",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate frozen H2 checkpoints for calibration diagnostics.")
    parser.add_argument("--config", default="configs/persistent_trajectory_formal.yaml")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-episodes", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    checkpoint_root = Path(args.checkpoint_root)
    output_root = Path(args.output_root)
    seeds = args.seeds or sorted(
        int(path.name.removeprefix("seed"))
        for path in checkpoint_root.glob("seed*")
        if path.is_dir()
    )
    if not seeds:
        raise ValueError("no checkpoint seeds found")
    for seed in seeds:
        _set_seed(seed)
        seed_dir = checkpoint_root / f"seed{seed}"
        if not (seed_dir / "response.pt").exists() or not (seed_dir / "recurrent.pt").exists():
            raise FileNotFoundError(f"missing checkpoint pair under {seed_dir}")
        output_dir = output_root / f"seed{seed}"
        print(f"calibration_eval_start seed={seed} output={output_dir}", flush=True)
        evaluate_seed(
            config,
            seed_dir,
            output_dir,
            args.device,
            args.batch_size,
            args.limit_episodes,
        )
        print(f"calibration_eval_done seed={seed}", flush=True)


if __name__ == "__main__":
    main()
