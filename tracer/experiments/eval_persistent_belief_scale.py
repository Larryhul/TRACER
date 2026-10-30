from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from tracer.experiments.train_persistent_trajectory_response import (
    _evaluate,
    _make_dataset,
    _set_seed,
    _write_raw_rows,
)
from tracer.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    TrajectoryRecurrentBelief,
    TrajectoryRecurrentConfig,
)


def _parse_scales(raw: str) -> list[float]:
    scales = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not scales or any(scale < 0.0 for scale in scales):
        raise ValueError("scales must contain one or more non-negative values")
    return scales


def _load_models(
    checkpoint_dir: Path,
    device: str,
) -> tuple[BeliefConditionedTeamResponseModel, TrajectoryRecurrentBelief, int]:
    response_checkpoint = torch.load(
        checkpoint_dir / "response.pt",
        map_location=device,
        weights_only=False,
    )
    recurrent_checkpoint = torch.load(
        checkpoint_dir / "recurrent.pt",
        map_location=device,
        weights_only=False,
    )
    response_model = BeliefConditionedTeamResponseModel(
        PersistentTrajectoryModelConfig(**response_checkpoint["model_config"])
    ).to(device)
    response_model.load_state_dict(response_checkpoint["model"])
    recurrent_model = TrajectoryRecurrentBelief(
        TrajectoryRecurrentConfig(**recurrent_checkpoint["model_config"])
    ).to(device)
    recurrent_model.load_state_dict(recurrent_checkpoint["model"])
    return response_model.eval(), recurrent_model.eval(), int(response_checkpoint["seed"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-pilot calibration sweep for persistent-belief likelihood scale."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--scales",
        default="0,0.025,0.05,0.1,0.15,0.2,0.3,0.5,1.0",
    )
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    training_cfg = raw.get("training", {})
    dataset_cfg = raw["dataset"]
    evaluation_cfg = raw.get("evaluation", {})
    response_model, recurrent_model, seed = _load_models(
        Path(args.checkpoint_dir), args.device
    )
    _set_seed(seed)
    torch.set_num_threads(int(training_cfg.get("num_threads", 4)))
    if args.limit_episodes > 0:
        raw["dataset"]["val_episodes"] = args.limit_episodes
        raw["dataset"]["ood_episodes"] = args.limit_episodes

    loader_kwargs = {
        "batch_size": int(training_cfg.get("batch_size", 8)),
        "num_workers": int(training_cfg.get("num_workers", 0)),
    }
    loaders = {
        "id": DataLoader(
            _make_dataset(raw, "val", seed),
            shuffle=False,
            **loader_kwargs,
        ),
        "ood": DataLoader(
            _make_dataset(
                raw,
                "ood",
                seed,
                action_policy=str(dataset_cfg.get("ood_action_policy", "correlated")),
            ),
            shuffle=False,
            **loader_kwargs,
        ),
    }

    scales = _parse_scales(args.scales)
    rows: list[dict[str, float | int | str]] = []
    single_scale_metrics = {}
    single_scale_raw_rows = []
    for scale in scales:
        print(f"belief_scale={scale:g}", flush=True)
        for split, loader in loaders.items():
            metrics, raw_rows = _evaluate(
                response_model,
                recurrent_model,
                loader,
                args.device,
                dt=float(dataset_cfg.get("dt", 0.4)),
                likelihood_scale=scale,
                confidence_threshold=float(
                    evaluation_cfg.get("confidence_threshold", 0.8)
                ),
                execute_steps=int(dataset_cfg.get("execute_steps", 3)),
                split_name=split,
            )
            if len(scales) == 1:
                single_scale_metrics[split] = metrics
                single_scale_raw_rows.extend(raw_rows)
            for method in ("no-belief", "reset", "persistent", "recurrent", "oracle"):
                values = metrics[method]
                rows.append(
                    {
                        "seed": seed,
                        "scale": scale,
                        "split": split,
                        "method": method,
                        "predictive_trajectory_nll": values[
                            "predictive_trajectory_nll"
                        ],
                        "trajectory_ade": values["trajectory_ade"],
                        "trajectory_fde": values["trajectory_fde"],
                        "decision_mode_nll": values["decision_mode_nll"],
                        "decision_mode_accuracy": values[
                            "decision_mode_accuracy"
                        ],
                    }
                )

    id_persistent = [
        row
        for row in rows
        if row["split"] == "id" and row["method"] == "persistent"
    ]
    selected = min(id_persistent, key=lambda row: row["predictive_trajectory_nll"])
    selected_scale = float(selected["scale"])
    if len(scales) == 1:
        selected_metrics = single_scale_metrics
        selected_raw_rows = single_scale_raw_rows
    else:
        selected_metrics = {}
        selected_raw_rows = []
        for split, loader in loaders.items():
            metrics, raw_rows = _evaluate(
                response_model,
                recurrent_model,
                loader,
                args.device,
                dt=float(dataset_cfg.get("dt", 0.4)),
                likelihood_scale=selected_scale,
                confidence_threshold=float(
                    evaluation_cfg.get("confidence_threshold", 0.8)
                ),
                execute_steps=int(dataset_cfg.get("execute_steps", 3)),
                split_name=split,
            )
            selected_metrics[split] = metrics
            selected_raw_rows.extend(raw_rows)
    payload = {
        "analysis_status": "post-pilot calibration diagnostic; not preregistered",
        "selection_rule": "minimum ID validation Persistent predictive trajectory NLL",
        "selected_scale": selected_scale,
        "rows": rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "selected_metrics.json").write_text(
        json.dumps(selected_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_raw_rows(output_dir / "selected_raw_metrics.csv", selected_raw_rows)
    print(f"selected_scale={selected_scale:g} output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
