from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the corrected H1 paired pilot.")
    parser.add_argument("--config", default="configs/h1_paired_pilot.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="experiments/pilot/h1_paired_v1")
    parser.add_argument("--modes", nargs="+", choices=["none", "additive", "nonlinear"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=None)
    parser.add_argument("--split", choices=["val", "test"], default=None)
    parser.add_argument("--include-context-baselines", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    suite_cfg = raw.get("suite", {})
    training_cfg = raw.get("training", {})
    evaluation_cfg = raw.get("evaluation", {})
    causal_cfg = raw.get("dataset", {}).get("causal_response", {})
    modes = args.modes or list(suite_cfg.get("modes", ["none", "additive", "nonlinear"]))
    seeds = args.seeds or [int(value) for value in suite_cfg.get("seeds", [2027, 2028, 2029])]
    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 30))
    bootstrap_resamples = int(
        args.bootstrap_resamples
        if args.bootstrap_resamples is not None
        else evaluation_cfg.get("bootstrap_resamples", 10000)
    )
    data_seed = int(causal_cfg.get("seed", 81173))
    evaluation_split = str(args.split or suite_cfg.get("evaluation_split", "val"))
    output_dir = Path(args.output_dir)
    checkpoint_root = output_dir / "checkpoints"
    evaluation_dir = output_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_command = [
        sys.executable,
        "-u",
        "-m",
        "sage.experiments.train_causal_response_suite",
        "--config",
        str(config_path),
        "--device",
        args.device,
        "--output-dir",
        str(checkpoint_root),
        "--epochs",
        str(epochs),
        "--data-seed",
        str(data_seed),
        "--modes",
        *modes,
        "--seeds",
        *(str(seed) for seed in seeds),
    ]
    if args.limit_train_batches > 0:
        train_command.extend(["--limit-train-batches", str(args.limit_train_batches)])
    if args.limit_val_batches > 0:
        train_command.extend(["--limit-val-batches", str(args.limit_val_batches)])

    eval_command = [
        sys.executable,
        "-u",
        "-m",
        "sage.experiments.eval_h1_paired_pilot",
        "--config",
        str(config_path),
        "--checkpoint-root",
        str(checkpoint_root),
        "--output-dir",
        str(evaluation_dir),
        "--device",
        args.device,
        "--data-seed",
        str(data_seed),
        "--split",
        evaluation_split,
        "--bootstrap-resamples",
        str(bootstrap_resamples),
        "--modes",
        *modes,
        "--seeds",
        *(str(seed) for seed in seeds),
    ]
    if args.include_context_baselines:
        eval_command.append("--include-context-baselines")
    manifest = {
        "status": "RUNNING",
        "purpose": "H1_PARAMETER_MATCHED_PAIRED_PILOT",
        "confirmatory": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "device": args.device,
        "modes": modes,
        "model_seeds": seeds,
        "data_seed": data_seed,
        "epochs": epochs,
        "independent_unit": "base_scene",
        "checkpoint_rule": str(training_cfg.get("checkpoint_rule", "final_epoch")),
        "evaluation_split": evaluation_split,
        "include_context_baselines": args.include_context_baselines,
        "train_command": train_command,
        "evaluation_command": eval_command,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("h1_stage=train", flush=True)
    subprocess.run(train_command, check=True)
    print("h1_stage=evaluate", flush=True)
    subprocess.run(eval_command, check=True)
    manifest["status"] = "COMPLETED"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"h1_stage=complete summary={evaluation_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
