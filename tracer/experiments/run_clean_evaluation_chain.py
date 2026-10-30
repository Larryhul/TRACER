"""Re-evaluate frozen H2 checkpoints and H3 from one clean Git commit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from tracer.experiments.run_persistent_trajectory_suite import _aggregate
from tracer.experiments.train_persistent_trajectory_response import (
    _evaluate,
    _make_dataset,
)
from tracer.experiments.eval_persistent_closed_loop import _load_model, _load_recurrent_model


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _h3_config(source: Path, output_dir: Path, checkpoint_dir: Path) -> Path:
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    raw.setdefault("suite", {})["checkpoint_dir"] = str(checkpoint_dir)
    raw["suite"]["output_dir"] = str(output_dir / "h3")
    path = output_dir / "h3_clean_evaluation.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _evaluate_h2(
    config_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    seeds: list[int],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset_cfg = raw["dataset"]
    data_seed = int(dataset_cfg["data_seed"])
    id_split = str(raw.get("evaluation", {}).get("id_split", "test"))
    id_dataset = _make_dataset(raw, id_split, data_seed)
    ood_dataset = _make_dataset(
        raw,
        "ood",
        data_seed,
        action_policy=str(dataset_cfg.get("ood_action_policy", "correlated")),
    )
    loaders = {
        "test": DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        "ood": DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    }
    runs: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        response = _load_model(checkpoint_dir / f"seed{seed}" / "response.pt", device)
        recurrent = _load_recurrent_model(checkpoint_dir / f"seed{seed}" / "recurrent.pt", device)
        runs[str(seed)] = {}
        for split, loader in loaders.items():
            runs[str(seed)][split], rows = _evaluate(
                response,
                recurrent,
                loader,
                device=device,
                dt=float(dataset_cfg.get("dt", 0.4)),
                likelihood_scale=float(raw["evaluation"].get("likelihood_scale", 0.05)),
                confidence_threshold=float(raw["evaluation"].get("confidence_threshold", 0.8)),
                execute_steps=int(dataset_cfg.get("execute_steps", 3)),
                split_name=split,
            )
            (output_dir / f"seed{seed}_{split}_rows.json").write_text(
                json.dumps(rows, indent=2) + "\n", encoding="utf-8"
            )
        print(f"clean_eval_h2_complete seed={seed}", flush=True)
    summary = {split: _aggregate(runs, split) for split in ("test", "ood")}
    return {"runs": runs, "summary": summary, "data_seed": data_seed}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    start_unix = time.time()
    git = _git_state()
    if git["dirty"]:
        raise RuntimeError("clean evaluation chain requires a clean Git worktree")
    h2_output = output / "h2_evaluation"
    h2_output.mkdir(parents=True, exist_ok=True)
    h2 = _evaluate_h2(
        Path(args.h2_config),
        Path(args.h2_checkpoint_dir),
        h2_output,
        args.seeds,
        args.device,
        args.batch_size,
    )
    (h2_output / "summary.json").write_text(json.dumps(h2, indent=2) + "\n", encoding="utf-8")
    h3_config = _h3_config(Path(args.h3_config), output, Path(args.h3_checkpoint_dir))
    h3_command = [
        sys.executable, "-u", "-m", "tracer.experiments.run_persistent_closed_loop_suite",
        "--config", str(h3_config), "--device", args.device,
        "--episodes", str(args.h3_episodes), "--output-dir", str(output / "h3"),
    ]
    with (output / "h3.log").open("w", encoding="utf-8") as handle:
        subprocess.run(h3_command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    payload = {
        "status": "COMPLETED",
        "protocol": "formal-v2-clean-code-evaluation-chain-v1",
        "started_unix": start_unix,
        "finished_unix": time.time(),
        "git": git,
        "h2_scope": "evaluation-only frozen checkpoints; no retraining",
        "h3_scope": "closed-loop evaluation using the same frozen H2 checkpoints",
        "h2": h2,
        "h3_command": h3_command,
        "h2_checkpoint_dir": str(Path(args.h2_checkpoint_dir)),
        "h3_checkpoint_dir": str(Path(args.h3_checkpoint_dir)),
    }
    (output / "chain_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/formal_v2/clean_evaluation_chain_20260902")
    parser.add_argument("--h2-config", default="configs/persistent_trajectory_formal_tracer_effect.yaml")
    parser.add_argument("--h2-checkpoint-dir", default="experiments/formal_v2/h2/persistent_trajectory_suite_tracer_effect")
    parser.add_argument("--h3-config", default="configs/persistent_closed_loop_formal_tracer_effect.yaml")
    parser.add_argument("--h3-checkpoint-dir", default="experiments/formal_v2/h2/persistent_trajectory_suite_tracer_effect")
    parser.add_argument("--h3-episodes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
