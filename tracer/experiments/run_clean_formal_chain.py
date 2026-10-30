"""Run the Formal-v2 H1/H2/H3 chain from one clean Git commit.

The chain intentionally writes to a fresh ignored experiment directory. H1
uses the frozen paired-response checkpoints required by its formal protocol;
H2 trains fresh response/belief checkpoints; H3 receives those exact H2
checkpoints through a generated frozen config. The manifest records the
distinction so a clean evaluation-only rerun is never confused with a clean
train-and-evaluate rerun.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _h3_config(source: Path, output_dir: Path, h2_dir: Path) -> Path:
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    raw.setdefault("suite", {})["checkpoint_dir"] = str(h2_dir)
    raw["suite"]["output_dir"] = str(output_dir / "h3")
    path = output_dir / "persistent_closed_loop_formal_clean.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _h1_config(source: Path, output_dir: Path) -> Path:
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    raw.setdefault("suite", {})["output_dir"] = str(output_dir / "h1" / "evaluation")
    path = output_dir / "h1_paired_formal_clean.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _h2_config(source: Path, output_dir: Path, batch_size: int) -> Path:
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    raw.setdefault("suite", {})["output_dir"] = str(output_dir / "h2")
    raw.setdefault("training", {})["batch_size"] = int(batch_size)
    path = output_dir / "persistent_trajectory_formal_clean.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started_unix = time.time()
    start_git = _git_state()
    if start_git["dirty"]:
        raise RuntimeError("clean formal chain requires an empty Git worktree at startup")
    python = str(Path(sys.executable).resolve())
    h1 = output / "h1"
    h2 = output / "h2"
    h3 = output / "h3"
    commands: list[dict[str, Any]] = []
    h1_config = _h1_config(Path(args.h1_config), output)
    h2_config = _h2_config(Path(args.h2_config), output, args.h2_batch_size)

    def execute(stage: str, command: list[str], log: Path) -> None:
        commands.append({"stage": stage, "command": command, "log": str(log.relative_to(root))})
        print(f"clean_chain_start stage={stage}", flush=True)
        _run(command, log)
        print(f"clean_chain_complete stage={stage}", flush=True)

    if args.stage in {"all", "h1"}:
        execute(
            "H1-evaluation-only-frozen-checkpoints",
            [
                python, "-u", "-m", "tracer.experiments.eval_h1_paired_pilot",
                "--config", str(h1_config),
                "--checkpoint-root", "experiments/formal_v2/h1_tracer_rerun/checkpoints",
                "--output-dir", str(h1 / "evaluation"),
                "--device", args.device,
                "--data-seed", "328731",
                "--allow-data-seed-mismatch",
                "--split", "test",
                "--bootstrap-resamples", "10000",
                "--modes", "none", "additive", "nonlinear",
                "--seeds", "2027", "2028", "2029", "2030", "2031",
            ],
            output / "h1.log",
        )
    if args.stage in {"all", "h2"}:
        execute(
            "H2-train-and-evaluate",
            [
                python, "-u", "-m", "tracer.experiments.run_persistent_trajectory_suite",
                "--config", str(h2_config),
                "--device", args.device,
                "--seeds", "2027", "2028", "2029", "2030", "2031",
                "--output-dir", str(h2),
            ],
            output / "h2.log",
        )
    if args.stage in {"all", "h3"}:
        if not h2.exists():
            raise RuntimeError("H3 requires the clean-chain H2 output directory")
        h3_config = _h3_config(Path(args.h3_config), output, h2)
        execute(
            "H3-train-checkpoint-closed-loop",
            [
                python, "-u", "-m", "tracer.experiments.run_persistent_closed_loop_suite",
                "--config", str(h3_config),
                "--device", args.device,
                "--episodes", str(args.h3_episodes),
                "--output-dir", str(h3),
            ],
            output / "h3.log",
        )
    end_git = _git_state()
    payload = {
        "status": "COMPLETED",
        "protocol": "formal-v2-clean-commit-chain-v1",
        "started_unix": started_unix,
        "git_at_start": start_git,
        "git_at_end": end_git,
        "all_stages_started_clean": not start_git["dirty"],
        "stage": args.stage,
        "h1_checkpoint_scope": "frozen_existing_formal_checkpoints; evaluation-only",
        "h2_checkpoint_scope": "freshly trained by this chain",
        "h2_batch_size": args.h2_batch_size,
        "h3_checkpoint_scope": "exactly the H2 output generated by this chain",
        "commands": commands,
    }
    (output / "chain_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/formal_v2/clean_chain_20260902")
    parser.add_argument("--stage", choices=["all", "h1", "h2", "h3"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--h1-config", default="configs/h1_paired_formal_tracer.yaml")
    parser.add_argument("--h2-config", default="configs/persistent_trajectory_formal_tracer_effect.yaml")
    parser.add_argument("--h3-config", default="configs/persistent_closed_loop_formal_tracer_effect.yaml")
    parser.add_argument("--h3-episodes", type=int, default=256)
    parser.add_argument("--h2-batch-size", type=int, default=64)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
