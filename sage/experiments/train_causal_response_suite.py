from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

from sage.experiments.formal_manifest import mark_formal_run, prepare_formal_run


def _aggregate(run_results: dict[str, dict[str, dict[str, float]]]) -> dict:
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
    for run_key, metrics in run_results.items():
        mode, _seed = run_key.split("/", maxsplit=1)
        for variant, values in metrics.items():
            for metric, value in values.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    grouped.setdefault(mode, {}).setdefault(variant, {}).setdefault(metric, []).append(
                        float(value)
                    )

    summary: dict[str, dict] = {}
    for mode, variants in grouped.items():
        summary[mode] = {}
        for variant, metrics in variants.items():
            summary[mode][variant] = {
                metric: {
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "count": len(values),
                }
                for metric, values in metrics.items()
            }
        if "joint" in variants and "additive" in variants:
            summary[mode]["contrasts"] = {
                "additive_minus_joint_nll": summary[mode]["additive"]["nll"]["mean"]
                - summary[mode]["joint"]["nll"]["mean"],
                "shuffled_minus_joint_nll": summary[mode].get("shuffled-joint", {})
                .get("nll", {})
                .get("mean", float("nan"))
                - summary[mode]["joint"]["nll"]["mean"],
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paired causal-response mechanism suite.")
    parser.add_argument("--config", default="configs/response_causal_suite.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--modes", nargs="+", default=None, choices=["none", "additive", "nonlinear"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    suite_cfg = raw.get("suite", {})
    training_cfg = raw.get("training", {})
    modes = args.modes or list(suite_cfg.get("modes", ["none", "additive", "nonlinear"]))
    seeds = args.seeds or [int(seed) for seed in suite_cfg.get("seeds", [2027, 2028, 2029])]
    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 15))
    dataset_cfg = raw.get("dataset", {}).get("causal_response", {})
    data_seed = int(
        args.data_seed if args.data_seed is not None else dataset_cfg.get("seed", 2027)
    )
    output_dir = Path(args.output_dir or suite_cfg.get("output_dir", "experiments/response/causal_suite"))
    formal = "formal_protocol" in raw
    if formal:
        execution = raw["formal_protocol"].get("execution", "train_and_evaluate")
        if execution != "train_and_evaluate":
            raise RuntimeError(
                f"this formal config is {execution}; use the frozen-checkpoint evaluator"
            )
        prepare_formal_run(
            config_path,
            output_dir,
            Path.cwd(),
            command=list(sys.argv),
            overrides={
                "device": args.device,
                "modes": modes,
                "seeds": seeds,
                "data_seed": data_seed,
                "epochs": epochs,
                "output_dir": str(output_dir),
            },
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_results: dict[str, dict[str, dict[str, float]]] = {}

    for mode in modes:
        for seed in seeds:
            run_dir = output_dir / mode / f"seed{seed}"
            command = [
                sys.executable,
                "-u",
                "-m",
                "sage.experiments.train_response_baselines",
                "--config",
                str(config_path),
                "--device",
                args.device,
                "--effect-mode",
                mode,
                "--seed",
                str(seed),
                "--data-seed",
                str(data_seed),
                "--epochs",
                str(epochs),
                "--output-dir",
                str(run_dir),
            ]
            if args.limit_train_batches > 0:
                command.extend(["--limit-train-batches", str(args.limit_train_batches)])
            if args.limit_val_batches > 0:
                command.extend(["--limit-val-batches", str(args.limit_val_batches)])
            if args.resume_existing:
                command.append("--resume-existing")
            print(f"suite_run mode={mode} seed={seed} output={run_dir}", flush=True)
            subprocess.run(command, check=True)
            metrics_path = run_dir / "metrics.json"
            run_results[f"{mode}/{seed}"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            (output_dir / "runs.json").write_text(
                json.dumps(run_results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    summary = _aggregate(run_results)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for mode in modes:
        contrasts = summary[mode].get("contrasts", {})
        print(
            f"suite_summary mode={mode} "
            f"additive_minus_joint_nll={contrasts.get('additive_minus_joint_nll', float('nan')):.6g} "
            f"shuffled_minus_joint_nll={contrasts.get('shuffled_minus_joint_nll', float('nan')):.6g}",
            flush=True,
        )
    if formal:
        mark_formal_run(output_dir, "COMPLETED", Path.cwd())
    print(f"saved_summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
