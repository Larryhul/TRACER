from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import yaml


def _aggregate(runs: dict[str, dict[str, dict]], split: str) -> dict[str, dict]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for run in runs.values():
        results = run[split]
        for baseline, metrics in results.items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    grouped.setdefault(baseline, {}).setdefault(metric, []).append(float(value))
    summary: dict[str, dict] = {}
    for baseline, metrics in grouped.items():
        summary[baseline] = {
            metric: {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "count": len(values),
            }
            for metric, values in metrics.items()
        }
    persistent = summary["persistent"]
    reset = summary["reset"]
    recurrent = summary["recurrent"]
    summary["contrasts"] = {
        "reset_minus_persistent_predictive_nll": (
            reset["predictive_response_nll"]["mean"]
            - persistent["predictive_response_nll"]["mean"]
        ),
        "recurrent_minus_persistent_predictive_nll": (
            recurrent["predictive_response_nll"]["mean"]
            - persistent["predictive_response_nll"]["mean"]
        ),
        "recurrent_minus_persistent_windows_to_confidence": (
            recurrent["windows_to_confidence"]["mean"]
            - persistent["windows_to_confidence"]["mean"]
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the persistent-belief H2 seed suite.")
    parser.add_argument("--config", default="configs/belief_h2.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    suite_cfg = raw.get("suite", {})
    training_cfg = raw.get("training", {})
    seeds = args.seeds or [int(seed) for seed in suite_cfg.get("seeds", [2027, 2028, 2029])]
    epochs = int(args.epochs or training_cfg.get("epochs", 20))
    output_dir = Path(
        args.output_dir or suite_cfg.get("output_dir", "experiments/belief/h2_suite")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, dict]] = {}

    for seed in seeds:
        run_dir = output_dir / f"seed{seed}"
        command = [
            sys.executable,
            "-u",
            "-m",
            "tracer.experiments.eval_synthetic_belief",
            "--config",
            str(config_path),
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--output-dir",
            str(run_dir),
        ]
        print(f"belief_suite_run seed={seed} output={run_dir}", flush=True)
        subprocess.run(command, check=True)
        runs[str(seed)] = {
            "in_distribution": json.loads(
                (run_dir / "metrics.json").read_text(encoding="utf-8")
            ),
            "ood": json.loads((run_dir / "ood_metrics.json").read_text(encoding="utf-8")),
        }
        (output_dir / "runs.json").write_text(
            json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    summary = {
        "in_distribution": _aggregate(runs, "in_distribution"),
        "ood": _aggregate(runs, "ood"),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "belief_suite_id_contrasts="
        f"{json.dumps(summary['in_distribution']['contrasts'], sort_keys=True)}"
    )
    print(
        "belief_suite_ood_contrasts="
        f"{json.dumps(summary['ood']['contrasts'], sort_keys=True)}"
    )
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
