from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

from tracer.experiments.formal_manifest import mark_formal_run, prepare_formal_run


def _aggregate(runs: dict[str, dict], split: str) -> dict[str, dict]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for run in runs.values():
        for baseline, metrics in run[split].items():
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
    shuffled = summary.get("shuffled-id")
    summary["contrasts"] = {
        "reset_minus_persistent_nll": (
            reset["predictive_trajectory_nll"]["mean"]
            - persistent["predictive_trajectory_nll"]["mean"]
        ),
        "reset_minus_persistent_ade": (
            reset["trajectory_ade"]["mean"] - persistent["trajectory_ade"]["mean"]
        ),
        "reset_minus_persistent_fde": (
            reset["trajectory_fde"]["mean"] - persistent["trajectory_fde"]["mean"]
        ),
        "persistent_minus_recurrent_final_mode_accuracy": (
            persistent["final_mode_accuracy"]["mean"]
            - recurrent["final_mode_accuracy"]["mean"]
        ),
        "persistent_minus_recurrent_nll": (
            persistent["predictive_trajectory_nll"]["mean"]
            - recurrent["predictive_trajectory_nll"]["mean"]
        ),
        "persistent_minus_recurrent_ade": (
            persistent["trajectory_ade"]["mean"] - recurrent["trajectory_ade"]["mean"]
        ),
    }
    if shuffled is not None:
        summary["contrasts"].update(
            {
                "shuffled_id_minus_persistent_nll": (
                    shuffled["predictive_trajectory_nll"]["mean"]
                    - persistent["predictive_trajectory_nll"]["mean"]
                ),
                "shuffled_id_minus_persistent_ade": (
                    shuffled["trajectory_ade"]["mean"]
                    - persistent["trajectory_ade"]["mean"]
                ),
            }
        )
    return summary


def _write_flat_csv(runs: dict[str, dict], output_path: Path) -> None:
    fields = [
        "seed",
        "split",
        "baseline",
        "predictive_trajectory_nll",
        "trajectory_ade",
        "trajectory_fde",
        "final_mode_accuracy",
        "windows_to_confidence",
        "likelihood_mode_accuracy",
        "likelihood_margin_per_coordinate",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed, run in sorted(runs.items()):
            for split in ("id", "ood"):
                for baseline, metrics in sorted(run[split].items()):
                    writer.writerow(
                        {
                            "seed": seed,
                            "split": split,
                            "baseline": baseline,
                            **{field: metrics[field] for field in fields[3:]},
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a persistent trajectory-response seed suite."
    )
    parser.add_argument("--config", default="configs/persistent_trajectory_formal.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--response-epochs", type=int, default=None)
    parser.add_argument("--recurrent-epochs", type=int, default=None)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    suite_cfg = raw.get("suite", {})
    training_cfg = raw.get("training", {})
    seeds = args.seeds or [int(seed) for seed in suite_cfg.get("seeds", [2027, 2028, 2029])]
    epochs = int(args.epochs or training_cfg.get("epochs", 25))
    response_epochs = int(
        args.response_epochs
        or args.epochs
        or training_cfg.get("response_epochs", epochs)
    )
    recurrent_epochs = int(
        args.recurrent_epochs
        or args.epochs
        or training_cfg.get("recurrent_epochs", epochs)
    )
    output_dir = Path(
        args.output_dir
        or suite_cfg.get("output_dir", "experiments/response/persistent_trajectory_formal_suite")
    )
    formal = "formal_protocol" in raw
    if formal:
        prepare_formal_run(
            config_path,
            output_dir,
            Path.cwd(),
            command=list(sys.argv),
            overrides={
                "device": args.device,
                "seeds": seeds,
                "response_epochs": response_epochs,
                "recurrent_epochs": recurrent_epochs,
                "output_dir": str(output_dir),
            },
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "runs.json"
    runs: dict[str, dict] = {}
    if args.resume_existing and runs_path.exists():
        runs = json.loads(runs_path.read_text(encoding="utf-8"))

    for seed in seeds:
        run_dir = output_dir / f"seed{seed}"
        if (
            args.resume_existing
            and str(seed) in runs
            and (run_dir / "metrics.json").exists()
            and (run_dir / "ood_metrics.json").exists()
        ):
            print(f"trajectory_suite_skip_complete seed={seed}", flush=True)
            continue
        command = [
            sys.executable,
            "-u",
            "-m",
            "tracer.experiments.train_persistent_trajectory_response",
            "--config",
            str(config_path),
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--response-epochs",
            str(response_epochs),
            "--recurrent-epochs",
            str(recurrent_epochs),
            "--output-dir",
            str(run_dir),
        ]
        print(f"trajectory_suite_run seed={seed} output={run_dir}", flush=True)
        with (run_dir.parent / f"seed{seed}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        runs[str(seed)] = {
            "id": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
            "ood": json.loads((run_dir / "ood_metrics.json").read_text(encoding="utf-8")),
        }
        runs_path.write_text(
            json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"trajectory_suite_completed seed={seed}", flush=True)

    summary = {split: _aggregate(runs, split) for split in ("id", "ood")}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_flat_csv(runs, output_dir / "runs.csv")
    for split in ("id", "ood"):
        print(
            f"trajectory_suite_{split}_contrasts="
            f"{json.dumps(summary[split]['contrasts'], sort_keys=True)}",
            flush=True,
        )
    if formal:
        analysis_cfg = raw.get("analysis", {})
        raw_paths = [output_dir / f"seed{seed}" / "raw_metrics.csv" for seed in seeds]
        analysis_command = [
            sys.executable,
            "-u",
            "-m",
            "tracer.experiments.analyze_h2_p0",
            "--raw",
            *[str(path) for path in raw_paths],
            "--output-dir",
            str(output_dir / "episode_analysis"),
            "--bootstrap-samples",
            str(int(analysis_cfg["bootstrap_samples"])),
            "--seed",
            str(int(analysis_cfg["seed"])),
            "--analysis-status",
            "formal-confirmatory",
            "--formal-manifest",
            str(output_dir / "manifest.json"),
        ]
        subprocess.run(analysis_command, check=True)
        mark_formal_run(output_dir, "COMPLETED", Path.cwd())
    print(f"saved_summary={output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
