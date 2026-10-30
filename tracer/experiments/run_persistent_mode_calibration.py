from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def _parse_weights(raw: str) -> list[float]:
    weights = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not weights or any(weight < 0.0 for weight in weights):
        raise ValueError("weights must contain one or more non-negative values")
    return weights


def _run_name(weight: float) -> str:
    return f"beta{weight:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate mode-discrimination weight.")
    parser.add_argument("--config", default="configs/persistent_trajectory_response.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--weights", default="0,0.02,0.05,0.1,0.2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--output-dir",
        default="experiments/response/persistent_mode_calibration_seed2027",
    )
    args = parser.parse_args()

    weights = _parse_weights(args.weights)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []

    for weight in weights:
        run_dir = output_root / _run_name(weight)
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "tracer.experiments.train_persistent_trajectory_response",
            "--config",
            args.config,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--mode-discrimination-weight",
            str(weight),
            "--mode-temperature",
            str(args.temperature),
            "--output-dir",
            str(run_dir),
        ]
        print(f"starting weight={weight:g} output={run_dir}", flush=True)
        with (run_dir / "train.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

        for split, filename in (("id", "metrics.json"), ("ood", "ood_metrics.json")):
            metrics = json.loads((run_dir / filename).read_text(encoding="utf-8"))
            for baseline, values in metrics.items():
                rows.append(
                    {
                        "weight": weight,
                        "temperature": args.temperature,
                        "split": split,
                        "baseline": baseline,
                        "trajectory_nll": values["predictive_trajectory_nll"],
                        "ade": values["trajectory_ade"],
                        "fde": values["trajectory_fde"],
                        "final_mode_accuracy": values["final_mode_accuracy"],
                        "likelihood_mode_accuracy": values["likelihood_mode_accuracy"],
                        "likelihood_margin_per_coordinate": values[
                            "likelihood_margin_per_coordinate"
                        ],
                    }
                )
        print(f"completed weight={weight:g}", flush=True)

    summary = {
        "seed": args.seed,
        "epochs": args.epochs,
        "temperature": args.temperature,
        "weights": weights,
        "rows": rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_summary={output_root}", flush=True)


if __name__ == "__main__":
    main()
