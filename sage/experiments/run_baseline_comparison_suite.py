"""Run deterministic baseline comparisons over multiple synthetic dataset seeds."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


BASELINES = (
    "constant_velocity",
    "constant_acceleration",
    "direct_goal",
    "social_force",
)


def _merge_csv(paths: list[Path], output: Path) -> None:
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in fields:
                    fields.append(field)
            rows.extend(reader)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic synthetic baseline seed sweep.")
    parser.add_argument("--config", default="configs/sage_synthetic.yaml")
    parser.add_argument("--output-dir", default="experiments/baselines/synthetic_multiseed_20260820")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--limit-samples", type=int, default=1024)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    csv_paths: list[Path] = []
    for seed in args.seeds:
        for baseline in args.baselines:
            output = output_dir / f"seed{seed}_{baseline}.csv"
            csv_paths.append(output)
            if args.resume_existing and output.exists():
                print(f"baseline_suite_skip seed={seed} baseline={baseline}", flush=True)
                continue
            if baseline == "social_force":
                module = "sage.experiments.eval_social_force_baseline"
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    module,
                    "--config",
                    args.config,
                    "--benchmark",
                    "synthetic",
                    "--split",
                    "test",
                    "--seed",
                    str(seed),
                    "--limit-samples",
                    str(args.limit_samples),
                    "--role-metrics",
                    "--device",
                    args.device,
                    "--save-csv",
                    str(output),
                ]
            else:
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "sage.experiments.eval_baseline_comparison",
                    "--config",
                    args.config,
                    "--dataset",
                    "synthetic",
                    "--split",
                    "test",
                    "--baseline",
                    baseline,
                    "--seed",
                    str(seed),
                    "--limit-samples",
                    str(args.limit_samples),
                    "--role-metrics",
                    "--device",
                    args.device,
                    "--save-csv",
                    str(output),
                ]
            print(f"baseline_suite_run seed={seed} baseline={baseline}", flush=True)
            subprocess.run(command, check=True)
            records.append({"seed": seed, "baseline": baseline, "csv": str(output)})

    existing = [path for path in csv_paths if path.exists()]
    _merge_csv(existing, output_dir / "results.csv")
    manifest = {
        "config": args.config,
        "device": args.device,
        "seeds": args.seeds,
        "baselines": args.baselines,
        "limit_samples": args.limit_samples,
        "results": str(output_dir / "results.csv"),
        "records_created_this_run": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved_results={output_dir / 'results.csv'}", flush=True)
    print(f"saved_manifest={output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
