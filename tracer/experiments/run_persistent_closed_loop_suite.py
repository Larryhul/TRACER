from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from tracer.experiments.eval_persistent_closed_loop import (
    _aggregate_runs,
    _configured_policies,
    _write_csv,
    _write_episode_csv,
)
from tracer.experiments.formal_manifest import mark_formal_run, prepare_formal_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run closed-loop seeds in parallel.")
    parser.add_argument("--config", default="configs/persistent_closed_loop_formal.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    suite_cfg = raw["suite"]
    seeds = [int(seed) for seed in suite_cfg["seeds"]]
    episodes = int(args.episodes or raw["simulation"]["episodes"])
    output_dir = Path(args.output_dir or suite_cfg["output_dir"])
    policies = _configured_policies(raw)
    formal = "formal_protocol" in raw
    if formal:
        artifacts = [Path(raw["candidate_source"]["checkpoint"])]
        artifacts.extend(
            Path(suite_cfg["checkpoint_dir"]) / f"seed{seed}" / "response.pt"
            for seed in seeds
        )
        if "recurrent-joint" in raw.get("additional_policies", []):
            artifacts.extend(
                Path(suite_cfg["checkpoint_dir"]) / f"seed{seed}" / "recurrent.pt"
                for seed in seeds
            )
        rescue_selection = raw.get("formal_protocol", {}).get("rescue_selection")
        if rescue_selection:
            artifacts.append(Path(str(rescue_selection)))
        candidate_selection = raw.get("formal_protocol", {}).get("candidate_selection")
        if candidate_selection:
            artifacts.append(Path(str(candidate_selection)))
        candidate_gate = raw.get("formal_protocol", {}).get("candidate_gate")
        if candidate_gate:
            artifacts.append(Path(str(candidate_gate)))
        prepare_formal_run(
            config_path,
            output_dir,
            Path.cwd(),
            command=list(sys.argv),
            overrides={
                "device": args.device,
                "episodes": episodes,
                "seeds": seeds,
                "output_dir": str(output_dir),
            },
            input_artifacts=artifacts,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    def run_seed(seed: int) -> int:
        seed_dir = output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        if args.resume_existing and (seed_dir / "runs.json").exists():
            print(f"closed_loop_suite_skip_complete seed={seed}", flush=True)
            return seed
        command = [
            sys.executable,
            "-u",
            "-m",
            "tracer.experiments.eval_persistent_closed_loop",
            "--config",
            str(config_path),
            "--device",
            args.device,
            "--seeds",
            str(seed),
            "--episodes",
            str(episodes),
            "--output-dir",
            str(seed_dir),
        ]
        with (output_dir / f"seed{seed}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return seed

    workers = min(int(suite_cfg.get("parallel_seeds", 1)), len(seeds))
    print(f"closed_loop_suite seeds={seeds} episodes={episodes} workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in seeds}
        for future in as_completed(futures):
            seed = future.result()
            print(f"closed_loop_suite_completed seed={seed}", flush=True)

    runs: dict[str, dict] = {}
    for seed in seeds:
        seed_runs = json.loads(
            (output_dir / f"seed{seed}" / "runs.json").read_text(encoding="utf-8")
        )
        runs[str(seed)] = seed_runs[str(seed)]
    (output_dir / "runs.json").write_text(
        json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = _aggregate_runs(runs, policies)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(runs, output_dir / "runs.csv")
    _write_episode_csv(runs, output_dir / "episode_metrics.csv")
    for split in raw["simulation"]["splits"]:
        print(
            f"closed_loop_suite_{split}_contrasts="
            f"{json.dumps(summary[str(split)]['contrasts'], sort_keys=True)}",
            flush=True,
        )
    if formal:
        analysis_cfg = raw.get("analysis", {})
        replication_role = raw.get("formal_protocol", {}).get("replication_role")
        stage = str(raw.get("formal_protocol", {}).get("stage", ""))
        analysis_status = (
            "secondary-robustness"
            if (
                str(replication_role).startswith("independent_data_seed_secondary_")
                or stage.endswith("-ablation")
            )
            else "formal-confirmatory"
        )
        analysis_command = [
            sys.executable,
            "-u",
            "-m",
            "tracer.experiments.analyze_h3_p0",
            "--episode-metrics",
            str(output_dir / "episode_metrics.csv"),
            "--runs",
            str(output_dir / "runs.json"),
            "--output-dir",
            str(output_dir / "episode_analysis"),
            "--bootstrap-samples",
            str(int(analysis_cfg["bootstrap_samples"])),
            "--seed",
            str(int(analysis_cfg["seed"])),
            "--analysis-status",
            analysis_status,
        ]
        if analysis_status == "formal-confirmatory":
            analysis_command.extend(
                ["--formal-manifest", str(output_dir / "manifest.json")]
            )
        subprocess.run(analysis_command, check=True)
        mark_formal_run(output_dir, "COMPLETED", Path.cwd())
    print(f"saved_summary={output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
