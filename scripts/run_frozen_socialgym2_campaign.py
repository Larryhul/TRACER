"""Run the complete paired campaign after independently verified prerequisites."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Event
import time


SEEDS = (2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377)
SCENARIOS = ("envs/scenario/intersection", "envs/scenario/open")
POLICIES = ("tracer-persistent", "tracer-reset", "tracer-uniform", "tracer-no-pair", "official-goalone", "official-halt")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args) -> None:
    root = args.root.resolve()
    frozen = args.frozen_code.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    from sage.experiments.socialgym2_physics_contract import verify_physics_audits

    analysis_paths = [root / "sage/experiments" / name for name in (
        "aggregate_socialgym2_tracer.py", "socialgym2_physics_contract.py",
    )]
    analysis_hashes = {str(path): sha256(path) for path in analysis_paths}
    response = root / "experiments/formal_v2/h2/persistent_trajectory_suite_tracer_effect/seed2027/response.pt"
    sage = root / "experiments/pilot/h3_sage_generator_p0_seed2027/synthetic_sage_epoch20.pt"
    runtime_path = args.runtime_manifest.resolve()
    runtime = json.loads(runtime_path.read_text())
    runtime_contract = {"sha256": sha256(runtime_path), "protocol": runtime["protocol"], "boundary": runtime["boundary"]}
    calibration_path = output / "validation_calibration.json"
    inputs = [args.calibration_input_root.resolve() / Path(scenario).name for scenario in SCENARIOS]
    audits = [args.physics_root.resolve() / f"{Path(scenario).name}.json" for scenario in SCENARIOS]
    protocol = {
        "protocol": "socialgym2-frozen-paired-campaign-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_seeds": SEEDS, "scenarios": SCENARIOS, "policies": POLICIES,
        "episodes_per_seed_scenario": 16, "maximum_planning_windows": 60,
        "expected_independent_scenes": 320, "expected_policy_rows": 1920,
        "controllable_robots": 3, "humans": 2, "partial_observation": True,
        "sim_dt": 0.025, "sim_ticks_per_window": 16,
        "training_checkpoint_seeds": [2027], "noninferiority_claim": False,
        "primary_outcomes": ["collision_free_team_success", "team_success", "collision", "task_progress"],
        "statistics": "paired unique-scene bootstrap; secondary seed/map effects, Wilson intervals, Holm-adjusted sign tests",
        "initial_collision_handling": "retain all scenes in primary analysis; common initial-collision-free secondary stratum",
        "source_snapshot": str(frozen), "source_manifest_sha256": sha256(frozen / "source_manifest.json"),
        "runtime_contract": runtime_contract, "response_sha256": sha256(response), "sage_sha256": sha256(sage),
        "analysis_sources": analysis_hashes, "validation_inputs": [str(path) for path in inputs],
        "physics_audits": [str(path) for path in audits],
        "claim_boundary": "shared patched external runtime and discrete joint action adapter; not official unmodified reproduction",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    status_path = output / "status.json"

    def status(state, **values):
        status_path.write_text(json.dumps({"status": state, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **values}, indent=2) + "\n")

    try:
        for filename, expected in json.loads((frozen / "source_manifest.json").read_text()).items():
            if sha256(frozen / filename) != expected:
                raise ValueError(f"frozen source hash mismatch: {filename}")
        required = audits + [path / "manifest.json" for path in inputs]
        deadline = time.monotonic() + args.input_timeout
        while missing := [str(path) for path in required if not path.exists()]:
            status("WAITING_FOR_PREREQUISITES", missing=missing)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"prerequisite artifacts missing: {missing}")
            time.sleep(30)
        verified = verify_physics_audits(audits, runtime_contract, list(SCENARIOS), minimum_ticks=960)
        (output / "physics_verification.json").write_text(json.dumps(verified, indent=2) + "\n")
        for path in inputs:
            metadata = json.loads((path / "metadata.json").read_text())
            manifest = json.loads((path / "manifest.json").read_text())
            if manifest.get("status") != "COMPLETED" or manifest.get("records") != 16:
                raise ValueError("validation does not contain 16 completed episode records")
            if (metadata["parameters"]["seed"] in SEEDS or metadata["parameters"]["max_steps"] != 60
                    or metadata["parameters"]["episodes"] != 16):
                raise ValueError("validation overlaps evaluation or uses a different horizon")
        status("CALIBRATING")
        subprocess.run([sys.executable, "-m", "sage.experiments.calibrate_socialgym2_belief",
                        "--output", str(calibration_path), *map(str, inputs)], cwd=frozen, check=True)
        scale = json.loads(calibration_path.read_text())["selected"]["likelihood_scale"]
        jobs = Queue()
        for scenario in SCENARIOS:
            for seed in SEEDS:
                jobs.put((scenario, seed))
        status("RUNNING", expected_policy_rows=1920, calibrated_scale=scale)
        failed = Event()

        def worker(number):
            environment = os.environ.copy()
            environment.update(ROS_MASTER_URI=f"http://127.0.0.1:{11321 + number}",
                               ROS_HOME=f"/tmp/h3_joint_workers/worker{number}",
                               ROS_LOG_DIR=f"/tmp/h3_joint_workers/worker{number}/log",
                               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            source = args.worker_root.resolve() / f"worker{number}/social_gym"
            while not failed.is_set():
                try:
                    scenario, seed = jobs.get_nowait()
                except Empty:
                    return
                for policy in POLICIES:
                    if failed.is_set():
                        return
                    target = output / f"{Path(scenario).name}_seed{seed}" / policy
                    target.parent.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, "-u", "-m", "sage.experiments.run_socialgym2_tracer_loop",
                               "--source", str(source), "--output-dir", str(target),
                               "--response-checkpoint", str(response), "--sage-checkpoint", str(sage),
                               "--runtime-manifest", str(runtime_path), "--calibration-artifact", str(calibration_path),
                               "--likelihood-scale", str(scale), "--scenario", scenario, "--seed", str(seed),
                               "--episodes", "16", "--max-steps", "60", "--num-agents", "3", "--num-humans", "2",
                               "--partial-observation", "--belief-selection-audit", "--policies", policy,
                               "--device", args.device]
                    print(f"worker={number} scenario={scenario} seed={seed} policy={policy}", flush=True)
                    with (target.parent / f"{policy}.log").open("w") as log:
                        try:
                            subprocess.run(command, cwd=frozen, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                        except BaseException:
                            failed.set()
                            raise
                    time.sleep(2)
                jobs.task_done()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker, number) for number in range(args.workers)]
            for future in futures:
                future.result()
        for filename, expected in analysis_hashes.items():
            if sha256(Path(filename)) != expected:
                raise RuntimeError("analysis source changed during the frozen campaign")
        all_inputs = [output / f"{Path(scenario).name}_seed{seed}" / policy
                      for scenario in SCENARIOS for seed in SEEDS for policy in POLICIES]
        status("AGGREGATING")
        subprocess.run([sys.executable, "-m", "sage.experiments.aggregate_socialgym2_tracer",
                        "--output-dir", str(output / "aggregate"), "--expected-policies", *POLICIES,
                        "--expected-seeds", *map(str, SEEDS), "--expected-scenarios", *SCENARIOS,
                        "--expected-episodes", "16", "--physics-audits", *map(str, audits), "--", *map(str, all_inputs)],
                       cwd=root, check=True)
        status("COMPLETED", expected_policy_rows=1920)
    except BaseException as error:
        status("FAILED", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frozen-code", type=Path, required=True)
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--calibration-input-root", type=Path, required=True)
    parser.add_argument("--physics-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-timeout", type=float, default=7200)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
