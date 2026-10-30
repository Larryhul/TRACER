from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from run_final_socialgym2_confirmation import (
    _clone_workers,
    _snapshot_source,
    _start_ros_masters,
    _stop_ros_masters,
    _worker_environment,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2461)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--humans", type=int, default=8)
    parser.add_argument("--port", type=int, default=11941)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selection-note", default="First four episodes of seed 2461, no geometry prescreen.")
    parser.add_argument("--policies", nargs="+", default=[
        "tracer-persistent", "tracer-no-pair", "tracer-reset", "official-goalone",
    ])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root / "experiments"):
        raise ValueError("output must be under the experiments directory mounted in the ROS runtime")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    _snapshot_source(root, snapshot)
    workers = output / "workers"
    _clone_workers(root / "experiments/socialgym2_joint_workers_20260906", workers, 1)
    source = workers / "worker0/social_gym"
    environment = _worker_environment(root, snapshot, source, 0, args.port, output / "ros_state")
    command = [
        sys.executable, "-u", "-m", "tracer.experiments.run_socialgym2_tracer_loop",
        "--source", str(source), "--output-dir", str(output / "run"),
        "--response-checkpoint", str(root / "experiments/formal_v2/h2/persistent_trajectory_suite_tracer_effect/seed2027/response.pt"),
        "--sage-checkpoint", str(root / "experiments/pilot/h3_sage_generator_p0_seed2027/synthetic_sage_epoch20.pt"),
        "--runtime-manifest", str(root / "experiments/socialgym2_joint_runtime_manifest_20260906_v7/manifest.json"),
        "--likelihood-scale", "0.05", "--scenario", "envs/scenario/open",
        "--seed", str(args.seed), "--episodes", str(args.episodes),
        "--max-steps", "60", "--num-agents", "3", "--num-humans", str(args.humans),
        "--belief-selection-audit", "--execution-diagnostics",
        "--candidate-prefix-model", "executable", "--task-steps", "2",
        "--joint-geometry-filter", "--execution-position-margin", "0.1",
        "--stagnation-trigger-seconds", "1.2", "--stagnation-motion-threshold", "0.02",
        "--stagnation-halt-penalty", "1.0", "--device", args.device,
        "--policies", *args.policies,
    ]
    protocol = {
        "status": "RUNNING", "purpose": "density-transfer qualitative screening only",
        "seed": args.seed, "episodes": list(range(args.episodes)), "robots": 3,
        "humans": args.humans, "policies": args.policies, "command": command,
        "scene_selection": args.selection_note,
        "observation": "full simulator observations; no partial-observation masking",
        "calibration": "Likelihood scale 0.05 transferred unchanged from two-human validation; not certified at the new density. The two-human calibration artifact is deliberately not passed as if valid for eight humans.",
        "boundary": "New exploratory density protocol, separate from the ongoing frozen H3 campaign. No SICNav or MPC-CVMM multi-robot integration is claimed.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    masters = _start_ros_masters(root, 1, args.port, output / "ros_masters")
    process = None
    try:
        with (output / "runner.log").open("x") as log:
            process = subprocess.Popen(command, cwd=snapshot, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            return_code = process.wait(timeout=1800)
            protocol["return_code"] = return_code
            protocol["status"] = "COMPLETED" if return_code == 0 else "FAILED"
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            protocol["status"] = "INTERRUPTED"
        _stop_ros_masters(masters)
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({key: protocol[key] for key in ("status", "seed", "robots", "humans")}))
    if protocol["status"] != "COMPLETED":
        raise SystemExit(protocol.get("return_code", 1))


if __name__ == "__main__":
    main()
