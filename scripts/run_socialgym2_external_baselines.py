#!/usr/bin/env python3
"""Run official PIPS controllers on the frozen v19 SocialGym2 scene lattice."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import sys
from threading import Event, Lock
import time

try:
    from scripts.run_final_socialgym2_confirmation import (
        POLICIES as V19_POLICIES,
        SCENARIOS,
        SEEDS,
        _clone_workers,
        _ros_overlay,
        _snapshot_source,
        _start_ros_masters,
        _stop_ros_masters,
        _worker_environment,
    )
except ModuleNotFoundError:
    from run_final_socialgym2_confirmation import (
        POLICIES as V19_POLICIES,
        SCENARIOS,
        SEEDS,
        _clone_workers,
        _ros_overlay,
        _snapshot_source,
        _start_ros_masters,
        _stop_ros_masters,
        _worker_environment,
    )


EXTERNAL_POLICIES = ("social-ref", "social-greedy")
PIPS_RUNTIME_LIBRARIES = (
    "libboost_thread.so.1.71.0",
    "libboost_chrono.so.1.71.0",
    "libboost_filesystem.so.1.71.0",
    "libboost_regex.so.1.71.0",
    "libz3.so.4",
    "liblog4cxx.so.10",
    "libconsole_bridge.so.0.4",
    "libglog.so.0",
    "libapr-1.so.0",
    "libaprutil-1.so.0",
    "libgflags.so.2.2",
    "libicui18n.so.66",
    "libicuuc.so.66",
    "libicudata.so.66",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stage_runtime_libraries(overlay: Path, destination: Path) -> dict[str, str]:
    destination.mkdir()
    search_roots = (
        overlay / "usr/lib/x86_64-linux-gnu",
        overlay / "lib/x86_64-linux-gnu",
    )
    staged = {}
    for name in PIPS_RUNTIME_LIBRARIES:
        candidates = [
            root / name
            for root in search_roots
            if (root / name).is_file()
        ]
        if not candidates:
            raise FileNotFoundError(f"PIPS runtime library is missing: {name}")
        target = destination / name
        target.symlink_to(candidates[0])
        staged[name] = _sha256(candidates[0])
    return staged


def run(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    source_manifest = _snapshot_source(root, snapshot)
    worker_templates = (
        root / "experiments/.tracer_final_closed_loop_confirmation_20260908_v5_workers"
    )
    workers_root = root / "experiments" / f".{output.name}_workers"
    _clone_workers(worker_templates, workers_root, args.workers)

    overlay = _ros_overlay(root)
    runtime_pips = overlay / "home/rosdev/social_gym/submodules/pips"
    runtime_libraries = output / "pips_runtime_libraries"
    runtime_library_hashes = _stage_runtime_libraries(overlay, runtime_libraries)
    binaries = {
        policy: runtime_pips / "bin" / executable
        for policy, executable in (
            ("social-ref", "social_ref"),
            ("social-greedy", "social_greed"),
        )
    }
    missing = [str(path) for path in binaries.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"official PIPS binaries are missing: {missing}")
    loader_environment = {
        **os.environ,
        "LD_LIBRARY_PATH": (
            f"{runtime_pips / 'lib'}:{runtime_libraries}:"
            f"{overlay / 'opt/ros/noetic/lib'}"
        ),
    }
    for binary in binaries.values():
        loader = subprocess.run(
            ["ldd", str(binary)],
            env=loader_environment,
            check=True,
            capture_output=True,
            text=True,
        )
        if "not found" in loader.stdout:
            raise RuntimeError(f"unresolved PIPS binary dependency:\n{loader.stdout}")

    python = root / ".conda/envs/hulan/bin/python"
    response = (
        root
        / "experiments/formal_v2/h2/persistent_trajectory_suite_tracer_effect/"
        "seed2027/response.pt"
    )
    sage = (
        root
        / "experiments/pilot/h3_sage_generator_p0_seed2027/"
        "synthetic_sage_epoch20.pt"
    )
    runtime = (
        root / "experiments/socialgym2_joint_runtime_manifest_20260906_v7/manifest.json"
    )
    calibration = (
        root
        / "experiments/socialgym2_joint_frozen_campaign_20260906_v1/"
        "validation_calibration.json"
    )
    physics = root / "experiments/socialgym2_joint_physics_full_20260906_v1"
    baseline = args.baseline.resolve()
    required = (
        python,
        response,
        sage,
        runtime,
        calibration,
        physics / "intersection.json",
        physics / "open.json",
        baseline / "aggregate/manifest.json",
        *(worker_templates / f"worker{index}/social_gym" for index in range(4)),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen SocialGym2 inputs: {missing}")

    official_commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(root / "raw_data/external/source_repos/socialgym2_official"),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    protocol = {
        "protocol": "socialgym2-official-pips-external-confirmation-v1",
        "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "paired_baseline": str(baseline),
        "seeds": list(SEEDS),
        "scenarios": list(SCENARIOS),
        "policies": list(EXTERNAL_POLICIES),
        "episodes_per_seed_scenario_policy": args.episodes,
        "maximum_planning_windows": args.max_steps,
        "expected_policy_jobs": len(SEEDS) * len(SCENARIOS) * len(EXTERNAL_POLICIES),
        "expected_external_rows": (
            len(SEEDS) * len(SCENARIOS) * len(EXTERNAL_POLICIES) * args.episodes
        ),
        "robots": 3,
        "humans": 2,
        "partial_observation": True,
        "action_contract": {
            "PIPS": "native GoAlone/Halt/Follow/Pass",
            "TRACER_v19": "frozen GoAlone/Halt",
            "projection": "none for PIPS",
        },
        "pips_binaries": {
            policy: {"path": str(path), "sha256": _sha256(path)}
            for policy, path in binaries.items()
        },
        "pips_runtime_libraries": runtime_library_hashes,
        "official_source_commit": official_commit,
        "source_manifest_sha256": _sha256(snapshot / "source_manifest.json"),
        "source_file_count": len(source_manifest),
        "isolated_worker_root": str(workers_root),
        "worker_count": args.workers,
        "ros_master_ports": [
            args.ros_master_base + worker for worker in range(args.workers)
        ],
        "claim_boundary": (
            "Same frozen SocialGym2 scene lattice and runtime as v19. PIPS executes "
            "its native four actions; TRACER v19 retains its frozen binary interface. "
            "This is an external-controller comparison, not action-space equivalence."
        ),
    }
    _write_json(output / "protocol.json", protocol)

    jobs: Queue[tuple[str, int, str]] = Queue()
    for scenario in SCENARIOS:
        for seed in SEEDS:
            for policy in EXTERNAL_POLICIES:
                jobs.put((scenario, seed, policy))
    failed = Event()
    errors: list[str] = []
    status_lock = Lock()
    progress = {"completed_policy_jobs": 0, "expected_policy_jobs": jobs.qsize()}

    def set_status(state: str, **values: object) -> None:
        with status_lock:
            _write_json(
                output / "status.json",
                {
                    "status": state,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    **progress,
                    **values,
                },
            )

    masters = []
    ros_state_root = Path("/tmp/h3_ros_overlay") / f"{output.name}_runner"
    try:
        masters = _start_ros_masters(
            root,
            args.workers,
            args.ros_master_base,
            output / "ros_master_logs",
        )
        set_status("RUNNING")

        def worker(number: int) -> None:
            source = workers_root / f"worker{number}/social_gym"
            environment = _worker_environment(
                root,
                snapshot,
                source,
                number,
                args.ros_master_base,
                ros_state_root,
            )
            environment["ROS_PACKAGE_PATH"] = (
                f"{runtime_pips}:{environment['ROS_PACKAGE_PATH']}"
            )
            environment["LD_LIBRARY_PATH"] = (
                f"{runtime_pips / 'lib'}:"
                f"{runtime_libraries}:"
                f"{environment['LD_LIBRARY_PATH']}"
            )
            environment["SOCIALGYM_SOURCE_COMMIT"] = official_commit
            time.sleep(number * args.launch_stagger_seconds)
            while not failed.is_set():
                try:
                    scenario, seed, policy = jobs.get_nowait()
                except Empty:
                    return
                target = output / f"{Path(scenario).name}_seed{seed}" / policy
                target.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(python),
                    "-u",
                    "-m",
                    "sage.experiments.run_socialgym2_tracer_loop",
                    "--source",
                    str(source),
                    "--output-dir",
                    str(target),
                    "--response-checkpoint",
                    str(response),
                    "--sage-checkpoint",
                    str(sage),
                    "--runtime-manifest",
                    str(runtime),
                    "--calibration-artifact",
                    str(calibration),
                    "--likelihood-scale",
                    "0.05",
                    "--scenario",
                    scenario,
                    "--seed",
                    str(seed),
                    "--episodes",
                    str(args.episodes),
                    "--max-steps",
                    str(args.max_steps),
                    "--num-agents",
                    "3",
                    "--num-humans",
                    "2",
                    "--partial-observation",
                    "--belief-selection-audit",
                    "--candidate-prefix-model",
                    "executable",
                    "--task-steps",
                    "2",
                    "--execution-position-margin",
                    "0.1",
                    "--stagnation-trigger-seconds",
                    "1.2",
                    "--stagnation-motion-threshold",
                    "0.02",
                    "--stagnation-halt-penalty",
                    "1.0",
                    "--policies",
                    policy,
                    "--device",
                    args.device,
                ]
                try:
                    for attempt in range(1, args.job_attempts + 1):
                        log_path = target.parent / f"{policy}.attempt{attempt}.log"
                        try:
                            with log_path.open("x", encoding="utf-8") as log:
                                subprocess.run(
                                    command,
                                    cwd=snapshot,
                                    env=environment,
                                    stdout=log,
                                    stderr=subprocess.STDOUT,
                                    check=True,
                                )
                            break
                        except subprocess.CalledProcessError:
                            if target.exists():
                                shutil.rmtree(target)
                            if attempt == args.job_attempts:
                                raise
                            time.sleep(10)
                    with status_lock:
                        progress["completed_policy_jobs"] += 1
                    set_status(
                        "RUNNING",
                        last_completed={
                            "scenario": scenario,
                            "seed": seed,
                            "policy": policy,
                        },
                    )
                except BaseException as error:
                    failed.set()
                    errors.append(
                        f"worker={number} scenario={scenario} seed={seed} "
                        f"policy={policy}: {error!r}"
                    )
                    set_status("FAILED", errors=errors)
                    return

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker, number) for number in range(args.workers)]
            for future in futures:
                future.result()
        if failed.is_set():
            raise RuntimeError("; ".join(errors))
    finally:
        _stop_ros_masters(masters)

    existing_inputs = [
        baseline / f"{Path(scenario).name}_seed{seed}" / policy
        for scenario in SCENARIOS
        for seed in SEEDS
        for policy in V19_POLICIES
    ]
    external_inputs = [
        output / f"{Path(scenario).name}_seed{seed}" / policy
        for scenario in SCENARIOS
        for seed in SEEDS
        for policy in EXTERNAL_POLICIES
    ]
    aggregate_command = [
        str(python),
        "-m",
        "sage.experiments.aggregate_socialgym2_tracer",
        "--output-dir",
        str(output / "aggregate"),
        "--expected-policies",
        *V19_POLICIES,
        *EXTERNAL_POLICIES,
        "--expected-seeds",
        *(str(seed) for seed in SEEDS),
        "--expected-scenarios",
        *SCENARIOS,
        "--expected-episodes",
        str(args.episodes),
        "--physics-audits",
        str(physics / "intersection.json"),
        str(physics / "open.json"),
        "--",
        *(str(path) for path in existing_inputs),
        *(str(path) for path in external_inputs),
    ]
    subprocess.run(aggregate_command, cwd=snapshot, check=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.annotate_socialgym2_policy_roles import annotate

    annotate(
        output / "aggregate",
        output / "aggregate/policy_roles.json",
        len(SEEDS) * len(SCENARIOS) * args.episodes,
    )
    protocol["status"] = "COMPLETED"
    protocol["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["aggregate"] = str(output / "aggregate")
    _write_json(output / "protocol.json", protocol)
    set_status("COMPLETED", aggregate=str(output / "aggregate"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/socialgym2_external_pips_confirmation_20260909_v1"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("experiments/tracer_final_closed_loop_confirmation_20260908_v7"),
    )
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ros-master-base", type=int, default=11461)
    parser.add_argument("--launch-stagger-seconds", type=float, default=2.0)
    parser.add_argument("--job-attempts", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.episodes != 16 or args.max_steps != 60:
        raise ValueError("v19 paired confirmation requires 16 episodes and 60 windows")
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must lie in [1, 16]")
    run(args)


if __name__ == "__main__":
    main()
