#!/usr/bin/env python3
"""Freeze and run the final paired SocialGym2 closed-loop confirmation."""

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
import signal
import socket
import subprocess
import sys
from threading import Event, Lock
import time


SEEDS = (3109, 3119, 3121, 3137, 3163, 3167, 3181, 3187, 3191, 3203)
SCENARIOS = ("envs/scenario/intersection", "envs/scenario/open")
POLICIES = (
    "tracer-persistent",
    "tracer-reset",
    "tracer-uniform",
    "tracer-no-pair",
    "official-goalone",
    "official-halt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _snapshot_source(root: Path, destination: Path) -> dict[str, str]:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name == "__pycache__" or name.endswith((".pyc", ".pyo"))}

    shutil.copytree(root / "sage", destination / "sage", ignore=ignore)
    manifest = {
        str(path.relative_to(destination)): _sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    _write_json(destination / "source_manifest.json", manifest)
    return manifest


def _verify_snapshot(snapshot: Path, manifest: dict[str, str]) -> None:
    actual_files = {
        str(path.relative_to(snapshot))
        for path in snapshot.rglob("*")
        if (
            path.is_file()
            and path.name != "source_manifest.json"
            and "__pycache__" not in path.parts
            and path.suffix not in (".pyc", ".pyo")
        )
    }
    if actual_files != set(manifest):
        raise RuntimeError("frozen source file set changed")
    changed = [relative for relative, expected in manifest.items() if _sha256(snapshot / relative) != expected]
    if changed:
        raise RuntimeError(f"frozen source hash mismatch: {changed[:3]}")


def _ros_overlay(root: Path) -> Path:
    runtime_manifest = root / "experiments/socialgym2_joint_runtime_manifest_20260906_v7/manifest.json"
    if runtime_manifest.is_file():
        runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
        roots = {
            Path(path.split("/home/rosdev/", 1)[0])
            for path in runtime.get("binaries", {})
            if "/home/rosdev/" in path
        }
        if len(roots) == 1 and next(iter(roots)).is_dir():
            return next(iter(roots))
    candidates = sorted((root / ".tools/docker-data/vfs/dir").glob(
        "*/opt/ros/noetic/lib/python3/dist-packages/roslib"
    ))
    candidates = [
        path
        for path in candidates
        if (path.parents[6] / "home/rosdev/social_gym/submodules/amrl_msgs/src/amrl_msgs").is_dir()
    ] or candidates
    if not candidates:
        raise RuntimeError("the ROS Python overlay used by the SocialGym2 workers is unavailable")
    return candidates[0].parents[6]


def _worker_environment(
    root: Path,
    snapshot: Path,
    source: Path,
    worker: int,
    master_base: int,
    ros_state_root: Path,
) -> dict[str, str]:
    overlay = _ros_overlay(root)
    worker_ros_state = ros_state_root / f"worker{worker}"
    (worker_ros_state / "log").mkdir(parents=True, exist_ok=True)
    runtime_source = overlay / "home/rosdev/social_gym"
    package_paths = (
        overlay / "opt/ros/noetic/share",
        source,
        source / "src",
        source / "submodules/ut_multirobot_sim",
        source / "submodules/ut_multirobot_sim/src",
        source / "submodules/amrl_msgs",
        source / "submodules/amrl_msgs/src",
        source / "submodules/amrl_maps",
        source / "submodules/pedsim_ros",
        source / "submodules/pedsim_ros/src",
        source / "submodules/graph_navigation",
        source / "submodules/graph_navigation/src",
        runtime_source,
        runtime_source / "submodules/ut_multirobot_sim",
        runtime_source / "submodules/amrl_msgs",
        runtime_source / "submodules/amrl_maps",
        runtime_source / "submodules/pedsim_ros",
        runtime_source / "submodules/graph_navigation",
    )
    python_paths = (
        overlay / "opt/ros/noetic/lib/python3/dist-packages",
        source,
        source / "src",
        runtime_source,
        runtime_source / "src",
        runtime_source / "submodules/amrl_msgs/src",
        runtime_source / "submodules/ut_multirobot_sim/src",
        runtime_source / "submodules/pedsim_ros/src",
        runtime_source / "submodules/graph_navigation/src",
        snapshot,
    )
    return {
        **os.environ,
        "PYTHONPATH": ":".join(str(path) for path in python_paths),
        "ROS_VERSION": "1",
        "ROS_PYTHON_VERSION": "3",
        "ROS_DISTRO": "noetic",
        "ROS_PACKAGE_PATH": ":".join(str(path) for path in package_paths),
        "ROS_MASTER_URI": f"http://127.0.0.1:{master_base + worker}",
        "ROS_HOME": str(worker_ros_state),
        "ROS_LOG_DIR": str(worker_ros_state / "log"),
        "ROS_IP": "127.0.0.1",
        "QT_QPA_PLATFORM": "offscreen",
        "PATH": f"{root / '.conda/envs/hulan/bin'}:{overlay / 'opt/ros/noetic/bin'}:{overlay / 'usr/bin'}:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{overlay / 'opt/ros/noetic/lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def _clone_workers(source_root: Path, destination: Path, workers: int) -> None:
    destination.mkdir()
    for worker in range(workers):
        source = source_root / f"worker{worker % 4}/social_gym"
        shutil.copytree(source, destination / f"worker{worker}/social_gym", symlinks=True)


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _start_ros_masters(
    root: Path,
    workers: int,
    master_base: int,
    log_root: Path,
) -> list[tuple[subprocess.Popen, object]]:
    occupied = [master_base + worker for worker in range(workers) if _port_open(master_base + worker)]
    if occupied:
        raise RuntimeError(f"dedicated ROS master ports are already occupied: {occupied}")
    overlay = _ros_overlay(root)
    log_root.mkdir()
    processes = []
    try:
        for worker in range(workers):
            state = f"/tmp/h3_ros_overlay/final_confirmation_{master_base + worker}"
            environment = {
                **os.environ,
                "HOME": "/root",
                "PATH": "/opt/ros/noetic/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "PYTHONPATH": "/opt/ros/noetic/lib/python3/dist-packages:/usr/lib/python3/dist-packages",
                "ROS_VERSION": "1",
                "ROS_PYTHON_VERSION": "3",
                "ROS_DISTRO": "noetic",
                "ROS_ROOT": "/opt/ros/noetic/share/ros",
                "ROS_ETC_DIR": "/opt/ros/noetic/etc/ros",
                "ROS_PACKAGE_PATH": "/opt/ros/noetic/share",
                "ROS_MASTER_URI": f"http://127.0.0.1:{master_base + worker}",
                "ROS_HOME": state,
                "ROS_LOG_DIR": f"{state}/log",
                "ROS_HOSTNAME": "127.0.0.1",
                "ROS_IP": "127.0.0.1",
                "CMAKE_PREFIX_PATH": "/opt/ros/noetic",
                "PKG_CONFIG_PATH": "/opt/ros/noetic/lib/pkgconfig",
                "LD_LIBRARY_PATH": "/opt/ros/noetic/lib:/usr/local/cuda/lib64",
            }
            log = (log_root / f"master{worker}.log").open("x", encoding="utf-8")
            process = subprocess.Popen(
                [
                    "chroot",
                    str(overlay),
                    "/usr/bin/python3",
                    "/opt/ros/noetic/bin/roscore",
                    "-p",
                    str(master_base + worker),
                ],
                cwd=overlay,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((process, log))
        deadline = time.monotonic() + 30.0
        while not all(_port_open(master_base + worker) for worker in range(workers)):
            failed = [process.returncode for process, _ in processes if process.poll() is not None]
            if failed:
                raise RuntimeError(f"a dedicated ROS master exited during startup: {failed}")
            if time.monotonic() >= deadline:
                raise TimeoutError("dedicated ROS masters did not become ready")
            time.sleep(0.25)
        return processes
    except BaseException:
        _stop_ros_masters(processes)
        raise


def _stop_ros_masters(processes: list[tuple[subprocess.Popen, object]]) -> None:
    for process, _ in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process, log in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        log.close()


def run(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    source_origin = args.frozen_source.resolve() if args.frozen_source else root
    if args.frozen_source:
        origin_manifest_path = source_origin / "source_manifest.json"
        if not origin_manifest_path.is_file():
            raise FileNotFoundError(f"frozen source manifest is missing: {origin_manifest_path}")
        origin_manifest = json.loads(origin_manifest_path.read_text(encoding="utf-8"))
        _verify_snapshot(source_origin, origin_manifest)
    else:
        origin_manifest = None
    manifest = _snapshot_source(source_origin, snapshot)
    if origin_manifest is not None and manifest != origin_manifest:
        raise RuntimeError("copied source does not match the selected frozen source")
    _verify_snapshot(snapshot, manifest)

    python = root / ".conda/envs/hulan/bin/python"
    response = root / "experiments/formal_v2/h2/persistent_trajectory_suite_tracer_effect/seed2027/response.pt"
    sage = root / "experiments/pilot/h3_sage_generator_p0_seed2027/synthetic_sage_epoch20.pt"
    runtime = root / "experiments/socialgym2_joint_runtime_manifest_20260906_v7/manifest.json"
    calibration = root / "experiments/socialgym2_joint_frozen_campaign_20260906_v1/validation_calibration.json"
    physics = root / "experiments/socialgym2_joint_physics_full_20260906_v1"
    worker_templates = root / "experiments/.tracer_final_closed_loop_confirmation_20260908_v5_workers"
    workers_root = root / "experiments" / f".{output.name}_workers"
    _clone_workers(worker_templates, workers_root, args.workers)
    launcher = Path(__file__).resolve()
    development_inputs = tuple(
        Path(
            f"experiments/socialgym2_execution_safety_20260907/"
            f"execution_prefix_dev_{scenario}_{version}_seed{seed}/summary.json"
        )
        for version in ("v19", "v20_hardrisk")
        for seed in (2309, 2311)
        for scenario in ("intersection", "open")
    )
    development_metadata = tuple(path.with_name("metadata.json") for path in development_inputs)
    required = (
        python,
        response,
        sage,
        runtime,
        calibration,
        physics / "intersection.json",
        physics / "open.json",
        launcher,
        *development_inputs,
        *development_metadata,
        *(worker_templates / f"worker{worker}/social_gym" for worker in range(4)),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen confirmation inputs: {missing}")

    frozen_inputs = {
        str(path): _sha256(path)
        for path in required
        if path.is_file()
    }
    for metadata_path in development_metadata:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for relative, expected in metadata["algorithm_sources"].items():
            source_path = snapshot / relative
            if not source_path.is_file() or _sha256(source_path) != expected:
                raise RuntimeError(
                    f"frozen source does not match development metadata: {metadata_path}: {relative}"
                )
    runtime_contract = json.loads(runtime.read_text(encoding="utf-8"))
    protocol = {
        "protocol": "tracer-socialgym2-final-closed-loop-confirmation-v1",
        "phase": "independent-final-confirmation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmation_seeds": list(SEEDS),
        "seed_audit": "absent from repository experiments and reports before launch",
        "scenarios": list(SCENARIOS),
        "policies": list(POLICIES),
        "episodes_per_seed_scenario_policy": args.episodes,
        "maximum_planning_windows": args.max_steps,
        "expected_independent_scenes": len(SEEDS) * len(SCENARIOS) * args.episodes,
        "expected_policy_rows": len(SEEDS) * len(SCENARIOS) * args.episodes * len(POLICIES),
        "controllable_robots": 3,
        "humans": 2,
        "partial_observation": True,
        "candidate_prefix_model": "executable",
        "task_steps": 2,
        "joint_geometry_filter": True,
        "hard_risk_filter": False,
        "execution_position_margin": 0.1,
        "stagnation_trigger_seconds": 1.2,
        "stagnation_motion_threshold": 0.02,
        "stagnation_halt_penalty": 1.0,
        "likelihood_scale": 0.05,
        "primary_outcomes": [
            "collision_free_team_success",
            "team_success",
            "collision",
            "task_progress",
        ],
        "secondary_outcomes": [
            "robot_success_fraction",
            "min_distance",
            "fallback_rate",
            "stagnation_window_rate",
        ],
        "statistics": "paired unique-scene bootstrap with map-stratified descriptive results",
        "initial_collision_handling": "retain all scenes; report the common initial-clean stratum secondarily",
        "development_selection": {
            "selected_version": "v19",
            "rejected_version": "v20_hardrisk",
            "seeds": [2309, 2311],
            "reserved_confirmation_used": False,
            "decision": "Retain v19 soft-risk: v20 hard-risk kept mean collision unchanged at 0.28125 while reducing mean collision-free team success from 0.375 to 0.25, team success from 0.5 to 0.34375, and task progress from 0.851485 to 0.783323.",
            "inputs": {
                str(path): frozen_inputs[str(path)]
                for path in (*development_inputs, *development_metadata)
            },
        },
        "source_snapshot": str(snapshot),
        "source_origin": str(source_origin),
        "source_origin_manifest_sha256": (
            _sha256(source_origin / "source_manifest.json") if args.frozen_source else None
        ),
        "source_manifest_sha256": _sha256(snapshot / "source_manifest.json"),
        "runtime_manifest_path": str(runtime),
        "runtime_contract": {
            "sha256": frozen_inputs[str(runtime)],
            "protocol": runtime_contract["protocol"],
            "boundary": runtime_contract["boundary"],
        },
        "frozen_inputs": frozen_inputs,
        "launcher_sha256": frozen_inputs[str(launcher)],
        "training_checkpoint_seeds": [2027],
        "interim_tuning": False,
        "noninferiority_claim": False,
        "full_tick_diagnostics": False,
        "full_tick_diagnostics_reason": "Primary confirmation stores episode and belief evidence; tick telemetry remains a prespecified separate diagnostic because it exceeds the workspace storage budget.",
        "isolated_worker_root": str(workers_root),
        "worker_count": args.workers,
        "parallelization": "independent map-seed-policy jobs",
        "worker_launch_stagger_seconds": args.launch_stagger_seconds,
        "transient_job_attempts": args.job_attempts,
        "dedicated_ros_master_ports": [args.ros_master_base + worker for worker in range(args.workers)],
        "claim_boundary": "Final paired external closed-loop confirmation in the shared patched SocialGym2 runtime; not unmodified official reproduction or a hard-safety guarantee.",
    }
    _write_json(output / "protocol.json", protocol)

    status_path = output / "status.json"
    status_lock = Lock()
    progress = {"completed_policy_jobs": 0, "expected_policy_jobs": len(SEEDS) * len(SCENARIOS) * len(POLICIES)}

    def set_status(state: str, **values: object) -> None:
        with status_lock:
            _write_json(status_path, {
                "status": state,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                **progress,
                **values,
            })

    masters = []
    ros_state_root = Path("/tmp/h3_ros_overlay") / f"{output.name}_runner"
    try:
        sys.path.insert(0, str(snapshot))
        from sage.experiments.socialgym2_physics_contract import verify_physics_audits

        verification = verify_physics_audits(
            [physics / "intersection.json", physics / "open.json"],
            protocol["runtime_contract"],
            list(SCENARIOS),
            minimum_ticks=960,
        )
        _write_json(output / "physics_verification.json", verification)
        masters = _start_ros_masters(
            root,
            args.workers,
            args.ros_master_base,
            output / "ros_master_logs",
        )
        set_status("RUNNING")

        jobs: Queue[tuple[str, int, str]] = Queue()
        for scenario in SCENARIOS:
            for seed in SEEDS:
                for policy in POLICIES:
                    jobs.put((scenario, seed, policy))
        failed = Event()

        def worker(number: int) -> None:
            source = workers_root / f"worker{number}/social_gym"
            environment = _worker_environment(
                root, snapshot, source, number, args.ros_master_base, ros_state_root,
            )
            time.sleep(number * args.launch_stagger_seconds)
            while not failed.is_set():
                try:
                    scenario, seed, policy = jobs.get_nowait()
                except Empty:
                    return
                try:
                    target = output / f"{Path(scenario).name}_seed{seed}" / policy
                    target.parent.mkdir(parents=True, exist_ok=True)
                    command = [
                        str(python), "-u", "-m", "sage.experiments.run_socialgym2_tracer_loop",
                        "--source", str(source),
                        "--output-dir", str(target),
                        "--response-checkpoint", str(response),
                        "--sage-checkpoint", str(sage),
                        "--runtime-manifest", str(runtime),
                        "--calibration-artifact", str(calibration),
                        "--likelihood-scale", "0.05",
                        "--scenario", scenario,
                        "--seed", str(seed),
                        "--episodes", str(args.episodes),
                        "--max-steps", str(args.max_steps),
                        "--num-agents", "3",
                        "--num-humans", "2",
                        "--partial-observation",
                        "--belief-selection-audit",
                        "--candidate-prefix-model", "executable",
                        "--task-steps", "2",
                        "--joint-geometry-filter",
                        "--execution-position-margin", "0.1",
                        "--stagnation-trigger-seconds", "1.2",
                        "--stagnation-motion-threshold", "0.02",
                        "--stagnation-halt-penalty", "1.0",
                        "--policies", policy,
                        "--device", args.device,
                    ]
                    for attempt in range(1, args.job_attempts + 1):
                        print(
                            f"worker={number} scenario={scenario} seed={seed} "
                            f"policy={policy} attempt={attempt}",
                            flush=True,
                        )
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
                    set_status("RUNNING", last_completed={"scenario": scenario, "seed": seed, "policy": policy})
                    time.sleep(2)
                except BaseException:
                    failed.set()
                    raise
                finally:
                    jobs.task_done()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker, number) for number in range(args.workers)]
            for future in futures:
                future.result()

        _verify_snapshot(snapshot, manifest)
        changed_inputs = [str(path) for path, expected in frozen_inputs.items() if _sha256(Path(path)) != expected]
        if changed_inputs:
            raise RuntimeError(f"frozen confirmation inputs changed: {changed_inputs}")

        inputs = [
            output / f"{Path(scenario).name}_seed{seed}" / policy
            for scenario in SCENARIOS
            for seed in SEEDS
            for policy in POLICIES
        ]
        set_status("AGGREGATING")
        aggregate_command = [
            str(python), "-u", "-m", "sage.experiments.aggregate_socialgym2_tracer",
            "--output-dir", str(output / "aggregate"),
            "--expected-policies", *POLICIES,
            "--expected-seeds", *map(str, SEEDS),
            "--expected-scenarios", *SCENARIOS,
            "--expected-episodes", str(args.episodes),
            "--physics-audits", str(physics / "intersection.json"), str(physics / "open.json"),
            "--", *map(str, inputs),
        ]
        with (output / "aggregate.log").open("x", encoding="utf-8") as log:
            subprocess.run(
                aggregate_command,
                cwd=snapshot,
                env=_worker_environment(
                    root,
                    snapshot,
                    workers_root / "worker0/social_gym",
                    0,
                    args.ros_master_base,
                    ros_state_root,
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        set_status("COMPLETED", aggregate=str(output / "aggregate/summary.json"))
    except BaseException as error:
        set_status("FAILED", error=repr(error))
        raise
    finally:
        _stop_ros_masters(masters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--frozen-source", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ros-master-base", type=int, default=11421)
    parser.add_argument("--launch-stagger-seconds", type=float, default=1.5)
    parser.add_argument("--job-attempts", type=int, default=2)
    args = parser.parse_args()
    if args.episodes != 16 or args.max_steps != 60:
        raise ValueError("the final confirmation protocol is frozen at 16 episodes and 60 planning windows")
    if not 1 <= args.workers <= 40:
        raise ValueError("workers must be between 1 and the memory-safe limit of 40")
    if args.launch_stagger_seconds < 0 or args.job_attempts not in (1, 2):
        raise ValueError("launch staggering must be nonnegative and job attempts must be one or two")
    run(args)


if __name__ == "__main__":
    main()
