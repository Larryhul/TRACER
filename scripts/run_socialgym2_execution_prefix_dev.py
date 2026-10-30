#!/usr/bin/env python3
"""Paired development run for the observable execution-prefix candidate."""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = Path(os.environ.get("SOCIALGYM2_EXECUTION_SOURCE_ROOT", ROOT)).resolve()
OUTPUT = Path(os.environ.get("SOCIALGYM2_EXECUTION_DEV_ROOT", "experiments/socialgym2_execution_safety_20260907"))
VERSION = os.environ.get("SOCIALGYM2_EXECUTION_DEV_VERSION", "v1")
SCENARIOS = tuple(filter(None, os.environ.get("SOCIALGYM2_EXECUTION_DEV_SCENARIOS", "intersection,open").split(",")))
CANDIDATE_PREFIX_MODEL = os.environ.get("SOCIALGYM2_CANDIDATE_PREFIX_MODEL", "executable")
CANDIDATE_MAX_SPEED = float(os.environ.get("SOCIALGYM2_CANDIDATE_MAX_SPEED", "1.5"))
CANDIDATE_NOMINAL_START_SPEED = float(
    os.environ.get("SOCIALGYM2_CANDIDATE_NOMINAL_START_SPEED", "1.5")
)
TASK_AGGREGATION = os.environ.get("SOCIALGYM2_TASK_AGGREGATION", "mean")
TASK_MIN_BLEND = float(os.environ.get("SOCIALGYM2_TASK_MIN_BLEND", "0.5"))
SEED = int(os.environ.get("SOCIALGYM2_EXECUTION_DEV_SEED", "2309"))
EPISODES = int(os.environ.get("SOCIALGYM2_EXECUTION_DEV_EPISODES", "8"))
MAX_STEPS = int(os.environ.get("SOCIALGYM2_EXECUTION_DEV_MAX_STEPS", "60"))
WORKER_OFFSET = int(os.environ.get("SOCIALGYM2_EXECUTION_DEV_WORKER_OFFSET", "0"))
HARD_RISK_FILTER = os.environ.get("SOCIALGYM2_EXECUTION_DEV_HARD_RISK", "0") == "1"
STAGNATION_RELEASE_HARD_SAFE = os.environ.get(
    "SOCIALGYM2_STAGNATION_RELEASE_HARD_SAFE", "0"
) == "1"
STAGNATION_RELEASE_REQUIRE_ROBUST_ROBOT = os.environ.get(
    "SOCIALGYM2_STAGNATION_RELEASE_REQUIRE_ROBUST_ROBOT", "0"
) == "1"
STAGNATION_RELEASE_REQUIRE_NONWORSENING_ROBOT = os.environ.get(
    "SOCIALGYM2_STAGNATION_RELEASE_REQUIRE_NONWORSENING_ROBOT", "0"
) == "1"
_robot_floor = os.environ.get("SOCIALGYM2_STAGNATION_RELEASE_ROBOT_CLEARANCE_FLOOR")
STAGNATION_RELEASE_ROBOT_CLEARANCE_FLOOR = (
    float(_robot_floor) if _robot_floor is not None else None
)
STAGNATION_RECOVERY_SAFETY_FIRST = os.environ.get(
    "SOCIALGYM2_STAGNATION_RECOVERY_SAFETY_FIRST", "0"
) == "1"
ROS_WORKER_BASE = int(os.environ.get("SOCIALGYM2_ROS_WORKER_BASE", "0"))
RUN_ROLE = os.environ.get(
    "SOCIALGYM2_EXECUTION_ROLE",
    "iterative development evaluation of execution consistency and joint safety",
)
RESERVED_CONFIRMATION_USED = os.environ.get("SOCIALGYM2_RESERVED_CONFIRMATION_USED", "0") == "1"
PYTHON = ROOT / ".conda/envs/hulan/bin/python"
SOURCE_ROOT = Path(os.environ.get(
    "SOCIALGYM2_EXECUTION_WORKERS_ROOT",
    ROOT / "experiments/socialgym2_joint_workers_20260906",
)).resolve()
RESPONSE = ROOT / "experiments/formal_v2/h2/persistent_trajectory_suite_tracer_effect/seed2027/response.pt"
SAGE = ROOT / "experiments/pilot/h3_sage_generator_p0_seed2027/synthetic_sage_epoch20.pt"
RUNTIME = ROOT / "experiments/socialgym2_joint_runtime_manifest_20260906_v7/manifest.json"
CALIBRATION = ROOT / "experiments/socialgym2_joint_frozen_campaign_20260906_v1/validation_calibration.json"


def _ros_overlay() -> Path:
    candidates = sorted((ROOT / ".tools/docker-data/vfs/dir").glob(
        "*/opt/ros/noetic/lib/python3/dist-packages/roslib"
    ))
    candidates = [
        path for path in candidates
        if (path.parents[6] / "home/rosdev/social_gym/submodules/amrl_msgs/src/amrl_msgs").is_dir()
    ] or candidates
    if not candidates:
        raise RuntimeError("the ROS Python overlay used by the SocialGym2 workers is unavailable")
    return candidates[0].parents[6]


def evaluate(scenario: str, worker: int) -> str:
    output = OUTPUT / f"execution_prefix_dev_{scenario}_{VERSION}"
    overlay = _ros_overlay()
    ros_python = overlay / "opt/ros/noetic/lib/python3/dist-packages"
    source = SOURCE_ROOT / f"worker{worker}/social_gym"
    runtime_source = overlay / "home/rosdev/social_gym"
    ros_worker = ROS_WORKER_BASE + worker
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
    environment = dict(
        os.environ,
        PYTHONPATH=":".join(str(path) for path in (
            ros_python,
            source,
            source / "src",
            runtime_source,
            runtime_source / "src",
            runtime_source / "submodules/amrl_msgs/src",
            runtime_source / "submodules/ut_multirobot_sim/src",
            runtime_source / "submodules/pedsim_ros/src",
            runtime_source / "submodules/graph_navigation/src",
            ALGORITHM_ROOT,
        )),
        ROS_VERSION="1",
        ROS_PYTHON_VERSION="3",
        ROS_DISTRO="noetic",
        ROS_PACKAGE_PATH=":".join(str(path) for path in package_paths),
        ROS_MASTER_URI=f"http://127.0.0.1:{11321 + ros_worker}",
        ROS_HOME=f"/tmp/h3_joint_workers/worker{ros_worker}",
        ROS_LOG_DIR=f"/tmp/h3_joint_workers/worker{ros_worker}/log",
        ROS_IP="127.0.0.1",
        QT_QPA_PLATFORM="offscreen",
        PATH=f"{ROOT / '.conda/envs/hulan/bin'}:{overlay / 'opt/ros/noetic/bin'}:{overlay / 'usr/bin'}:{os.environ.get('PATH', '')}",
        LD_LIBRARY_PATH=f"{overlay / 'opt/ros/noetic/lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    command = [
        str(PYTHON), "-u", "-m", "tracer.experiments.run_socialgym2_tracer_loop",
        "--source", str(SOURCE_ROOT / f"worker{worker}/social_gym"),
        "--output-dir", str(output),
        "--response-checkpoint", str(RESPONSE),
        "--sage-checkpoint", str(SAGE),
        "--runtime-manifest", str(RUNTIME),
        "--calibration-artifact", str(CALIBRATION),
        "--likelihood-scale", ".05", "--scenario", f"envs/scenario/{scenario}",
        "--seed", str(SEED), "--episodes", str(EPISODES), "--max-steps", str(MAX_STEPS),
        "--num-agents", "3", "--num-humans", "2", "--partial-observation",
        "--belief-selection-audit", "--policies", "tracer-persistent",
        "--execution-diagnostics", "--joint-geometry-filter",
        "--candidate-prefix-model", CANDIDATE_PREFIX_MODEL,
        "--candidate-max-speed", str(CANDIDATE_MAX_SPEED),
        "--candidate-nominal-start-speed", str(CANDIDATE_NOMINAL_START_SPEED),
        "--task-aggregation", TASK_AGGREGATION,
        "--task-min-blend", str(TASK_MIN_BLEND),
        "--device", "cuda",
    ]
    if HARD_RISK_FILTER:
        command.append("--hard-risk-filter")
    if STAGNATION_RELEASE_HARD_SAFE:
        command.append("--stagnation-release-hard-safe")
    if STAGNATION_RELEASE_REQUIRE_ROBUST_ROBOT:
        command.append("--stagnation-release-require-robust-robot")
    if STAGNATION_RELEASE_REQUIRE_NONWORSENING_ROBOT:
        command.append("--stagnation-release-require-nonworsening-robot")
    if STAGNATION_RELEASE_ROBOT_CLEARANCE_FLOOR is not None:
        command.extend([
            "--stagnation-release-robot-clearance-floor="
            f"{STAGNATION_RELEASE_ROBOT_CLEARANCE_FLOOR}",
        ])
    if STAGNATION_RECOVERY_SAFETY_FIRST:
        command.append("--stagnation-recovery-safety-first")
    log_path = OUTPUT / f"execution_prefix_dev_{scenario}_{VERSION}.log"
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ALGORITHM_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    subprocess.run([
        str(PYTHON), "-m", "tracer.experiments.analyze_socialgym2_execution",
        "--input", str(output),
        "--output", str(OUTPUT / f"execution_prefix_dev_{scenario}_diagnosis_{VERSION}.json"),
    ], cwd=ALGORITHM_ROOT, env=environment, check=True)
    return str(output)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "role": RUN_ROLE,
        "source": str(ALGORITHM_ROOT), "workers_root": str(SOURCE_ROOT),
        "seed": SEED, "episodes_per_map": EPISODES, "max_steps": MAX_STEPS,
        "maps": list(SCENARIOS), "policy": "tracer-persistent",
        "candidate_prefix_model": CANDIDATE_PREFIX_MODEL,
        "candidate_max_speed": CANDIDATE_MAX_SPEED,
        "candidate_nominal_start_speed": CANDIDATE_NOMINAL_START_SPEED,
        "task_aggregation": TASK_AGGREGATION,
        "task_min_blend": TASK_MIN_BLEND,
        "hard_risk_filter": HARD_RISK_FILTER,
        "development_changes": [
            "observable executable candidate prefix",
            "swept joint-geometry filter with initial-overlap recovery",
            "measured execution-error safety margin",
            "observable one-window swept robot-human guard",
            "hard-safe nonworsening recovery inside the robust margin after observed stagnation",
            "stagnation-aware action ranking",
        ] + ([
            "optional hard-safe stagnation release without relaxing hard robot/static/human thresholds",
        ] if STAGNATION_RELEASE_HARD_SAFE else []),
        "baseline": {
            "intersection": str(OUTPUT / "baseline_diagnostics_intersection_v2"),
            "open": str(OUTPUT / "baseline_diagnostics_open_v2"),
        },
        "reserved_confirmation_used": RESERVED_CONFIRMATION_USED,
        "stagnation_release_hard_safe": STAGNATION_RELEASE_HARD_SAFE,
        "stagnation_release_require_robust_robot": STAGNATION_RELEASE_REQUIRE_ROBUST_ROBOT,
        "stagnation_release_require_nonworsening_robot": STAGNATION_RELEASE_REQUIRE_NONWORSENING_ROBOT,
        "stagnation_release_robot_clearance_floor": STAGNATION_RELEASE_ROBOT_CLEARANCE_FLOOR,
        "stagnation_recovery_safety_first": STAGNATION_RECOVERY_SAFETY_FIRST,
        "ros_worker_base": ROS_WORKER_BASE,
        "selection_rule": "retain only if collision or collision-free team success improves without material team-success/progress regression across maps",
    }
    protocol_path = OUTPUT / f"execution_prefix_dev_protocol_{VERSION}.json"
    if not protocol_path.exists():
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    worker_by_scenario = {
        "intersection": (1 + WORKER_OFFSET) % 4,
        "open": (2 + WORKER_OFFSET) % 4,
    }
    jobs = [(scenario, worker_by_scenario[scenario]) for scenario in SCENARIOS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        outputs = list(executor.map(lambda item: evaluate(*item), jobs))
    (OUTPUT / f"execution_prefix_dev_completed_{VERSION}.json").write_text(
        json.dumps({"status": "COMPLETED", "outputs": outputs}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
