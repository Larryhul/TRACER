"""Run the official SICNav policies on the official CrowdSimPlus simulator.

This is a literature-baseline audit for the TRACER paper.  It deliberately
keeps the SICNav implementation and simulator in the ignored external source
tree; the runner only supplies the official config, fixed test cases, and
collects comparable navigation diagnostics.  It is not a TRACER response-
belief loop and it does not claim multi-robot equivalence.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tracer.experiments.navigation_contract import initial_navigation_state, navigation_outcome


POLICY_NAMES = (
    "sicnav-np",
    "sicnav-p",
    "official-mpc-cvmm",
    "official-dwa",
    "official-orca",
    "official-linear",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_for_metadata(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    status = run("status", "--porcelain").splitlines()
    # Permit a clean audit worktree to mount the immutable external checkout.
    unexpected_status = [line for line in status if line != "?? raw_data/external"]
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(unexpected_status),
    }


def _bootstrap(values: list[float], seed: int, samples: int = 4000) -> dict[str, Any]:
    if not values:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "count": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "count": int(len(array)),
    }


def _import_official(external_root: Path):
    source = str(external_root.resolve())
    rvo2_source = external_root.parent / "python_rvo2_official"
    for item in (source, str(rvo2_source.resolve())):
        if item not in sys.path:
            sys.path.insert(0, item)
    import gym  # noqa: F401  # imported for the official environment package
    from crowd_sim_plus.envs.crowd_sim_plus import CrowdSimPlus
    from crowd_sim_plus.envs.policy.linear import Linear
    from crowd_sim_plus.envs.policy.orca import ORCA
    from crowd_sim_plus.envs.policy.policy_factory import policy_factory
    from crowd_sim_plus.envs.utils.action import ActionRot, ActionXY
    from crowd_sim_plus.envs.utils.robot_plus import Robot, RobotFullKnowledge
    from sicnav.policy.campc import CollisionAvoidMPC
    from sicnav.policy.dwa import DynamicWindowApproach

    policy_factory["campc"] = CollisionAvoidMPC
    policy_factory["dwa"] = DynamicWindowApproach
    return CrowdSimPlus, Linear, ORCA, ActionRot, ActionXY, Robot, RobotFullKnowledge, CollisionAvoidMPC, DynamicWindowApproach


def _new_episode(
    external_root: Path,
    policy_name: str,
    case: int,
    env_config_path: Path,
    policy_config_path: Path,
):
    CrowdSimPlus, Linear, ORCA, ActionRot, ActionXY, Robot, RobotFullKnowledge, CollisionAvoidMPC, DynamicWindowApproach = _import_official(external_root)
    env_config = configparser.RawConfigParser()
    if not env_config.read(env_config_path):
        raise FileNotFoundError(env_config_path)
    policy_config = configparser.RawConfigParser()
    if not policy_config.read(policy_config_path):
        raise FileNotFoundError(policy_config_path)

    env_config.set("robot", "visible", "true")
    if policy_name in {"sicnav-np", "sicnav-p", "official-mpc-cvmm"}:
        env_config.set("robot", "policy", "campc")
    elif policy_name == "official-orca":
        env_config.set("robot", "policy", "orca_plus")
    elif policy_name == "official-dwa":
        env_config.set("robot", "policy", "dwa")
    else:
        env_config.set("robot", "policy", "linear")

    env = CrowdSimPlus()
    env.configure(env_config)
    if policy_name in {"sicnav-np", "sicnav-p", "official-mpc-cvmm"}:
        if policy_name == "official-mpc-cvmm":
            policy_config.set("mpc_env", "hum_model", "cvmm")
        policy_config.set("mpc_env", "priviledged_info", "true" if policy_name == "sicnav-p" else "false")
        policy = CollisionAvoidMPC()
        policy.configure(policy_config)
    elif policy_name == "official-orca":
        policy = ORCA()
        policy.configure(policy_config)
    elif policy_name == "official-dwa":
        policy = DynamicWindowApproach()
        policy.configure(policy_config)
        policy.configure_dwa(policy_config, env_config)
    else:
        policy = Linear()
        policy.configure(policy_config)
    robot_class = RobotFullKnowledge if policy_name == "sicnav-p" else Robot
    robot = robot_class(env_config, "robot")
    robot.set_policy(policy)
    env.set_robot(robot)
    policy.set_phase("test")
    policy.set_device(None)
    policy.set_env(env)
    observation, static_obs = env.reset("test", case, return_stat=True)
    return env, robot, policy, observation, static_obs, ActionRot, ActionXY


def _val(info: Any, key: str) -> float:
    value = info.get(key)
    return float(value.val) if value is not None else 0.0


def _run_episode(
    external_root: Path,
    policy_name: str,
    case: int,
    env_config_path: Path,
    policy_config_path: Path,
    max_steps: int,
) -> dict[str, Any]:
    env, robot, policy, observation, static_obs, ActionRot, ActionXY = _new_episode(
        external_root, policy_name, case, env_config_path, policy_config_path
    )
    del observation
    initial_position = np.asarray(robot.get_position(), dtype=np.float64)
    initial_contract = initial_navigation_state(env, robot, static_obs)
    goal = np.asarray(robot.get_goal_position(), dtype=np.float64)
    initial_goal_distance = max(float(np.linalg.norm(goal - initial_position)), 1e-6)
    min_distance = float("inf")
    collision_events = 0
    wall_collision_events = 0
    frozen_events = 0
    danger_events = 0
    final_info = ""
    solver_times: list[float] = []
    steps = 0
    started = time.perf_counter()
    while steps < max_steps:
        current_observation = [human.get_observable_state() for human in env.humans]
        action_started = time.perf_counter()
        action = robot.act(current_observation, static_obs)
        solver_times.append(time.perf_counter() - action_started)
        if not isinstance(action, (ActionRot, ActionXY)):
            raise TypeError(f"official policy returned unsupported action: {type(action)!r}")
        _, _, done, info = env.step(action)
        steps += 1
        final_info = "|".join(sorted(info)) if info else ""
        collision_events += int(_val(info, "Collision") != 0.0)
        wall_collision_events += int(_val(info, "WallCollision") != 0.0)
        frozen_events += int(_val(info, "Frozen") != 0.0)
        danger_events += int(_val(info, "Danger") != 0.0)
        robot_position = np.asarray(robot.get_position(), dtype=np.float64)
        for human in env.humans:
            min_distance = min(min_distance, float(np.linalg.norm(robot_position - np.asarray(human.get_position()))))
        if done:
            break

    reached_goal = float(_val(info, "ReachGoal") != 0.0)
    timeout = float(_val(info, "Timeout") != 0.0)
    final_position = np.asarray(robot.get_position(), dtype=np.float64)
    task_progress = (initial_goal_distance - float(np.linalg.norm(goal - final_position))) / initial_goal_distance
    official_solver_times = getattr(policy, "calc_times", [])
    if official_solver_times:
        solver_times = [float(value) for value in official_solver_times if math.isfinite(float(value))]
    return {
        "policy": policy_name,
        "case": int(case),
        "steps": int(steps),
        **navigation_outcome(reached_goal, timeout, steps, max_steps, collision_events, wall_collision_events),
        **initial_contract,
        "collision": float(collision_events > 0),
        "collision_events": int(collision_events),
        "wall_collision": float(wall_collision_events > 0),
        "wall_collision_events": int(wall_collision_events),
        "frozen_events": int(frozen_events),
        "danger_events": int(danger_events),
        "task_progress": float(task_progress),
        "total_time": float(env.global_time),
        "min_distance": float(min_distance) if math.isfinite(min_distance) else float("nan"),
        "solver_time_mean": float(statistics.fmean(solver_times)) if solver_times else float("nan"),
        "solver_time_p95": float(np.quantile(solver_times, 0.95)) if solver_times else float("nan"),
        "wall_time_seconds": float(time.perf_counter() - started),
        "final_info": final_info,
    }


def _summary(rows: list[dict[str, Any]], seed: int, policies: tuple[str, ...]) -> dict[str, Any]:
    metrics = ("success", "collision_free_success", "timeout", "environment_timeout", "budget_exhausted", "collision", "wall_collision", "task_progress", "total_time", "min_distance", "solver_time_mean", "solver_time_p95")
    result: dict[str, Any] = {"policies": {}}
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        result["policies"][policy] = {
            metric: _bootstrap(
                [float(row[metric]) for row in selected if math.isfinite(float(row[metric]))],
                seed + index,
            )
            for index, metric in enumerate(metrics)
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    git = _git_state(root)
    if git["dirty"] and not getattr(args, "allow_dirty", False):
        raise RuntimeError("official SICNav experiment requires a clean main-repository worktree")
    external_root = (root / args.external_root).resolve()
    env_config_path = (root / args.env_config).resolve()
    policy_config_path = (root / args.policy_config).resolve()
    output = (root / args.output_dir).resolve()
    if (output / "episode_metrics.csv").exists():
        raise ValueError("official benchmark output already contains results; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    cases = list(range(args.case_start, args.case_start + args.cases))
    policies = tuple(args.policies)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        for policy in policies:
            print(f"sicnav_official case={case} policy={policy}", flush=True)
            rows.append(_run_episode(external_root, policy, case, env_config_path, policy_config_path, args.max_steps))

    fieldnames = list(rows[0]) if rows else []
    with (output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows, args.bootstrap_seed, policies)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    external_commit = subprocess.check_output(["git", "-C", str(external_root), "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "protocol": "sicnav-official-crowdsimplus-baseline-v2",
        "claim_boundary": "official SICNav navigation baseline on official CrowdSimPlus; not a TRACER response-belief loop and not a multi-robot result",
        "git": git,
        "development_run": bool(git["dirty"]),
        "source_hashes": {_path_for_metadata(path, root): _sha256(path) for path in (
            Path(__file__).resolve(), root / "tracer/experiments/navigation_contract.py", env_config_path, policy_config_path
        )},
        "measurement_contract": {
            "min_distance": "minimum post-step center-to-center human distance",
            "timeout": "environment timeout or exhausted step budget without arrival",
            "kinematics": "official policy native kinematics; check each episode before paired claims",
        },
        "device": "cpu-casadi-ipopt",
        "external_repository": {"path": _path_for_metadata(external_root, root), "commit": external_commit},
        "parameters": {
            "external_root": _path_for_metadata(external_root, root),
            "env_config": _path_for_metadata(env_config_path, root),
            "policy_config": _path_for_metadata(policy_config_path, root),
            "cases": cases,
            "case_start": args.case_start,
            "policies": policies,
            "max_steps": args.max_steps,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        f"# Official SICNav baseline\n\nOfficial source commit `{external_commit}`; main commit `{git['commit']}`; clean `{not git['dirty']}`; cases `{args.case_start}--{args.case_start + args.cases - 1}`.",
        "\nThe run uses the official CrowdSimPlus simulator and official SICNav CAMPC implementation. It is a single-robot baseline and is not counted as TRACER execution.",
        "\n## Results\n",
    ]
    for policy in policies:
        metrics = summary["policies"][policy]
        lines.append(
            f"- `{policy}`: success `{metrics['success']['mean']:.3f}`, collision `{metrics['collision']['mean']:.3f}`, "
            f"task progress `{metrics['task_progress']['mean']:.3f}`, min distance `{metrics['min_distance']['mean']:.3f}`, "
            f"solver mean `{metrics['solver_time_mean']['mean']:.3f}s`."
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = {name: _sha256(output / name) for name in ("metadata.json", "summary.json", "episode_metrics.csv", "report.md")}
    manifest = {
        "status": "COMPLETED",
        "protocol": metadata["protocol"],
        "git": git,
        "external_repository": metadata["external_repository"],
        "artifacts": artifacts,
        "claim_boundary": metadata["claim_boundary"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"metadata": metadata, "summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", default="raw_data/external/source_repos/sicnav_official_filtered")
    parser.add_argument("--env-config", default="raw_data/external/source_repos/sicnav_official_filtered/sicnav/configs/env.config")
    parser.add_argument("--policy-config", default="raw_data/external/source_repos/sicnav_official_filtered/sicnav/configs/policy.config")
    parser.add_argument("--output-dir", default="experiments/tracer_sicnav_official_20260903")
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--allow-dirty", action="store_true", help="development run with explicit dirty/source-hash metadata")
    parser.add_argument("--bootstrap-seed", type=int, default=828731)
    parser.add_argument("--policies", nargs="+", choices=POLICY_NAMES, default=list(POLICY_NAMES))
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
