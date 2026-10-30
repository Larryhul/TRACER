"""Run the official SocialGym2 ROS environment without SB3 or RViz.

The runner calls the official ``RosSocialEnv`` and ``ManualScenario`` classes
from the mounted SocialGym2 checkout. It is deliberately a protocol smoke and
integration benchmark: it does not alter the official simulator or claim a
TRACER comparison until the TRACER controller is connected to this service
interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path


SCENARIO_FACTORIES = {
    "envs/scenario/intersection": "envs_intersection",
    "envs/scenario/hallway": "envs_hallway",
    "envs/scenario/door": "envs_door",
    "envs/scenario/round_about": "envs_round_about",
}


def _commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("SOCIALGYM_SOURCE_COMMIT", "unavailable")


def build_env(
    source: Path,
    num_agents: int,
    num_humans: int,
    partial: bool,
    launch_config: Path,
    scenario_name: str,
):
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "src"))
    from src.environment import RosSocialEnv
    from src.environment.observations import (
        AgentsGoalDistance,
        AgentsPose,
        CollisionObservation,
        OtherAgentObservables,
        SuccessObservation,
    )
    from src.environment.observations.observer import Observer
    from src.environment.rewards import ExistencePenalty, GoalDistanceChange, Rewarder
    from src.environment.scenarios.common_scenarios import (
        envs_door,
        envs_hallway,
        envs_intersection,
        envs_round_about,
    )

    scenario_factory = {
        "envs/scenario/intersection": envs_intersection,
        "envs/scenario/hallway": envs_hallway,
        "envs/scenario/door": envs_door,
        "envs/scenario/round_about": envs_round_about,
    }[scenario_name]
    scenario, _ = scenario_factory(partially_observable=partial, config_runner=True, all_config=False)

    original_make_scenario = scenario.make_scenario

    def make_runtime_scenario(config):
        result = original_make_scenario(config)
        sim_config = source / "config/gym_gen/sim_config.lua"
        text = sim_config.read_text(encoding="utf-8")
        map_path = source / "submodules/ut_multirobot_sim/maps" / scenario_name / f"{Path(scenario_name).name}.vectormap.txt"
        text = re.sub(r'(?m)^\s*human_config\s*=.*$', f'  human_config = "{source / "config/gym_gen/humans.lua"}"', text)
        text = re.sub(r'(?m)^\s*robot_config\s*=.*$', f'  robot_config = "{source / "submodules/ut_multirobot_sim/config/ut_jackal_config.lua"}"', text)
        text = re.sub(r'(?m)^\s*map_name\s*=.*$', f'  map_name = "{map_path}"', text)
        sim_config.write_text(text, encoding="utf-8")
        return result

    scenario.make_scenario = make_runtime_scenario
    observations = [
        AgentsGoalDistance(history_length=2),
        AgentsPose(ignore_theta=True),
        CollisionObservation(),
        SuccessObservation(),
        OtherAgentObservables(pos_x=True, pos_y=True, pos_theta=False),
    ]
    rewards = [ExistencePenalty(1.0), GoalDistanceChange()]
    observer = Observer(observations)
    rewarder = Rewarder(rewards)
    return RosSocialEnv(
        observer=observer,
        rewarder=rewarder,
        scenarios=[scenario],
        num_humans=num_humans,
        num_agents=num_agents,
        debug=False,
        launch_config=str(launch_config),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("third_party/socialgym2_official"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-humans", type=int, default=2)
    parser.add_argument("--partial-observation", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_FACTORIES),
        default="envs/scenario/intersection",
        help="Official SocialGym2 scenario with a non-empty vector map.",
    )
    parser.add_argument(
        "--launch-config",
        type=Path,
        default=None,
        help="Official launch file; defaults to the test full-loop configuration.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    source = args.source.resolve()
    launch_config = (args.launch_config or source / "config/gym_gen/launch.launch").resolve()
    metadata = {
        "protocol": "socialgym2-official-headless-ros-v1",
        "source": str(source),
        "source_commit": _commit(source),
        "seed": args.seed,
        "episodes": args.episodes,
        "steps_per_episode": args.steps,
        "num_agents": args.num_agents,
        "num_humans": args.num_humans,
        "partial_observation": args.partial_observation,
        "scenario": args.scenario,
        "launch_config": str(launch_config),
        "ros_master_uri": os.environ.get("ROS_MASTER_URI"),
        "claim_boundary": "official SocialGym2 RosSocialEnv reset/step integration only; no TRACER same-protocol or superiority claim",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    fields = [
        "episode",
        "step",
        "num_agents",
        "num_humans",
        "active_agents",
        "finite_observation",
        "mean_reward",
        "success_count",
        "collision_count",
        "mean_robot_x",
        "mean_robot_y",
    ]
    rows: list[dict[str, object]] = []
    env = None
    error = None
    started = time.time()
    try:
        env = build_env(
            source,
            args.num_agents,
            args.num_humans,
            args.partial_observation,
            launch_config,
            args.scenario,
        )
        generated_config = source / "config/gym_gen/sim_config.lua"
        if generated_config.is_file():
            (args.output_dir / "runtime_sim_config.lua").write_text(
                generated_config.read_text(encoding="utf-8"), encoding="utf-8"
            )
            metadata["runtime_sim_config_sha256"] = hashlib.sha256(
                generated_config.read_bytes()
            ).hexdigest()
        for episode in range(args.episodes):
            observations, _ = env.reset(seed=args.seed + episode, return_info=True)
            for step in range(args.steps):
                action = {agent: 1 for agent in env.agents}
                obs, rewards, terms, truncs, infos = env.step(action)
                positions = [info.get("position", [float("nan")]) for info in infos.values() if isinstance(info, dict) and "position" in info]
                xs = [float(pos[0]) for pos in positions if len(pos) >= 2]
                ys = [float(pos[1]) for pos in positions if len(pos) >= 2]
                finite = all(value == value and abs(value) < float("inf") for value in [*xs, *ys, *map(float, rewards.values())])
                rows.append({
                    "episode": episode,
                    "step": step,
                    "num_agents": args.num_agents,
                    "num_humans": args.num_humans,
                    "active_agents": len(obs),
                    "finite_observation": int(finite),
                    "mean_reward": sum(map(float, rewards.values())) / max(1, len(rewards)),
                    "success_count": sum(int(info.get("succeeded", False)) for info in infos.values() if isinstance(info, dict)),
                    "collision_count": sum(int(info.get("collision", False)) for info in infos.values() if isinstance(info, dict)),
                    "mean_robot_x": sum(xs) / max(1, len(xs)),
                    "mean_robot_y": sum(ys) / max(1, len(ys)),
                })
                if all(terms.get(agent, False) or truncs.get(agent, False) for agent in list(terms)):
                    break
    except Exception as exc:  # preserve the exact runtime boundary in metadata
        error = f"{type(exc).__name__}: {exc}"
        metadata["error"] = error
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                metadata["close_error"] = f"{type(exc).__name__}: {exc}"
            try:
                env.launch.shutdown()
            except Exception as exc:
                metadata["launch_shutdown_error"] = f"{type(exc).__name__}: {exc}"

    with (args.output_dir / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    completed = error is None and len(rows) == args.episodes * args.steps and all(row["finite_observation"] for row in rows)
    summary = {
        "protocol": metadata["protocol"],
        "status": "COMPLETED" if completed else "FAILED",
        "rows": len(rows),
        "expected_rows": args.episodes * args.steps,
        "finite_rows": sum(int(row["finite_observation"]) for row in rows),
        "elapsed_seconds": time.time() - started,
        "error": error,
        "claim_boundary": metadata["claim_boundary"],
        "episode_metrics_sha256": hashlib.sha256((args.output_dir / "episode_metrics.csv").read_bytes()).hexdigest(),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
