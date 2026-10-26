"""Audit measured multi-robot intervention effects and synchronous reset."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np


from sage.experiments.run_socialgym2_tracer_loop import (
    _build_env,
    _capture_observations,
    _episode_seed,
    _initial_state_hash,
    _runtime_manifest,
    _scene_from_responses,
    _seed_episode,
)


REPEAT_TOLERANCE_METERS = 1e-4
MIN_INTERVENTION_TO_REPEAT_RATIO = 10.0


def _human_state(responses) -> np.ndarray:
    robots, _, _, observations = _scene_from_responses(responses, partial_observation=False)
    if len(robots) != 3 or len(observations) != 2:
        raise RuntimeError("joint physics audit requires all three robots and both humans")
    return np.asarray(sorted([position.tolist() for position, _ in observations]), dtype=np.float64)


def _human_distance(first: np.ndarray, second: np.ndarray) -> float:
    return min(float(np.linalg.norm(first - second[list(order)], axis=-1).mean())
               for order in itertools.permutations(range(len(second))))


def audit(
    source: Path,
    scenario: str,
    seed: int,
    episodes: int,
    ticks: int,
    output: Path,
    runtime_manifest: Path | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime audit: {output}")
    runtime_contract = _runtime_manifest(runtime_manifest, source)
    launch_config = source / "config/gym_gen/launch.launch"
    env = _build_env(source, 3, 2, False, launch_config, scenario)
    capture = _capture_observations(env)
    records = []
    action_vectors = list(itertools.product((0, 1), repeat=3)) + [(0, 0, 0)]
    try:
        for episode in range(episodes):
            branches = []
            for branch_index, action in enumerate(action_vectors):
                episode_seed = _seed_episode(_episode_seed(seed, episode))
                env.new_scenario(num_humans=2, num_agents=3)
                env.reset(seed=episode_seed)
                responses = capture.pop("responses")
                initial_hash = _initial_state_hash(*_scene_from_responses(responses, partial_observation=False))
                if env.h3_simulation_time() != 0.0:
                    raise RuntimeError("runtime audit reset advances time")
                humans = []
                robots = []
                for tick in range(ticks):
                    if branch_index == len(action_vectors) - 1:
                        time.sleep(0.005)
                    env.step(dict(zip(env.agents, action)))
                    responses = capture.pop("responses")
                    humans.append(_human_state(responses))
                    robots.append(_scene_from_responses(responses, partial_observation=False)[0].tolist())
                    if not np.isclose(env.h3_simulation_time(), (tick + 1) * 0.025, atol=2e-5):
                        raise RuntimeError("runtime audit clock mismatch")
                branches.append({"action": action, "initial_hash": initial_hash,
                                 "human_trajectory": np.asarray(humans).tolist(), "robot_trajectory": robots})
            if len({branch["initial_hash"] for branch in branches}) != 1:
                raise RuntimeError("intervention branches have different initial states")
            reference = np.asarray(branches[0]["human_trajectory"])
            repeat = np.asarray(branches[-1]["human_trajectory"])
            repeat_error = max(_human_distance(first, second) for first, second in zip(reference, repeat))
            robot_repeat_error = float(np.linalg.norm(
                np.asarray(branches[0]["robot_trajectory"])
                - np.asarray(branches[-1]["robot_trajectory"]), axis=-1,
            ).max())
            effects = []
            branch_by_action = {tuple(branch["action"]): branch for branch in branches[:-1]}
            for robot_index in range(3):
                intervention = tuple(int(index == robot_index) for index in range(3))
                branch = branch_by_action[intervention]
                changed = np.asarray(branch["human_trajectory"])
                effects.append(max(_human_distance(first, second) for first, second in zip(reference, changed)))
            records.append({"episode": episode, "seed": episode_seed, "repeat_max_human_distance": repeat_error,
                            "repeat_max_robot_distance": robot_repeat_error,
                            "robot_intervention_max_human_distance": effects, "branches": branches})
            print(f"joint runtime scenario={scenario} episode={episode} effects={effects} repeat={repeat_error}", flush=True)
    finally:
        env.close()
        env.launch.shutdown()
    max_repeat = max(record["repeat_max_human_distance"] for record in records)
    max_robot_repeat = max(record["repeat_max_robot_distance"] for record in records)
    if _runtime_manifest(runtime_manifest, source) != runtime_contract:
        raise RuntimeError("external runtime changed during physical audit")
    max_effects = np.max([record["robot_intervention_max_human_distance"] for record in records], axis=0)
    result = {"protocol": "socialgym2-joint-physical-intervention-audit-v4", "scenario": scenario,
              "seed": seed, "episodes": episodes, "ticks": ticks, "partial_observation": False,
              "max_repeat_human_distance": max_repeat, "max_robot_intervention_effect": max_effects.tolist(),
              "max_repeat_robot_distance": max_robot_repeat,
              "repeat_branch_wall_delay_seconds_per_tick": 0.005,
              "runtime_contract": runtime_contract,
              "repeat_tolerance_meters": REPEAT_TOLERANCE_METERS,
              "minimum_intervention_to_repeat_ratio": MIN_INTERVENTION_TO_REPEAT_RATIO,
              "every_robot_has_observed_human_effect": bool(
                  np.all(max_effects > max(1e-4, MIN_INTERVENTION_TO_REPEAT_RATIO * max_repeat))
              ),
              "records": records}
    result["status"] = "PASSED" if (
        max_repeat <= REPEAT_TOLERANCE_METERS
        and max_robot_repeat <= REPEAT_TOLERANCE_METERS
        and result["every_robot_has_observed_human_effect"]
    ) else "FAILED"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=2221)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=160)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(
        args.source.resolve(), args.scenario, args.seed, args.episodes, args.ticks,
        args.output, runtime_manifest=args.runtime_manifest,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    if result["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
