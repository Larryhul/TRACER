"""Verify full-horizon joint interventions from saved simulator trajectories."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def verify_physics_audits(
    paths: list[Path],
    runtime_contract: dict,
    scenarios: list[str],
    *,
    minimum_ticks: int,
    minimum_episodes: int = 4,
) -> list[dict]:
    seen = set()
    verified = []
    lattice = set(itertools.product((0, 1), repeat=3))
    for path in paths:
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("protocol") != "socialgym2-joint-physical-intervention-audit-v4":
            raise ValueError("physical audit does not cover the complete joint lattice")
        if audit.get("status") != "PASSED" or audit.get("runtime_contract") != runtime_contract:
            raise ValueError("physical audit failed or belongs to a different runtime")
        if audit.get("repeat_tolerance_meters", float("inf")) > 1e-4:
            raise ValueError("physical audit relaxes the strict repeat tolerance")
        scenario = audit["scenario"]
        if scenario in seen or scenario not in scenarios:
            raise ValueError("physical audit scenario is duplicate or unexpected")
        seen.add(scenario)
        ticks = audit["ticks"]
        records = audit["records"]
        if ticks < minimum_ticks or len(records) < minimum_episodes:
            raise ValueError("physical audit is shorter than the declared evaluation")
        if len({record["seed"] for record in records}) != len(records):
            raise ValueError("physical audit repeats an episode seed")
        maximum_repeat = 0.0
        maximum_robot_repeat = 0.0
        maximum_effect = np.zeros(3)
        for record in records:
            branches = record["branches"]
            if len(branches) != 9 or {tuple(branch["action"]) for branch in branches[:-1]} != lattice:
                raise ValueError("physical audit omits a joint intervention")
            if tuple(branches[0]["action"]) != (0, 0, 0) or branches[0]["action"] != branches[-1]["action"]:
                raise ValueError("physical audit does not repeat its reference action")
            if len({branch["initial_hash"] for branch in branches}) != 1:
                raise ValueError("physical intervention branches have different initial states")
            for branch in branches:
                for field, entities in (("human_trajectory", 2), ("robot_trajectory", 3)):
                    trajectory = np.asarray(branch[field], dtype=np.float64)
                    if trajectory.shape != (ticks, entities, 2) or not np.isfinite(trajectory).all():
                        raise ValueError("physical audit trajectory is incomplete or non-finite")
            reference = np.asarray(branches[0]["human_trajectory"])
            repeated = np.asarray(branches[-1]["human_trajectory"])

            def human_difference(trajectory):
                distances = [
                    np.linalg.norm(reference - trajectory[:, order, :], axis=-1).mean(axis=-1)
                    for order in ((0, 1), (1, 0))
                ]
                return float(np.minimum(*distances).max())

            repeat_error = human_difference(repeated)
            robot_error = float(np.linalg.norm(
                np.asarray(branches[0]["robot_trajectory"])
                - np.asarray(branches[-1]["robot_trajectory"]), axis=-1,
            ).max())
            if max(repeat_error, robot_error) > 1e-4:
                raise ValueError("saved trajectories fail strict physical repeatability")
            maximum_repeat = max(maximum_repeat, repeat_error)
            maximum_robot_repeat = max(maximum_robot_repeat, robot_error)
            lookup = {tuple(branch["action"]): branch for branch in branches[:-1]}
            for robot_index in range(3):
                intervention = tuple(int(index == robot_index) for index in range(3))
                effect = human_difference(np.asarray(lookup[intervention]["human_trajectory"]))
                maximum_effect[robot_index] = max(maximum_effect[robot_index], effect)
        if not np.all(maximum_effect > max(1e-4, 10 * maximum_repeat)):
            raise ValueError("not every controllable robot has a measured human-response effect")
        verified.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "scenario": scenario, "ticks": ticks, "episodes": len(records),
                         "max_repeat_human_distance": maximum_repeat,
                         "max_repeat_robot_distance": maximum_robot_repeat,
                         "max_robot_intervention_effect": maximum_effect.tolist()})
    if seen != set(scenarios):
        raise ValueError("physical audits do not cover every evaluated scenario")
    return verified
