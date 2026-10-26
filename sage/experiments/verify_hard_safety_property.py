"""Executable finite-candidate hard-safety property audit.

The audit verifies the exact guarantee implemented by TRACER's local static
filter.  It deliberately does not claim dynamic collision avoidance: the
property is only about the supplied candidate positions, static circular
obstacles, and pairwise robot clearance at the checked horizon samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from sage.experiments.run_independent_multientity_suite import _build_spec
from sage.experiments.run_tracer_gap_closure import _obstacles, _static_safe_mask
from sage.response.independent_social_force import MultiEntityIndependentSocialForceSimulator


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "dirty": bool(run("status", "--porcelain"))}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    started = time.perf_counter()
    git = _git_state(root)
    if git["dirty"]:
        raise RuntimeError("hard-safety property audit requires a clean Git worktree")
    checked_pools = 0
    checked_candidates = 0
    selected_without_fallback = 0
    fallback_pools = 0
    unsafe_selected_without_fallback = 0
    safe_geometry_mismatches = 0
    all_rejected_cases = 0
    records: list[dict[str, Any]] = []
    for seed in args.seeds:
        for episode in range(args.episodes):
            spec = _build_spec(args, "ood", args.num_robots)
            simulator = MultiEntityIndependentSocialForceSimulator(
                spec, episode + seed * 100, args.num_entities, args.active_entities
            )
            rng = np.random.default_rng(args.random_seed + seed * 1009 + episode * 9176)
            for pool_index in range(args.pools_per_episode):
                pool = simulator.candidate_pool()
                obstacles = _obstacles(simulator, args.random_seed + episode + pool_index)
                if pool_index % 5 == 0:
                    # Force the explicit all-rejected branch so it is tested,
                    # rather than relying on the sampled pool to trigger it.
                    center = simulator.robot_positions[0] + simulator.entity_positions[0]
                    obstacles = np.asarray([[center[0], center[1], 10.0]], dtype=np.float32)
                hard_safe, obstacle_violation, robot_violation = _static_safe_mask(
                    simulator, pool, obstacles, args.hard_safety_margin
                )
                candidate = pool.robot_candidates.numpy()[:, 0, :, :, :2]
                world = candidate + simulator.entity_positions[0][None, None, None, :]
                direct_obstacle = np.zeros(len(candidate), dtype=bool)
                for center_x, center_y, radius in obstacles:
                    direct_obstacle |= (
                        np.linalg.norm(world - np.asarray([center_x, center_y]), axis=-1)
                        < float(radius) + args.hard_safety_margin
                    ).any(axis=(1, 2))
                direct_robot = np.zeros(len(candidate), dtype=bool)
                for first in range(args.num_robots):
                    for second in range(first + 1, args.num_robots):
                        direct_robot |= (
                            np.linalg.norm(world[:, first] - world[:, second], axis=-1)
                            < args.hard_safety_margin
                        ).any(axis=1)
                safe_geometry_mismatches += int(
                    not np.array_equal(obstacle_violation, direct_obstacle)
                    or not np.array_equal(robot_violation, direct_robot)
                    or not np.array_equal(hard_safe, ~(direct_obstacle | direct_robot))
                )
                scores = rng.normal(size=len(candidate))
                scores[~hard_safe] = np.inf
                fallback = not np.isfinite(scores).any()
                if fallback:
                    selected = int(np.argmin(obstacle_violation.astype(float)))
                    fallback_pools += 1
                    all_rejected_cases += int(not hard_safe.any())
                else:
                    selected = int(np.argmin(scores))
                    selected_without_fallback += 1
                selected_is_unsafe = bool(not hard_safe[selected])
                unsafe_selected_without_fallback += int(selected_is_unsafe and not fallback)
                checked_pools += 1
                checked_candidates += len(candidate)
                records.append({
                    "seed": int(seed),
                    "episode": int(episode),
                    "pool": int(pool_index),
                    "safe_candidates": int(hard_safe.sum()),
                    "candidate_count": int(len(candidate)),
                    "fallback": bool(fallback),
                    "selected": selected,
                    "selected_is_unsafe": selected_is_unsafe,
                    "obstacle_violations": int(obstacle_violation.sum()),
                    "robot_violations": int(robot_violation.sum()),
                })
    summary = {
        "checked_pools": checked_pools,
        "checked_candidates": checked_candidates,
        "selected_without_fallback": selected_without_fallback,
        "fallback_pools": fallback_pools,
        "all_rejected_cases": all_rejected_cases,
        "unsafe_selected_without_fallback": unsafe_selected_without_fallback,
        "safe_geometry_mismatches": safe_geometry_mismatches,
        "property_holds": bool(
            unsafe_selected_without_fallback == 0 and safe_geometry_mismatches == 0
        ),
    }
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "protocol": "tracer-finite-candidate-hard-safety-property-v1",
        "claim_boundary": "finite candidate positions checked against supplied static circular obstacles and pairwise robot clearance; no dynamic or arbitrary-control safety guarantee",
        "git": git,
        "parameters": vars(args),
        "summary": summary,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = (
        "# Finite-candidate hard-safety property audit\n\n"
        f"Commit `{git['commit']}`; clean `{not git['dirty']}`.\n\n"
        f"Checked `{checked_pools}` pools and `{checked_candidates}` candidate trajectories. "
        f"The exact finite-candidate property holds: `{summary['property_holds']}`. "
        f"Unsafe selections without fallback: `{unsafe_selected_without_fallback}`; "
        f"geometry mismatches: `{safe_geometry_mismatches}`; explicit all-rejected pools: `{all_rejected_cases}`.\n\n"
        "This is a static finite-candidate audit. It is not a formal guarantee for dynamic humans, unenumerated controls, or the official simulator.\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "status": "COMPLETED" if summary["property_holds"] else "FAILED",
        "protocol": metadata["protocol"],
        "git": git,
        "claim_boundary": metadata["claim_boundary"],
        "artifacts": {name: _sha256(output / name) for name in ("records.json", "metadata.json", "summary.json", "report.md")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not summary["property_holds"]:
        raise RuntimeError(f"hard-safety property failed: {summary}")
    return {"metadata": metadata, "summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/tracer_hard_safety_property_20260903")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2031, 2039, 2053, 2063])
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--pools-per-episode", type=int, default=16)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=959731)
    parser.add_argument("--data-seed", type=int, default=969731)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--num-entities", type=int, default=6)
    parser.add_argument("--active-entities", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.3)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--response-noise", type=float, default=0.08)
    parser.add_argument("--entity-max-speed", type=float, default=2.2)
    parser.add_argument("--scene-family", default="crossing")
    parser.add_argument("--hard-safety-margin", type=float, default=0.08)
    args = parser.parse_args()
    result = _run(args)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
