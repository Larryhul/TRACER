"""Run an isolated, frozen, paired navigation-objective comparison."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Event, Lock
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_final_socialgym2_confirmation import (
    _clone_workers,
    _port_open,
    _ros_overlay,
    _sha256,
    _snapshot_source,
    _stop_ros_masters,
    _verify_snapshot,
    _worker_environment,
    _write_json,
)


def verify_run(directory: Path, episodes: int) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "COMPLETED" or manifest["records"] != episodes:
        raise ValueError(f"incomplete navigation run: {directory}")
    for name, expected in manifest["artifacts"].items():
        if _sha256(directory / name) != expected:
            raise ValueError(f"artifact hash mismatch: {directory / name}")
    diagnosis = json.loads(directory.with_suffix(".diagnosis.json").read_text(encoding="utf-8"))
    if diagnosis["status"] != "COMPLETED" or len(diagnosis["episodes"]) != episodes:
        raise ValueError(f"incomplete execution audit: {directory}")
    if diagnosis["stream_sha256"] != _sha256(directory / "execution_diagnostics.jsonl"):
        raise ValueError(f"execution audit stream mismatch: {directory}")
    return diagnosis


def paired_metrics(baseline: list[dict], candidate: list[dict], episodes: int) -> dict:
    def index(rows):
        indexed = {}
        for row in rows:
            metrics = row["metrics"]
            key = (metrics["seed"], metrics["scenario"], metrics["episode"])
            if key in indexed:
                raise ValueError(f"duplicate episode: {key}")
            indexed[key] = row
        return indexed

    first, second = index(baseline), index(candidate)
    if not first or first.keys() != second.keys() or len(first) != episodes:
        raise ValueError("comparison requires a complete, nonempty paired episode set")
    for key in first:
        for field in ("initial_state_hash", "initial_scene_snapshot", "scenario_hash", "initial_collision"):
            if first[key]["metrics"][field] != second[key]["metrics"][field]:
                raise ValueError(f"unmatched {field}: {key}")
    keys = sorted(first)
    clean_keys = [key for key in keys if not first[key]["metrics"]["initial_collision"]]
    result = {"paired_scenes": len(keys), "initially_clean_scenes": len(clean_keys), "metrics": {}}
    rng = np.random.default_rng(20260909)
    fields = (
        "team_success", "collision_free_team_success", "collision", "task_progress",
        "stagnation_window_rate", "robot_success_fraction", "belief_action_change_rate",
        "fallback_rate",
    )
    for field in (*fields, "budget_exhausted_before_team_success", "robot_halt_fraction"):
        values = lambda rows: np.asarray([
            rows[key]["metrics"][field] if field in fields else rows[key][field] for key in keys
        ], dtype=float)
        baseline_values, candidate_values = values(first), values(second)
        differences = candidate_values - baseline_values
        if not np.isfinite(differences).all():
            raise ValueError(f"nonfinite paired metric: {field}")
        draws = rng.choice(differences, size=(20000, len(keys)), replace=True).mean(axis=1)
        result["metrics"][field] = {
            "baseline": float(baseline_values.mean()),
            "candidate": float(candidate_values.mean()),
            "delta": float(differences.mean()),
            "paired_scene_bootstrap95": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    result["new_collision_scenes"] = [list(key) for key in keys if (
        not first[key]["metrics"]["collision"] and second[key]["metrics"]["collision"]
    )]
    result["removed_collision_scenes"] = [list(key) for key in keys if (
        first[key]["metrics"]["collision"] and not second[key]["metrics"]["collision"]
    )]
    result["initially_clean_collision_counts"] = {
        "baseline": sum(int(first[key]["metrics"]["collision"]) for key in clean_keys),
        "candidate": sum(int(second[key]["metrics"]["collision"]) for key in clean_keys),
    }
    result["boundary"] = "Scene-paired descriptive screening; bootstrap intervals are not a safety or noninferiority guarantee."
    return result


def adoption_decision(results: dict) -> dict:
    reasons = []
    if set(results) != {"open", "intersection"}:
        reasons.append("both maps are required before adoption")
    gains = 0.0
    for scenario, result in results.items():
        metrics = result["metrics"]
        if result["new_collision_scenes"]:
            reasons.append(f"{scenario}: new collision scenes")
        if metrics["collision_free_team_success"]["delta"] < -1e-12:
            reasons.append(f"{scenario}: collision-free team-success regression")
        if metrics["task_progress"]["delta"] < -0.01:
            reasons.append(f"{scenario}: task-progress regression exceeds 0.01")
        gains += metrics["collision_free_team_success"]["delta"] * result["paired_scenes"]
    if gains <= 1e-12:
        reasons.append("no positive pooled collision-free team-success gain")
    return {
        "adopt": not reasons, "reasons": reasons,
        "net_additional_collision_free_team_successes": round(gains),
        "boundary": "Operational selection rule, not a test of statistical significance or guaranteed improvement.",
    }


def validated_profile(path: Path) -> dict:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema") != "socialgym2-selected-navigation-profile-v1":
        raise ValueError("unsupported navigation profile")
    snapshot = Path(profile["frozen_source"])
    manifest_path = snapshot / "source_manifest.json"
    if _sha256(manifest_path) != profile["source_manifest_sha256"]:
        raise ValueError("selected profile source manifest changed")
    _verify_snapshot(snapshot, json.loads(manifest_path.read_text(encoding="utf-8")))
    comparison_path = Path(profile["comparison"])
    if _sha256(comparison_path) != profile["comparison_sha256"]:
        raise ValueError("selected profile comparison changed")
    if not adoption_decision(json.loads(comparison_path.read_text(encoding="utf-8")))["adopt"]:
        raise ValueError("selected profile does not satisfy adoption rules")
    protocol = json.loads(comparison_path.with_name("protocol.json").read_text(encoding="utf-8"))
    if (
        protocol["source_manifest_sha256"] != profile["source_manifest_sha256"]
        or protocol["input_hashes"] != profile["input_hashes"]
        or profile["candidate"] not in protocol["conditions"]
    ):
        raise ValueError("selected profile does not match the evaluated protocol")
    return profile


def start_masters(root: Path, output: Path, workers: int, port_base: int):
    overlay = _ros_overlay(root)
    processes = []
    try:
        for worker in range(workers):
            port = port_base + worker
            if _port_open(port):
                raise RuntimeError(f"dedicated ROS port already occupied: {port}")
            state = output / "ros" / f"master{worker}"
            state.mkdir(parents=True)
            log = (state / "master.log").open("x")
            environment = {
                **os.environ,
                "PYTHONPATH": str(overlay / "opt/ros/noetic/lib/python3/dist-packages"),
                "ROS_HOME": str(state), "ROS_LOG_DIR": str(state),
                "ROS_MASTER_URI": f"http://127.0.0.1:{port}", "ROS_IP": "127.0.0.1",
            }
            process = subprocess.Popen([
                str(root / ".conda/envs/hulan/bin/python"),
                str(overlay / "opt/ros/noetic/bin/rosmaster"), "--core", "-p", str(port),
            ], env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((process, log))
        deadline = time.monotonic() + 30
        while not all(_port_open(port_base + worker) for worker in range(workers)):
            if any(process.poll() is not None for process, _ in processes):
                raise RuntimeError("isolated ROS master failed to start")
            if time.monotonic() >= deadline:
                raise TimeoutError("isolated ROS masters unavailable")
            time.sleep(0.25)
        return processes
    except BaseException:
        _stop_ros_masters(processes)
        raise


def run(args):
    root, output = args.root.resolve(), args.output.resolve()
    profile = validated_profile(args.profile) if args.profile else None
    if profile:
        if args.candidate is not None and args.candidate != profile["candidate"]:
            raise ValueError("candidate cannot override a verified navigation profile")
        args.candidate = profile["candidate"]
        if args.frozen_source and args.frozen_source.resolve() != Path(profile["frozen_source"]).resolve():
            raise ValueError("source cannot override a verified navigation profile")
        args.frozen_source = Path(profile["frozen_source"])
    args.candidate = args.candidate or "lower_tail25"
    if args.episodes < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("episodes must be positive and seeds unique/nonempty")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    origin = args.frozen_source.resolve() if args.frozen_source else root
    if args.frozen_source:
        _verify_snapshot(origin, json.loads((origin / "source_manifest.json").read_text(encoding="utf-8")))
    source_manifest = _snapshot_source(origin, snapshot)
    workers = root / "experiments" / f".{output.name}_workers"
    _clone_workers(root / "experiments/socialgym2_joint_workers_20260906", workers, len(args.seeds))
    python = root / ".conda/envs/hulan/bin/python"
    inputs = {
        "response": root / "experiments/formal_v2/h2/persistent_trajectory_suite_tracer_effect/seed2027/response.pt",
        "sage": root / "experiments/pilot/h3_sage_generator_p0_seed2027/synthetic_sage_epoch20.pt",
        "runtime": root / "experiments/socialgym2_joint_runtime_manifest_20260906_v7/manifest.json",
        "calibration": root / "experiments/socialgym2_joint_frozen_campaign_20260906_v1/validation_calibration.json",
    }
    input_hashes = {name: _sha256(path) for name, path in inputs.items()}
    if profile and input_hashes != profile["input_hashes"]:
        raise ValueError("selected profile requires the evaluated model, calibration and runtime inputs")
    shutil.copy2(__file__, output / "runner.py")
    shutil.copy2(Path(__file__).with_name("run_final_socialgym2_confirmation.py"), output / "runner_helpers.py")
    protocol = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds, "scenarios": args.scenarios, "episodes_per_seed_map": args.episodes,
        "conditions": {"mean": {"aggregation": "mean", "blend": 0.5},
                       args.candidate: {
                           "aggregation": "lower-tail" if args.candidate.startswith("lower_tail25") else "mean",
                           "blend": 0.25 if args.candidate.startswith("lower_tail25") else 0.5,
                           "completed_robot_halt": args.candidate == "completed_halt",
                           "stagnation_release_hard_safe": args.candidate.endswith("release"),
                           "response_cost_steps": 2 if args.candidate == "short_response" else None,
                       }},
        "source_manifest_sha256": _sha256(snapshot / "source_manifest.json"),
        "selected_profile": str(args.profile.resolve()) if args.profile else None,
        "source_origin": str(origin),
        "isolated_workers": str(workers),
        "input_hashes": input_hashes, "inputs": {key: str(value) for key, value in inputs.items()},
        "candidate_prefix": "executable", "task_steps": 2, "max_steps": 60,
        "policy": "tracer-persistent", "pairing": "same scene, seed, worker and frozen code; balanced condition order",
        "selection_rule": "Do not adopt if either map has extra collision scenes, worse collision-free team success, or progress loss exceeding 0.01; require positive pooled collision-free team-success gain.",
        "boundary": "Two-map navigation-objective confirmation, not component attribution or external-baseline comparison. Prior source-drift attempts excluded in full.",
    }
    _write_json(output / "protocol.json", protocol)
    masters = []
    try:
        masters = start_masters(root, output, len(args.seeds), args.port_base)
        _write_json(output / "status.json", {"status": "RUNNING"})
        failure = Event()
        status_lock = Lock()
        completed_jobs = []

        def worker(number):
            seed = args.seeds[number]
            source = workers / f"worker{number}/social_gym"
            environment = _worker_environment(
                root, snapshot, source, number, args.port_base,
                Path("/tmp/h3_ros_overlay") / output.name,
            )
            conditions = list(protocol["conditions"])
            if number % 2:
                conditions.reverse()
            for scenario in args.scenarios:
                for condition in conditions:
                    if failure.is_set():
                        return
                    _verify_snapshot(snapshot, source_manifest)
                    target = output / f"{scenario}_{condition}_seed{seed}"
                    settings = protocol["conditions"][condition]
                    command = [
                        str(python), "-u", "-m", "tracer.experiments.run_socialgym2_tracer_loop",
                        "--source", str(source), "--output-dir", str(target),
                        "--response-checkpoint", str(inputs["response"]), "--sage-checkpoint", str(inputs["sage"]),
                        "--runtime-manifest", str(inputs["runtime"]), "--calibration-artifact", str(inputs["calibration"]),
                        "--likelihood-scale", "0.05", "--scenario", f"envs/scenario/{scenario}",
                        "--seed", str(seed), "--episodes", str(args.episodes), "--max-steps", "60",
                        "--num-agents", "3", "--num-humans", "2", "--partial-observation",
                        "--belief-selection-audit", "--policies", "tracer-persistent",
                        "--execution-diagnostics", "--joint-geometry-filter", "--candidate-prefix-model", "executable",
                        "--task-steps", "2", "--task-aggregation", settings["aggregation"],
                        "--task-min-blend", str(settings["blend"]), "--device", "cuda",
                    ]
                    if settings.get("completed_robot_halt", False):
                        command.append("--completed-robot-halt")
                    if settings.get("stagnation_release_hard_safe", False):
                        command.append("--stagnation-release-hard-safe")
                    if settings.get("response_cost_steps") is not None:
                        command.extend(["--response-cost-steps", str(settings["response_cost_steps"])])
                    print(f"START {target.name}", flush=True)
                    with target.with_suffix(".log").open("x") as log:
                        subprocess.run(command, cwd=snapshot, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                        subprocess.run([
                            str(python), "-m", "tracer.experiments.analyze_socialgym2_execution",
                            "--input", str(target), "--output", str(target.with_suffix(".diagnosis.json")),
                        ], cwd=snapshot, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                    verify_run(target, args.episodes)
                    with status_lock:
                        completed_jobs.append(target.name)
                        _write_json(output / "status.json", {
                            "status": "RUNNING", "completed_jobs": completed_jobs,
                            "expected_jobs": len(args.scenarios) * len(args.seeds) * len(conditions),
                        })
                    print(f"VERIFIED {target.name}", flush=True)

        def guarded_worker(number):
            try:
                worker(number)
            except BaseException:
                failure.set()
                raise

        with ThreadPoolExecutor(max_workers=len(args.seeds)) as executor:
            list(executor.map(guarded_worker, range(len(args.seeds))))
        _verify_snapshot(snapshot, source_manifest)
        if {name: _sha256(path) for name, path in inputs.items()} != input_hashes:
            raise RuntimeError("model, calibration or runtime inputs changed during comparison")
        results = {}
        for scenario in args.scenarios:
            collected = {condition: [] for condition in protocol["conditions"]}
            for seed in args.seeds:
                metadata = []
                for condition in collected:
                    target = output / f"{scenario}_{condition}_seed{seed}"
                    collected[condition].extend(verify_run(target, args.episodes)["episodes"])
                    metadata.append(json.loads((target / "metadata.json").read_text(encoding="utf-8")))
                for field in ("algorithm_sources", "checkpoint_hashes", "runtime_contract"):
                    if metadata[0][field] != metadata[1][field]:
                        raise ValueError(f"unmatched {field}: {scenario}, {seed}")
            results[scenario] = paired_metrics(collected["mean"], collected[args.candidate], len(args.seeds) * args.episodes)
        _write_json(output / "comparison.json", results)
        _write_json(output / "decision.json", adoption_decision(results))
        _write_json(output / "status.json", {"status": "COMPLETED", "verified_paired_scenes": len(args.seeds) * args.episodes * len(args.scenarios)})
        print(json.dumps(results, indent=2), flush=True)
    except BaseException as error:
        _write_json(output / "status.json", {"status": "FAILED", "error": repr(error)})
        raise
    finally:
        _stop_ros_masters(masters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2861, 2873, 2887, 2897])
    parser.add_argument("--scenarios", nargs="+", choices=["open", "intersection"], default=["open", "intersection"])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--port-base", type=int, default=11341)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--frozen-source", type=Path)
    parser.add_argument(
        "--candidate",
        choices=["lower_tail25", "completed_halt", "conditional_release", "lower_tail25_release", "short_response"],
        default=None,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
