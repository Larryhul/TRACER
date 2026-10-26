from __future__ import annotations

import argparse
import configparser
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path("experiments/external_baseline_calibrated_20260908_v1")
OUTPUT = ROOT / "reports/paper_trajectories_refined_20260908"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def captured_arrays(environment, start):
    states = environment.states[start:]
    return {
        "robot": np.asarray([state[0].position for state in states] + [environment.robot.get_position()]),
        "humans": np.asarray([[human.position for human in state[1]] for state in states]
                             + [[human.get_position() for human in environment.humans]]),
        "goal": np.asarray(environment.robot.get_goal_position()),
        "obstacles": np.asarray(environment.static_obstacles).reshape(-1, 2, 2),
        "robot_radius": np.asarray(environment.robot.radius),
        "human_radii": np.asarray([human.radius for human in environment.humans]),
        "time": np.arange(len(states) + 1) * environment.time_step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(144, 152)))
    parser.add_argument("--policies", nargs="+", default=["tracer-persistent", "sicnav-np", "official-mpc-cvmm", "official-orca"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--humans", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--scenario", choices=["circle_crossing", "hallway_bottleneck"], default="circle_crossing")
    parser.add_argument("--radius", type=float, default=3.5)
    options = parser.parse_args()
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "h3_data"
    data_dir.mkdir(exist_ok=True)
    snapshot = REFERENCE / "frozen/source_snapshot"
    sys.path.insert(0, str(snapshot))
    os.chdir(ROOT)
    torch.set_num_threads(2)
    from sage.experiments import run_sicnav_tracer_loop as tracer
    from sage.experiments import run_sicnav_official_benchmark as official
    from sage.experiments.calibrate_persistent_belief import load_belief_calibration

    metadata_path = REFERENCE / "tracer/metadata.json"
    parameters = dict(json.loads(metadata_path.read_text())["parameters"])
    environment_config = configparser.RawConfigParser()
    environment_config.read(ROOT / parameters["env_config"])
    environment_config.set("sim", "test_sim", options.scenario)
    environment_config.set("sim", "human_num", str(options.humans))
    environment_config.set("sim", "circle_radius", str(options.radius))
    environment_config.set("sim", "starts_moving", str(options.warmup))
    config_path = output / f"{options.scenario}_{options.humans}h.config"
    serialized = io.StringIO()
    environment_config.write(serialized)
    config_text = serialized.getvalue()
    if config_path.exists() and config_path.read_text() != config_text:
        raise ValueError("Environment config changed; use a fresh output directory")
    config_path.write_text(config_text)
    parameters.update(env_config=str(config_path), output_dir=str(output), device=options.device)
    args = argparse.Namespace(**parameters)
    checkpoint_path = ROOT / args.checkpoint_dir / "seed2027/response.pt"
    calibration_path = ROOT / str(args.belief_calibration_path).format(seed=2027)
    model = tracer._load_model(checkpoint_path, args.device, expected_dt=args.env_dt)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    calibration = load_belief_calibration(calibration_path, checkpoint_path, execute_steps=1,
                                        dt=args.env_dt, num_robots=model.config.num_robots)
    for case in options.cases:
        tracer._validate_external_belief_cases(checkpoint, case, 1)
        tracer._validate_external_belief_cases({"provenance": {"case_splits": calibration["case_splits"]}}, case, 1)
    args.belief_update_stride = int(calibration["observation_stride"])
    args.likelihood_scale = float(calibration["selected"]["likelihood_scale"])
    args.adaptive_max_hazard = args.hazard_rate
    args.adaptive_surprise_threshold = args.change_point_threshold
    model.mode_semantics = checkpoint["mode_semantics"]
    protocol = {
        "kind": "exploratory_qualitative_domain_transfer_not_original_benchmark",
        "frozen_controller_metadata": str(metadata_path), "metadata_sha256": digest(metadata_path),
        "source_snapshot": str(snapshot), "tracer_runner_sha256": digest(tracer.__file__),
        "official_runner_sha256": digest(official.__file__),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": digest(checkpoint_path),
        "calibration_sha256": digest(calibration_path), "environment_config_sha256": digest(config_path),
        "environment": f"official {options.scenario}; {options.humans} reactive humans; radius {options.radius}m; {options.warmup} warmup steps",
        "changed_environment_fields": {"test_sim": options.scenario, "human_num": options.humans,
                                       "circle_radius": options.radius, "starts_moving": options.warmup},
        "controller_parameters": parameters, "model_seed": 2027,
        "screening_cases": options.cases,
        "screening_policies": ["tracer-persistent", "sicnav-np", "official-mpc-cvmm", "official-orca"],
        "budget": "All methods: 60 steps, dt=0.25s, 15s; native official kinematics.",
        "selection_rule": "Retain the complete eight-case screen. For an explicitly selected advantage example, require collision-free TRACER success; rank first by nonprivileged SICNav safety advantage, then other baseline safety failures, then TRACER-minus-SICNav travel-time advantage. Do not invent an advantage if none qualifies.",
        "boundary": "No retraining, controller retuning, blind-test claim or overall-superiority claim. ORCA is holonomic; other displayed methods are unicycle. New open-space environment is a domain-transfer illustration of a hallway-calibrated controller.",
    }
    protocol_path = output / "h3_screening_protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text())
        keys = ["metadata_sha256", "tracer_runner_sha256", "official_runner_sha256", "checkpoint_sha256",
                "calibration_sha256", "environment_config_sha256", "controller_parameters", "model_seed",
                "screening_cases", "screening_policies", "budget", "selection_rule"]
        if any(existing.get(key) != protocol.get(key) for key in keys):
            raise ValueError("Protocol differs; use a fresh output directory")
    else:
        write_json(protocol_path, protocol)
    adapter = tracer.SAGECandidateAdapter.from_checkpoint(
        ROOT / args.sage_checkpoint, args.device,
        tracer.SAGECandidateAdapterConfig(
            num_candidates=args.num_candidates, guided=True, step_stride=args.sage_step_stride,
            goal_speed=args.sage_goal_speed, oversample_factor=args.sage_oversample_factor,
            task_prefilter=True, task_prefix_steps=args.execute_steps, goal_direction="task_goal",
            include_task_anchor=True, task_anchor_speed_scales=(0.45,), batch_size=args.num_candidates,
            guidance_step_clip_norm=0.1, diversity_weight=args.sage_diversity_weight,
        ),
    )
    for case in options.cases:
        for policy in options.policies:
            stem = data_dir / f"case{case}_{policy}"
            if stem.with_suffix(".json").exists():
                continue
            print(f"CASE {case} POLICY {policy}", flush=True)
            torch.manual_seed(2027)
            np.random.seed(2027)
            random.seed(2027)
            module = tracer if policy == "tracer-persistent" else official
            original = module._new_episode
            captures = []

            def capture(*episode_args, **episode_kwargs):
                result = original(*episode_args, **episode_kwargs)
                captures.append((result[0], len(result[0].states)))
                return result

            module._new_episode = capture
            try:
                if policy == "tracer-persistent":
                    windows = []
                    metrics = tracer._run_episode(model, adapter, policy, ROOT / args.external_root,
                                                  case, config_path, args, 2027, [], windows)
                else:
                    windows = []
                    metrics = official._run_episode(ROOT / args.external_root, policy, case,
                                                    config_path, ROOT / args.motion_policy_config, args.max_steps)
                arrays = captured_arrays(*captures[0])
                clearance = np.linalg.norm(arrays["humans"] - arrays["robot"][:, None], axis=-1)
                clearance -= arrays["robot_radius"] + arrays["human_radii"][None]
                np.savez_compressed(stem.with_suffix(".npz"), **arrays)
                write_json(stem.with_suffix(".json"), {"metrics": metrics, "windows": windows,
                                                       "minimum_sampled_surface_gap_m": float(clearance.min())})
                print(f"DONE {case} {policy} success={metrics['success']} collision={metrics['collision']} time={metrics['total_time']}", flush=True)
            except Exception as error:
                write_json(stem.with_suffix(".error.json"), {"error": str(error), "traceback": traceback.format_exc()})
                raise
            finally:
                module._new_episode = original


if __name__ == "__main__":
    main()
