from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "experiments/h3_external_same_protocol_20260906"
DEFAULT_OUTPUT = ROOT / "reports/paper_trajectories_20260908"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    def finite_values(value):
        if isinstance(value, dict):
            return {key: finite_values(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_values(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    Path(path).write_text(json.dumps(finite_values(data), indent=2, allow_nan=False) + "\n")


def capture_environment(module, captures):
    original = module._new_episode

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        environment = result[0]
        captures.append((environment, len(environment.states)))
        return result

    module._new_episode = wrapped
    return original


def environment_arrays(environment, start):
    states = environment.states[start:]
    robot = np.asarray([state[0].position for state in states] + [environment.robot.get_position()])
    humans = np.asarray([[human.position for human in state[1]] for state in states]
                        + [[human.get_position() for human in environment.humans]])
    return {
        "robot": robot,
        "humans": humans,
        "goal": np.asarray(environment.robot.get_goal_position()),
        "obstacles": np.asarray(environment.static_obstacles),
        "robot_radius": np.asarray(environment.robot.radius),
        "human_radii": np.asarray([human.radius for human in environment.humans]),
        "time": np.arange(len(robot)) * float(environment.time_step),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", type=int, nargs="+", default=[88, 96])
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", choices=["tracer", "official", "all"], default="all")
    options = parser.parse_args()
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data"
    data_dir.mkdir(exist_ok=True)
    snapshot = ARCHIVE / "tracer_three_mode_belief_v2/source_snapshot"
    sys.path.insert(0, str(snapshot))
    os.chdir(ROOT)
    torch.set_num_threads(2)
    from sage.experiments import run_sicnav_tracer_loop as tracer
    from sage.experiments import run_sicnav_official_benchmark as official

    metadata_path = ARCHIVE / "tracer_three_mode_belief_v2/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    parameters = dict(metadata["parameters"])
    parameters.update(device=options.device, output_dir=str(output), belief_selection_audit=True)
    args = argparse.Namespace(**parameters)
    checkpoint_path = ROOT / args.checkpoint_dir / f"seed{options.seed}/response.pt"
    external_root = ROOT / args.external_root
    environment_config = ROOT / args.env_config
    policy_config = ROOT / args.motion_policy_config
    provenance = {
        "status": "qualitative_development_replay_not_new_aggregate_results",
        "reference_metadata": str(metadata_path),
        "reference_metadata_sha256": digest(metadata_path),
        "source_snapshot": str(snapshot),
        "runner_sha256": digest(tracer.__file__),
        "official_runner_sha256": digest(official.__file__),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest(checkpoint_path),
        "environment_config_sha256": digest(environment_config),
        "policy_config_sha256": digest(policy_config),
        "seed": options.seed,
        "cases": options.cases,
        "device": options.device,
        "selection_rule": "Case 88 is the first archived audit case; case 96 is the first seed-2027 case with Persistent/Reset candidate-index divergence. No outcome-based case filtering.",
        "audit": "Uniform same-state score probes preserve RNG state and use cached predictions; trajectories come from simulator states, not integrated plotting splines.",
        "claim_boundary": "Archived September 6 controller, single-robot CrowdSimPlus, illustrative rerun; not the current in-progress H3 multi-robot campaign. No claim of universal superiority.",
    }
    provenance_path = output / "replay_provenance.json"
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text())
        if existing != provenance:
            raise ValueError("Replay provenance changed; use a fresh output directory")
    write_json(output / "replay_provenance.json", provenance)
    if options.only in {"tracer", "all"}:
        model = tracer._load_model(checkpoint_path, args.device, expected_dt=args.env_dt)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        calibration = checkpoint["belief_calibration"]
        if calibration["phase"] != "validation":
            raise ValueError("Expected validation-only belief calibration")
        tracer._validate_external_belief_cases(checkpoint, min(options.cases), max(options.cases) - min(options.cases) + 1)
        args.belief_update_stride = int(calibration["observation_stride"])
        args.likelihood_scale = float(calibration["selected"]["likelihood_scale"])
        model.mode_semantics = checkpoint["mode_semantics"]
        adapter = tracer.SAGECandidateAdapter.from_checkpoint(
            ROOT / args.sage_checkpoint, args.device,
            tracer.SAGECandidateAdapterConfig(
                num_candidates=args.num_candidates, guided=True, step_stride=args.sage_step_stride,
                goal_speed=args.sage_goal_speed, oversample_factor=args.sage_oversample_factor,
                task_prefilter=True, task_prefix_steps=args.execute_steps, goal_direction="task_goal",
                include_task_anchor=True, task_anchor_speed_scales=(0.45,),
                batch_size=args.num_candidates, guidance_step_clip_norm=0.1,
                diversity_weight=args.sage_diversity_weight,
            ),
        )
        original_score = tracer._score_candidates
        signature = inspect.signature(original_score)
        decisions = []

        def score_and_capture(*score_args, **score_kwargs):
            bound = signature.bind(*score_args, **score_kwargs).arguments
            counterfactual = bound.get("cached_predictions") is not None
            if counterfactual:
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                numpy_rng = np.random.get_state()
                python_rng = random.getstate()
            try:
                result = original_score(*score_args, **score_kwargs)
            finally:
                if counterfactual:
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state_all(cuda_rng)
                    np.random.set_state(numpy_rng)
                    random.setstate(python_rng)
            candidates = np.asarray(bound["global_candidates"])
            if counterfactual:
                decisions[-1]["uniform_selected"] = int(result[0])
                decisions[-1]["uniform_path"] = candidates[int(result[0])].copy()
            else:
                decisions.append({
                    "belief": bound["beliefs"].detach().exp().cpu().numpy().copy(),
                    "selected": int(result[0]),
                    "selected_path": candidates[int(result[0])].copy(),
                    "robot": np.asarray(bound["robot"].get_position()),
                    "humans": np.asarray([human.get_position() for human in bound["humans_list"]]),
                })
            return result

        tracer._score_candidates = score_and_capture
        captures = []
        original_episode = capture_environment(tracer, captures)
        try:
            for case in options.cases:
                policies = ["tracer-persistent", "tracer-reset"] if case == 96 else ["tracer-persistent"]
                for policy in policies:
                    stem = data_dir / f"case{case}_{policy}"
                    if stem.with_suffix(".json").exists():
                        continue
                    print(f"Replaying case={case} policy={policy}", flush=True)
                    torch.manual_seed(options.seed)
                    np.random.seed(options.seed)
                    random.seed(options.seed)
                    captures.clear()
                    decisions.clear()
                    windows = []
                    metrics = tracer._run_episode(model, adapter, policy, external_root, case,
                                                  environment_config, args, options.seed, [], windows)
                    if len(captures) != 1:
                        raise ValueError("Expected exactly one factual environment")
                    arrays = environment_arrays(*captures[0])
                    for key in ["belief", "selected", "uniform_selected", "selected_path", "uniform_path"]:
                        arrays[key] = np.asarray([decision[key] for decision in decisions])
                    arrays["decision_robot"] = np.asarray([decision["robot"] for decision in decisions])
                    arrays["decision_humans"] = np.asarray([decision["humans"] for decision in decisions])
                    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
                    write_json(stem.with_suffix(".json"), {
                        "metrics": metrics, "windows": windows,
                        "mode_semantics": checkpoint["mode_semantics"],
                    })
                    print(f"Saved {stem.name}: {len(arrays['robot'])} frames", flush=True)
        finally:
            tracer._score_candidates = original_score
            tracer._new_episode = original_episode
    if options.only in {"official", "all"}:
        captures = []
        original_episode = capture_environment(official, captures)
        try:
            for case in options.cases:
                for policy in ["sicnav-np", "sicnav-p", "official-orca"]:
                    stem = data_dir / f"case{case}_{policy}"
                    if stem.with_suffix(".json").exists():
                        continue
                    print(f"Replaying case={case} policy={policy}", flush=True)
                    captures.clear()
                    metrics = official._run_episode(external_root, policy, case, environment_config,
                                                    policy_config, args.max_steps)
                    arrays = environment_arrays(*captures[0])
                    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
                    write_json(stem.with_suffix(".json"), {"metrics": metrics})
                    print(f"Saved {stem.name}: {len(arrays['robot'])} frames", flush=True)
        finally:
            official._new_episode = original_episode


if __name__ == "__main__":
    main()
