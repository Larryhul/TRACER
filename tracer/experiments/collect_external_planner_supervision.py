from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from tracer.experiments.calibrate_external_belief_cadence import _controller_args, _validate_splits, validation_case_splits
from tracer.experiments.calibrate_persistent_belief import load_belief_calibration
from tracer.experiments.eval_persistent_closed_loop import _load_model
from tracer.experiments.external_information_value import DecisionInformationAudit
from tracer.experiments.external_value_policy import validate_value_runtime, value_policy_digest
from tracer.experiments.run_sicnav_tracer_loop import _goal_response_mask, _planning_cost_config, _run_episode
from tracer.response.sage_candidate_adapter import SAGECandidateAdapter, SAGECandidateAdapterConfig


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collection_cases(splits, split):
    if split not in {"train", "val"}:
        raise ValueError("planner supervision excludes test cases")
    groups = [set(splits[name]) for name in ("train", "val", "test")]
    if any(not group for group in groups) or any(
        groups[first] & groups[second] for first, second in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("planner supervision requires nonempty disjoint case splits")
    return sorted(groups[("train", "val").index(split)])


def expanded_case_splits(original, proposed):
    proposed = _validate_splits(proposed)
    for name in ("train", "val", "test"):
        if not set(original[name]).issubset(proposed[name]):
            raise ValueError(f"expanded supervision must retain the original {name} cases in their original role")
    return proposed


def parse_case_windows(values, cases, maximum_window):
    if not values:
        return None
    result = set()
    for value in values:
        try:
            case_text, window_text = value.split(":", 1)
            item = (int(case_text), int(window_text))
        except (TypeError, ValueError):
            raise ValueError("targeted audit windows must use case:window integers") from None
        if item[0] not in cases or not 0 <= item[1] < maximum_window or item in result:
            raise ValueError("targeted audit windows must be unique and inside the selected cases/runtime")
        result.add(item)
    return result


class PlannerSupervisionAudit(DecisionInformationAudit):
    @torch.no_grad()
    def observe(self, **values):
        count = len(self.records)
        super().observe(**values)
        if len(self.records) == count:
            return
        env, args = values["env"], values["args"]
        horizon = int(values["model"].config.horizon)
        candidates = values["global_candidates"][:, :horizon]
        direction = np.asarray(env.robot.get_goal_position()) - env.robot.get_position()
        direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
        active = np.ones(candidates.shape[:2], dtype=bool)
        if getattr(args, "goal_terminal_response", False):
            active = _goal_response_mask(candidates, np.asarray(env.robot.get_goal_position()), env.robot.radius)
        means, log_stds, references, _, _ = values["cached_predictions"]
        name = f"seed{values['seed']}_case{values['case']}_window{values['window']}"
        inputs = self.output.parent / "inputs"
        inputs.mkdir(exist_ok=True)
        np.savez_compressed(
            inputs / f"{name}.npz", histories=values["histories"], current=values["current"],
            candidates=values["candidates"][:, :, :horizon], static_obstacles=np.asarray(values["static_obstacles"]),
            beliefs=values["beliefs"].cpu().numpy(), direction=direction, response_valid_steps=active,
            penalty=np.asarray(values["diagnostics"]["observable_penalty"]) + np.asarray(values["candidate_bias"]),
            teacher_means=means.cpu().numpy(), teacher_log_stds=log_stds.cpu().numpy(),
            teacher_references=references.cpu().numpy(),
            robot_position=np.asarray(env.robot.get_position()),
            robot_velocity=np.asarray(env.robot.get_velocity()),
            robot_goal=np.asarray(env.robot.get_goal_position()),
            robot_heading=np.asarray(env.robot.theta), robot_radius=np.asarray(env.robot.radius),
            robot_preferred_speed=np.asarray(env.robot.v_pref),
            human_positions=np.asarray([human.get_position() for human in env.humans]),
            human_velocities=np.asarray([human.get_velocity() for human in env.humans]),
            human_radii=np.asarray([human.radius for human in env.humans]),
            decision_scores=values["decision_scores"].cpu().numpy(),
            **{f"score_component_{name}": np.asarray(value) for name, value in values["diagnostics"].items()},
        )


def main():
    parser = argparse.ArgumentParser(description="Collect exact planner candidate supervision without using test scenes.")
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "diagnostic"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-stride", type=int, default=1)
    parser.add_argument("--prefix-diversity", action="store_true")
    parser.add_argument("--continuation-windows", type=int, default=0)
    parser.add_argument("--continuation-branches", type=int, default=6)
    parser.add_argument("--behavior-value-model")
    parser.add_argument("--case-splits")
    parser.add_argument("--case-subset", type=int, nargs="+")
    parser.add_argument("--audit-case-window", nargs="+")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    checkpoint = Path(args.checkpoint).resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parameters = json.loads(Path(args.source_metadata).read_text(encoding="utf-8"))["parameters"]
    splits = validation_case_splits(saved, parameters)
    if args.case_splits:
        splits = expanded_case_splits(splits, json.loads(Path(args.case_splits).read_text(encoding="utf-8")))
    if args.split == "diagnostic":
        disclosed = set().union(*(set(values) for values in splits.values()))
        if (not args.case_subset or len(set(args.case_subset)) != len(args.case_subset)
                or disclosed.intersection(args.case_subset)):
            raise ValueError("diagnostic cases must be unique, nonempty and outside frozen train/val/test")
        cases = sorted(args.case_subset)
    else:
        cases = collection_cases(splits, args.split)
        if args.case_subset is not None:
            if not args.case_subset or len(set(args.case_subset)) != len(args.case_subset) or not set(args.case_subset).issubset(cases):
                raise ValueError("collection shard must be a unique nonempty subset of its declared split")
            cases = sorted(args.case_subset)
    seed = int(saved["seed"])
    runtime = _controller_args(parameters, checkpoint, output, splits, seed, args.device)
    runtime.cases = len(cases)
    runtime.case_start = min(cases)
    runtime.policies = ["tracer-persistent"]
    runtime.belief_calibration_path = str(Path(args.calibration).resolve())
    runtime.belief_cadence_source = "checkpoint_bound_validation_calibration"
    calibration = load_belief_calibration(args.calibration, checkpoint, execute_steps=1, dt=.25, num_robots=1)
    runtime.likelihood_scale = calibration["selected"]["likelihood_scale"]
    runtime.belief_selection_audit = True
    runtime.belief_evidence_audit = False
    runtime.planner_sensitivity_audit = False
    runtime.belief_cost_leverage = 1.0
    runtime.prefix_diversity = args.prefix_diversity
    case_windows = parse_case_windows(args.audit_case_window, set(cases), runtime.windows)
    if case_windows is not None and not args.continuation_windows:
        raise ValueError("targeted audit windows require continuation supervision")
    behavior_value_model = None
    if args.behavior_value_model:
        if not args.continuation_windows:
            raise ValueError("learned behavior collection requires verified complete continuations")
        behavior_value_model = json.loads(Path(args.behavior_value_model).read_text(encoding="utf-8"))
        expanded_case_splits(splits, behavior_value_model["case_splits"])
        validate_value_runtime(behavior_value_model, runtime, file_hash(checkpoint))
        runtime.belief_selection_audit = False
    if runtime.hard_risk_filter:
        raise ValueError("supervised candidate support must use belief-independent hard safety masks")
    paths = [checkpoint, Path(args.calibration).resolve(), Path(args.source_metadata).resolve(),
             Path(runtime.sage_checkpoint).resolve(), Path(runtime.env_config).resolve(), Path(runtime.motion_policy_config).resolve()]
    paths += list(Path(runtime.external_root).parent.joinpath("python_rvo2_official").glob("*.so"))
    if args.case_splits:
        paths.append(Path(args.case_splits).resolve())
    if args.behavior_value_model:
        paths.append(Path(args.behavior_value_model).resolve())
    paths += list(Path(__file__).resolve().parents[1].rglob("*.py")) + list(Path(runtime.external_root).rglob("*.py"))
    hashes = {str(path.resolve()): file_hash(path) for path in paths}
    protocol = {
        "version": "planner-supervision-v1", "split": args.split, "cases": cases, "case_splits": splits,
        "seed": seed, "parameters": vars(runtime), "planning": asdict(_planning_cost_config(runtime)),
        "dt": runtime.model_dt, "execute_steps": runtime.execute_steps, "source_hashes": hashes,
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
        "boundary": "Frozen factual Persistent collector; cloned futures and speed labels are offline supervision only; never controller inputs. Validation groups are selection-only, never gradient updates.",
        "continuation_windows": args.continuation_windows,
        "continuation_branches": args.continuation_branches if args.continuation_windows else 0,
        "continuation_outcome_schema": "navigation-reliability-v1" if args.continuation_windows >= runtime.windows else None,
        "behavior_value_model": str(Path(args.behavior_value_model).resolve()) if args.behavior_value_model else None,
        "behavior_value_sha256": value_policy_digest(behavior_value_model),
        "targeted_case_windows": sorted(case_windows) if case_windows is not None else None,
        "behavior_boundary": "When configured, the identical frozen learned value policy generates factual states and every continuation. This is policy-iteration supervision, not a belief attribution experiment or a policy promotion.",
        "continuation_target": "Execute one supported distinct prefix, then the identical frozen controller with branch-specific factual history and belief updates; save actual collision, success, time and remaining distance. These offline futures never enter the factual controller.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"status": "COLLECTING"}) + "\n")
    model = _load_model(checkpoint, args.device, expected_dt=.25)
    adapter = SAGECandidateAdapter.from_checkpoint(
        Path(runtime.sage_checkpoint).resolve(), args.device,
        SAGECandidateAdapterConfig(
            num_candidates=runtime.num_candidates, guided=True, step_stride=runtime.sage_step_stride,
            goal_speed=runtime.sage_goal_speed, oversample_factor=runtime.sage_oversample_factor,
            task_prefilter=True, task_prefix_steps=runtime.execute_steps, goal_direction="task_goal",
            include_task_anchor=True, task_anchor_speed_scales=(.45,), batch_size=runtime.num_candidates,
            guidance_step_clip_norm=.1, diversity_weight=getattr(runtime, "sage_diversity_weight", 0.0),
        ),
    )
    audit = PlannerSupervisionAudit(output / "audit", saved["speed_thresholds_train_only"], args.audit_stride)
    if args.continuation_windows:
        from tracer.experiments.external_replanning_supervision import ReplanningSupervisionAudit
        audit = ReplanningSupervisionAudit(
            output / "audit", saved["speed_thresholds_train_only"], args.audit_stride,
            adapter, args.continuation_windows, args.continuation_branches,
            executable_value_model=behavior_value_model,
            case_windows=case_windows,
        )
    episodes = []
    for case in cases:
        episodes.append(_run_episode(
            model, adapter, "tracer-persistent", Path(runtime.external_root).resolve(), case,
            Path(runtime.env_config).resolve(), runtime, seed, decision_audit=audit,
            executable_value_model=behavior_value_model,
        ))
        if args.continuation_windows:
            audit.verify_episode(episodes[-1])
        with (output / "episodes.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
            writer.writeheader()
            writer.writerows(episodes)
        print(f"planner supervision split={args.split} case={case} groups={len(audit.records)}", flush=True)
    changed = [path for path, digest in hashes.items() if not Path(path).exists() or file_hash(path) != digest]
    manifest = {"status": "SOURCE_CHANGED" if changed else "COMPLETED", "changed_sources": changed,
                "groups": len(audit.records), "verified_groups": sum(row.get("live_prefix_verified", False) for row in audit.records),
                "cases": cases}
    if args.continuation_windows:
        manifest.update(audit.finalize())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if changed:
        raise RuntimeError("sources or protected inputs changed during collection")


if __name__ == "__main__":
    main()
