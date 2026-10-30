from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tracer.experiments.collect_external_planner_supervision import file_hash
from tracer.experiments.external_value_policy import value_policy_active
from tracer.experiments.train_executable_value import load_continuation_groups
from tracer.experiments.train_executable_value_ensemble import resample_scene_groups
from tracer.response.executable_value import FEATURE_VERSION, fit_paired_value, prefix_value_feature_indices, value_action
from tracer.response.reliable_executable_value import (
    DEFAULT_MINIMUM_GAINS, RELIABILITY_IMPROVEMENT_TIERS, RELIABILITY_OBJECTIVE,
    RELIABILITY_TARGETS, RELIABILITY_VETO_INDICES,
)


def training_support(groups, feature_indices):
    eligible = [group for group in groups if group["rerank_allowed"]]
    features = np.concatenate([group["features"][:, feature_indices] for group in eligible])
    cases = np.concatenate([np.full(len(group["features"]), group["case"]) for group in eligible])
    if len(set(cases)) < 4:
        raise ValueError("reliable value support requires at least four training scenes")
    center, scale = features.mean(axis=0), np.maximum(features.std(axis=0), 1e-3)
    normalized = (features - center) / scale
    generator = np.random.default_rng(970907)
    selected = np.sort(generator.choice(len(features), min(1024, len(features)), replace=False))
    anchors, anchor_cases = normalized[selected], cases[selected]
    nearest = []
    for anchor, case in zip(anchors, anchor_cases):
        others = anchors[anchor_cases != case]
        nearest.append(float(np.sqrt(np.square(others - anchor).mean(axis=1)).min()))
    return {"feature_indices": feature_indices, "center": center.tolist(), "scale": scale.tolist(),
            "anchors": anchors.tolist(), "maximum_distance": float(np.quantile(nearest, .95)),
            "scope": "Train-only cross-scene nearest-neighbour abstention; not calibrated safety probability."}


def evaluate_reliable_value(model, groups):
    records = []
    intervention_counts = {}
    for group in sorted(groups, key=lambda item: (item["case"], item["window"])):
        intervention_count = intervention_counts.get(group["case"], 0)
        active = group["rerank_allowed"] and value_policy_active(
            model, int(group["window"]), intervention_count
        )
        if active:
            selected, _ = value_action(
                model, group["features"], group["parent"], 0., group["response_risk_support"]
            )
        else:
            selected = group["parent"]
        intervention_counts[group["case"]] = intervention_count + int(selected != group["parent"])
        differences = group["targets"][selected] - group["targets"][group["parent"]]
        records.append({"case": group["case"], "window": group["window"],
                        "selected": group["candidate_indices"][selected],
                        "parent_selected": group["candidate_indices"][group["parent"]],
                        "changed": selected != group["parent"],
                        "differences": dict(zip(RELIABILITY_TARGETS, differences.tolist()))})
    cases = sorted({row["case"] for row in records})
    indices = np.random.default_rng(970908).integers(0, len(cases), (4000, len(cases)))
    metrics = {}
    for name in RELIABILITY_TARGETS:
        means = np.asarray([np.mean([row["differences"][name] for row in records if row["case"] == case]) for case in cases])
        metrics[name] = {"mean_difference": float(means.mean()),
                         "case_bootstrap_ci95": np.quantile(means[indices].mean(axis=1), [.025, .975]).tolist()}
    veto_names = [RELIABILITY_TARGETS[index] for index in RELIABILITY_VETO_INDICES]
    worsening = [row for row in records if any(row["differences"][name] > 0 for name in veto_names)]
    tiers = tuple(tuple((RELIABILITY_TARGETS[index], 1) for index in tier)
                  for tier in RELIABILITY_IMPROVEMENT_TIERS)
    first_improvement, earlier_tiers_not_worse = None, True
    for index, tier in enumerate(tiers):
        differences = [direction * metrics[name]["mean_difference"] for name, direction in tier]
        if any(value > 1e-10 for value in differences):
            earlier_tiers_not_worse = False
            break
        if any(value < -1e-10 for value in differences):
            first_improvement = index
            break
    changed_cases = len({row["case"] for row in records if row["changed"]})
    eligible = not worsening and earlier_tiers_not_worse and first_improvement is not None and changed_cases >= 2
    return {"records": records, "metrics": metrics, "changed_cases": changed_cases,
            "added_safety_or_task_failure_groups": len(worsening), "eligible_for_development_navigation": eligible,
            "first_improved_priority_tier": first_improvement,
            "deployment_stride": model.get("deployment_stride", 1),
            "maximum_episode_interventions": model.get("maximum_episode_interventions"),
            "boundary": "Scheduled-intervention development screening only; safety/failure heads are veto-only and cannot create a deployment gain; actual closed-loop safety and progress remain unverified."}


def main():
    parser = argparse.ArgumentParser(description="Fit conservative, safety-first executable actions with complete navigation outcomes.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--regularization", nargs="+", type=float, default=[.1, 1., 10.])
    parser.add_argument("--uncertainty-multiplier", nargs="+", type=float, default=[.01, .05, .1, .25, .5, 1., 2.])
    parser.add_argument("--nonlinear-bandwidth", nargs="+", type=float, default=[.5, 1., 2.])
    parser.add_argument("--deployment-stride", type=int, default=1)
    parser.add_argument("--maximum-episode-interventions", type=int, default=1)
    parser.add_argument("--minimum-human-gain", type=float, default=DEFAULT_MINIMUM_GAINS[7])
    args = parser.parse_args()
    if (args.members < 8 or args.deployment_stride < 1 or args.maximum_episode_interventions < 1
        or not np.isfinite(args.minimum_human_gain) or args.minimum_human_gain <= 0
        or not args.regularization or any(
        not np.isfinite(value) or value <= 0 for value in args.regularization
    ) or not args.uncertainty_multiplier or any(
        not np.isfinite(value) or value <= 0 for value in args.uncertainty_multiplier
    ) or not args.nonlinear_bandwidth or any(
        not np.isfinite(value) or value <= 0 for value in args.nonlinear_bandwidth
    )):
        raise ValueError("reliable training requires at least eight scene fits and positive grid values")
    train, train_protocol, hashes = load_continuation_groups(args.train_dir, "train", "reliability")
    validation, val_protocol, val_hashes = load_continuation_groups(args.val_dir, "val", "reliability")
    hashes.update(val_hashes)
    for field in ("checkpoint_sha256", "case_splits", "seed", "dt", "execute_steps", "continuation_branches", "behavior_value_sha256"):
        if train_protocol.get(field) != val_protocol.get(field):
            raise ValueError(f"reliable train/validation protocol differs: {field}")
    if set(train_protocol["cases"]) & set(val_protocol["cases"]):
        raise ValueError("reliable train/validation scenes must be disjoint")
    if train_protocol.get("behavior_value_model"):
        raise ValueError("the learned collection reference has no safety acceptance; do not deploy it as a certified fallback")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes.update({str(path): file_hash(path) for path in Path(__file__).resolve().parents[1].rglob("*.py")})
    feature_indices = prefix_value_feature_indices(train[0]["names"])
    generator = np.random.default_rng(970909)
    models = []
    trials = []
    for regularization in args.regularization:
        for nonlinear_bandwidth in args.nonlinear_bandwidth:
            members = []
            for index in range(args.members):
                members.append(fit_paired_value(resample_scene_groups(train, generator), regularization,
                                                nonlinear_bandwidth, feature_indices))
                print(f"reliability scene fit regularization={regularization} bandwidth={nonlinear_bandwidth} "
                      f"{index + 1}/{args.members}", flush=True)
            for uncertainty_multiplier in args.uncertainty_multiplier:
                model = {"feature_version": FEATURE_VERSION, "selection_objective": RELIABILITY_OBJECTIVE,
                         "target_names": list(RELIABILITY_TARGETS), "ensemble": members,
                         "regularization": regularization, "nonlinear_bandwidth": nonlinear_bandwidth,
                         "uncertainty_multiplier": uncertainty_multiplier,
                         "minimum_gains": [
                             *DEFAULT_MINIMUM_GAINS[:7], args.minimum_human_gain, DEFAULT_MINIMUM_GAINS[8]
                         ],
                         "training_support": training_support(train, feature_indices), "response_risk_guard": True,
                         "feature_names": train[0]["names"], "minimum_gain": 0.,
                         "branch_count": train_protocol["continuation_branches"],
                         "response_checkpoint_sha256": train_protocol["checkpoint_sha256"],
                         "case_splits": train_protocol["case_splits"], "parent_parameters": train_protocol["parameters"],
                         "parent_artifact_hashes": train_protocol["parent_artifact_hashes"],
                         "training_behavior_value_sha256": None, "reference_value_model": None,
                         "reference_scope": "Original constrained controller, not a promoted learned collection policy.",
                         "deployment_stride": args.deployment_stride,
                         "maximum_episode_interventions": args.maximum_episode_interventions,
                         "source_and_input_hashes": hashes, "deployment_status": "DEVELOPMENT_ONLY"}
                evaluation = evaluate_reliable_value(model, validation)
                tier = evaluation["first_improved_priority_tier"]
                trial = {"regularization": regularization, "nonlinear_bandwidth": nonlinear_bandwidth,
                         "uncertainty_multiplier": uncertainty_multiplier,
                         "eligible_for_development_navigation": evaluation["eligible_for_development_navigation"],
                         "first_improved_priority_tier": tier, "changed_cases": evaluation["changed_cases"],
                         "added_safety_or_task_failure_groups": evaluation["added_safety_or_task_failure_groups"],
                         "validation_metrics": evaluation["metrics"]}
                trials.append(trial)
                models.append((model, evaluation))
                print(json.dumps(trial, allow_nan=False), flush=True)
    changed = [path for path, digest in hashes.items() if not Path(path).exists() or file_hash(path) != digest]
    eligible = [(model, evaluation) for model, evaluation in models
                if evaluation["eligible_for_development_navigation"]]
    def selection_key(pair):
        evaluation = pair[1]
        tier = evaluation["first_improved_priority_tier"]
        metrics = evaluation["metrics"]
        priority = (
            metrics["stagnation_time"]["mean_difference"],
            metrics["goal_shortfall"]["mean_difference"],
            metrics["human_velocity_variation"]["mean_difference"],
            metrics["elapsed_time"]["mean_difference"],
        )
        return (tier, *priority)
    selected = min(eligible, key=selection_key) if eligible else None
    fallback = min(models, key=lambda pair: (pair[1]["added_safety_or_task_failure_groups"],
                                               pair[1]["changed_cases"],
                                               pair[1]["metrics"]["elapsed_time"]["mean_difference"]))
    model, evaluation = selected or fallback
    summary = {"status": "SOURCE_CHANGED" if changed else "COMPLETED", "changed_inputs": changed,
               "train_cases": train_protocol["cases"], "validation_cases": val_protocol["cases"],
               "target_names": list(RELIABILITY_TARGETS), "validation": evaluation, "trials": trials,
               "selected_regularization": model["regularization"],
               "selected_nonlinear_bandwidth": model["nonlinear_bandwidth"],
               "selected_uncertainty_multiplier": model["uncertainty_multiplier"],
               "deployment_stride": model["deployment_stride"],
               "maximum_episode_interventions": model["maximum_episode_interventions"],
               "minimum_human_gain": model["minimum_gains"][7],
               "eligible_model": selected is not None, "candidate_policy_saved": not changed,
               "boundary": "No model promotion; safety, stagnation, progress and human perturbation precede elapsed time."}
    if not changed:
        model["validation"] = evaluation
        (output / "candidate_policy.json").write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if changed:
        raise RuntimeError("reliable training inputs changed")


if __name__ == "__main__":
    main()
