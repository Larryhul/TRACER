from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sage.experiments.collect_external_planner_supervision import file_hash
from sage.experiments.train_executable_value import load_continuation_groups
from sage.response.executable_value import value_action, value_representation
from sage.response.reliable_executable_value import (
    DEFAULT_MINIMUM_GAINS,
    PAIRWISE_EFFECT_OBJECTIVE,
    PAIRWISE_RELIABILITY_OBJECTIVE,
    RELIABILITY_TARGETS,
    pairwise_reliability_features,
    pairwise_reliable_model_action,
    reliable_action,
)


def prediction_upper(model, features, parent):
    members = []
    for member in model["ensemble"]:
        represented = value_representation(features, member)
        relative = (represented - represented[parent]) / np.asarray(member["scale"])
        members.append(relative @ np.asarray(member["coefficients"]))
    members = np.asarray(members)
    predictions = members.mean(axis=0)
    return predictions + float(model["uncertainty_multiplier"]) * members.std(axis=0, ddof=1)


def training_support(model, features, parent=None):
    domain = model["training_support"]
    if model.get("selection_objective") in {
        PAIRWISE_RELIABILITY_OBJECTIVE, PAIRWISE_EFFECT_OBJECTIVE,
    }:
        if parent is None:
            raise ValueError("pairwise training support requires the reference action")
        features = pairwise_reliability_features(features, parent, model["basic_feature_count"])
    else:
        features = np.asarray(features, dtype=float)[:, domain["feature_indices"]]
    selected = np.asarray(features, dtype=float)
    normalized = (selected - np.asarray(domain["center"])) / np.asarray(domain["scale"])
    anchors = np.asarray(domain["anchors"])
    distances = np.sqrt(np.square(normalized[:, None] - anchors[None]).mean(axis=-1)).min(axis=1)
    return distances <= domain["maximum_distance"], distances


def parse_reference_actions(values):
    result = {}
    for value in values or ():
        try:
            case_text, window_text, candidate_text = value.split(":", 2)
            key, candidate = (int(case_text), int(window_text)), int(candidate_text)
        except (TypeError, ValueError):
            raise ValueError("reference actions must use case:window:candidate integers") from None
        if key in result:
            raise ValueError("reference actions must identify unique case/window groups")
        result[key] = candidate
    return result


def audit_opportunity(groups, model, reference_actions=None, planner_score_tolerance=0.0):
    pairwise = model.get("selection_objective") in {
        PAIRWISE_RELIABILITY_OBJECTIVE, PAIRWISE_EFFECT_OBJECTIVE,
    }
    gains = DEFAULT_MINIMUM_GAINS if pairwise else model["minimum_gains"]
    if not np.isfinite(planner_score_tolerance) or planner_score_tolerance < 0.0:
        raise ValueError("planner score tolerance must be finite and nonnegative")
    reference_actions = reference_actions or {}
    available = {(int(group["case"]), int(group["window"])) for group in groups}
    if not set(reference_actions).issubset(available):
        raise ValueError("reference action override does not match a continuation group")
    records = []
    for group in groups:
        if not group["rerank_allowed"]:
            continue
        key = (int(group["case"]), int(group["window"]))
        reference = reference_actions.get(key)
        if reference is None:
            parent = group["parent"]
        elif reference not in group["candidate_indices"]:
            raise ValueError("reference action override is absent from continuation branches")
        else:
            parent = group["candidate_indices"].index(reference)
        actual = group["targets"] - group["targets"][parent]
        risk = np.asarray(group["response_risk_support"], dtype=bool)
        if reference is not None:
            from sage.response.executable_value import response_risk_admissible

            risk = response_risk_admissible(group["diagnostics"], reference)[group["candidate_indices"]]
        planner_scores = np.asarray(group.get("planner_scores", np.zeros(len(group["candidate_indices"]))))
        if planner_scores.shape != (len(group["candidate_indices"]),):
            raise ValueError("planner scores must align with continuation candidates")
        planner_support = np.isfinite(planner_scores) & (
            planner_scores <= planner_scores[parent] + planner_score_tolerance + 1e-8
        )
        domain, distances = training_support(model, group["features"], parent)
        upper = None if pairwise else prediction_upper(model, group["features"], parent)
        actual_unconstrained = reliable_action(actual, parent, gains)
        actual_risk = reliable_action(actual, parent, gains, risk)
        actual_deployment = reliable_action(actual, parent, gains, risk & planner_support)
        if pairwise:
            predicted_risk = pairwise_reliable_model_action(model, group["features"], parent, risk)[0]
            predicted_supported = (
                parent if not domain[parent]
                else pairwise_reliable_model_action(model, group["features"], parent, risk & domain)[0]
            )
        else:
            predicted_risk = reliable_action(upper, parent, gains, risk)
            predicted_supported = parent if not domain[parent] else reliable_action(upper, parent, gains, risk & domain)
        deployed, _ = value_action(model, group["features"], parent, 0., risk & planner_support)
        deployed_difference = actual[deployed]
        if pairwise:
            from sage.response.reliable_executable_value import pairwise_reliability_predictions

            _, _, lower_human, upper_harm = pairwise_reliability_predictions(
                model, group["features"], parent
            )
            deployed_prediction_upper = {
                "human_improvement_lower_probability": float(lower_human[deployed]),
                "safety_or_task_harm_upper_probability": float(upper_harm[deployed]),
            }
        else:
            deployed_prediction_upper = dict(zip(RELIABILITY_TARGETS, upper[deployed].tolist()))
        records.append({
            "case": group["case"],
            "window": group["window"],
            "parent": group["candidate_indices"][parent],
            "actual_unconstrained": group["candidate_indices"][actual_unconstrained],
            "actual_risk_supported": group["candidate_indices"][actual_risk],
            "actual_deployment_supported": group["candidate_indices"][actual_deployment],
            "predicted_risk_supported": group["candidate_indices"][predicted_risk],
            "predicted_domain_supported": group["candidate_indices"][predicted_supported],
            "deployed": group["candidate_indices"][deployed],
            "parent_in_training_support": bool(domain[parent]),
            "domain_supported_candidates": int((domain & risk).sum()),
            "planner_supported_candidates": int((planner_support & risk).sum()),
            "planner_supported_alternatives": int(
                (planner_support & risk).sum() - int(planner_support[parent] and risk[parent])
            ),
            "closest_training_distance": float(distances.min()),
            "deployed_training_distance": float(distances[deployed]),
            "actual_risk_supported_differences": dict(zip(RELIABILITY_TARGETS, actual[actual_risk].tolist())),
            "deployed_differences": dict(zip(RELIABILITY_TARGETS, deployed_difference.tolist())),
            "deployed_prediction_upper": deployed_prediction_upper,
        })
    changed = lambda name: [row for row in records if row[name] != row["parent"]]
    unconstrained = changed("actual_unconstrained")
    risk_supported = changed("actual_risk_supported")
    deployment_supported = changed("actual_deployment_supported")
    predicted_risk = changed("predicted_risk_supported")
    predicted_domain = changed("predicted_domain_supported")
    deployed = changed("deployed")
    deployed_harm = [row for row in records if any(
        row["deployed_differences"][name] > 1e-10
        for name in (RELIABILITY_TARGETS[index] for index in range(6))
    )]
    deployed_time = [row["deployed_differences"]["elapsed_time"] for row in records]
    return {
        "groups": len(records),
        "cases": len({row["case"] for row in records}),
        "actual_opportunity_groups": len(unconstrained),
        "actual_risk_supported_opportunity_groups": len(risk_supported),
        "actual_opportunities_rejected_by_risk_guard": len(unconstrained) - len(risk_supported),
        "actual_deployment_supported_opportunity_groups": len(deployment_supported),
        "actual_risk_opportunities_rejected_by_planner_support": (
            len(risk_supported) - len(deployment_supported)
        ),
        "planner_supported_alternative_pairs": sum(
            row["planner_supported_alternatives"] for row in records
        ),
        "predicted_risk_supported_opportunity_groups": len(predicted_risk),
        "predicted_opportunities_rejected_by_training_support": len(predicted_risk) - len(predicted_domain),
        "deployed_opportunities_rejected_by_planner_support": len(predicted_risk) - len(deployed),
        "parent_out_of_training_support_groups": sum(not row["parent_in_training_support"] for row in records),
        "deployed_change_groups": len(deployed),
        "deployed_safe_or_task_harm_groups": len(deployed_harm),
        "deployed_mean_elapsed_time_difference": float(np.mean(deployed_time)) if deployed_time else 0.0,
        "records": records,
        "boundary": "Development diagnostic on one-intervention outcomes; actual opportunity is not a deployable oracle or repeated-policy result.",
    }


def main():
    parser = argparse.ArgumentParser(description="Explain conservative reliable-value abstention on frozen continuations.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "diagnostic"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-action", action="append")
    parser.add_argument("--planner-score-tolerance", type=float, default=0.0)
    args = parser.parse_args()
    model_path = Path(args.model).resolve()
    model = json.loads(model_path.read_text())
    groups, _, hashes = load_continuation_groups(args.data_dir, args.split, "reliability")
    hashes[str(model_path)] = file_hash(model_path)
    report = audit_opportunity(
        groups, model, parse_reference_actions(args.reference_action), args.planner_score_tolerance
    )
    changed = [path for path, digest in hashes.items() if not Path(path).exists() or file_hash(path) != digest]
    report.update({"status": "SOURCE_CHANGED" if changed else "COMPLETED", "changed_inputs": changed,
                   "input_hashes": hashes})
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("reliable opportunity audit output already exists")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("records", "input_hashes")}, indent=2))


if __name__ == "__main__":
    main()
