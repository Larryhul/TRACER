from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sage.experiments.collect_external_planner_supervision import collection_cases, file_hash
from sage.response.executable_value import executable_value_features, fit_paired_value, prefix_value_feature_indices, response_risk_admissible, value_action


def continuation_cases(protocol, split):
    if split != "diagnostic":
        return collection_cases(protocol["case_splits"], split)
    cases = set(protocol["cases"])
    disclosed = set().union(*(set(values) for values in protocol["case_splits"].values()))
    if not cases or cases & disclosed:
        raise ValueError("diagnostic continuations must remain outside frozen train/val/test cases")
    return sorted(cases)


def load_continuation_groups(root, split, target_schema="legacy"):
    if target_schema not in {"legacy", "reliability"}:
        raise ValueError("unknown executable value target schema")
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "COMPLETED" or manifest.get("changed_sources") or protocol["split"] != split:
        raise ValueError("executable value training requires unchanged completed split-labelled continuations")
    allowed = set(continuation_cases(protocol, split))
    if not set(protocol["cases"]).issubset(allowed) or manifest["cases"] != protocol["cases"]:
        raise ValueError("continuation cases must belong to their declared split")
    if protocol["continuation_windows"] < protocol["parameters"]["windows"]:
        raise ValueError("navigation value supervision must continue to the original episode termination")
    groups, hashes, names = [], {}, None
    stored_sources = protocol["source_hashes"]
    parent_artifacts = {}
    behavior_path = protocol.get("behavior_value_model")
    if behavior_path:
        from sage.experiments.external_value_policy import value_policy_digest

        behavior_path = Path(behavior_path).resolve()
        behavior = json.loads(behavior_path.read_text())
        if (stored_sources.get(str(behavior_path)) != file_hash(behavior_path)
                or protocol.get("behavior_value_sha256") != value_policy_digest(behavior)):
            raise ValueError("continuation behavior value model changed")
    for name in ("sage_checkpoint", "env_config", "motion_policy_config", "belief_calibration_path"):
        path = str(Path(protocol["parameters"][name]).resolve())
        if path not in stored_sources or file_hash(path) != stored_sources[path]:
            raise ValueError(f"continuation parent artifact changed: {name}")
        parent_artifacts[path] = stored_sources[path]
    protocol["parent_artifact_hashes"] = parent_artifacts
    audit_rows_path = root / "audit" / "windows.jsonl"
    audit_rows = {(row["case"], row["window"]): row for row in
                  (json.loads(line) for line in audit_rows_path.read_text().splitlines())}
    files = [root / "protocol.json", root / "manifest.json", audit_rows_path]
    if behavior_path:
        files.append(behavior_path)
    for path in sorted((root / "continuations").glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("behavior_value_sha256") != protocol.get("behavior_value_sha256"):
            raise ValueError("continuation group uses a different behavior value policy")
        if row["case"] not in protocol["cases"]:
            raise ValueError("continuation group belongs to an undeclared case")
        input_path = root / "inputs" / path.with_suffix(".npz").name
        with np.load(input_path, allow_pickle=False) as inputs:
            diagnostics = {name.removeprefix("score_component_"): inputs[name] for name in inputs.files
                           if name.startswith("score_component_")}
            planner_scores = np.asarray(inputs["decision_scores"]).copy()
            audit_path = root / "audit" / path.with_suffix(".npz").name
            with np.load(audit_path, allow_pickle=False) as audit:
                features, current_names = executable_value_features(
                    candidates=audit["planned_candidates"], headings=audit["planned_headings"],
                    robot_position=inputs["robot_position"], robot_velocity=inputs["robot_velocity"],
                    robot_goal=inputs["robot_goal"], robot_heading=float(inputs["robot_heading"]),
                    robot_radius=float(inputs["robot_radius"]), human_positions=inputs["human_positions"],
                    human_velocities=inputs["human_velocities"], human_radii=inputs["human_radii"],
                    beliefs=np.exp(inputs["beliefs"]), means=inputs["teacher_means"],
                    log_stds=inputs["teacher_log_stds"], diagnostics=diagnostics,
                    dt=protocol["dt"], execute_steps=protocol["execute_steps"],
                )
            if names is not None and names != current_names:
                raise ValueError("value feature contract changed between groups")
            names = current_names
            branches = row["branches"]
            indices = [branch["candidate"] for branch in branches]
            if len(set(indices)) != len(indices) or row["parent_selected"] not in indices:
                raise ValueError("continuation group must contain a unique factual parent action")
            reference = row.get("decision_reference_selected", row["parent_selected"])
            if reference not in indices:
                raise ValueError("continuation is missing the deployed base reference action")
            if behavior_path and reference != row["parent_selected"]:
                raise ValueError("policy-iteration targets must use the factual behavior reference action")
            if any(not branch["first_window_verified"] for branch in branches):
                raise ValueError("continuation first state must replay the parent exactly")
            targets = []
            for branch in branches:
                original_budget = min(protocol["parameters"]["max_steps"],
                                      protocol["parameters"]["windows"] * protocol["execute_steps"])
                budget_finished = branch["elapsed_steps"] + row["window"] * protocol["execute_steps"] >= original_budget
                if not branch["success"] and not branch["environment_timeout"] and not budget_finished:
                    raise ValueError("value target is censored before navigation termination")
                if target_schema == "reliability":
                    from sage.response.reliable_executable_value import reliability_targets

                    targets.append(reliability_targets(branch, float(inputs["robot_radius"])))
                    continue
                residual = 0. if branch["success"] else max(0., branch["terminal_goal_distance"] - float(inputs["robot_radius"])) / max(float(inputs["robot_preferred_speed"]), 1e-6)
                targets.append([branch["elapsed_time"] + residual,
                                1. - branch["collision_free_success"],
                                max(branch["collision"], branch["wall_collision"])])
        groups.append({"case": row["case"], "window": row["window"], "features": features[indices],
                       "parent": indices.index(reference), "candidate_indices": indices,
                       "planner_scores": planner_scores[indices],
                       "behavior_parent": indices.index(row["parent_selected"]),
                       "rerank_allowed": audit_rows[(row["case"], row["window"])]["status"] == "matched",
                       "diagnostics": {name: np.asarray(value).copy() for name, value in diagnostics.items()},
                       "response_risk_support": response_risk_admissible(diagnostics, reference)[indices],
                       "targets": np.asarray(targets), "names": names})
        files.extend([path, input_path, audit_path])
    if len(groups) != manifest["continuation_groups"] or {group["case"] for group in groups} != set(protocol["cases"]):
        raise ValueError("continuation group coverage differs from the completed manifest")
    if manifest["factual_continuation_checks"] < len(groups):
        raise ValueError("parent continuations have insufficient live verification")
    if target_schema == "reliability" and (protocol.get("continuation_outcome_schema") != "navigation-reliability-v1"
                                          or manifest.get("factual_terminal_checks", 0) < len(groups)):
        raise ValueError("reliability targets need live factual terminal outcome verification")
    for path in files:
        hashes[str(path)] = file_hash(path)
    return groups, protocol, hashes


def evaluate_value(model, groups, minimum_gain):
    records = []
    for group in groups:
        support = group["response_risk_support"] if model.get("response_risk_guard", False) else None
        selected, predictions = value_action(model, group["features"], group["parent"], minimum_gain, support)
        if not group["rerank_allowed"]:
            selected = group["parent"]
        difference = group["targets"][selected] - group["targets"][group["parent"]]
        records.append({"case": group["case"], "window": group["window"],
                        "selected": group["candidate_indices"][selected],
                        "parent_selected": group["candidate_indices"][group["parent"]],
                        "changed": int(selected != group["parent"]),
                        "time_difference": float(difference[0]), "failure_difference": float(difference[1]),
                        "collision_difference": float(difference[2]),
                        "predicted_time_difference": float(predictions[selected, 0])})
    cases = sorted({row["case"] for row in records})
    result = {"cases": len(cases), "groups": len(records), "changed_groups": sum(row["changed"] for row in records),
              "changed_cases": len({row["case"] for row in records if row["changed"]}), "records": records}
    for metric in ("time_difference", "failure_difference", "collision_difference"):
        means = np.asarray([np.mean([row[metric] for row in records if row["case"] == case]) for case in cases])
        indices = np.random.default_rng(70219).integers(0, len(means), size=(4000, len(means)))
        result[metric] = {"mean": float(means.mean()), "case_bootstrap_ci95": np.quantile(means[indices].mean(1), [.025, .975]).tolist()}
    result["added_failure_groups"] = sum(row["failure_difference"] > 0 for row in records)
    result["added_collision_groups"] = sum(row["collision_difference"] > 0 for row in records)
    return result


def main():
    parser = argparse.ArgumentParser(description="Fit observable executable-prefix values to verified terminal replanning outcomes.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--regularization", nargs="+", type=float, default=[.01, .1, 1., 10.])
    parser.add_argument("--minimum-gain", nargs="+", type=float, default=[.25, .5, 1.])
    parser.add_argument("--nonlinear-bandwidth", type=float)
    parser.add_argument("--feature-set", choices=("all", "cost", "prefix"), default="all")
    parser.add_argument("--response-risk-guard", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    train, train_protocol, hashes = load_continuation_groups(args.train_dir, "train")
    val, val_protocol, val_hashes = load_continuation_groups(args.val_dir, "val")
    hashes.update(val_hashes)
    for field in ("checkpoint_sha256", "case_splits", "seed", "dt", "execute_steps", "continuation_branches", "behavior_value_sha256"):
        if train_protocol.get(field) != val_protocol.get(field):
            raise ValueError(f"continuation protocols disagree on {field}")
    if {group["case"] for group in train} & {group["case"] for group in val}:
        raise ValueError("value training and validation scenes must be disjoint")
    sources = list(Path(__file__).resolve().parents[1].rglob("*.py"))
    hashes.update({str(path): file_hash(path) for path in sources})
    trials, selected = [], None
    feature_indices = None
    if args.feature_set == "cost":
        feature_indices = [train[0]["names"].index(name) for name in
                           ("weighted_task", "weighted_safety", "weighted_human", "weighted_risk", "observable_penalty")]
    elif args.feature_set == "prefix":
        feature_indices = prefix_value_feature_indices(train[0]["names"])
        if not feature_indices:
            raise ValueError("prefix-only value model has no prefix-consistent observable features")
    for regularization in args.regularization:
        model = fit_paired_value(train, regularization, args.nonlinear_bandwidth, feature_indices)
        model["response_risk_guard"] = args.response_risk_guard
        for minimum_gain in args.minimum_gain:
            evaluation = evaluate_value(model, val, minimum_gain)
            guard = evaluation["added_failure_groups"] == 0 and evaluation["added_collision_groups"] == 0
            eligible = guard and evaluation["changed_cases"] >= 2 and evaluation["time_difference"]["mean"] < -1e-8
            trial = {"regularization": regularization, "minimum_gain": minimum_gain, "eligible": eligible, "validation": evaluation}
            trials.append(trial)
            print(json.dumps({key: value for key, value in trial.items() if key != "validation"} | {
                "time_difference": evaluation["time_difference"], "changed_cases": evaluation["changed_cases"],
                "added_failure_groups": evaluation["added_failure_groups"]}), flush=True)
            if eligible and (selected is None or evaluation["time_difference"]["mean"] < selected[1]["time_difference"]["mean"]):
                selected = (model.copy(), evaluation, minimum_gain)
    changed = [path for path, digest in hashes.items() if not Path(path).exists() or file_hash(path) != digest]
    summary = {"status": "SOURCE_CHANGED" if changed else "COMPLETED", "changed_inputs": changed,
               "nonlinear_bandwidth": args.nonlinear_bandwidth,
               "feature_set": args.feature_set,
               "response_risk_guard": args.response_risk_guard,
               "eligible_model": selected is not None, "train_cases": train_protocol["cases"],
               "validation_cases": val_protocol["cases"], "trials": trials,
               "boundary": "Development-only selection using paired frozen-parent continuations; not proof of repeatedly executing the learned reranker or independent confirmation."}
    if selected is not None and not changed:
        model, evaluation, minimum_gain = selected
        model.update({"feature_names": train[0]["names"], "minimum_gain": minimum_gain,
                      "branch_count": train_protocol["continuation_branches"],
                      "response_checkpoint_sha256": train_protocol["checkpoint_sha256"],
                      "case_splits": train_protocol["case_splits"],
                      "parent_parameters": train_protocol["parameters"],
                      "parent_artifact_hashes": train_protocol["parent_artifact_hashes"],
                      "validation": evaluation, "source_and_input_hashes": hashes})
        model["training_behavior_value_sha256"] = train_protocol.get("behavior_value_sha256")
        if train_protocol.get("behavior_value_model"):
            model["reference_value_model"] = json.loads(Path(train_protocol["behavior_value_model"]).read_text())
        (output / "value_model.json").write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
        summary["selected_regularization"] = model["regularization"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if changed:
        raise RuntimeError("value training inputs changed")


if __name__ == "__main__":
    main()
