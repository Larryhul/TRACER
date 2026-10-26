import argparse
import json
from pathlib import Path

import numpy as np

from sage.experiments.closed_loop_protocol import audit_candidate_protocol
from sage.experiments.eval_component_factorial import FACTORIAL_POLICIES, audit_shared_emissions, summarize_factorial
from sage.experiments.formal_manifest import sha256_file


def read_evidence(path, hashes):
    path = Path(path).resolve()
    before = sha256_file(path)
    payload = json.loads(path.read_text())
    if sha256_file(path) != before or (str(path) in hashes and hashes[str(path)] != before):
        raise ValueError(f"evidence changed while reading: {path}")
    hashes[str(path)] = before
    return payload


def decision_contribution(policies, first, second):
    first_episodes = policies[first]["window_records"]
    second_episodes = policies[second]["window_records"]
    if len(first_episodes) != len(second_episodes):
        raise ValueError("paired decision episodes must align")
    differences, changed = [], []
    for first_windows, second_windows in zip(first_episodes, second_episodes, strict=True):
        if len(first_windows) != len(second_windows):
            raise ValueError("paired decision windows must align")
        for first_row, second_row in zip(first_windows, second_windows, strict=True):
            for key in ("window", "state_hash", "candidate_hash", "candidate_seed", "environment_rng_hash", "executed_candidate", "executed_response_hash"):
                if first_row[key] != second_row[key]:
                    raise ValueError(f"decision contribution requires shared factual states: {key}")
            differences.append(first_row["selected_true_cost"] - second_row["selected_true_cost"])
            changed.append(first_row["selected_candidate"] != second_row["selected_candidate"])
    differences = np.asarray(differences, dtype=float)
    changed = np.asarray(changed, dtype=bool)
    if not len(differences) or not np.isfinite(differences).all():
        raise ValueError("nonempty finite paired decision costs required")
    if np.any(np.abs(differences[~changed]) > 1e-8):
        raise ValueError("identical same-state selections must have identical costs")
    return {
        "windows": len(differences),
        "candidate_changes": int(changed.sum()),
        "candidate_change_fraction": float(changed.mean()),
        "beneficial_changes": int((differences < -1e-8).sum()),
        "harmful_changes": int((differences > 1e-8).sum()),
        "mean_cost_difference": float(differences.mean()),
        "mean_cost_difference_when_changed": float(differences[changed].mean()) if changed.any() else None,
        "scope": "Same-state candidate selections, not necessarily distinct executed prefixes; windows are not independent samples.",
    }


def audit_directory(directory):
    directory = Path(directory).resolve()
    hashes = {}
    summary = read_evidence(directory / "summary.json", hashes)
    if summary.get("status") != "COMPLETED" or summary.get("changed_sources") or summary.get("changed_checkpoints"):
        raise ValueError("completed unchanged factorial evaluation required")
    manifest = read_evidence(directory / "run_manifest.json", hashes)
    for category in ("sources", "checkpoints"):
        for path, expected in manifest[category].items():
            if sha256_file(Path(path)) != expected:
                raise ValueError(f"frozen input mismatch: {path}")
            hashes[str(Path(path).resolve())] = expected
    arguments = manifest["config"]["arguments"]
    names = [policy.name for policy in FACTORIAL_POLICIES]
    modes = arguments["evaluation_modes"]
    if set(modes) != {"same_state", "closed_loop"}:
        raise ValueError("attribution audit requires same-state and closed-loop evidence")
    runs = {mode: {} for mode in modes}
    audits, decisions = {}, {}
    for seed in arguments["seeds"]:
        decisions[str(seed)] = {}
        for mode in modes:
            runs[mode][str(seed)] = {}
            for split in arguments["splits"]:
                policies = read_evidence(directory / f"seed{seed}_{split}_{mode}.json", hashes)
                if set(policies) != set(names):
                    raise ValueError("all six factorial policies required")
                for policy in policies.values():
                    if len(policy["episode_records"]) != arguments["episodes"] or len(policy["window_records"]) != arguments["episodes"]:
                        raise ValueError("manifest episode count does not match evidence")
                    if any(len(rows) != manifest["config"]["config"]["simulation"]["windows"] for rows in policy["window_records"]):
                        raise ValueError("manifest window count does not match evidence")
                for name, policy in policies.items():
                    expected_weight = 0. if name.endswith("-additive") else summary["effective_settings"][str(seed)]["joint_weight"]
                    if any(not np.isclose(row["response_gate_weight"], expected_weight, rtol=0, atol=1e-7)
                           for rows in policy["window_records"] for row in rows):
                        raise ValueError("runtime Joint weight does not match effective settings")
                runs[mode][str(seed)][split] = policies
                audit = audit_candidate_protocol({str(seed): {split: policies}}, names, mode)
                if mode == "same_state":
                    audit["emissions"] = audit_shared_emissions(policies)
                    decisions[str(seed)][split] = {
                        "memory_at_joint": decision_contribution(policies, "persistent-joint", "reset-joint"),
                        "joint_at_persistent": decision_contribution(policies, "persistent-joint", "persistent-additive"),
                    }
                audits[f"{seed}/{split}/{mode}"] = audit
    additional = ("terminal_goal_distance", "terminal_goal_success") if manifest["config"]["config"]["simulation"].get("task_type") == "fixed_goal" else ()
    results = {mode: summarize_factorial(data, arguments["bootstrap_resamples"], additional_metrics=additional) for mode, data in runs.items()}
    if results != summary["results"]:
        raise ValueError("published summary does not match reconstructed episode evidence")
    settings = summary["effective_settings"]
    for seed in arguments["seeds"]:
        setting = settings[str(seed)]
        for template, field in (("calibration_template", "likelihood_scale"), ("residual_weight_calibration_template", "joint_weight")):
            if arguments.get(template):
                calibration = read_evidence(arguments[template].format(seed=seed), hashes)
                if calibration.get("status") != "COMPLETED":
                    raise ValueError("completed bound calibration required")
                selected = calibration["selected"]["weight" if field == "joint_weight" else field]
            else:
                selected = arguments[field]
            if setting[field] != selected:
                raise ValueError(f"recorded {field} differs from the supplied calibration")
    return {
        "status": "COMPLETED", "directory": str(directory), "phase": summary["phase"],
        "effective_settings": settings, "run_configuration": manifest["config"],
        "audits": audits, "decision_contributions": decisions,
        "results": results, "evidence_sha256": hashes,
        "claim_boundary": "Descriptive synthetic attribution on the recorded scenes, conditional on fixed response seeds; this audit does not establish universal or external superiority.",
    }, runs


def compare_inference_settings(parent, candidate):
    for key in ("simulation", "planning", "candidate_source"):
        if parent["run_configuration"]["config"][key] != candidate["run_configuration"]["config"][key]:
            raise ValueError(f"inference comparison changed {key}")
    for key in ("seeds", "splits", "episodes", "data_seed", "device", "num_threads"):
        if parent["run_configuration"]["arguments"][key] != candidate["run_configuration"]["arguments"][key]:
            raise ValueError(f"inference comparison changed {key}")
    for seed, setting in parent["effective_settings"].items():
        for key in ("planning_checkpoint", "joint_weight", "prefix_objective", "retain_previous_plan", "continuation_value_checkpoint", "continuation_value_weight"):
            if setting[key] != candidate["effective_settings"][seed][key]:
                raise ValueError(f"inference comparison changed {key}")


def compare_inference_only(parent_runs, candidate_runs):
    parent = parent_runs["closed_loop"]
    candidate = candidate_runs["closed_loop"]
    if parent.keys() != candidate.keys():
        raise ValueError("inference comparison model seeds must match")
    comparisons = 0
    for seed, splits in parent.items():
        if splits.keys() != candidate[seed].keys():
            raise ValueError("inference comparison domains must match")
        for split, policies in splits.items():
            for name, result in policies.items():
                updated = candidate[seed][split][name]
                for old_windows, new_windows in zip(result["window_records"], updated["window_records"], strict=True):
                    for key in ("state_hash", "candidate_hash", "environment_rng_hash"):
                        if old_windows[0][key] != new_windows[0][key]:
                            raise ValueError(f"inference comparison initial state mismatch: {key}")
                    for old, new in zip(old_windows, new_windows, strict=True):
                        for key in ("window", "candidate_seed", "environment_rng_hash"):
                            if old[key] != new[key]:
                                raise ValueError(f"inference comparison random streams mismatch: {key}")
                        if name.startswith("uniform-"):
                            for key in ("state_hash", "candidate_hash", "executed_candidate", "executed_response_hash", "selected_candidate", "selected_true_cost"):
                                if old[key] != new[key]:
                                    raise ValueError(f"inference-only change altered Uniform control: {key}")
                            comparisons += 1
    return {"status": "pass", "unchanged_uniform_windows": comparisons}


def paired_pipeline_contrasts(parent_runs, candidate_runs, resamples=4000):
    if resamples < 1:
        raise ValueError("positive bootstrap budget required")
    results = {}
    metrics = ("realized_total_cost", "episode_collision", "decision_prefix_nll", "planning_prefix_nll", "planning_prefix_ade", "belief_selection_changed")
    for mode, parent_seeds in parent_runs.items():
        results[mode] = {}
        seeds = list(parent_seeds)
        for split in parent_seeds[seeds[0]]:
            policies = {}
            for policy in FACTORIAL_POLICIES:
                contrasts = {}
                for metric in metrics:
                    differences, identifiers = [], None
                    for seed in seeds:
                        old = parent_seeds[seed][split][policy.name]["episode_records"]
                        new = candidate_runs[mode][seed][split][policy.name]["episode_records"]
                        old_ids = [row["episode"] for row in old]
                        if old_ids != [row["episode"] for row in new] or (identifiers is not None and old_ids != identifiers):
                            raise ValueError("pipeline contrast episodes must align")
                        identifiers = old_ids
                        differences.append([new_row[metric] - old_row[metric] for old_row, new_row in zip(old, new, strict=True)])
                    values = np.asarray(differences, dtype=float)
                    if not np.isfinite(values).all() or not values.size:
                        raise ValueError("finite nonempty paired metric values required")
                    scene_means = values.mean(0)
                    generator = np.random.default_rng(614203)
                    sampled = generator.integers(0, len(scene_means), size=(resamples, len(scene_means)))
                    contrasts[metric] = {
                        "mean_difference": float(scene_means.mean()),
                        "case_bootstrap_ci95": np.quantile(scene_means[sampled].mean(1), [.025, .975]).tolist(),
                        "per_model_seed": dict(zip(seeds, values.mean(1).tolist(), strict=True)),
                        "independent_scenes": len(scene_means),
                    }
                policies[policy.name] = contrasts
            results[mode][split] = policies
    return results


def main():
    parser = argparse.ArgumentParser(description="Reconstruct factorial evidence and isolate decision contributions")
    parser.add_argument("--directory", required=True)
    parser.add_argument("--inference-parent")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload, runs = audit_directory(args.directory)
    if args.inference_parent:
        parent, parent_runs = audit_directory(args.inference_parent)
        compare_inference_settings(parent, payload)
        payload["inference_parent"] = parent
        payload["inference_control_audit"] = compare_inference_only(parent_runs, runs)
        payload["candidate_minus_parent"] = paired_pipeline_contrasts(parent_runs, runs)
    for record in (payload, payload.get("inference_parent", {})):
        for path, expected in record.get("evidence_sha256", {}).items():
            if sha256_file(Path(path)) != expected:
                raise ValueError(f"evidence changed during audit: {path}")
    payload["audit_source_sha256"] = sha256_file(Path(__file__))
    output = Path(args.output)
    if output.exists():
        raise ValueError("audit output already exists; choose a new path")
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(output.resolve())}))


if __name__ == "__main__":
    main()
