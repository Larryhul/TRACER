from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def paired_interval(parent, candidate):
    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(parent, dtype=np.float64)
    generator = np.random.default_rng(98137)
    samples = differences[generator.integers(0, len(differences), (20000, len(differences)))].mean(1)
    return {"parent": float(np.mean(parent)), "candidate": float(np.mean(candidate)),
            "candidate_minus_parent": float(differences.mean()),
            "case_bootstrap_ci95": np.quantile(samples, [.025, .975]).tolist()}


def summarize(root):
    root = Path(root).resolve()
    selection = json.loads((root / "confirmation_selection.json").read_text())
    for name in ("checkpoint", "calibration"):
        digest = hashlib.sha256(Path(selection[name]).read_bytes()).hexdigest()
        if digest != selection[f"{name}_sha256"]:
            raise ValueError(f"frozen confirmation {name} changed")
    expected = list(range(selection["case_start"], selection["case_start"] + selection["cases"]))
    summaries, protocols, episodes = {}, {}, {}
    for name in ("parent", "candidate"):
        directory = root / f"confirmation_{name}"
        summaries[name] = json.loads((directory / "summary.json").read_text())
        protocols[name] = json.loads((directory / "protocol.json").read_text())
        if summaries[name]["status"] != "COMPLETED" or summaries[name]["changed_required_sources"]:
            raise ValueError("confirmation requires completed immutable audits")
        if protocols[name]["phase"] != "confirmation" or protocols[name]["evaluation_cases"] != expected:
            raise ValueError("confirmation must use exactly the prespecified cases")
        if protocols[name]["seed"] != selection["seed"]:
            raise ValueError("confirmation model seed mismatch")
        with (directory / "episodes.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(expected) or sorted(int(row["case"]) for row in rows) != expected:
            raise ValueError("confirmation needs exactly one episode per independent case")
        if any(row["policy"] != "tracer-persistent" for row in rows):
            raise ValueError("confirmation must execute the actual Persistent controller")
        episodes[name] = {int(row["case"]): row for row in rows}
    if len({row["initial_scene_hash"] for row in episodes["parent"].values()}) != len(expected):
        raise ValueError("confirmation cases contain repeated initial scenes")
    for case in expected:
        for name in ("initial_scene_hash", "robot_kinematics", "robot_initial_heading", "environment_dt", "candidate_profile", "human_num"):
            if episodes["parent"][case][name] != episodes["candidate"][case][name]:
                raise ValueError(f"unmatched navigation contract: case={case} field={name}")
    differences = {name: {"parent": protocols["parent"]["parameters"].get(name),
                          "candidate": protocols["candidate"]["parameters"].get(name)}
                   for name in protocols["parent"]["parameters"]
                   if protocols["parent"]["parameters"].get(name) != protocols["candidate"]["parameters"].get(name)}
    allowed = {"checkpoint_dir", "external_response_checkpoint_dir", "output_dir", "belief_calibration_path", "likelihood_scale"}
    if set(differences) - allowed:
        raise ValueError(f"confirmation changed controller settings: {set(differences) - allowed}")
    metrics = ("success", "collision_free_success", "collision", "wall_collision", "timeout", "task_progress", "total_time",
               "belief_prediction_nll_gain", "belief_action_change_rate")
    navigation = {name: paired_interval(
        [float(episodes["parent"][case][name]) for case in expected],
        [float(episodes["candidate"][case][name]) for case in expected],
    ) for name in metrics}
    ranking = summaries["parent"]["results"]["paired_model_comparison"]
    selected_hash = selection["checkpoint_sha256"]
    for name in ("parent", "candidate"):
        if protocols[name]["source_hashes"].get(str(Path(selection["checkpoint"]).resolve())) != selected_hash:
            raise ValueError("both confirmation audits must protect the frozen candidate checkpoint")
    prediction = summaries["parent"]["results"]["case_averaged_metrics"]
    coverage = {name: summaries["parent"]["results"][name]
                for name in ("audited_windows", "matched_windows", "all_selections_supported_windows", "independent_cases")}
    if coverage["independent_cases"] != len(expected):
        raise ValueError("every confirmation scene must contribute a paired model audit")
    warnings = []
    if any(protocol["parameters"].get("policies") != ["tracer-persistent"] for protocol in protocols.values()):
        warnings.append("Frozen audit runtime parameters inherited the Uniform calibration scaffold's policies label; actual episode rows and observer windows execute Persistent. The current runner corrects this descriptive metadata field.")
    return {
        "status": "COMPLETED", "selection": selection, "unique_paired_scenes": len(expected),
        "controller_parameter_differences": differences, "audit_coverage": coverage,
        "fixed_parent_state_ranking": ranking,
        "fixed_parent_state_metrics": {name: prediction[name] for name in
                                       ("persistent_regret", "candidate_regret", "persistent_prefix_regret", "candidate_prefix_regret")},
        "actual_navigation": navigation,
        "candidate_on_policy_regret": summaries["candidate"]["results"]["case_averaged_metrics"]["persistent_regret"],
        "observed_safety_not_worse": all(navigation[name]["candidate"] <= navigation[name]["parent"]
                                        for name in ("collision", "wall_collision"))
                                     and navigation["collision_free_success"]["candidate"] >= navigation["collision_free_success"]["parent"],
        "ranking_directionally_improved": all(ranking[name]["parent_minus_candidate"] > 0 for name in ("cost", "prefix_regret")),
        "ranking_case_ci_positive": all(ranking[name]["case_bootstrap_ci95"][0] > 0 for name in ("cost", "prefix_regret")),
        "provenance_notes": warnings,
        "boundary": "One model seed, prespecified simulator scenes. Same-state finite-horizon oracle regret is not episode return. Observed safety is not a statistical noninferiority or universal-safety guarantee. No claim that Persistent beats Uniform or that multi-robot interaction improves.",
    }


def main():
    parser = argparse.ArgumentParser(description="Validate and summarize prespecified planner-alignment confirmation.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite a confirmation summary")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
