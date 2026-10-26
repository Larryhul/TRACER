from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sage.experiments.collect_external_planner_supervision import file_hash
from sage.experiments.train_executable_value import evaluate_value, load_continuation_groups
from sage.response.executable_value import FEATURE_VERSION, fit_paired_value, prefix_value_feature_indices


def resample_scene_groups(groups, generator):
    cases = sorted({group["case"] for group in groups})
    sampled = generator.choice(cases, size=len(cases), replace=True)
    resampled = []
    for slot, original in enumerate(sampled):
        for group in groups:
            if group["case"] == original:
                resampled.append({**group, "case": slot, "original_case": original})
    return resampled


def main():
    parser = argparse.ArgumentParser(description="Fit scene-resampled executable values and abstain on uncertain action gains.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--members", type=int, default=24)
    parser.add_argument("--regularization", type=float, default=1.)
    parser.add_argument("--nonlinear-bandwidth", type=float, default=1.)
    parser.add_argument("--uncertainty-multiplier", type=float, nargs="+", default=[0., 1., 2.])
    parser.add_argument("--minimum-gain", type=float, nargs="+", default=[.05, .125, .25])
    args = parser.parse_args()
    if args.members < 8 or any(value < 0 for value in args.uncertainty_multiplier):
        raise ValueError("scene uncertainty requires at least8 members and nonnegative multipliers")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    train, train_protocol, hashes = load_continuation_groups(args.train_dir, "train")
    validation, validation_protocol, val_hashes = load_continuation_groups(args.val_dir, "val")
    hashes.update(val_hashes)
    for field in ("checkpoint_sha256", "case_splits", "seed", "dt", "execute_steps", "continuation_branches", "behavior_value_sha256"):
        if train_protocol.get(field) != validation_protocol.get(field):
            raise ValueError(f"ensemble training/validation protocol differs: {field}")
    if {group["case"] for group in train} & {group["case"] for group in validation}:
        raise ValueError("ensemble scene splits overlap")
    source_paths = list(Path(__file__).resolve().parents[1].rglob("*.py"))
    hashes.update({str(path): file_hash(path) for path in source_paths})
    indices = prefix_value_feature_indices(train[0]["names"])
    generator = np.random.default_rng(601943)
    members = []
    for index in range(args.members):
        sampled = resample_scene_groups(train, generator)
        members.append(fit_paired_value(sampled, args.regularization, args.nonlinear_bandwidth, indices))
        print(f"scene-bootstrap value member {index+1}/{args.members}", flush=True)
    trials, best = [], None
    for multiplier in args.uncertainty_multiplier:
        model = {"feature_version": FEATURE_VERSION, "ensemble": members, "uncertainty_multiplier": multiplier}
        for minimum_gain in args.minimum_gain:
            metrics = evaluate_value(model, validation, minimum_gain)
            eligible = metrics["added_failure_groups"] == 0 and metrics["added_collision_groups"] == 0 and metrics["changed_cases"] >= 2 and metrics["time_difference"]["mean"] < 0
            trial = {"uncertainty_multiplier": multiplier, "minimum_gain": minimum_gain, "eligible": eligible, "validation": metrics}
            trials.append(trial)
            print(json.dumps({"uncertainty_multiplier": multiplier,"minimum_gain":minimum_gain,
                              "eligible":eligible,"changed_cases":metrics["changed_cases"],
                              "time_difference":metrics["time_difference"],"added_failures":metrics["added_failure_groups"]}),flush=True)
            if eligible and (best is None or metrics["time_difference"]["mean"] < best[1]["time_difference"]["mean"]):
                best = (model.copy(), metrics, minimum_gain)
    changed = [path for path, digest in hashes.items() if not Path(path).exists() or file_hash(path) != digest]
    summary = {"status":"SOURCE_CHANGED" if changed else "COMPLETED","changed_inputs":changed,
               "eligible_model":best is not None,"members":args.members,"trials":trials,
               "boundary":"Train-scene bootstrap dispersion is an empirical abstention measure, not a calibrated guarantee. Selection validation only; no use of failed confirmation cases300–427."}
    if best is not None and not changed:
        model, metrics, minimum_gain = best
        model.update({"feature_names":train[0]["names"],"minimum_gain":minimum_gain,
                      "branch_count":train_protocol["continuation_branches"],
                      "response_checkpoint_sha256":train_protocol["checkpoint_sha256"],
                      "case_splits":train_protocol["case_splits"],"parent_parameters":train_protocol["parameters"],
                      "parent_artifact_hashes":train_protocol["parent_artifact_hashes"],
                      "source_and_input_hashes":hashes,"validation":metrics})
        model["training_behavior_value_sha256"] = train_protocol.get("behavior_value_sha256")
        if train_protocol.get("behavior_value_model"):
            model["reference_value_model"] = json.loads(Path(train_protocol["behavior_value_model"]).read_text())
        (output/'value_model.json').write_text(json.dumps(model,indent=2,allow_nan=False)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if changed:
        raise RuntimeError("ensemble sources or data changed")


if __name__ == '__main__':
    main()
