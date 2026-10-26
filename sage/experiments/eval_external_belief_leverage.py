from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from sage.experiments.calibrate_external_belief_cadence import _controller_args, validation_case_splits
from sage.experiments.calibrate_persistent_belief import load_belief_calibration
from sage.experiments.eval_persistent_closed_loop import _load_model
from sage.experiments.run_sicnav_tracer_loop import _run_episode
from sage.response.sage_candidate_adapter import SAGECandidateAdapter, SAGECandidateAdapterConfig


METRICS = (
    "collision_free_success", "collision", "task_progress", "total_time",
    "belief_action_change_rate", "belief_selection_change_rate", "belief_prediction_nll_gain",
)


def summarize(records):
    results = []
    for leverage in sorted({row["belief_cost_leverage"] for row in records}):
        selected = [row for row in records if row["belief_cost_leverage"] == leverage]
        policies = {
            policy: {metric: float(np.mean([row[metric] for row in selected if row["policy"] == policy]))
                     for metric in METRICS}
            for policy in ("tracer-uniform", "tracer-persistent")
        }
        paired = {(row["case"], row["policy"]): row for row in selected}
        contrasts = {}
        cases = sorted({row["case"] for row in selected})
        for metric in ("task_progress", "total_time", "collision"):
            differences = np.asarray([
                paired[case, "tracer-persistent"][metric] - paired[case, "tracer-uniform"][metric]
                for case in cases
            ])
            generator = np.random.default_rng(61731)
            draws = differences[generator.integers(0, len(cases), (10000, len(cases)))].mean(axis=1)
            contrasts[metric] = {"mean": float(differences.mean()),
                                 "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
                                 "independent_cases": len(cases)}
        results.append({"leverage": leverage, "policies": policies,
                        "persistent_minus_uniform": contrasts})
    return results


def main():
    parser = argparse.ArgumentParser(description="Validation-only belief decision-leverage ablation; never tunes test cases.")
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--leverage", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prefix-diversity", action="store_true")
    args = parser.parse_args()
    if not args.leverage or any(not np.isfinite(value) or value <= 0 for value in args.leverage):
        raise ValueError("leverage values must be positive and finite")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    checkpoint_path = Path(args.checkpoint).resolve()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = int(saved["seed"])
    parameters = json.loads(Path(args.source_metadata).read_text())["parameters"]
    splits = validation_case_splits(saved, parameters)
    calibration = load_belief_calibration(args.calibration, checkpoint_path,
                                         execute_steps=1, dt=0.25, num_robots=1)
    runtime = _controller_args(parameters, checkpoint_path, output, splits, seed, args.device)
    runtime.likelihood_scale = calibration["selected"]["likelihood_scale"]
    runtime.belief_selection_audit = True
    runtime.planner_sensitivity_audit = True
    runtime.belief_evidence_audit = False
    runtime.prefix_diversity = args.prefix_diversity
    source_paths = [Path(__file__).resolve(), *Path("sage").rglob("*.py"),
                    checkpoint_path, Path(args.calibration).resolve(), Path(args.source_metadata).resolve()]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    (output / "protocol.json").write_text(json.dumps({
        "phase": "validation", "seed": seed, "case_splits": splits, "parameters": vars(runtime),
        "leverage": args.leverage, "calibration": str(Path(args.calibration).resolve()),
        "source_hashes": hashes,
        "source_check": "all source changes are recorded; completion requires every imported local module to remain unchanged",
        "boundary": "development ablation on reused validation cases; extrapolated score correction is not a posterior or a calibrated risk probability; no promotion based on action flips",
    }, indent=2) + "\n")
    model = _load_model(checkpoint_path, args.device, expected_dt=0.25)
    adapter = SAGECandidateAdapter.from_checkpoint(
        Path(runtime.sage_checkpoint).resolve(), args.device,
        SAGECandidateAdapterConfig(
            num_candidates=runtime.num_candidates, guided=True, step_stride=runtime.sage_step_stride,
            goal_speed=runtime.sage_goal_speed, oversample_factor=runtime.sage_oversample_factor,
            task_prefilter=True, task_prefix_steps=runtime.execute_steps, goal_direction="task_goal",
            include_task_anchor=True, task_anchor_speed_scales=(0.45,), batch_size=runtime.num_candidates,
            guidance_step_clip_norm=0.1, diversity_weight=getattr(runtime, "sage_diversity_weight", 0.0),
        ),
    )
    records = []
    with (output / "windows.jsonl").open("x") as window_file:
        for leverage in args.leverage:
            runtime.belief_cost_leverage = leverage
            for case in splits["val"]:
                for policy in ("tracer-uniform", "tracer-persistent"):
                    print(f"validation seed={seed} leverage={leverage} case={case} policy={policy}", flush=True)
                    windows = []
                    record = _run_episode(model, adapter, policy, Path(runtime.external_root).resolve(),
                                          case, Path(runtime.env_config).resolve(), runtime, seed,
                                          window_audit_rows=windows)
                    record["belief_cost_leverage"] = leverage
                    records.append(record)
                    for row in windows:
                        window_file.write(json.dumps({"belief_cost_leverage": leverage, **row}) + "\n")
                    window_file.flush()
                    with (output / "episodes.csv").open("w") as file:
                        writer = csv.DictWriter(file, fieldnames=list(records[0]))
                        writer.writeheader()
                        writer.writerows(records)
    changed = [path for path, digest in hashes.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
    imported = {str(Path(module.__file__).resolve()) for module in list(sys.modules.values())
                if getattr(module, "__file__", None)}
    relevant_changed = [path for path in changed if str(Path(path).resolve()) in imported]
    report = {"status": "COMPLETED" if not relevant_changed else "SOURCE_CHANGED", "changed_sources": changed,
              "changed_imported_sources": relevant_changed,
              "results": summarize(records)}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if relevant_changed:
        raise RuntimeError("sources changed during the validation ablation")


if __name__ == "__main__":
    main()
