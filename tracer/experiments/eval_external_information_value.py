from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from tracer.experiments.calibrate_external_belief_cadence import _controller_args, validation_case_splits
from tracer.experiments.calibrate_persistent_belief import load_belief_calibration
from tracer.experiments.eval_persistent_closed_loop import _load_model
from tracer.experiments.external_information_value import DecisionInformationAudit
from tracer.experiments.run_sicnav_tracer_loop import _run_episode
from tracer.response.sage_candidate_adapter import SAGECandidateAdapter, SAGECandidateAdapterConfig


def information_evaluation_cases(splits, confirmation_start=None, confirmation_count=24):
    if confirmation_start is None:
        return "validation", list(splits["val"])
    if confirmation_start < 0 or confirmation_count < 1:
        raise ValueError("confirmation cases must be nonnegative with a positive count")
    cases = list(range(confirmation_start, confirmation_start + confirmation_count))
    disclosed = set().union(*(set(values) for values in splits.values()))
    if disclosed.intersection(cases):
        raise ValueError("new confirmation cases must be disjoint from all disclosed development and test cases")
    return "confirmation", cases


def summarize_information(records):
    matched = [row for row in records if row["status"] == "matched" and row.get("live_prefix_verified")]
    names = ["uniform", "persistent", "true_mode"]
    if any("candidate_supported" in row for row in records):
        names.append("candidate")
    comparable = [row for row in matched if all(row.get(f"{name}_supported", False) for name in names)]
    result = {"audited_windows": len(records), "matched_windows": len(matched),
              "all_selections_supported_windows": len(comparable),
              "independent_cases": len({row["case"] for row in comparable})}
    if not comparable:
        return result
    for name in (*names, "oracle"):
        result[f"{name}_action_changes_vs_uniform"] = sum(row[f"{name}_action_change_vs_uniform"] for row in comparable)
    result["uniform_positive_prefix_regret_windows"] = sum(row["uniform_prefix_regret"] > 1e-6 for row in comparable)
    comparisons = {}
    for name in (*names[1:], "oracle"):
        cases = sorted({row["case"] for row in comparable})
        per_case = [float(np.mean([row["uniform_cost"] - row[f"{name}_cost"] for row in comparable if row["case"] == case]))
                    for case in cases]
        generator = np.random.default_rng(61731)
        draws = np.asarray(per_case)[generator.integers(0, len(cases), (10000, len(cases)))].mean(axis=1)
        comparisons[name] = {"mean_cost_reduction_vs_uniform": float(np.mean(per_case)),
                             "case_bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
                             "per_case": dict(zip(map(str, cases), per_case))}
    result["cost_comparisons"] = comparisons
    if "candidate" in names:
        cases = sorted({row["case"] for row in comparable})
        changes = {}
        for metric in ("cost", "prefix_regret"):
            differences = np.asarray([np.mean([
                row[f"persistent_{metric}"] - row[f"candidate_{metric}"] for row in comparable if row["case"] == case
            ]) for case in cases])
            generator = np.random.default_rng(61731)
            draws = differences[generator.integers(0, len(cases), (10000, len(cases)))].mean(1)
            changes[metric] = {"parent_minus_candidate": float(differences.mean()),
                               "case_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist(),
                               "per_case": dict(zip(map(str, cases), differences.tolist()))}
        result["paired_model_comparison"] = changes
    metrics = ["decision_mode_accuracy", "decision_true_mode_probability", "motion_valid_fraction", "projected_fraction"]
    metrics += [f"{name}_{metric}" for name in names
                for metric in ("regret", "prefix_regret", "spearman", "horizon_collision", "horizon_wall_collision")]
    metrics += ["oracle_horizon_collision", "oracle_horizon_wall_collision"]
    result["case_averaged_metrics"] = {}
    for metric in metrics:
        means = []
        for case in sorted({row["case"] for row in comparable}):
            values = [row[metric] for row in comparable if row["case"] == case and row.get(metric) is not None]
            if values:
                means.append(float(np.mean(values)))
        result["case_averaged_metrics"][metric] = float(np.mean(means)) if means else None
    return result


def main():
    parser = argparse.ArgumentParser(description="Read-only true-mode and simulator-response information-value audit on validation cases.")
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--audit-stride", type=int, default=1)
    parser.add_argument("--prefix-diversity", action="store_true")
    parser.add_argument("--confirmation-case-start", type=int)
    parser.add_argument("--confirmation-cases", type=int, default=24)
    parser.add_argument("--case-subset", type=int, nargs="+")
    parser.add_argument("--case-splits")
    parser.add_argument("--comparison-checkpoint")
    parser.add_argument("--navigation-only", action="store_true")
    parser.add_argument("--executable-value-model")
    parser.add_argument("--candidate-debug-windows", type=int, default=0)
    parser.add_argument("--planner-score-tolerance", type=float, default=None)
    args = parser.parse_args()
    if args.audit_stride < 1:
        raise ValueError("audit stride must be positive")
    if args.navigation_only and args.comparison_checkpoint:
        raise ValueError("paired checkpoint information audit cannot be navigation-only")
    if args.executable_value_model and not args.navigation_only:
        raise ValueError("executable value deployment requires a separate actual-navigation evaluation")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    checkpoint_path = Path(args.checkpoint).resolve()
    protected = [checkpoint_path, Path(args.calibration).resolve(), Path(args.source_metadata).resolve()]
    if args.comparison_checkpoint:
        protected.append(Path(args.comparison_checkpoint).resolve())
    if args.executable_value_model:
        protected.append(Path(args.executable_value_model).resolve())
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = int(saved["seed"])
    parameters = json.loads(Path(args.source_metadata).read_text(encoding="utf-8"))["parameters"]
    splits = validation_case_splits(saved, parameters)
    if args.case_splits:
        from tracer.experiments.collect_external_planner_supervision import expanded_case_splits

        proposed_splits = json.loads(Path(args.case_splits).read_text(encoding="utf-8"))
        splits = expanded_case_splits(splits, proposed_splits)
    executable_value_model = None
    if args.executable_value_model:
        from tracer.experiments.collect_external_planner_supervision import expanded_case_splits
        executable_value_model = json.loads(Path(args.executable_value_model).read_text(encoding="utf-8"))
        if args.planner_score_tolerance is not None:
            if not np.isfinite(args.planner_score_tolerance) or args.planner_score_tolerance < 0.0:
                raise ValueError("planner score tolerance must be finite and nonnegative")
            from tracer.experiments.external_value_policy import value_policy_stages

            for stage in value_policy_stages(executable_value_model):
                if stage.get("reference_value_model") is not None:
                    stage["planner_score_tolerance"] = float(args.planner_score_tolerance)
        if executable_value_model["response_checkpoint_sha256"] != hashes[str(checkpoint_path)]:
            raise ValueError("executable value head must use its frozen response checkpoint")
        for path, digest in executable_value_model["parent_artifact_hashes"].items():
            if not Path(path).exists() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"executable value parent artifact changed: {path}")
        splits = expanded_case_splits(splits, executable_value_model["case_splits"])
    phase, evaluation_cases = information_evaluation_cases(splits, args.confirmation_case_start, args.confirmation_cases)
    if args.case_subset is not None:
        requested = list(args.case_subset)
        requested_set = set(requested)
        if not requested or len(requested_set) != len(requested):
            raise ValueError("case subset must be unique and nonempty")
        if requested_set.issubset(evaluation_cases):
            evaluation_cases = sorted(requested)
        elif args.confirmation_case_start is None and not requested_set.intersection(
            set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        ):
            phase, evaluation_cases = "explicit_subset", sorted(requested)
        else:
            raise ValueError("case subset must be in the selected evaluation split or be disjoint explicit cases")
    calibration = load_belief_calibration(args.calibration, checkpoint_path, execute_steps=1, dt=0.25, num_robots=1)
    runtime = _controller_args(parameters, checkpoint_path, output, splits, seed, args.device)
    runtime.cases = len(evaluation_cases)
    runtime.case_start = min(evaluation_cases)
    runtime.policies = ["tracer-persistent"]
    runtime.belief_calibration_path = str(Path(args.calibration).resolve())
    runtime.belief_cadence_source = "checkpoint_bound_validation_calibration"
    runtime.likelihood_scale = calibration["selected"]["likelihood_scale"]
    runtime.belief_selection_audit = True
    runtime.planner_sensitivity_audit = True
    runtime.belief_evidence_audit = False
    runtime.belief_cost_leverage = 1.0
    runtime.prefix_diversity = args.prefix_diversity
    runtime.candidate_debug_windows = args.candidate_debug_windows
    if executable_value_model is not None:
        from tracer.experiments.external_value_policy import validate_value_runtime

        validate_value_runtime(executable_value_model, runtime, hashes[str(checkpoint_path)])
        allowed_differences = {
            "output_dir", "cases", "case_start", "belief_selection_audit", "planner_sensitivity_audit",
            "belief_evidence_audit", "ranking_audit_windows", "candidate_debug_windows",
        }
        parent_parameters = executable_value_model["parent_parameters"]
        differences = [name for name in set(parent_parameters) | set(vars(runtime))
                       if name not in allowed_differences and parent_parameters.get(name) != vars(runtime).get(name)]
        if differences:
            raise ValueError(f"executable value runtime differs from its frozen parent: {differences}")
        runtime.belief_selection_audit = False
        runtime.planner_sensitivity_audit = False
    if runtime.hard_risk_filter:
        raise ValueError("reference protocol must have belief-independent hard safety masks")
    protected += [Path(runtime.sage_checkpoint).resolve(), Path(runtime.env_config).resolve(), Path(runtime.motion_policy_config).resolve()]
    source_paths = [*Path(__file__).resolve().parents[1].rglob("*.py"), *Path(runtime.external_root).rglob("*.py"),
                    *Path(runtime.external_root).parent.joinpath("python_rvo2_official").glob("*.so")]
    for path in [*protected, *source_paths]:
        hashes.setdefault(str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest())
    protocol = {
        "phase": phase, "evaluation_cases": evaluation_cases, "seed": seed, "case_splits": splits, "parameters": vars(runtime),
        "audit_stride": args.audit_stride, "source_hashes": hashes,
        "navigation_only": args.navigation_only,
        "executable_value_model": str(Path(args.executable_value_model).resolve()) if args.executable_value_model else None,
        "planner_score_tolerance": args.planner_score_tolerance,
        "executable_value_boundary": "Observable learned prefix-value reranking within the parent safe top-prefix shortlist; original response and belief updates unchanged. Actual closed-loop evaluation, not frozen-parent continuation validation. Original response score is retained in diagnostics and is not the learned time-value score.",
        "source_check": "all changes recorded; imported recorded modules and immutable checkpoint/config inputs must remain unchanged",
        "true_mode_semantics": saved["mode_semantics"],
        "comparison_checkpoint": str(Path(args.comparison_checkpoint).resolve()) if args.comparison_checkpoint else None,
        "comparison_checkpoint_semantics": "Offline only: candidate checkpoint uses the factual parent's current belief, state, candidates and safety support. It never controls the factual rollout.",
        "speed_thresholds_train_only": saved["speed_thresholds_train_only"],
        "boundary": "offline-only speed-bin labels and cloned futures never enter factual control or belief updates; confirmation cases must be prespecified and excluded from tuning; validation cases are reused development evidence",
        "oracle_objective": "same finite-horizon expected-cost formula evaluated on deterministic simulator responses and actual executor paths, with original observable penalties and common decision support; not episode navigation return",
        "oracle_reference": "stationary-robot simulator branch matching the model's reference convention; an intervention, not an offered feasible braking candidate",
        "prefix_regret": "best oracle tail among candidates sharing a selected executable prefix minus global oracle cost; optimistic finite-pool diagnostic, not a closed-loop policy value",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    model = _load_model(checkpoint_path, args.device, expected_dt=0.25)
    comparison_model = None
    if args.comparison_checkpoint:
        comparison_path = Path(args.comparison_checkpoint).resolve()
        comparison_checkpoint = torch.load(comparison_path, map_location="cpu", weights_only=False)
        contract_keys = set(saved["model_config"]) | set(comparison_checkpoint["model_config"])
        contract_keys.discard("candidate_uncertainty")
        if any(comparison_checkpoint["model_config"].get(name) != saved["model_config"].get(name) for name in contract_keys):
            raise ValueError("paired model audit requires identical observable-input and mode contracts")
        for name in ("speed_thresholds_train_only", "mode_semantics", "seed"):
            if comparison_checkpoint.get(name) != saved.get(name):
                raise ValueError(f"paired model audit requires aligned {name}")
        comparison_model = _load_model(comparison_path, args.device, expected_dt=.25)
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
    audit = DecisionInformationAudit(output / "audit", saved["speed_thresholds_train_only"], args.audit_stride,
                                     comparison_model=comparison_model)
    episodes = []
    for case in evaluation_cases:
        print(f"information audit seed={seed} case={case} stride={args.audit_stride}", flush=True)
        episodes.append(_run_episode(model, adapter, "tracer-persistent", Path(runtime.external_root).resolve(),
                                     case, Path(runtime.env_config).resolve(), runtime, seed,
                                     decision_audit=None if args.navigation_only else audit,
                                     executable_value_model=executable_value_model))
        with (output / "episodes.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
            writer.writeheader()
            writer.writerows(episodes)
        print(json.dumps(summarize_information(audit.records), allow_nan=False), flush=True)
    changed = [path for path, digest in hashes.items()
               if not Path(path).exists() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
    imported = {str(Path(module.__file__).resolve()) for module in list(sys.modules.values()) if getattr(module, "__file__", None)}
    required = imported | {str(path) for path in protected}
    relevant_changed = [path for path in changed if path in required]
    report = {"status": "SOURCE_CHANGED" if relevant_changed else "COMPLETED", "changed_sources": changed,
              "changed_required_sources": relevant_changed, "results": summarize_information(audit.records),
              "factual_navigation": {name: float(np.mean([row[name] for row in episodes]))
                                     for name in ("task_progress", "total_time", "collision", "collision_free_success")}}
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if relevant_changed:
        raise RuntimeError("required sources or inputs changed during the information audit")


if __name__ == "__main__":
    main()
