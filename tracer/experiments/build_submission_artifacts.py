"""Build submission tables and an evidence manifest from frozen experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LATEX_ROW_END = r" \\"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _fmt_optional(row: dict[str, str], field: str, digits: int = 4) -> str:
    value = row.get(field, "")
    return "-" if value in (None, "", "nan", "None") else _fmt(float(value), digits)


def _fmt_ci(row: dict[str, Any], digits: int = 4) -> str:
    return (
        f"{_fmt(row['mean'], digits)}"
        f" [{_fmt(row['bootstrap_ci95_low'], digits)},"
        f" {_fmt(row['bootstrap_ci95_high'], digits)}]"
    )


def _paired_csv_bootstrap(
    path: Path,
    split: str,
    first_policy: str,
    second_policy: str,
    field: str,
    seed: int,
    samples: int = 5000,
) -> tuple[float, float, float, int]:
    """Return second-minus-first paired mean and episode bootstrap interval."""

    if not path.exists():
        return float("nan"), float("nan"), float("nan"), 0
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split or row.get("policy") not in {
                first_policy,
                second_policy,
            }:
                continue
            key = (row.get("seed", ""), row.get("episode", ""))
            grouped.setdefault(key, {})[row["policy"]] = float(row[field])
    values = [
        pair[second_policy] - pair[first_policy]
        for pair in grouped.values()
        if first_policy in pair and second_policy in pair
    ]
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values)))
        for _ in range(samples)
    )
    low_index = int(0.025 * len(means))
    high_index = int(0.975 * len(means)) - 1
    return (
        statistics.fmean(values),
        means[low_index],
        means[high_index],
        len(values),
    )


def _table(
    path: Path,
    caption: str,
    label: str,
    columns: str,
    header: str,
    rows: list[str],
    size_command: str | None = None,
) -> None:
    content = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        *( [size_command] if size_command else [] ),
        "\\resizebox{\\columnwidth}{!}{%",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        header,
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def _matrix_contrast_summary(
    path: Path,
    contrast: str,
    split: str | None = None,
) -> tuple[int, float, int]:
    """Return cell count, mean contrast, and number of positive cells."""

    if not path.exists():
        return 0, float("nan"), 0
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [
        float(row["value"])
        for row in rows
        if row.get("contrast") == contrast
        and (split is None or row.get("split") == split)
        and row.get("value") not in (None, "", "nan")
    ]
    if not values:
        return 0, float("nan"), 0
    return len(values), statistics.fmean(values), sum(value > 0.0 for value in values)


def _matrix_mean(path: Path, field: str, method: str, split: str | None = None) -> tuple[int, float]:
    if not path.exists():
        return 0, float("nan")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [
        float(row[field])
        for row in rows
        if row.get("method") == method
        and (split is None or row.get("split") == split)
        and row.get(field) not in (None, "", "nan")
    ]
    return len(values), (statistics.fmean(values) if values else float("nan"))


def _paired_matrix_contrast(
    path: Path,
    first_method: str,
    second_method: str,
    field: str,
    key_fields: tuple[str, ...],
) -> tuple[int, float, int]:
    if not path.exists():
        return 0, float("nan"), 0
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(field_name, "") for field_name in key_fields)
        grouped.setdefault(key, {})[row.get("method", "")] = row.get(field, "nan")
    values = [
        float(methods[first_method]) - float(methods[second_method])
        for methods in grouped.values()
        if first_method in methods
        and second_method in methods
        and methods[first_method] not in (None, "", "nan")
        and methods[second_method] not in (None, "", "nan")
    ]
    return len(values), (statistics.fmean(values) if values else float("nan")), sum(
        value > 0.0 for value in values
    )


def _plot_h3_secondary(analysis: dict[str, Any], output: Path) -> None:
    metrics = (("realized_total_cost", "Cost"), ("task_progress", "Task progress"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=220)
    colors = {"id": "#276FBF", "ood": "#D1495B"}
    for axis, (metric, title) in zip(axes, metrics):
        for index, split in enumerate(("id", "ood")):
            row = next(
                row
                for row in analysis["contrasts"]
                if row["split"] == split
                and row["contrast"] == "recurrent-minus-persistent"
                and row["metric"] == metric
            )
            mean = float(row["mean"])
            low = mean - float(row["bootstrap_ci95_low"])
            high = float(row["bootstrap_ci95_high"]) - mean
            axis.errorbar(
                index,
                mean,
                yerr=[[low], [high]],
                fmt="o",
                color=colors[split],
                capsize=3,
                markersize=5,
                label=split.upper(),
            )
        axis.axhline(0.0, color="#666666", linewidth=0.8)
        axis.set_title(title)
        axis.set_xticks([0, 1], ["ID", "OOD"])
        axis.set_ylabel("Recurrent-joint minus Persistent-joint")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


TRACER_EXCLUDED_EXTERNAL_TABLES = (
    "external.tex",
    "external_sage.tex",
    "sdd_strong.tex",
    "eth_ucy_portability.tex",
)


def build(root: Path, output_dir: Path) -> dict[str, Any]:
    h1_path = root / "experiments/formal_v2/h1_tracer_formal/evaluation/summary.json"
    h2_analysis_path = root / "experiments/formal_v2/h2/persistent_trajectory_suite_tracer_effect/episode_analysis/analysis.json"
    h2_calibration_path = root / "reports/h2_calibration_2026-08-20/calibration.json"
    h2_prefix_path = root / "reports/h2_prefix_sensitivity_2026-08-21/prefix_summary.csv"
    h2_prefix_scale_path = root / "reports/h2_prefix_scale_sensitivity_2026-08-23/summary.json"
    h3_analysis_path = root / "experiments/formal_v2/h3/persistent_closed_loop_suite_tracer_effect_safetyfix/episode_analysis/analysis.json"
    h3_prior_replication_analysis_path = root / "experiments/pilot/h3_candidate_rescue_replication_20260819/episode_analysis/analysis.json"
    h3_replication_summary_path = root / "reports/h3_core_replication_20260827.json"
    h3_recurrent_analysis_path = root / "experiments/pilot/h3_recurrent_baseline_20260820/episode_analysis/analysis.json"
    h3_recurrent_summary_path = root / "experiments/pilot/h3_recurrent_baseline_20260820/summary.json"
    h3_mediation_path = root / "experiments/formal_v2/h3/persistent_closed_loop_suite_tracer_effect_safetyfix/summary.json"
    h3_safety_ablation_path = root / "reports/h3_safety_filter_ablation_20260902.md"
    legacy_h1_matrix_path = root / "experiments/causal_benchmark_matrix_20260823_mechanisms/summary.csv"
    corrected_h1_matrix_path = root / "experiments/causal_benchmark_matrix_20260824_corrected_shuffle/summary.csv"
    h1_matrix_path = (
        corrected_h1_matrix_path if corrected_h1_matrix_path.exists() else legacy_h1_matrix_path
    )
    continuous_intervention_path = root / "experiments/continuous_intervention_benchmark_20260824/summary.json"
    mode_switch_id_path = root / "experiments/mode_switch_benchmark_20260824_id/summary.json"
    mode_switch_ood_path = root / "experiments/mode_switch_benchmark_20260824_ood/summary.json"
    h2_matrix_path = root / "experiments/persistent_benchmark_matrix_20260824_coverage/summary.csv"
    h3_matrix_path = root / "experiments/closed_loop_benchmark_matrix_20260824_formal_scale/contrasts.csv"
    independent_formal_path = root / "experiments/independent_social_force_formal_20260902"
    independent_h1_path = independent_formal_path / "h1_response_ablation/summary.json"
    external_shadow_path = root / "experiments/external_tracer_shadow_replay_20260902/summary.json"
    independent_multientity_path = root / "experiments/independent_multientity_suite_20260902_final/summary.json"
    official_crowdnav_manifest_path = root / "experiments/tracer_crowdnav_20260903_full/manifest.json"
    official_sicnav_manifest_paths = (
        root / "experiments/tracer_sicnav_official_20260903_v3/manifest.json",
        root / "experiments/tracer_sicnav_official_20260903_v4/manifest.json",
    )
    official_sicnav_tracer_manifest_path = root / "experiments/tracer_sicnav_crowdsimplus_20260903_full/manifest.json"
    official_sicnav_same_protocol_manifest_path = root / "experiments/sicnav_same_protocol_20260904/manifest.json"
    official_socialgym2_manifest_path = root / "experiments/socialgym2_tracer_20260904_v5_aggregate/manifest.json"
    socialgym2_twoagent_sensitivity_report_path = root / "reports/socialgym2_tracer_twoagent_onehuman_20260904.md"
    atom_headless_manifest_path = root / "experiments/atom_headless_20260903/manifest.json"
    algames_headless_results_path = root / "experiments/lucidgames_algames_headless_20260903/results.csv"
    hard_safety_property_path = root / "experiments/tracer_hard_safety_property_20260903/summary.json"
    belief_cost_selection_path = root / "experiments/tracer_belief_cost_selection_20260903/manifest.json"
    external_baseline_audit_path = root / "experiments/external_baseline_audit_20260904_final_clean/audit.json"
    external_validation_holdout_report_path = root / "reports/external_validation_20260904_crowdsimplus_holdout.md"
    external_validation_holdout_manifest_path = root / "experiments/external_validation_dev/crowdsimplus_safety4_margin08_holdout_cases16_31_gpu_20260904/manifest.json"
    local_same_protocol_manifest_path = root / "experiments/tracer_gap_closure_20260903_mpogames_final/manifest.json"
    local_same_protocol_csv_path = root / "experiments/tracer_gap_closure_20260903_mpogames_final/episode_metrics.csv"
    clean_formal_chain_manifest_path = root / "experiments/formal_v2/clean_chain_submission_20260904/chain_manifest.json"
    multientity_system_manifest_path = root / "experiments/tracer_gap_closure_20260903_mpogames_final/manifest.json"
    external_crowdnav_holdout_report_path = root / "reports/external_validation_20260904_crowdnav_holdout.md"
    external_crowdnav_full_budget_report_path = root / "reports/external_validation_20260904_crowdnav_full_budget.md"
    external_crowdnav_full_budget_validation_manifest_path = root / "experiments/external_validation_dev/crowdnav_fullbudget_validation_cases0_15_20260904/manifest.json"
    external_crowdnav_full_budget_holdout_manifest_path = root / "experiments/external_validation_dev/crowdnav_fullbudget_holdout_cases16_31_20260904/manifest.json"
    external_crowdnav_full_budget_variants_manifest_path = root / "experiments/external_validation_dev/crowdnav_fullbudget_variants_holdout_cases16_31_20260904/manifest.json"
    external_crowdnav_holdout_manifest_candidates = (
        root / "experiments/external_validation_dev/crowdnav_holdout_cases16_31_cuda_20260904/manifest.json",
        root / "experiments/external_validation_dev/crowdnav_holdout_cases16_31_fixed_20260904/manifest.json",
    )
    external_crowdnav_holdout_manifest_path = next(
        (path for path in external_crowdnav_holdout_manifest_candidates if path.exists()),
        external_crowdnav_holdout_manifest_candidates[0],
    )
    external_kinematic_guard_report_path = root / "reports/external_kinematic_guard_ablation_20260904.md"
    external_candidate_budget_report_path = root / "reports/external_validation_20260904_crowdsimplus_candidate_budget.md"
    external_static_obstacle_guard_report_path = root / "reports/external_validation_20260904_crowdsimplus_static_guard.md"
    external_static_obstacle_holdout_manifest_path = root / "experiments/external_validation_dev/crowdsimplus_final_static010_safety4_margin08_holdout16_31_cuda_20260904/manifest.json"
    external_static_same_case_holdout_manifest_path = root / "experiments/sicnav_same_protocol_holdout_static010_20260904/manifest.json"
    external_multientity_scale_report_path = root / "reports/external_multientity_scale_20260904.md"
    external_multientity_scale_manifest_paths = tuple(
        root / f"experiments/external_multientity_scale_20260904_humans{human_num}/manifest.json"
        for human_num in (1, 3, 5)
    )
    external_dynamic_fallback_report_path = root / "reports/external_dynamic_fallback_holdout_20260904.md"
    external_dynamic_fallback_holdout_manifest_path = root / "experiments/external_dynamic_holdout_20260904_orca_fallback/manifest.json"
    external_dynamic_pure_holdout_manifest_path = root / "experiments/external_dynamic_holdout_20260904_goal_random/manifest.json"
    external_dynamic_same_case_manifest_path = root / "experiments/external_dynamic_samecase_holdout_20260904_orca_fallback/manifest.json"
    external_dynamic_sicnav_same_case_manifest_path = root / "experiments/external_dynamic_samecase_holdout_20260904_orca_fallback_sicnav_v2/manifest.json"
    external_dynamic_kinematic_holdout_report_path = root / "reports/external_dynamic_kinematic_holdout_20260904.md"
    external_dynamic_kinematic_holdout_manifest_path = root / "experiments/external_dynamic_holdout_20260904_hybrid_kin010/manifest.json"
    external_dynamic_validation_manifest_paths = tuple(
        root / "experiments" / name / "manifest.json"
        for name in (
            "external_dynamic_validation_20260904_cfg_static",
            "external_dynamic_validation_20260904_cfg_safety8",
            "external_dynamic_validation_20260904_cfg_margin10",
            "external_dynamic_validation_20260904_cfg_kin005",
            "external_dynamic_validation_20260904_cfg_kin010_prefix2",
            "external_dynamic_validation_20260904_cfg_static020",
            "external_dynamic_validation_20260904_cfg_orca_augmented",
            "external_dynamic_validation_20260904_cfg_orca_fallback",
        )
    )
    external_sage_sampling_report_path = root / "reports/external_validation_20260904/external_sage_sampling_sensitivity.md"
    external_eth_ucy_budget_report_path = root / "reports/external_validation_20260904/eth_ucy_sage_10epoch/summary.md"
    external_response_weight1_audit_path = root / "reports/external_validation_20260904/response_weight1_test_audit.md"
    submission_compile_audit_path = root / "reports/submission_compile_audit_20260904.md"

    h1 = _read(h1_path)
    h2_analysis = _read(h2_analysis_path)
    h2_calibration = _read(h2_calibration_path)
    h3_analysis = _read(h3_analysis_path)
    h3_replication = _read(h3_replication_summary_path)
    h3_recurrent_analysis = _read(h3_recurrent_analysis_path)
    h3_recurrent_summary = _read(h3_recurrent_summary_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for table_name in TRACER_EXCLUDED_EXTERNAL_TABLES:
        (tables_dir / table_name).unlink(missing_ok=True)

    nonlinear = h1["pooled_across_seeds"]["nonlinear"]
    h1_rows = [
        f"Matched additive $-$ joint NLL & {_fmt(nonlinear['matched_additive_minus_joint_nll']['mean'])} & "
        f"[{_fmt(nonlinear['matched_additive_minus_joint_nll']['mean_ci95'][0])}, {_fmt(nonlinear['matched_additive_minus_joint_nll']['mean_ci95'][1])}] & "
        f"{_fmt(nonlinear['matched_additive_minus_joint_nll']['dz'], 2)}{LATEX_ROW_END}",
        f"Interaction RMSE & {_fmt(nonlinear['matched_additive_minus_joint_interaction_rmse']['mean'])} & "
        f"[{_fmt(nonlinear['matched_additive_minus_joint_interaction_rmse']['mean_ci95'][0])}, {_fmt(nonlinear['matched_additive_minus_joint_interaction_rmse']['mean_ci95'][1])}] & "
        f"{_fmt(nonlinear['matched_additive_minus_joint_interaction_rmse']['dz'], 2)}{LATEX_ROW_END}",
        f"Shuffled $-$ joint NLL & {_fmt(nonlinear['shuffled_minus_joint_nll']['mean'])} & "
        f"[{_fmt(nonlinear['shuffled_minus_joint_nll']['mean_ci95'][0])}, {_fmt(nonlinear['shuffled_minus_joint_nll']['mean_ci95'][1])}] & "
        f"{_fmt(nonlinear['shuffled_minus_joint_nll']['dz'], 2)}{LATEX_ROW_END}",
    ]
    _table(
        tables_dir / "h1.tex",
        "H1 nonlinear joint-response evidence on the held-out paired-intervention corpus.",
        "tab:h1",
        "lccc",
        r"Contrast & Mean & 95\% CI & $d_z$ \\",
        h1_rows,
    )

    continuous_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Continuous action-support response benchmark on 25 joint candidate branches per held-out base scene.}",
        "\\label{tab:continuousintervention}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Mechanism & Variant & NLL & Effect RMSE & Effect rank $\\rho$ & Top-3 overlap \\\\",
        "\\midrule",
    ]
    if continuous_intervention_path.exists():
        continuous_payload = _read(continuous_intervention_path)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in continuous_payload["summary"]:
            grouped.setdefault((str(row["effect_mode"]), str(row["variant"])), []).append(row)
        for mode in ("none", "additive", "nonlinear"):
            for variant in ("additive", "team-concat", "joint", "shuffled-joint"):
                values = grouped.get((mode, variant))
                if not values:
                    continue
                def mean_field(field: str) -> float:
                    finite = [float(value[field]) for value in values if math.isfinite(float(value[field]))]
                    return statistics.fmean(finite) if finite else float("nan")
                def fmt_field(field: str, digits: int) -> str:
                    value = mean_field(field)
                    return "-" if not math.isfinite(value) else _fmt(value, digits)
                continuous_lines.append(
                    f"{mode.title()} & {variant} & {fmt_field('nll_mean', 3)} & "
                    f"{fmt_field('action_effect_rmse_mean', 3)} & "
                    f"{fmt_field('action_effect_rank_spearman_mean', 3)} & "
                    f"{fmt_field('action_effect_top3_mean', 3)}{LATEX_ROW_END}"
                )
    else:
        continuous_lines.append(f"Continuous action support & not run & - & - & - & -{LATEX_ROW_END}")
    continuous_lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table*}", ""))
    (tables_dir / "continuous_intervention.tex").write_text(
        "\n".join(continuous_lines), encoding="utf-8"
    )

    mode_switch_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Mode-switch re-entry audit. Persistent is evaluated without a change-point reset.}",
        "\\label{tab:modeswitch}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Split & Method & First post acc. & Delay & Success & NLL \\\\",
        "\\midrule",
    ]
    mode_switch_payloads = {
        "ID": _read(mode_switch_id_path) if mode_switch_id_path.exists() else None,
        "OOD": _read(mode_switch_ood_path) if mode_switch_ood_path.exists() else None,
    }
    mode_switch_row_count = 0
    for split, payload in mode_switch_payloads.items():
        if payload is None:
            continue
        for method in ("persistent", "reset", "recurrent", "no-belief"):
            row = payload["summary"].get(method)
            if row is None:
                continue
            mode_switch_row_count += 1
            mode_switch_lines.append(
                f"{split} & {method} & {_fmt(row['post_switch_first_window_accuracy'], 3)} & "
                f"{_fmt(row['adaptation_delay_windows'], 2)} & "
                f"{_fmt(row['adaptation_success_rate'], 3)} & "
                f"{_fmt(row['predictive_nll'], 3)}{LATEX_ROW_END}"
            )
    if mode_switch_row_count == 0:
        mode_switch_lines.append(f"Mode switch & not run & - & - & - & -{LATEX_ROW_END}")
    mode_switch_lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table}", ""))
    (tables_dir / "h2_mode_switch.tex").write_text(
        "\n".join(mode_switch_lines), encoding="utf-8"
    )

    policy_shift_path = root / "reports/external_validation_20260824/candidate_policy_shift/summary.json"
    policy_shift_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Held-out candidate-policy response generalization. Values are mean over five model seeds.}",
        "\\label{tab:policyshift}",
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Mode & Policy & Variant & NLL & Effect RMSE & Rank $\\rho$ & Top-1 \\\\",
        "\\midrule",
    ]
    if policy_shift_path.exists():
        policy_shift_payload = _read(policy_shift_path)
        lookup = {
            (str(row["mode"]), str(row["policy"]), str(row["variant"])): row
            for row in policy_shift_payload["summary"]
        }
        def policy_fmt(row: dict[str, Any], field: str, digits: int) -> str:
            value = float(row[field])
            return "-" if not math.isfinite(value) else _fmt(value, digits)
        for mode in policy_shift_payload["protocol"]["modes"]:
            for policy in policy_shift_payload["protocol"]["policies"]:
                for variant in ("additive", "joint", "team-concat"):
                    row = lookup.get((mode, policy, variant))
                    if row is None:
                        continue
                    policy_shift_lines.append(
                        f"{mode} & {policy} & {variant} & {policy_fmt(row, 'nll_mean', 3)} & "
                        f"{policy_fmt(row, 'effect_rmse_mean', 3)} & "
                        f"{policy_fmt(row, 'effect_rank_spearman_mean', 3)} & "
                        f"{policy_fmt(row, 'effect_top1_mean', 3)}{LATEX_ROW_END}"
                    )
    else:
        policy_shift_lines.append(f"Policy shift & not run & - & - & - & - & -{LATEX_ROW_END}")
    policy_shift_lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table*}", ""))
    (tables_dir / "candidate_policy_shift.tex").write_text(
        "\n".join(policy_shift_lines), encoding="utf-8"
    )

    h2_rows = []
    h2_lookup = {
        (row["split"], row["contrast"], row["metric"]): row
        for row in h2_analysis["contrasts"]
        if row.get("scope") in (None, "all-post-initial")
    }
    for split in ("id", "ood"):
        for contrast, name in (
            ("reset-minus-persistent", "Reset $-$ Persistent"),
            ("shuffled-id-minus-persistent", "Shuffled-ID $-$ Persistent"),
            ("recurrent-minus-persistent", "Recurrent $-$ Persistent"),
        ):
            row = h2_lookup[(split, contrast, "predictive_trajectory_nll")]
            h2_rows.append(
                f"{split.upper()} {name} & {_fmt(row['mean'])} & "
                f"[{_fmt(row['bootstrap_ci95_low'])}, {_fmt(row['bootstrap_ci95_high'])}] & "
                f"{_fmt(row['paired_dz'], 2)}{LATEX_ROW_END}")
    _table(
        tables_dir / "h2.tex",
        "H2 episode-level predictive NLL contrasts; positive values favor Persistent.",
        "tab:h2",
        "llcc",
        r"Split / contrast & Mean & 95\% CI & $d_z$ \\",
        h2_rows,
    )

    prefix_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{H2 executed-prefix sensitivity. Positive values favor Persistent.}",
        "\\label{tab:h2prefix}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Prefix & Split & Reset $-$ P. & Shuffled-ID $-$ P. & Recurrent $-$ P. \\\\",
        "\\midrule",
    ]
    if h2_prefix_path.exists():
        with h2_prefix_path.open(encoding="utf-8", newline="") as handle:
            prefix_rows = list(csv.DictReader(handle))
        prefix_lookup = {
            (row["prefix"], row["split"], row["contrast"]): row for row in prefix_rows
        }
        for prefix in ("1", "3", "6"):
            for split in ("id", "ood"):
                values = [
                    prefix_lookup[(prefix, split, contrast)]["mean"]
                    for contrast in (
                        "reset_minus_persistent_nll",
                        "shuffled_id_minus_persistent_nll",
                        "recurrent_minus_persistent_nll",
                    )
                ]
                prefix_lines.append(
                    f"{prefix} & {split.upper()} & "
                    + " & ".join(f"{float(value):.4f}" for value in values)
                    + LATEX_ROW_END
                )
    else:
        prefix_lines.append(f"Prefix sweep & not run & - & - & -{LATEX_ROW_END}")
    prefix_lines.extend(("\\bottomrule", "\\end{tabular}}", "\\end{table}", ""))
    (tables_dir / "h2_prefix.tex").write_text("\n".join(prefix_lines), encoding="utf-8")

    h3_rows = []
    for split in ("id", "ood"):
        for contrast, metric, label in (
            ("reset-minus-persistent", "realized_total_cost", "Reset $-$ Persistent cost"),
            ("persistent-minus-reset", "task_progress", "Persistent $-$ Reset progress"),
            ("task-only-minus-persistent", "realized_total_cost", "Task-only $-$ Persistent cost"),
        ):
            row = next(
                row
                for row in h3_analysis["contrasts"]
                if row["split"] == split and row["contrast"] == contrast and row["metric"] == metric
            )
            h3_rows.append(
                f"{split.upper()} {label} & {_fmt(row['mean'])} & "
                f"[{_fmt(row['bootstrap_ci95_low'])}, {_fmt(row['bootstrap_ci95_high'])}]{LATEX_ROW_END}")
    _table(
        tables_dir / "h3.tex",
        "H3 bounded TRACER closed-loop evidence after probabilistic safety filtering. Positive cost and progress contrasts favor Persistent.",
        "tab:h3",
        "llc",
        r"Split / contrast & Mean & 95\% CI \\",
        h3_rows,
    )

    mediation_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{H3 belief-to-selection mediation audit. Cost deltas are uniform-belief true cost minus Persistent-selected true cost.}",
        "\\label{tab:h3mediation}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Split & Flip rate & Cost delta & Oracle-regret delta \\\\",
        "\\midrule",
    ]
    if h3_mediation_path.exists():
        mediation_payload = _read(h3_mediation_path)
        for split in ("id", "ood"):
            contrasts = mediation_payload[split]["contrasts"]
            mediation_lines.append(
                f"{split.upper()} & {_fmt(contrasts['persistent_selection_flip_rate'], 3)} & "
                f"{_fmt(contrasts['persistent_selection_cost_delta'], 5)} & "
                f"{_fmt(contrasts['persistent_selection_oracle_regret_delta'], 5)}{LATEX_ROW_END}"
            )
    else:
        mediation_lines.append(f"Mediation & - & - & -{LATEX_ROW_END}")
    mediation_lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table}", ""))
    (tables_dir / "h3_mediation.tex").write_text(
        "\n".join(mediation_lines), encoding="utf-8"
    )

    h3_replication_rows = []
    for split in ("id", "ood"):
        contrasts = h3_replication["splits"][split]["contrasts"]
        for key, label in (
            ("task_only_minus_persistent_cost", "Task-only $-$ Persistent"),
            ("reset_minus_persistent_cost", "Reset $-$ Persistent"),
            ("recurrent_minus_persistent_cost", "Recurrent $-$ Persistent"),
            ("additive_minus_joint_cost", "Additive $-$ Joint"),
        ):
            row = contrasts[key]
            h3_replication_rows.append(
                f"{split.upper()} {label} & {_fmt(row['mean'])} & "
                f"[{_fmt(row['ci95'][0])}, {_fmt(row['ci95'][1])}]{LATEX_ROW_END}"
            )
    _table(
        tables_dir / "h3_replication.tex",
        "Independent-data-seed full-system H3 replication. Cost differences are first minus second; positive values favor the second policy.",
        "tab:h3replication",
        "llc",
        r"Split / realized-cost contrast & Mean & 95\% CI \\",
        h3_replication_rows,
    )

    recurrent_rows = []
    for split in ("id", "ood"):
        for metric, label in (
            ("realized_total_cost", "Cost"),
            ("task_progress", "Task progress"),
            ("human_cost", "Human cost"),
        ):
            row = next(
                row
                for row in h3_recurrent_analysis["contrasts"]
                if row["split"] == split
                and row["contrast"] == "recurrent-minus-persistent"
                and row["metric"] == metric
            )
            recurrent_rows.append(
                f"{split.upper()} {label} & {_fmt(row['mean'])} & "
                f"[{_fmt(row['bootstrap_ci95_low'])}, {_fmt(row['bootstrap_ci95_high'])}]{LATEX_ROW_END}")
    _table(
        tables_dir / "h3_recurrent.tex",
        "Secondary recurrent-joint baseline on an independent data seed; recurrent-joint minus Persistent-joint.",
        "tab:h3recurrent",
        "llc",
        r"Split / metric & Mean & 95\% CI \\",
        recurrent_rows,
    )

    calibration_rows = []
    for row in h2_calibration["aggregate"]:
        if row["method"] not in {"persistent", "recurrent"}:
            continue
        calibration_rows.append(
            f"{row['split'].upper()} {row['method']} & {_fmt(row['ece_mean'])} & "
            f"{_fmt(row['coverage_1sigma_mean'])} & {_fmt(row['coverage_2sigma_mean'])}{LATEX_ROW_END}")
    _table(
        tables_dir / "calibration.tex",
        "H2 calibration diagnostics from checkpoint-only replay. Lower ECE is better.",
        "tab:calibration",
        "lccc",
        r"Split / method & ECE & 1-$\sigma$ coverage & 2-$\sigma$ coverage \\",
        calibration_rows,
    )
    _plot_h3_secondary(
        h3_recurrent_analysis,
        output_dir / "figures" / "h3_recurrent_secondary.png",
    )

    submission_baseline_path = root / "reports/baseline_multiseed_2026-08-21/combined_results.csv"
    baseline_path = root / "experiments/baselines/synthetic_multiseed_20260820/results.csv"
    legacy_baseline_path = root / "experiments/baselines/synthetic_20260819/results.csv"
    if submission_baseline_path.exists():
        source_paths = [submission_baseline_path]
    else:
        source_paths = [path for path in (baseline_path, legacy_baseline_path) if path.exists()]
    if not source_paths:
        raise FileNotFoundError("no baseline comparison CSV is available")
    baseline_rows = []
    for source_path in source_paths:
        with source_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(source_path)
                baseline_rows.append(row)
    baseline_names = {
        "social_force",
        "constant_velocity",
        "constant_acceleration",
        "direct_goal",
        "trajectron_plus_plus",
        "mid",
    }
    selected_baselines: dict[str, list[dict[str, str]]] = {}
    for row in baseline_rows:
        name = row.get("baseline", "") or row.get("model_variant", "")
        if (
            not submission_baseline_path.exists()
            and baseline_path.exists()
            and row["_source"] != str(baseline_path)
            and name
            in {
                "constant_velocity",
                "constant_acceleration",
                "direct_goal",
                "social_force",
            }
        ):
            continue
        if name in baseline_names:
            selected_baselines.setdefault(name, []).append(row)
        elif row.get("model_variant") == "joint_heterogeneous_semantic":
            selected_baselines.setdefault("sage_medium", []).append(row)
    baseline_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Supporting navigation-family comparison on the controlled synthetic protocol. Values are mean $\\pm$ SD over available dataset seeds.}",
        "\\label{tab:baselines}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Method & $n$ & Robot ADE & Robot FDE & Entity ADE & Goal FDE & Goal success & R-E collision & R-R collision \\\\",
        "\\midrule",
    ]
    def baseline_cell(rows: list[dict[str, str]], field: str, digits: int) -> str:
        values = [float(row[field]) for row in rows]
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"${statistics.fmean(values):.{digits}f}\\pm{deviation:.{digits}f}$"

    for name, rows in selected_baselines.items():
        baseline_lines.append(
            f"{name.replace('_', ' ')} & {len(rows)} & {baseline_cell(rows, 'robot_ade', 3)} & "
            f"{baseline_cell(rows, 'robot_fde', 3)} & {baseline_cell(rows, 'entity_ade', 3)} & "
            f"{baseline_cell(rows, 'goal_fde', 3)} & {baseline_cell(rows, 'goal_success_rate', 3)} & "
            f"{baseline_cell(rows, 'robot_entity_collision_rate', 4)} & "
            f"{baseline_cell(rows, 'robot_robot_collision_rate', 4)}{LATEX_ROW_END}"
        )
    baseline_lines.extend(("\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""))
    (tables_dir / "baselines.tex").write_text(
        "\n".join(baseline_lines), encoding="utf-8"
    )

    if local_same_protocol_csv_path.exists():
        with local_same_protocol_csv_path.open(encoding="utf-8", newline="") as handle:
            local_rows = list(csv.DictReader(handle))
        policy_order = (
            "tracer-persistent",
            "tracer-hazard",
            "tracer-change-point",
            "tracer-forgetting",
            "tracer-reset",
            "mpogames-style",
            "orca-style",
            "mpc-style",
        )
        local_summary_rows = []
        for policy in policy_order:
            policy_rows = [row for row in local_rows if row.get("policy") == policy]
            if not policy_rows:
                continue

            def local_mean(field: str) -> float:
                return statistics.fmean(float(row[field]) for row in policy_rows)

            local_summary_rows.append(
                f"{policy.replace('-', ' ').title()} & {len(policy_rows)} & "
                f"{_fmt(local_mean('realized_total_cost'), 4)} & "
                f"{_fmt(local_mean('task_progress'), 3)} & "
                f"{_fmt(local_mean('collision_rate'), 3)} & "
                f"{_fmt(local_mean('hard_safe_fraction'), 3)} & "
                f"{_fmt(local_mean('hard_safety_fallback'), 3)}{LATEX_ROW_END}"
            )
        _table(
            tables_dir / "same_protocol_local_baselines.tex",
            "Same-protocol local multi-entity system baselines. Values average 720 episode rows per policy across five seeds and nine stress scenarios; these are style baselines, not official literature reimplementations.",
            "tab:sameprotocollocalbaselines",
            "lrrrrrr",
            f"Policy & $n$ & Cost & Progress & Collision & Hard-safe & Fallback{LATEX_ROW_END}",
            local_summary_rows,
            size_command="\\scriptsize",
        )
    baseline_coverage = {
        name: {
            "n": len(rows),
            "sources": sorted({row["_source"] for row in rows}),
        }
        for name, rows in selected_baselines.items()
    }

    external_current_method_path = (
        root / "reports/external_response_benchmark_20260901/summary.json"
    )
    if external_current_method_path.exists():
        external_current = _read(external_current_method_path)["datasets"]

        def external_contrast_row(
            dataset: str,
            label: str,
            key: str,
            *,
            scale: float = 1.0,
        ) -> str:
            comparison = external_current[dataset]["comparisons"][key]
            mean = float(comparison["mean"]) * scale
            low = float(comparison["ci95_low"]) * scale
            high = float(comparison["ci95_high"]) * scale
            directions = (
                f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']}"
            )
            dataset_labels = {
                "thor_magni": "THOR-MAGNI",
                "huron_sacson": "HuRoN/SACSoN",
            }
            return (
                f"{dataset_labels[dataset]} & {label} & {mean:.4f} & "
                f"[{low:.4f}, {high:.4f}] & {directions}{LATEX_ROW_END}"
            )

        external_current_rows = [
            external_contrast_row(
                "thor_magni",
                "Action NLL gain",
                "action_vs_history_nll",
            ),
            external_contrast_row(
                "thor_magni",
                "Reset $-$ Persistent NLL",
                "persistent_vs_reset_nll",
            ),
            external_contrast_row(
                "thor_magni",
                "Persistent $-$ shuffled acc. (pp)",
                "persistent_vs_shuffled_accuracy",
                scale=100.0,
            ),
            external_contrast_row(
                "thor_magni",
                "Recurrent $-$ Persistent acc. (pp)",
                "recurrent_vs_persistent_accuracy",
                scale=100.0,
            ),
            external_contrast_row(
                "huron_sacson",
                "Action NLL gain",
                "action_vs_history_nll",
            ),
            external_contrast_row(
                "huron_sacson",
                "Reset $-$ Persistent NLL",
                "persistent_vs_reset_nll",
            ),
        ]
        _table(
            tables_dir / "external_response_belief.tex",
            "Current-method observational degeneration on held-out recordings. "
            "Directions are named by each contrast; intervals are across five model seeds. "
            "These results support conditional association and recording-local continuity only.",
            "tab:external_response_belief",
            "llrrr",
            f"Dataset & Contrast & Mean & 95\\% CI & W/T/L{LATEX_ROW_END}",
            external_current_rows,
            size_command="\\scriptsize",
        )

    if independent_h1_path.exists() and (independent_formal_path / "h3_summary.json").exists():
        independent_h1 = _read(independent_h1_path)["aggregate"]
        independent_h2 = _read(independent_formal_path / "h2_summary.json")
        independent_h3 = _read(independent_formal_path / "h3_summary.json")
        independent_rows = [
            f"H1 ADE (additive $-$ joint) & {_fmt(independent_h1['additive_minus_joint']['ade']['mean'])} & {_fmt(independent_h1['additive_minus_joint']['ade']['std'])} & all 5{LATEX_ROW_END}",
            f"H1 FDE (additive $-$ joint) & {_fmt(independent_h1['additive_minus_joint']['fde']['mean'])} & {_fmt(independent_h1['additive_minus_joint']['fde']['std'])} & all 5{LATEX_ROW_END}",
            f"H1 NLL (additive $-$ joint) & {_fmt(independent_h1['additive_minus_joint']['nll']['mean'])} & {_fmt(independent_h1['additive_minus_joint']['nll']['std'])} & all 5{LATEX_ROW_END}",
            f"H2 test final mode accuracy (Persistent) & {_fmt(independent_h2['test']['persistent']['final_mode_accuracy'], 3)} & -- & --{LATEX_ROW_END}",
            f"H3 ID reset $-$ persistent cost & {_fmt(independent_h3['id']['contrasts']['reset_minus_persistent_total_cost'], 4)} & -- & bounded{LATEX_ROW_END}",
            f"H3 OOD reset $-$ persistent cost & {_fmt(independent_h3['ood']['contrasts']['reset_minus_persistent_total_cost'], 4)} & -- & bounded{LATEX_ROW_END}",
        ]
        _table(
            tables_dir / "independent_interaction.tex",
            "Independent nonlinear Social-Force interaction check. Positive H1 values favor the joint pair residual; H3 is a bounded integration diagnostic.",
            "tab:independentinteraction",
            "lccr",
            f"Metric & Mean & SD / note & Direction{LATEX_ROW_END}",
            independent_rows,
            size_command="\\scriptsize",
        )

    if independent_multientity_path.exists():
        multientity = _read(independent_multientity_path)
        h2_test = multientity["h2"]["summary"]["test"]
        mismatch = multientity["continuous_mismatch"]["aggregate"]
        baselines = multientity["baseline_scale"]["baselines"]
        scale = multientity["baseline_scale"]["scale"]
        h2_contrasts = multientity["h2"]["summary"]["test"]["episode_contrasts"]

        def value_sd(metric: dict[str, Any], digits: int = 4) -> str:
            return f"{_fmt(metric['mean'], digits)} $\\pm$ {_fmt(metric['std'], digits)}"

        h3_csv_path = independent_multientity_path.parent / "h3_multientity_episode.csv"
        h3_id_delta = _paired_csv_bootstrap(
            h3_csv_path, "id", "persistent-joint", "reset-joint",
            "realized_total_cost", 618731,
        )
        h3_ood_delta = _paired_csv_bootstrap(
            h3_csv_path, "ood", "persistent-joint", "reset-joint",
            "realized_total_cost", 618732,
        )

        multientity_rows = [
            f"H2 test final mode accuracy (Persistent) & {value_sd(h2_test['persistent']['final_mode_accuracy'], 3)} & -- & 640 episode contrasts{LATEX_ROW_END}",
            f"H2 test final mode accuracy (Reset) & {value_sd(h2_test['reset']['final_mode_accuracy'], 3)} & -- & matched control{LATEX_ROW_END}",
            f"H2 test final mode accuracy (Shuffled-ID) & {value_sd(h2_test['shuffled-id']['final_mode_accuracy'], 3)} & -- & identity control{LATEX_ROW_END}",
            f"H2 test Reset $-$ Persistent ADE & {_fmt(h2_contrasts['reset_minus_persistent_trajectory_ade']['mean'], 4)} & [{_fmt(h2_contrasts['reset_minus_persistent_trajectory_ade']['ci95_low'], 4)}, {_fmt(h2_contrasts['reset_minus_persistent_trajectory_ade']['ci95_high'], 4)}] & positive favors Persistent{LATEX_ROW_END}",
            f"H2 test Shuffled-ID $-$ Persistent ADE & {_fmt(h2_contrasts['shuffled_minus_persistent_trajectory_ade']['mean'], 4)} & [{_fmt(h2_contrasts['shuffled_minus_persistent_trajectory_ade']['ci95_low'], 4)}, {_fmt(h2_contrasts['shuffled_minus_persistent_trajectory_ade']['ci95_high'], 4)}] & positive favors Persistent{LATEX_ROW_END}",
            f"H3 ID reset $-$ Persistent cost & {_fmt(h3_id_delta[0], 4)} & [{_fmt(h3_id_delta[1], 4)}, {_fmt(h3_id_delta[2], 4)}] & {h3_id_delta[3]} paired episodes{LATEX_ROW_END}",
            f"H3 OOD reset $-$ Persistent cost & {_fmt(h3_ood_delta[0], 4)} & [{_fmt(h3_ood_delta[1], 4)}, {_fmt(h3_ood_delta[2], 4)}] & {h3_ood_delta[3]} paired episodes{LATEX_ROW_END}",
            f"Continuous mismatch additive $-$ joint ADE & {_fmt(mismatch['additive_minus_joint']['ade']['mean'], 4)} & [{_fmt(mismatch['additive_minus_joint']['ade']['ci95_low'], 4)}, {_fmt(mismatch['additive_minus_joint']['ade']['ci95_high'], 4)}] & positive favors joint{LATEX_ROW_END}",
            f"Constant-velocity response ADE & {_fmt(baselines['constant_velocity']['mean'], 4)} & [{_fmt(baselines['constant_velocity']['ci95_low'], 4)}, {_fmt(baselines['constant_velocity']['ci95_high'], 4)}] & independent baseline{LATEX_ROW_END}",
            f"Constant-acceleration response ADE & {_fmt(baselines['constant_acceleration']['mean'], 4)} & [{_fmt(baselines['constant_acceleration']['ci95_low'], 4)}, {_fmt(baselines['constant_acceleration']['ci95_high'], 4)}] & independent baseline{LATEX_ROW_END}",
            f"R=6 counterfactual finite rate & {_fmt(scale['R6']['finite_rate'], 3)} & -- & {scale['R6']['ms_per_counterfactual_episode']:.2f} ms/episode{LATEX_ROW_END}",
        ]
        _table(
            tables_dir / "independent_multientity.tex",
            "Independent multi-entity Social-Force robustness suite. H2 and H3 intervals are episode-level bootstrap intervals from paired frozen runs.",
            "tab:independentmultientity",
            "lccc",
            f"Metric & Value & 95\\% CI & Note{LATEX_ROW_END}",
            multientity_rows,
            size_command="\\scriptsize",
        )

    scale_path = root / "reports/synthetic_robot_scale_2026-08-21/scale_results.csv"
    dataset_sensitivity_path = root / "reports/sage_dataset_sensitivity_2026-08-21/dataset_sensitivity.json"
    scale_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Fixed-budget robot-count scale diagnostic on the controlled synthetic protocol.}",
        "\\label{tab:scale}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        r"Robots & Method & Robot ADE & Robot FDE & Goal success & R-E coll. & R-R coll. \\",
        "\\midrule",
    ]
    if scale_path.exists():
        with scale_path.open(encoding="utf-8", newline="") as handle:
            scale_rows = list(csv.DictReader(handle))
        method_order = {"sage": 0, "constant_velocity": 1, "direct_goal": 2}
        scale_rows.sort(
            key=lambda row: (
                int(row.get("num_robots", 0)),
                method_order.get(row.get("scale_method", ""), 99),
            )
        )
        for row in scale_rows:
            scale_lines.append(
                f"{row.get('num_robots', '-')} & {row.get('scale_method', '-').replace('_', ' ')} & "
                f"{_fmt_optional(row, 'robot_ade', 3)} & {_fmt_optional(row, 'robot_fde', 3)} & "
                f"{_fmt_optional(row, 'goal_success_rate', 3)} & "
                f"{_fmt_optional(row, 'robot_entity_collision_rate', 4)} & "
                f"{_fmt_optional(row, 'robot_robot_collision_rate', 4)}{LATEX_ROW_END}"
            )
    else:
        scale_lines.append(f"Scale sweep & not run & - & - & - & - & -{LATEX_ROW_END}")
    scale_lines.extend(("\\bottomrule", "\\end{tabular}}", "\\end{table}", ""))
    (tables_dir / "scale.tex").write_text("\n".join(scale_lines), encoding="utf-8")

    dataset_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{SAGE synthetic dataset-seed sensitivity. Values are mean $\\pm$ SD over model seeds.}",
        "\\label{tab:datasetsensitivity}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Dataset seed & Robot ADE & Robot FDE & Entity ADE & Goal success & R-E coll. & R-R coll. \\\\",
        "\\midrule",
    ]
    if dataset_sensitivity_path.exists():
        dataset_payload = _read(dataset_sensitivity_path)
        records = list(dataset_payload["by_dataset_seed"].items())
        records.append(("pooled", {"metrics": dataset_payload["pooled"]}))
        for dataset_seed, record in records:
            metrics = record["metrics"]
            dataset_lines.append(
                "{} & {}{:.3f}\\pm{:.3f}{} & {}{:.3f}\\pm{:.3f}{} & "
                "{}{:.3f}\\pm{:.3f}{} & {}{:.3f}\\pm{:.3f}{} & "
                "{}{:.4f}\\pm{:.4f}{} & {}{:.4f}\\pm{:.4f}{}{}".format(
                    dataset_seed,
                    "$",
                    metrics["robot_ade"]["mean"],
                    metrics["robot_ade"]["std"],
                    "$",
                    "$",
                    metrics["robot_fde"]["mean"],
                    metrics["robot_fde"]["std"],
                    "$",
                    "$",
                    metrics["entity_ade"]["mean"],
                    metrics["entity_ade"]["std"],
                    "$",
                    "$",
                    metrics["goal_success_rate"]["mean"],
                    metrics["goal_success_rate"]["std"],
                    "$",
                    "$",
                    metrics["robot_entity_collision_rate"]["mean"],
                    metrics["robot_entity_collision_rate"]["std"],
                    "$",
                    "$",
                    metrics["robot_robot_collision_rate"]["mean"],
                    metrics["robot_robot_collision_rate"]["std"],
                    "$",
                    LATEX_ROW_END,
                )
            )
    else:
        dataset_lines.append(f"Dataset sweep & - & - & - & - & - & -{LATEX_ROW_END}")
    dataset_lines.extend(("\\bottomrule", "\\end{tabular}}", "\\end{table}", ""))
    (tables_dir / "dataset_sensitivity.tex").write_text(
        "\n".join(dataset_lines), encoding="utf-8"
    )

    lofo_path = root / "reports/response_family_lofo_2026-08-21/lofo_summary.csv"
    lofo_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Response-family transfer diagnostic. Entries are matched-additive $-$ joint NLL; positive values favor the joint model.}",
        "\\label{tab:lofo}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        r"Train $\backslash$ Test & None & Additive & Nonlinear \\",
        "\\midrule",
    ]
    if lofo_path.exists():
        with lofo_path.open(encoding="utf-8", newline="") as handle:
            lofo_rows = list(csv.DictReader(handle))
        lookup = {
            (row.get("train_mode", ""), row.get("eval_mode", "")): row
            for row in lofo_rows
            if row.get("contrast") == "matched_additive_minus_joint_nll"
        }
        for train_mode in ("none", "additive", "nonlinear"):
            cells = []
            for eval_mode in ("none", "additive", "nonlinear"):
                row = lookup.get((train_mode, eval_mode))
                cells.append("-" if row is None else f"{float(row['mean']):.3f}")
            lofo_lines.append(f"{train_mode.title()} & {' & '.join(cells)}{LATEX_ROW_END}")
    else:
        lofo_lines.append(f"LOFO sweep & not run & - & -{LATEX_ROW_END}")
    lofo_lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table}", ""))
    (tables_dir / "lofo.tex").write_text("\n".join(lofo_lines), encoding="utf-8")

    h1_cells, h1_delta, h1_positive = _paired_matrix_contrast(
        h1_matrix_path,
        "matched-additive",
        "joint",
        "nll_mean",
        ("mode", "num_robots", "scene_family", "response_family"),
    )
    _, h1_shuffle, h1_shuffle_positive = _paired_matrix_contrast(
        h1_matrix_path,
        "shuffled-joint",
        "joint",
        "nll_mean",
        ("mode", "num_robots", "scene_family", "response_family"),
    )
    h1_r2_n, h1_r2 = _matrix_mean(
        h1_matrix_path, "interaction_r2_mean", "joint"
    )
    h2_rows = []
    for split in ("id", "ood"):
        reset_n, reset_mean, reset_positive = 0, float("nan"), 0
        # The matrix records store methods, so derive the paired NLL contrasts
        # from the complete cell rows rather than relying on a hand-written CSV.
        if h2_matrix_path.exists():
            with h2_matrix_path.open(encoding="utf-8", newline="") as handle:
                h2_rows_all = list(csv.DictReader(handle))
            by_cell = {}
            for row in h2_rows_all:
                if row.get("split") == split:
                    key = (
                        row.get("num_robots"), row.get("scene_family"),
                        row.get("response_family"),
                    )
                    by_cell.setdefault(key, {})[row.get("method")] = row
            contrasts = [
                float(methods["reset"]["predictive_trajectory_nll_mean"])
                - float(methods["persistent"]["predictive_trajectory_nll_mean"])
                for methods in by_cell.values()
                if "reset" in methods and "persistent" in methods
            ]
            reset_n = len(contrasts)
            reset_mean = statistics.fmean(contrasts) if contrasts else float("nan")
            reset_positive = sum(value > 0.0 for value in contrasts)
            recurrent_contrasts = [
                float(methods["recurrent"]["predictive_trajectory_nll_mean"])
                - float(methods["persistent"]["predictive_trajectory_nll_mean"])
                for methods in by_cell.values()
                if "recurrent" in methods and "persistent" in methods
            ]
            recurrent_mean = statistics.fmean(recurrent_contrasts) if recurrent_contrasts else float("nan")
            recurrent_positive = sum(value > 0.0 for value in recurrent_contrasts)
        else:
            recurrent_mean = float("nan")
            recurrent_positive = 0
        h2_rows.append(
            f"H2 {split.upper()} reset $-$ persistent & {reset_n} & {_fmt(reset_mean, 4)} & "
            f"{reset_positive}/{reset_n} & recurrent $-$ persistent: {_fmt(recurrent_mean, 4)} "
            f"({recurrent_positive}/{reset_n}){LATEX_ROW_END}"
        )
    h3_rows = []
    for split in ("id", "ood"):
        reset_n, reset_mean, reset_positive = _matrix_contrast_summary(
            h3_matrix_path, "reset_minus_persistent_total_cost", split
        )
        recurrent_n, recurrent_mean, recurrent_positive = _matrix_contrast_summary(
            h3_matrix_path, "recurrent_minus_persistent_total_cost", split
        )
        h3_rows.append(
            f"H3 {split.upper()} reset $-$ persistent cost & {reset_n} & {_fmt(reset_mean, 4)} & "
            f"{reset_positive}/{reset_n} & recurrent: {_fmt(recurrent_mean, 4)} "
            f"({recurrent_positive}/{recurrent_n}){LATEX_ROW_END}"
        )
    matrix_rows = [
        f"H1 matched-additive $-$ joint NLL & {h1_cells} & {_fmt(h1_delta, 4)} & "
        f"{h1_positive}/{h1_cells} & shuffle $-$ joint: {_fmt(h1_shuffle, 4)} "
        f"({h1_shuffle_positive}/{h1_cells}), interaction $R^2$: {_fmt(h1_r2, 3)}{LATEX_ROW_END}",
        *h2_rows,
        *h3_rows,
    ]
    _table(
        tables_dir / "matrix_robustness.tex",
        "Submission-scale simulation matrix audit. Positive counts indicate cells with the favorable direction; mixed directions are retained.",
        "tab:matrixrobustness",
        "lcccl",
        r"Audit & Cells & Mean & Favorable cells & Secondary diagnostic \\",
        matrix_rows,
        size_command="\\scriptsize",
    )

    evidence = {
        "generated_from": {
            "h1": str(h1_path.relative_to(root)),
            "h2_analysis": str(h2_analysis_path.relative_to(root)),
            "h2_calibration": str(h2_calibration_path.relative_to(root)),
            "h2_prefix_sensitivity": (
                str(h2_prefix_path.relative_to(root)) if h2_prefix_path.exists() else None
            ),
            "h2_prefix_scale_sensitivity": (
                str(h2_prefix_scale_path.relative_to(root))
                if h2_prefix_scale_path.exists()
                else None
            ),
            "h3_analysis": str(h3_analysis_path.relative_to(root)),
            "h3_replication_summary": str(h3_replication_summary_path.relative_to(root)),
            "h3_prior_replication_analysis": str(h3_prior_replication_analysis_path.relative_to(root)),
            "h3_recurrent_analysis": str(h3_recurrent_analysis_path.relative_to(root)),
            "h3_recurrent_summary": str(h3_recurrent_summary_path.relative_to(root)),
            "h3_recurrent_figure": "figures/h3_recurrent_secondary.png",
            "h3_safety_ablation": (
                str(h3_safety_ablation_path.relative_to(root))
                if h3_safety_ablation_path.exists()
                else None
            ),
            "baselines": [str(path.relative_to(root)) for path in source_paths],
            "robot_scale": (
                str(scale_path.relative_to(root)) if scale_path.exists() else None
            ),
            "dataset_sensitivity": (
                str(dataset_sensitivity_path.relative_to(root))
                if dataset_sensitivity_path.exists()
                else None
            ),
            "response_family_lofo": (
                str(lofo_path.relative_to(root)) if lofo_path.exists() else None
            ),
            "external_current_method": (
                str(external_current_method_path.relative_to(root))
                if external_current_method_path.exists()
                else None
            ),
            "h1_matrix": str(h1_matrix_path.relative_to(root)) if h1_matrix_path.exists() else None,
            "continuous_intervention": (
                str(continuous_intervention_path.relative_to(root))
                if continuous_intervention_path.exists()
                else None
            ),
            "h2_mode_switch_id": (
                str(mode_switch_id_path.relative_to(root))
                if mode_switch_id_path.exists()
                else None
            ),
            "h2_mode_switch_ood": (
                str(mode_switch_ood_path.relative_to(root))
                if mode_switch_ood_path.exists()
                else None
            ),
            "h3_mediation": (
                str(h3_mediation_path.relative_to(root))
                if h3_mediation_path.exists()
                else None
            ),
            "candidate_policy_shift": (
                str(policy_shift_path.relative_to(root))
                if policy_shift_path.exists()
                else None
            ),
            "h2_matrix": str(h2_matrix_path.relative_to(root)) if h2_matrix_path.exists() else None,
            "h3_matrix": str(h3_matrix_path.relative_to(root)) if h3_matrix_path.exists() else None,
            "independent_interaction": (
                str(independent_formal_path.relative_to(root))
                if independent_formal_path.exists()
                else None
            ),
            "independent_h1_ablation": (
                str(independent_h1_path.relative_to(root))
                if independent_h1_path.exists()
                else None
            ),
            "external_tracer_shadow_replay": (
                str(external_shadow_path.relative_to(root))
                if external_shadow_path.exists()
                else None
            ),
            "official_crowdnav_tracer": (
                str(official_crowdnav_manifest_path.relative_to(root))
                if official_crowdnav_manifest_path.exists()
                else None
            ),
            "official_sicnav_baseline": [
                str(path.relative_to(root))
                for path in official_sicnav_manifest_paths
                if path.exists()
            ],
            "official_sicnav_tracer_loop": (
                str(official_sicnav_tracer_manifest_path.relative_to(root))
                if official_sicnav_tracer_manifest_path.exists()
                else None
            ),
            "official_sicnav_same_protocol": (
                str(official_sicnav_same_protocol_manifest_path.relative_to(root))
                if official_sicnav_same_protocol_manifest_path.exists()
                else None
            ),
            "official_socialgym2_tracer_long_horizon": (
                str(official_socialgym2_manifest_path.relative_to(root))
                if official_socialgym2_manifest_path.exists()
                else None
            ),
            "socialgym2_twoagent_sensitivity_report": (
                str(socialgym2_twoagent_sensitivity_report_path.relative_to(root))
                if socialgym2_twoagent_sensitivity_report_path.exists()
                else None
            ),
            "official_atom_headless_core": (
                str(atom_headless_manifest_path.relative_to(root))
                if atom_headless_manifest_path.exists()
                else None
            ),
            "official_algames_headless_core": (
                str(algames_headless_results_path.relative_to(root))
                if algames_headless_results_path.exists()
                else None
            ),
            "same_protocol_local_baselines": (
                str(local_same_protocol_manifest_path.relative_to(root))
                if local_same_protocol_manifest_path.exists()
                else None
            ),
            "external_baseline_audit": (
                str(external_baseline_audit_path.relative_to(root))
                if external_baseline_audit_path.exists()
                else None
            ),
            "external_validation_holdout_report": (
                str(external_validation_holdout_report_path.relative_to(root))
                if external_validation_holdout_report_path.exists()
                else None
            ),
            "external_validation_holdout_manifest": (
                str(external_validation_holdout_manifest_path.relative_to(root))
                if external_validation_holdout_manifest_path.exists()
                else None
            ),
            "hard_safety_property": (
                str(hard_safety_property_path.relative_to(root))
                if hard_safety_property_path.exists()
                else None
            ),
            "belief_cost_selection": (
                str(belief_cost_selection_path.relative_to(root))
                if belief_cost_selection_path.exists()
                else None
            ),
            "clean_formal_chain": (
                str(clean_formal_chain_manifest_path.relative_to(root))
                if clean_formal_chain_manifest_path.exists()
                else None
            ),
            "external_crowdnav_holdout_report": (
                str(external_crowdnav_holdout_report_path.relative_to(root))
                if external_crowdnav_holdout_report_path.exists()
                else None
            ),
            "external_crowdnav_holdout_manifest": (
                str(external_crowdnav_holdout_manifest_path.relative_to(root))
                if external_crowdnav_holdout_manifest_path.exists()
                else None
            ),
            "external_crowdnav_full_budget_report": (
                str(external_crowdnav_full_budget_report_path.relative_to(root))
                if external_crowdnav_full_budget_report_path.exists()
                else None
            ),
            "external_crowdnav_full_budget_validation_manifest": (
                str(external_crowdnav_full_budget_validation_manifest_path.relative_to(root))
                if external_crowdnav_full_budget_validation_manifest_path.exists()
                else None
            ),
            "external_crowdnav_full_budget_holdout_manifest": (
                str(external_crowdnav_full_budget_holdout_manifest_path.relative_to(root))
                if external_crowdnav_full_budget_holdout_manifest_path.exists()
                else None
            ),
            "external_crowdnav_full_budget_variants_manifest": (
                str(external_crowdnav_full_budget_variants_manifest_path.relative_to(root))
                if external_crowdnav_full_budget_variants_manifest_path.exists()
                else None
            ),
            "external_kinematic_guard_ablation": (
                str(external_kinematic_guard_report_path.relative_to(root))
                if external_kinematic_guard_report_path.exists()
                else None
            ),
            "external_candidate_budget": (
                str(external_candidate_budget_report_path.relative_to(root))
                if external_candidate_budget_report_path.exists()
                else None
            ),
            "external_static_obstacle_guard": (
                str(external_static_obstacle_guard_report_path.relative_to(root))
                if external_static_obstacle_guard_report_path.exists()
                else None
            ),
            "external_static_obstacle_holdout_manifest": (
                str(external_static_obstacle_holdout_manifest_path.relative_to(root))
                if external_static_obstacle_holdout_manifest_path.exists()
                else None
            ),
            "external_static_same_case_holdout_manifest": (
                str(external_static_same_case_holdout_manifest_path.relative_to(root))
                if external_static_same_case_holdout_manifest_path.exists()
                else None
            ),
            "external_multientity_scale_report": (
                str(external_multientity_scale_report_path.relative_to(root))
                if external_multientity_scale_report_path.exists()
                else None
            ),
            "external_multientity_scale_manifests": [
                str(path.relative_to(root))
                for path in external_multientity_scale_manifest_paths
                if path.exists()
            ],
            "external_dynamic_fallback_report": (
                str(external_dynamic_fallback_report_path.relative_to(root))
                if external_dynamic_fallback_report_path.exists()
                else None
            ),
            "external_dynamic_fallback_holdout_manifest": (
                str(external_dynamic_fallback_holdout_manifest_path.relative_to(root))
                if external_dynamic_fallback_holdout_manifest_path.exists()
                else None
            ),
            "external_dynamic_pure_holdout_manifest": (
                str(external_dynamic_pure_holdout_manifest_path.relative_to(root))
                if external_dynamic_pure_holdout_manifest_path.exists()
                else None
            ),
            "external_dynamic_same_case_manifest": (
                str(external_dynamic_same_case_manifest_path.relative_to(root))
                if external_dynamic_same_case_manifest_path.exists()
                else None
            ),
            "external_dynamic_sicnav_same_case_manifest": (
                str(external_dynamic_sicnav_same_case_manifest_path.relative_to(root))
                if external_dynamic_sicnav_same_case_manifest_path.exists()
                else None
            ),
            "external_dynamic_kinematic_holdout_report": (
                str(external_dynamic_kinematic_holdout_report_path.relative_to(root))
                if external_dynamic_kinematic_holdout_report_path.exists()
                else None
            ),
            "external_dynamic_kinematic_holdout_manifest": (
                str(external_dynamic_kinematic_holdout_manifest_path.relative_to(root))
                if external_dynamic_kinematic_holdout_manifest_path.exists()
                else None
            ),
            "external_dynamic_validation_manifests": [
                str(path.relative_to(root))
                for path in external_dynamic_validation_manifest_paths
                if path.exists()
            ],
            "external_sage_sampling_sensitivity": (
                str(external_sage_sampling_report_path.relative_to(root))
                if external_sage_sampling_report_path.exists()
                else None
            ),
            "external_eth_ucy_budget_audit": (
                str(external_eth_ucy_budget_report_path.relative_to(root))
                if external_eth_ucy_budget_report_path.exists()
                else None
            ),
            "external_response_weight1_audit": (
                str(external_response_weight1_audit_path.relative_to(root))
                if external_response_weight1_audit_path.exists()
                else None
            ),
            "submission_compile_audit": (
                str(submission_compile_audit_path.relative_to(root))
                if submission_compile_audit_path.exists()
                else None
            ),
            "multientity_system_matrix": (
                str(multientity_system_manifest_path.relative_to(root))
                if multientity_system_manifest_path.exists()
                else None
            ),
            "independent_multientity_suite": (
                str(independent_multientity_path.relative_to(root))
                if independent_multientity_path.exists()
                else None
            ),
            "independent_multientity_h3_episode_csv": (
                str((independent_multientity_path.parent / "h3_multientity_episode.csv").relative_to(root))
                if (independent_multientity_path.parent / "h3_multientity_episode.csv").exists()
                else None
            ),
            "independent_multientity_h2_rows": sorted(
                str(path.relative_to(root))
                for path in (independent_multientity_path.parent / "h2").glob("*_rows.csv")
            ) if (independent_multientity_path.parent / "h2").exists() else [],
        },
        "baseline_coverage": baseline_coverage,
        "claims": {
            "h1": "supported nonlinear joint-response structure under the paired synthetic intervention protocol",
            **({
                "continuous_action_support": "continuous unseen team-action interpolation and response ranking diagnostic",
            } if continuous_intervention_path.exists() else {}),
            "h2": "supported predictive benefit of identity-persistent belief over reset and shuffled-ID controls",
            "h2_prefix_sensitivity": "executed-prefix sensitivity diagnostic; formal claim is restricted to the three-step prefix",
            "h2_prefix_scale_sensitivity": "post-hoc long-prefix likelihood-scale stability diagnostic; not a Formal-v1 gate",
            **({
                "h2_mode_switch": "mode-switch re-entry audit; persistent filtering is not claimed to dominate change-point reset",
            } if mode_switch_id_path.exists() and mode_switch_ood_path.exists() else {}),
            **({
                "h3_mediation": "direct belief-to-selection mediation audit; selection changes are reported without a superiority claim",
            } if h3_mediation_path.exists() else {}),
            **({
                "candidate_policy_shift": "held-out candidate-policy response generalization diagnostic; not a closed-loop superiority claim",
            } if policy_shift_path.exists() else {}),
            "h3": "bounded candidate-rescue integration and task non-inferiority; not universal superiority",
            "h3_replication": "independent full-system data-seed replication with all eight policies; confirms bounded integration but not Persistent or joint planner superiority",
            "h3_recurrent": "secondary independent recurrent baseline result with a human-cost trade-off",
            **({
                "h3_safety_ablation": "secondary probabilistic safety-filter ablation with bounded risk/task trade-offs",
            } if h3_safety_ablation_path.exists() else {}),
            "robot_scale": "fixed-budget single-/multi-robot stress diagnostic, not a superiority gate",
            "dataset_sensitivity": "synthetic dataset-seed sensitivity diagnostic for SAGE, not a Formal-v1 gate",
            "response_family_lofo": "controlled synthetic response-family transfer diagnostic, not a real-human claim",
            "external_current_method": "single-robot held-out-recording response and identity-continuity benchmark; conditional association only, with mixed action directions and the stronger recurrent control retained",
            "matrix_robustness": "submission-scale synthetic matrix audit with mixed closed-loop directions retained; not universal superiority",
            **({
                "official_crowdnav_tracer": "official CrowdNav single-robot full TRACER loop; finite third-party integration evidence, not multi-robot superiority",
            } if official_crowdnav_manifest_path.exists() else {}),
            **({
                "official_sicnav_baseline": "official SICNav policies on official CrowdSimPlus; standalone single-robot baseline, not same-protocol TRACER",
            } if any(path.exists() for path in official_sicnav_manifest_paths) else {}),
            **({
                "official_sicnav_tracer_loop": "official CrowdSimPlus single-robot TRACER response-belief-planning execution with ORCA+ humans; integration evidence, not superiority",
            } if official_sicnav_tracer_manifest_path.exists() else {}),
            **({
                "official_sicnav_same_protocol": "official CrowdSimPlus ORCA/MPC/DWA/Linear controls compared with TRACER on the same fixed cases and time budget; negative superiority result retained",
            } if official_sicnav_same_protocol_manifest_path.exists() else {}),
            **({
                "official_socialgym2_tracer_long_horizon": "official SocialGym2 ROS headless TRACER long-horizon integration under partial observation; discrete action projection and bounded claim",
            } if official_socialgym2_manifest_path.exists() else {}),
            **({
                "socialgym2_twoagent_sensitivity": "official SocialGym2 minimum valid two-robot/one-human sensitivity across five seeds; zero-success regime and bounded progress/collision direction, not continuous H3 or superiority",
            } if socialgym2_twoagent_sensitivity_report_path.exists() else {}),
            **({
                "official_atom_headless_core": "official AToM three-agent ILQGames numerical-core audit; not ROS runtime and not same-protocol TRACER",
            } if atom_headless_manifest_path.exists() else {}),
            **({
                "official_algames_headless_core": "official ALGAMES numerical examples in isolated Julia; not LUCIDGames GUI, ROS runtime, or same-protocol TRACER",
            } if algames_headless_results_path.exists() else {}),
            **({
                "same_protocol_local_baselines": "same resettable multi-entity candidate/execution protocol with local MPOGames-style, ORCA-style, and MPC-style controls; not official literature reimplementations",
            } if local_same_protocol_manifest_path.exists() else {}),
            **({
                "external_baseline_audit": "read-only source/runtime/evidence audit for official external baselines",
            } if external_baseline_audit_path.exists() else {}),
            **({
                "external_validation_holdout": "validation-only CrowdSimPlus safety/margin selection followed by a disjoint five-seed holdout; bounded external compatibility evidence, not superiority",
            } if external_validation_holdout_report_path.exists() and external_validation_holdout_manifest_path.exists() else {}),
            **({
                "hard_safety_property": "executable finite-candidate static geometry and fallback property audit; no dynamic safety guarantee",
            } if hard_safety_property_path.exists() else {}),
            **({
                "belief_cost_selection": "validation-only belief-to-cost interface selection with an independent held-out test; diagnostic, not superiority",
            } if belief_cost_selection_path.exists() else {}),
            **({
                "clean_formal_chain": "single clean-commit H1 evaluation -> H2 fresh training/evaluation -> H3 exact-checkpoint closed loop, five seeds, CUDA; bounded formal evidence",
            } if clean_formal_chain_manifest_path.exists() else {}),
            **({
                "external_crowdnav_holdout": "disjoint official CrowdNav cases and five-seed holdout including ORCA/Linear; explicit cost-safety trade-off, not superiority",
            } if external_crowdnav_holdout_report_path.exists() and external_crowdnav_holdout_manifest_path.exists() else {}),
            **({
                "external_crowdnav_full_budget": "corrected official CrowdNav 25-second full-budget validation and disjoint five-seed holdout; success/collision trade-off retained, not superiority",
            } if external_crowdnav_full_budget_report_path.exists() and external_crowdnav_full_budget_validation_manifest_path.exists() and external_crowdnav_full_budget_holdout_manifest_path.exists() else {}),
            **({
                "external_kinematic_guard_ablation": "validation-selected observed-state kinematic guard fails to generalize on the disjoint CrowdSimPlus holdout; negative safety result retained and guard disabled in the primary protocol",
            } if external_kinematic_guard_report_path.exists() else {}),
            **({
                "external_candidate_budget": "validation-selected CrowdSimPlus candidate budget with a disjoint five-seed holdout; negative sensitivity result retained, not a superiority claim",
            } if external_candidate_budget_report_path.exists() else {}),
            **({
                "external_eth_ucy_budget_audit": "ETH/UCY SAGE training-budget sensitivity with fixed held-out scenes and seeds; observational portability trade-off, not closed-loop H3 or uniform superiority",
            } if external_eth_ucy_budget_report_path.exists() else {}),
            **({
                "external_response_weight1_audit": "validation-selected emission-weight alternative on five-seed external recordings; THOR identity-continuity Pareto point with HuRoN mixed direction, not a primary replacement",
            } if external_response_weight1_audit_path.exists() else {}),
            **({
                "external_static_obstacle_guard": "validation-selected CrowdSimPlus swept-path static-obstacle guard with a disjoint five-seed holdout; task/safety operating-point evidence, not a hard safety guarantee",
            } if external_static_obstacle_guard_report_path.exists() else {}),
            **({
                "external_static_obstacle_holdout": "final clean-commit CrowdSimPlus swept-path guard holdout with five model seeds; bounded external operating-point evidence with all belief-policy directions retained",
            } if external_static_obstacle_guard_report_path.exists() and external_static_obstacle_holdout_manifest_path.exists() else {}),
            **({
                "external_static_same_case_holdout": "same holdout case range and episode budget for final guarded TRACER and official CrowdSimPlus controls; mixed system-level directions retained",
            } if external_static_obstacle_guard_report_path.exists() and external_static_same_case_holdout_manifest_path.exists() else {}),
            **({
                "external_multientity_scale": "official CrowdSimPlus single-robot multi-entity TRACER scale audit at 1/3/5 humans; execution and density-failure evidence, not three-robot H3 or superiority",
            } if external_multientity_scale_report_path.exists() and all(path.exists() for path in external_multientity_scale_manifest_paths) else {}),
            **({
                "external_dynamic_fallback": "validation-selected ORCA proposal and emergency-fallback hybrid on disjoint official CrowdSimPlus cases; pure control, all belief policies, and same-case official controls retained; not pure TRACER superiority or hard safety",
            } if all(
                path.exists()
                for path in (
                    external_dynamic_fallback_report_path,
                    external_dynamic_fallback_holdout_manifest_path,
                    external_dynamic_pure_holdout_manifest_path,
                    external_dynamic_same_case_manifest_path,
                    *external_dynamic_validation_manifest_paths,
                )
            ) else {}),
            **({
                "external_dynamic_sicnav_controls": "official SICNav-np and SICNav-p added to the same fixed-case CrowdSimPlus comparison; standalone official controller rows, not a multi-robot TRACER result",
            } if external_dynamic_sicnav_same_case_manifest_path.exists() else {}),
            **({
                "external_dynamic_kinematic_holdout": "validation-selected kinematic safety margin evaluated on an untouched CrowdSimPlus holdout; negative generalization result retained and not promoted to the primary protocol",
            } if external_dynamic_kinematic_holdout_report_path.exists() and external_dynamic_kinematic_holdout_manifest_path.exists() else {}),
            **({
                "external_sage_sampling_sensitivity": "fixed five-sample external SAGE stochastic-coverage sensitivity over three datasets and three seeds; best-of-K navigation with all safety diagnostics retained, not a primary replacement",
            } if external_sage_sampling_report_path.exists() else {}),
            **({
                "multientity_system_matrix": "multi-entity TRACER closed-loop robustness and local system-baseline matrix on an independent resettable Social-Force environment; no official literature reimplementation claim",
            } if multientity_system_manifest_path.exists() else {}),
            **({
                "independent_interaction": "independent nonlinear Social-Force reset/replay, joint-response, persistent-belief, and bounded closed-loop evidence; no real-human causal claim",
            } if independent_formal_path.exists() else {}),
            **({
                "external_tracer_shadow_replay": "full TRACER software-stack shadow execution on THOR-MAGNI and HuRoN logged recordings; no unexecuted alternative-action outcome claim",
            } if external_shadow_path.exists() else {}),
            **({
                "independent_multientity_suite": "independent multi-entity Social-Force H2/H3 replay, continuous-parameter mismatch, response baselines, and R=6 scalability diagnostics; no real-human causal claim",
            } if independent_multientity_path.exists() else {}),
        },
        "h3_recurrent_summary": {
            split: h3_recurrent_summary[split]["contrasts"]
            for split in ("id", "ood")
        },
        "limitations": [
            "All formal claims are restricted to the disclosed synthetic protocol.",
            "The current-method THOR-MAGNI and HuRoN runs are single-robot observational degenerations, not multi-robot counterfactual or real-human causal evidence; HuRoN has only 19 single-entity test episodes and no valid shuffled-ID control.",
            "No hard collision guarantee or universal navigation superiority is claimed.",
            "Official conference class files must be supplied before final compilation.",
        ],
    }
    (output_dir / "evidence_manifest.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible ICRA submission tables.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="PAPER/icra2027")
    args = parser.parse_args()
    evidence = build(Path(args.root).resolve(), Path(args.output_dir))
    print(f"built_submission_artifacts={args.output_dir}")
    print(f"claim_keys={','.join(sorted(evidence['claims']))}")


if __name__ == "__main__":
    main()
