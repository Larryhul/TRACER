from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest

from tracer.experiments.build_submission_artifacts import build


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_SUMMARY = PROJECT_ROOT / "experiments/formal_v2/h1_tracer_formal/evaluation/summary.json"


@pytest.mark.archived_assets
@pytest.mark.skipif(not ARCHIVED_SUMMARY.is_file(), reason="Restore archived formal-run outputs to build manuscript tables.")
class SubmissionArtifactTest(unittest.TestCase):
    def test_builder_recreates_all_claim_tables_from_frozen_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "submission"
            stale_tables_dir = output_dir / "tables"
            stale_tables_dir.mkdir(parents=True)
            for table_name in (
                "external.tex",
                "external_sage.tex",
                "sdd_strong.tex",
                "eth_ucy_portability.tex",
            ):
                (stale_tables_dir / table_name).write_text("stale", encoding="utf-8")
            evidence = build(PROJECT_ROOT, output_dir)
            self.assertEqual(
                set(evidence["claims"]),
                {
                    "h1",
                    "h2",
                    "h2_prefix_sensitivity",
                    "h2_prefix_scale_sensitivity",
                    "h3",
                    "h3_replication",
                    "h3_recurrent",
                    *(["h3_safety_ablation"] if (
                        PROJECT_ROOT
                        / "reports/h3_safety_filter_ablation_20260902.md"
                    ).exists() else []),
                    "robot_scale",
                    "dataset_sensitivity",
                    "response_family_lofo",
                    "matrix_robustness",
                    *( ["independent_interaction"] if (
                        PROJECT_ROOT
                        / "experiments/independent_social_force_formal_20260902"
                    ).exists() else []),
                    *( ["independent_multientity_suite"] if (
                        PROJECT_ROOT
                        / "experiments/independent_multientity_suite_20260902_final/summary.json"
                    ).exists() else []),
                    *( ["external_tracer_shadow_replay"] if (
                        PROJECT_ROOT
                        / "experiments/external_tracer_shadow_replay_20260902/summary.json"
                    ).exists() else []),
                    *(["external_current_method"] if (
                        PROJECT_ROOT
                        / "reports/external_response_benchmark_20260901/summary.json"
                    ).exists() else []),
                    *(["continuous_action_support"] if (
                        PROJECT_ROOT
                        / "experiments/continuous_intervention_benchmark_20260824/summary.json"
                    ).exists() else []),
                    *(["h2_mode_switch"] if all(
                        (
                            PROJECT_ROOT
                            / f"experiments/mode_switch_benchmark_20260824_{split}/summary.json"
                        ).exists()
                        for split in ("id", "ood")
                    ) else []),
                    *(["h3_mediation"] if (
                        PROJECT_ROOT
                        / "experiments/formal_v2/h3/persistent_closed_loop_suite_tracer_effect_safetyfix/summary.json"
                    ).exists() else []),
                    *(["candidate_policy_shift"] if (
                        PROJECT_ROOT
                        / "reports/external_validation_20260824/candidate_policy_shift/summary.json"
                    ).exists() else []),
                    *( ["official_crowdnav_tracer"] if (
                        PROJECT_ROOT
                        / "experiments/tracer_crowdnav_20260903_full/manifest.json"
                    ).exists() else []),
                    *( ["official_sicnav_baseline"] if any(
                        (
                            PROJECT_ROOT
                            / f"experiments/tracer_sicnav_official_20260903_v{version}/manifest.json"
                        ).exists()
                        for version in (3, 4)
                    ) else []),
                    *( ["official_sicnav_tracer_loop"] if (
                        PROJECT_ROOT
                        / "experiments/tracer_sicnav_crowdsimplus_20260903_full/manifest.json"
                    ).exists() else []),
                    *( ["official_sicnav_same_protocol"] if (
                        PROJECT_ROOT
                        / "experiments/sicnav_same_protocol_20260904/manifest.json"
                    ).exists() else []),
                    *( ["official_socialgym2_tracer_long_horizon"] if (
                        PROJECT_ROOT
                        / "experiments/socialgym2_tracer_20260904_v5_aggregate/manifest.json"
                    ).exists() else []),
                    *( ["socialgym2_twoagent_sensitivity"] if (
                        PROJECT_ROOT
                        / "reports/socialgym2_tracer_twoagent_onehuman_20260904.md"
                    ).exists() else []),
                    *( ["external_baseline_audit"] if (
                        PROJECT_ROOT
                        / "experiments/external_baseline_audit_20260903/audit.json"
                    ).exists() else []),
                    *( ["official_atom_headless_core"] if (
                        PROJECT_ROOT
                        / "experiments/atom_headless_20260903/manifest.json"
                    ).exists() else []),
                    *( ["official_algames_headless_core"] if (
                        PROJECT_ROOT
                        / "experiments/lucidgames_algames_headless_20260903/results.csv"
                    ).exists() else []),
                    *( ["same_protocol_local_baselines"] if (
                        PROJECT_ROOT
                        / "experiments/tracer_gap_closure_20260903_mpogames_final/manifest.json"
                    ).exists() else []),
                    *( ["hard_safety_property"] if (
                        PROJECT_ROOT
                        / "experiments/tracer_hard_safety_property_20260903/summary.json"
                    ).exists() else []),
                    *( ["belief_cost_selection"] if (
                        PROJECT_ROOT
                        / "experiments/tracer_belief_cost_selection_20260903/manifest.json"
                    ).exists() else []),
                    *( ["clean_formal_chain"] if (
                        PROJECT_ROOT
                        / "experiments/formal_v2/clean_chain_submission_20260904/chain_manifest.json"
                    ).exists() else []),
                    *( ["external_crowdnav_holdout"] if (
                        (PROJECT_ROOT / "reports/external_validation_20260904_crowdnav_holdout.md").exists()
                        and any(
                            (
                                PROJECT_ROOT / path
                            ).exists()
                            for path in (
                                "experiments/external_validation_dev/crowdnav_holdout_cases16_31_cuda_20260904/manifest.json",
                                "experiments/external_validation_dev/crowdnav_holdout_cases16_31_fixed_20260904/manifest.json",
                            )
                        )
                    ) else []),
                    *( ["external_crowdnav_full_budget"] if all(
                        (
                            PROJECT_ROOT / path
                        ).exists()
                        for path in (
                            "reports/external_validation_20260904_crowdnav_full_budget.md",
                            "experiments/external_validation_dev/crowdnav_fullbudget_validation_cases0_15_20260904/manifest.json",
                            "experiments/external_validation_dev/crowdnav_fullbudget_holdout_cases16_31_20260904/manifest.json",
                        )
                    ) else []),
                    *( ["external_kinematic_guard_ablation"] if (
                        PROJECT_ROOT
                        / "reports/external_kinematic_guard_ablation_20260904.md"
                    ).exists() else []),
                    *( ["external_candidate_budget"] if (
                        PROJECT_ROOT
                        / "reports/external_validation_20260904_crowdsimplus_candidate_budget.md"
                    ).exists() else []),
                    *( ["external_static_obstacle_guard"] if (
                        PROJECT_ROOT
                        / "reports/external_validation_20260904_crowdsimplus_static_guard.md"
                    ).exists() else []),
                    *( ["external_static_obstacle_holdout"] if all(
                        (
                            PROJECT_ROOT / path
                        ).exists()
                        for path in (
                            "reports/external_validation_20260904_crowdsimplus_static_guard.md",
                            "experiments/external_validation_dev/crowdsimplus_final_static010_safety4_margin08_holdout16_31_cuda_20260904/manifest.json",
                        )
                    ) else []),
                    *( ["external_static_same_case_holdout"] if all(
                        (
                            PROJECT_ROOT / path
                        ).exists()
                        for path in (
                            "reports/external_validation_20260904_crowdsimplus_static_guard.md",
                            "experiments/sicnav_same_protocol_holdout_static010_20260904/manifest.json",
                        )
                    ) else []),
                    *( ["external_multientity_scale"] if all(
                        (
                            PROJECT_ROOT / path
                        ).exists()
                        for path in (
                            "reports/external_multientity_scale_20260904.md",
                            "experiments/external_multientity_scale_20260904_humans1/manifest.json",
                            "experiments/external_multientity_scale_20260904_humans3/manifest.json",
                            "experiments/external_multientity_scale_20260904_humans5/manifest.json",
                        )
                    ) else []),
                    *( ["external_dynamic_fallback"] if all(
                        (
                            PROJECT_ROOT / path
                        ).exists()
                        for path in (
                            "reports/external_dynamic_fallback_holdout_20260904.md",
                            "experiments/external_dynamic_holdout_20260904_orca_fallback/manifest.json",
                            "experiments/external_dynamic_holdout_20260904_goal_random/manifest.json",
                            "experiments/external_dynamic_samecase_holdout_20260904_orca_fallback/manifest.json",
                            "experiments/external_dynamic_validation_20260904_cfg_static/manifest.json",
                            "experiments/external_dynamic_validation_20260904_cfg_safety8/manifest.json",
                            "experiments/external_dynamic_validation_20260904_cfg_margin10/manifest.json",
                            "experiments/external_dynamic_validation_20260904_cfg_kin005/manifest.json",
                            "experiments/external_dynamic_validation_20260904_cfg_kin010_prefix2/manifest.json",
                            "experiments/external_dynamic_validation_20260904_cfg_static020/manifest.json",
                            "experiments/external_dynamic_validation_20260904_cfg_orca_augmented/manifest.json",
                            "experiments/external_dynamic_validation_20260904_cfg_orca_fallback/manifest.json",
                        )
                    ) else []),
                    *( ["external_dynamic_sicnav_controls"] if (
                        PROJECT_ROOT
                        / "experiments/external_dynamic_samecase_holdout_20260904_orca_fallback_sicnav_v2/manifest.json"
                    ).exists() else []),
                    *( ["external_dynamic_kinematic_holdout"] if all(
                        (
                            PROJECT_ROOT / path
                        ).exists()
                        for path in (
                            "reports/external_dynamic_kinematic_holdout_20260904.md",
                            "experiments/external_dynamic_holdout_20260904_hybrid_kin010/manifest.json",
                        )
                    ) else []),
                    *( ["external_eth_ucy_budget_audit"] if (
                        PROJECT_ROOT
                        / "reports/external_validation_20260904/eth_ucy_sage_10epoch/summary.md"
                    ).exists() else []),
                    *( ["external_response_weight1_audit"] if (
                        PROJECT_ROOT
                        / "reports/external_validation_20260904/response_weight1_test_audit.md"
                    ).exists() else []),
                    *( ["external_sage_sampling_sensitivity"] if (
                        PROJECT_ROOT
                        / "reports/external_validation_20260904/external_sage_sampling_sensitivity.md"
                    ).exists() else []),
                    *( ["multientity_system_matrix"] if (
                        PROJECT_ROOT
                        / "experiments/tracer_gap_closure_20260903_mpogames_final/manifest.json"
                    ).exists() else []),
                    *( ["external_validation_holdout"] if all(
                        (
                            PROJECT_ROOT / path
                        ).exists()
                        for path in (
                            "reports/external_validation_20260904_crowdsimplus_holdout.md",
                            "experiments/external_validation_dev/crowdsimplus_safety4_margin08_holdout_cases16_31_gpu_20260904/manifest.json",
                        )
                    ) else []),
                },
            )
            for table in (
                "h1",
                "h2",
                "h2_prefix",
                "h3",
                "h3_replication",
                "h3_recurrent",
                "calibration",
                "external_response_belief",
                "scale",
                "dataset_sensitivity",
                "lofo",
                "matrix_robustness",
                "continuous_intervention",
                "h2_mode_switch",
                "h3_mediation",
                "candidate_policy_shift",
                *( ["independent_interaction"] if (
                    PROJECT_ROOT
                    / "experiments/independent_social_force_formal_20260902"
                ).exists() else []),
                *( ["independent_multientity"] if (
                    PROJECT_ROOT
                    / "experiments/independent_multientity_suite_20260902_final/summary.json"
                ).exists() else []),
                *( ["same_protocol_local_baselines"] if (
                    PROJECT_ROOT
                    / "experiments/tracer_gap_closure_20260903_mpogames_final/episode_metrics.csv"
                ).exists() else []),
            ):
                text = (output_dir / "tables" / f"{table}.tex").read_text(
                    encoding="utf-8"
                )
                self.assertTrue(
                    "\\begin{table}" in text or "\\begin{table*}" in text
                )
                self.assertTrue(
                    "\\end{table}" in text or "\\end{table*}" in text
                )
                self.assertNotIn("95\\% CI & \\\\", text)
            for excluded_table in (
                "external.tex",
                "external_sage.tex",
                "sdd_strong.tex",
                "eth_ucy_portability.tex",
            ):
                self.assertFalse((output_dir / "tables" / excluded_table).exists())
            external_current_table = (
                output_dir / "tables" / "external_response_belief.tex"
            ).read_text(encoding="utf-8")
            self.assertIn("THOR-MAGNI", external_current_table)
            self.assertIn("HuRoN/SACSoN", external_current_table)
            self.assertIn("Recurrent $-$ Persistent", external_current_table)
            self.assertTrue(
                (output_dir / "figures" / "h3_recurrent_secondary.png").exists()
            )
            manifest = json.loads(
                (output_dir / "evidence_manifest.json").read_text(encoding="utf-8")
            )
            for excluded_key in (
                "external_validation",
                "external_response_audit",
                "external_sage_portability",
                "external_sdd_sage_smoke",
                "external_sdd_sage_multiseed",
                "external_sdd_baseline_comparison",
                "external_sdd_strong_comparison",
                "eth_ucy_portability",
            ):
                self.assertNotIn(excluded_key, manifest["claims"])
                self.assertNotIn(excluded_key, manifest["generated_from"])
            self.assertIn("h3_recurrent_summary", manifest)
            expected_h1 = "experiments/formal_v2/h1_tracer_formal/evaluation/summary.json"
            self.assertEqual(manifest["generated_from"]["h1"], expected_h1)
            self.assertEqual(
                manifest["generated_from"]["h3_replication_summary"],
                "reports/h3_core_replication_20260827.json",
            )
            self.assertEqual(
                manifest["generated_from"]["h2_prefix_sensitivity"],
                "reports/h2_prefix_sensitivity_2026-08-21/prefix_summary.csv",
            )
            self.assertEqual(
                manifest["generated_from"]["h2_prefix_scale_sensitivity"],
                "reports/h2_prefix_scale_sensitivity_2026-08-23/summary.json",
            )
            self.assertEqual(manifest["baseline_coverage"]["social_force"]["n"], 5)
            learned_n = 5 if (
                PROJECT_ROOT / "reports/baseline_multiseed_2026-08-21/combined_results.csv"
            ).exists() else 1
            self.assertEqual(manifest["baseline_coverage"]["mid"]["n"], learned_n)
            self.assertEqual(
                manifest["baseline_coverage"]["trajectron_plus_plus"]["n"], learned_n
            )
            self.assertEqual(manifest["baseline_coverage"]["sage_medium"]["n"], learned_n)
            self.assertEqual(
                manifest["generated_from"]["external_current_method"],
                "reports/external_response_benchmark_20260901/summary.json",
            )
            self.assertEqual(
                manifest["generated_from"]["robot_scale"],
                "reports/synthetic_robot_scale_2026-08-21/scale_results.csv",
            )
            self.assertEqual(
                manifest["generated_from"]["independent_multientity_suite"],
                "experiments/independent_multientity_suite_20260902_final/summary.json",
            )
            self.assertEqual(
                manifest["generated_from"]["response_family_lofo"],
                "reports/response_family_lofo_2026-08-21/lofo_summary.csv",
            )
            self.assertEqual(
                manifest["generated_from"]["dataset_sensitivity"],
                "reports/sage_dataset_sensitivity_2026-08-21/dataset_sensitivity.json",
            )
            self.assertEqual(
                manifest["generated_from"]["official_crowdnav_tracer"],
                "experiments/tracer_crowdnav_20260903_full/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["official_sicnav_baseline"],
                [
                    "experiments/tracer_sicnav_official_20260903_v3/manifest.json",
                    "experiments/tracer_sicnav_official_20260903_v4/manifest.json",
                ],
            )
            self.assertEqual(
                manifest["generated_from"]["official_sicnav_tracer_loop"],
                "experiments/tracer_sicnav_crowdsimplus_20260903_full/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["official_sicnav_same_protocol"],
                "experiments/sicnav_same_protocol_20260904/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["external_multientity_scale_report"],
                "reports/external_multientity_scale_20260904.md",
            )
            self.assertEqual(
                manifest["generated_from"]["external_multientity_scale_manifests"],
                [
                    "experiments/external_multientity_scale_20260904_humans1/manifest.json",
                    "experiments/external_multientity_scale_20260904_humans3/manifest.json",
                    "experiments/external_multientity_scale_20260904_humans5/manifest.json",
                ],
            )
            self.assertEqual(
                manifest["generated_from"]["external_dynamic_fallback_report"],
                "reports/external_dynamic_fallback_holdout_20260904.md",
            )
            self.assertEqual(
                manifest["generated_from"]["external_dynamic_fallback_holdout_manifest"],
                "experiments/external_dynamic_holdout_20260904_orca_fallback/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["external_dynamic_pure_holdout_manifest"],
                "experiments/external_dynamic_holdout_20260904_goal_random/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["external_dynamic_same_case_manifest"],
                "experiments/external_dynamic_samecase_holdout_20260904_orca_fallback/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["external_dynamic_sicnav_same_case_manifest"],
                "experiments/external_dynamic_samecase_holdout_20260904_orca_fallback_sicnav_v2/manifest.json",
            )
            expected_dynamic_kinematic_report = (
                "reports/external_dynamic_kinematic_holdout_20260904.md"
                if (
                    PROJECT_ROOT / "reports/external_dynamic_kinematic_holdout_20260904.md"
                ).exists()
                else None
            )
            expected_dynamic_kinematic_manifest = (
                "experiments/external_dynamic_holdout_20260904_hybrid_kin010/manifest.json"
                if (
                    PROJECT_ROOT
                    / "experiments/external_dynamic_holdout_20260904_hybrid_kin010/manifest.json"
                ).exists()
                else None
            )
            self.assertEqual(
                manifest["generated_from"]["external_dynamic_kinematic_holdout_report"],
                expected_dynamic_kinematic_report,
            )
            self.assertEqual(
                manifest["generated_from"]["external_dynamic_kinematic_holdout_manifest"],
                expected_dynamic_kinematic_manifest,
            )
            self.assertEqual(
                manifest["generated_from"]["external_dynamic_validation_manifests"],
                [
                    "experiments/external_dynamic_validation_20260904_cfg_static/manifest.json",
                    "experiments/external_dynamic_validation_20260904_cfg_safety8/manifest.json",
                    "experiments/external_dynamic_validation_20260904_cfg_margin10/manifest.json",
                    "experiments/external_dynamic_validation_20260904_cfg_kin005/manifest.json",
                    "experiments/external_dynamic_validation_20260904_cfg_kin010_prefix2/manifest.json",
                    "experiments/external_dynamic_validation_20260904_cfg_static020/manifest.json",
                    "experiments/external_dynamic_validation_20260904_cfg_orca_augmented/manifest.json",
                    "experiments/external_dynamic_validation_20260904_cfg_orca_fallback/manifest.json",
                ],
            )
            self.assertEqual(
                manifest["generated_from"]["official_socialgym2_tracer_long_horizon"],
                "experiments/socialgym2_tracer_20260904_v5_aggregate/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["external_baseline_audit"],
                "experiments/external_baseline_audit_20260904_final_clean/audit.json",
            )
            self.assertEqual(
                manifest["generated_from"]["official_atom_headless_core"],
                "experiments/atom_headless_20260903/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["official_algames_headless_core"],
                "experiments/lucidgames_algames_headless_20260903/results.csv",
            )
            self.assertEqual(
                manifest["generated_from"]["same_protocol_local_baselines"],
                "experiments/tracer_gap_closure_20260903_mpogames_final/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["hard_safety_property"],
                "experiments/tracer_hard_safety_property_20260903/summary.json",
            )
            self.assertEqual(
                manifest["generated_from"]["belief_cost_selection"],
                "experiments/tracer_belief_cost_selection_20260903/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["clean_formal_chain"],
                "experiments/formal_v2/clean_chain_submission_20260904/chain_manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["external_crowdnav_holdout_report"],
                "reports/external_validation_20260904_crowdnav_holdout.md",
            )
            crowdnav_holdout_candidates = (
                "experiments/external_validation_dev/crowdnav_holdout_cases16_31_cuda_20260904/manifest.json",
                "experiments/external_validation_dev/crowdnav_holdout_cases16_31_fixed_20260904/manifest.json",
            )
            expected_crowdnav_holdout_manifest = next(
                (
                    path
                    for path in crowdnav_holdout_candidates
                    if (PROJECT_ROOT / path).exists()
                ),
                None,
            )
            self.assertEqual(
                manifest["generated_from"]["external_crowdnav_holdout_manifest"],
                expected_crowdnav_holdout_manifest,
            )
            self.assertEqual(
                manifest["generated_from"]["external_crowdnav_full_budget_report"],
                "reports/external_validation_20260904_crowdnav_full_budget.md",
            )
            self.assertEqual(
                manifest["generated_from"]["external_crowdnav_full_budget_validation_manifest"],
                "experiments/external_validation_dev/crowdnav_fullbudget_validation_cases0_15_20260904/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["external_crowdnav_full_budget_holdout_manifest"],
                "experiments/external_validation_dev/crowdnav_fullbudget_holdout_cases16_31_20260904/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["external_crowdnav_full_budget_variants_manifest"],
                "experiments/external_validation_dev/crowdnav_fullbudget_variants_holdout_cases16_31_20260904/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["multientity_system_matrix"],
                "experiments/tracer_gap_closure_20260903_mpogames_final/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["external_validation_holdout_report"],
                "reports/external_validation_20260904_crowdsimplus_holdout.md",
            )
            self.assertEqual(
                manifest["generated_from"]["external_validation_holdout_manifest"],
                "experiments/external_validation_dev/crowdsimplus_safety4_margin08_holdout_cases16_31_gpu_20260904/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["external_candidate_budget"],
                "reports/external_validation_20260904_crowdsimplus_candidate_budget.md",
            )
            self.assertEqual(
                manifest["generated_from"]["external_static_obstacle_guard"],
                "reports/external_validation_20260904_crowdsimplus_static_guard.md",
            )
            self.assertEqual(
                manifest["generated_from"]["external_static_obstacle_holdout_manifest"],
                "experiments/external_validation_dev/crowdsimplus_final_static010_safety4_margin08_holdout16_31_cuda_20260904/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["external_static_same_case_holdout_manifest"],
                "experiments/sicnav_same_protocol_holdout_static010_20260904/manifest.json",
            )
            self.assertEqual(
                manifest["generated_from"]["external_sage_sampling_sensitivity"],
                "reports/external_validation_20260904/external_sage_sampling_sensitivity.md",
            )
            expected_h1_matrix = (
                "experiments/causal_benchmark_matrix_20260824_corrected_shuffle/summary.csv"
                if (
                    PROJECT_ROOT
                    / "experiments/causal_benchmark_matrix_20260824_corrected_shuffle/summary.csv"
                ).exists()
                else "experiments/causal_benchmark_matrix_20260823_mechanisms/summary.csv"
            )
            self.assertEqual(manifest["generated_from"]["h1_matrix"], expected_h1_matrix)
            expected_continuous = (
                "experiments/continuous_intervention_benchmark_20260824/summary.json"
                if (
                    PROJECT_ROOT
                    / "experiments/continuous_intervention_benchmark_20260824/summary.json"
                ).exists()
                else None
            )
            self.assertEqual(
                manifest["generated_from"]["continuous_intervention"],
                expected_continuous,
            )
            expected_mode_switch_id = (
                "experiments/mode_switch_benchmark_20260824_id/summary.json"
                if (
                    PROJECT_ROOT
                    / "experiments/mode_switch_benchmark_20260824_id/summary.json"
                ).exists()
                else None
            )
            expected_mode_switch_ood = (
                "experiments/mode_switch_benchmark_20260824_ood/summary.json"
                if (
                    PROJECT_ROOT
                    / "experiments/mode_switch_benchmark_20260824_ood/summary.json"
                ).exists()
                else None
            )
            self.assertEqual(
                manifest["generated_from"]["h2_mode_switch_id"], expected_mode_switch_id
            )
            self.assertEqual(
                manifest["generated_from"]["h2_mode_switch_ood"], expected_mode_switch_ood
            )
            expected_h3_mediation = (
                "experiments/formal_v2/h3/persistent_closed_loop_suite_tracer_effect_safetyfix/summary.json"
                if (
                    PROJECT_ROOT
                    / "experiments/formal_v2/h3/persistent_closed_loop_suite_tracer_effect_safetyfix/summary.json"
                ).exists()
                else None
            )
            self.assertEqual(
                manifest["generated_from"]["h3_mediation"], expected_h3_mediation
            )
            expected_policy_shift = (
                "reports/external_validation_20260824/candidate_policy_shift/summary.json"
                if (
                    PROJECT_ROOT
                    / "reports/external_validation_20260824/candidate_policy_shift/summary.json"
                ).exists()
                else None
            )
            self.assertEqual(
                manifest["generated_from"]["candidate_policy_shift"], expected_policy_shift
            )
            self.assertEqual(
                manifest["generated_from"]["h2_matrix"],
                "experiments/persistent_benchmark_matrix_20260824_coverage/summary.csv",
            )
            self.assertEqual(
                manifest["generated_from"]["h3_matrix"],
                "experiments/closed_loop_benchmark_matrix_20260824_formal_scale/contrasts.csv",
            )


if __name__ == "__main__":
    unittest.main()
