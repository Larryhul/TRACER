from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml

from tracer.experiments.eval_persistent_closed_loop import (
    POLICIES,
    _aggregate_runs,
    _configured_policies,
)
from tracer.experiments.audit_formal_h2_data import audit_h2_data
from tracer.experiments.formal_manifest import (
    prepare_formal_run,
    require_formal_gate,
    sha256_file,
    validate_formal_config,
)
from tracer.experiments.review_h3_formal import review_formal_h3
from tracer.experiments.paired_statistics import (
    blocked_paired_bootstrap,
    validate_formal_episode_analysis,
    validate_formal_episode_records,
)
from tracer.experiments.review_h1_rescue import (
    EXPECTED_METHODS,
    EXPECTED_MODES,
    EXPECTED_SEEDS,
    TRAINED_VARIANTS,
    review_h1_formal,
    review_h1_rescue,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FormalH3ReviewTest(unittest.TestCase):
    def test_recurrent_closed_loop_baseline_is_explicitly_opt_in(self) -> None:
        policies = _configured_policies({"additional_policies": ["recurrent-joint"]})
        self.assertEqual(policies[-1].name, "recurrent-joint")
        self.assertEqual(policies[-1].belief_type, "recurrent")
        with self.assertRaises(ValueError):
            _configured_policies({"additional_policies": ["unknown-policy"]})

    def test_complete_failed_task_gate_stays_descriptive(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis_path = root / "analysis.json"
            episode_path = root / "episode_metrics.csv"
            runs_path = root / "runs.json"
            manifest_path = root / "manifest.json"
            candidate_gate_path = root / "candidate_gate.json"
            analysis_path.write_text(
                json.dumps(
                    {
                        "analysis_status": "FORMAL_CONFIRMATORY",
                        "bootstrap_samples": 10_000,
                        "seed": 20270810,
                        "gate": {"status": "fail"},
                    }
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "COMPLETED",
                        "independent_unit": "episode",
                        "model_seed_role": "nuisance_block",
                    }
                ),
                encoding="utf-8",
            )
            candidate_gate_path.write_text(
                json.dumps({"status": "evaluated", "passed": False}),
                encoding="utf-8",
            )
            fields = ["seed", "split", "policy", "episode", *(
                "realized_total_cost",
                "task_progress",
                "safety_cost",
                "human_cost",
                "min_distance",
                "collision_rate",
                "mean_activity",
                "oracle_regret",
                "oracle_hit_rate",
                "candidate_rank_spearman",
                "candidate_rank_kendall",
                "task_feasible_candidate_count",
                "task_feasible_candidate_fraction",
                "candidate_true_cost_range",
                "final_belief_entropy",
                "final_mode_accuracy",
            )]
            runs = {}
            with episode_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for seed in (2027, 2028, 2029, 2030, 2031):
                    runs[str(seed)] = {}
                    for split in ("id", "ood"):
                        runs[str(seed)][split] = {}
                        for policy in (
                            "task-only",
                            "no-belief-joint",
                            "reset-additive",
                            "reset-joint",
                            "persistent-additive",
                            "persistent-joint",
                            "oracle-joint",
                        ):
                            writer.writerow(
                                {
                                    "seed": seed,
                                    "split": split,
                                    "policy": policy,
                                    "episode": 0,
                                    **{field: 0.0 for field in fields[4:]},
                                }
                            )
                            window_record = {
                                "window": 0,
                                "candidate_hash": "a" * 64,
                                "candidate_seed": 17,
                                "candidate_generation_key": "fixture",
                                "selected_candidate": 0,
                                "predicted_candidate_utility": [0.0],
                                "ground_truth_candidate_utility": [0.0],
                            }
                            runs[str(seed)][split][policy] = {
                                "episode_records": [{"episode": 0}],
                                "selected_candidates": [[0]],
                                "window_records": [[window_record]],
                            }
            runs_path.write_text(json.dumps(runs), encoding="utf-8")
            result = review_formal_h3(
                analysis_path,
                episode_path,
                runs_path,
                manifest_path,
                candidate_gate_path,
                episodes=1,
                windows=1,
                candidates=1,
            )
            self.assertEqual(
                result["review_status"], "ELIGIBLE_FOR_H3_DESCRIPTIVE_REVIEW"
            )
            self.assertFalse(result["claim_gate"]["full_closed_loop_claim_passed"])
            self.assertEqual(
                result["claim_gate"]["decision"],
                "BLOCKED_BY_SAGE_CANDIDATE_GATE_AND_TASK_GATE",
            )
            analysis_path.write_text(
                json.dumps(
                    {
                        "analysis_status": "FORMAL_CONFIRMATORY",
                        "bootstrap_samples": 10_000,
                        "seed": 20270810,
                        "gate": {"status": "pass"},
                    }
                ),
                encoding="utf-8",
            )
            result = review_formal_h3(
                analysis_path,
                episode_path,
                runs_path,
                manifest_path,
                candidate_gate_path,
                episodes=1,
                windows=1,
                candidates=1,
            )
            self.assertEqual(
                result["claim_gate"]["decision"],
                "BLOCKED_BY_SAGE_CANDIDATE_GATE",
            )


class FormalConfigurationTest(unittest.TestCase):
    def test_formal_configs_use_five_seed_blocks_and_frozen_data_seeds(self) -> None:
        h1 = yaml.safe_load(
            (PROJECT_ROOT / "configs/h1_paired_formal.yaml").read_text(encoding="utf-8")
        )
        h2 = yaml.safe_load(
            (PROJECT_ROOT / "configs/persistent_trajectory_formal.yaml").read_text(
                encoding="utf-8"
            )
        )
        h3 = yaml.safe_load(
            (PROJECT_ROOT / "configs/persistent_closed_loop_formal.yaml").read_text(
                encoding="utf-8"
            )
        )
        h3_corrected = yaml.safe_load(
            (PROJECT_ROOT / "configs/persistent_closed_loop_formal_corrected.yaml").read_text(
                encoding="utf-8"
            )
        )
        h3_rescue = yaml.safe_load(
            (PROJECT_ROOT / "configs/persistent_closed_loop_formal_rescue.yaml").read_text(
                encoding="utf-8"
            )
        )
        h3_candidate_rescue = yaml.safe_load(
            (
                PROJECT_ROOT
                / "configs/persistent_closed_loop_formal_candidate_rescue.yaml"
            ).read_text(encoding="utf-8")
        )
        for raw in (h1, h2, h3, h3_candidate_rescue):
            validate_formal_config(raw)
            self.assertEqual(len(set(raw["suite"]["seeds"])), 5)
            self.assertEqual(
                raw["formal_protocol"]["model_seed_role"], "nuisance_block"
            )
        self.assertEqual(h2["evaluation"]["id_split"], "test")
        self.assertEqual(
            h1["formal_protocol"]["execution"],
            "evaluation_only_frozen_checkpoints",
        )
        self.assertEqual(h1["dataset"]["causal_response"]["seed"], 328731)
        self.assertEqual(
            h1["suite"]["checkpoint_root"],
            "experiments/pilot/h1_paired_rescue_stage1_v2/checkpoints",
        )
        self.assertEqual(h2["evaluation"]["likelihood_scale"], 0.05)
        self.assertEqual(h2["dataset"]["num_robots"], 3)
        self.assertEqual(h2["dataset"]["data_seed"], 418731)
        self.assertEqual(h3["belief"]["likelihood_scale"], 0.05)
        self.assertEqual(h3["simulation"]["num_robots"], 3)
        self.assertEqual(h3["simulation"]["data_seed"], 518731)
        self.assertEqual(h2["analysis"]["bootstrap_samples"], 10_000)
        self.assertEqual(h3["analysis"]["bootstrap_samples"], 10_000)
        self.assertEqual(h2["analysis"]["seed"], 20270810)
        self.assertEqual(h3["analysis"]["seed"], 20270810)
        self.assertEqual(h3_corrected["simulation"]["data_seed"], 528731)
        self.assertEqual(
            h3_corrected["planning"]["task_steps"],
            h3_corrected["simulation"]["execute_steps"],
        )
        self.assertEqual(h3_rescue["simulation"]["data_seed"], 538731)
        self.assertEqual(
            h3_rescue["candidate_source"]["task_anchor_speed_scales"], [0.4]
        )
        self.assertEqual(h3_candidate_rescue["simulation"]["data_seed"], 548731)
        self.assertEqual(
            h3_candidate_rescue["candidate_source"]["guidance_step_clip_norm"],
            0.1,
        )
        self.assertEqual(
            h3_candidate_rescue["formal_protocol"]["candidate_gate"],
            "experiments/pilot/h3_sage_generator_gate_repair_20260819/evaluation_exact/gate.json",
        )
        h3_replication = yaml.safe_load(
            (
                PROJECT_ROOT
                / "configs/persistent_closed_loop_candidate_replication.yaml"
            ).read_text(encoding="utf-8")
        )
        validate_formal_config(h3_replication)
        self.assertEqual(h3_replication["simulation"]["data_seed"], 558731)
        self.assertEqual(
            h3_replication["formal_protocol"]["replication_role"],
            "independent_data_seed_secondary_robustness",
        )
        self.assertEqual(
            h3_replication["candidate_source"]["task_anchor_speed_scales"],
            [0.4],
        )
        h3_recurrent = yaml.safe_load(
            (
                PROJECT_ROOT
                / "configs/persistent_closed_loop_recurrent_baseline_20260820.yaml"
            ).read_text(encoding="utf-8")
        )
        validate_formal_config(h3_recurrent)
        self.assertEqual(h3_recurrent["simulation"]["data_seed"], 578731)
        self.assertEqual(h3_recurrent["additional_policies"], ["recurrent-joint"])
        self.assertEqual(
            h3_recurrent["formal_protocol"]["replication_role"],
            "independent_data_seed_secondary_recurrent_baseline",
        )
        self.assertIn(
            "recurrent_minus_persistent_predictive_trajectory_nll",
            h2["formal_protocol"]["primary_contrasts"],
        )

    def test_h3_contains_the_complete_belief_response_factorial(self) -> None:
        core_names = {
            "reset-additive",
            "reset-joint",
            "persistent-additive",
            "persistent-joint",
        }
        core = [policy for policy in POLICIES if policy.name in core_names]
        self.assertEqual({policy.name for policy in core}, core_names)
        self.assertEqual(
            {(policy.belief_type, policy.include_pair_interactions) for policy in core},
            {
                ("reset", False),
                ("reset", True),
                ("persistent", False),
                ("persistent", True),
            },
        )

    def test_h3_summary_reports_the_two_by_two_interaction(self) -> None:
        costs = {
            "reset-additive": 4.0,
            "reset-joint": 3.0,
            "persistent-additive": 2.5,
            "persistent-joint": 2.0,
        }
        policy_results = {}
        for policy in POLICIES:
            policy_results[policy.name] = {
                "realized_total_cost": {"mean": costs.get(policy.name, 1.0)},
                "human_cost": {"mean": 1.0},
                "oracle_regret": {"mean": 1.0},
            }
        summary = _aggregate_runs({"2027": {"id": policy_results}})
        interaction = summary["id"]["contrasts"][
            "interaction_of_belief_and_response_realized_total_cost"
        ]
        self.assertAlmostEqual(interaction, 0.5)

    def test_h3_summary_preserves_opt_in_recurrent_policy(self) -> None:
        policy_results = {}
        for policy in (*POLICIES, _configured_policies({"additional_policies": ["recurrent-joint"]})[-1]):
            policy_results[policy.name] = {
                "realized_total_cost": {"mean": 1.0},
                "task_progress": {"mean": 1.0},
                "human_cost": {"mean": 1.0},
                "oracle_regret": {"mean": 1.0},
                "runtime_ms_per_window": {"mean": 2.0},
            }
        summary = _aggregate_runs(
            {"2027": {"id": policy_results}},
            _configured_policies({"additional_policies": ["recurrent-joint"]}),
        )
        self.assertIn("recurrent-joint", summary["id"])
        self.assertEqual(summary["id"]["recurrent-joint"]["runtime_ms_per_window"]["mean"], 2.0)
        self.assertEqual(
            summary["id"]["contrasts"]["recurrent_minus_persistent_total_cost"],
            0.0,
        )


class FormalManifestTest(unittest.TestCase):
    @staticmethod
    def _config() -> dict:
        return {
            "formal_protocol": {
                "version": "formal-v1",
                "stage": "H2",
                "status": "blocked_pending_h1_gate",
                "gate_file": "gates/h1.json",
                "independent_unit": "episode",
                "model_seed_role": "nuisance_block",
                "confirmatory_splits": ["test", "ood"],
                "primary_contrasts": ["reset_minus_persistent_nll"],
                "exclusion_rules": ["nonfinite_only"],
            },
            "suite": {"seeds": [2027, 2028, 2029, 2030, 2031]},
        }

    def test_gate_is_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = self._config()
            with self.assertRaises(RuntimeError):
                require_formal_gate(raw, root)
            gate = root / "gates/h1.json"
            gate.parent.mkdir()
            gate.write_text('{"status": "failed"}\n', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                require_formal_gate(raw, root)
            gate.write_text('{"status": "passed", "source": "unit-test"}\n', encoding="utf-8")
            record = require_formal_gate(raw, root)
            self.assertEqual(record["content"]["status"], "passed")
            self.assertEqual(record["sha256"], sha256_file(gate))

    def test_manifest_freezes_config_environment_and_artifact_hashes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(self._config(), sort_keys=False), encoding="utf-8"
            )
            gate = root / "gates/h1.json"
            gate.parent.mkdir()
            gate.write_text('{"status": "passed"}\n', encoding="utf-8")
            artifact = root / "checkpoint.pt"
            artifact.write_bytes(b"frozen-checkpoint")
            output = root / "run"
            manifest = prepare_formal_run(
                config_path,
                output,
                root,
                command=["python", "formal.py"],
                overrides={"device": "cpu"},
                input_artifacts=[artifact],
            )
            self.assertEqual(manifest["status"], "STARTED")
            self.assertEqual(manifest["input_artifacts"][0]["sha256"], sha256_file(artifact))
            self.assertEqual((output / "config_frozen.yaml").read_bytes(), config_path.read_bytes())
            self.assertTrue((output / "environment.json").is_file())
            self.assertTrue((output / "deviation_log.jsonl").is_file())
            persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["schema"], "sage-formal-run/v1")

            config_path.write_text(
                yaml.safe_dump({**self._config(), "changed": True}), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                prepare_formal_run(
                    config_path,
                    output,
                    root,
                    command=["python", "formal.py"],
                )

    def test_started_manifest_requires_all_declared_artifacts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(self._config(), sort_keys=False), encoding="utf-8"
            )
            gate = root / "gates/h1.json"
            gate.parent.mkdir()
            gate.write_text('{"status": "passed"}\n', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                prepare_formal_run(
                    config_path,
                    root / "run",
                    root,
                    command=["python", "formal.py"],
                    input_artifacts=[root / "missing.pt"],
                )


class PairedStatisticsTest(unittest.TestCase):
    def test_model_seeds_are_blocks_and_episodes_are_independent_units(self) -> None:
        summary = blocked_paired_bootstrap(
            {
                2027: {0: 1.0, 1: 3.0},
                2028: {0: 3.0, 1: 5.0},
            },
            np.random.default_rng(7),
            samples=2_000,
        )
        self.assertEqual(summary["model_seed_blocks"], 2)
        self.assertEqual(summary["independent_units"], 2)
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["paired_observations"], 4)
        self.assertAlmostEqual(float(summary["mean"]), 3.0)
        self.assertAlmostEqual(float(summary["standard_deviation"]), 2.0**0.5)

    def test_missing_episode_in_one_seed_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical paired units"):
            blocked_paired_bootstrap(
                {
                    2027: {0: 1.0, 1: 2.0},
                    2028: {0: 1.5},
                },
                np.random.default_rng(7),
                samples=100,
            )

    def test_formal_episode_analysis_requires_active_matching_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            input_path = run_root / "episode_metrics.csv"
            input_path.write_text("header\n", encoding="utf-8")
            manifest_path = run_root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "sage-formal-run/v1",
                        "status": "STARTED",
                        "stage": "H3",
                        "independent_unit": "episode",
                        "model_seed_role": "nuisance_block",
                    }
                ),
                encoding="utf-8",
            )
            (run_root / "config_frozen.yaml").write_text(
                yaml.safe_dump(
                    {
                        "analysis": {
                            "bootstrap_samples": 10_000,
                            "seed": 20270810,
                        }
                    }
                ),
                encoding="utf-8",
            )
            manifest = validate_formal_episode_analysis(
                manifest_path,
                "H3",
                [input_path],
                bootstrap_samples=10_000,
                analysis_seed=20270810,
            )
            self.assertEqual(manifest["manifest"]["stage"], "H3")
            manifest["manifest"]["status"] = "COMPLETED"
            manifest_path.write_text(
                json.dumps(manifest["manifest"]), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "failed checks"):
                validate_formal_episode_analysis(
                    manifest_path,
                    "H3",
                    [input_path],
                    bootstrap_samples=10_000,
                    analysis_seed=20270810,
                )

    def test_formal_episode_records_require_five_seeds_and_frozen_units(self) -> None:
        frozen = {
            "suite": {"seeds": list(EXPECTED_SEEDS)},
            "simulation": {"episodes": 2, "splits": ["id", "ood"]},
        }
        records = [
            {
                "split": split,
                "contrast": "example",
                "metric": "cost",
                "independent_units": 2,
                "model_seed_blocks": 5,
                "paired_observations": 10,
            }
            for split in ("id", "ood")
        ]
        validate_formal_episode_records(
            records, frozen, "H3", set(EXPECTED_SEEDS)
        )
        records[0]["independent_units"] = 1
        with self.assertRaisesRegex(RuntimeError, "dimensions failed"):
            validate_formal_episode_records(
                records, frozen, "H3", set(EXPECTED_SEEDS)
            )

        with self.assertRaisesRegex(RuntimeError, "complete frozen five-seed set"):
            validate_formal_episode_records(
                records,
                frozen,
                "H3",
                set(EXPECTED_SEEDS[:3]),
            )

    def test_formal_h2_records_use_frozen_id_alias_and_episode_counts(self) -> None:
        frozen = {
            "suite": {"seeds": list(EXPECTED_SEEDS)},
            "dataset": {"test_episodes": 2, "ood_episodes": 3},
            "evaluation": {"id_split": "test"},
        }
        records = [
            {
                "split": split,
                "contrast": "example",
                "metric": "nll",
                "independent_units": episodes,
                "model_seed_blocks": 5,
                "paired_observations": episodes * 5,
            }
            for split, episodes in (("id", 2), ("ood", 3))
        ]
        validate_formal_episode_records(
            records, frozen, "H2", set(EXPECTED_SEEDS)
        )
        records[1]["paired_observations"] = 10
        with self.assertRaisesRegex(RuntimeError, "dimensions failed"):
            validate_formal_episode_records(
                records, frozen, "H2", set(EXPECTED_SEEDS)
            )


class FormalDataAuditTest(unittest.TestCase):
    def test_small_h2_audit_checks_split_identity_and_ood_shift(self) -> None:
        raw = yaml.safe_load(
            (PROJECT_ROOT / "configs/persistent_trajectory_formal.yaml").read_text(
                encoding="utf-8"
            )
        )
        result = audit_h2_data(raw, episodes_per_split=4)
        self.assertEqual(result["audit_status"], "PASS")
        self.assertTrue(
            result["checks"][
                "no_audited_content_overlap_and_full_id_ranges_disjoint"
            ]
        )
        self.assertTrue(result["checks"]["ood_action_correlation_shift_detected"])
        self.assertFalse(result["gate_created"])


class H1RescueReviewTest(unittest.TestCase):
    @staticmethod
    def _write_fixture(
        root: Path, additive_leakage: float = 0.02
    ) -> tuple[Path, Path, Path]:
        contrast = {
            "mean": 0.0,
            "mean_ci95": [-0.001, 0.001],
            "n_base_scenes": 2,
        }
        positive = {
            "mean": 0.2,
            "mean_ci95": [0.1, 0.3],
            "n_base_scenes": 2,
        }
        summary = {
            "status": "PILOT_ONLY_NOT_CONFIRMATORY",
            "independent_unit": "base_scene",
            "data_seed": 318731,
            "split": "test",
            "bootstrap_resamples": 10000,
            "model_seeds": list(EXPECTED_SEEDS),
            "modes": list(EXPECTED_MODES),
            "per_seed": {},
            "pooled_across_seeds": {
                "none": {"matched_additive_minus_joint_nll": contrast},
                "additive": {"matched_additive_minus_joint_nll": contrast},
                "nonlinear": {
                    "matched_additive_minus_joint_nll": positive,
                    "matched_additive_minus_joint_interaction_rmse": positive,
                },
            },
            "diagnostic_maxima": {
                "matched_vs_joint_relative_difference": 0.001,
                "robot_permutation_max_abs_error": 0.0,
                "pair_zero_condition_max_abs_error": 0.0,
                "pair_one_robot_mask_max_abs_error": 0.0,
                "matched_additive_second_difference_max_abs": 0.0,
                "joint_without_pair_matches_additive_max_abs_error": 0.0,
            },
            "gates": {"negative_controls_equivalent_dz_0p2": False},
        }
        for mode in EXPECTED_MODES:
            for seed in EXPECTED_SEEDS:
                summary["per_seed"][f"{mode}/{seed}"] = {
                    "matched_additive_minus_joint_nll": {
                        "mean": 0.2,
                        "n_base_scenes": 2,
                    },
                    "matched_additive_minus_joint_interaction_rmse": {
                        "mean": 0.2,
                        "n_base_scenes": 2,
                    },
                }
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        records_path = root / "paired_records.csv"
        fields = [
            "effect_mode",
            "model_seed",
            "data_seed",
            "split",
            "base_scene_id",
            "method",
            "nll",
            "interaction_rmse",
            "interaction_target_rms",
            "interaction_prediction_rms",
        ]
        with records_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for mode in EXPECTED_MODES:
                for seed in EXPECTED_SEEDS:
                    for scene in range(2):
                        for method in EXPECTED_METHODS:
                            prediction = 0.0
                            if method == "joint" and mode == "none":
                                prediction = 0.01
                            if method == "joint" and mode == "additive":
                                prediction = additive_leakage
                            writer.writerow(
                                {
                                    "effect_mode": mode,
                                    "model_seed": seed,
                                    "data_seed": 318731,
                                    "split": "pilot_test",
                                    "base_scene_id": scene,
                                    "method": method,
                                    "nll": 0.0,
                                    "interaction_rmse": 0.0,
                                    "interaction_target_rms": (
                                        1.0 if mode == "nonlinear" else 0.0
                                    ),
                                    "interaction_prediction_rms": prediction,
                                }
                            )

        checkpoint_root = root / "checkpoints"
        for mode in EXPECTED_MODES:
            for seed in EXPECTED_SEEDS:
                for variant in TRAINED_VARIANTS:
                    checkpoint = (
                        checkpoint_root
                        / mode
                        / f"seed{seed}"
                        / f"{variant}.pt"
                    )
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "effect_mode": mode,
                            "seed": seed,
                            "data_seed": 318731,
                            "variant": variant,
                        },
                        checkpoint,
                    )
        return summary_path, records_path, checkpoint_root

    @staticmethod
    def _convert_fixture_to_formal(
        root: Path, paths: tuple[Path, Path, Path]
    ) -> Path:
        summary_path, records_path, checkpoint_root = paths
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "status": "FORMAL_CONFIRMATORY",
                "data_seed": 328731,
                "allow_data_seed_mismatch": True,
            }
        )
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        with records_path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        for record in records:
            record["data_seed"] = "328731"
            record["split"] = "formal_test"
        with records_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

        frozen_config = root / "config_frozen.yaml"
        frozen_config.write_text("formal_protocol:\n  stage: H1\n", encoding="utf-8")
        (root / "environment.json").write_text(
            json.dumps({"torch": {"cuda_available": True}}), encoding="utf-8"
        )
        artifacts = []
        for checkpoint in sorted(checkpoint_root.rglob("*.pt")):
            artifacts.append(
                {
                    "path": str(checkpoint),
                    "exists": True,
                    "sha256": sha256_file(checkpoint),
                }
            )
        manifest = {
            "schema": "sage-formal-run/v1",
            "status": "COMPLETED",
            "stage": "H1",
            "independent_unit": "base_scene",
            "model_seed_role": "nuisance_block",
            "config": {"sha256": sha256_file(frozen_config)},
            "gate": {"content": {"status": "passed"}},
            "command": ["python", "eval", "--allow-data-seed-mismatch"],
            "overrides": {
                "device": "cuda",
                "modes": list(EXPECTED_MODES),
                "seeds": list(EXPECTED_SEEDS),
                "data_seed": 328731,
                "split": "test",
                "bootstrap_resamples": 10_000,
                "checkpoint_root": str(checkpoint_root),
                "output_dir": str(root),
            },
            "input_artifacts": artifacts,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_complete_passing_inputs_only_become_eligible_for_manual_review(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = self._write_fixture(Path(temporary))
            result = review_h1_rescue(*paths, expected_base_scenes=2)
            self.assertEqual(result["review_status"], "ELIGIBLE_FOR_MANUAL_GATE_REVIEW")
            self.assertTrue(result["revised_gate_pass"])
            self.assertFalse(result["gate_created"])
            self.assertFalse(result["formal_h1_authorized"])

    def test_missing_checkpoint_fails_closed_as_incomplete(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = self._write_fixture(Path(temporary))
            missing = paths[2] / "none" / "seed2027" / "joint.pt"
            missing.unlink()
            result = review_h1_rescue(*paths, expected_base_scenes=2)
            self.assertEqual(
                result["review_status"], "INCOMPLETE_AWAITING_ALL_FIVE_SEEDS"
            )
            self.assertFalse(result["revised_gate_pass"])

    def test_checkpoint_metadata_mismatch_fails_closed_as_incomplete(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = self._write_fixture(Path(temporary))
            mismatched = paths[2] / "none" / "seed2027" / "joint.pt"
            torch.save(
                {
                    "effect_mode": "none",
                    "seed": 9999,
                    "data_seed": 318731,
                    "variant": "joint",
                },
                mismatched,
            )
            result = review_h1_rescue(*paths, expected_base_scenes=2)
            self.assertEqual(
                result["review_status"], "INCOMPLETE_AWAITING_ALL_FIVE_SEEDS"
            )
            self.assertTrue(
                result["completeness"]["checkpoints"]["metadata_errors"]
            )

    def test_excess_mechanism_leakage_blocks_gate(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = self._write_fixture(Path(temporary), additive_leakage=0.10)
            result = review_h1_rescue(*paths, expected_base_scenes=2)
            self.assertEqual(result["review_status"], "DO_NOT_PASS_RESCUE_GATE")
            self.assertFalse(
                result["revised_gate_checks"][
                    "additive_mechanism_leakage_at_most_5pct"
                ]
            )

    def test_formal_profile_checks_manifest_and_independent_corpus(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._write_fixture(root)
            manifest_path = self._convert_fixture_to_formal(root, paths)
            result = review_h1_formal(
                *paths,
                manifest_path=manifest_path,
                expected_base_scenes=2,
            )
            self.assertEqual(
                result["review_status"], "ELIGIBLE_FOR_H1_PASS_GATE_REVIEW"
            )
            self.assertTrue(result["revised_gate_pass"])
            self.assertTrue(result["completeness"]["formal_manifest"]["complete"])
            self.assertFalse(result["h2_authorized"])

    def test_formal_profile_rejects_tampered_checkpoint_manifest_hash(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._write_fixture(root)
            manifest_path = self._convert_fixture_to_formal(root, paths)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["input_artifacts"][0]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = review_h1_formal(
                *paths,
                manifest_path=manifest_path,
                expected_base_scenes=2,
            )
            self.assertEqual(result["review_status"], "INCOMPLETE_FORMAL_H1_INPUTS")
            self.assertFalse(result["revised_gate_pass"])
            self.assertFalse(
                result["completeness"]["formal_manifest"]["checks"][
                    "all_checkpoint_hashes_match_manifest"
                ]
            )


if __name__ == "__main__":
    unittest.main()
