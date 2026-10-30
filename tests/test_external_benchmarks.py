from __future__ import annotations

import unittest

import numpy as np
from tracer.experiments.run_crowdnav_tracer import _update_beliefs

from tracer.experiments.run_candidate_policy_shift_benchmark import _policy_actions
from tracer.experiments.run_external_response_benchmarks import (
    _bootstrap_ci,
    _paired_summary,
    _sign_flip_pvalue,
    _trajectory_metrics,
)
from tracer.experiments.run_eth_ucy_portability_benchmark import _aggregate
from tracer.response.causal_synthetic import CausalResponseSpec, CausalTeamResponseDataset


class ExternalBenchmarkTest(unittest.TestCase):
    def test_crowdnav_nonstationary_belief_variants_are_normalized(self) -> None:
        beliefs = np.full((2, 3), -np.log(3.0), dtype=np.float32)
        means = np.zeros((2, 1, 3, 2, 2), dtype=np.float32)
        log_stds = np.zeros_like(means)
        observed_velocity = np.zeros((2, 2, 2), dtype=np.float32)
        for policy in (
            "tracer-persistent",
            "tracer-hazard",
            "tracer-forgetting",
            "tracer-change-point",
            "tracer-reset",
        ):
            updated, change_points = _update_beliefs(
                beliefs,
                means,
                log_stds,
                0,
                observed_velocity,
                2,
                0.05,
                policy,
                0.05,
                0.05,
                1.0,
            )
            self.assertEqual(updated.shape, beliefs.shape)
            self.assertTrue(np.isfinite(updated).all())
            np.testing.assert_allclose(np.exp(updated).sum(axis=-1), 1.0)
            self.assertGreaterEqual(change_points, 0)

    def test_candidate_policy_shift_preserves_factorial_shape_and_changes_policy(self) -> None:
        dataset = CausalTeamResponseDataset(
            CausalResponseSpec(
                split="test",
                num_base_scenes=2,
                interventions_per_scene=4,
                num_entities=3,
                seed=123,
            )
        )
        base = dataset._base_scene(0)
        standard = _policy_actions(dataset, 0, base, "standard")
        shifted = _policy_actions(dataset, 0, base, "wide-lateral")
        self.assertEqual(standard.shape, (4, 2, dataset.horizon, 2))
        self.assertEqual(shifted.shape, standard.shape)
        self.assertFalse(np.allclose(standard, shifted))

    def test_eth_ucy_aggregate_handles_dataset_subset(self) -> None:
        metrics = {
            "robot_ade": 1.0,
            "robot_fde": 1.0,
            "entity_ade": 1.0,
            "entity_fde": 1.0,
            "goal_success_rate": 1.0,
            "collision_rate": 1.0,
            "robot_entity_collision_rate": 1.0,
            "robot_robot_collision_rate": 1.0,
            "min_distance": 1.0,
        }
        rows = [
            {"dataset": "eth", "method": "sage", "model_seed": 2027, **metrics},
            {"dataset": "eth", "method": "mid", "model_seed": 2027, **metrics},
        ]
        aggregate = _aggregate(rows, datasets=("eth",), methods=("sage", "mid"))
        self.assertEqual(aggregate["macro/sage"]["metrics"]["robot_ade"]["mean"], 1.0)
        self.assertEqual(aggregate["macro/mid"]["metrics"]["collision_rate"]["mean"], 1.0)

    def test_uf_coordinate_jump_is_flagged(self) -> None:
        metrics = _trajectory_metrics(
            ["0", "100", "200"],
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.1, 0.0, 0.0)],
        )
        self.assertEqual(metrics["quality_flag"], "suspect_coordinate_jump")

    def test_uf_primary_pairs_exclude_quality_flagged_trials(self) -> None:
        def record(condition: str, quality_flag: str = "ok") -> dict[str, object]:
            return {
                "participant": "P001",
                "task": "inventory_checking",
                "repetition": 1,
                "condition": condition,
                "status": "complete",
                "quality_flag": quality_flag,
                "duration_s": 10.0 if condition == "no_robot" else 12.0,
                "path_length_m": 1.0,
                "displacement_m": 1.0,
                "mean_speed_mps": 0.1,
                "straightness": 1.0,
                "turning_radians": 0.0,
                "file": f"{condition}.xlsx",
            }

        result = _paired_summary([record("no_robot"), record("with_robot", "suspect_coordinate_jump")])
        self.assertEqual(result["summary"]["quality_excluded_trials"], 1)
        self.assertEqual(result["summary"]["paired_groups"], 0)

    def test_uf_paired_inference_is_reproducible(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(_bootstrap_ci(values, seed=7), _bootstrap_ci(values, seed=7))
        self.assertLess(_sign_flip_pvalue(values, seed=7), 0.2)


if __name__ == "__main__":
    unittest.main()
