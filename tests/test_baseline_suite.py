from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from sage.experiments.summarize_baseline_suite import summarize


class BaselineSuiteSummaryTest(unittest.TestCase):
    def test_summary_reports_seed_coverage_and_mean_std(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv"
            fields = [
                "baseline",
                "model_variant",
                "dataset_seed",
                "robot_ade",
                "robot_fde",
                "entity_ade",
                "goal_fde",
                "goal_success_rate",
                "robot_entity_collision_rate",
                "robot_robot_collision_rate",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for seed, ade in ((2027, 1.0), (2028, 3.0)):
                    writer.writerow(
                        {
                            "baseline": "constant_velocity",
                            "model_variant": "constant_velocity",
                            "dataset_seed": seed,
                            "robot_ade": ade,
                            "robot_fde": ade,
                            "entity_ade": ade,
                            "goal_fde": ade,
                            "goal_success_rate": 0.0,
                            "robot_entity_collision_rate": 0.1,
                            "robot_robot_collision_rate": 0.2,
                        }
                    )
            payload = summarize(path)
            record = payload["baselines"]["constant_velocity"]
            self.assertEqual(record["dataset_seeds"], ["2027", "2028"])
            self.assertAlmostEqual(record["metrics"]["robot_ade"]["mean"], 2.0)
            self.assertAlmostEqual(record["metrics"]["robot_ade"]["std"], 2 ** 0.5)

    def test_additional_source_does_not_duplicate_primary_deterministic_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fields = [
                "baseline",
                "model_variant",
                "dataset_seed",
                "robot_ade",
                "robot_fde",
                "entity_ade",
                "goal_fde",
                "goal_success_rate",
                "robot_entity_collision_rate",
                "robot_robot_collision_rate",
            ]
            primary = root / "primary.csv"
            legacy = root / "legacy.csv"
            for path, rows in (
                (
                    primary,
                    [("constant_velocity", "constant_velocity", "202027", 1.0)],
                ),
                (
                    legacy,
                    [
                        ("constant_velocity", "constant_velocity", "202027", 9.0),
                        ("", "joint_heterogeneous_semantic", "202027", 2.0),
                    ],
                ),
            ):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for baseline, variant, seed, ade in rows:
                        writer.writerow(
                            {
                                "baseline": baseline,
                                "model_variant": variant,
                                "dataset_seed": seed,
                                "robot_ade": ade,
                                "robot_fde": ade,
                                "entity_ade": ade,
                                "goal_fde": ade,
                                "goal_success_rate": 0.0,
                                "robot_entity_collision_rate": 0.0,
                                "robot_robot_collision_rate": 0.0,
                            }
                        )
            payload = summarize(primary, [legacy])
            self.assertEqual(payload["baselines"]["constant_velocity"]["n_rows"], 1)
            self.assertEqual(payload["baselines"]["sage_medium"]["n_rows"], 1)


if __name__ == "__main__":
    unittest.main()
