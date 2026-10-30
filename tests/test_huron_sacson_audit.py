from __future__ import annotations

import unittest

import numpy as np

from tracer.experiments.run_huron_sacson_audit import (
    _condition_fields,
    _distance_bin,
    _human_response_audit,
    _motion_bin,
)


class HuronSacsonAuditTest(unittest.TestCase):
    def test_condition_fields_preserve_interaction_loss_variant(self) -> None:
        fields = _condition_fields("Feb-16-2023-bww1-intloss")
        self.assertEqual(fields["date"], "Feb-16-2023")
        self.assertEqual(fields["robot_platform"], "bww1")
        self.assertTrue(fields["interaction_loss_condition"])

    def test_distance_bins_are_ordered_and_fail_closed(self) -> None:
        self.assertEqual(_distance_bin(None), "no_pedestrian_detection")
        self.assertEqual(_distance_bin(0.99), "near_lt_1m")
        self.assertEqual(_distance_bin(1.0), "mid_1_to_2m")
        self.assertEqual(_distance_bin(2.0), "far_2_to_3m")
        self.assertEqual(_distance_bin(3.0), "over_3m")

    def test_motion_bins_use_executed_pose_speed(self) -> None:
        self.assertEqual(_motion_bin(None), "unknown_motion")
        self.assertEqual(_motion_bin(0.0), "stationary_lt_0.05mps")
        self.assertEqual(_motion_bin(0.2), "slow_0.05_to_0.5mps")
        self.assertEqual(_motion_bin(0.5), "moving_ge_0.5mps")

    def test_marker_jump_is_excluded_from_primary_response_summary(self) -> None:
        rows, summary = _human_response_audit(
            [(0, 7, 1.0, 0.0, 1.0), (1_000_000_000, 7, 1.5, 0.0, 1.5), (2_000_000_000, 7, 20.0, 0.0, 20.0)],
            np.asarray([0, 1_000_000_000, 2_000_000_000], dtype=np.int64),
            np.asarray([0.0, 0.2, 0.2]),
            np.zeros(3),
            np.zeros(3),
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["primary_observations"], 1)
        self.assertEqual(summary["quality_excluded_observations"], 1)


if __name__ == "__main__":
    unittest.main()
