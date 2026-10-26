from __future__ import annotations

import unittest

from sage.experiments.run_thor_magni_full_audit import (
    _condition,
    _gaze_columns,
    _recording_pair_key,
)


class ThorMagniFullAuditTest(unittest.TestCase):
    def test_condition_mapping_uses_dataset_native_scenario_codes(self) -> None:
        self.assertEqual(
            _condition("Scenario_3", "170522_SC3A_R1"),
            "moving_robot_differential_drive",
        )
        self.assertEqual(
            _condition("Scenario_4", "300922_SC4B_R2"),
            "hri_speech_gesture_gaze",
        )

    def test_recording_pair_key_keeps_date_scenario_and_run(self) -> None:
        self.assertEqual(_recording_pair_key("170522_SC3A_R2"), "170522_SC3_R2")
        self.assertIsNone(_recording_pair_key("180522_SC2_R1"))

    def test_gaze_columns_accept_tracker_prefixes(self) -> None:
        header = [
            "Helmet_5 TB2_G2D_X",
            "Helmet_5 TB2_G2D_Y",
            "Helmet_5 TB2_G3D_X",
            "Helmet_5 TB2_G3D_Y",
            "Helmet_5 TB2_G3D_Z",
            "Helmet_5 TB2_Movement",
        ]
        columns = _gaze_columns(header, "Helmet_5")
        self.assertEqual(columns["gaze_2d"], (0, 1))
        self.assertEqual(columns["gaze_3d"], (2, 3, 4))
        self.assertEqual(columns["movement"], (5,))


if __name__ == "__main__":
    unittest.main()
