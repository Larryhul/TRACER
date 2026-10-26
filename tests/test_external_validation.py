from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from sage.experiments.audit_external_validation import audit
from sage.experiments.run_external_sdd_multiseed import aggregate
from sage.experiments.summarize_external_sdd_baselines import summarize, write_outputs


class ExternalValidationAuditTest(unittest.TestCase):
    def test_multiseed_aggregate_reports_mean_and_sample_std(self) -> None:
        rows = [
            {
                "seed": "1",
                "robot_ade": "1.0",
                "robot_fde": "2.0",
                "entity_ade": "3.0",
                "goal_fde": "4.0",
                "goal_success_rate": "0.1",
                "robot_entity_collision_rate": "0.2",
                "robot_robot_collision_rate": "0.3",
            },
            {
                "seed": "2",
                "robot_ade": "3.0",
                "robot_fde": "4.0",
                "entity_ade": "5.0",
                "goal_fde": "6.0",
                "goal_success_rate": "0.3",
                "robot_entity_collision_rate": "0.4",
                "robot_robot_collision_rate": "0.5",
            },
        ]
        summary = aggregate(rows)
        self.assertEqual(summary["metrics"]["robot_ade"]["mean"], 2.0)
        self.assertAlmostEqual(summary["metrics"]["robot_ade"]["std"], 2**0.5)
        self.assertEqual(summary["metrics"]["robot_ade"]["n"], 2)

    def test_audit_reports_external_metrics_and_fail_closed_jrdb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdd_root = root / "sdd" / "annotations" / "bookstore" / "video0"
            sdd_root.mkdir(parents=True)
            sdd_path = sdd_root / "annotations.txt"
            lines = []
            for frame in range(40):
                x = 100 + frame
                lines.append(f"0 {x} 100 {x + 10} 110 {frame * 12} 0 0 0 \"Pedestrian\"")
            sdd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            thor_path = root / "thor.csv"
            fields = ["Time", "frame_id", "x", "y", "ag_id", "agent_type", "file_name", "tracklet_id", "dataset_name", "action"]
            with thor_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for frame in range(40):
                    writer.writerow(
                        {
                            "Time": frame * 0.4,
                            "frame_id": frame,
                            "x": frame * 0.1,
                            "y": 0.0,
                            "ag_id": "Helmet_1",
                            "agent_type": "Visitors-Alone",
                            "file_name": "scenario.csv",
                            "tracklet_id": 1,
                            "dataset_name": "Scenario_1",
                            "action": "Walk",
                        }
                    )

            payload = audit(
                sdd_root.parents[2],
                thor_path,
                root / "jrdb",
                root / "report",
                history=8,
                horizon=12,
                sdd_frame_stride=12,
                sdd_coordinate_scale=50.0,
            )
            self.assertEqual(payload["datasets"]["SDD"]["status"], "complete")
            self.assertGreater(payload["datasets"]["SDD"]["windows"], 0)
            self.assertEqual(payload["datasets"]["THOR-MAGNI"]["status"], "complete")
            self.assertGreater(payload["datasets"]["THOR-MAGNI"]["windows"], 0)
            self.assertEqual(
                payload["datasets"]["JRDB"]["status"],
                "blocked_missing_or_authenticated_assets",
            )
            self.assertTrue((root / "report" / "external_validation.md").exists())

    def test_sdd_summary_preserves_learned_rows_and_protocol(self) -> None:
        fields = [
            "method",
            "baseline",
            "robot_ade",
            "robot_fde",
            "entity_ade",
            "goal_success_rate",
            "robot_entity_collision_rate",
            "robot_robot_collision_rate",
        ]
        values = {
            "robot_ade": "1.0",
            "robot_fde": "2.0",
            "entity_ade": "3.0",
            "goal_success_rate": "0.4",
            "robot_entity_collision_rate": "0.1",
            "robot_robot_collision_rate": "0.2",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "cv.csv"
            sage = root / "sage.csv"
            learned = root / "mid.csv"
            for path, discriminator in (
                (baseline, {"baseline": "constant_velocity"}),
                (sage, {"method": "sage"}),
                (learned, {"method": "mid"}),
            ):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    row = {field: "" for field in fields}
                    row.update(values)
                    row.update(discriminator)
                    writer.writerow(row)
            payload = summarize(
                [baseline],
                sage,
                test_windows=1024,
                sage_train_windows=2048,
                sage_epochs=15,
                sage_seeds=5,
                sage_samples=20,
                learned_paths=[learned],
            )
            self.assertEqual(payload["protocol"]["sage_samples"], 20)
            self.assertEqual(payload["methods"]["mid"]["n"], 1)
            self.assertIn(str(learned), payload["sources"])
            output_dir = root / "summary"
            write_outputs(output_dir, payload)
            written = json.loads(
                (output_dir / "sdd_external_comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(written["methods"]), {"constant_velocity", "sage", "mid"})


if __name__ == "__main__":
    unittest.main()
