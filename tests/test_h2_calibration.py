from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sage.experiments.analyze_h2_calibration import _ece
from sage.experiments.summarize_h2_probabilistic_diagnostics import summarize


class H2CalibrationTest(unittest.TestCase):
    def test_ece_uses_posterior_confidence_and_correctness(self) -> None:
        rows = [
            {"posterior_confidence": "0.9", "posterior_mode_correct": "1"},
            {"posterior_confidence": "0.8", "posterior_mode_correct": "0"},
            {"posterior_confidence": "0.2", "posterior_mode_correct": "0"},
            {"posterior_confidence": "0.1", "posterior_mode_correct": "1"},
        ]
        ece, bins = _ece(rows)
        self.assertAlmostEqual(ece, 0.5, places=6)
        self.assertEqual(len(bins), 4)

    def test_probabilistic_summary_preserves_split_method_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (2027, 2028):
                seed_dir = root / f"seed{seed}"
                seed_dir.mkdir()
                payload = {
                    "persistent": {
                        "final_mode_nll": 0.5 + seed / 100000,
                        "final_mode_brier": 0.2,
                        "final_mode_accuracy": 0.8,
                        "final_posterior_entropy": 0.4,
                        "mode_nll": 0.6,
                        "mode_brier": 0.3,
                        "posterior_entropy": 0.5,
                        "windows_to_confidence": 2.0,
                        "entropy_by_window": [0.8, 0.4],
                        "true_probability_by_window": [0.6, 0.8],
                    }
                }
                (seed_dir / "metrics.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                (seed_dir / "ood_metrics.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            summary = summarize(root)
            scalar = [
                row
                for row in summary["scalar_metrics"]
                if row["split"] == "ood"
                and row["method"] == "persistent"
                and row["metric"] == "final_mode_nll"
            ]
            self.assertEqual(len(scalar), 1)
            self.assertEqual(scalar[0]["n"], 2)
            self.assertAlmostEqual(scalar[0]["mean"], 0.520275, places=6)
            self.assertEqual(summary["num_seeds"], 2)
            self.assertTrue(summary["limitations"])


if __name__ == "__main__":
    unittest.main()
