from __future__ import annotations

import unittest

from sage.experiments.audit_h2_belief_protocol import _ece, _row_summary


class H2BeliefProtocolAuditTest(unittest.TestCase):
    def test_ece_is_zero_for_perfectly_reliable_confidence(self) -> None:
        self.assertAlmostEqual(_ece([0.0, 1.0, 1.0], [0.0, 1.0, 1.0]), 0.0)

    def test_row_summary_separates_decision_and_posterior(self) -> None:
        rows = []
        for window, decision_probability, posterior_probability in (
            (0, 1.0 / 3.0, 1.0 / 3.0),
            (1, 0.4, 0.9),
        ):
            rows.append(
                {
                    "active": "1",
                    "window": str(window),
                    "decision_mode_correct": "0",
                    "posterior_mode_correct": "1",
                    "decision_true_probability": str(decision_probability),
                    "posterior_true_probability": str(posterior_probability),
                    "decision_confidence": str(decision_probability),
                    "posterior_confidence": str(posterior_probability),
                    "predictive_trajectory_nll": "0.25",
                }
            )
        summary = _row_summary(rows)
        self.assertEqual(summary["post_initial_rows"], 1)
        self.assertGreater(summary["posterior_minus_decision_accuracy"], 0.0)
        self.assertLess(summary["posterior_minus_decision_nll"], 0.0)


if __name__ == "__main__":
    unittest.main()
