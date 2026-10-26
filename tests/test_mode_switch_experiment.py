from __future__ import annotations

import unittest

import torch

from sage.experiments.eval_mode_switch_benchmark import _adaptation_delay


class ModeSwitchExperimentTest(unittest.TestCase):
    def test_adaptation_delay_counts_from_switch_and_marks_missing(self) -> None:
        correct = torch.tensor(
            [
                [True, True, False, False, True, False, False, False],
                [True, True, False, False, False, False, False, False],
            ]
        )
        torch.testing.assert_close(
            _adaptation_delay(correct, switch_window=4), torch.tensor([1, 5])
        )


if __name__ == "__main__":
    unittest.main()
