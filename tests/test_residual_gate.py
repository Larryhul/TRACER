from __future__ import annotations

import unittest

import torch

from tracer.response.residual_gate import (
    ResidualGateConfig,
    apply_joint_residual_gate,
    compute_belief_confidence,
    compute_joint_residual_gate,
)


class ResidualGateTest(unittest.TestCase):
    def test_uniform_belief_has_low_confidence(self) -> None:
        belief = torch.full((2, 3, 4, 4), 0.25)
        confidence = compute_belief_confidence(belief)
        self.assertEqual(confidence.shape, (2, 3, 4))
        self.assertTrue(torch.allclose(confidence, torch.zeros_like(confidence)))

    def test_confident_low_uncertainty_has_large_gate(self) -> None:
        belief = torch.zeros((2, 3, 4, 4))
        belief[..., 1] = 1.0
        uncertainty = torch.zeros((2, 3, 4))
        gate = compute_joint_residual_gate(belief, uncertainty)
        self.assertEqual(gate.shape, (2, 3, 4))
        self.assertTrue(torch.all(gate > 0.9))
        self.assertTrue(torch.all(gate <= 1.0))

    def test_uniform_or_high_uncertainty_suppresses_gate(self) -> None:
        uniform_belief = torch.full((3, 2, 5, 4), 0.25)
        confident_belief = torch.zeros_like(uniform_belief)
        confident_belief[..., 0] = 1.0
        low_uncertainty = torch.zeros((3, 2, 5))
        high_uncertainty = torch.full((3, 2, 5), 100.0)

        uniform_gate = compute_joint_residual_gate(uniform_belief, low_uncertainty)
        uncertain_gate = compute_joint_residual_gate(
            confident_belief, high_uncertainty
        )

        self.assertTrue(torch.all(uniform_gate < 0.1))
        self.assertTrue(torch.all(uncertain_gate < 0.1))

    def test_extra_uncertainty_dimensions_are_aggregated(self) -> None:
        belief = torch.rand((2, 3, 4, 5))
        uncertainty = torch.rand((2, 3, 4, 6, 7))
        gate = compute_joint_residual_gate(belief, uncertainty)
        self.assertEqual(gate.shape, (2, 3, 4))
        self.assertTrue(torch.all(torch.isfinite(gate)))
        self.assertTrue(torch.all((gate >= 0.0) & (gate <= 1.0)))

    def test_extreme_inputs_remain_finite(self) -> None:
        belief = torch.tensor(
            [[[[1.0e30, 1.0e-30, 0.0], [0.0, 1.0e30, 1.0e-30]]]],
            dtype=torch.float32,
        )
        uncertainty = torch.tensor([[[1.0e30, 0.0]]], dtype=torch.float32)
        gate = compute_joint_residual_gate(belief, uncertainty)
        self.assertTrue(torch.all(torch.isfinite(gate)))
        self.assertTrue(torch.all((gate >= 0.0) & (gate <= 1.0)))

    def test_interpolation_stays_between_predictions(self) -> None:
        additive = torch.tensor(
            [[[[0.0, 3.0], [2.0, -1.0]], [[4.0, 2.0], [-3.0, 1.0]]]]
        )
        joint = torch.tensor(
            [[[[10.0, -2.0], [-4.0, 5.0]], [[-2.0, 8.0], [6.0, -5.0]]]]
        )
        gate = torch.tensor([[[0.0, 0.25], [0.75, 1.0]]])
        interpolated = apply_joint_residual_gate(additive, joint, gate)
        lower = torch.minimum(additive, joint)
        upper = torch.maximum(additive, joint)
        self.assertTrue(torch.all(interpolated >= lower))
        self.assertTrue(torch.all(interpolated <= upper))
        self.assertTrue(torch.allclose(interpolated[..., 0, :][0, 0], additive[..., 0, :][0, 0]))
        self.assertTrue(torch.allclose(interpolated[..., 1, :][0, 1], joint[..., 1, :][0, 1]))

    def test_invalid_shapes_and_types_fail_fast(self) -> None:
        with self.assertRaises(TypeError):
            compute_belief_confidence(torch.ones((2, 3), dtype=torch.int64))
        with self.assertRaises(ValueError):
            compute_joint_residual_gate(
                torch.ones((2, 3, 4)), torch.ones((2, 2, 5))
            )
        with self.assertRaises(ValueError):
            apply_joint_residual_gate(
                torch.zeros((2, 3, 4)),
                torch.ones((2, 3, 5)),
                torch.ones((2, 3)),
            )
        with self.assertRaises(ValueError):
            apply_joint_residual_gate(
                torch.zeros((2, 3)), torch.ones((2, 3)), torch.full((2, 3), 1.1)
            )

    def test_custom_gate_bounds_are_respected(self) -> None:
        config = ResidualGateConfig(min_gate=0.2, max_gate=0.8)
        belief = torch.full((2, 3), 0.5)
        uncertainty = torch.zeros((2,))
        gate = compute_joint_residual_gate(belief, uncertainty, config)
        self.assertTrue(torch.all(gate >= 0.2))
        self.assertTrue(torch.all(gate <= 0.8))


if __name__ == "__main__":
    unittest.main()
