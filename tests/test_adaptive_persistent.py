from __future__ import annotations

import unittest

import torch

from sage.belief.adaptive_persistent import (
    AdaptivePersistentBelief,
    AdaptivePersistentBeliefConfig,
)


class AdaptivePersistentBeliefTest(unittest.TestCase):
    def test_persistent_state_keeps_long_term_memory(self) -> None:
        state = AdaptivePersistentBelief(
            3,
            config=AdaptivePersistentBeliefConfig(
                base_hazard=0.0,
                max_hazard=0.2,
                surprise_threshold=2.0,
                surprise_scale=0.1,
            ),
        )

        posterior, _ = state.update(torch.tensor([[-2.0, 4.0, -2.0]]))
        remembered_probability = posterior.exp()[0, 1].item()
        for _ in range(8):
            posterior, diagnostics = state.update(torch.zeros(1, 3))

        self.assertGreater(remembered_probability, 0.98)
        self.assertGreater(float(posterior.exp()[0, 1]), 0.98)
        self.assertLess(float(diagnostics["hazard"][0]), 1e-5)
        self.assertTrue(torch.isfinite(posterior).all())

    def test_untempered_surprise_increases_hazard(self) -> None:
        state = AdaptivePersistentBelief(
            3,
            prior=torch.tensor([0.98, 0.01, 0.01]),
            config=AdaptivePersistentBeliefConfig(
                base_hazard=0.02,
                max_hazard=0.40,
                surprise_threshold=2.0,
                surprise_scale=0.1,
                likelihood_scale=0.05,
            ),
        )

        _, diagnostics = state.update(torch.full((3,), -10.0))

        self.assertGreater(float(diagnostics["surprise"]), 9.9)
        self.assertGreater(float(diagnostics["hazard"]), 0.39)
        self.assertTrue(bool(diagnostics["changed"]))
        self.assertTrue(bool(diagnostics["valid"]))

    def test_hysteresis_and_cooldown_prevent_repeated_resets(self) -> None:
        state = AdaptivePersistentBelief(
            3,
            prior=torch.tensor([0.98, 0.01, 0.01]),
            config=AdaptivePersistentBeliefConfig(
                base_hazard=0.0,
                max_hazard=0.0,
                surprise_threshold=2.0,
                surprise_scale=0.2,
                hard_reset=True,
                threshold_hysteresis=0.5,
                cooldown_steps=3,
            ),
        )
        high = torch.full((3,), -3.0)
        low = torch.zeros(3)

        _, first = state.update(high)
        _, second = state.update(high)
        _, third = state.update(low)
        _, fourth = state.update(high)

        self.assertTrue(bool(first["changed"]))
        self.assertTrue(bool(first["reset"]))
        self.assertFalse(bool(second["changed"]))
        self.assertFalse(bool(second["reset"]))
        self.assertFalse(bool(third["active"]))
        self.assertTrue(bool(fourth["changed"]))
        self.assertFalse(bool(fourth["reset"]))

        _, fifth = state.update(low)
        _, sixth = state.update(high)
        self.assertTrue(bool(sixth["reset"]))
        self.assertEqual(int(fifth["cooldown"]), 0)

    def test_batch_updates_and_nonfinite_likelihood_are_safe(self) -> None:
        state = AdaptivePersistentBelief(
            3,
            config=AdaptivePersistentBeliefConfig(
                base_hazard=0.1,
                max_hazard=0.3,
                surprise_threshold=1.0,
                surprise_scale=0.2,
            ),
        )
        first_likelihood = torch.tensor(
            [[0.0, 2.0, -1.0], [float("nan"), 0.0, 0.0]]
        )
        first_posterior, first_diagnostics = state.update(first_likelihood)
        second_posterior, second_diagnostics = state.update(
            torch.tensor([[float("inf"), -float("inf"), 0.0], [0.0, 0.0, 0.0]])
        )

        self.assertEqual(tuple(first_posterior.shape), (2, 3))
        self.assertEqual(tuple(second_posterior.shape), (2, 3))
        self.assertTrue(torch.isfinite(second_posterior).all())
        torch.testing.assert_close(
            second_posterior.exp().sum(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6
        )
        self.assertTrue(bool(first_diagnostics["valid"][0]))
        self.assertFalse(bool(first_diagnostics["valid"][1]))
        self.assertFalse(bool(second_diagnostics["valid"][0]))
        self.assertTrue(bool(second_diagnostics["valid"][1]))
        torch.testing.assert_close(
            second_posterior[0], first_posterior[0], atol=1e-6, rtol=1e-6
        )

    def test_observation_mask_keeps_unobserved_batch_items_persistent(self) -> None:
        state = AdaptivePersistentBelief(3)
        first, _ = state.update(torch.tensor([[4.0, -2.0, -2.0], [0.0, 2.0, -2.0]]))
        second, diagnostics = state.update(
            torch.tensor([[-4.0, 2.0, 2.0], [4.0, -2.0, -2.0]]),
            observation_mask=torch.tensor([True, False]),
        )
        torch.testing.assert_close(second[1], first[1])
        self.assertFalse(bool(diagnostics["valid"][1]))

    def test_reset_restores_initial_prior(self) -> None:
        prior = torch.tensor([0.2, 0.3, 0.5])
        state = AdaptivePersistentBelief(3, prior=prior)
        state.update(torch.tensor([4.0, -2.0, -2.0]))

        restored = state.reset()

        assert restored is not None
        torch.testing.assert_close(restored.exp(), prior, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
