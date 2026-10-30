from __future__ import annotations

import unittest

import torch

from tracer.belief.baselines import RecurrentBeliefBaseline, RecurrentBeliefConfig
from tracer.belief.filter import (
    PersistentBeliefStore,
    categorical_bayes_update,
    lagged_reset_beliefs,
    run_categorical_filter,
)
from tracer.belief.metrics import belief_classification_metrics, predictive_mixture_nll
from tracer.belief.likelihood import ModeConditionedResponseLikelihood, ModeLikelihoodConfig
from tracer.belief.related_work import (
    UKFBeliefConfig,
    UKFParameterBelief,
    categorical_pomdp_update,
    run_ukf_filter,
)
from tracer.belief.synthetic import (
    BeliefSequenceSpec,
    PersistentBeliefSequenceDataset,
    mode_log_likelihood,
    mode_response_means,
)


class CategoricalBeliefTest(unittest.TestCase):
    def test_bayes_update_normalizes_and_increases_supported_mode(self) -> None:
        prior = torch.full((3,), 1.0 / 3.0).log()
        posterior = categorical_bayes_update(prior, torch.tensor([0.0, -2.0, -3.0]))
        torch.testing.assert_close(posterior.exp().sum(), torch.tensor(1.0))
        self.assertGreater(float(posterior.exp()[0]), 1.0 / 3.0)

    def test_persistent_filter_accumulates_while_reset_does_not(self) -> None:
        likelihoods = torch.tensor([[[0.0, -1.0], [0.0, -1.0], [0.0, -1.0]]])
        _, persistent = run_categorical_filter(likelihoods)
        _, reset = run_categorical_filter(likelihoods, reset_each_window=True)
        self.assertGreater(float(persistent[0, -1, 0]), float(persistent[0, 0, 0]))
        torch.testing.assert_close(reset[0, 0], reset[0, -1])

    def test_lagged_reset_uses_only_the_previous_window(self) -> None:
        likelihoods = torch.tensor([[[0.0, -2.0], [-3.0, 0.0], [0.0, -1.0]]])
        before, after = lagged_reset_beliefs(likelihoods)
        torch.testing.assert_close(before[:, 0], torch.full((1, 2), 0.5).log())
        torch.testing.assert_close(before[:, 1], after[:, 0])
        torch.testing.assert_close(before[:, 2], after[:, 1])

    def test_filter_mask_preserves_posterior_without_an_observation(self) -> None:
        likelihoods = torch.tensor([[[0.0, -2.0], [-4.0, 0.0], [0.0, -1.0]]])
        mask = torch.tensor([[True, False, True]])
        before, after = run_categorical_filter(likelihoods, observation_mask=mask)
        torch.testing.assert_close(after[:, 1], after[:, 0])
        torch.testing.assert_close(before[:, 2], after[:, 0])

    def test_reentry_resumes_the_same_identity_posterior(self) -> None:
        likelihoods = torch.tensor(
            [[[2.0, 0.0], [2.0, 0.0], [-9.0, 9.0], [-9.0, 9.0], [2.0, 0.0]]]
        )
        present = torch.tensor([[True, True, False, False, True]])
        before, after = run_categorical_filter(likelihoods, observation_mask=present)
        torch.testing.assert_close(after[:, 2], after[:, 1])
        torch.testing.assert_close(after[:, 3], after[:, 1])
        torch.testing.assert_close(before[:, 4], after[:, 1])
        self.assertGreater(float(after[0, 4, 0]), float(before[0, 4, 0]))

    def test_store_keeps_entity_posteriors_independent(self) -> None:
        store = PersistentBeliefStore(num_modes=3)
        first = store.update("entity-a", torch.tensor([0.0, -4.0, -4.0]))
        second = store.update("entity-b", torch.tensor([-4.0, 0.0, -4.0]))
        self.assertEqual(len(store), 2)
        self.assertEqual(int(first.argmax()), 0)
        self.assertEqual(int(second.argmax()), 1)
        self.assertEqual(int(store.get("entity-a").argmax()), 0)
        store.reset("entity-a")
        torch.testing.assert_close(store.get("entity-a"), torch.full((3,), 1.0 / 3.0))


class BeliefSequenceDatasetTest(unittest.TestCase):
    def test_episode_has_stable_mode_and_disjoint_split_ids(self) -> None:
        train = PersistentBeliefSequenceDataset(
            BeliefSequenceSpec(split="train", num_episodes=4, windows=6, seed=17)
        )
        validation = PersistentBeliefSequenceDataset(
            BeliefSequenceSpec(split="val", num_episodes=4, windows=6, seed=17)
        )
        sample = train[0]
        self.assertEqual(sample["actions"].shape, (6, 4))
        self.assertEqual(sample["responses"].shape, (6, 2))
        train_ids = {train[index]["entity_id"] for index in range(len(train))}
        validation_ids = {validation[index]["entity_id"] for index in range(len(validation))}
        self.assertTrue(train_ids.isdisjoint(validation_ids))

    def test_true_mode_has_best_average_likelihood(self) -> None:
        dataset = PersistentBeliefSequenceDataset(
            BeliefSequenceSpec(num_episodes=32, windows=8, seed=23, response_noise=0.10)
        )
        actions = torch.stack([dataset[index]["actions"] for index in range(len(dataset))])
        responses = torch.stack([dataset[index]["responses"] for index in range(len(dataset))])
        modes = torch.stack([dataset[index]["mode"] for index in range(len(dataset))])
        likelihoods = mode_log_likelihood(actions, responses, response_noise=0.10)
        predicted = likelihoods.sum(dim=1).argmax(dim=-1)
        self.assertGreater(float((predicted == modes).float().mean()), 0.95)

    def test_response_mean_and_likelihood_shapes(self) -> None:
        actions = torch.rand(2, 5, 4)
        responses = torch.rand(2, 5, 2)
        self.assertEqual(mode_response_means(actions).shape, (2, 5, 3, 2))
        self.assertEqual(mode_log_likelihood(actions, responses, 0.18).shape, (2, 5, 3))

    def test_ood_policy_correlates_team_actions_without_changing_range(self) -> None:
        dataset = PersistentBeliefSequenceDataset(
            BeliefSequenceSpec(
                split="test",
                num_episodes=64,
                windows=10,
                seed=29,
                action_policy="correlated",
            )
        )
        activities = torch.cat(
            [dataset[index]["actions"][:, :2] for index in range(len(dataset))]
        )
        correlation = torch.corrcoef(activities.T)[0, 1]
        self.assertGreater(float(correlation), 0.9)
        self.assertGreaterEqual(float(activities.min()), 0.15)
        self.assertLessEqual(float(activities.max()), 1.0)


class BeliefBaselineAndMetricsTest(unittest.TestCase):
    def test_pomdp_belief_update_normalizes(self) -> None:
        probabilities = torch.full((4, 3), 1.0 / 3.0)
        likelihood = torch.tensor([[2.0, 0.0, -1.0]]).expand(4, -1)
        predicted, posterior = categorical_pomdp_update(probabilities, likelihood, 0.1)
        torch.testing.assert_close(predicted.sum(dim=-1), torch.ones(4))
        torch.testing.assert_close(posterior.sum(dim=-1), torch.ones(4))
        self.assertTrue(torch.all(posterior[:, 0] > posterior[:, 2]))

    def test_ukf_filter_returns_finite_normalized_projected_beliefs(self) -> None:
        actions = torch.rand(5, 6, 4)
        means = mode_response_means(actions)
        responses = means[:, :, 1] + 0.03 * torch.randn(5, 6, 2)
        before, after, nll = run_ukf_filter(
            means,
            responses,
            UKFBeliefConfig(observation_variance=0.18**2, adaptive_process=True),
        )
        self.assertEqual(before.shape, (5, 6, 3))
        self.assertEqual(after.shape, (5, 6, 3))
        self.assertEqual(nll.shape, (5, 6))
        self.assertTrue(torch.isfinite(after).all())
        torch.testing.assert_close(after.sum(dim=-1), torch.ones(5, 6), atol=1e-5, rtol=1e-5)

    def test_streaming_ukf_updates_from_a_response_prefix(self) -> None:
        actions = torch.rand(4, 4)
        means = mode_response_means(actions).permute(1, 0, 2)
        state = UKFParameterBelief(UKFBeliefConfig(), device="cpu")
        prior = state.probabilities()
        posterior = state.update(means, means[1] + 0.02 * torch.randn(4, 2))
        self.assertEqual(tuple(posterior.shape), (3,))
        torch.testing.assert_close(posterior.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.allclose(prior, posterior))

    def test_recurrent_baseline_forward_backward(self) -> None:
        config = RecurrentBeliefConfig(hidden_dim=8)
        model = RecurrentBeliefBaseline(config)
        output = model(torch.rand(4, 6, 4), torch.rand(4, 6, 2))
        self.assertEqual(output.shape, (4, 6, 3))
        loss = -output[..., 0].mean()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_learned_mode_likelihood_scores_all_modes_and_backpropagates(self) -> None:
        model = ModeConditionedResponseLikelihood(ModeLikelihoodConfig(hidden_dim=16))
        actions = torch.rand(4, 6, 4)
        responses = torch.rand(4, 6, 2)
        mode_ids = torch.tensor([0, 1, 2, 0])[:, None].expand(-1, 6)
        output = model(actions, mode_ids)
        self.assertEqual(output["mean"].shape, (4, 6, 2))
        likelihoods = model.all_mode_log_likelihood(actions, responses)
        self.assertEqual(likelihoods.shape, (4, 6, 3))
        loss = -likelihoods.gather(-1, mode_ids[..., None]).mean()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_metrics_reward_correct_confident_belief(self) -> None:
        probabilities = torch.tensor(
            [
                [[0.6, 0.2, 0.2], [0.9, 0.05, 0.05]],
                [[0.2, 0.6, 0.2], [0.05, 0.9, 0.05]],
            ]
        )
        targets = torch.tensor([0, 1])
        metrics = belief_classification_metrics(probabilities, targets)
        self.assertEqual(metrics["final_mode_accuracy"], 1.0)
        self.assertLess(metrics["final_mode_nll"], metrics["mode_nll"])
        log_likelihoods = probabilities.log()
        predictive_nll, curve = predictive_mixture_nll(probabilities, log_likelihoods)
        self.assertTrue(torch.isfinite(torch.tensor(predictive_nll)))
        self.assertEqual(len(curve), 2)


if __name__ == "__main__":
    unittest.main()
