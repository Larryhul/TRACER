from types import SimpleNamespace

import numpy as np
import torch

from tracer.experiments.train_rolling_external_response import aligned_history, prediction_loss, response_group_batches


def test_rolling_history_includes_actual_current_position():
    class Agent:
        def get_position(self):
            return (2.0, 0.0)
        def get_velocity(self):
            return (1.0, 0.0)
    robot = Agent()
    human = Agent()
    env = SimpleNamespace(robot=robot, humans=[human], time_step=.25,
                          states=[(SimpleNamespace(position=(1.5, 0.0)), [SimpleNamespace(position=(1.5, 0.0))]),
                                  (SimpleNamespace(position=(1.75, 0.0)), [SimpleNamespace(position=(1.75, 0.0))])])
    result = aligned_history(env, human, 4)
    np.testing.assert_allclose(result[:, 0], [1.25, 1.5, 1.75, 2.0])


def test_rolling_gaussian_loss_remains_differentiable():
    mean = torch.zeros(3, 6, 2, requires_grad=True)
    log_std = torch.zeros_like(mean, requires_grad=True)
    loss = prediction_loss({"mean": mean, "log_std": log_std}, torch.ones_like(mean))
    loss.backward()
    assert torch.isfinite(loss)
    assert mean.grad.abs().sum() > 0


def test_response_group_batches_keep_counterfactuals_together_without_duplicates():
    data = {"case": np.repeat(np.arange(4), 3), "window": np.zeros(12, dtype=int), "human": np.zeros(12, dtype=int)}
    batches = list(response_group_batches(data, 7, np.random.default_rng(7)))
    assert all(len(batch) <= 7 for batch in batches)
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(12))
    for case in range(4):
        assert sum(np.any(data["case"][batch] == case) for batch in batches) == 1
