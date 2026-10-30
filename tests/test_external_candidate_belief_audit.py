import numpy as np
import torch

from tracer.experiments.audit_external_candidate_belief import causal_row_beliefs, paired_case_interval


def _observations():
    data = {"case": np.ones(6), "human": np.zeros(6), "window": np.repeat([2, 0, 1], 2),
            "target": np.zeros((6, 2, 2))}
    means = torch.zeros(6, 2, 2, 2)
    means[:, 1] = 1.
    return data, {"mean": means, "log_std": torch.zeros_like(means)}, torch.zeros(6, 2, 2)


def test_candidate_beliefs_are_before_observation_and_not_repeated_per_branch():
    data, prediction, target = _observations()
    result = causal_row_beliefs(data, prediction, target, 1.)
    torch.testing.assert_close(result["persistent"][2:4], torch.full((2, 2), .5))
    torch.testing.assert_close(result["persistent"][4], torch.tensor([0., -1.]).softmax(-1))
    torch.testing.assert_close(result["persistent"][0], torch.tensor([0., -2.]).softmax(-1))
    torch.testing.assert_close(result["reset"][0], result["reset"][4])
    changed = target.clone()
    changed[:2] = 100.
    data["target"] = changed.numpy()
    second = causal_row_beliefs(data, prediction, changed, 1.)
    torch.testing.assert_close(second["persistent"], result["persistent"])


def test_confidence_interval_clusters_by_case_not_repeated_windows():
    baseline = paired_case_interval([0., 2.], [1, 2])
    duplicated = paired_case_interval([0.] * 100 + [2.], [1] * 100 + [2])
    assert baseline == duplicated
