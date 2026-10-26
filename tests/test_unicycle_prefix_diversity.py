import numpy as np
import pytest

from sage.response.unicycle_candidates import UnicycleLimits, diversify_unicycle_prefixes, project_unicycle_candidates


@pytest.mark.parametrize("velocity", [[0.0, 0.0], [0.45, 0.0], [-0.45, 0.0]])
def test_variants_change_executed_controls_and_respect_dynamics(velocity):
    desired = np.zeros((2, 8, 4), dtype=np.float32)
    desired[:, :, 2] = 0.8 if velocity[0] >= 0 else -0.8
    limits = UnicycleLimits()
    original = project_unicycle_candidates(desired, [0.0, 0.0], velocity, 0.0, 0.25, limits)
    snapshot = original.copy()
    variants, headings = diversify_unicycle_prefixes(original, [0.0, 0.0], velocity, 0.0, 0.25, limits)
    np.testing.assert_array_equal(original, snapshot)
    assert len(variants) > 0
    assert len(np.unique(np.round(variants[:, 0, 2:], 5), axis=0)) >= 2
    speeds = np.linalg.norm(variants[..., 2:], axis=-1)
    increments = np.diff(np.concatenate([np.full((len(variants), 1), np.linalg.norm(velocity)), speeds], axis=1), axis=1)
    assert increments.max() <= limits.acceleration * 0.25 + 1e-6
    assert increments.min() >= -limits.deceleration * 0.25 - 1e-6
    assert np.abs(np.diff(headings, axis=1)).max() <= limits.max_yaw_rate * 0.25 + 1e-6
    np.testing.assert_allclose(variants[..., :2], np.cumsum(variants[..., 2:] * 0.25, axis=1), atol=1e-6)
