import numpy as np
import pytest

from sage.experiments.navigation_exposure import NavigationExposure


def test_exposure_uses_observed_velocity_changes_and_actual_time():
    exposure = NavigationExposure()
    exposure.observe(2., 1.9, [[1., 0.]], [[.8, 0.]], .25)
    exposure.observe(1.9, 1.9, [[.8, 0.]], [[.8, .3]], .1)
    assert exposure.stagnation_time == pytest.approx(.1)
    assert exposure.human_velocity_variation == pytest.approx(.5)


def test_exposure_snapshot_restores_full_episode_and_future_increment():
    parent = NavigationExposure(.5, .7)
    restored = NavigationExposure(**parent.snapshot())
    restored.observe(2., 2.1, [[1., 0.]], [[.8, 0.]], .25)
    assert restored.stagnation_time - parent.stagnation_time == pytest.approx(.25)
    assert restored.human_velocity_variation - parent.human_velocity_variation == pytest.approx(.2)
    assert parent.snapshot() == {"stagnation_time": .5, "human_velocity_variation": .7}


def test_exposure_rejects_missing_or_nonfinite_observations():
    exposure = NavigationExposure()
    with pytest.raises(ValueError, match="aligned finite"):
        exposure.observe(1., 1., [[1., 0.]], [], .25)
    with pytest.raises(ValueError, match="aligned finite"):
        exposure.observe(1., np.nan, [[1., 0.]], [[1., 0.]], .25)
