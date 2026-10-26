from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sage.experiments.plot_paper_evidence import (
    H3_POLICIES,
    bootstrap_weights,
    cumulative_safe_completion,
    decision_gain_matrix,
    load_h2,
    mean_interval,
    paired_h1,
    panel_caption,
    panel_grid,
    reliability_statistics,
    save_figure,
    strength_groups,
    validate_h3,
)


def test_completion_denominator_retains_failures_and_timeouts():
    values = cumulative_safe_completion([2, 3, 1, 8], [1, 1, 0, 0], [0, 1, 2, 3, 8])
    np.testing.assert_allclose(values.mean(axis=0), [0, 0, 0.25, 0.5, 0.5])
    assert np.all(np.diff(values, axis=1) >= 0)


def test_completion_includes_exact_terminal_timestamp():
    np.testing.assert_array_equal(cumulative_safe_completion([1.025], [1], [1., 1.025]), [[0, 1]])


@pytest.mark.parametrize("times,safe", [([np.nan], [1]), ([-1], [1]), ([1], [0.5])])
def test_invalid_completion_values_are_rejected(times, safe):
    with pytest.raises(ValueError):
        cumulative_safe_completion(times, safe, [0, 1])


def h3_fixture():
    rows = []
    for policy in H3_POLICIES:
        for episode in range(2):
            rows.append({"policy": policy, "seed": 3109, "episode": episode, "scenario": "open",
                         "episode_seed": 3109000 + episode, "initial_state_hash": f"scene{episode}",
                         "initial_collision": 0., "team_success": 1., "collision": float(episode),
                         "collision_free_team_success": float(not episode)})
    return pd.DataFrame(rows)


def test_h3_pairing_requires_all_methods_and_identical_reset_state():
    data = h3_fixture()
    validate_h3(data)
    with pytest.raises(ValueError, match="incomplete"):
        validate_h3(data.iloc[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_h3(pd.concat([data, data.iloc[:1]]))
    data.loc[0, "initial_state_hash"] = "other"
    with pytest.raises(ValueError, match="reset mismatch"):
        validate_h3(data)


def test_h3_arrival_with_collision_cannot_be_safe_completion():
    data = h3_fixture()
    data.loc[1, "collision_free_team_success"] = 1
    with pytest.raises(ValueError, match="inconsistent"):
        validate_h3(data)


def test_bootstrap_is_reproducible_and_uses_complete_cluster_rows():
    first = bootstrap_weights(4, 30, 7)
    np.testing.assert_array_equal(first, bootstrap_weights(4, 30, 7))
    np.testing.assert_array_equal(first.sum(axis=1), np.full(30, 4))
    mean, lower, upper = mean_interval(np.ones((4, 3)), first)
    np.testing.assert_allclose([mean, lower, upper], np.ones((3, 3)))
    with pytest.raises(ValueError):
        mean_interval(np.ones((3, 2)), first)


def h1_fixture():
    rows = []
    for mode in ("none", "additive", "nonlinear"):
        for scene in range(4):
            for seed in (2027, 2028):
                for method in ("joint", "matched-additive"):
                    rows.append({"effect_mode": mode, "data_seed": 328731, "split": "formal_test",
                                 "base_scene_id": scene, "model_seed": seed, "method": method,
                                 "interaction_target_rms": float(scene) if mode == "nonlinear" else 0.,
                                 "interaction_rmse": 0.1 if method == "joint" else (0.2 if mode == "nonlinear" else 0.)})
    return pd.DataFrame(rows)


def test_h1_model_seeds_are_not_counted_as_independent_scenes():
    scenes, seeds = paired_h1(h1_fixture())
    assert seeds == 2
    assert len(scenes) == 12
    np.testing.assert_allclose(scenes.loc[scenes.effect_mode.eq("nonlinear"), "gain"], 0.1)
    np.testing.assert_allclose(scenes.loc[scenes.effect_mode.eq("none"), "gain"], -0.1)


def test_h1_missing_pair_or_truth_mismatch_is_rejected():
    data = h1_fixture()
    with pytest.raises(ValueError, match="incomplete"):
        paired_h1(data.iloc[:-1])
    data.loc[0, "interaction_target_rms"] = 2
    with pytest.raises(ValueError, match="truth differs"):
        paired_h1(data)


def test_h1_strength_bins_retain_extreme_and_negative_cases():
    data = pd.DataFrame({"strength": np.arange(32) / 10, "gain": np.arange(32) - 15})
    grouped, edges = strength_groups(data, 8)
    assert len(grouped) == 32
    assert grouped.iloc[-1].strength_bin == 7
    assert grouped.iloc[0].strength_bin == 0
    assert grouped.groupby("strength_bin").size().eq(4).all()
    np.testing.assert_allclose(edges[[0, -1]], [0., 3.1])


def h2_fixture():
    rows = []
    for episode in range(2):
        for seed in (2027, 2028):
            for entity in range(2):
                for window in (1, 2):
                    for method in ("persistent", "reset", "recurrent"):
                        rows.append({"split": "id", "episode_id": episode, "model_seed": seed,
                                     "entity_id": entity, "window": window, "method": method,
                                     "active": 1, "main_evaluation": 1,
                                     "predictive_trajectory_nll": 0.5 if method == "reset" else 0.3,
                                     "decision_confidence": 0.5, "decision_mode_correct": episode})
    return pd.DataFrame(rows)


def test_h2_prediction_gain_pairs_before_averaging_seeds_and_entities():
    data = h2_fixture()
    gain = decision_gain_matrix(data, "persistent")
    assert gain.shape == (2, 2)
    np.testing.assert_allclose(gain.to_numpy(), 0.2)
    data = data.drop(data[(data.method == "reset")].index[0])
    with pytest.raises(ValueError, match="missing"):
        decision_gain_matrix(data, "persistent")


def test_calibration_bootstrap_resamples_episode_not_each_observation():
    data = h2_fixture().query("method == 'persistent'")
    weights = np.asarray([[2, 0], [0, 2]])
    rows, ece, accuracy = reliability_statistics(data, weights, minimum_count=1, minimum_episodes=1)
    nonempty = [row for row in rows if row["count"]]
    assert len(nonempty) == 1
    assert accuracy == 0.5 and ece == 0.
    assert nonempty[0]["ci95_low"] == pytest.approx(0.025)
    assert nonempty[0]["ci95_high"] == pytest.approx(0.975)


def test_calibration_includes_confidence_one_and_retains_sparse_bin_in_ece():
    data = pd.DataFrame({"episode_id": [0, 1, 2], "decision_confidence": [1., 0.4, 0.4],
                         "decision_mode_correct": [0, 0, 0]})
    rows, ece, _ = reliability_statistics(data, bootstrap_weights(3, 20), minimum_count=2, minimum_episodes=1)
    assert rows[-1]["count"] == 1
    assert not rows[-1]["plotted"]
    assert sum(row["count"] for row in rows) == 3
    assert ece == pytest.approx(0.6)


def test_h2_loader_excludes_inactive_window_zero_and_never_uses_posterior(tmp_path):
    data = h2_fixture()
    extra = data.iloc[:3].copy()
    extra["window"] = 0
    extra["main_evaluation"] = 0
    data = pd.concat([data, extra])
    data["posterior_confidence"] = 0.999
    for seed, rows in data.groupby("model_seed"):
        directory = tmp_path / f"seed{seed}"
        directory.mkdir()
        rows.to_csv(directory / "raw_metrics.csv", index=False)
    loaded, paths = load_h2(tmp_path)
    assert len(paths) == 2
    assert loaded.window.min() == 1
    assert "posterior_confidence" not in loaded.columns
    assert loaded.decision_confidence.eq(0.5).all()


@pytest.mark.parametrize("rows", [1, 2])
def test_reference_layout_keeps_square_panels_and_unobstructed_captions(tmp_path, rows):
    figure, axes = panel_grid(rows)
    assert plt.rcParams["font.family"] == ["STIXGeneral"]
    for index, axis in enumerate(axes.flat):
        assert axis.get_box_aspect() == 1
        axis.set_xlabel("Elapsed simulation time (s)")
        axis.set_ylabel("Collision-free team completion (%)")
        panel_caption(axis, f"({chr(97 + index)}) Panel")
    save_figure(figure, tmp_path, f"layout{rows}")
    assert all((tmp_path / f"layout{rows}.{suffix}").is_file() for suffix in ("png", "pdf", "svg"))
