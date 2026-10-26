from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sage.experiments.plot_supervisor_revision import (
    METHODS,
    WINDOWS,
    build_fig3,
    build_fig4,
    cluster_bands,
    execution_status,
    full_pair_limit,
    marker_indices,
    style,
    trajectory_grid,
)


def episode_totals():
    rows = []
    for split in ("id", "ood"):
        for episode in range(5):
            for method_index, method in enumerate(METHODS):
                for window in WINDOWS:
                    count = episode + 1
                    value = 0.01 * (episode + window + method_index)
                    rows.append({"split": split, "episode_id": episode, "method": method,
                                 "window": window, "count": count, "sum": count * value})
    return pd.DataFrame(rows)


def test_cluster_bootstrap_preserves_pooled_active_record_means():
    totals = episode_totals()
    summary = cluster_bands(totals, repeats=100)
    expected = totals.groupby(["split", "method", "window"])[["sum", "count"]].sum()
    actual = summary.set_index(["split", "method", "window"]).sort_index()
    np.testing.assert_allclose(actual["mean"], expected["sum"] / expected["count"])
    assert len(summary) == 42
    assert summary.total_episode_clusters.eq(5).all()
    assert (summary.lower <= summary.upper).all()


def test_extra_correlated_seed_rows_do_not_shrink_episode_intervals():
    totals = episode_totals()
    first = cluster_bands(totals, repeats=100)
    totals[["sum", "count"]] *= 5
    replicated = cluster_bands(totals, repeats=100)
    np.testing.assert_allclose(first[["mean", "lower", "upper"]], replicated[["mean", "lower", "upper"]])


def test_shared_cluster_draws_preserve_cross_method_dependence():
    result = cluster_bands(episode_totals(), repeats=100)
    first = result[result.method.eq("persistent")][["mean", "lower", "upper"]].to_numpy()
    second = result[result.method.eq("reset")][["mean", "lower", "upper"]].to_numpy()
    np.testing.assert_allclose(second - first, 0.01)


def test_duplicate_cluster_rows_are_rejected():
    totals = episode_totals()
    with pytest.raises(ValueError, match="duplicate"):
        cluster_bands(pd.concat([totals, totals.iloc[:1]]), repeats=100)


def test_h1_limit_includes_every_outlier():
    records = pd.DataFrame({"x": [0.1] * 1000 + [0.852], "y": [0.09] * 1000 + [0.2]})
    assert full_pair_limit(records) >= records[["x", "y"]].to_numpy().max()
    assert full_pair_limit(records) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        full_pair_limit(pd.DataFrame({"x": [np.nan], "y": [0.2]}))


def test_h3_collision_time_is_not_reported_as_team_completion():
    status, label, _ = execution_status({"collision": 1, "team_success": 1}, 10.88)
    assert status == "Collision"
    assert label == "Elapsed: 10.88 s"
    assert execution_status({"collision": 0, "team_success": 1}, 16.4)[1] == "Team time: 16.40 s"


def test_timestamp_markers_include_exact_final_tick():
    times = np.asarray([0, 0.09, 0.17, 0.25, 0.34])
    indices = marker_indices(times)
    assert indices[0] == 0
    assert indices[-1] == len(times) - 1
    with pytest.raises(ValueError):
        marker_indices(np.asarray([0, 0.2, 0.1]))


def test_compact_row_has_equal_square_panels_and_centered_groups(tmp_path):
    pd.DataFrame({"x": [0.1, 0.852], "y": [0.09, 0.2]}).to_csv(tmp_path / "fig3_h1_paired_records.csv", index=False)
    pd.DataFrame([{"split": split, "policy": policy, "task": -20, "safety": 40, "human": 42, "collision": 99}
                  for split in ("id", "ood") for policy in ("reset-joint", "persistent-joint", "oracle-joint")]).to_csv(
        tmp_path / "fig3_h3_relative_changes.csv", index=False)
    style()
    figure, audit = build_fig3(tmp_path)
    figure.canvas.draw()
    sizes = [(axis.get_window_extent().width, axis.get_window_extent().height) for axis in figure.axes]
    np.testing.assert_allclose(sizes[0], sizes[1], atol=1e-8)
    assert sizes[0][0] == pytest.approx(sizes[0][1])
    right = figure.axes[1]
    first_group = [container.patches[0].get_x() + container.patches[0].get_width() / 2 for container in right.containers]
    assert np.mean(first_group) == pytest.approx(0, abs=1e-12)
    assert audit["hidden_records"] == 0
    plt.close(figure)


def test_h2_has_all_six_curves_and_unambiguous_semantics():
    style()
    figure, _ = build_fig4(cluster_bands(episode_totals(), repeats=100))
    axis = figure.axes[0]
    assert len(axis.lines) == 6
    assert "Replanning window" in axis.get_xlabel()
    assert "Posterior concentration" in axis.get_ylabel()
    assert len(figure.legends) == 0
    assert len(axis.get_legend().get_texts()) == 6
    assert axis.get_legend().get_frame().get_linewidth() < axis.spines["left"].get_linewidth()
    assert figure.get_size_inches()[0] == pytest.approx(7.16)
    plt.close(figure)


@pytest.mark.parametrize("width", [3.4, 7.0])
def test_trajectory_panels_are_square_at_both_publication_widths(width):
    style()
    figure, axes = trajectory_grid(width)
    figure.canvas.draw()
    for axis in axes:
        extent = axis.get_window_extent()
        assert extent.width == pytest.approx(extent.height)
        assert min(tick.get_fontsize() for tick in axis.get_xticklabels()) >= 7.5
    plt.close(figure)
