import pytest
import torch

from sage.experiments.audit_component_mode_switch import aggregate_switch_contrasts
from sage.experiments.eval_mode_switch_benchmark import _evaluate_split, _switch_sample
from sage.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig, TrajectoryRecurrentBelief, TrajectoryRecurrentConfig
from sage.response.persistent_trajectory import PersistentTrajectorySpec


def test_mode_switch_raw_scores_reproduce_method_means(monkeypatch):
    torch.manual_seed(91)
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(history=4, horizon=4, num_robots=2, hidden_dim=16, dropout=0.)).eval()
    recurrent = TrajectoryRecurrentBelief(TrajectoryRecurrentConfig(horizon=4, num_robots=2, execute_steps=1, hidden_dim=16)).eval()
    monkeypatch.setattr("sage.experiments.eval_mode_switch_benchmark._load_models", lambda *_: (model, recurrent))
    spec = PersistentTrajectorySpec(history=4, horizon=4, windows=4, num_robots=2, execute_steps=1)
    first = _switch_sample(spec, 0, 2)
    later = _switch_sample(spec, 0, 3)
    torch.testing.assert_close(first["target_future_vel"][:2], later["target_future_vel"][:2])
    rows = []
    summary = _evaluate_split(None, spec, 2, 2, "cpu", .5, episode_records=rows)
    for item in summary:
        selected = [row for row in rows if row["method"] == item["method"] and row["phase"] == "all"]
        for metric in ("trajectory_ade", "trajectory_fde", "predictive_nll"):
            assert sum(row[metric] for row in selected) / len(selected) == pytest.approx(item[metric], abs=1e-6)
    uniform = [row for row in rows if row["method"] == "no-belief"]
    assert all(row["decision_brier"] == pytest.approx(2. / 3.) for row in uniform)
    expanded = [{**row, "model_seed": seed} for seed in (1, 2, 3) for row in rows]
    contrasts = aggregate_switch_contrasts(expanded, resamples=20)
    assert all(row["independent_episodes"] == 2 for row in contrasts)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_switch_contrasts(expanded + [expanded[0]])
    with pytest.raises(ValueError, match="align"):
        aggregate_switch_contrasts(expanded[1:])
