from __future__ import annotations

import numpy as np
import pytest
import hashlib
import json
from types import SimpleNamespace

from sage.experiments.calibrate_socialgym2_belief import _score, calibrate
from sage.experiments.run_socialgym2_tracer_loop import _validate_calibration_artifact


def test_calibration_compares_the_same_unscaled_predictive_density() -> None:
    groups = [{"rows": [{"likelihood": np.array([-2.0, -5.0, -8.0])}]}]
    assert _score(groups, 0.001) == _score(groups, 0.05)


def test_calibration_rewards_useful_past_evidence_not_small_temperature() -> None:
    groups = [{"rows": [{"likelihood": np.array([-2.0, -5.0, -8.0])}] * 4}]
    low_score, count = _score(groups, 0.001)
    high_score, _ = _score(groups, 0.05)
    assert high_score < low_score
    assert count == 4


@pytest.mark.parametrize("scales", [(), (0.0,), (-1.0,), (float("nan"),)])
def test_calibration_rejects_invalid_scales(tmp_path, scales) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        calibrate([], tmp_path / "calibration.json", scales)


def test_calibration_runtime_contract_matches_writer_and_rejects_overlap(tmp_path) -> None:
    checkpoint = tmp_path / "response.pt"
    checkpoint.write_bytes(b"checkpoint")
    artifact = {
        "protocol": "tracer-socialgym2-joint-belief-calibration-v1",
        "source_protocol": "tracer-socialgym2-joint-belief-synchronous-v5",
        "phase": "validation",
        "split": "validation",
        "validation_seeds": [2201],
        "sim_dt": 0.025,
        "sim_steps_per_planning_window": 16,
        "planning_window_dt": 0.4,
        "num_agents": 3,
        "num_humans": 2,
        "source_checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "selected": {"likelihood_scale": 0.05},
    }
    artifact_path = tmp_path / "calibration.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    args = SimpleNamespace(seed=2027, sim_dt=0.025, sim_steps_per_window=16,
                           env_dt=0.4, num_agents=3, num_humans=2, likelihood_scale=0.05)
    assert _validate_calibration_artifact(artifact_path, checkpoint, args) == artifact
    args.seed = 2201
    with pytest.raises(ValueError, match="overlaps"):
        _validate_calibration_artifact(artifact_path, checkpoint, args)
