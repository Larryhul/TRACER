import json

import pytest
import torch

from sage.belief.correlated_emission import CorrelatedEmission, sequence_emission_likelihoods
from sage.experiments.calibrate_component_memory import component_protocol_id, load_component_calibration
from sage.experiments.component_emission_protocol import load_emission_noise
from sage.experiments.eval_component_factorial import FACTORIAL_POLICIES, audit_shared_emissions
from sage.experiments.eval_persistent_closed_loop import _evaluate_policy
from sage.experiments.formal_manifest import sha256_file
from sage.experiments.train_component_emission_noise import factual_emission_dataset, fit_emission_noise
from sage.response.closed_loop import ClosedLoopSpec, PlanningCostConfig
from sage.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig, trajectory_mode_log_likelihood


def test_diagonal_limit_matches_original_likelihood():
    torch.manual_seed(316)
    means = torch.randn(4, 3, 3, 2, dtype=torch.float64)
    log_std = torch.randn_like(means) * .2
    observation = torch.randn(4, 3, 2, dtype=torch.float64)
    model = CorrelatedEmission(1., 0., 0., 3, .4)
    likelihood, state = model.step(means, log_std, observation)
    torch.testing.assert_close(likelihood, trajectory_mode_log_likelihood(means, log_std, observation))
    repeated, _ = model.step(means, log_std, observation, state)
    torch.testing.assert_close(repeated, likelihood)


def test_sequential_likelihood_matches_full_joint_gaussian():
    torch.manual_seed(823)
    episodes, windows, modes, steps = 2, 4, 3, 3
    means = torch.randn(episodes, windows, modes, steps, 2, dtype=torch.float64)
    log_std = torch.randn_like(means) * .1
    observation = torch.randn(episodes, windows, steps, 2, dtype=torch.float64)
    data = {"means": means, "log_std": log_std, "target": observation}
    noise, offset, correlation = .8, .6, .45
    actual = sequence_emission_likelihoods(data, noise, offset, correlation).sum(1)
    times = torch.arange(windows).repeat_interleave(steps)
    covariance = offset * correlation ** (times[:, None] - times[None]).abs().double()
    covariance += torch.eye(windows * steps, dtype=torch.float64) * noise
    distribution = torch.distributions.MultivariateNormal(torch.zeros(windows * steps, dtype=torch.float64), covariance)
    residual = ((observation[:, :, None] - means) * (-log_std).exp()).permute(0, 2, 4, 1, 3).flatten(-2)
    expected = distribution.log_prob(residual).sum(-1) - log_std.sum((1, 3, 4))
    torch.testing.assert_close(actual, expected)


def test_correlated_updates_are_causal_and_identity_separated():
    torch.manual_seed(103)
    data = {"means": torch.randn(2, 5, 3, 3, 2), "log_std": torch.zeros(2, 5, 3, 3, 2), "target": torch.randn(2, 5, 3, 2)}
    original = sequence_emission_likelihoods(data, .8, .6, .4)
    changed = {key: value.clone() for key, value in data.items()}
    changed["target"][0, 3:] += 5.
    updated = sequence_emission_likelihoods(changed, .8, .6, .4)
    torch.testing.assert_close(updated[0, :3], original[0, :3])
    torch.testing.assert_close(updated[1], original[1])
    assert not torch.equal(updated[0, 3:], original[0, 3:])
    for episode in range(2):
        individual = sequence_emission_likelihoods({key: value[episode:episode + 1] for key, value in data.items()}, .8, .6, .4)
        torch.testing.assert_close(individual[0], original[episode])


def test_correlated_emission_validates_clock_and_parameters():
    with pytest.raises(ValueError, match="variances"):
        CorrelatedEmission(-1., .2, .5, 3, .4)
    with pytest.raises(ValueError, match="finite"):
        CorrelatedEmission(1., float("nan"), .5, 3, .4)
    model = CorrelatedEmission(1., .2, .5, 3, .4)
    with pytest.raises(ValueError, match="executed prefix"):
        model.step(torch.zeros(3, 4, 2), torch.zeros(3, 4, 2), torch.zeros(4, 2))


@pytest.mark.parametrize("architecture", ["legacy", "tracer"])
def test_shared_correlated_emission_preserves_factorial_attribution(architecture):
    torch.manual_seed(51)
    response = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(history=4, horizon=4, num_robots=2, hidden_dim=16, dropout=0., architecture=architecture)).eval()
    spec = ClosedLoopSpec(history=4, horizon=4, windows=3, num_robots=2, num_candidates=6, execute_steps=2)
    common = dict(model=response, spec=spec, episodes=2, device="cpu", planning_cost=PlanningCostConfig(task_steps=2), likelihood_scale=.5,
                  prefix_objective=True, evaluation_mode="same_state", response_gate_min=1., response_gate_max=1.)
    trace = []
    original = _evaluate_policy(**common, policy=FACTORIAL_POLICIES[0], captured_reference_trace=trace)
    noise = CorrelatedEmission(.8, .6, .4, 2, spec.dt)
    policies = {}
    for policy in FACTORIAL_POLICIES:
        policies[policy.name] = _evaluate_policy(**common, policy=policy, reference_trace=trace, emission_noise_model=noise)
    assert policies["uniform-joint"]["selected_candidates"] == original["selected_candidates"]
    assert audit_shared_emissions(policies)["status"] == "pass"
    for episode in range(2):
        for window in range(3):
            likelihoods = [policies[policy.name]["window_records"][episode][window]["executed_mode_log_likelihood"] for policy in FACTORIAL_POLICIES]
            assert all(value == likelihoods[0] for value in likelihoods)
    with pytest.raises(ValueError, match="clock"):
        _evaluate_policy(**common, policy=FACTORIAL_POLICIES[0], reference_trace=trace, emission_noise_model=CorrelatedEmission(.8, .6, .4, 1, spec.dt))


@pytest.mark.parametrize("architecture", ["legacy", "tracer"])
def test_factual_training_consumes_only_selected_executed_responses(architecture):
    torch.manual_seed(612)
    response = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(history=4, horizon=4, num_robots=2, hidden_dim=16, dropout=0., architecture=architecture)).eval()
    data = {"history": torch.randn(2, 3, 4, 2), "current": torch.randn(2, 3, 5, 2, 4), "candidates": torch.randn(2, 3, 5, 2, 4, 4), "responses": torch.randn(2, 3, 5, 4, 2), "mode": torch.tensor([0, 2])}
    trace = [{"episode": episode, "window": window, "executed_candidate": 0} for episode in range(2) for window in range(3)]
    original = factual_emission_dataset(response, data, trace, 2, "cpu")
    data["responses"][:, :, 1:] += 100.
    data["responses"][:, :, 0, 2:] += 100.
    changed = factual_emission_dataset(response, data, trace, 2, "cpu")
    for key in original:
        torch.testing.assert_close(original[key], changed[key])
    fitted = fit_emission_noise(original, iterations=2)
    assert fitted["noise_variance"] > 0 and 0 <= fitted["correlation"] < 1
    with pytest.raises(ValueError, match="duplicate"):
        factual_emission_dataset(response, data, trace + trace[:1], 2, "cpu")


def test_correlated_artifact_and_calibration_are_bound(tmp_path):
    checkpoint = tmp_path / "response.pt"
    checkpoint.write_bytes(b"frozen response")
    raw = {"simulation": {}, "planning": {}, "candidate_source": {"type": "synthetic"}}
    spec = ClosedLoopSpec()
    noise_path = tmp_path / "emission_noise.json"
    record = {"schema": "component-correlated-emission-v1", "status": "COMPLETED", "changed_inputs": [],
              "emission_checkpoint_sha256": sha256_file(checkpoint), "component_protocol_id": component_protocol_id(raw, True), "fitting_data_seeds": [2, 3],
              "parameters": {"noise_variance": .8, "offset_variance": .6, "correlation": .4, "execute_steps": spec.execute_steps, "dt": spec.dt}}
    noise_path.write_text(json.dumps(record))
    assert load_emission_noise(noise_path, checkpoint, raw, 4).noise_variance == .8
    with pytest.raises(ValueError, match="separate"):
        load_emission_noise(noise_path, checkpoint, raw, 3)
    calibration = {"schema": "component-memory-factual-v1", "status": "COMPLETED", "phase": "validation", "checkpoint_sha256": sha256_file(checkpoint),
                   "data_seed": 4, "emission_noise_sha256": sha256_file(noise_path), "selected": {"likelihood_scale": .5},
                   **{name: getattr(spec, name) for name in ("dt", "execute_steps", "num_robots", "horizon")}}
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    assert load_component_calibration(calibration_path, checkpoint, spec, 5, emission_noise_checkpoint=noise_path) == .5
    with pytest.raises(ValueError, match="correlated emission"):
        load_component_calibration(calibration_path, checkpoint, spec, 5)
    checkpoint.write_bytes(b"other response")
    with pytest.raises(ValueError, match="checkpoint"):
        load_emission_noise(noise_path, checkpoint, raw, 4)
