from __future__ import annotations

import csv
import hashlib
import itertools
import json

import numpy as np
import pytest

from sage.experiments.aggregate_socialgym2_tracer import METRICS, aggregate, factual_memory_contrasts
from sage.experiments.socialgym2_physics_contract import verify_physics_audits
from sage.experiments.socialgym2_policy_roles import POLICY_ROLE_SCHEMA, socialgym2_policy_roles


RUNTIME = {"protocol": "socialgym2-joint-synchronous-runtime-v5", "sha256": "frozen"}
POLICIES = ["tracer-persistent", "tracer-reset", "tracer-uniform", "tracer-no-pair"]


def seal(path):
    artifacts = {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in ("metadata.json", "episode_metrics.csv", "belief_evidence.csv")
    }
    if (path / "belief_trace.csv").is_file():
        artifacts["belief_trace.csv"] = hashlib.sha256((path / "belief_trace.csv").read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps({"status": "COMPLETED", "artifacts": artifacts}))


@pytest.fixture
def inputs(tmp_path):
    paths = []
    for policy in POLICIES:
        path = tmp_path / policy
        path.mkdir()
        metadata = {
            "protocol": "tracer-socialgym2-joint-belief-synchronous-v5",
            "parameters": {"seed": 2027, "scenario": "intersection", "num_agents": 3,
                           "num_humans": 2, "partial_observation": True,
                           "max_steps": 2, "policies": [policy]},
            "joint_response_contract": {"controllable_robots": 3, "response_checkpoint_robots": 3,
                                        "pair_interactions": True, "same_state_uniform_counterfactual": True,
                                        "pair_interactions_by_policy": {
                                            policy: socialgym2_policy_roles([policy], True)[policy]["pair_interactions"]},
                                        "policy_role_schema": POLICY_ROLE_SCHEMA,
                                        "policy_roles_by_policy": socialgym2_policy_roles([policy], True)},
            "clock_contract": {"simulator_delta_t": 0.025, "planning_window_dt": 0.4,
                               "simulator_steps_per_planning_window": 16,
                               **{name: True for name in (
                                   "observed_transition_used_for_belief_update", "zero_step_reset",
                                   "post_step_observation", "clock_verified_every_window",
                                   "pedestrian_rng_seeded_per_episode", "all_controllable_robots_exposed_to_pedestrians",
                               )}},
            "runtime_contract": RUNTIME, "algorithm_sources": {"sage/core.py": "fixed"},
            "checkpoint_hashes": {"response": "response-fixed", "sage": "sage-fixed"},
            "calibration_contract": {"artifact_sha256": "calibration-fixed"},
        }
        (path / "metadata.json").write_text(json.dumps(metadata))
        rows = []
        for episode in range(2):
            rows.append({**{metric: 0.0 for metric in METRICS}, "policy": policy, "seed": 2027,
                         "episode": episode, "episode_seed": 2027000 + episode,
                         "scenario": "intersection", "scenario_hash": f"config{episode}",
                         "initial_state_hash": f"state{episode}",
                         "initial_scene_snapshot": json.dumps({"episode": episode}), "steps": 2,
                         "executed_sim_ticks": 32, "clock_max_abs_error": 0,
                         "last_window_sim_ticks": 16, "terminal_reached": False})
        with (path / "episode_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (path / "belief_evidence.csv").write_text("policy,step\n")
        (path / "belief_trace.csv").write_text(
            f"policy,seed,episode,step,track_id,action_vector\n{policy},2027,0,0,0,\"[0,0,1]\"\n"
        )
        seal(path)
        paths.append(path)
    return paths


def test_development_aggregate_accepts_one_policy_per_source(inputs, tmp_path):
    result = aggregate(inputs, tmp_path / "aggregate")
    assert result["summary"]["paired_keys"] == 2
    assert result["summary"]["run_class"] == "development_partial_lattice"
    assert result["manifest"]["records"] == 8
    assert result["metadata"]["source_policy_role_metadata"]["legacy_sources"] == 0
    with (tmp_path / "aggregate" / "belief_trace.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["scenario"] for row in rows} == {"intersection"}


def test_aggregate_ignores_worker_paths_but_requires_reset_snapshot(inputs, tmp_path):
    for index, path in enumerate(inputs):
        metrics_path = path / "episode_metrics.csv"
        with metrics_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["scenario_hash"] = f"worker-{index}-{row['episode']}"
        with metrics_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        seal(path)

    result = aggregate(inputs, tmp_path / "aggregate")

    expected_mismatches = (len(inputs) - 1) * 2
    assert result["summary"]["raw_scenario_hash_mismatch_count"] == expected_mismatches
    assert result["metadata"]["pairing_audit"]["raw_scenario_hash_mismatch_count"] == expected_mismatches

    metrics_path = inputs[1] / "episode_metrics.csv"
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["initial_scene_snapshot"] = json.dumps({"episode": "different"})
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    seal(inputs[1])
    with pytest.raises(ValueError, match="initial_scene_snapshot"):
        aggregate(inputs, tmp_path / "mismatched")


def test_aggregate_rejects_tampered_trace(inputs, tmp_path):
    with (inputs[0] / "belief_trace.csv").open("a") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        aggregate(inputs, tmp_path / "aggregate")


def test_legacy_policy_roles_are_flagged_for_hash_bound_correction(inputs, tmp_path):
    for path in inputs:
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        contract = metadata["joint_response_contract"]
        contract.pop("policy_role_schema")
        contract.pop("policy_roles_by_policy")
        policy = metadata["parameters"]["policies"][0]
        contract["pair_interactions_by_policy"] = {policy: policy != "tracer-no-pair"}
        metadata_path.write_text(json.dumps(metadata))
        seal(path)

    result = aggregate(inputs, tmp_path / "aggregate")

    role_metadata = result["metadata"]["source_policy_role_metadata"]
    assert role_metadata["legacy_sources"] == len(inputs)
    assert role_metadata["corrected_sources"] == 0
    assert role_metadata["legacy_boundary"] is not None


def test_corrected_official_control_is_not_labelled_as_joint_response(inputs, tmp_path):
    path = inputs[1]
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["parameters"]["policies"] = ["official-goalone"]
    roles = socialgym2_policy_roles(["official-goalone"], True)
    contract = metadata["joint_response_contract"]
    contract["pair_interactions_by_policy"] = {"official-goalone": None}
    contract["policy_roles_by_policy"] = roles
    contract["same_state_uniform_counterfactual"] = False
    metadata_path.write_text(json.dumps(metadata))
    metrics_path = path / "episode_metrics.csv"
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["policy"] = "official-goalone"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    seal(path)

    result = aggregate(inputs, tmp_path / "aggregate")

    official = result["metadata"]["policy_roles"]["official-goalone"]
    assert official["response_planner_evaluated"] is False
    assert official["pair_interactions"] is None


def test_native_external_policy_has_independent_provenance_and_action_usage(inputs, tmp_path):
    path = inputs[1]
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["parameters"]["policies"] = ["social-ref"]
    metadata["algorithm_sources"] = {"sage/external_adapter.py": "external-fixed"}
    metadata["external_controller"] = {
        "policy": "social-ref",
        "source_hashes": {"submodules/pips/src/policy/social_ref.cpp": "pips-fixed"},
    }
    roles = socialgym2_policy_roles(["social-ref"], True)
    contract = metadata["joint_response_contract"]
    contract["pair_interactions_by_policy"] = {"social-ref": None}
    contract["policy_roles_by_policy"] = roles
    contract["same_state_uniform_counterfactual"] = False
    metadata_path.write_text(json.dumps(metadata))
    metrics_path = path / "episode_metrics.csv"
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["policy"] = "social-ref"
        row["external_raw_action_counts"] = "[3, 2, 1, 0]"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    seal(path)

    result = aggregate(inputs, tmp_path / "aggregate")

    policy = result["summary"]["policies"]["social-ref"]
    assert policy["native_action_count"] == 12
    assert policy["native_action_usage"]["GoAlone"]["rate"] == 0.5
    assert result["metadata"]["external_controller_provenance"]["social-ref"] == {
        "submodules/pips/src/policy/social_ref.cpp": "pips-fixed"
    }


def test_aggregate_rejects_tampered_result_before_writing(inputs, tmp_path):
    with (inputs[0] / "episode_metrics.csv").open("a") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        aggregate(inputs, tmp_path / "aggregate")
    assert not (tmp_path / "aggregate").exists()


@pytest.mark.parametrize("field", ["runtime_contract", "algorithm_sources", "checkpoint_hashes"])
def test_aggregate_rejects_mixed_frozen_inputs(inputs, tmp_path, field):
    path = inputs[-1] / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field]["changed"] = "different"
    path.write_text(json.dumps(metadata))
    seal(inputs[-1])
    with pytest.raises(ValueError, match=field):
        aggregate(inputs, tmp_path / "aggregate")


def test_formal_aggregate_requires_physical_evidence(inputs, tmp_path):
    with pytest.raises(ValueError, match="physical audits"):
        aggregate(inputs, tmp_path / "aggregate", expected_policies=POLICIES,
                  expected_seeds=["2027"], expected_scenarios=["intersection"], expected_episodes=2)


def test_formal_aggregate_rejects_missing_scenes(inputs, tmp_path):
    with pytest.raises(ValueError, match="lattice mismatch"):
        aggregate(inputs, tmp_path / "aggregate", expected_policies=POLICIES,
                  expected_seeds=["2027"], expected_scenarios=["intersection"], expected_episodes=3)


@pytest.fixture
def physics_path(tmp_path):
    branches = []
    for action in list(itertools.product((0, 1), repeat=3)) + [(0, 0, 0)]:
        humans = np.zeros((2, 2, 2))
        humans[:, :, 0] += sum(action) * 0.1
        branches.append({"action": action, "initial_hash": "fixed", "human_trajectory": humans.tolist(),
                         "robot_trajectory": np.zeros((2, 3, 2)).tolist()})
    artifact = {"protocol": "socialgym2-joint-physical-intervention-audit-v4", "status": "PASSED",
                "runtime_contract": RUNTIME, "repeat_tolerance_meters": 1e-4,
                "scenario": "intersection", "ticks": 2,
                "records": [{"seed": seed, "branches": branches} for seed in range(4)]}
    path = tmp_path / "physics.json"
    path.write_text(json.dumps(artifact))
    return path


def test_physics_contract_recomputes_actual_trajectories(physics_path):
    verified = verify_physics_audits([physics_path], RUNTIME, ["intersection"], minimum_ticks=2)
    assert verified[0]["max_repeat_human_distance"] == 0
    artifact = json.loads(physics_path.read_text())
    artifact["records"][0]["branches"][-1]["human_trajectory"][0][0][0] += 0.02
    physics_path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="repeatability"):
        verify_physics_audits([physics_path], RUNTIME, ["intersection"], minimum_ticks=2)


def test_physics_contract_rejects_short_smoke(physics_path):
    with pytest.raises(ValueError, match="shorter"):
        verify_physics_audits([physics_path], RUNTIME, ["intersection"], minimum_ticks=960)


def test_physics_contract_rejects_duplicate_seed(physics_path):
    artifact = json.loads(physics_path.read_text())
    artifact["records"][1]["seed"] = artifact["records"][0]["seed"]
    physics_path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="episode seed"):
        verify_physics_audits([physics_path], RUNTIME, ["intersection"], minimum_ticks=2)


def test_factual_memory_scores_before_update_and_preserves_empty_state():
    uniform = json.dumps([-float(np.log(3))] * 3)
    first = {"policy": "tracer-uniform", "seed": "2027", "scenario": "intersection",
             "episode": "0", "track_id": "human-0", "step": "0", "initial_state_hash": "scene0",
             "sim_steps_executed": "16", "decision_log_belief": uniform, "posterior_log_belief": uniform,
             "factual_mode_log_likelihood": json.dumps([0.0, -4.0, -4.0])}
    first_result = factual_memory_contrasts([first], 1.0)
    assert first_result["contrasts"]["uniform-minus-persistent"]["mean"] == pytest.approx(0)
    rows = [{**first, "step": str(step)} for step in range(4)]
    result = factual_memory_contrasts(rows, 1.0)
    assert result["independent_observed_scenes"] == 1
    assert result["evidence_rows"] == 4
    assert result["contrasts"]["uniform-minus-persistent"]["mean"] > 0
    assert result["contrasts"]["reset-minus-persistent"]["mean"] > 0
    assert factual_memory_contrasts([], 1.0)["status"] == "NO_OBSERVED_UNIFORM_EVIDENCE"
    with pytest.raises(ValueError, match="duplicate"):
        factual_memory_contrasts([first, first], 1.0)
    with pytest.raises(ValueError, match="partial terminal"):
        factual_memory_contrasts([{**first, "sim_steps_executed": "1"}], 1.0)
