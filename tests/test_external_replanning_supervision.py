import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tracer.experiments.external_controller_state import ExternalControllerState, FirstActionOverride, RandomState
from tracer.experiments.external_replanning_supervision import ContinuationTrace, ReplanningSupervisionAudit, distinct_prefix_candidates
from tracer.experiments.external_information_value import simulator_snapshot
from tracer.response.executable_value import FEATURE_VERSION, executable_value_features, fit_paired_value, value_action
from tracer.experiments.train_executable_value import evaluate_value
from tracer.experiments.summarize_executable_value_navigation import paired_navigation
from tracer.response.executable_value import value_representation
from tracer.response.executable_value import prefix_value_feature_indices
from tracer.response.executable_value import response_risk_admissible
from tracer.experiments.merge_planner_supervision import merge
from tracer.experiments.audit_replanning_target_alignment import prefix_cost_alignment
from tracer.experiments.confirm_executable_value import confirmation_acceptance, confirmation_cases, confirmation_reference_model
from tracer.experiments.collect_external_planner_supervision import parse_case_windows
from tracer.experiments.train_executable_value_ensemble import resample_scene_groups
from tracer.experiments.external_replanning_supervision import continuation_candidates
from tracer.experiments.external_value_policy import value_policy_active, value_policy_digest, validate_value_runtime, select_value_policy, select_scheduled_value_policy, value_policy_stages
from tracer.experiments.verify_executable_reference_identity import verify_reference_identity


def test_branch_random_state_restores_parent_and_replays_all_generators():
    before = RandomState.capture()
    with before.isolated():
        first = (np.random.rand(4), random.random(), torch.rand(4))
    after = RandomState.capture()
    assert np.array_equal(before.numpy[1], after.numpy[1])
    assert before.numpy[2:] == after.numpy[2:]
    assert before.python == after.python
    assert torch.equal(before.torch_cpu, after.torch_cpu)
    with before.isolated():
        second = (np.random.rand(4), random.random(), torch.rand(4))
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_branch_random_state_restores_after_failure():
    before = RandomState.capture()
    with pytest.raises(RuntimeError):
        with before.isolated():
            np.random.rand(4)
            random.random()
            torch.rand(4)
            raise RuntimeError("branch failed")
    after = RandomState.capture()
    assert np.array_equal(before.numpy[1], after.numpy[1])
    assert before.python == after.python
    assert torch.equal(before.torch_cpu, after.torch_cpu)


def test_capture_isolated_state_preserves_aliases_and_controller_memory():
    robot = SimpleNamespace(position=np.zeros(2))
    env = SimpleNamespace(robot=robot)
    histories = [[np.ones(2)]]
    beliefs = torch.log(torch.tensor([[0.2, 0.8]]))
    rng = np.random.default_rng(91)
    state = ExternalControllerState.capture(
        environment=(env, robot), histories=histories, beliefs=beliefs, adaptive_states=None,
        rng=rng, previous_plan=np.ones((3, 4)), fallback_streak=2,
        steps=6, windows=3, case=12, seed=2027, policy="tracer-persistent",
    )
    robot.position += 4
    histories[0][0] += 2
    beliefs.fill_(0)
    assert state.environment[0].robot is state.environment[1]
    assert np.array_equal(state.environment[1].position, np.zeros(2))
    assert np.array_equal(state.histories[0][0], np.ones(2))
    assert torch.allclose(state.beliefs.exp(), torch.tensor([[0.2, 0.8]]))
    resumed = np.random.default_rng()
    resumed.bit_generator.state = copy.deepcopy(state.candidate_rng)
    assert resumed.random() == rng.random()
    state.validate(12, 2027, "tracer-persistent")
    with pytest.raises(ValueError, match="preserve"):
        state.validate(13, 2027, "tracer-persistent")


def test_prefix_branch_selection_excludes_duplicate_tails_and_unsafe_actions():
    candidates = np.zeros((5, 3, 4))
    candidates[1, 1:, 2] = 9
    candidates[2, 0, 2] = 1
    candidates[3, 0, 2] = 2
    candidates[4, 0, 2] = 3
    scores = np.array([2., 0., 1., np.inf, 4.])
    assert distinct_prefix_candidates(candidates, None, scores, 0, 1, 6) == [0, 2, 4]
    headings = np.zeros((5, 3))
    headings[1, 0] = 0.1
    assert distinct_prefix_candidates(candidates, headings, scores, 0, 1, 3) == [0, 1, 2]


def test_first_action_intervention_requires_exact_pool_headings_and_support():
    candidates = np.zeros((3, 4, 4))
    headings = np.zeros((3, 4))
    override = FirstActionOverride(candidates.copy(), headings.copy(), 1)
    assert override.validate(candidates, headings, np.array([1., 2., np.inf])) == 1
    with pytest.raises(ValueError, match="support"):
        override.validate(candidates, headings, np.array([1., np.inf, 0.]))
    with pytest.raises(ValueError, match="headings"):
        override.validate(candidates, None, np.zeros(3))
    with pytest.raises(ValueError, match="pool"):
        override.validate(candidates + 0.01, headings, np.zeros(3))


def test_learned_behavior_continuations_preserve_the_deployed_base_shortlist():
    candidates = np.zeros((5, 3, 4))
    candidates[:, 0, 2] = np.arange(5)
    scores = np.array([2., 0., 1., np.inf, np.inf])
    indices, reference = continuation_candidates(candidates, None, scores, 2, 1, 3, True)
    assert indices == [2, 1, 0] and reference == 2
    indices, reference = continuation_candidates(candidates, None, scores, 2, 1, 3, False)
    assert indices == [2, 1, 0] and reference == 2
    with pytest.raises(ValueError, match="supported factual selection"):
        continuation_candidates(candidates, None, scores, 4, 1, 3, True)


def test_continuation_state_cannot_silently_switch_learned_behavior():
    state = ExternalControllerState(
        environment=(), histories=[], beliefs=torch.zeros(1), adaptive_states=None,
        candidate_rng={}, random_state=RandomState.capture(), previous_plan=None,
        fallback_streak=0, steps=0, windows=0, case=1, seed=2, policy="tracer-persistent",
        executable_value_sha256="frozen-value",
        value_action_changes=1, value_action_change_details=[(4, 2, 3)],
        value_stage_action_changes=[2, 1],
        value_stage_action_change_details=[[(0, 1, 2), (4, 1, 2)], [(4, 2, 3)]],
    )
    state.validate(1, 2, "tracer-persistent", "frozen-value")
    assert state.value_action_changes == 1
    assert state.value_action_change_details == [(4, 2, 3)]
    assert state.value_stage_action_changes == [2, 1]
    assert state.value_stage_action_change_details == [[(0, 1, 2), (4, 1, 2)], [(4, 2, 3)]]
    with pytest.raises(ValueError, match="executable value policy"):
        state.validate(1, 2, "tracer-persistent")
    with pytest.raises(ValueError, match="executable value policy"):
        state.validate(1, 2, "tracer-persistent", "different-value")


def test_value_policy_digest_and_runtime_preserve_model_and_safety_contract(tmp_path):
    import hashlib

    artifact = tmp_path / "response.pt"
    artifact.write_bytes(b"frozen")
    parameters = {"safety_weight": 2., "execute_steps": 1, "output_dir": "old"}
    model = {"response_checkpoint_sha256": "response", "parent_parameters": parameters,
             "parent_artifact_hashes": {str(artifact): hashlib.sha256(artifact.read_bytes()).hexdigest()}}
    assert value_policy_digest(model) == value_policy_digest(dict(reversed(list(model.items()))))
    assert value_policy_digest(None) is None
    validate_value_runtime(model, SimpleNamespace(**{**parameters, "output_dir": "new"}), "response")
    with pytest.raises(ValueError, match="runtime differs"):
        validate_value_runtime(model, SimpleNamespace(**{**parameters, "safety_weight": 1.}), "response")
    with pytest.raises(ValueError, match="frozen response"):
        validate_value_runtime(model, SimpleNamespace(**parameters), "other")
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_value_runtime(model, SimpleNamespace(**parameters), "response")


def test_collection_rejects_a_different_live_learned_policy(tmp_path):
    model = {"coefficient": 2.}
    audit = ReplanningSupervisionAudit(tmp_path / "audit", [.8, 1.2], 1, None, 60, 6, model)
    model["coefficient"] = 3.
    assert audit.executable_value_sha256 == value_policy_digest({"coefficient": 2.})
    with pytest.raises(ValueError, match="policies must match"):
        audit.capture_controller_state(executable_value_sha256=value_policy_digest(model))


def test_policy_iteration_abstains_to_the_actual_behavior_action_not_base_argmin():
    behavior = {"feature_version": FEATURE_VERSION, "feature_names": ["prefix_speed"],
                "scale": [1.], "coefficients": [[-1., 0., 0.]], "branch_count": 2, "minimum_gain": .05}
    refinement = {**behavior, "coefficients": [[0., 0., 0.]], "reference_value_model": behavior,
                  "training_behavior_value_sha256": value_policy_digest(behavior)}
    features = np.arange(3., dtype=float)[:, None]
    candidates = np.zeros((3, 2, 4))
    candidates[:, 0, 2] = features[:, 0]
    scores = np.array([0., 1., 2.])
    assert value_action(refinement, features, 0, .05)[0] == 0
    selected = select_value_policy(refinement, features, ["prefix_speed"], candidates, None, scores, 0, 1, {})
    assert selected == 1
    assert len(value_policy_stages(refinement)) == 2
    changed = copy.deepcopy(refinement)
    changed["reference_value_model"]["minimum_gain"] = .5
    with pytest.raises(ValueError, match="reference differs"):
        value_policy_stages(changed)


def test_policy_iteration_cannot_add_unsafe_candidates_to_reference_support():
    behavior = {"feature_version": FEATURE_VERSION, "feature_names": ["prefix_speed"],
                "scale": [1.], "coefficients": [[-1., 0., 0.]], "branch_count": 3, "minimum_gain": .05}
    candidates = np.zeros((3, 2, 4))
    candidates[:, 0, 2] = np.arange(3.)
    selected = select_value_policy(behavior, np.arange(3.)[:, None], ["prefix_speed"], candidates,
                                   None, np.array([0., 1., np.inf]), 0, 1, {})
    assert selected == 1


def test_policy_iteration_preserves_planner_safety_support():
    behavior = {"feature_version": FEATURE_VERSION, "feature_names": ["prefix_speed"],
                "scale": [1.], "coefficients": [[-1., 0., 0.]], "branch_count": 3, "minimum_gain": .05}
    candidates = np.zeros((3, 2, 4))
    candidates[:, 0, 2] = np.arange(3.)
    selected = select_value_policy(
        behavior, np.arange(3.)[:, None], ["prefix_speed"], candidates, None,
        np.array([0., 1., 2.]), 0, 1, {}, admissible_mask=np.array([True, False, True])
    )
    assert selected == 2
    with pytest.raises(ValueError, match="not safety-admissible"):
        select_value_policy(
            behavior, np.arange(3.)[:, None], ["prefix_speed"], candidates, None,
            np.array([0., 1., 2.]), 1, 1, {}, admissible_mask=np.array([True, False, True])
        )


def test_refinement_preserves_planner_aligned_score_support():
    behavior = {"feature_version": FEATURE_VERSION, "feature_names": ["prefix_speed"],
                "scale": [1.], "coefficients": [[1., 0., 0.]], "branch_count": 3,
                "minimum_gain": .05}
    refinement = {**behavior, "reference_value_model": behavior,
                  "training_behavior_value_sha256": value_policy_digest(behavior),
                  "response_risk_guard": False}
    candidates = np.zeros((3, 2, 4))
    candidates[:, 0, 2] = np.asarray([0., -1., 1.])
    selected = select_value_policy(
        refinement, np.asarray([0., -1., 1.])[:, None], ["prefix_speed"], candidates, None,
        np.array([0., 1., 2.]), 0, 1, {}, admissible_mask=np.ones(3, dtype=bool)
    )
    assert selected == 1


def test_chained_policy_applies_each_stage_schedule_and_counts_refinement_only():
    behavior = {"feature_version": FEATURE_VERSION, "feature_names": ["prefix_speed"],
                "scale": [1.], "coefficients": [[-1., 0., 0.]], "branch_count": 3,
                "minimum_gain": .05}
    refinement = {**behavior, "coefficients": [[1., 0., 0.]], "branch_count": 2,
                  "reference_value_model": behavior,
                  "training_behavior_value_sha256": value_policy_digest(behavior),
                  "deployment_stride": 4, "maximum_episode_interventions": 1}
    features = np.arange(3., dtype=float)[:, None]
    candidates = np.zeros((3, 2, 4))
    candidates[:, 0, 2] = features[:, 0]
    scores = np.array([0., 1., 2.])
    selected, counts, trace = select_scheduled_value_policy(
        refinement, features, ["prefix_speed"], candidates, None, scores, 0, 1, {}, 1, [0, 0]
    )
    assert selected == 2 and counts == [1, 0]
    assert [stage["active"] for stage in trace] == [True, False]
    selected, counts, trace = select_scheduled_value_policy(
        refinement, features, ["prefix_speed"], candidates, None, scores, 0, 1, {}, 4, counts
    )
    assert selected == 0 and counts == [2, 1]
    assert trace[-1] == {"stage": 1, "active": True, "before": 2, "after": 0, "changed": True}
    selected, counts, trace = select_scheduled_value_policy(
        refinement, features, ["prefix_speed"], candidates, None, scores, 0, 1, {}, 8, counts
    )
    assert selected == 2 and counts == [3, 1]
    assert not trace[-1]["active"]


def test_value_policy_schedule_limits_actual_episode_interventions():
    assert value_policy_active({}, 3, 9)
    model = {"deployment_stride": 4, "maximum_episode_interventions": 1}
    assert value_policy_active(model, 0, 0)
    assert not value_policy_active(model, 3, 0)
    assert value_policy_active(model, 4, 0)
    assert not value_policy_active(model, 4, 1)
    with pytest.raises(ValueError, match="stride"):
        value_policy_active({"deployment_stride": 0}, 0, 0)
    with pytest.raises(ValueError, match="intervention limit"):
        value_policy_active({"maximum_episode_interventions": 0}, 0, 0)


def test_pairwise_reliability_runtime_requires_schedule_and_original_reference(tmp_path):
    artifact = tmp_path / "response.pt"
    artifact.write_bytes(b"frozen")
    import hashlib
    base = {
        "selection_objective": "pairwise-reliability-classifier-v1",
        "response_checkpoint_sha256": "response", "parent_parameters": {"output_dir": "old"},
        "parent_artifact_hashes": {str(artifact): hashlib.sha256(artifact.read_bytes()).hexdigest()},
        "training_behavior_value_sha256": None, "reference_value_model": None,
    }
    with pytest.raises(ValueError, match="cadence"):
        validate_value_runtime(base, SimpleNamespace(output_dir="new"), "response")
    scheduled = {**base, "deployment_stride": 4, "maximum_episode_interventions": 1}
    validate_value_runtime(scheduled, SimpleNamespace(output_dir="new"), "response")
    chained = {**scheduled, "reference_value_model": {}}
    chained["training_behavior_value_sha256"] = value_policy_digest({})
    with pytest.raises(ValueError, match="safe reference"):
        validate_value_runtime(chained, SimpleNamespace(output_dir="new"), "response")


def test_pairwise_reliability_allows_explicit_frozen_collection_reference(tmp_path):
    artifact = tmp_path / "response.pt"
    artifact.write_bytes(b"frozen")
    import hashlib
    reference = {
        "response_checkpoint_sha256": "response",
        "parent_parameters": {"output_dir": "old"},
        "parent_artifact_hashes": {str(artifact): hashlib.sha256(artifact.read_bytes()).hexdigest()},
    }
    model = {
        "selection_objective": "pairwise-reliability-effect-v1",
        "response_checkpoint_sha256": "response",
        "parent_parameters": {"output_dir": "old"},
        "parent_artifact_hashes": {str(artifact): hashlib.sha256(artifact.read_bytes()).hexdigest()},
        "training_behavior_value_sha256": value_policy_digest(reference),
        "reference_value_model": reference,
        "reference_policy_role": "frozen_collection_behavior",
        "deployment_stride": 4,
        "maximum_episode_interventions": 1,
    }
    validate_value_runtime(model, SimpleNamespace(output_dir="new"), "response")


def value_feature_example():
    candidates = np.zeros((4, 3, 4))
    candidates[:, :, 2] = np.arange(4)[:, None] * .2
    candidates[..., :2] = np.cumsum(candidates[..., 2:] * .25, axis=1)
    from tracer.response.executable_value import COMPONENTS
    return dict(
        candidates=candidates, headings=np.zeros((4, 3)), robot_position=np.zeros(2),
        robot_velocity=np.array([.1, 0.]), robot_goal=np.array([3., 0.]), robot_heading=0.,
        human_positions=np.array([[1., .5], [2., -.5]]), human_velocities=np.array([[0., -.1], [.1, .2]]),
        human_radii=np.array([.2, .3]), robot_radius=.3, beliefs=np.array([[.2, .3, .5], [.5, .2, .3]]),
        means=np.ones((2, 4, 3, 3, 2)) * .1, log_stds=np.full((2, 4, 3, 3, 2), -2.),
        diagnostics={name:np.arange(4) * .01 for name in COMPONENTS}, dt=.25, execute_steps=1,
    )


def test_value_features_only_observable_and_invariant_to_entity_order_and_translation():
    inputs = value_feature_example()
    original, names = executable_value_features(**inputs)
    assert original.shape == (4, len(names)) and np.isfinite(original).all()
    shuffled = copy.deepcopy(inputs)
    for name in ("human_positions", "human_velocities", "human_radii", "beliefs", "means", "log_stds"):
        shuffled[name] = shuffled[name][::-1]
    assert np.allclose(executable_value_features(**shuffled)[0], original)
    moved = copy.deepcopy(inputs)
    for name in ("robot_position", "robot_goal", "human_positions"):
        moved[name] += np.array([7., -4.])
    moved["candidates"][..., :2] += np.array([7., -4.])
    assert np.allclose(executable_value_features(**moved)[0], original)
    rotated = copy.deepcopy(inputs)
    rotation = np.array([[0., -1.], [1., 0.]])
    for name in ("robot_position", "robot_velocity", "robot_goal", "human_positions", "human_velocities", "means"):
        rotated[name] = rotated[name] @ rotation.T
    for interval in (slice(0, 2), slice(2, 4)):
        rotated["candidates"][..., interval] = rotated["candidates"][..., interval] @ rotation.T
    rotated["robot_heading"] += np.pi / 2
    rotated["headings"] += np.pi / 2
    assert np.allclose(executable_value_features(**rotated)[0], original)


def test_paired_value_fit_ignores_scene_return_offsets_and_predicts_time_advantage():
    groups=[]
    for case in range(5):
        features=np.array([[0., 0.], [1., 0.], [0., 1.]])
        targets=np.array([[4., 0., 0.], [3., 0., 0.], [5., 1., 1.]])
        targets[:, 0] += case * 10
        groups.append({"case":case, "features":features, "targets":targets, "parent":0})
    fitted=fit_paired_value(groups, .001)
    selected, predictions=value_action(fitted, features, 0)
    assert selected == 1 and predictions[1, 0] < -.99
    modified=copy.deepcopy(groups)
    for group in modified:
        group["targets"] += np.array([17., .1, .1])
    other=fit_paired_value(modified, .001)
    assert np.allclose(fitted["coefficients"], other["coefficients"])


def test_value_selector_rejects_predicted_failure_and_subthreshold_advantage():
    model={"feature_version":FEATURE_VERSION,"scale":[1.,1.,1.],"coefficients":np.eye(3).tolist()}
    features=np.array([[0.,0.,0.],[-2.,.1,0.],[-1.,0.,.1],[-.1,0.,0.]])
    assert value_action(model,features,0,.25)[0] == 0
    features[3,0]=-.5
    assert value_action(model,features,0,.25)[0] == 3


def test_value_features_reject_nonfinite_predictions_and_invalid_beliefs():
    inputs=value_feature_example()
    inputs["beliefs"] *= 2
    with pytest.raises(ValueError,match="normalized"):
        executable_value_features(**inputs)
    inputs=value_feature_example()
    inputs["means"][0,0,0,0,0]=np.nan
    with pytest.raises(ValueError,match="finite"):
        executable_value_features(**inputs)


def test_continuation_trace_rejects_changed_belief_scores_and_support():
    expected={name:np.ones((2,3)) for name in ("histories","current","candidates","global_candidates")}
    expected.update(beliefs=torch.log(torch.tensor([[.2,.3,.5]])),decision_scores=torch.tensor([1.,2.,float("inf")]))
    values={**expected,"env":object(),"window":4}
    trace=ContinuationTrace(expected)
    trace.observe(**values)
    assert trace.first_verified
    changed={**values,"beliefs":torch.log(torch.tensor([[.3,.2,.5]]))}
    with pytest.raises(ValueError,match="belief"):
        ContinuationTrace(expected).observe(**changed)
    changed={**values,"decision_scores":torch.tensor([1.,float("inf"),2.])}
    with pytest.raises(ValueError,match="support"):
        ContinuationTrace(expected).observe(**changed)
    changed={**values,"decision_scores":torch.tensor([1.,2.1,float("inf")])}
    with pytest.raises(ValueError,match="scores"):
        ContinuationTrace(expected).observe(**changed)


def test_parent_continuation_cannot_complete_without_matching_live_windows(tmp_path):
    agent=SimpleNamespace(
        get_position=lambda:np.array([0.,0.]),get_velocity=lambda:np.array([.1,0.]),
        get_goal_position=lambda:np.array([2.,0.]),theta=0.,v_pref=1.,
    )
    env=SimpleNamespace(robot=agent,humans=[agent])
    audit=ReplanningSupervisionAudit(tmp_path/'audit',[.8,1.2],1,None,3,3)
    audit.current_key=(3,2)
    audit.expected_factual[(3,2)]=simulator_snapshot(env)
    with pytest.raises(ValueError,match="not verified"):
        audit.finalize()
    audit.verify_execution(env,1)
    assert audit.finalize()['factual_continuation_checks']==1
    audit.expected_factual[(3,2)]=simulator_snapshot(env)+.1
    with pytest.raises(ValueError,match="subsequent closed loop"):
        audit.verify_execution(env,1)


def test_parent_terminal_safety_and_exposure_must_match_live_episode(tmp_path):
    audit = ReplanningSupervisionAudit(tmp_path / "audit", [.8, 1.2], 1, None, 60, 3)
    audit.expected_outcomes[1] = [{"collision_free_success": 1., "human_velocity_variation": .7}]
    with pytest.raises(ValueError, match="terminal safety"):
        audit.finalize()
    with pytest.raises(ValueError, match="terminal collision_free_success"):
        audit.verify_episode({"case": 1, "collision_free_success": 0., "human_velocity_variation": .7})
    audit.verify_episode({"case": 1, "collision_free_success": 1., "human_velocity_variation": .7})
    assert audit.finalize()["factual_terminal_checks"] == 1


def test_fallback_groups_never_train_or_change_under_value_selection():
    active={"case":1,"window":0,"features":np.array([[0.],[1.]]),"parent":0,
            "targets":np.array([[4.,0.,0.],[3.,0.,0.]]),"candidate_indices":[2,4],"rerank_allowed":True}
    fallback=copy.deepcopy(active)
    fallback.update(case=2,rerank_allowed=False)
    fallback['targets'][1,0]=1000.
    fit_active=fit_paired_value([active],.001)
    fit_both=fit_paired_value([active,fallback],.001)
    assert np.allclose(fit_active['coefficients'],fit_both['coefficients'])
    result=evaluate_value(fit_both,[active,fallback],.25)
    assert result['changed_groups']==1 and result['records'][1]['selected']==2
    assert result['records'][1]['time_difference']==0


def test_navigation_acceptance_checks_actual_safety_and_initial_hashes():
    baseline={"seed":"2027","case":"64","initial_scene_hash":"fixed","environment_dt":".25",
              "robot_kinematics":"unicycle","robot_initial_heading":"0","policy":"tracer-persistent",
              "collision_free_success":"1","collision":"0","wall_collision":"0","timeout":"0",
              "total_time":"5","task_progress":".95"}
    candidate={**baseline,"total_time":"4.5","executable_value_action_changes":"2"}
    result=paired_navigation([baseline],[candidate])
    assert result['observed_safety_not_worse'] and result['time_directionally_improved']
    collision={**candidate,"collision":"1","collision_free_success":"0"}
    assert not paired_navigation([baseline],[collision])['observed_safety_not_worse']
    with pytest.raises(ValueError,match="initial_scene_hash"):
        paired_navigation([baseline],[{**candidate,"initial_scene_hash":"other"}])


def test_nonlinear_value_representation_is_deterministic_and_preserves_parent_zero():
    groups=[]
    for case in range(4):
        features=np.array([[float(case),0.],[float(case),1.],[float(case),-1.]])
        targets=np.array([[5.,0.,0.],[4.+.1*case,0.,0.],[6.,1.,1.]])
        groups.append({"case":case,"features":features,"targets":targets,"parent":0})
    first=fit_paired_value(groups,.1,1.)
    second=fit_paired_value(groups,.1,1.)
    assert first['projection']==second['projection']
    features=groups[0]['features']
    assert value_representation(features,first).shape==(3,130)
    selected,prediction=value_action(first,features,0,.1)
    assert np.array_equal(prediction[0],np.zeros(3))
    assert selected==1


def test_continuation_shard_merge_requires_consistent_horizon_and_preserves_live_checks(tmp_path):
    splits={"train":[0,1],"val":[64],"test":[88]}
    shards=[]
    for case in (0,1):
        shard=tmp_path/f'shard{case}'
        for name in ('audit','inputs','continuations'):
            (shard/name).mkdir(parents=True)
        protocol={"split":"train","cases":[case],"planning":{},"dt":.25,"execute_steps":1,
                  "checkpoint_sha256":"frozen","seed":2027,"parameters":{"cases":1,"case_start":case},
                  "continuation_windows":60,"continuation_branches":6}
        (shard/'protocol.json').write_text(json.dumps(protocol))
        (shard/'manifest.json').write_text(json.dumps({"status":"COMPLETED","continuation_groups":1,
            "continuation_branches":4,"factual_continuation_checks":10}))
        row={"seed":2027,"case":case,"window":0,"live_prefix_verified":True}
        (shard/'audit/windows.jsonl').write_text(json.dumps(row)+'\n')
        for name in ('audit','inputs'):
            np.savez(shard/name/f'seed2027_case{case}_window0.npz',values=np.zeros(1))
        (shard/'continuations'/f'seed2027_case{case}_window0.json').write_text(json.dumps(row))
        shards.append(shard)
    merge(tmp_path/'merged',shards,splits,'train')
    merge(tmp_path/'subset',[shards[0]],splits,'train',[0])
    subset=json.loads((tmp_path/'subset/protocol.json').read_text(encoding="utf-8"))
    assert subset['cases']==[0] and subset['case_splits']==splits
    with pytest.raises(ValueError,match='original declared split'):
        merge(tmp_path/'invalid_subset',[shards[0]],splits,'train',[64])
    manifest=json.loads((tmp_path/'merged/manifest.json').read_text(encoding="utf-8"))
    assert manifest['continuation_groups']==2 and manifest['factual_continuation_checks']==20
    assert (tmp_path/'merged/continuations/seed2027_case1_window0.json').read_bytes() == (tmp_path/'shard1/continuations/seed2027_case1_window0.json').read_bytes()
    protocol['continuation_windows']=8
    (shards[1]/'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='continuation mismatch'):
        merge(tmp_path/'bad_horizon',shards,splits,'train')


def test_temporal_value_features_preserve_permutation_invariance_at_equal_distances():
    inputs=value_feature_example()
    inputs['human_positions']=np.array([[1.,.5],[1.,-.5]])
    inputs['human_velocities']=np.array([[.1,.2],[-.3,.4]])
    inputs['human_radii']=np.array([.2,.2])
    original,names=executable_value_features(**inputs)
    for name in ('human_positions','human_velocities','human_radii','beliefs','means','log_stds'):
        inputs[name]=inputs[name][::-1]
    actual,other_names=executable_value_features(**inputs)
    assert names==other_names and np.allclose(original,actual)


def test_value_feature_subset_ignores_unselected_observables():
    groups=[]
    for case in range(3):
        features=np.array([[0.,float(case)],[1.,float(case)+20.]])
        groups.append({'case':case,'features':features,'parent':0,'targets':np.array([[4.,0.,0.],[3.,0.,0.]])})
    model=fit_paired_value(groups,.01,feature_indices=[0])
    selected,first=value_action(model,groups[0]['features'],0,.25)
    changed=groups[0]['features'].copy()
    changed[:,1]+=np.array([1000.,-1000.])
    other,second=value_action(model,changed,0,.25)
    assert selected==other==1 and np.allclose(first,second)


def test_temporal_value_features_resolve_stop_turn_alignment_and_approach_timing():
    inputs=value_feature_example()
    inputs['candidates'][:]=0
    inputs['headings'][1,:]=np.pi/2
    features,names=executable_value_features(**inputs)
    assert features[0,names.index('prefix_goal_heading_cos')]>features[1,names.index('prefix_goal_heading_cos')]
    assert 'linear_closest_time' in names and 'response_closest_time' in names


def test_alignment_audit_distinguishes_safe_navigation_gain_from_open_loop_cost(tmp_path):
    (tmp_path/'audit').mkdir()
    snapshots=np.zeros((3,2,2,8))
    snapshots[:,0,0,0]=np.arange(3)
    np.savez(tmp_path/'audit/seed9_case64_window0.npz',snapshots=snapshots,
             factual_total=np.array([1.,2.,0.]),eligible=np.ones(3,dtype=bool))
    group={'case':64,'window':0,'rerank_allowed':True,'candidate_indices':[0,1,2],
           'parent':0,'targets':np.array([[5.,0.,0.],[4.,0.,0.],[3.,1.,1.]])}
    summary=prefix_cost_alignment([group],tmp_path,1,9)
    assert summary['safe_faster_alternatives']==1
    assert summary['safe_faster_not_recognized_by_open_loop']==1
    assert summary['open_loop_better_alternatives']==1
    assert summary['open_loop_better_but_no_safe_time_gain']==1


def test_prefix_value_features_ignore_discarded_candidate_tail_and_costs():
    inputs=value_feature_example()
    original,names=executable_value_features(**inputs)
    indices=prefix_value_feature_indices(names)
    assert len(indices)>0
    changed=copy.deepcopy(inputs)
    changed['candidates'][:,1:]+=4.
    changed['headings'][:,1:]+=.4
    changed['means'][:,:,:,1:]+=2.
    changed['log_stds'][:,:,:,1:]+=1.
    for name in changed['diagnostics']:
        changed['diagnostics'][name]+=3.
    actual,other_names=executable_value_features(**changed)
    assert names==other_names
    assert np.allclose(original[:,indices],actual[:,indices])
    assert not np.allclose(original,actual)


def test_confirmation_blocks_disclosed_cases_and_requires_prespecified_size():
    splits={'train':[0,1], 'val':[64,65], 'test':[88,89]}
    cases,excluded=confirmation_cases(300,128,splits,[160,200])
    assert cases==list(range(300,428)) and 160 in excluded
    with pytest.raises(ValueError,match='overlap'):
        confirmation_cases(160,64,splits,[160,200])
    with pytest.raises(ValueError,match='at least64'):
        confirmation_cases(300,32,splits,[])


def test_confirmation_uses_the_frozen_training_behavior_as_parent():
    reference = {"selection_objective": "legacy-reference"}
    candidate = {
        "selection_objective": "pairwise-reliability-effect-v1",
        "reference_value_model": reference,
        "training_behavior_value_sha256": value_policy_digest(reference),
    }
    assert confirmation_reference_model(candidate) == reference
    changed = copy.deepcopy(candidate)
    changed["reference_value_model"]["minimum_gain"] = 0.5
    with pytest.raises(ValueError, match="training behavior"):
        confirmation_reference_model(changed)


def test_targeted_continuation_windows_are_explicit_and_bounded():
    assert parse_case_windows(["502:8", "506:16"], {502, 506}, 60) == {(502, 8), (506, 16)}
    with pytest.raises(ValueError, match="unique"):
        parse_case_windows(["502:8", "502:8"], {502}, 60)
    with pytest.raises(ValueError, match="selected cases"):
        parse_case_windows(["503:8"], {502}, 60)
    with pytest.raises(ValueError, match="case:window"):
        parse_case_windows(["502"], {502}, 60)


def test_confirmation_acceptance_requires_safety_stall_and_robust_human_gain():
    parent = [dict(seed=1, case=case, human_velocity_variation=1., stagnation_time=0.) for case in range(128)]
    candidate = [dict(row, executable_value_action_changes=int(row["case"] < 6)) for row in parent]
    for row in candidate[:6]:
        row["human_velocity_variation"] = .75
    report = {"observed_safety_not_worse": True,
              "metrics": {"goal_shortfall": {"candidate_minus_parent": 0.},
                          "human_velocity_variation": {"case_bootstrap_ci95": [-.02, -.001]}}}
    assert confirmation_acceptance(report, parent, candidate)["primary_acceptance_passed"]
    candidate[0]["human_velocity_variation"] = -7.
    for row in candidate[1:6]:
        row["human_velocity_variation"] = 1.
    assert not confirmation_acceptance(report, parent, candidate)["primary_acceptance_passed"]
    candidate = [dict(row, executable_value_action_changes=int(row["case"] < 6)) for row in parent]
    for row in candidate[:6]:
        row["human_velocity_variation"] = .75
    candidate[0]["stagnation_time"] = .25
    assert not confirmation_acceptance(report, parent, candidate)["primary_acceptance_passed"]
    assert not confirmation_acceptance(
        {**report, "observed_safety_not_worse": False}, parent, candidate
    )["primary_acceptance_passed"]


def test_zero_refinement_identity_requires_same_inner_actions_and_outcomes():
    base = {
        "seed": "1", "case": "9", "initial_scene_hash": "scene", "environment_dt": ".25",
        "robot_kinematics": "unicycle", "robot_initial_heading": "0", "policy": "tracer-persistent",
        "collision_free_success": "1", "collision": "0", "wall_collision": "0", "timeout": "0",
        "total_time": "4", "task_progress": ".9",
    }
    parent = [{**base, "executable_value_stage_action_change_details": "[[[0, 1, 2]]]"}]
    candidate = [{
        **base, "executable_value_stage_action_change_details": "[[[0, 1, 2]], []]",
        "executable_value_action_changes": "0",
    }]
    report = verify_reference_identity(parent, candidate)
    assert report["reference_action_sequences_identical"]
    changed = copy.deepcopy(candidate)
    changed[0]["executable_value_stage_action_change_details"] = "[[[0, 1, 0]], []]"
    with pytest.raises(ValueError, match="behavior action sequence"):
        verify_reference_identity(parent, changed)


def test_scene_bootstrap_keeps_windows_together_and_does_not_cancel_duplicates():
    groups=[{'case':case,'window':window,'features':np.ones((2,1))} for case in (3,4,5) for window in (0,4)]
    generator=SimpleNamespace(choice=lambda cases,size,replace:np.array([3,3,5]))
    resampled=resample_scene_groups(groups,generator)
    assert len(resampled)==6
    assert [group['case'] for group in resampled]==[0,0,1,1,2,2]
    assert [group['original_case'] for group in resampled]==[3,3,3,3,5,5]
    assert [group['window'] for group in resampled]==[0,4,0,4,0,4]
    assert [group['case'] for group in groups]==[3,3,4,4,5,5]


def test_value_ensemble_abstains_when_time_gain_is_disputed():
    def member(gain):
        return {'feature_version':FEATURE_VERSION,'scale':[1.],'coefficients':[[gain,0.,0.]]}
    model={'feature_version':FEATURE_VERSION,'ensemble':[member(-.8),member(.2)],'uncertainty_multiplier':0.}
    features=np.array([[0.],[1.]])
    assert value_action(model,features,0,.05)[0]==1
    model['uncertainty_multiplier']=1.
    selected,predictions=value_action(model,features,0,.05)
    assert selected==0 and predictions[1,0]<0
    assert np.array_equal(predictions[0],np.zeros(3))


def test_pairwise_reliability_network_accepts_a_small_capacity_width():
    from tracer.experiments.train_pairwise_reliable_value import PairwiseReliabilityNetwork

    model = PairwiseReliabilityNetwork(12, width=8)
    predictions = model(torch.zeros(3, 12))
    assert tuple(predictions.shape) == (3, 2)


def test_value_ensemble_safety_disagreement_prevents_time_only_switch():
    first={'feature_version':FEATURE_VERSION,'scale':[1.],'coefficients':[[-1.,-.2,-.2]]}
    second={'feature_version':FEATURE_VERSION,'scale':[1.],'coefficients':[[-1.,.1,.1]]}
    model={'feature_version':FEATURE_VERSION,'ensemble':[first,second],'uncertainty_multiplier':1.}
    assert value_action(model,np.array([[0.],[1.]]),0,.05)[0]==0


def test_response_risk_guard_preserves_parent_and_rejects_either_cost_increase():
    diagnostics={'weighted_risk':np.array([.1,.05,.11,.05]),'weighted_safety':np.array([.2,.1,.1,.21])}
    support=response_risk_admissible(diagnostics,0)
    assert support.tolist()==[True,True,False,False]
    model={'feature_version':FEATURE_VERSION,'scale':[1.],'coefficients':[[-1.,0.,0.]]}
    features=np.arange(4.)[:,None]
    assert value_action(model,features,0,.05,support)[0]==1
    with pytest.raises(ValueError,match='preserve the parent'):
        value_action(model,features,0,.05,np.array([False,True,True,True]))
