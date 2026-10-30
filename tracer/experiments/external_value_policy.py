from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tracer.response.executable_value import response_risk_admissible, value_action


def value_policy_digest(model):
    if model is None:
        return None
    encoded = json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _allows_collection_reference(model):
    return (
        model.get("reference_policy_role") == "frozen_collection_behavior"
        and model.get("reference_value_model") is not None
    )


def value_policy_stages(model):
    stages, seen = [], set()
    while model is not None:
        if id(model) in seen:
            raise ValueError("executable value reference policies cannot contain cycles")
        seen.add(id(model))
        reference = model.get("reference_value_model")
        expected = model.get("training_behavior_value_sha256")
        if expected != value_policy_digest(reference):
            raise ValueError("executable value reference differs from its training behavior")
        stages.append(model)
        model = reference
    return list(reversed(stages))


def _selection_support(candidates, selected, diagnostics, stage, admissible_mask, scores):
    candidates = list(candidates)
    if not 0 <= selected < len(candidates):
        raise ValueError("executable value reference action is outside the candidate pool")
    support = np.ones(len(candidates), dtype=bool)
    if admissible_mask is not None:
        supplied = np.asarray(admissible_mask, dtype=bool)
        if supplied.shape != (len(candidates),):
            raise ValueError("executable value safety support must align with candidates")
        if not supplied[selected]:
            raise ValueError("executable value reference action is not safety-admissible")
        support &= supplied
    if stage.get("response_risk_guard", False):
        support &= response_risk_admissible(diagnostics, selected)
    if stage.get("reference_value_model") is not None:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(candidates),) or not np.isfinite(scores[selected]):
            raise ValueError("planner-aligned value support requires finite reference scores")
        tolerance = float(stage.get("planner_score_tolerance", 0.0))
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("planner score tolerance must be finite and nonnegative")
        support &= np.isfinite(scores) & (scores <= scores[selected] + tolerance + 1e-8)
    return support


def select_value_policy(model, features, names, candidates, headings, scores, selected, execute_steps, diagnostics,
                        admissible_mask=None):
    from tracer.experiments.external_replanning_supervision import distinct_prefix_candidates

    for stage in value_policy_stages(model):
        if names != stage["feature_names"]:
            raise ValueError("learned value feature contract differs from training")
        shortlist = distinct_prefix_candidates(candidates, headings, scores, selected, execute_steps, stage["branch_count"])
        support = _selection_support(candidates, selected, diagnostics, stage, admissible_mask, scores)[shortlist]
        local, _ = value_action(stage, features[shortlist], 0, stage["minimum_gain"], support)
        selected = shortlist[local]
    return int(selected)


def select_scheduled_value_policy(model, features, names, candidates, headings, scores, selected,
                                  execute_steps, diagnostics, window, intervention_counts, admissible_mask=None):
    from tracer.experiments.external_replanning_supervision import distinct_prefix_candidates

    stages = value_policy_stages(model)
    counts = list(intervention_counts)
    if len(counts) != len(stages) or any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts
    ):
        raise ValueError("scheduled value policy requires one nonnegative intervention count per stage")
    trace = []
    for index, stage in enumerate(stages):
        before = int(selected)
        active = value_policy_active(stage, window, counts[index])
        if active:
            if names != stage["feature_names"]:
                raise ValueError("learned value feature contract differs from training")
            shortlist = distinct_prefix_candidates(
                candidates, headings, scores, selected, execute_steps, stage["branch_count"]
            )
            support = _selection_support(candidates, selected, diagnostics, stage, admissible_mask, scores)[shortlist]
            local, _ = value_action(stage, features[shortlist], 0, stage["minimum_gain"], support)
            selected = shortlist[local]
        changed = int(selected) != before
        counts[index] += int(changed)
        trace.append({
            "stage": index, "active": bool(active), "before": before,
            "after": int(selected), "changed": bool(changed),
        })
    return int(selected), counts, trace


def value_policy_active(model, window, intervention_count):
    if not isinstance(window, int) or window < 0:
        raise ValueError("executable value window must be a nonnegative integer")
    if not isinstance(intervention_count, int) or intervention_count < 0:
        raise ValueError("executable value intervention count must be a nonnegative integer")
    stride = model.get("deployment_stride", 1)
    maximum = model.get("maximum_episode_interventions")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("executable value deployment stride must be a positive integer")
    if maximum is not None and (
        not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1
    ):
        raise ValueError("executable value episode intervention limit must be a positive integer")
    return window % stride == 0 and (maximum is None or intervention_count < maximum)


def validate_value_runtime(model, runtime, response_checkpoint_sha256):
    if (model is not None
            and model.get("selection_objective") in {
                "safe-completion-lexicographic-v1", "pairwise-reliability-classifier-v1",
                "pairwise-reliability-effect-v1",
            }
            and model.get("reference_value_model") is not None
            and not _allows_collection_reference(model)):
        raise ValueError("reliable deployment cannot treat an unaccepted learned collection policy as its safe reference")
    for stage in value_policy_stages(model):
        _validate_value_stage(stage, runtime, response_checkpoint_sha256)


def _validate_value_stage(model, runtime, response_checkpoint_sha256):
    reliable_objectives = {
        "safe-completion-lexicographic-v1", "pairwise-reliability-classifier-v1",
        "pairwise-reliability-effect-v1",
    }
    if (model.get("selection_objective") in reliable_objectives
            and model.get("reference_value_model") is not None
            and not _allows_collection_reference(model)):
        raise ValueError("reliable deployment cannot treat an unaccepted learned collection policy as its safe reference")
    if model.get("selection_objective") in reliable_objectives and (
        "deployment_stride" not in model or "maximum_episode_interventions" not in model
    ):
        raise ValueError("reliable deployment must declare its audited cadence and episode intervention limit")
    value_policy_active(model, 0, 0)
    if model["response_checkpoint_sha256"] != response_checkpoint_sha256:
        raise ValueError("executable value head must use its frozen response checkpoint")
    for path, digest in model["parent_artifact_hashes"].items():
        artifact = Path(path)
        if not artifact.exists() or hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
            raise ValueError(f"executable value parent artifact changed: {path}")
    allowed = {
        "output_dir", "cases", "case_start", "belief_selection_audit", "planner_sensitivity_audit",
        "belief_evidence_audit", "ranking_audit_windows", "candidate_debug_windows",
    }
    parent = model["parent_parameters"]
    actual = vars(runtime)
    differences = sorted(name for name in set(parent) | set(actual)
                         if name not in allowed and parent.get(name) != actual.get(name))
    if differences:
        raise ValueError(f"executable value runtime differs from its frozen parent: {differences}")
