from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from sage.experiments.collect_external_planner_supervision import PlannerSupervisionAudit
from sage.experiments.external_controller_state import ExternalControllerState, FirstActionOverride
from sage.experiments.external_information_value import simulator_snapshot
from sage.experiments.external_value_policy import value_policy_digest
from sage.experiments.run_sicnav_tracer_loop import _run_episode


def distinct_prefix_candidates(candidates, headings, scores, selected, execute_steps, count):
    if execute_steps < 1 or execute_steps > candidates.shape[1] or count < 1:
        raise ValueError("continuation needs a valid executed prefix and positive branch count")
    scores = np.asarray(scores)
    if scores.shape != (len(candidates),) or not 0 <= selected < len(candidates) or not np.isfinite(scores[selected]):
        raise ValueError("continuation requires the supported factual selection")
    prefixes = candidates[:, :execute_steps].reshape(len(candidates), -1)
    if headings is not None:
        prefixes = np.concatenate([prefixes, headings[:, :execute_steps]], axis=1)
    supported = np.flatnonzero(np.isfinite(scores))
    order = [int(selected), *supported[np.argsort(scores[supported], kind="stable")].tolist()]
    result = []
    for index in order:
        if not any(np.allclose(prefixes[index], prefixes[previous], atol=1e-6, rtol=0) for previous in result):
            result.append(index)
        if len(result) == count:
            break
    return result


def continuation_candidates(candidates, headings, scores, selected, execute_steps, count, learned_behavior):
    reference = int(selected)
    indices = distinct_prefix_candidates(candidates, headings, scores, reference, execute_steps, count)
    if learned_behavior and selected not in indices:
        raise ValueError("factual learned selection differs from the deployment prefix shortlist")
    return indices, reference


class ContinuationTrace:
    def __init__(self, expected=None):
        self.expected = expected
        self.records = []
        self.pending_window = None
        self.env = None
        self.first_verified = False

    def observe(self, **values):
        self.env = values["env"]
        self.pending_window = int(values["window"])
        if not self.first_verified and self.expected is not None:
            for name in ("histories", "current", "candidates", "global_candidates"):
                if not np.allclose(values[name], self.expected[name], atol=2e-6, rtol=0):
                    raise ValueError(f"continuation changed factual first-window {name}")
            if not torch.allclose(values["beliefs"], self.expected["beliefs"], atol=2e-6, rtol=0):
                raise ValueError("continuation changed the factual decision belief")
            actual = values["decision_scores"].cpu().numpy()
            expected = self.expected["decision_scores"].cpu().numpy()
            if not np.array_equal(np.isfinite(actual), np.isfinite(expected)):
                raise ValueError("continuation changed the factual decision support")
            finite = np.isfinite(actual)
            if not np.allclose(actual[finite], expected[finite], atol=2e-6, rtol=0):
                raise ValueError("continuation changed factual candidate scores")
        self.first_verified = True

    def verify_execution(self, env, steps):
        if self.pending_window is None or steps < 1:
            raise ValueError("continuation execution has no corresponding decision")
        self.records.append((self.pending_window, simulator_snapshot(env).copy()))
        self.pending_window = None


class ReplanningSupervisionAudit(PlannerSupervisionAudit):
    def __init__(self, output, thresholds, stride, adapter, rollout_windows, branch_count,
                 executable_value_model=None, case_windows=None):
        super().__init__(output, thresholds, stride)
        if rollout_windows < 2 or branch_count < 2:
            raise ValueError("replanning supervision requires multiple windows and distinct branches")
        self.adapter = adapter
        self.rollout_windows = int(rollout_windows)
        self.branch_count = int(branch_count)
        self.controller_state = None
        self.expected_factual = {}
        self.continuation_records = []
        self.factual_continuation_checks = 0
        self.current_key = None
        self.executable_value_model = copy.deepcopy(executable_value_model)
        self.executable_value_sha256 = value_policy_digest(self.executable_value_model)
        self.expected_outcomes = {}
        self.factual_terminal_checks = 0
        self.case_windows = None if case_windows is None else set(case_windows)

    def _targeted(self, case, window):
        return self.case_windows is None or (int(case), int(window)) in self.case_windows

    def capture_controller_state(self, **values):
        if values.get("executable_value_sha256") != self.executable_value_sha256:
            raise ValueError("collection and continuation executable value policies must match")
        self.current_key = (int(values["case"]), int(values["windows"]))
        self.controller_state = (
            ExternalControllerState.capture(**values)
            if values["windows"] % self.stride == 0 and self._targeted(values["case"], values["windows"])
            else None
        )

    @torch.no_grad()
    def observe(self, **values):
        if not self._targeted(values["case"], values["window"]):
            return
        before_count = len(self.records)
        super().observe(**values)
        if len(self.records) == before_count:
            return
        state = self.controller_state
        if state is None or state.windows != values["window"]:
            raise ValueError("replanning supervision requires the pre-decision controller state")
        if not np.array_equal(simulator_snapshot(state.environment[0]), simulator_snapshot(values["env"])):
            raise ValueError("captured controller state differs from the factual decision environment")
        runtime = copy.copy(values["args"])
        if getattr(runtime, "include_incumbent", False):
            raise ValueError("prefix-only continuation requires a controller independent of the discarded candidate tail")
        runtime.windows = min(runtime.windows, state.windows + self.rollout_windows)
        runtime.max_steps = min(runtime.max_steps, state.steps + self.rollout_windows * runtime.execute_steps)
        runtime.ranking_audit_windows = 0
        runtime.candidate_debug_windows = 0
        runtime.response_audit_stride = 0
        runtime.belief_selection_audit = False
        runtime.planner_sensitivity_audit = False
        runtime.belief_evidence_audit = False
        indices, reference = continuation_candidates(
            values["global_candidates"], values["headings"], values["decision_scores"].cpu().numpy(),
            values["selected"], runtime.execute_steps, self.branch_count, self.executable_value_model is not None,
        )
        records = []
        for index in indices:
            trace = ContinuationTrace(values)
            intervention = FirstActionOverride(values["global_candidates"], values["headings"], index)
            try:
                with state.random_state.isolated():
                    result = _run_episode(
                        values["model"], self.adapter, values["policy"], Path(runtime.external_root).resolve(),
                        values["case"], Path(runtime.env_config).resolve(), runtime, values["seed"],
                        decision_audit=trace, controller_state=state, initial_action=intervention,
                        executable_value_model=self.executable_value_model,
                    )
            finally:
                if trace.env is not None:
                    trace.env.close()
            if not trace.first_verified or not trace.records:
                raise ValueError("counterfactual continuation did not execute a verified prefix")
            goal = state.environment[0].robot.get_goal_position()
            terminal_distance = float(np.linalg.norm(trace.records[-1][1][0, :2] - goal))
            record = {
                "candidate": index, "first_window_verified": True,
                "elapsed_steps": int(result["steps"] - state.steps),
                "elapsed_time": float(result["total_time"] - state.environment[0].global_time),
                "success": result["success"], "collision_free_success": result["collision_free_success"],
                "collision": result["collision"], "wall_collision": result["wall_collision"],
                "environment_timeout": result["environment_timeout"],
                "timeout": result["timeout"],
                "terminal_goal_distance": terminal_distance,
                "distance_progress": float(np.linalg.norm(simulator_snapshot(state.environment[0])[0, :2] - goal)) - terminal_distance,
            }
            if state.outcome_state is not None:
                record.update({
                    "outcome_history_restored": True,
                    "stagnation_time": result["stagnation_time"] - state.outcome_state["exposure"]["stagnation_time"],
                    "human_velocity_variation": result["human_velocity_variation"] - state.outcome_state["exposure"]["human_velocity_variation"],
                })
            records.append(record)
            if index == values["selected"]:
                if state.outcome_state is not None and self.rollout_windows >= values["args"].windows:
                    fields = ("success", "collision_free_success", "collision", "wall_collision", "timeout",
                              "total_time", "task_progress", "stagnation_time", "human_velocity_variation",
                              "terminal_goal_distance")
                    self.expected_outcomes.setdefault(int(values["case"]), []).append(
                        {name: result[name] for name in fields})
                for window, snapshot in trace.records:
                    key = (int(values["case"]), int(window))
                    if key in self.expected_factual and not np.allclose(self.expected_factual[key], snapshot, atol=2e-5, rtol=0):
                        raise ValueError("overlapping parent continuations disagree before live verification")
                    self.expected_factual[key] = snapshot
        row = {
            "case": values["case"], "window": values["window"], "seed": values["seed"],
            "parent_selected": values["selected"], "requested_windows": self.rollout_windows,
            "decision_reference_selected": reference,
            "branches": records,
            "behavior_value_sha256": self.executable_value_sha256,
        }
        destination = self.output.parent / "continuations"
        destination.mkdir(exist_ok=True)
        path = destination / f"seed{values['seed']}_case{values['case']}_window{values['window']}.json"
        if path.exists():
            raise ValueError("continuation group already exists")
        path.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        self.continuation_records.append(row)

    def verify_execution(self, env, steps):
        super().verify_execution(env, steps)
        if self.current_key in self.expected_factual:
            expected = self.expected_factual.pop(self.current_key)
            if not np.allclose(simulator_snapshot(env), expected, atol=2e-5, rtol=0):
                raise ValueError("parent replanning branch differs from the actual subsequent closed loop")
            self.factual_continuation_checks += 1

    def finalize(self):
        if self.expected_factual:
            raise ValueError("some counterfactual parent continuation windows were not verified live")
        if self.expected_outcomes:
            raise ValueError("some parent terminal safety and exposure outcomes were not verified live")
        return {
            "continuation_groups": len(self.continuation_records),
            "continuation_branches": sum(len(row["branches"]) for row in self.continuation_records),
            "factual_continuation_checks": self.factual_continuation_checks,
            "factual_terminal_checks": self.factual_terminal_checks,
        }

    def verify_episode(self, result):
        case = int(result["case"])
        expected = self.expected_outcomes.get(case, [])
        for outcomes in expected:
            for name, value in outcomes.items():
                if not np.isclose(result[name], value, atol=2e-5, rtol=0):
                    raise ValueError(f"factual terminal {name} differs from the actual completed episode")
        self.factual_terminal_checks += len(expected)
        self.expected_outcomes.pop(case, None)
