from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RandomState:
    numpy: tuple
    python: object
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] | None

    @classmethod
    def capture(cls):
        return cls(np.random.get_state(), random.getstate(), torch.random.get_rng_state(),
                   torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)

    def restore(self):
        np.random.set_state(self.numpy)
        random.setstate(self.python)
        torch.random.set_rng_state(self.torch_cpu)
        if self.torch_cuda is not None:
            torch.cuda.set_rng_state_all(self.torch_cuda)

    @contextmanager
    def isolated(self):
        outside = RandomState.capture()
        try:
            self.restore()
            yield
        finally:
            outside.restore()


@dataclass
class ExternalControllerState:
    environment: tuple
    histories: list
    beliefs: torch.Tensor
    adaptive_states: object
    candidate_rng: dict
    random_state: RandomState
    previous_plan: np.ndarray | None
    fallback_streak: int
    steps: int
    windows: int
    case: int
    seed: int
    policy: str
    executable_value_sha256: str | None = None
    value_action_changes: int = 0
    value_action_change_details: list[tuple[int, int, int]] | None = None
    value_stage_action_changes: list[int] | None = None
    value_stage_action_change_details: list[list[tuple[int, int, int]]] | None = None
    outcome_state: dict | None = None

    @classmethod
    def capture(cls, *, environment, histories, beliefs, adaptive_states, rng, previous_plan,
                fallback_streak, steps, windows, case, seed, policy, executable_value_sha256=None,
                value_action_changes=0, value_action_change_details=None,
                value_stage_action_changes=None, value_stage_action_change_details=None,
                outcome_state=None):
        return cls(
            copy.deepcopy(environment), copy.deepcopy(histories), beliefs.detach().clone(),
            copy.deepcopy(adaptive_states), copy.deepcopy(rng.bit_generator.state),
            RandomState.capture(), None if previous_plan is None else previous_plan.copy(),
            int(fallback_streak), int(steps), int(windows), int(case), int(seed), str(policy),
            executable_value_sha256,
            int(value_action_changes),
            copy.deepcopy(value_action_change_details),
            copy.deepcopy(value_stage_action_changes),
            copy.deepcopy(value_stage_action_change_details),
            copy.deepcopy(outcome_state),
        )

    def validate(self, case, seed, policy, executable_value_sha256=None):
        if (case, seed, policy) != (self.case, self.seed, self.policy):
            raise ValueError("continuation must preserve the captured scene, seed and controller policy")
        if executable_value_sha256 != self.executable_value_sha256:
            raise ValueError("continuation must preserve the captured executable value policy")


@dataclass
class FirstActionOverride:
    candidates: np.ndarray
    headings: np.ndarray | None
    selected: int

    def validate(self, candidates, headings, decision_scores):
        if self.candidates.shape != candidates.shape or not np.allclose(self.candidates, candidates, atol=1e-6, rtol=0):
            raise ValueError("resumed candidate pool differs from the captured parent state")
        if (self.headings is None) != (headings is None) or (
            headings is not None and (self.headings.shape != headings.shape or
                                      not np.allclose(self.headings, headings, atol=1e-6, rtol=0))
        ):
            raise ValueError("resumed candidate headings differ from the captured parent state")
        if not 0 <= self.selected < len(candidates) or not np.isfinite(decision_scores[self.selected]):
            raise ValueError("offline first-action override must retain the parent's actual decision support")
        return int(self.selected)
