from __future__ import annotations

from collections.abc import Hashable
import math

import torch


def uniform_log_belief(
    num_modes: int,
    *batch_shape: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if num_modes < 2:
        raise ValueError("num_modes must be at least two")
    logits = torch.zeros(*batch_shape, num_modes, device=device, dtype=dtype)
    return torch.log_softmax(logits, dim=-1)


def categorical_bayes_update(
    log_belief: torch.Tensor,
    log_likelihood: torch.Tensor,
) -> torch.Tensor:
    """Apply one categorical Bayes update in stable log space."""

    if log_belief.shape != log_likelihood.shape:
        raise ValueError(
            "log_belief and log_likelihood must have identical shapes, got "
            f"{tuple(log_belief.shape)} and {tuple(log_likelihood.shape)}"
        )
    if log_belief.ndim < 1:
        raise ValueError("belief tensors must include a mode dimension")
    return torch.log_softmax(log_belief + log_likelihood, dim=-1)


def apply_belief_forgetting(
    log_belief: torch.Tensor,
    forgetting_rate: float,
) -> torch.Tensor:
    """Mix a posterior with the uniform prior before the next decision.

    The operation is defined in probability space so the forgetting rate has
    the same interpretation for any number of modes.  ``0`` preserves the
    posterior and ``1`` returns a uniform belief.
    """

    if log_belief.ndim < 1:
        raise ValueError("log_belief must include a mode dimension")
    if not 0.0 <= float(forgetting_rate) <= 1.0:
        raise ValueError("forgetting_rate must lie in [0, 1]")
    probability = log_belief.exp()
    uniform = torch.full_like(probability, 1.0 / probability.shape[-1])
    mixed = (1.0 - float(forgetting_rate)) * probability + float(forgetting_rate) * uniform
    # ``mixed`` is already a normalized probability vector.  Applying
    # ``log_softmax`` to the probabilities would treat them as logits and
    # unintentionally flatten a confident posterior.  Convert back to log
    # probability directly instead.
    return mixed.clamp_min(1e-9).log()


def hazard_belief_update(
    log_belief: torch.Tensor,
    log_likelihood: torch.Tensor,
    hazard_rate: float,
) -> torch.Tensor:
    """Apply a categorical update with a constant probability of mode change."""

    return categorical_bayes_update(
        apply_belief_forgetting(log_belief, hazard_rate), log_likelihood
    )


def detect_change_point(
    log_belief: torch.Tensor,
    log_likelihood: torch.Tensor,
    surprise_threshold: float,
) -> torch.Tensor:
    """Return a scalar boolean indicating an unexpectedly poor observation.

    The score is the negative predictive log likelihood under the current
    belief.  The threshold is frozen on validation data by the experiment
    runner and is never fitted on the test episodes.
    """

    if log_belief.shape != log_likelihood.shape:
        raise ValueError("belief and likelihood shapes must match")
    if float(surprise_threshold) <= 0.0:
        raise ValueError("surprise_threshold must be positive")
    predictive_log_probability = torch.logsumexp(
        log_belief + log_likelihood, dim=-1
    )
    return (-predictive_log_probability > float(surprise_threshold)).detach()


def adaptive_categorical_update(
    log_belief: torch.Tensor,
    log_likelihood: torch.Tensor,
    base_hazard: float = 0.0,
    max_hazard: float = 0.10,
    surprise_threshold: float = 2.0,
    surprise_scale: float = 0.10,
    hard_reset: bool = False,
    likelihood_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update a categorical belief with surprise-triggered adaptation.

    The prior is mixed with a uniform transition distribution.  The mixing
    rate stays near ``base_hazard`` for expected observations and rises
    toward ``max_hazard`` when the current posterior assigns low predictive
    probability to the observation.  A hard reset is optional and is only
    activated at the same validated surprise threshold; it prevents a stale,
    near-zero mode probability from blocking recovery after a switch.

    Returns ``(posterior, hazard, surprise, changed)``.  ``hazard`` and
    ``surprise`` retain the batch shape without the mode dimension.
    Supply untempered ``log_likelihood``; ``likelihood_scale`` affects only
    the observation update, never the surprise detector's units.
    """

    if log_belief.shape != log_likelihood.shape:
        raise ValueError("belief and likelihood shapes must match")
    if log_belief.ndim < 1:
        raise ValueError("belief tensors must include a mode dimension")
    if not 0.0 <= float(base_hazard) <= 1.0:
        raise ValueError("base_hazard must lie in [0, 1]")
    if not 0.0 <= float(max_hazard) <= 1.0:
        raise ValueError("max_hazard must lie in [0, 1]")
    if float(max_hazard) < float(base_hazard):
        raise ValueError("max_hazard must be >= base_hazard")
    if float(surprise_threshold) <= 0.0 or float(surprise_scale) <= 0.0:
        raise ValueError("surprise threshold and scale must be positive")
    if not math.isfinite(float(likelihood_scale)) or float(likelihood_scale) < 0.0:
        raise ValueError("likelihood_scale must be finite and non-negative")

    predictive_log_probability = torch.logsumexp(
        log_belief + log_likelihood, dim=-1
    )
    surprise = -predictive_log_probability
    excess = (surprise - float(surprise_threshold)) / float(surprise_scale)
    intensity = torch.sigmoid(excess)
    hazard = float(base_hazard) + (
        float(max_hazard) - float(base_hazard)
    ) * intensity
    changed = surprise > float(surprise_threshold)
    mixed_prior = apply_belief_forgetting(log_belief, 0.0)
    probability = mixed_prior.exp()
    uniform = torch.full_like(probability, 1.0 / probability.shape[-1])
    mixed_probability = (1.0 - hazard[..., None]) * probability + hazard[..., None] * uniform
    if hard_reset:
        mixed_probability = torch.where(
            changed[..., None], uniform, mixed_probability
        )
    posterior = categorical_bayes_update(
        mixed_probability.clamp_min(1e-9).log(), float(likelihood_scale) * log_likelihood
    )
    return posterior, hazard.detach(), surprise.detach(), changed.detach()


def run_categorical_filter(
    log_likelihoods: torch.Tensor,
    prior_log_belief: torch.Tensor | None = None,
    reset_each_window: bool = False,
    observation_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return beliefs immediately before and after every observation.

    log_likelihoods has shape (..., windows, modes). The returned tensors have
    the same shape. Persistent filtering feeds each posterior into the next
    window; reset filtering reuses the supplied prior at every window.
    """

    if log_likelihoods.ndim < 2:
        raise ValueError("log_likelihoods must have shape (..., windows, modes)")
    num_modes = int(log_likelihoods.shape[-1])
    batch_shape = tuple(log_likelihoods.shape[:-2])
    if prior_log_belief is None:
        prior = uniform_log_belief(
            num_modes,
            *batch_shape,
            device=log_likelihoods.device,
            dtype=log_likelihoods.dtype,
        )
    else:
        expected = (*batch_shape, num_modes)
        if tuple(prior_log_belief.shape) != expected:
            raise ValueError(
                f"prior_log_belief must have shape {expected}, got {tuple(prior_log_belief.shape)}"
            )
        prior = torch.log_softmax(prior_log_belief, dim=-1)

    if observation_mask is not None:
        expected_mask = (*batch_shape, int(log_likelihoods.shape[-2]))
        if tuple(observation_mask.shape) != expected_mask:
            raise ValueError(
                f"observation_mask must have shape {expected_mask}, got "
                f"{tuple(observation_mask.shape)}"
            )
        observation_mask = observation_mask.to(
            device=log_likelihoods.device, dtype=torch.bool
        )

    current = prior
    before = []
    after = []
    for window in range(log_likelihoods.shape[-2]):
        if reset_each_window:
            current = prior
        before.append(current)
        updated = categorical_bayes_update(current, log_likelihoods[..., window, :])
        if observation_mask is None:
            current = updated
        else:
            current = torch.where(
                observation_mask[..., window, None], updated, current
            )
        after.append(current)
    return torch.stack(before, dim=-2), torch.stack(after, dim=-2)


def lagged_reset_beliefs(
    log_likelihoods: torch.Tensor,
    prior_log_belief: torch.Tensor | None = None,
    observation_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use only the previous window's observation at each decision.

    Window zero uses the common prior. At window ``t > 0``, the returned
    ``before`` belief is the reset posterior from window ``t - 1``. This is the
    legal single-window baseline for receding-horizon prediction.
    """

    reset_before_observation, reset_after = run_categorical_filter(
        log_likelihoods,
        prior_log_belief=prior_log_belief,
        reset_each_window=True,
        observation_mask=observation_mask,
    )
    prior = reset_before_observation[..., :1, :]
    decision_belief = torch.cat([prior, reset_after[..., :-1, :]], dim=-2)
    return decision_belief, reset_after


class PersistentBeliefStore:
    """Perfect-ID categorical posterior store for the minimal ICRA setting."""

    def __init__(self, num_modes: int, prior: torch.Tensor | None = None) -> None:
        if num_modes < 2:
            raise ValueError("num_modes must be at least two")
        self.num_modes = int(num_modes)
        if prior is None:
            self._prior_log = uniform_log_belief(self.num_modes)
        else:
            if tuple(prior.shape) != (self.num_modes,):
                raise ValueError(f"prior must have shape {(self.num_modes,)}")
            if (prior < 0).any() or not torch.isfinite(prior).all() or float(prior.sum()) <= 0.0:
                raise ValueError("prior must contain finite non-negative mass")
            self._prior_log = torch.log(prior / prior.sum()).clamp_min(-1e9)
        self._beliefs: dict[Hashable, torch.Tensor] = {}

    def get_log(self, entity_id: Hashable) -> torch.Tensor:
        return self._beliefs.get(entity_id, self._prior_log).clone()

    def get(self, entity_id: Hashable) -> torch.Tensor:
        return self.get_log(entity_id).exp()

    def update(self, entity_id: Hashable, log_likelihood: torch.Tensor) -> torch.Tensor:
        if tuple(log_likelihood.shape) != (self.num_modes,):
            raise ValueError(f"log_likelihood must have shape {(self.num_modes,)}")
        posterior = categorical_bayes_update(self.get_log(entity_id), log_likelihood.detach().cpu())
        self._beliefs[entity_id] = posterior
        return posterior.exp().clone()

    def reset(self, entity_id: Hashable | None = None) -> None:
        if entity_id is None:
            self._beliefs.clear()
        else:
            self._beliefs.pop(entity_id, None)

    def __len__(self) -> int:
        return len(self._beliefs)
