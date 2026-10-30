"""Streaming adaptive persistent categorical belief state.

This module is intentionally independent from the experiment runners.  It
keeps a posterior between observations, increases the transition hazard when
the current observation is surprising under the *untempered* likelihood, and
can optionally reset to the initial prior on a hysteretic change-point event.
The state is stored in log space and supports arbitrary leading batch
dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AdaptivePersistentBeliefConfig:
    """Numerical and adaptation settings for :class:`AdaptivePersistentBelief`."""

    base_hazard: float = 0.0
    max_hazard: float = 0.10
    surprise_threshold: float = 2.0
    surprise_scale: float = 0.10
    likelihood_scale: float = 1.0
    hard_reset: bool = False
    threshold_hysteresis: float = 0.0
    cooldown_steps: int = 0
    likelihood_clip: float = 80.0
    min_probability: float = 1e-12

    def __post_init__(self) -> None:
        scalar_fields = (
            ("base_hazard", self.base_hazard),
            ("max_hazard", self.max_hazard),
            ("surprise_threshold", self.surprise_threshold),
            ("surprise_scale", self.surprise_scale),
            ("likelihood_scale", self.likelihood_scale),
            ("threshold_hysteresis", self.threshold_hysteresis),
            ("likelihood_clip", self.likelihood_clip),
            ("min_probability", self.min_probability),
        )
        for name, value in scalar_fields:
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= float(self.base_hazard) <= 1.0:
            raise ValueError("base_hazard must lie in [0, 1]")
        if not 0.0 <= float(self.max_hazard) <= 1.0:
            raise ValueError("max_hazard must lie in [0, 1]")
        if float(self.max_hazard) < float(self.base_hazard):
            raise ValueError("max_hazard must be >= base_hazard")
        if float(self.surprise_threshold) <= 0.0:
            raise ValueError("surprise_threshold must be positive")
        if float(self.surprise_scale) <= 0.0:
            raise ValueError("surprise_scale must be positive")
        if float(self.likelihood_scale) < 0.0:
            raise ValueError("likelihood_scale must be non-negative")
        if float(self.threshold_hysteresis) < 0.0:
            raise ValueError("threshold_hysteresis must be non-negative")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative")
        if float(self.likelihood_clip) <= 0.0:
            raise ValueError("likelihood_clip must be positive")
        if not 0.0 < float(self.min_probability) <= 1.0:
            raise ValueError("min_probability must lie in (0, 1]")


class AdaptivePersistentBelief:
    """Maintain a persistent categorical posterior with adaptive forgetting.

    ``prior`` is a probability tensor with shape ``(..., num_modes)``.  If it
    is omitted, a uniform prior is created lazily from the first update.  Each
    call to :meth:`update` accepts an untempered log-likelihood with shape
    ``(..., num_modes)`` and returns ``(log_posterior, diagnostics)``.  The
    posterior is detached before it is stored so a long online rollout does
    not retain an autograd graph.

    An observation containing any non-finite likelihood value is treated as
    invalid for that batch item: it produces no belief update, no forgetting,
    and finite diagnostics.  This avoids allowing a single malformed model
    output to poison a persistent state.
    """

    def __init__(
        self,
        num_modes: int,
        prior: torch.Tensor | None = None,
        config: AdaptivePersistentBeliefConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_modes < 2:
            raise ValueError("num_modes must be at least two")
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            raise ValueError("dtype must be floating point")
        self.num_modes = int(num_modes)
        self.config = config or AdaptivePersistentBeliefConfig()
        self._device = torch.device(device) if device is not None else None
        self._dtype = dtype
        self._initial_log_belief = (
            self._normalise_prior(prior) if prior is not None else None
        )
        self._log_belief: torch.Tensor | None = None
        self._surprise_active: torch.Tensor | None = None
        self._cooldown_remaining: torch.Tensor | None = None

    @property
    def log_belief(self) -> torch.Tensor | None:
        """Return a copy of the current normalized log posterior."""

        return None if self._log_belief is None else self._log_belief.clone()

    @property
    def probabilities(self) -> torch.Tensor | None:
        """Return the current posterior in probability space."""

        return None if self._log_belief is None else self._log_belief.exp()

    @property
    def initialized(self) -> bool:
        return self._log_belief is not None

    def reset(self, prior: torch.Tensor | None = None) -> torch.Tensor | None:
        """Reset the state to ``prior`` or to the original initial prior."""

        if prior is not None:
            self._initial_log_belief = self._normalise_prior(prior)
        if self._log_belief is None:
            self._surprise_active = None
            self._cooldown_remaining = None
            return None

        target = self._initial_for_shape(self._log_belief.shape[:-1])
        self._log_belief = target.clone()
        batch_shape = self._log_belief.shape[:-1]
        self._surprise_active = torch.zeros(
            batch_shape, dtype=torch.bool, device=self._log_belief.device
        )
        self._cooldown_remaining = torch.zeros(
            batch_shape, dtype=torch.long, device=self._log_belief.device
        )
        return self.log_belief

    def update(
        self,
        log_likelihood: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply one adaptive persistent update.

        Surprise is computed from the current predictive probability using the
        raw, untempered likelihood.  ``likelihood_scale`` is applied only to
        the posterior update, so calibration of the adaptation signal is not
        changed by a downstream soft update temperature. Numerical clipping
        bounds relative, tempered evidence, never the absolute log density.
        """

        likelihood = torch.as_tensor(log_likelihood)
        if likelihood.ndim < 1:
            raise ValueError("log_likelihood must include a mode dimension")
        if likelihood.shape[-1] != self.num_modes:
            raise ValueError(
                f"log_likelihood must have last dimension {self.num_modes}, "
                f"got {likelihood.shape[-1]}"
            )
        if not torch.is_floating_point(likelihood):
            likelihood = likelihood.to(torch.float32)
        if likelihood.dtype in (torch.float16, torch.bfloat16):
            likelihood = likelihood.to(torch.float32)
        if self._device is not None:
            likelihood = likelihood.to(self._device)
        if observation_mask is not None:
            observation_mask = torch.as_tensor(observation_mask, device=likelihood.device)
            if tuple(observation_mask.shape) != tuple(likelihood.shape[:-1]):
                raise ValueError(
                    "observation_mask must match log_likelihood batch dimensions"
                )
            observation_mask = observation_mask.to(dtype=torch.bool)

        with torch.no_grad():
            self._ensure_state(likelihood)
            assert self._log_belief is not None
            assert self._surprise_active is not None
            assert self._cooldown_remaining is not None

            current = torch.log_softmax(self._log_belief, dim=-1)
            finite = torch.isfinite(likelihood).all(dim=-1)
            if observation_mask is not None:
                finite = finite & observation_mask
            safe_likelihood = torch.where(
                finite[..., None], likelihood, torch.zeros_like(likelihood)
            )
            relative_likelihood = safe_likelihood - safe_likelihood.amax(dim=-1, keepdim=True)
            tempered_likelihood = (
                float(self.config.likelihood_scale) * relative_likelihood
                if self.config.likelihood_scale > 0.0
                else torch.zeros_like(relative_likelihood)
            ).clamp(min=-float(self.config.likelihood_clip), max=0.0)

            predictive_log_probability = torch.logsumexp(
                current + safe_likelihood, dim=-1
            )
            surprise = (-predictive_log_probability).nan_to_num(
                nan=0.0,
                posinf=float(self.config.likelihood_clip),
                neginf=-float(self.config.likelihood_clip),
            )
            surprise = torch.where(finite, surprise, torch.zeros_like(surprise))

            intensity = torch.sigmoid(
                (surprise - float(self.config.surprise_threshold))
                / float(self.config.surprise_scale)
            )
            hazard = float(self.config.base_hazard) + (
                float(self.config.max_hazard) - float(self.config.base_hazard)
            ) * intensity
            hazard = torch.where(finite, hazard, torch.zeros_like(hazard))

            previous_active = self._surprise_active
            threshold_on = float(
                self.config.surprise_threshold + self.config.threshold_hysteresis
            )
            threshold_off = float(
                max(
                    0.0,
                    self.config.surprise_threshold
                    - self.config.threshold_hysteresis,
                )
            )
            entered_active = surprise >= threshold_on
            remained_active = surprise >= threshold_off
            active = torch.where(
                previous_active, remained_active, entered_active
            )
            active = torch.where(finite, active, previous_active)
            changed = finite & (~previous_active) & active

            cooldown_before = self._cooldown_remaining
            reset = (
                changed
                & bool(self.config.hard_reset)
                & (cooldown_before <= 0)
            )
            reset_target = self._initial_for_shape(current.shape[:-1])
            reset_prior = torch.where(
                reset[..., None], reset_target, current
            )

            mixed_log_prior = torch.logaddexp(
                torch.log1p(-hazard[..., None]) + reset_prior,
                hazard[..., None].log() - math.log(self.num_modes),
            )
            candidate = torch.log_softmax(
                mixed_log_prior + tempered_likelihood,
                dim=-1,
            )
            posterior = torch.where(finite[..., None], candidate, current)
            posterior = torch.log_softmax(posterior, dim=-1)

            cooldown = torch.clamp(cooldown_before - finite.to(torch.long), min=0)
            if self.config.cooldown_steps > 0:
                cooldown = torch.where(
                    reset,
                    torch.full_like(cooldown, int(self.config.cooldown_steps)),
                    cooldown,
                )

            self._log_belief = posterior.detach()
            self._surprise_active = active.detach()
            self._cooldown_remaining = cooldown.detach()

            diagnostics = {
                "hazard": hazard.detach(),
                "surprise": surprise.detach(),
                "reset": reset.detach(),
                "changed": changed.detach(),
                "active": active.detach(),
                "valid": finite.detach(),
                "cooldown": cooldown.detach(),
            }
            return self._log_belief.clone(), diagnostics

    def _ensure_state(self, likelihood: torch.Tensor) -> None:
        batch_shape = likelihood.shape[:-1]
        if self._log_belief is None:
            initial = self._initial_for_shape(batch_shape, likelihood)
            self._log_belief = initial.clone()
            if self._initial_log_belief is None:
                self._initial_log_belief = initial.clone()
            self._surprise_active = torch.zeros(
                batch_shape, dtype=torch.bool, device=likelihood.device
            )
            self._cooldown_remaining = torch.zeros(
                batch_shape, dtype=torch.long, device=likelihood.device
            )
            return

        expected_shape = (*batch_shape, self.num_modes)
        if tuple(self._log_belief.shape) != expected_shape:
            raise ValueError(
                f"log_likelihood batch shape {batch_shape} does not match "
                f"state shape {tuple(self._log_belief.shape[:-1])}"
            )
        self._log_belief = self._log_belief.to(
            device=likelihood.device, dtype=likelihood.dtype
        )
        self._initial_log_belief = self._initial_log_belief.to(
            device=likelihood.device, dtype=likelihood.dtype
        )
        self._surprise_active = self._surprise_active.to(device=likelihood.device)
        self._cooldown_remaining = self._cooldown_remaining.to(
            device=likelihood.device
        )

    def _normalise_prior(self, prior: torch.Tensor) -> torch.Tensor:
        prior = torch.as_tensor(prior)
        if prior.ndim < 1 or prior.shape[-1] != self.num_modes:
            raise ValueError(
                f"prior must have last dimension {self.num_modes}"
            )
        if not torch.is_floating_point(prior):
            prior = prior.to(torch.float32)
        if not torch.isfinite(prior).all():
            raise ValueError("prior must contain only finite values")
        if (prior < 0).any():
            raise ValueError("prior must contain non-negative probabilities")
        mass = prior.sum(dim=-1, keepdim=True)
        if (mass <= 0).any():
            raise ValueError("each prior probability vector must have positive mass")
        if self._device is not None:
            prior = prior.to(self._device)
        if prior.dtype in (torch.float16, torch.bfloat16):
            prior = prior.to(torch.float32)
        return (prior / mass.to(prior) ).clamp_min(
            float(self.config.min_probability)
        ).log()

    def _initial_for_shape(
        self,
        batch_shape: tuple[int, ...],
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reference is None:
            if self._log_belief is not None:
                reference = self._log_belief
            elif self._initial_log_belief is not None:
                reference = self._initial_log_belief
        if reference is not None:
            device = reference.device
            dtype = reference.dtype
        elif self._device is not None:
            device = self._device
            dtype = self._dtype
        else:
            device = torch.device("cpu")
            dtype = self._dtype

        if self._initial_log_belief is None:
            return torch.full(
                (*batch_shape, self.num_modes),
                -math.log(self.num_modes),
                device=device,
                dtype=dtype,
            )

        initial = self._initial_log_belief.to(device=device, dtype=dtype)
        initial_batch_shape = tuple(initial.shape[:-1])
        if initial_batch_shape == batch_shape:
            return torch.log_softmax(initial, dim=-1)
        if not initial_batch_shape:
            return torch.log_softmax(
                initial.expand(*batch_shape, self.num_modes), dim=-1
            )
        raise ValueError(
            f"prior batch shape {initial_batch_shape} does not match "
            f"likelihood batch shape {batch_shape}"
        )


__all__ = [
    "AdaptivePersistentBelief",
    "AdaptivePersistentBeliefConfig",
]
