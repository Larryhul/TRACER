"""Belief-maintenance components used for related-work comparisons.

These implementations deliberately expose only the belief-update part that is
common to the current TRACER protocol. They are method-level baselines, not
official reimplementations of the complete LUCIDGames, MPOGames, or AToM
planners. LUCIDGames and AToM use continuous UKF parameter beliefs, while the
synthetic TRACER audit has a discrete response-mode truth. The UKF therefore
provides an explicit continuous-model-mismatch baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class UKFBeliefConfig:
    """Configuration for a one-dimensional continuous behavior parameter."""

    num_modes: int = 3
    prior_mean: float | None = None
    prior_variance: float = 1.0
    process_variance: float = 0.02
    observation_variance: float = 0.18**2
    anchor_temperature: float = 8.0
    adaptive_process: bool = False
    max_process_variance: float = 0.5

    def __post_init__(self) -> None:
        if self.num_modes < 2:
            raise ValueError("num_modes must be at least two")
        if self.prior_variance <= 0.0:
            raise ValueError("prior_variance must be positive")
        if self.process_variance < 0.0:
            raise ValueError("process_variance must be non-negative")
        if self.observation_variance <= 0.0:
            raise ValueError("observation_variance must be positive")
        if self.anchor_temperature <= 0.0:
            raise ValueError("anchor_temperature must be positive")
        if self.max_process_variance < self.process_variance:
            raise ValueError("max_process_variance must be >= process_variance")


def categorical_pomdp_update(
    probabilities: torch.Tensor,
    log_likelihood: torch.Tensor,
    transition_probability: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a discrete-mode POMDP prediction and Bayes observation update.

    The transition is a symmetric mode-switch kernel. The returned pair is
    the predicted belief before the current observation and the posterior
    after it. This is the belief-maintenance component used for the
    MPOGames-style comparison.
    """

    if probabilities.ndim < 1 or probabilities.shape != log_likelihood.shape:
        raise ValueError("probabilities and log_likelihood must have matching mode shapes")
    if not 0.0 <= float(transition_probability) <= 1.0:
        raise ValueError("transition_probability must lie in [0, 1]")
    num_modes = int(probabilities.shape[-1])
    if num_modes < 2:
        raise ValueError("at least two modes are required")
    predicted = categorical_pomdp_predict(probabilities, transition_probability)
    posterior = torch.softmax(predicted.clamp_min(1e-9).log() + log_likelihood, dim=-1)
    return predicted, posterior


def categorical_pomdp_predict(
    probabilities: torch.Tensor,
    transition_probability: float = 0.10,
) -> torch.Tensor:
    """Propagate a categorical belief through the symmetric mode kernel."""

    if probabilities.ndim < 1:
        raise ValueError("probabilities must include a mode dimension")
    if not 0.0 <= float(transition_probability) <= 1.0:
        raise ValueError("transition_probability must lie in [0, 1]")
    num_modes = int(probabilities.shape[-1])
    if num_modes < 2:
        raise ValueError("at least two modes are required")
    transition = torch.full(
        (num_modes, num_modes),
        float(transition_probability) / (num_modes - 1),
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    transition.fill_diagonal_(1.0 - float(transition_probability))
    return probabilities @ transition


def _measurement_from_parameter(
    parameter: torch.Tensor,
    mode_means: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Interpolate mode-conditioned response means at continuous parameters."""

    num_modes = int(mode_means.shape[-2])
    anchors = torch.arange(
        num_modes, device=mode_means.device, dtype=mode_means.dtype
    )
    weights = torch.softmax(
        -float(temperature) * (parameter[..., None] - anchors).square(), dim=-1
    )
    expanded_means = mode_means[:, None].expand(
        -1, parameter.shape[-1], -1, -1
    )
    return (weights[..., None] * expanded_means).sum(dim=-2)


def _parameter_mode_probabilities(
    mean: torch.Tensor,
    variance: torch.Tensor,
    num_modes: int,
) -> torch.Tensor:
    anchors = torch.arange(num_modes, device=mean.device, dtype=mean.dtype)
    log_scores = -0.5 * (anchors - mean[..., None]).square() / variance[..., None]
    return torch.softmax(log_scores, dim=-1)


def run_ukf_filter(
    mode_means: torch.Tensor,
    responses: torch.Tensor,
    config: UKFBeliefConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a batched scalar UKF over response-conditioned mode means.

    Args:
        mode_means: Tensor shaped ``(episodes, windows, modes, response_dim)``.
        responses: Tensor shaped ``(episodes, windows, response_dim)``.
        config: UKF parameter and process-noise settings.

    Returns:
        ``(before_probabilities, after_probabilities, predictive_nll)``. The
        probabilities are projected from the continuous parameter posterior
        onto the discrete response anchors only for common scoring.
    """

    if mode_means.ndim != 4 or responses.ndim != 3:
        raise ValueError("mode_means must be (episodes, windows, modes, dim)")
    if mode_means.shape[:2] != responses.shape[:2] or mode_means.shape[-1] != responses.shape[-1]:
        raise ValueError("mode_means and responses must have aligned episode/window dimensions")
    if int(mode_means.shape[-2]) != config.num_modes:
        raise ValueError("mode_means mode count does not match config")

    episodes, windows, _, response_dim = mode_means.shape
    mean = torch.full(
        (episodes,),
        float(config.prior_mean if config.prior_mean is not None else (config.num_modes - 1) / 2.0),
        device=mode_means.device,
        dtype=mode_means.dtype,
    )
    variance = torch.full_like(mean, float(config.prior_variance))
    process_variance = torch.full_like(mean, float(config.process_variance))
    identity = torch.eye(response_dim, device=mode_means.device, dtype=mode_means.dtype)
    before: list[torch.Tensor] = []
    after: list[torch.Tensor] = []
    predictive_nll: list[torch.Tensor] = []

    for window in range(windows):
        variance = (variance + process_variance).clamp_min(1e-6)
        predicted_mean = mean
        predicted_variance = variance
        standard_deviation = variance.sqrt()
        sigma_parameters = torch.stack(
            [mean, mean - standard_deviation, mean + standard_deviation], dim=-1
        )
        sigma_measurements = _measurement_from_parameter(
            sigma_parameters,
            mode_means[:, window],
            config.anchor_temperature,
        )
        measurement_mean = sigma_measurements.mean(dim=-2)
        centered_measurements = sigma_measurements - measurement_mean[:, None]
        measurement_covariance = 0.5 * torch.einsum(
            "bsi,bsj->bij", centered_measurements[:, 1:], centered_measurements[:, 1:]
        )
        measurement_covariance = measurement_covariance + float(config.observation_variance) * identity
        innovation = responses[:, window] - measurement_mean
        solved_innovation = torch.linalg.solve(
            measurement_covariance, innovation[..., None]
        ).squeeze(-1)
        mahalanobis = (innovation * solved_innovation).sum(dim=-1).clamp_min(0.0)
        logdet = torch.linalg.slogdet(measurement_covariance)[1]
        predictive_nll.append(
            0.5 * (float(response_dim) * torch.log(torch.tensor(2.0 * torch.pi, device=mode_means.device, dtype=mode_means.dtype)) + logdet + mahalanobis)
        )
        cross_covariance = 0.5 * (
            (-standard_deviation)[..., None] * centered_measurements[:, 1]
            + standard_deviation[..., None] * centered_measurements[:, 2]
        )
        gain = torch.linalg.solve(
            measurement_covariance, cross_covariance[..., None]
        ).squeeze(-1)
        mean = mean + (gain * innovation).sum(dim=-1)
        variance = variance - torch.einsum(
            "bi,bij,bj->b", gain, measurement_covariance, gain
        )
        variance = variance.clamp_min(1e-6)
        mean = mean.clamp(0.0, float(config.num_modes - 1))
        before.append(
            _parameter_mode_probabilities(
                predicted_mean,
                predicted_variance,
                config.num_modes,
            )
        )
        after.append(_parameter_mode_probabilities(mean, variance, config.num_modes))
        if config.adaptive_process:
            process_variance = (
                float(config.process_variance)
                * (1.0 + 0.20 * mahalanobis.clamp(0.0, 10.0))
            ).clamp(float(config.process_variance), float(config.max_process_variance))

    return torch.stack(before, dim=1), torch.stack(after, dim=1), torch.stack(predictive_nll, dim=1)


class UKFParameterBelief:
    """Streaming scalar UKF state for a receding-horizon planner."""

    def __init__(self, config: UKFBeliefConfig, device: torch.device | str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        # The UKF state is tiny. Keeping its covariance solve on CPU avoids a
        # Triton JIT dependency for small outer products on some CUDA builds.
        self.state_device = torch.device("cpu")
        self.mean = torch.tensor(
            float(config.prior_mean if config.prior_mean is not None else (config.num_modes - 1) / 2.0),
            device=self.state_device,
            dtype=torch.float32,
        )
        self.variance = torch.tensor(
            float(config.prior_variance), device=self.state_device, dtype=torch.float32
        )
        self.process_variance = torch.tensor(
            float(config.process_variance), device=self.state_device, dtype=torch.float32
        )

    def probabilities(self) -> torch.Tensor:
        """Return the predicted mode probabilities before a new observation."""

        return _parameter_mode_probabilities(
            self.mean,
            (self.variance + self.process_variance).clamp_min(1e-6),
            self.config.num_modes,
        ).to(self.device)

    @torch.no_grad()
    def update(self, mode_means: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        """Update from one executed response prefix and return posterior modes."""

        if mode_means.ndim < 2 or response.ndim < 1:
            raise ValueError("mode_means and response must include response dimensions")
        if int(mode_means.shape[0]) != self.config.num_modes:
            raise ValueError("mode_means mode count does not match config")
        if tuple(mode_means.shape[1:]) != tuple(response.shape):
            raise ValueError("mode_means and response must have aligned dimensions")
        mode_means = mode_means.detach().to(self.state_device, dtype=torch.float32).reshape(
            self.config.num_modes, -1
        )
        response = response.detach().to(self.state_device, dtype=torch.float32).reshape(-1)
        variance = (self.variance + self.process_variance).clamp_min(1e-6)
        standard_deviation = variance.sqrt()
        sigma_parameters = torch.stack(
            [self.mean, self.mean - standard_deviation, self.mean + standard_deviation]
        )
        sigma_measurements = _measurement_from_parameter(
            sigma_parameters[None], mode_means[None], self.config.anchor_temperature
        )[0]
        measurement_mean = sigma_measurements.mean(dim=0)
        centered = sigma_measurements - measurement_mean
        identity = torch.eye(response.numel(), device=self.state_device)
        measurement_covariance = 0.5 * (
            centered[1:, :, None] @ centered[1:, None, :]
        ).sum(dim=0)
        measurement_covariance = measurement_covariance + float(self.config.observation_variance) * identity
        innovation = response - measurement_mean
        solved_innovation = torch.linalg.solve(measurement_covariance, innovation)
        mahalanobis = (innovation * solved_innovation).sum().clamp_min(0.0)
        cross_covariance = 0.5 * (
            (-standard_deviation) * centered[1]
            + standard_deviation * centered[2]
        )
        gain = torch.linalg.solve(measurement_covariance, cross_covariance)
        self.mean = (self.mean + (gain * innovation).sum()).clamp(
            0.0, float(self.config.num_modes - 1)
        )
        self.variance = (
            variance - gain @ measurement_covariance @ gain
        ).clamp_min(1e-6)
        if self.config.adaptive_process:
            self.process_variance = (
                float(self.config.process_variance)
                * (1.0 + 0.20 * mahalanobis.clamp(0.0, 10.0))
            ).clamp(
                float(self.config.process_variance),
                float(self.config.max_process_variance),
            )
        return _parameter_mode_probabilities(
            self.mean, self.variance, self.config.num_modes
        ).to(self.device)
