from dataclasses import dataclass
import math

import torch


def conditional_emission_step(mean, log_std, observation, noise_variance, offset_variance, correlation, state=None):
    if mean.shape != log_std.shape or mean.ndim < 3 or mean.shape[:-3] != observation.shape[:-2] or mean.shape[-2:] != observation.shape[-2:]:
        raise ValueError("mode emissions and executed observation must align")
    if not all(torch.isfinite(value).all() for value in (mean, log_std, observation)):
        raise ValueError("emissions and executed observations must be finite")
    residual = (observation[..., None, :, :] - mean) * (-log_std).exp()
    steps, coordinates = mean.shape[-2:]
    noise = mean.new_tensor(noise_variance) if not isinstance(noise_variance, torch.Tensor) else noise_variance.to(mean)
    offset = mean.new_tensor(offset_variance) if not isinstance(offset_variance, torch.Tensor) else offset_variance.to(mean)
    coefficient = mean.new_tensor(correlation) if not isinstance(correlation, torch.Tensor) else correlation.to(mean)
    if state is None:
        prior_mean = torch.zeros_like(residual[..., 0, :])
        prior_variance = offset
    else:
        prior_mean, prior_variance = state
        if prior_mean.shape != residual.shape[:-2] + (coordinates,):
            raise ValueError("conditional emission state must match identity and mode dimensions")
    error = residual - prior_mean[..., None, :]
    determinant = (steps - 1) * noise.log() + (noise + steps * prior_variance).log()
    quadratic = error.square().sum((-1, -2)) / noise
    quadratic = quadratic - prior_variance * error.sum(-2).square().sum(-1) / (noise * (noise + steps * prior_variance))
    likelihood = -.5 * (quadratic + coordinates * determinant + steps * coordinates * math.log(2 * math.pi)) - log_std.sum((-1, -2))
    gain = prior_variance / (noise + steps * prior_variance)
    posterior_mean = prior_mean + gain * error.sum(-2)
    posterior_variance = prior_variance * noise / (noise + steps * prior_variance)
    next_state = (
        coefficient * posterior_mean,
        coefficient.square() * posterior_variance + (1 - coefficient.square()) * offset,
    )
    return likelihood, next_state


@dataclass(frozen=True)
class CorrelatedEmission:
    noise_variance: float
    offset_variance: float
    correlation: float
    execute_steps: int
    dt: float

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.noise_variance, self.offset_variance, self.correlation, self.dt)):
            raise ValueError("correlated emission parameters must be finite")
        if self.noise_variance <= 0 or self.offset_variance < 0 or not 0 <= self.correlation < 1:
            raise ValueError("correlated emission variances and correlation are invalid")
        if self.execute_steps < 1 or self.dt <= 0:
            raise ValueError("correlated emission requires a positive execution clock")

    def step(self, mean, log_std, observation, state=None):
        if mean.shape[-2] != self.execute_steps:
            raise ValueError("correlated emission must consume exactly the executed prefix")
        return conditional_emission_step(mean, log_std, observation, self.noise_variance, self.offset_variance, self.correlation, state)


def sequence_emission_likelihoods(data, noise_variance, offset_variance, correlation):
    mean, log_std, observation = (data[name].double() for name in ("means", "log_std", "target"))
    if mean.ndim != 5:
        raise ValueError("emission sequences must have episode, window, mode, time, coordinate dimensions")
    state = None
    likelihoods = []
    for window in range(mean.shape[1]):
        likelihood, state = conditional_emission_step(mean[:, window], log_std[:, window], observation[:, window], noise_variance, offset_variance, correlation, state)
        likelihoods.append(likelihood)
    return torch.stack(likelihoods, dim=1)
