from __future__ import annotations

import torch
import torch.nn.functional as F


def belief_classification_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    confidence_threshold: float = 0.8,
) -> dict[str, float | list[float]]:
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape (episodes, windows, modes)")
    if tuple(targets.shape) != (probabilities.shape[0],):
        raise ValueError("targets must have shape (episodes,)")
    if not 0.0 < confidence_threshold < 1.0:
        raise ValueError("confidence_threshold must lie in (0, 1)")
    probabilities = probabilities.clamp_min(1e-9)
    one_hot = F.one_hot(targets, num_classes=probabilities.shape[-1]).to(probabilities.dtype)
    expanded_targets = targets[:, None, None].expand(-1, probabilities.shape[1], 1)
    true_probability = probabilities.gather(dim=-1, index=expanded_targets).squeeze(-1)
    predictions = probabilities.argmax(dim=-1)
    accuracy = (predictions == targets[:, None]).to(probabilities.dtype)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    brier = (probabilities - one_hot[:, None, :]).square().sum(dim=-1)

    confident = true_probability >= confidence_threshold
    window_indices = torch.arange(
        1, probabilities.shape[1] + 1, device=probabilities.device
    )[None, :].expand_as(true_probability)
    sentinel = torch.full_like(window_indices, probabilities.shape[1] + 1)
    windows_to_confidence = torch.where(confident, window_indices, sentinel).min(dim=1).values

    return {
        "mode_nll": float((-true_probability.log()).mean()),
        "mode_accuracy": float(accuracy.mean()),
        "mode_brier": float(brier.mean()),
        "posterior_entropy": float(entropy.mean()),
        "final_mode_nll": float((-true_probability[:, -1].log()).mean()),
        "final_mode_accuracy": float(accuracy[:, -1].mean()),
        "final_mode_brier": float(brier[:, -1].mean()),
        "final_posterior_entropy": float(entropy[:, -1].mean()),
        "windows_to_confidence": float(windows_to_confidence.to(probabilities.dtype).mean()),
        "accuracy_by_window": accuracy.mean(dim=0).tolist(),
        "true_probability_by_window": true_probability.mean(dim=0).tolist(),
        "entropy_by_window": entropy.mean(dim=0).tolist(),
    }


def predictive_mixture_nll(
    prior_probabilities: torch.Tensor,
    mode_log_likelihoods: torch.Tensor,
) -> tuple[float, list[float]]:
    if prior_probabilities.shape != mode_log_likelihoods.shape:
        raise ValueError("prior probabilities and mode likelihoods must have identical shapes")
    log_prior = prior_probabilities.clamp_min(1e-9).log()
    nll = -torch.logsumexp(log_prior + mode_log_likelihoods, dim=-1)
    return float(nll.mean()), nll.mean(dim=0).tolist()
