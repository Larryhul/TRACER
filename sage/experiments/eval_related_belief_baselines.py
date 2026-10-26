"""Compare TRACER belief maintenance with related-work-style components.

The experiment uses the same response observations and splits as H2. It
isolates belief maintenance so that the comparison is interpretable:
MPOGames-POMDP is a discrete multimodal belief filter, while LUCIDGames-UKF
and AToM-UKF are continuous behavior-parameter UKF baselines. These are
method-level reimplementations of the shared belief components, not claims of
official complete planner reimplementations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from sage.belief.metrics import belief_classification_metrics, predictive_mixture_nll
from sage.belief.related_work import (
    UKFBeliefConfig,
    categorical_pomdp_update,
    run_ukf_filter,
)
from sage.belief.filter import adaptive_categorical_update
from sage.belief.synthetic import (
    BeliefSequenceSpec,
    PersistentBeliefSequenceDataset,
    mode_log_likelihood,
    mode_response_means,
)


METHODS = (
    "uniform-no-belief",
    "reset-bayes",
    "persistent-bayes",
    "adaptive-persistent",
    "mpogames-pomdp",
    "lucidgames-ukf",
    "atom-ukf",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _git_state() -> dict[str, str | bool]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
        "branch": run("branch", "--show-current"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset(raw: dict, split: str, seed: int, action_policy: str) -> PersistentBeliefSequenceDataset:
    cfg = raw["dataset"]
    count_key = "val_episodes" if split == "val" else "ood_episodes"
    return PersistentBeliefSequenceDataset(
        BeliefSequenceSpec(
            split=split,
            num_episodes=int(cfg[count_key]),
            windows=int(cfg.get("windows", 10)),
            seed=seed,
            response_noise=float(cfg.get("response_noise", 0.18)),
            num_modes=int(cfg.get("num_modes", 3)),
            action_policy=action_policy,
        )
    )


def _stack(dataset: PersistentBeliefSequenceDataset, device: str) -> tuple[torch.Tensor, ...]:
    actions = torch.stack([dataset[index]["actions"] for index in range(len(dataset))]).to(device)
    responses = torch.stack([dataset[index]["responses"] for index in range(len(dataset))]).to(device)
    modes = torch.stack([dataset[index]["mode"] for index in range(len(dataset))]).to(device)
    return actions, responses, modes


def _categorical_run(
    likelihoods: torch.Tensor,
    method: str,
    transition_probability: float,
    adaptive_base_hazard: float = 0.01,
    adaptive_max_hazard: float = 0.80,
    adaptive_surprise_threshold: float = 2.0,
    adaptive_surprise_scale: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    episodes, windows, modes = likelihoods.shape
    uniform = torch.full(
        (episodes, modes), 1.0 / modes, device=likelihoods.device, dtype=likelihoods.dtype
    )
    current = uniform
    before: list[torch.Tensor] = []
    after: list[torch.Tensor] = []
    nll: list[torch.Tensor] = []
    for window in range(windows):
        if method == "uniform-no-belief" or method == "reset-bayes":
            predicted = uniform
        elif method == "mpogames-pomdp":
            predicted, _ = categorical_pomdp_update(
                current, likelihoods[:, window], transition_probability
            )
        else:
            predicted = current
        before.append(predicted)
        if method == "adaptive-persistent":
            posterior, _, surprise, _ = adaptive_categorical_update(
                predicted.clamp_min(1e-9).log(),
                likelihoods[:, window],
                base_hazard=adaptive_base_hazard,
                max_hazard=adaptive_max_hazard,
                surprise_threshold=adaptive_surprise_threshold,
                surprise_scale=adaptive_surprise_scale,
            )
            nll.append(surprise)
        else:
            log_predictive = torch.logsumexp(
                predicted.clamp_min(1e-9).log() + likelihoods[:, window], dim=-1
            )
            nll.append(-log_predictive)
            posterior = torch.softmax(
                predicted.clamp_min(1e-9).log() + likelihoods[:, window], dim=-1
            )
        after.append(posterior.exp() if method == "adaptive-persistent" else posterior)
        # The adaptive helper returns log-probabilities; the other categorical
        # branches keep ``current`` in probability space.
        current = posterior.exp() if method == "adaptive-persistent" else posterior
    return torch.stack(before, dim=1), torch.stack(after, dim=1), torch.stack(nll, dim=1)


def _metric_row(
    probabilities: torch.Tensor,
    modes: torch.Tensor,
    predictive_nll: torch.Tensor,
    method: str,
    split: str,
    condition: str,
    seed: int,
) -> dict[str, float | int | str]:
    metrics = belief_classification_metrics(probabilities, modes)
    return {
        "seed": seed,
        "split": split,
        "condition": condition,
        "method": method,
        "final_mode_nll": metrics["final_mode_nll"],
        "final_mode_accuracy": metrics["final_mode_accuracy"],
        "final_mode_brier": metrics["final_mode_brier"],
        "final_posterior_entropy": metrics["final_posterior_entropy"],
        "windows_to_confidence": metrics["windows_to_confidence"],
        "predictive_response_nll": float(predictive_nll.mean()),
    }


def _evaluate(
    actions: torch.Tensor,
    responses: torch.Tensor,
    modes: torch.Tensor,
    response_noise: float,
    transition_probability: float,
    split: str,
    condition: str,
    seed: int,
    adaptive_base_hazard: float,
    adaptive_max_hazard: float,
    adaptive_surprise_threshold: float,
    adaptive_surprise_scale: float,
) -> list[dict[str, float | int | str]]:
    likelihoods = mode_log_likelihood(actions, responses, response_noise)
    mode_means = mode_response_means(actions)
    rows: list[dict[str, float | int | str]] = []
    for method in METHODS:
        if method in {"uniform-no-belief", "reset-bayes", "persistent-bayes", "adaptive-persistent", "mpogames-pomdp"}:
            before, after, nll = _categorical_run(
                likelihoods,
                "uniform-no-belief" if method == "uniform-no-belief" else method,
                transition_probability,
                adaptive_base_hazard,
                adaptive_max_hazard,
                adaptive_surprise_threshold,
                adaptive_surprise_scale,
            )
        else:
            config = UKFBeliefConfig(
                num_modes=int(mode_means.shape[-2]),
                observation_variance=float(response_noise) ** 2,
                process_variance=0.015 if method == "lucidgames-ukf" else 0.08,
                prior_variance=1.0,
                adaptive_process=method == "atom-ukf",
                max_process_variance=0.35,
            )
            before, after, nll = run_ukf_filter(mode_means, responses, config)
        del before
        rows.append(_metric_row(after, modes, nll, method, split, condition, seed))
    return rows


def _switch_data(
    actions: torch.Tensor,
    responses: torch.Tensor,
    modes: torch.Tensor,
    switch_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    means = mode_response_means(actions)
    residual = responses - means.gather(
        2, modes[:, None, None, None].expand(-1, actions.shape[1], 1, 2)
    ).squeeze(2)
    switched_modes = modes.clone()
    switched_modes = (switched_modes + 1) % means.shape[-2]
    response_modes = modes[:, None].expand(-1, actions.shape[1]).clone()
    response_modes[:, switch_window:] = switched_modes[:, None]
    switched_means = means.gather(
        2, response_modes[..., None, None].expand(-1, -1, 1, 2)
    ).squeeze(2)
    return switched_means + residual, response_modes


def _aggregate(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for method in METHODS:
        values = [
            float(row["predictive_response_nll"])
            for row in rows
            if row["method"] == method
        ]
        summary[method] = {
            "count": len(values),
            "predictive_response_nll_mean": statistics.fmean(values) if values else math.nan,
            "predictive_response_nll_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "final_mode_accuracy_mean": statistics.fmean(
                [float(row["final_mode_accuracy"]) for row in rows if row["method"] == method]
            ),
            "final_mode_nll_mean": statistics.fmean(
                [float(row["final_mode_nll"]) for row in rows if row["method"] == method]
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/belief_h2.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--output-dir", default="experiments/related_belief_baselines_20260905")
    parser.add_argument("--switch-window", type=int, default=5)
    parser.add_argument("--transition-probability", type=float, default=0.10)
    parser.add_argument("--adaptive-base-hazard", type=float, default=0.0)
    parser.add_argument("--adaptive-max-hazard", type=float, default=0.10)
    parser.add_argument("--adaptive-surprise-threshold", type=float, default=2.0)
    parser.add_argument("--adaptive-surprise-scale", type=float, default=0.10)
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    cfg = raw["dataset"]
    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        _set_seed(seed)
        for split, action_policy in (("val", "iid"), ("ood", str(cfg.get("ood_action_policy", "correlated")))):
            dataset = _dataset(raw, split, seed, action_policy)
            actions, responses, modes = _stack(dataset, args.device)
            rows.extend(_evaluate(
                actions, responses, modes, float(cfg.get("response_noise", 0.18)),
                args.transition_probability, split, "stationary", seed,
                args.adaptive_base_hazard, args.adaptive_max_hazard,
                args.adaptive_surprise_threshold, args.adaptive_surprise_scale,
            ))
            switched_responses, switched_mode_sequence = _switch_data(
                actions, responses, modes, args.switch_window
            )
            # Score switch runs against the time-varying mode sequence.
            switch_likelihoods = mode_log_likelihood(actions, switched_responses, float(cfg.get("response_noise", 0.18)))
            switch_means = mode_response_means(actions)
            for method in METHODS:
                if method in {"uniform-no-belief", "reset-bayes", "persistent-bayes", "adaptive-persistent", "mpogames-pomdp"}:
                    _, posterior, nll = _categorical_run(
                        switch_likelihoods, method, args.transition_probability,
                        args.adaptive_base_hazard, args.adaptive_max_hazard,
                        args.adaptive_surprise_threshold, args.adaptive_surprise_scale,
                    )
                else:
                    ukf_config = UKFBeliefConfig(
                        num_modes=int(switch_means.shape[-2]),
                        observation_variance=float(cfg.get("response_noise", 0.18)) ** 2,
                        process_variance=0.015 if method == "lucidgames-ukf" else 0.08,
                        adaptive_process=method == "atom-ukf",
                        max_process_variance=0.35,
                    )
                    _, posterior, nll = run_ukf_filter(switch_means, switched_responses, ukf_config)
                target = switched_mode_sequence
                true_probability = posterior.gather(-1, target[..., None]).squeeze(-1).clamp_min(1e-9)
                rows.append({
                    "seed": seed, "split": split, "condition": "mode-switch", "method": method,
                    "final_mode_nll": float((-true_probability[:, -1].log()).mean()),
                    "final_mode_accuracy": float((posterior[:, -1].argmax(-1) == target[:, -1]).float().mean()),
                    "final_mode_brier": float((posterior[:, -1] - torch.nn.functional.one_hot(target[:, -1], posterior.shape[-1]).float()).square().sum(-1).mean()),
                    "final_posterior_entropy": float((-(posterior[:, -1] * posterior[:, -1].clamp_min(1e-9).log()).sum(-1)).mean()),
                    "windows_to_confidence": float("nan"),
                    "predictive_response_nll": float(nll.mean()),
                })
            print(f"related_belief seed={seed} split={split} episodes={len(dataset)} device={args.device}", flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed", "split", "condition", "method", "final_mode_nll", "final_mode_accuracy",
        "final_mode_brier", "final_posterior_entropy", "windows_to_confidence", "predictive_response_nll",
    ]
    with (output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for condition in ("stationary", "mode-switch"):
        grouped[condition] = {}
        for split in ("val", "ood"):
            subset = [row for row in rows if row["condition"] == condition and row["split"] == split]
            grouped[condition][split] = _aggregate(subset)
    metadata = {
        "protocol": "tracer-related-belief-component-v1",
        "git": _git_state(),
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": args.seeds,
        "methods": METHODS,
        "transition_probability": args.transition_probability,
        "switch_window": args.switch_window,
        "adaptive_base_hazard": args.adaptive_base_hazard,
        "adaptive_max_hazard": args.adaptive_max_hazard,
        "adaptive_surprise_threshold": args.adaptive_surprise_threshold,
        "adaptive_surprise_scale": args.adaptive_surprise_scale,
        "claim_boundary": "method-level belief-maintenance comparison; not official complete planner reimplementation and no causal human claim",
        "runtime_seconds": None,
    }
    payload = {"metadata": metadata, "summary": grouped}
    start = time.perf_counter()
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    lines = [
        "# Related Belief-Maintenance Baselines",
        "",
        "This audit compares only the belief-maintenance components under the common TRACER H2 observation protocol.",
        "LUCIDGames-UKF and AToM-UKF are continuous parameter UKF baselines; MPOGames-POMDP is a discrete multimodal POMDP filter.",
        "They are method-level reimplementations, not official complete planner reimplementations.",
        "",
        f"Device: `{args.device}`; seeds: `{args.seeds}`; rows: `{len(rows)}`.",
        "",
    ]
    for condition in ("stationary", "mode-switch"):
        lines += [f"## {condition}", ""]
        for split in ("val", "ood"):
            lines.append(f"### {split}")
            lines.append("")
            lines.append("| Method | Final mode NLL | Final accuracy | Predictive response NLL |")
            lines.append("|---|---:|---:|---:|")
            for method in METHODS:
                item = grouped[condition][split][method]
                lines.append(
                    f"| `{method}` | {float(item['final_mode_nll_mean']):.4f} | {float(item['final_mode_accuracy_mean']):.4f} | {float(item['predictive_response_nll_mean']):.4f} |"
                )
            lines.append("")
    lines += [
        "## Interpretation Boundary",
        "",
        "The discrete MPOGames-style filter is evaluated in its natural categorical state space. The UKF baselines operate on a continuous latent parameter interpolating the three response anchors, so their results quantify robustness to a continuous-versus-discrete belief-model mismatch.",
        "The experiment does not claim that these rows reproduce the official game solvers or their original datasets.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    metadata["runtime_seconds"] = time.perf_counter() - start
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "protocol": metadata["protocol"],
                "git": metadata["git"],
                "claim_boundary": metadata["claim_boundary"],
                "artifacts": {
                    name: _sha256(output_dir / name)
                    for name in ("records.csv", "summary.json", "report.md", "metadata.json")
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved={output_dir / 'summary.json'} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
