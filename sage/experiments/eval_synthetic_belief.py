from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from sage.belief.baselines import RecurrentBeliefBaseline, RecurrentBeliefConfig
from sage.belief.filter import run_categorical_filter, uniform_log_belief
from sage.belief.likelihood import ModeConditionedResponseLikelihood, ModeLikelihoodConfig
from sage.belief.metrics import belief_classification_metrics, predictive_mixture_nll
from sage.belief.synthetic import (
    BeliefSequenceSpec,
    PersistentBeliefSequenceDataset,
    mode_log_likelihood,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_dataset(
    raw: dict,
    split: str,
    seed: int | None = None,
    action_policy: str = "iid",
) -> PersistentBeliefSequenceDataset:
    cfg = raw["dataset"]
    count_key = f"{split}_episodes"
    return PersistentBeliefSequenceDataset(
        BeliefSequenceSpec(
            split=split,
            num_episodes=int(cfg[count_key]),
            windows=int(cfg.get("windows", 10)),
            seed=int(seed if seed is not None else cfg.get("seed", 2027)),
            response_noise=float(cfg.get("response_noise", 0.18)),
            num_modes=int(cfg.get("num_modes", 3)),
            action_policy=action_policy,
        )
    )


def _train_recurrent(
    model: RecurrentBeliefBaseline,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in loader:
            actions = batch["actions"].to(device)
            responses = batch["responses"].to(device)
            modes = batch["mode"].to(device)
            log_probabilities = model(actions, responses)
            targets = modes[:, None].expand(-1, actions.shape[1]).reshape(-1)
            loss = F.nll_loss(log_probabilities.reshape(-1, model.config.num_modes), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * actions.shape[0]
            count += actions.shape[0]
        print(f"recurrent epoch={epoch}/{epochs} train_mode_nll={loss_sum / count:.6f}", flush=True)


def _train_likelihood(
    model: ModeConditionedResponseLikelihood,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in loader:
            actions = batch["actions"].to(device)
            responses = batch["responses"].to(device)
            modes = batch["mode"].to(device)[:, None].expand(-1, actions.shape[1])
            log_likelihood = model.all_mode_log_likelihood(actions, responses)
            selected = log_likelihood.gather(dim=-1, index=modes[..., None]).squeeze(-1)
            loss = -selected.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * actions.shape[0]
            count += actions.shape[0]
        print(f"likelihood epoch={epoch}/{epochs} train_response_nll={loss_sum / count:.6f}", flush=True)


@torch.no_grad()
def _evaluate(
    recurrent_model: RecurrentBeliefBaseline,
    likelihood_model: ModeConditionedResponseLikelihood,
    loader: DataLoader,
    device: str,
    response_noise: float,
    confidence_threshold: float,
) -> dict[str, dict]:
    recurrent_model.eval()
    likelihood_model.eval()
    actions_all = []
    responses_all = []
    modes_all = []
    recurrent_after_all = []
    for batch in loader:
        actions = batch["actions"].to(device)
        responses = batch["responses"].to(device)
        actions_all.append(actions.cpu())
        responses_all.append(responses.cpu())
        modes_all.append(batch["mode"].cpu())
        recurrent_after_all.append(recurrent_model(actions, responses).exp().cpu())

    actions = torch.cat(actions_all)
    responses = torch.cat(responses_all)
    modes = torch.cat(modes_all)
    recurrent_after = torch.cat(recurrent_after_all)
    analytic_log_likelihoods = mode_log_likelihood(actions, responses, response_noise)
    learned_log_likelihoods = likelihood_model.all_mode_log_likelihood(
        actions.to(device), responses.to(device)
    ).cpu()
    episodes, windows, num_modes = analytic_log_likelihoods.shape
    prior_log = uniform_log_belief(num_modes, episodes)
    prior = prior_log.exp()

    analytic_before_log, analytic_after_log = run_categorical_filter(
        analytic_log_likelihoods, prior_log_belief=prior_log
    )
    persistent_before_log, persistent_after_log = run_categorical_filter(
        learned_log_likelihoods, prior_log_belief=prior_log
    )
    reset_before_log, reset_after_log = run_categorical_filter(
        learned_log_likelihoods, prior_log_belief=prior_log, reset_each_window=True
    )
    frozen = prior[:, None, :].expand(-1, windows, -1)
    recurrent_before = torch.cat([prior[:, None, :], recurrent_after[:, :-1]], dim=1)
    oracle = F.one_hot(modes, num_classes=num_modes).to(torch.float32)
    oracle = oracle[:, None, :].expand(-1, windows, -1)

    beliefs = {
        "frozen": (frozen, frozen, learned_log_likelihoods),
        "reset": (reset_before_log.exp(), reset_after_log.exp(), learned_log_likelihoods),
        "persistent": (
            persistent_before_log.exp(),
            persistent_after_log.exp(),
            learned_log_likelihoods,
        ),
        "analytic-persistent": (
            analytic_before_log.exp(),
            analytic_after_log.exp(),
            analytic_log_likelihoods,
        ),
        "recurrent": (recurrent_before, recurrent_after, learned_log_likelihoods),
        "oracle": (oracle, oracle, learned_log_likelihoods),
    }
    results: dict[str, dict] = {}
    for name, (before, after, predictive_likelihoods) in beliefs.items():
        metrics = belief_classification_metrics(after, modes, confidence_threshold)
        predictive_nll, predictive_curve = predictive_mixture_nll(
            before, predictive_likelihoods
        )
        metrics["predictive_response_nll"] = predictive_nll
        metrics["predictive_response_nll_by_window"] = predictive_curve
        results[name] = metrics
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate persistent ID-bound beliefs on known likelihoods.")
    parser.add_argument("--config", default="configs/belief_h2.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    dataset_cfg = raw["dataset"]
    model_cfg = raw.get("model", {})
    training_cfg = raw.get("training", {})
    evaluation_cfg = raw.get("evaluation", {})
    seed = int(args.seed if args.seed is not None else dataset_cfg.get("seed", 2027))
    _set_seed(seed)
    torch.set_num_threads(int(training_cfg.get("num_threads", 2)))

    train_dataset = _make_dataset(raw, "train", seed)
    val_dataset = _make_dataset(raw, "val", seed)
    ood_dataset = _make_dataset(
        raw,
        "ood",
        seed,
        action_policy=str(dataset_cfg.get("ood_action_policy", "correlated")),
    )
    loader_kwargs = {
        "batch_size": int(training_cfg.get("batch_size", 128)),
        "num_workers": int(training_cfg.get("num_workers", 0)),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    ood_loader = DataLoader(ood_dataset, shuffle=False, **loader_kwargs)
    config = RecurrentBeliefConfig(
        num_modes=int(dataset_cfg.get("num_modes", 3)),
        hidden_dim=int(model_cfg.get("recurrent_hidden_dim", 32)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    recurrent_model = RecurrentBeliefBaseline(config).to(args.device)
    likelihood_config = ModeLikelihoodConfig(
        num_modes=int(dataset_cfg.get("num_modes", 3)),
        hidden_dim=int(model_cfg.get("likelihood_hidden_dim", 64)),
    )
    likelihood_model = ModeConditionedResponseLikelihood(likelihood_config).to(args.device)
    print(
        f"train_episodes={len(train_dataset)} val_episodes={len(val_dataset)} "
        f"windows={dataset_cfg.get('windows', 10)} recurrent_parameters="
        f"{sum(parameter.numel() for parameter in recurrent_model.parameters())} "
        f"likelihood_parameters={sum(parameter.numel() for parameter in likelihood_model.parameters())} "
        f"device={args.device}",
        flush=True,
    )
    _train_recurrent(
        recurrent_model,
        train_loader,
        args.device,
        epochs=int(args.epochs or training_cfg.get("epochs", 20)),
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    _train_likelihood(
        likelihood_model,
        train_loader,
        args.device,
        epochs=int(args.epochs or training_cfg.get("epochs", 20)),
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    results = _evaluate(
        recurrent_model,
        likelihood_model,
        val_loader,
        args.device,
        response_noise=float(dataset_cfg.get("response_noise", 0.18)),
        confidence_threshold=float(evaluation_cfg.get("confidence_threshold", 0.8)),
    )
    ood_results = _evaluate(
        recurrent_model,
        likelihood_model,
        ood_loader,
        args.device,
        response_noise=float(dataset_cfg.get("response_noise", 0.18)),
        confidence_threshold=float(evaluation_cfg.get("confidence_threshold", 0.8)),
    )
    output_dir = Path(args.output_dir or raw.get("output_dir", "experiments/belief/h2_analytic"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "ood_metrics.json").write_text(
        json.dumps(ood_results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "model": recurrent_model.state_dict(),
            "model_config": vars(config),
            "seed": seed,
        },
        output_dir / "recurrent.pt",
    )
    torch.save(
        {
            "model": likelihood_model.state_dict(),
            "model_config": vars(likelihood_config),
            "seed": seed,
        },
        output_dir / "likelihood.pt",
    )
    for name, metrics in results.items():
        print(
            f"baseline={name} final_accuracy={metrics['final_mode_accuracy']:.5f} "
            f"final_mode_nll={metrics['final_mode_nll']:.5f} "
            f"brier={metrics['final_mode_brier']:.5f} "
            f"windows_to_80={metrics['windows_to_confidence']:.5f} "
            f"predictive_response_nll={metrics['predictive_response_nll']:.5f}",
            flush=True,
        )
    for name, metrics in ood_results.items():
        print(
            f"ood_baseline={name} final_accuracy={metrics['final_mode_accuracy']:.5f} "
            f"final_mode_nll={metrics['final_mode_nll']:.5f} "
            f"windows_to_80={metrics['windows_to_confidence']:.5f} "
            f"predictive_response_nll={metrics['predictive_response_nll']:.5f}",
            flush=True,
        )
    print(f"saved_results={output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
