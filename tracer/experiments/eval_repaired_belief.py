from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from tracer.belief.filter import adaptive_categorical_update, categorical_bayes_update, uniform_log_belief
from tracer.experiments.eval_related_belief_baselines import _dataset, _stack, _switch_data
from tracer.belief.synthetic import mode_log_likelihood


def evaluate(raw: dict, seeds: list[int], device: str, likelihood_scale: float) -> list[dict]:
    records = []
    for seed in seeds:
        for split, action_policy in (("val", "iid"), ("ood", "correlated")):
            dataset = _dataset(raw, split, seed, action_policy)
            actions, responses, modes = _stack(dataset, device)
            switched, switch_modes = _switch_data(actions, responses, modes, actions.shape[1] // 2)
            for condition, observed, labels in (
                ("stationary", responses, modes[:, None].expand(-1, actions.shape[1])),
                ("switch", switched, switch_modes),
            ):
                likelihoods = mode_log_likelihood(actions, observed, float(raw["dataset"].get("response_noise", 0.18)))
                for method in ("persistent", "legacy-scaled-surprise", "raw-surprise"):
                    posterior = uniform_log_belief(likelihoods.shape[-1], len(actions), device=device)
                    hazards = []
                    changes = []
                    accuracies = []
                    for window in range(actions.shape[1]):
                        likelihood = likelihoods[:, window]
                        if method == "persistent":
                            posterior = categorical_bayes_update(posterior, likelihood_scale * likelihood)
                            hazard = torch.zeros(len(actions), device=device)
                            changed = torch.zeros(len(actions), device=device)
                        else:
                            evidence = likelihood_scale * likelihood if method == "legacy-scaled-surprise" else likelihood
                            scale = 1.0 if method == "legacy-scaled-surprise" else likelihood_scale
                            posterior, hazard, _, changed = adaptive_categorical_update(posterior, evidence, likelihood_scale=scale)
                        accuracies.append(float((posterior.argmax(dim=-1) == labels[:, window]).float().mean()))
                        hazards.append(float(hazard.mean()))
                        changes.append(float(changed.float().mean()))
                    records.append({
                        "seed": seed, "split": split, "condition": condition, "method": method,
                        "final_accuracy": accuracies[-1],
                        "final_mode_nll": float(-posterior.gather(-1, labels[:, -1, None]).mean()),
                        "accuracy_by_window": accuracies,
                        "mean_hazard": sum(hazards) / len(hazards),
                        "change_fraction": sum(changes) / len(changes),
                    })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired raw-versus-tempered surprise regression audit.")
    parser.add_argument("--config", default="configs/belief_h2.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--likelihood-scale", type=float, default=0.05)
    parser.add_argument("--output-dir", default="experiments/repair_20260905/belief")
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    records = evaluate(raw, args.seeds, args.device, args.likelihood_scale)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "records.json").write_text(json.dumps({"parameters": vars(args), "records": records}, indent=2) + "\n")
    for condition in ("stationary", "switch"):
        for method in ("persistent", "legacy-scaled-surprise", "raw-surprise"):
            rows = [row for row in records if row["condition"] == condition and row["method"] == method]
            summary = {key: sum(row[key] for row in rows) / len(rows) for key in ("final_accuracy", "final_mode_nll", "mean_hazard", "change_fraction")}
            print(condition, method, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
