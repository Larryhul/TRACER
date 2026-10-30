from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tracer.experiments.train_rolling_external_belief import (
    attach_modes,
    evaluate_factual_modes,
    first_prefix_rows,
    grouped_candidate_costs,
    predict_all_modes,
    prefix_likelihood,
    recover_training_labels,
    sha256,
)
from tracer.response.candidate_ranking import candidate_ranking_metrics
from tracer.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig


def causal_row_beliefs(data, prediction, target, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("likelihood scale must be positive and finite")
    factual_rows = first_prefix_rows(data)
    likelihoods = prefix_likelihood(prediction, target)
    modes = prediction["mean"].shape[1]
    uniform = likelihoods.new_full((modes,), -np.log(modes))
    persistent = uniform.expand(len(target), -1).clone()
    reset = persistent.clone()
    tracks = {}
    before_by_state = {}
    for row in factual_rows:
        identity = (int(data["case"][row]), int(data["human"][row]))
        current, last = tracks.get(identity, (uniform, uniform))
        before_by_state[(*identity, int(data["window"][row]))] = current, last
        evidence = likelihoods[row] * scale
        tracks[identity] = (current + evidence).log_softmax(-1), evidence.log_softmax(-1)
    for row in range(len(target)):
        key = int(data["case"][row]), int(data["human"][row]), int(data["window"][row])
        persistent[row], reset[row] = before_by_state[key]
    return {"persistent": persistent.exp(), "reset": reset.exp(),
            "uniform": uniform.exp().expand_as(persistent)}


def paired_case_interval(differences, cases, seed=839731):
    values = np.asarray(differences, dtype=np.float64)
    cases = np.asarray(cases)
    grouped = np.asarray([values[cases == case].mean() for case in np.unique(cases)])
    if len(grouped) < 2:
        return {"case_mean": float(grouped.mean()), "ci95": None, "case_count": len(grouped)}
    generator = np.random.default_rng(seed)
    bootstraps = generator.choice(grouped, (2000, len(grouped)), replace=True).mean(-1)
    return {"case_mean": float(grouped.mean()), "ci95": np.quantile(bootstraps, [.025, .975]).tolist(),
            "case_count": len(grouped)}


@torch.no_grad()
def audit_checkpoint(checkpoint_path, data_dir, external_root, batch_size, device):
    checkpoint_path, data_dir = Path(checkpoint_path), Path(data_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    provenance = checkpoint["provenance"]
    expected_hash = provenance["source_data_sha256"]["val"]
    if sha256(data_dir / "val.npz") != expected_hash:
        raise ValueError("validation data must match checkpoint provenance")
    with np.load(data_dir / "val.npz") as loaded:
        data = dict(loaded)
    if set(map(int, data["case"])) != set(provenance["case_splits"]["val"]):
        raise ValueError("validation case split does not match checkpoint")
    speeds = recover_training_labels({"val": data}, Path(external_root).resolve())
    data = attach_modes(data, speeds, np.asarray(checkpoint["speed_thresholds_train_only"]))
    model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    all_rows = np.arange(len(data["target"]))
    batches = [predict_all_modes(model, data, all_rows[start:start + batch_size], device)[0]
               for start in range(0, len(all_rows), batch_size)]
    prediction = {name: torch.cat([batch[name] for batch in batches]).cpu() for name in ("mean", "log_std")}
    target = torch.as_tensor(data["target"])
    calibration = checkpoint["belief_calibration"]
    if calibration["observation_stride"] != provenance["source_manifest"]["parameters"]["collection_stride"]:
        raise ValueError("observation stride must match validation collection cadence")
    weights = causal_row_beliefs(data, prediction, target, calibration["selected"]["likelihood_scale"])
    weights["true_mode_diagnostic"] = torch.nn.functional.one_hot(
        torch.as_tensor(data["mode"]), model.config.num_modes
    ).to(target)
    costs = {}
    for policy, probabilities in weights.items():
        costs[policy], oracle, valid = grouped_candidate_costs(
            data, all_rows, prediction, target, "cpu", model.config.dt, mode_weights=probabilities
        )
    costs["zero_response"] = torch.zeros_like(oracle)
    group_cases = list(dict.fromkeys(zip(data["case"], data["window"], data["human"])))
    group_cases = np.asarray([identity[0] for identity in group_cases])
    best = oracle.masked_fill(~valid, float("inf")).min(-1).values
    selections = {name: cost.masked_fill(~valid, float("inf")).argmin(-1) for name, cost in costs.items()}
    regrets = {name: oracle.gather(-1, selected[:, None]).squeeze(-1) - best
               for name, selected in selections.items()}
    metrics = {}
    for policy, cost in costs.items():
        metrics[policy] = candidate_ranking_metrics(cost, oracle, valid_mask=valid)
        metrics[policy]["response_only_selection_change_vs_uniform"] = float((selections[policy] != selections["uniform"]).float().mean())
        for baseline in ("uniform", "zero_response"):
            metrics[policy][f"regret_difference_vs_{baseline}"] = paired_case_interval(
                (regrets[policy] - regrets[baseline]).cpu().numpy(), group_cases
            )
    return {"checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": sha256(checkpoint_path),
            "audit_protocol": "mode-expectation-before-observation-v2",
            "audit_source_sha256": {name: sha256(Path(__file__).resolve().parents[1] / name)
                                    for name in ("experiments/audit_external_candidate_belief.py",
                                                 "experiments/train_rolling_external_belief.py", "response/candidate_ranking.py")},
            "phase": "validation", "case_count": len(np.unique(data["case"])), "group_count": len(group_cases),
            "cost_definition": "planner human cost: mean time, summed velocity coordinates, expectation after per-mode cost",
            "boundary": "per-human response-only ranking; not full robot action selection or closed-loop navigation improvement",
            "calibration": calibration["selected"],
            "factual_modes": evaluate_factual_modes(model, data, batch_size, device), "policies": metrics}


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--external-root", default="raw_data/external/source_repos/sicnav_official_filtered")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("output already exists")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    torch.set_num_threads(2)
    results = []
    for checkpoint in args.checkpoints:
        result = audit_checkpoint(checkpoint, args.data_dir, args.external_root, args.batch_size, args.device)
        results.append(result)
        print(json.dumps(finite_json(result)), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(finite_json(results), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
