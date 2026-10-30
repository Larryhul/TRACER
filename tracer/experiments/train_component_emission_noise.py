import argparse
import copy
import json
import math
from pathlib import Path

import torch
import yaml

from tracer.belief.correlated_emission import sequence_emission_likelihoods
from tracer.experiments.calibrate_component_memory import component_protocol_id, score_scales
from tracer.experiments.closed_loop_protocol import prepare_manifest
from tracer.experiments.eval_component_factorial import _local_sources
from tracer.experiments.eval_persistent_closed_loop import _load_model
from tracer.experiments.formal_manifest import sha256_file


@torch.no_grad()
def factual_emission_dataset(model, data, trace, execute_steps, device):
    episodes, windows = data["history"].shape[:2]
    selected = torch.full((episodes, windows), -1, dtype=torch.long)
    for row in trace:
        episode, window = row["episode"], row["window"]
        if selected[episode, window] != -1:
            raise ValueError("duplicate executed-prefix identity")
        selected[episode, window] = row["executed_candidate"]
    if (selected < 0).any() or (selected >= data["candidates"].shape[2]).any():
        raise ValueError("factual trace must identify every executed candidate")
    episode_indices, window_indices = torch.arange(episodes)[:, None], torch.arange(windows)[None]
    chosen = {key: data[key][episode_indices, window_indices, selected] for key in ("current", "candidates", "responses")}
    history = data["history"].reshape(episodes * windows, 1, -1, 2).to(device)
    current = chosen["current"].reshape(episodes * windows, 1, *chosen["current"].shape[-2:]).to(device)
    candidates = chosen["candidates"].reshape(episodes * windows, 1, *chosen["candidates"].shape[-3:]).to(device)
    means, log_stds = [], []
    for start in range(0, episodes * windows, 128):
        output = model(history[start:start + 128], current[start:start + 128], candidates[start:start + 128])
        means.append(output["mean"][:, 0, :, :execute_steps].cpu())
        log_stds.append(output["log_std"][:, 0, :, :execute_steps].cpu())
    shape = (episodes, windows, model.config.num_modes, execute_steps, 2)
    return {"means": torch.cat(means).reshape(shape), "log_std": torch.cat(log_stds).reshape(shape),
            "target": chosen["responses"][:, :, :execute_steps], "modes": data["mode"].long()}


def fit_emission_noise(data, iterations=250):
    if iterations < 1:
        raise ValueError("positive emission fitting iterations required")
    parameters = torch.tensor([math.log(.85), math.log(.55), .5], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([parameters], lr=.03)
    for iteration in range(iterations):
        noise, offset = parameters[:2].exp()
        correlation = parameters[2].sigmoid()
        likelihood = sequence_emission_likelihoods(data, noise, offset, correlation)
        mode = data["modes"][:, None, None].expand(-1, likelihood.shape[1], 1)
        loss = -likelihood.gather(-1, mode).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    noise, offset = parameters[:2].detach().exp()
    return {"noise_variance": float(noise), "offset_variance": float(offset), "correlation": float(parameters[2].detach().sigmoid())}


def main():
    parser = argparse.ArgumentParser(description="Fit shared correlated emission noise from executed training prefixes only")
    parser.add_argument("--config", required=True)
    parser.add_argument("--emission-checkpoint", required=True)
    parser.add_argument("--coverage-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    coverage = Path(args.coverage_dir).resolve()
    coverage_summary = json.loads((coverage / "summary.json").read_text(encoding="utf-8"))
    if coverage_summary.get("status") != "COMPLETED" or coverage_summary.get("changed_inputs"):
        raise ValueError("completed factual coverage data required")
    source_manifest = json.loads((coverage / "run_manifest.json").read_text(encoding="utf-8"))
    source_arguments = source_manifest["config"]["arguments"]
    source_raw = source_manifest["config"]["config"]
    comparison_raw = copy.deepcopy(source_raw)
    comparison_raw["candidate_source"].pop("retain_previous_plan", None)
    if component_protocol_id(comparison_raw, True) != component_protocol_id(raw, True):
        raise ValueError("coverage and emission execution protocols must match")
    checkpoint = Path(args.emission_checkpoint).resolve()
    model = _load_model(checkpoint, args.device, expected_dt=raw["simulation"]["dt"]).eval()
    paths = {
        split: [coverage / profile / f"{split}_rollouts.pt", coverage / profile / f"{split}_trace.json"]
        for split, profile in [("train", "uniform"), ("validation", "factorial")]
    }
    inputs = [checkpoint, coverage / "summary.json", coverage / "run_manifest.json", Path(args.config).resolve()] + [path for pair in paths.values() for path in pair]
    output = Path(args.output_dir)
    manifest = prepare_manifest(output, {"arguments": vars(args), "config": raw,
        "scope": "Shared conditional residual-noise model for every factorial policy, including Reset and Uniform; hidden mode only selects offline training likelihood, never a deployed input",
        "selection": "fixed training MLE iterations; validation proper predictive NLL is diagnostic only; no closed-loop outcome selection"}, inputs, _local_sources(), False)
    datasets = {}
    prefix = int(raw["simulation"]["execute_steps"])
    for split, pair in paths.items():
        data = torch.load(pair[0], map_location="cpu", weights_only=False)
        datasets[split] = factual_emission_dataset(model, data, json.loads(pair[1].read_text(encoding="utf-8")), prefix, args.device)
        torch.save(datasets[split], output / f"{split}.pt")
    parameters = fit_emission_noise(datasets["train"], args.iterations)
    diagnostics = {}
    for split, data in datasets.items():
        diagnostics[split] = {}
        for family, values in [("diagonal", {"noise_variance": 1., "offset_variance": 0., "correlation": 0.}), ("correlated", parameters)]:
            likelihood = sequence_emission_likelihoods(data, **values)
            diagnostics[split][family] = score_scales(likelihood, [.01, .025, .05, .1, .2, .5, 1.], 2 * prefix)
    changed = [path for group in ("sources", "checkpoints") for path, digest in manifest[group].items() if sha256_file(Path(path)) != digest]
    record = {"schema": "component-correlated-emission-v1", "status": "INVALIDATED" if changed else "COMPLETED", "changed_inputs": changed,
              "parameters": {**parameters, "execute_steps": prefix, "dt": raw["simulation"]["dt"]},
              "emission_checkpoint_sha256": sha256_file(checkpoint), "component_protocol_id": component_protocol_id(raw, True),
              "training_component_protocol_id": component_protocol_id(source_raw, True),
              "coverage_retains_previous_plan": bool(source_raw["candidate_source"].get("retain_previous_plan", False)),
              "fitting_data_seeds": [source_arguments["train_data_seed"], source_arguments["validation_data_seed"]],
              "training_scenes": len(datasets["train"]["modes"]), "validation_scenes": len(datasets["validation"]["modes"]), "diagnostics": diagnostics}
    (output / "emission_noise.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps({key: value for key, value in record.items() if key != "diagnostics"}, indent=2) + "\n")
    print(record["status"], record["parameters"], flush=True)
    if changed:
        raise RuntimeError("emission fitting inputs changed")


if __name__ == "__main__":
    main()
