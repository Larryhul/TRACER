import argparse
import json
from pathlib import Path

import torch

from tracer.experiments.closed_loop_protocol import prepare_manifest
from tracer.experiments.eval_component_factorial import _local_sources
from tracer.experiments.formal_manifest import sha256_file
from tracer.experiments.train_component_value import value_error
from tracer.response.continuation_value import FrozenContinuationValue


def pooled_value_coefficients(data, parent, regularization):
    if regularization <= 0 or not torch.isfinite(torch.as_tensor(regularization)):
        raise ValueError("positive finite pooled value regularization required")
    normalized = (data["features"] - parent.center) / parent.scale
    design = torch.cat([torch.ones_like(normalized[:, :1]), normalized], -1)
    target = data["returns"] / data["remaining"]
    penalty = torch.eye(design.shape[-1], dtype=design.dtype, device=design.device) * regularization
    penalty[0, 0] = 0.
    return torch.linalg.solve(design.T @ design + penalty, design.T @ target)


def choose_pooled_value_head(train, validation, parent, regularization, strengths):
    if not strengths or any(not 0. <= strength <= 1. for strength in strengths):
        raise ValueError("mode-specific value strengths must lie in [0, 1]")
    pooled = pooled_value_coefficients(train, parent, regularization)
    trials = []
    for strength in strengths:
        coefficients = parent.coefficients if strength == 1. else pooled[None] + strength * (parent.coefficients - pooled[None])
        head = FrozenContinuationValue(parent.center, parent.scale, coefficients, parent.total_windows, parent.dt, parent.execute_steps)
        trials.append((value_error(head, validation), strength, head))
    error, strength, head = min(trials, key=lambda entry: (entry[0], entry[1]))
    return head, {"validation_mse": error, "mode_specific_strength": strength}, [
        {"validation_mse": loss, "mode_specific_strength": weight} for loss, weight, _ in trials
    ]


def main():
    parser = argparse.ArgumentParser(description="Partially pool mode-specific continuation regressions using existing disjoint factual train/validation returns")
    parser.add_argument("--parent-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0., .25, .5, .75, 1.])
    args = parser.parse_args()
    torch.set_num_threads(4)
    parent_dir = Path(args.parent_dir).resolve()
    parent_summary = json.loads((parent_dir / "summary.json").read_text(encoding="utf-8"))
    if parent_summary.get("status") != "COMPLETED" or parent_summary.get("changed_inputs") or parent_summary["value_sha256"] != sha256_file(parent_dir / "value.pt"):
        raise ValueError("completed unchanged parent continuation value required")
    parent_manifest = json.loads((parent_dir / "run_manifest.json").read_text(encoding="utf-8"))
    for group in ("sources", "checkpoints"):
        for path, digest in parent_manifest[group].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError("parent value source/checkpoint changed")
    payload = torch.load(parent_dir / "value.pt", weights_only=True, map_location="cpu")
    if payload["training_data_seed"] == payload["validation_data_seed"]:
        raise ValueError("pooled value selection requires separate training and validation scenes")
    parent = FrozenContinuationValue(**{name: payload[name] for name in ("center", "scale", "coefficients", "total_windows", "dt", "execute_steps")})
    inputs = [parent_dir / filename for filename in ("summary.json", "run_manifest.json", "value.pt", "train.pt", "validation.pt")]
    output = Path(args.output_dir)
    manifest = prepare_manifest(output, {"arguments": vars(args),
        "selection": "validation proper factual per-window return MSE; parent and pooled models share training returns and parent train-only normalization",
        "scope": "Common partially pooled continuation head for all six policies. No mode label or hidden simulator state is supplied at deployment; no closed-loop cost or component difference selects pooling."}, inputs, _local_sources(), False)
    train = torch.load(parent_dir / "train.pt", weights_only=True, map_location="cpu")
    validation = torch.load(parent_dir / "validation.pt", weights_only=True, map_location="cpu")
    head, selected, trials = choose_pooled_value_head(train, validation, parent, payload["regularization"], args.strengths)
    updated = {**payload, "coefficients": head.coefficients, "mode_specific_strength": selected["mode_specific_strength"], "parent_value_sha256": sha256_file(parent_dir / "value.pt")}
    torch.save(updated, output / "value.pt")
    changed = [path for group in ("sources", "checkpoints") for path, digest in manifest[group].items() if sha256_file(Path(path)) != digest]
    summary = {"status": "INVALIDATED" if changed else "COMPLETED", "changed_inputs": changed, "selected": selected, "trials": trials,
               "parent_validation_mse": value_error(parent, validation), "value_sha256": sha256_file(output / "value.pt"),
               "training_scenes": len(train["modes"]) // parent.total_windows, "validation_scenes": len(validation["modes"]) // parent.total_windows}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(summary, flush=True)
    if changed:
        raise RuntimeError("pooled value refinement inputs changed")


if __name__ == "__main__":
    main()
