"""Choose continuation strength by Uniform actual returns, never ablation gaps."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from tracer.experiments.calibrate_component_memory import component_protocol_id
from tracer.experiments.closed_loop_protocol import prepare_manifest
from tracer.experiments.component_value_protocol import load_value_artifact
from tracer.experiments.eval_component_factorial import FACTORIAL_POLICIES, _local_sources
from tracer.experiments.eval_persistent_closed_loop import _evaluate_policy, _load_model
from tracer.experiments.finetune_h3_response_ranking import _adapter_from_config, _make_spec
from tracer.experiments.formal_manifest import sha256_file
from tracer.response.closed_loop import PlanningCostConfig


def choose_value_weight(records):
    if not records or not any(row["weight"] == 0. for row in records):
        raise ValueError("continuation calibration must include the unchanged parent")
    if any(not all(np.isfinite(row[name]) for name in ("weight", "cost", "collision")) or not 0 <= row["weight"] <= 1 for row in records):
        raise ValueError("finite continuation outcomes and weights in [0, 1] required")
    parent = next(row for row in records if row["weight"] == 0.)
    eligible = [row for row in records if row["collision"] <= parent["collision"]]
    return min(eligible, key=lambda row: (row["cost"], row["weight"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=924609)
    parser.add_argument("--weights", type=float, nargs="+", default=[0., .25, .5, 1.])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    choose_value_weight([{"weight": weight, "cost": 0., "collision": 0.} for weight in args.weights])
    torch.set_num_threads(4)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw["simulation"]["data_seed"] = args.data_seed
    checkpoint, value_checkpoint = Path(args.checkpoint).resolve(), Path(args.value_checkpoint).resolve()
    value_model, _ = load_value_artifact(value_checkpoint, checkpoint, raw, args.data_seed)
    model = _load_model(checkpoint, args.device)
    adapter = _adapter_from_config(raw, args.device)
    output = Path(args.output_dir)
    inputs = [checkpoint, value_checkpoint, value_checkpoint.parent / "summary.json", value_checkpoint.parent / "run_manifest.json", Path(raw["candidate_source"]["checkpoint"]).resolve()]
    manifest = prepare_manifest(output, {"arguments": vars(args), "config": raw, "selection": "minimum mean actual Uniform-Joint episode cost subject to no additional observed collision; ties choose smaller weight", "scope": "independent validation calibration; no Persistent/Reset/Joint/Additive gap selection"}, inputs, _local_sources(), False)
    records = []
    reference_rows = None
    for weight in args.weights:
        result = _evaluate_policy(model, FACTORIAL_POLICIES[0], _make_spec(raw, "id"), args.episodes, args.device, PlanningCostConfig(**raw["planning"]), .05,
                                  candidate_adapter=adapter, prefix_objective=True, response_gate_min=1., response_gate_max=1.,
                                  continuation_value_model=value_model, continuation_value_weight=weight)
        rows = result["window_records"]
        if reference_rows is None:
            reference_rows = rows
        for first, current in zip(reference_rows, rows):
            if first[0]["state_hash"] != current[0]["state_hash"] or any(left["candidate_seed"] != right["candidate_seed"] or left["environment_rng_hash"] != right["environment_rng_hash"] for left, right in zip(first, current)):
                raise ValueError("weight calibration must preserve paired states and random streams")
        record = {"weight": weight, "cost": float(np.mean([row["cumulative_executed_cost"] for row in result["episode_records"]])), "collision": float(np.mean([row["episode_collision"] for row in result["episode_records"]]))}
        records.append(record)
        (output / f"weight_{weight:g}.json").write_text(json.dumps(result, allow_nan=True))
        print(record, flush=True)
    changed = [path for group in ("sources", "checkpoints") for path, digest in manifest[group].items() if sha256_file(Path(path)) != digest]
    payload = {"status": "INVALIDATED" if changed else "COMPLETED", "phase": "validation", "changed_inputs": changed, "data_seed": args.data_seed,
               "component_protocol_id": component_protocol_id(raw, True), "value_sha256": sha256_file(value_checkpoint), "selected": choose_value_weight(records), "records": records}
    (output / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("selected", payload["selected"], flush=True)
    if changed:
        raise RuntimeError("value calibration inputs changed")


if __name__ == "__main__":
    main()
