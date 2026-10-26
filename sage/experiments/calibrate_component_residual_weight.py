"""Calibrate constant residual reliability independently of belief confidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sage.experiments.closed_loop_protocol import audit_candidate_protocol, prepare_manifest
from sage.experiments.eval_component_factorial import _local_sources
from sage.experiments.eval_persistent_closed_loop import Policy, _evaluate_policy, _load_model
from sage.experiments.finetune_h3_response_ranking import _adapter_from_config, _make_spec
from sage.experiments.formal_manifest import sha256_file
from sage.experiments.calibrate_component_memory import component_protocol_id
from sage.response.closed_loop import PlanningCostConfig


def choose_residual_weight(records):
    if not records or not any(row["weight"] == 0.0 for row in records):
        raise ValueError("residual calibration requires an additive control")
    if any(not all(np.isfinite(row[name]) for name in ("weight", "cost", "collision")) for row in records):
        raise ValueError("nonfinite residual calibration outcome")
    baseline_collision = next(row["collision"] for row in records if row["weight"] == 0.0)
    eligible = [row for row in records if row["collision"] <= baseline_collision]
    return min(eligible, key=lambda row: (row["cost"], row["weight"]))


def load_residual_weight(path, checkpoint, raw, data_seed):
    record = json.loads(Path(path).read_text())
    if record.get("status") != "COMPLETED" or record.get("phase") != "validation":
        raise ValueError("completed validation residual calibration required")
    if record["checkpoint_sha256"] != sha256_file(Path(checkpoint)):
        raise ValueError("residual calibration checkpoint mismatch")
    if record["component_protocol_id"] != component_protocol_id(raw, True):
        raise ValueError("residual calibration protocol mismatch")
    if record["data_seed"] == data_seed:
        raise ValueError("residual calibration and evaluation scene seeds must be separate")
    weight = record["selected"]["weight"]
    if not np.isfinite(weight) or not 0. <= weight <= 1.:
        raise ValueError("residual calibration weight must lie in [0, 1]")
    return float(weight)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=924611)
    parser.add_argument("--weights", type=float, nargs="+", default=[0., .25, .5, .75, 1.])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if any(not np.isfinite(weight) or not 0 <= weight <= 1 for weight in args.weights) or 0.0 not in args.weights:
        raise ValueError("weights must include zero and lie in [0, 1]")
    torch.set_num_threads(4)
    raw = yaml.safe_load(Path(args.config).read_text())
    raw["simulation"]["data_seed"] = args.data_seed
    spec = _make_spec(raw, "id")
    model = _load_model(Path(args.checkpoint), args.device)
    if model.config.dt != spec.dt:
        raise ValueError("residual calibration timestep mismatch")
    adapter = _adapter_from_config(raw, args.device)
    output = Path(args.output_dir)
    manifest = prepare_manifest(output, {
        "arguments": vars(args), "config": raw,
        "selection": "minimum Uniform same-state factual cost among weights with no increase in collision versus Additive",
        "belief_control": "Uniform only; never select a weight by Persistent-minus-Reset advantage",
        "claim_boundary": "validation-only constant reliability shrinkage, not adaptive belief-confidence gating",
    }, [Path(args.checkpoint).resolve(), Path(raw["candidate_source"]["checkpoint"]).resolve()], _local_sources(), False)
    arguments = dict(
        model=model, spec=spec, episodes=args.episodes, device=args.device,
        planning_cost=PlanningCostConfig(**raw["planning"]), likelihood_scale=0.05,
        candidate_adapter=adapter, evaluation_mode="same_state", prefix_objective=True,
        retain_previous_plan=bool(raw.get("candidate_source", {}).get("retain_previous_plan", False)),
    )
    trace = []
    reference = _evaluate_policy(
        **arguments, policy=Policy("reference", "uniform", True, response_gate=True),
        captured_reference_trace=trace, response_gate_min=1., response_gate_max=1.,
    )
    records = []
    policies = {"reference": reference}
    for weight in args.weights:
        result = reference if weight == 1. else _evaluate_policy(
            **arguments, policy=Policy(f"weight-{weight}", "uniform", True, response_gate=True),
            reference_trace=trace, response_gate_min=weight, response_gate_max=weight,
        )
        policies[f"weight-{weight}"] = result
        record = {"weight": weight, "cost": result["realized_total_cost"]["mean"], "collision": result["collision_rate"]["mean"]}
        records.append(record)
        print(record, flush=True)
    audit = audit_candidate_protocol({"checkpoint": {"id": policies}}, list(policies), "same_state")
    changed = [path for group in ("sources", "checkpoints") for path, digest in manifest[group].items() if sha256_file(Path(path)) != digest]
    summary = {
        "status": "COMPLETED" if not changed else "INVALIDATED", "phase": "validation",
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)), "data_seed": args.data_seed,
        "component_protocol_id": component_protocol_id(raw, True),
        "selected": choose_residual_weight(records), "records": records,
        "audit": audit, "changed_inputs": changed,
    }
    (output / "policies.json").write_text(json.dumps(policies, allow_nan=True))
    (output / "calibration.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if changed:
        raise RuntimeError("residual weight calibration inputs changed")


if __name__ == "__main__":
    main()
