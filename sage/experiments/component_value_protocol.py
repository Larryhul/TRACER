"""Immutable source and scene contracts for continuation-value experiments."""

import json
from pathlib import Path

import torch

from sage.experiments.calibrate_component_memory import component_protocol_id
from sage.experiments.formal_manifest import sha256_file
from sage.response.continuation_value import FrozenContinuationValue


def load_value_artifact(path, checkpoint, raw, evaluation_data_seed):
    path = Path(path).resolve()
    summary = json.loads((path.parent / "summary.json").read_text())
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if summary.get("status") != "COMPLETED" or summary.get("changed_inputs") or summary["value_sha256"] != sha256_file(path):
        raise ValueError("completed unchanged continuation value checkpoint required")
    if payload["planning_checkpoint_sha256"] != sha256_file(Path(checkpoint)) or payload["component_protocol_id"] != component_protocol_id(raw, True):
        raise ValueError("continuation value source checkpoint/protocol mismatch")
    if evaluation_data_seed in {payload["training_data_seed"], payload["validation_data_seed"]}:
        raise ValueError("continuation evaluation must use separate scenes from fitting and selection")
    head = FrozenContinuationValue(**{name: payload[name] for name in ("center", "scale", "coefficients", "total_windows", "dt", "execute_steps")})
    return head, payload


def load_value_weight(path, value_checkpoint, raw, evaluation_data_seed):
    record = json.loads(Path(path).read_text())
    if record.get("status") != "COMPLETED" or record.get("phase") != "validation" or record.get("changed_inputs"):
        raise ValueError("completed unchanged validation value calibration required")
    if record["value_sha256"] != sha256_file(Path(value_checkpoint)) or record["component_protocol_id"] != component_protocol_id(raw, True):
        raise ValueError("continuation weight checkpoint/protocol mismatch")
    if record["data_seed"] == evaluation_data_seed:
        raise ValueError("value calibration and evaluation scenes must be separate")
    weight = float(record["selected"]["weight"])
    if not 0. <= weight <= 1.:
        raise ValueError("calibrated continuation weight must lie in [0, 1]")
    return weight
