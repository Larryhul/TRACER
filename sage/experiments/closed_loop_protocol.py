from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sage.experiments.formal_manifest import sha256_file


PROTOCOL_VERSION = "h3-state-consistent-v1"


@dataclass(frozen=True)
class ReferenceWindow:
    simulator: Any
    pool: Any
    executed_candidate: int


def prepare_manifest(
    output_dir: Path,
    effective_config: dict[str, Any],
    checkpoints: list[Path],
    sources: list[Path],
    resume: bool,
) -> dict[str, Any]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "config": effective_config,
        "checkpoints": {str(path): sha256_file(path) for path in checkpoints},
        "sources": {str(path): sha256_file(path) for path in sources},
    }
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
    payload["fingerprint"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    path = output_dir / "run_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if not resume:
            raise ValueError("output directory already has a manifest; use --resume or a new directory")
        if previous != payload:
            raise ValueError("resume protocol/checkpoint/source mismatch; use a new output directory")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("cannot mix an unversioned/legacy result directory with the corrected protocol")
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def audit_candidate_protocol(
    runs: dict[str, Any], policy_names: list[str], evaluation_mode: str
) -> dict[str, Any]:
    checked_slots = 0
    checked_comparisons = 0
    same_state_comparisons = 0
    divergent_state_comparisons = 0
    divergent_candidate_context_comparisons = 0
    max_contract_error = 0.0
    for seed_runs in runs.values():
        for policies in seed_runs.values():
            reference_episodes = policies[policy_names[0]]["window_records"]
            for policy_name in policy_names:
                if len(policies[policy_name]["window_records"]) != len(reference_episodes):
                    raise ValueError("policy episode counts do not align")
            for episode, reference_windows in enumerate(reference_episodes):
                for policy_name in policy_names:
                    if len(policies[policy_name]["window_records"][episode]) != len(reference_windows):
                        raise ValueError("policy window counts do not align")
                for window, reference in enumerate(reference_windows):
                    checked_slots += 1
                    for policy_name in policy_names:
                        record = policies[policy_name]["window_records"][episode][window]
                        for key in ("candidate_position_error", "candidate_velocity_error", "candidate_integration_error", "execution_position_error"):
                            value = float(record[key])
                            if not 0.0 <= value <= 1e-3:
                                raise ValueError(f"invalid runtime contract diagnostic: {key}={value}")
                            max_contract_error = max(max_contract_error, value)
                        if policy_name == policy_names[0]:
                            continue
                        checked_comparisons += 1
                        if record["candidate_seed"] != reference["candidate_seed"]:
                            raise ValueError("candidate random streams are not paired")
                        if record["environment_rng_hash"] != reference["environment_rng_hash"]:
                            raise ValueError("environment random streams are not paired")
                        retention = bool(reference.get("retained_plan_enabled", False))
                        if bool(record.get("retained_plan_enabled", False)) != retention:
                            raise ValueError("policies must share the candidate retention rule")
                        if retention and (not reference.get("candidate_context_hash") or not record.get("candidate_context_hash")):
                            raise ValueError("retained candidates require recorded planning context")
                        same_context = not retention or record["candidate_context_hash"] == reference["candidate_context_hash"]
                        if not same_context:
                            divergent_candidate_context_comparisons += 1
                            if evaluation_mode == "same_state":
                                raise ValueError("same-state diagnostic must share candidate planning context")
                        same_state = record["state_hash"] == reference["state_hash"]
                        if same_state:
                            same_state_comparisons += 1
                            if same_context and record["candidate_hash"] != reference["candidate_hash"]:
                                raise ValueError("identical states/seeds produced different candidate pools")
                        else:
                            divergent_state_comparisons += 1
                            if evaluation_mode == "same_state":
                                raise ValueError("same-state diagnostic has divergent observable states")
                        if evaluation_mode == "same_state":
                            for key in ("executed_candidate", "executed_response_hash"):
                                if record[key] != reference[key]:
                                    raise ValueError("same-state diagnostic must share executed evidence")
    return {
        "status": "pass",
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_mode": evaluation_mode,
        "checked_slots": checked_slots,
        "checked_comparisons": checked_comparisons,
        "same_state_comparisons": same_state_comparisons,
        "divergent_state_comparisons": divergent_state_comparisons,
        "divergent_candidate_context_comparisons": divergent_candidate_context_comparisons,
        "max_contract_error": max_contract_error,
    }
