"""Select a SocialGym2 belief scale from held-out factual evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL = "tracer-socialgym2-joint-belief-synchronous-v5"
CALIBRATION_PROTOCOL = "tracer-socialgym2-joint-belief-calibration-v1"
DEFAULT_SCALES = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _load_groups(inputs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for path in inputs:
        metadata_path = path / "metadata.json"
        evidence_path = path / "belief_evidence.csv"
        if not metadata_path.is_file() or not evidence_path.is_file():
            raise FileNotFoundError(f"missing calibration artifact input: {path}")
        source = json.loads(metadata_path.read_text(encoding="utf-8"))
        if source.get("protocol") != PROTOCOL:
            raise ValueError(f"unsupported source protocol in {path}")
        if (source.get("runtime_contract") or {}).get("protocol") != "socialgym2-joint-synchronous-runtime-v5":
            raise ValueError(f"calibration input {path} lacks the frozen runtime v5 contract")
        if not source.get("algorithm_sources") or not source.get("checkpoint_hashes"):
            raise ValueError(f"calibration input {path} lacks source/checkpoint provenance")
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETED":
            raise ValueError(f"calibration input {path} is incomplete")
        for name in ("metadata.json", "belief_evidence.csv"):
            if manifest.get("artifacts", {}).get(name) != _sha256(path / name):
                raise ValueError(f"calibration input hash mismatch: {path / name}")
        parameters = source.get("parameters", {})
        if parameters.get("num_agents") != 3 or parameters.get("num_humans") != 2:
            raise ValueError(f"calibration input {path} is not the three-robot/two-human protocol")
        clock = source.get("clock_contract", {})
        if (
            clock.get("simulator_steps_per_planning_window") != 16
            or not math.isclose(float(clock.get("simulator_delta_t", 0)), 0.025)
            or not math.isclose(float(clock.get("planning_window_dt", 0)), 0.4)
            or clock.get("observed_transition_used_for_belief_update") is not True
        ):
            raise ValueError(f"calibration input {path} has an incompatible simulator clock")
        with evidence_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("policy") != "tracer-uniform":
                    continue
                row = dict(row)
                likelihood = np.asarray(json.loads(row["factual_mode_log_likelihood"]), dtype=np.float64)
                decision = np.asarray(json.loads(row["decision_log_belief"]), dtype=np.float64)
                posterior = np.asarray(json.loads(row["posterior_log_belief"]), dtype=np.float64)
                if likelihood.shape != (3,) or decision.shape != (3,) or posterior.shape != (3,):
                    raise ValueError(f"invalid evidence shape in {path}: {row}")
                if not all(np.isfinite(values).all() for values in (likelihood, decision, posterior)):
                    raise ValueError(f"non-finite evidence in {path}: {row}")
                if not np.allclose(decision, -math.log(3.0), atol=1e-5):
                    raise ValueError("Uniform calibration evidence must start from a uniform decision belief")
                if not np.allclose(posterior, -math.log(3.0), atol=1e-5):
                    raise ValueError("Uniform calibration evidence must not update its posterior")
                row["likelihood"] = likelihood
                row["decision"] = decision
                rows.append(row)
        metadata.append(source)
    if not rows:
        raise ValueError("no Uniform factual evidence was found")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["scenario_hash"], row["initial_state_hash"], row["track_id"])
        groups.setdefault(key, []).append(row)
    ordered = []
    for key, group in groups.items():
        if len({row["step"] for row in group}) != len(group):
            raise ValueError(f"duplicate calibration decisions for track {key}")
        ordered.append({"key": key, "rows": sorted(group, key=lambda row: int(row["step"]))})
    return ordered, metadata


def _score(groups: list[dict[str, Any]], scale: float) -> tuple[float, int]:
    total = 0.0
    count = 0
    for group in groups:
        prior = np.full(3, -math.log(3.0), dtype=np.float64)
        for row in group["rows"]:
            likelihood = row["likelihood"]
            total -= _logsumexp(prior + likelihood)
            posterior = prior + scale * likelihood
            posterior -= _logsumexp(posterior)
            prior = posterior
            count += 1
    return total / max(count, 1), count


def calibrate(inputs: list[Path], output: Path, scales: tuple[float, ...] = DEFAULT_SCALES) -> dict[str, Any]:
    if not scales or any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("calibration scales must be positive and finite")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration artifact: {output}")
    groups, metadata = _load_groups(inputs)
    checkpoints = {source.get("parameters", {}).get("response_checkpoint") for source in metadata}
    if len(checkpoints) != 1:
        raise ValueError("calibration inputs must use the same response checkpoint")
    for source in metadata[1:]:
        for field in ("runtime_contract", "algorithm_sources", "checkpoint_hashes"):
            if source[field] != metadata[0][field]:
                raise ValueError(f"calibration inputs have different {field}")
    scores = []
    for scale in scales:
        score, count = _score(groups, float(scale))
        scores.append({"likelihood_scale": float(scale), "decision_mixture_nll": score, "decisions": count})
    selected = min(scores, key=lambda item: (item["decision_mixture_nll"], item["likelihood_scale"]))
    source_checkpoint = metadata[0].get("parameters", {}).get("response_checkpoint")
    if not source_checkpoint:
        raise ValueError("calibration input does not record a response checkpoint")
    checkpoint_path = Path(source_checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    if _sha256(checkpoint_path) != metadata[0]["checkpoint_hashes"]["response"]:
        raise ValueError("calibration response checkpoint changed after evidence collection")
    artifact = {
        "protocol": CALIBRATION_PROTOCOL,
        "phase": "validation",
        "split": "validation",
        "selection_metric": "unscaled_factual_mixture_nll_before_update",
        "likelihood_scale_applies_to": "posterior_update_only_not_evaluation_density",
        "hidden_mode_labels_used": False,
        "source_protocol": PROTOCOL,
        "runtime_contract": metadata[0]["runtime_contract"],
        "algorithm_sources": metadata[0]["algorithm_sources"],
        "planning_horizon_windows": metadata[0]["parameters"]["max_steps"],
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "sim_dt": 0.025,
        "sim_steps_per_planning_window": 16,
        "planning_window_dt": 0.4,
        "num_agents": 3,
        "num_humans": 2,
        "input_artifacts": [
            {
                "path": str(path),
                "metadata_sha256": _sha256(path / "metadata.json"),
                "evidence_sha256": _sha256(path / "belief_evidence.csv"),
            }
            for path in inputs
        ],
        "scene_count": len({(group["key"][0], group["key"][1]) for group in groups}),
        "validation_seeds": sorted({source["parameters"]["seed"] for source in metadata}),
        "validation_scene_hashes": sorted({group["key"][1] for group in groups}),
        "uniform_decision_mixture_nll": _score(groups, 0.0)[0],
        "track_group_count": len(groups),
        "scores": scores,
        "selected": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    result = calibrate(args.inputs, args.output)
    print(json.dumps({"selected": result["selected"], "scene_count": result["scene_count"], "track_group_count": result["track_group_count"]}, indent=2))


if __name__ == "__main__":
    main()
