from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml


SCHEMA = "sage-formal-run/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inside_project(path: Path, project_root: Path) -> Path:
    root = project_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path must stay inside project root: {path}")
    return resolved


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_pool_sha256(pool: Any) -> str:
    """Hash candidate tensors with shape and dtype metadata in a stable order."""

    digest = hashlib.sha256()
    for name in ("robot_current", "robot_candidates", "activities"):
        tensor = getattr(pool, name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"candidate pool field {name} must be a torch.Tensor")
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _git_snapshot(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        porcelain = run("status", "--porcelain")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"available": False, "commit": None, "branch": None, "dirty": None}
    return {
        "available": True,
        "commit": commit,
        "branch": branch,
        "dirty": bool(porcelain),
        "status_porcelain": porcelain.splitlines(),
    }


def _environment_snapshot(probe_cuda: bool = False) -> dict[str, Any]:
    cuda_probe_error = None
    cuda_probe_warnings: list[str] = []
    cuda_available: bool | None = None
    if probe_cuda:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                cuda_available = torch.cuda.is_available()
            except (RuntimeError, OSError) as error:
                cuda_available = False
                cuda_probe_error = repr(error)
        cuda_probe_warnings = [str(item.message) for item in captured]
    devices = []
    if cuda_available:
        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "created_at_utc": _utc_now(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "torch": {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if probe_cuda else None,
            "devices": devices,
            "probe_error": cuda_probe_error,
            "probe_skipped": not probe_cuda,
            "probe_warnings": cuda_probe_warnings,
        },
    }


def validate_formal_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = raw.get("formal_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("formal config must contain a formal_protocol mapping")
    required = {
        "version",
        "stage",
        "status",
        "gate_file",
        "independent_unit",
        "model_seed_role",
        "confirmatory_splits",
        "primary_contrasts",
        "exclusion_rules",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"formal_protocol is missing fields: {missing}")
    if protocol["model_seed_role"] != "nuisance_block":
        raise ValueError("formal model seeds must be declared nuisance_block")
    seeds = raw.get("suite", {}).get("seeds", [])
    if len(set(int(seed) for seed in seeds)) < 5:
        raise ValueError("Formal-v1 requires at least five distinct model seeds")
    return protocol


def require_formal_gate(raw: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    protocol = validate_formal_config(raw)
    gate_path = _inside_project(Path(str(protocol["gate_file"])), project_root)
    if not gate_path.is_file():
        raise RuntimeError(
            f"formal stage {protocol['stage']} is blocked; missing gate file {gate_path}"
        )
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"formal gate is not valid JSON: {gate_path}") from error
    if gate.get("status") != "passed":
        raise RuntimeError(f"formal gate has not passed: {gate_path}")
    return {"path": str(gate_path), "sha256": sha256_file(gate_path), "content": gate}


def _artifact_records(paths: Iterable[Path], project_root: Path) -> list[dict[str, Any]]:
    records = []
    for requested in paths:
        path = _inside_project(requested, project_root)
        record: dict[str, Any] = {
            "path": str(path.relative_to(project_root)),
            "exists": path.is_file(),
        }
        if path.is_file():
            record.update({"size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        records.append(record)
    return records


def prepare_formal_run(
    config_path: Path,
    output_dir: Path,
    project_root: Path,
    command: list[str],
    overrides: Mapping[str, Any] | None = None,
    input_artifacts: Iterable[Path] = (),
    enforce_gate: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve()
    config = _inside_project(config_path, root)
    output = _inside_project(output_dir, root)
    raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    protocol = validate_formal_config(raw)
    gate = require_formal_gate(raw, root) if enforce_gate else None
    artifacts = _artifact_records(input_artifacts, root)
    missing_artifacts = [record["path"] for record in artifacts if not record["exists"]]
    if enforce_gate and missing_artifacts:
        raise RuntimeError(f"formal input artifacts are missing: {missing_artifacts}")
    output.mkdir(parents=True, exist_ok=True)

    frozen = output / "config_frozen.yaml"
    config_bytes = config.read_bytes()
    if frozen.exists() and frozen.read_bytes() != config_bytes:
        raise RuntimeError(f"frozen config mismatch in existing run directory: {frozen}")
    if not frozen.exists():
        frozen.write_bytes(config_bytes)

    environment_path = output / "environment.json"
    if not environment_path.exists():
        requested_device = str((overrides or {}).get("device", ""))
        environment_path.write_text(
            json.dumps(
                _environment_snapshot(probe_cuda=requested_device.startswith("cuda")),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    deviation_path = output / "deviation_log.jsonl"
    deviation_path.touch(exist_ok=True)

    manifest_path = output / "manifest.json"
    config_hash = sha256_file(config)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config", {}).get("sha256") != config_hash:
            raise RuntimeError(f"manifest config hash mismatch: {manifest_path}")
        return existing

    manifest = {
        "schema": SCHEMA,
        "status": "STARTED" if enforce_gate else "PREPARED",
        "created_at_utc": _utc_now(),
        "stage": protocol["stage"],
        "formal_protocol": dict(protocol),
        "independent_unit": protocol["independent_unit"],
        "model_seed_role": protocol["model_seed_role"],
        "config": {
            "source": str(config.relative_to(root)),
            "frozen": str(frozen.relative_to(root)),
            "sha256": config_hash,
        },
        "command": command,
        "overrides": dict(overrides or {}),
        "gate": gate,
        "git": _git_snapshot(root),
        "environment": str(environment_path.relative_to(root)),
        "deviation_log": str(deviation_path.relative_to(root)),
        "input_artifacts": artifacts,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def mark_formal_run(output_dir: Path, status: str, project_root: Path) -> None:
    if status not in {"COMPLETED", "FAILED"}:
        raise ValueError("formal run status must be COMPLETED or FAILED")
    output = _inside_project(output_dir, project_root)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["finished_at_utc"] = _utc_now()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an immutable Formal-v1 run manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--input-artifact", action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    manifest = prepare_formal_run(
        Path(args.config),
        Path(args.output_dir),
        project_root,
        command=list(sys.argv),
        input_artifacts=[Path(path) for path in args.input_artifact],
        enforce_gate=not args.prepare_only,
    )
    print(json.dumps({"stage": manifest["stage"], "status": manifest["status"]}))


if __name__ == "__main__":
    main()
