"""Audit external baseline sources, runtime endpoints, and saved evidence.

The audit is intentionally read-only with respect to external checkouts.  It
does not install ROS, Docker, Julia packages, or modify any source repository;
it records the exact boundary between a completed benchmark and an unavailable
runtime so the TRACER paper cannot silently overstate external coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_EXTERNAL_CSV_FIELDS = {
    "crowdnav": (
        "realized_total_cost",
        "success",
        "collision",
        "timeout",
        "total_time",
        "task_progress",
        "human_cost",
        "min_distance",
    ),
    "sicnav": (
        "success",
        "collision",
        "timeout",
        "total_time",
        "task_progress",
        "min_distance",
        "solver_time_mean",
    ),
    "atom_headless": (
        "solver_success",
        "converged",
        "finite",
        "iterates",
        "initial_min_distance",
        "final_min_distance",
        "final_goal_error",
        "final_state_norm",
    ),
}

REQUIRED_ALGAMES_CSV_FIELDS = ("converged", "iterations", "seconds")

REQUIRED_SOCIALGYM2_TRACER_FIELDS = (
    "success",
    "collision",
    "task_progress",
    "mean_reward",
    "min_distance",
    "fallback_rate",
    "final_belief_entropy",
    "action_change_rate",
)

REQUIRED_SICNAV_SAME_PROTOCOL_FIELDS = (
    "steps",
    "success",
    "timeout",
    "collision",
    "wall_collision",
    "task_progress",
    "total_time",
    "min_distance",
    "wall_time_seconds",
)

# These fields are defined for every controller in the local multi-entity
# evaluator.  Belief-only fields are intentionally excluded because ORCA/MPC
# controls do not maintain a TRACER belief and therefore emit NaN there.
REQUIRED_LOCAL_CSV_FIELDS = (
    "realized_total_cost",
    "task_progress",
    "collision_rate",
    "hard_safe_fraction",
    "hard_safety_fallback",
    "hard_safety_violation_without_fallback",
    "obstacle_collision",
    "robot_collision",
    "hard_safety_violation",
)


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source(root: Path, relative: str, role: str) -> dict[str, Any]:
    path = root / relative
    exists = path.is_dir()
    return {
        "role": role,
        "path": relative,
        "exists": exists,
        "commit": _git(path, "rev-parse", "HEAD") if exists else None,
        "dirty": (
            _git(path, "diff", "--quiet") is None
            if exists and _git(path, "rev-parse", "HEAD")
            else None
        ),
    }


def _submodules(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"entries": [], "initialized": False, "pinned": False}
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "submodule", "status"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return result
    entries = []
    for line in output.splitlines():
        if not line:
            continue
        marker = line[0]
        fields = line[1:].split()
        subpath = fields[1] if len(fields) > 1 else None
        subdir = path / subpath if subpath else None
        head = None
        if subdir and (subdir / ".git").exists():
            top = _git(subdir, "rev-parse", "--show-toplevel")
            if top and Path(top).resolve() == subdir.resolve():
                head = _git(subdir, "rev-parse", "HEAD")
        entries.append({"marker": marker, "commit": fields[0] if fields else None, "path": subpath, "head": head})
    result["entries"] = entries
    result["initialized"] = bool(entries) and all(item["marker"] != "-" and item["head"] for item in entries)
    result["pinned"] = bool(entries) and all(item["marker"] == " " and item["head"] == item["commit"] for item in entries)
    return result


def _finite_csv(path: Path, fields: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "rows": 0,
        "missing_fields": [],
        "nonfinite_cells": 0,
        "complete": False,
        "sha256": None,
    }
    if not path.is_file():
        return result

    result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        missing = sorted(set(fields) - header)
        result["missing_fields"] = missing
        for row in reader:
            result["rows"] += 1
            for field in fields:
                try:
                    value = float(row[field])
                except (KeyError, TypeError, ValueError):
                    result["nonfinite_cells"] += 1
                    continue
                if not math.isfinite(value):
                    result["nonfinite_cells"] += 1
    result["complete"] = bool(
        result["rows"] > 0
        and not result["missing_fields"]
        and result["nonfinite_cells"] == 0
    )
    return result


def _lucidgames_package_smoke(root: Path) -> dict[str, Any]:
    output_dir = root / "experiments/lucidgames_package_smoke_20260903"
    metadata_path = output_dir / "metadata.txt"
    status_path = output_dir / "status.txt"
    result: dict[str, Any] = {
        "path": str(output_dir),
        "exists": metadata_path.is_file() and status_path.is_file(),
        "module_loaded": False,
        "direct_games_solver_export": False,
        "complete": False,
        "sha256": None,
    }
    if not result["exists"]:
        return result
    metadata = metadata_path.read_text(encoding="utf-8")
    status = status_path.read_text(encoding="utf-8")
    result["module_loaded"] = "module_loaded=true" in metadata
    result["direct_games_solver_export"] = "has_direct_games_solver_export=true" in metadata
    result["complete"] = result["module_loaded"] and "exit_code=0" in status
    result["sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    return result


def _main_git(root: Path) -> dict[str, Any]:
    commit = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain") or ""
    return {"commit": commit, "dirty": bool(dirty), "status_lines": dirty.splitlines()}


def build_audit(root: Path) -> dict[str, Any]:
    socialgym2_path = root / "raw_data/external/source_repos/socialgym2_official"
    sources = {
        "crowdnav": _source(
            root,
            "raw_data/external/source_repos/crowdnav_official",
            "official CrowdNav simulator",
        ),
        "python_rvo2": _source(
            root,
            "raw_data/external/source_repos/python_rvo2_official",
            "official Python-RVO2 human policy",
        ),
        "sicnav": _source(
            root,
            "raw_data/external/source_repos/sicnav_official_filtered",
            "official SICNav implementation",
        ),
        "socialgym2": _source(
            root,
            "raw_data/external/source_repos/socialgym2_official",
            "official SocialGym2 simulator",
        ),
        "atom": _source(
            root,
            "raw_data/external/source_repos/atom_official",
            "official AToM implementation",
        ),
        "lucidgames": _source(
            root,
            "raw_data/external/source_repos/lucidgames_official",
            "official LUCIDGames implementation",
        ),
        "mpogames": {
            "role": "MPOGames paper baseline",
            "path": "raw_data/external/source_repos/mpogames_official",
            "exists": False,
            "commit": None,
            "dirty": None,
            "note": "No official source checkout is present in the workspace.",
        },
    }
    sources["socialgym2"]["submodules"] = _submodules(socialgym2_path)

    def available(command: str) -> bool:
        candidates = [
            Path(shutil.which(command)) if shutil.which(command) else None,
            Path(sys.executable).parent / command,
        ]
        local_tools = {
            "docker": root / ".tools/docker-static/usr/bin/docker",
            "julia": root / ".conda/envs/lucidgames-julia16/bin/julia",
        }
        if command in local_tools:
            candidates.append(local_tools[command])
        return any(path is not None and path.is_file() for path in candidates)

    commands = {
        command: available(command)
        for command in ("docker", "roscore", "roslaunch", "rviz", "julia")
    }
    evidence = {
        "crowdnav": _finite_csv(
            root / "experiments/tracer_crowdnav_20260903_full/episode_metrics.csv",
            REQUIRED_EXTERNAL_CSV_FIELDS["crowdnav"],
        ),
        "sicnav": _finite_csv(
            root / "experiments/tracer_sicnav_official_20260903_v3/episode_metrics.csv",
            REQUIRED_EXTERNAL_CSV_FIELDS["sicnav"],
        ),
        "sicnav_dwa": _finite_csv(
            root / "experiments/tracer_sicnav_official_20260903_v4/episode_metrics.csv",
            REQUIRED_EXTERNAL_CSV_FIELDS["sicnav"],
        ),
        "sicnav_tracer_loop": _finite_csv(
            root / "experiments/tracer_sicnav_crowdsimplus_20260903_full/episode_metrics.csv",
            ("success", "collision", "timeout", "total_time", "task_progress", "min_distance", "fallback_rate", "final_belief_entropy"),
        ),
        "socialgym2_tracer_loop": _finite_csv(
            root / "experiments/socialgym2_tracer_20260904_v5_aggregate/episode_metrics.csv",
            REQUIRED_SOCIALGYM2_TRACER_FIELDS,
        ),
        "sicnav_same_protocol": _finite_csv(
            root / "experiments/sicnav_same_protocol_20260904/episode_metrics.csv",
            REQUIRED_SICNAV_SAME_PROTOCOL_FIELDS,
        ),
        "atom_headless": _finite_csv(
            root / "experiments/atom_headless_20260903/results.csv",
            REQUIRED_EXTERNAL_CSV_FIELDS["atom_headless"],
        ),
        "algames_headless": _finite_csv(
            root / "experiments/lucidgames_algames_headless_20260903/results.csv",
            REQUIRED_ALGAMES_CSV_FIELDS,
        ),
        "local_same_protocol": _finite_csv(
            root / "experiments/tracer_gap_closure_20260903_mpogames_final/episode_metrics.csv",
            REQUIRED_LOCAL_CSV_FIELDS,
        ),
    }
    crowdnav_expected_rows = 5 * 16 * 7
    evidence["crowdnav"]["expected_rows"] = crowdnav_expected_rows
    evidence["crowdnav"]["expected_row_count"] = (
        evidence["crowdnav"]["rows"] == crowdnav_expected_rows
    )
    evidence["crowdnav"]["complete"] = bool(
        evidence["crowdnav"]["complete"]
        and evidence["crowdnav"]["expected_row_count"]
    )
    sicnav_tracer_expected_rows = 5 * 16 * 5
    evidence["sicnav_tracer_loop"]["expected_rows"] = sicnav_tracer_expected_rows
    evidence["sicnav_tracer_loop"]["expected_row_count"] = (
        evidence["sicnav_tracer_loop"]["rows"] == sicnav_tracer_expected_rows
    )
    evidence["sicnav_tracer_loop"]["complete"] = bool(
        evidence["sicnav_tracer_loop"]["complete"]
        and evidence["sicnav_tracer_loop"]["expected_row_count"]
    )
    socialgym2_expected_rows = 5 * 8
    evidence["socialgym2_tracer_loop"]["expected_rows"] = socialgym2_expected_rows
    evidence["socialgym2_tracer_loop"]["expected_row_count"] = (
        evidence["socialgym2_tracer_loop"]["rows"] == socialgym2_expected_rows
    )
    evidence["socialgym2_tracer_loop"]["complete"] = bool(
        evidence["socialgym2_tracer_loop"]["complete"]
        and evidence["socialgym2_tracer_loop"]["expected_row_count"]
    )
    sicnav_same_protocol_expected_rows = 5 * 16 * 5 + 16 * 4
    evidence["sicnav_same_protocol"]["expected_rows"] = sicnav_same_protocol_expected_rows
    evidence["sicnav_same_protocol"]["expected_row_count"] = (
        evidence["sicnav_same_protocol"]["rows"] == sicnav_same_protocol_expected_rows
    )
    evidence["sicnav_same_protocol"]["complete"] = bool(
        evidence["sicnav_same_protocol"]["complete"]
        and evidence["sicnav_same_protocol"]["expected_row_count"]
    )
    local_expected_rows = 5 * 9 * 8 * 16
    evidence["local_same_protocol"]["expected_rows"] = local_expected_rows
    evidence["local_same_protocol"]["expected_row_count"] = (
        evidence["local_same_protocol"]["rows"] == local_expected_rows
    )
    evidence["local_same_protocol"]["complete"] = bool(
        evidence["local_same_protocol"]["complete"]
        and evidence["local_same_protocol"]["expected_row_count"]
    )
    algames_expected_rows = 4
    evidence["algames_headless"]["expected_rows"] = algames_expected_rows
    evidence["algames_headless"]["expected_row_count"] = (
        evidence["algames_headless"]["rows"] == algames_expected_rows
    )
    evidence["algames_headless"]["complete"] = bool(
        evidence["algames_headless"]["complete"]
        and evidence["algames_headless"]["expected_row_count"]
    )
    safety_path = root / "experiments/tracer_hard_safety_property_20260903/summary.json"
    if safety_path.is_file():
        safety_summary = json.loads(safety_path.read_text(encoding="utf-8"))
        evidence["hard_safety_property"] = {
            "path": str(safety_path),
            "exists": True,
            "rows": int(safety_summary.get("checked_pools", 0)),
            "property_holds": bool(safety_summary.get("property_holds", False)),
            "checked_pools": int(safety_summary.get("checked_pools", 0)),
            "checked_candidates": int(safety_summary.get("checked_candidates", 0)),
            "unsafe_selected_without_fallback": int(safety_summary.get("unsafe_selected_without_fallback", -1)),
            "safe_geometry_mismatches": int(safety_summary.get("safe_geometry_mismatches", -1)),
            "complete": bool(safety_summary.get("property_holds", False)),
            "sha256": hashlib.sha256(safety_path.read_bytes()).hexdigest(),
        }
    selection_path = root / "experiments/tracer_belief_cost_selection_20260903/manifest.json"
    if selection_path.is_file():
        selection_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_summary_path = selection_path.parent / "summary.json"
        selection_summary = json.loads(selection_summary_path.read_text(encoding="utf-8")) if selection_summary_path.is_file() else {}
        selection_metadata_path = selection_path.parent / "metadata.json"
        selection_metadata = json.loads(selection_metadata_path.read_text(encoding="utf-8")) if selection_metadata_path.is_file() else {}
        evidence["belief_cost_selection"] = {
            "path": str(selection_path),
            "exists": True,
            "rows": sum(int(item.get("realized_total_cost", {}).get("count", 0)) for item in selection_summary.get("test", {}).values()) + sum(
                int(item.get("validation_rows", 0)) for item in selection_summary.get("validation_candidates", [])
            ),
            "status": selection_manifest.get("status"),
            "validation_only": selection_manifest.get("selection_role") == "validation-only",
            "test_role": selection_manifest.get("test_role"),
            "data_seeds_disjoint": bool(selection_metadata.get("data_seeds_disjoint", False)),
            "complete": bool(selection_manifest.get("status") == "COMPLETED" and selection_manifest.get("selection_role") == "validation-only"),
            "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        }
    package_smoke = _lucidgames_package_smoke(root)

    return {
        "protocol": "tracer-external-baseline-audit-v1",
        "main_repository": _main_git(root),
        "runtime_commands": commands,
        "sources": sources,
        "evidence": evidence,
        "runtime_smoke": {"lucidgames_package": package_smoke},
        "status": {
            "crowdnav_tracer_loop": evidence["crowdnav"]["complete"],
            "sicnav_standalone_baseline": evidence["sicnav"]["complete"],
            "sicnav_dwa_baseline": evidence["sicnav_dwa"]["complete"],
            "sicnav_tracer_loop": evidence["sicnav_tracer_loop"]["complete"],
            "socialgym2_tracer_loop": evidence["socialgym2_tracer_loop"]["complete"],
            "socialgym2_headless_runtime_complete": evidence["socialgym2_tracer_loop"]["complete"] and sources["socialgym2"]["submodules"]["initialized"],
            "socialgym2_runtime_available": all(
                commands[name] for name in ("docker", "roscore", "roslaunch", "rviz")
            ),
            "socialgym2_submodules_initialized": sources["socialgym2"]["submodules"]["initialized"],
            "socialgym2_submodules_pinned": sources["socialgym2"]["submodules"]["pinned"],
            "atom_runtime_available": all(
                commands[name] for name in ("roscore", "roslaunch", "rviz")
            ),
            "atom_headless_core_complete": evidence["atom_headless"]["complete"],
            "algames_headless_core_complete": evidence["algames_headless"]["complete"],
            "same_protocol_local_baselines_complete": evidence["local_same_protocol"]["complete"],
            "same_protocol_official_crowdsimplus_complete": evidence["sicnav_same_protocol"]["complete"],
            "lucidgames_julia_available": commands["julia"],
            "lucidgames_runtime_available": False,
            "lucidgames_official_smoke_passed": package_smoke["complete"],
            "lucidgames_package_import_passed": package_smoke["complete"],
            "lucidgames_solver_exported": package_smoke["direct_games_solver_export"],
            "mpogames_source_available": False,
            "same_protocol_strong_baselines_complete": False,
            "hard_safety_property_complete": evidence.get("hard_safety_property", {}).get("complete", False),
            "belief_cost_selection_complete": evidence.get("belief_cost_selection", {}).get("complete", False),
        },
    }


def _markdown(audit: dict[str, Any]) -> str:
    runtime = audit["runtime_commands"]
    evidence = audit["evidence"]
    lines = [
        "# External baseline audit",
        "",
        "This is a read-only audit of external source checkouts and local evidence.",
        "It does not convert a standalone baseline into a same-protocol TRACER result.",
        "",
        f"Main commit: `{audit['main_repository']['commit']}`; dirty: `{audit['main_repository']['dirty']}`.",
        "",
        "## Runtime endpoints",
        "",
        "| Endpoint | Available |",
        "|---|---:|",
    ]
    for name, available in runtime.items():
        lines.append(f"| `{name}` | `{available}` |")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "| Benchmark | Rows | Complete finite CSV | SHA-256 |",
            "|---|---:|---:|---|",
        ]
    )
    for name, item in evidence.items():
        lines.append(
            f"| `{name}` | {item['rows']} | `{item['complete']}` | `{item['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- CrowdNav is a completed official single-robot TRACER loop.",
            "- SICNav is a completed official standalone CrowdSimPlus baseline; its MPC/DWA extensions are separately recorded.",
            "- The full TRACER response-belief-planning loop is also completed in official CrowdSimPlus; it is a single-robot execution/integration result, not a strong-baseline superiority comparison.",
            "- Official ORCA/MPC/DWA/Linear controls are now rerun in the same fixed-case CrowdSimPlus protocol as TRACER; this closes the executable official baseline comparison while preserving the negative TRACER superiority result.",
            "- SocialGym2 now has a completed 5-seed official ROS headless TRACER loop with three robots, two requested humans, partial observation, ID-free spatial tracking, and the official GoAlone/Halt action projection; it is an external integration result, not continuous H3 replication.",
            "- AToM cannot be counted as a ROS episode or a TRACER same-protocol result.",
            "- The AToM headless row is an official three-agent numerical-core audit; it is not a ROS episode or a TRACER same-protocol result.",
            "- The ALGAMES headless rows execute four official solver examples in an isolated Julia 1.6 environment; the legacy GUI dependency does not pass, so these are numerical-core evidence only. The top-level LUCIDGames package import smoke passes, but it exposes no DirectGamesSolver export.",
            "- The local same-protocol matrix is a completed 5-seed x 9-scenario x 8-policy x 16-episode evaluation, including MPOGames-style, ORCA-style, and MPC-style controls; these are explicitly local style baselines, not official literature reimplementations.",
            "- LUCIDGames/MPOGames are not integrated into the same TRACER protocol.",
            "- The finite-candidate property audit and validation-only belief-cost selection are separate local evidence tracks with explicit claim boundaries.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/external_baseline_audit_20260903")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(root)
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output_dir / "report.md").write_text(_markdown(audit))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
