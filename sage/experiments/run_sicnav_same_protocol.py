"""Run TRACER and official CrowdSimPlus controls under one paired protocol.

The TRACER and official-policy runners remain separate because they own
different action-selection implementations.  This orchestrator gives both
the same external source, fixed test cases, episode budget, and reporting
schema, then computes case-matched contrasts without treating model seeds as
independent environment seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


TRACER_POLICIES = (
    "tracer-persistent",
    "tracer-hazard",
    "tracer-forgetting",
    "tracer-change-point",
    "tracer-reset",
)
OFFICIAL_POLICIES = (
    "official-sicnav-np",
    "official-sicnav-p",
    "official-mpc-cvmm",
    "official-dwa",
    "official-orca",
    "official-linear",
)
OFFICIAL_RUNNER_POLICIES = {
    "official-sicnav-np": "sicnav-np",
    "official-sicnav-p": "sicnav-p",
    "official-mpc-cvmm": "official-mpc-cvmm",
    "official-dwa": "official-dwa",
    "official-orca": "official-orca",
    "official-linear": "official-linear",
}
OFFICIAL_OUTPUT_POLICIES = {
    value: key for key, value in OFFICIAL_RUNNER_POLICIES.items()
}
COMMON_FIELDS = (
    "source",
    "policy",
    "seed",
    "case",
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
METRICS = (
    "success",
    "timeout",
    "collision",
    "wall_collision",
    "task_progress",
    "total_time",
    "min_distance",
)


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    status = run("status", "--porcelain").splitlines()
    # A clean audit worktree may mount the immutable ignored external checkout
    # as a symlink; that mount is not part of the experiment code state.
    unexpected_status = [line for line in status if line != "?? raw_data/external"]
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(unexpected_status),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_for_metadata(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _bootstrap(values: list[float], seed: int, samples: int = 4000) -> dict[str, Any]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "count": 0}
    rng = np.random.default_rng(seed)
    draws = finite[rng.integers(0, finite.size, size=(samples, finite.size))].mean(axis=1)
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "count": int(finite.size),
    }


def _normalise(rows: list[dict[str, str]], source: str) -> list[dict[str, Any]]:
    normalised = []
    for row in rows:
        item: dict[str, Any] = {field: row.get(field, "nan") for field in COMMON_FIELDS}
        item["source"] = source
        if source == "official":
            item["policy"] = OFFICIAL_OUTPUT_POLICIES.get(item["policy"], item["policy"])
        item["seed"] = row.get("seed", "official")
        item["case"] = int(row["case"])
        item["steps"] = int(row["steps"])
        for field in COMMON_FIELDS:
            if field not in {"source", "policy", "seed", "case", "steps"}:
                item[field] = _float(item[field])
        normalised.append(item)
    return normalised


def _summarise(rows: list[dict[str, Any]], policies: tuple[str, ...], seed_offset: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for policy_index, policy in enumerate(policies):
        selected = [row for row in rows if row["policy"] == policy]
        result[policy] = {
            metric: _bootstrap(
                [_float(row[metric]) for row in selected],
                seed_offset + policy_index * 31 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    return result


def _case_means(rows: list[dict[str, Any]], policy: str, metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row["policy"] != policy:
            continue
        value = _float(row[metric])
        if math.isfinite(value):
            grouped.setdefault(int(row["case"]), []).append(value)
    return {case: float(np.mean(values)) for case, values in grouped.items() if values}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd().resolve()
    git = _git_state(root)
    if git["dirty"]:
        raise RuntimeError("same-protocol benchmark requires a clean main-repository worktree")
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tracer_output = output / "tracer"
    official_output = output / "official"
    python = str(Path(sys.executable).resolve())
    started = time.perf_counter()
    tracer_command = [
        python,
        "-u",
        "-m",
        "sage.experiments.run_sicnav_tracer_loop",
        "--external-root",
        args.external_root,
        "--env-config",
        args.env_config,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--output-dir",
        str(tracer_output),
        "--device",
        args.device,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--cases",
        str(args.cases),
        "--case-start",
        str(args.case_start),
        "--max-steps",
        str(args.max_steps),
        "--candidate-profile",
        args.candidate_profile,
        "--fallback-profile",
        args.fallback_profile,
        "--windows",
        str(args.windows),
        "--execute-steps",
        str(args.execute_steps),
        "--num-candidates",
        str(args.num_candidates),
        "--human-cost-weight",
        str(args.human_cost_weight),
        "--safety-weight",
        str(args.safety_weight),
        "--likelihood-scale",
        str(args.likelihood_scale),
        "--collision-margin",
        str(args.collision_margin),
        "--kinematic-safety-margin",
        str(args.kinematic_safety_margin),
        "--static-safety-margin",
        str(args.static_safety_margin),
        "--geometric-safety-prefix-steps",
        str(args.geometric_safety_prefix_steps),
        "--candidate-seed",
        str(args.candidate_seed),
        "--policies",
        *TRACER_POLICIES,
    ]
    official_command = [
        python,
        "-u",
        "-m",
        "sage.experiments.run_sicnav_official_benchmark",
        "--external-root",
        args.external_root,
        "--env-config",
        args.env_config,
        "--policy-config",
        args.policy_config,
        "--output-dir",
        str(official_output),
        "--cases",
        str(args.cases),
        "--max-steps",
        str(args.max_steps),
        "--case-start",
        str(args.case_start),
        "--policies",
        *[OFFICIAL_RUNNER_POLICIES[policy] for policy in OFFICIAL_POLICIES],
    ]
    _run(tracer_command, output / "tracer.log")
    _run(official_command, output / "official.log")

    rows = _normalise(_read_rows(tracer_output / "episode_metrics.csv"), "tracer")
    rows.extend(_normalise(_read_rows(official_output / "episode_metrics.csv"), "official"))
    if not rows:
        raise RuntimeError("same-protocol benchmark produced no rows")
    metrics_path = output / "episode_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMMON_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    all_policies = TRACER_POLICIES + OFFICIAL_POLICIES
    summary: dict[str, Any] = {
        "records": len(rows),
        "tracer_records": sum(row["source"] == "tracer" for row in rows),
        "official_records": sum(row["source"] == "official" for row in rows),
        "tracer": _summarise([row for row in rows if row["source"] == "tracer"], TRACER_POLICIES, args.bootstrap_seed),
        "official": _summarise([row for row in rows if row["source"] == "official"], OFFICIAL_POLICIES, args.bootstrap_seed + 1000),
        "paired_case_contrasts": {},
    }
    tracer_rows = [row for row in rows if row["source"] == "tracer"]
    official_rows = [row for row in rows if row["source"] == "official"]
    persistent_by_case = _case_means(tracer_rows, "tracer-persistent", "task_progress")
    for index, policy in enumerate(OFFICIAL_POLICIES):
        baseline_by_case = _case_means(official_rows, policy, "task_progress")
        keys = sorted(persistent_by_case.keys() & baseline_by_case.keys())
        delta = [baseline_by_case[key] - persistent_by_case[key] for key in keys]
        summary["paired_case_contrasts"][f"{policy}-persistent:task_progress"] = _bootstrap(
            delta, args.bootstrap_seed + 2000 + index
        )

    external_root = (root / args.external_root).resolve()
    external_commit = subprocess.check_output(
        ["git", "-C", str(external_root), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata = {
        "protocol": "tracer-official-crowdsimplus-same-case-protocol-v1",
        "claim_boundary": "TRACER and official CrowdSimPlus controls use the same fixed cases, external simulator, and episode budget; controller-specific action selection remains distinct and this is not a multi-robot or real-human result",
        "git": git,
        "external_repository": {
            "path": _path_for_metadata(external_root, root),
            "commit": external_commit,
        },
        "parameters": {
            "external_root": args.external_root,
            "env_config": args.env_config,
            "policy_config": args.policy_config,
            "checkpoint_dir": args.checkpoint_dir,
            "seeds": list(args.seeds),
            "cases": args.cases,
            "case_start": args.case_start,
            "max_steps": args.max_steps,
            "windows": args.windows,
            "execute_steps": args.execute_steps,
            "num_candidates": args.num_candidates,
            "human_cost_weight": args.human_cost_weight,
            "safety_weight": args.safety_weight,
            "likelihood_scale": args.likelihood_scale,
            "collision_margin": args.collision_margin,
            "kinematic_safety_margin": args.kinematic_safety_margin,
            "static_safety_margin": args.static_safety_margin,
            "geometric_safety_prefix_steps": args.geometric_safety_prefix_steps,
            "candidate_profile": args.candidate_profile,
            "fallback_profile": args.fallback_profile,
            "device": args.device,
            "tracer_policies": list(TRACER_POLICIES),
            "official_policies": list(OFFICIAL_POLICIES),
        },
        "records": len(rows),
        "runtime_seconds": time.perf_counter() - started,
        "source_artifacts": {
            "tracer": {
                "metadata_sha256": _sha256(tracer_output / "metadata.json"),
                "metrics_sha256": _sha256(tracer_output / "episode_metrics.csv"),
            },
            "official": {
                "metadata_sha256": _sha256(official_output / "metadata.json"),
                "metrics_sha256": _sha256(official_output / "episode_metrics.csv"),
            },
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_lines = [
        "# TRACER and official CrowdSimPlus same-case protocol",
        "",
        f"Main commit `{git['commit']}`; clean `{not git['dirty']}`; external source `{external_commit}`.",
        f"Fixed cases: `{args.case_start}--{args.case_start + args.cases - 1}`; episode budget: `{args.max_steps}` simulator steps; records: `{len(rows)}` (`{summary['tracer_records']}` TRACER and `{summary['official_records']}` official controls).",
        "",
        "Both runs use the same CrowdSimPlus test cases and time budget. TRACER uses its response-belief-planning controller; official policies use their own action selection. The comparison is therefore a same-environment, same-case controller comparison, not a claim that every controller evaluates the same candidate lattice.",
        "",
        "## Results",
        "",
        "| Source | Policy | N | Success | Collision | Progress | Min distance |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for source, policies in (("tracer", TRACER_POLICIES), ("official", OFFICIAL_POLICIES)):
        for policy in policies:
            item = summary[source][policy]
            report_lines.append(
                f"| {source} | `{policy}` | {item['success']['count']} | {item['success']['mean']:.3f} | {item['collision']['mean']:.3f} | {item['task_progress']['mean']:.3f} | {item['min_distance']['mean']:.3f} |"
            )
    report_lines.extend(["", "## Case-matched contrasts", "", "A positive task-progress contrast means the official policy progressed farther than the mean TRACER Persistent result on the same case.", ""])
    for name, item in summary["paired_case_contrasts"].items():
        report_lines.append(
            f"- `{name}`: `{item['mean']:.4f}` with bootstrap CI `[{item['ci95_low']:.4f}, {item['ci95_high']:.4f}]`, case count `{item['count']}`."
        )
    report_lines.extend([
        "",
        "## Boundary",
        "",
        "This closes the executable official SICNav-np/SICNav-p/MPC-CVMM/DWA/ORCA/Linear same-case comparison on CrowdSimPlus. It does not make official LUCIDGames or MPOGames available, does not replace their missing solver integrations, and does not support real-human causality or multi-robot superiority claims.",
        "",
    ])
    report_path = output / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    manifest = {
        "status": "COMPLETED",
        "protocol": metadata["protocol"],
        "git": git,
        "external_repository": metadata["external_repository"],
        "records": len(rows),
        "artifacts": {
            name: _sha256(output / name)
            for name in ("episode_metrics.csv", "metadata.json", "summary.json", "report.md")
        },
        "claim_boundary": metadata["claim_boundary"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"metadata": metadata, "summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/sicnav_same_protocol_20260904")
    parser.add_argument("--external-root", default="raw_data/external/source_repos/sicnav_official_filtered")
    parser.add_argument("--env-config", default="raw_data/external/source_repos/sicnav_official_filtered/sicnav/configs/env.config")
    parser.add_argument("--policy-config", default="raw_data/external/source_repos/sicnav_official_filtered/sicnav/configs/policy.config")
    parser.add_argument("--checkpoint-dir", default="experiments/independent_multientity_suite_20260902_final/checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2031, 2039, 2053, 2063])
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument(
        "--case-start",
        type=int,
        default=0,
        help="First official test-case index; use disjoint ranges for validation and holdout.",
    )
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument(
        "--candidate-profile",
        choices=("goal-random", "orca-augmented"),
        default="goal-random",
        help="TRACER candidate support profile passed to the paired runner.",
    )
    parser.add_argument(
        "--fallback-profile",
        choices=("least-risk", "orca-proposal"),
        default="least-risk",
        help="TRACER fallback profile passed to the paired runner.",
    )
    parser.add_argument("--windows", type=int, default=30)
    parser.add_argument("--execute-steps", type=int, default=2)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--human-cost-weight", type=float, default=0.6)
    parser.add_argument("--safety-weight", type=float, default=2.0)
    parser.add_argument("--likelihood-scale", type=float, default=0.05)
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument("--kinematic-safety-margin", type=float, default=-1.0)
    parser.add_argument("--static-safety-margin", type=float, default=-1.0)
    parser.add_argument("--geometric-safety-prefix-steps", type=int, default=0)
    parser.add_argument("--candidate-seed", type=int, default=839731)
    parser.add_argument("--bootstrap-seed", type=int, default=849731)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["manifest"]["status"], "records": result["manifest"]["records"]}, indent=2))


if __name__ == "__main__":
    main()
