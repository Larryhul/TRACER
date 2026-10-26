from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "collision_free_success",
    "success",
    "collision",
    "wall_collision",
    "timeout",
    "task_progress",
    "total_time",
    "min_distance",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_manifest(directory: Path) -> dict[str, Any]:
    manifest = read_json(directory / "manifest.json")
    if manifest.get("status") != "COMPLETED":
        raise ValueError(f"Incomplete run: {directory}")
    for name, expected in manifest["artifacts"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {directory / name}")
    return manifest


def bootstrap(values: list[float], *, seed: int, samples: int = 10_000) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Bootstrap requires finite case-level values")
    generator = np.random.default_rng(seed)
    draws = array[generator.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "case_count": int(len(array)),
        "case_bootstrap_ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
    }


def role(policy: str, kinematics: list[str]) -> str:
    if policy == "sicnav-p":
        return "privileged-information unicycle reference"
    if kinematics == ["holonomic"]:
        return "holonomic reference"
    return "unicycle controller"


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    root = root.resolve()
    protocol_path = protocol_path.resolve()
    protocol_copy = root / "protocol.json"
    if digest(protocol_copy) != digest(protocol_path):
        raise ValueError("Executed protocol copy differs from the frozen protocol")
    protocol = read_json(protocol_path)
    if protocol["status_at_freeze"] != "PREREGISTERED_BEFORE_EXECUTION":
        raise ValueError("Protocol is not marked preregistered")

    tracer_dir = root / "tracer"
    official_dir = root / "official"
    tracer_manifest = verify_manifest(tracer_dir)
    official_manifest = verify_manifest(official_dir)
    tracer_meta = read_json(tracer_dir / "metadata.json")
    official_meta = read_json(official_dir / "metadata.json")

    cases = tuple(range(protocol["cases"]["start"], protocol["cases"]["start"] + protocol["cases"]["count"]))
    seeds = tuple(protocol["tracer"]["model_seeds"])
    tracer_policy = protocol["tracer"]["policy"]
    official_policies = tuple(protocol["official_baselines"]["policies"])
    if tracer_meta["external_repository"]["commit"] != protocol["external_repository_commit"]:
        raise ValueError("TRACER simulator commit differs from protocol")
    if official_meta["external_repository"]["commit"] != protocol["external_repository_commit"]:
        raise ValueError("Official simulator commit differs from protocol")
    if tracer_meta["source_hashes"]["experiments/run_sicnav_tracer_loop.py"] != protocol["tracer"]["runner_sha256"]:
        raise ValueError("TRACER runner differs from frozen source")
    official_runner = official_meta["source_hashes"]["sage/experiments/run_sicnav_official_benchmark.py"]
    if official_runner != protocol["official_baselines"]["runner_sha256"]:
        raise ValueError("Official runner differs from frozen source")
    if int(tracer_meta["parameters"]["case_start"]) != cases[0] or int(tracer_meta["parameters"]["cases"]) != len(cases):
        raise ValueError("TRACER case range differs from protocol")
    if tuple(int(value) for value in official_meta["parameters"]["cases"]) != cases:
        raise ValueError("Official case range differs from protocol")

    tracer_rows = read_rows(tracer_dir / "episode_metrics.csv")
    official_rows = read_rows(official_dir / "episode_metrics.csv")
    tracer = {(row["policy"], int(row["seed"]), int(row["case"])): row for row in tracer_rows}
    official = {(row["policy"], int(row["case"])): row for row in official_rows}
    expected_tracer = {(tracer_policy, seed, case) for seed in seeds for case in cases}
    expected_official = {(policy, case) for policy in official_policies for case in cases}
    if set(tracer) != expected_tracer:
        raise ValueError("Missing or unexpected TRACER rows")
    if set(official) != expected_official:
        raise ValueError("Missing or unexpected official rows")

    for case in cases:
        reference = official[official_policies[0], case]
        rows = [official[policy, case] for policy in official_policies]
        rows.extend(tracer[tracer_policy, seed, case] for seed in seeds)
        for row in rows:
            if row["initial_scene_hash"] != reference["initial_scene_hash"]:
                raise ValueError(f"Initial scene mismatch for case {case}: {row['policy']}")
            if not math.isclose(float(row["environment_dt"]), float(reference["environment_dt"]), abs_tol=1e-12):
                raise ValueError(f"Environment clock mismatch for case {case}: {row['policy']}")
            if row["robot_kinematics"] == reference["robot_kinematics"] and not math.isclose(
                float(row["robot_initial_heading"]), float(reference["robot_initial_heading"]), abs_tol=1e-12
            ):
                raise ValueError(f"Initial heading mismatch for case {case}: {row['policy']}")
            for metric in METRICS:
                if not math.isfinite(float(row[metric])):
                    raise ValueError(f"Nonfinite {metric} for case {case}: {row['policy']}")

    scene_values: dict[str, dict[str, list[float]]] = {}
    policies: dict[str, Any] = {}
    for policy in (tracer_policy, *official_policies):
        values = {
            metric: [
                float(np.mean([float(tracer[policy, seed, case][metric]) for seed in seeds]))
                if policy == tracer_policy
                else float(official[policy, case][metric])
                for case in cases
            ]
            for metric in METRICS
        }
        scene_values[policy] = values
        source_rows = tracer_rows if policy == tracer_policy else official_rows
        selected = [row for row in source_rows if row["policy"] == policy]
        kinematics = sorted({row["robot_kinematics"] for row in selected})
        policies[policy] = {
            "records": len(selected),
            "independent_cases": len(cases),
            "robot_kinematics": kinematics,
            "comparison_role": role(policy, kinematics),
            "metrics": {
                metric: bootstrap(metric_values, seed=849731 + index)
                for index, (metric, metric_values) in enumerate(values.items())
            },
            "per_model_seed": {
                str(seed): {
                    metric: float(np.mean([float(tracer[policy, seed, case][metric]) for case in cases]))
                    for metric in METRICS
                }
                for seed in seeds
            }
            if policy == tracer_policy
            else {},
        }

    contrasts = {
        baseline: {
            metric: bootstrap(
                list(np.asarray(scene_values[tracer_policy][metric]) - np.asarray(scene_values[baseline][metric])),
                seed=859731 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
        for baseline in official_policies
    }
    summary = {
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "source-frozen unseen-case confirmation on cases 144-159; not pooled with development cases",
        "claim_boundary": protocol["claim_boundary"],
        "protocol_sha256": digest(protocol_path),
        "tracer_records": len(tracer_rows),
        "official_records": len(official_rows),
        "independent_cases": len(cases),
        "model_seeds": list(seeds),
        "scene_hash_clock_and_heading_pairing": "PASS",
        "uncertainty": "Paired case bootstrap with 10000 resamples after averaging TRACER model seeds within case",
        "policies": policies,
        "persistent_minus_official": contrasts,
        "input_hashes": {
            str(protocol_path): digest(protocol_path),
            str(tracer_dir / "manifest.json"): digest(tracer_dir / "manifest.json"),
            str(official_dir / "manifest.json"): digest(official_dir / "manifest.json"),
            str(tracer_dir / "episode_metrics.csv"): digest(tracer_dir / "episode_metrics.csv"),
            str(official_dir / "episode_metrics.csv"): digest(official_dir / "episode_metrics.csv"),
        },
    }
    write_json(root / "summary.json", summary)

    columns = ["source", "policy", "seed", "case", "initial_scene_hash", "robot_kinematics", "environment_dt", *METRICS]
    with (root / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for source, rows in (("tracer", tracer_rows), ("official", official_rows)):
            for row in rows:
                writer.writerow({field: source if field == "source" else row.get(field, "") for field in columns})

    lines = [
        "# CrowdSimPlus source-frozen unseen-case confirmation",
        "",
        "Cases 144-159 were frozen before execution and are disjoint from response train (112-135), validation (136-143), and prior development evaluation (88-111).",
        "TRACER uses the selected September 8 controller and frozen source. Three model seeds are averaged within each of 16 independent cases.",
        "",
        "## Results",
        "",
        "| Policy | Role | Collision-free success | Success | Collision | Timeout | Progress | Time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, values in policies.items():
        metrics = values["metrics"]
        lines.append(
            f"| {policy} | {values['comparison_role']} | {metrics['collision_free_success']['mean']:.2%} | "
            f"{metrics['success']['mean']:.2%} | {metrics['collision']['mean']:.2%} | {metrics['timeout']['mean']:.2%} | "
            f"{metrics['task_progress']['mean']:.6f} | {metrics['total_time']['mean']:.6f} |"
        )
    lines += [
        "",
        "## Persistent minus official",
        "",
        "| Baseline | Collision-free success difference (pp), 95% CI | Collision difference (pp), 95% CI | Progress difference, 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for baseline, metrics in contrasts.items():
        collision_free = metrics["collision_free_success"]
        collision = metrics["collision"]
        progress = metrics["task_progress"]
        lines.append(
            f"| {baseline} | {100 * collision_free['mean']:+.2f} "
            f"[{100 * collision_free['case_bootstrap_ci95'][0]:+.2f}, {100 * collision_free['case_bootstrap_ci95'][1]:+.2f}] | "
            f"{100 * collision['mean']:+.2f} [{100 * collision['case_bootstrap_ci95'][0]:+.2f}, {100 * collision['case_bootstrap_ci95'][1]:+.2f}] | "
            f"{progress['mean']:+.4f} [{progress['case_bootstrap_ci95'][0]:+.4f}, {progress['case_bootstrap_ci95'][1]:+.4f}] |"
        )
    lines += [
        "",
        "## Boundaries",
        "",
        "- This is a source-frozen unseen-case confirmation, not a clean-git or real-human blind test.",
        "- SICNav-P has privileged information; ORCA and Linear retain native holonomic kinematics.",
        "- Candidate sets differ, so this is a same-scene and same-budget comparison, not action-space equivalence.",
        "- CrowdSimPlus is single-robot and does not establish multi-robot generalization.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        root / "manifest.json",
        {
            "status": "COMPLETED",
            "protocol_sha256": digest(protocol_path),
            "input_manifests": {
                "tracer": digest(tracer_dir / "manifest.json"),
                "official": digest(official_dir / "manifest.json"),
            },
            "artifacts": {
                name: digest(root / name)
                for name in ("protocol.json", "summary.json", "report.md", "episode_metrics.csv")
            },
            "source_manifests": {"tracer": tracer_manifest["status"], "official": official_manifest["status"]},
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.root, args.protocol)
    print(json.dumps({"status": summary["status"], "independent_cases": summary["independent_cases"]}, indent=2))


if __name__ == "__main__":
    main()
