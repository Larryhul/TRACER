from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SEEDS = (2027, 2031, 2039)
CASES = tuple(range(88, 112))
TRACER_POLICIES = ("tracer-persistent",)
OFFICIAL_POLICIES = (
    "sicnav-np", "sicnav-p", "official-mpc-cvmm", "official-dwa",
    "official-orca", "official-linear",
)
METRICS = (
    "collision_free_success", "success", "collision", "wall_collision",
    "timeout", "task_progress", "total_time", "min_distance",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_artifacts(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "COMPLETED":
        raise ValueError(f"Incomplete run: {directory}")
    for name, expected in manifest["artifacts"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {directory / name}")
    return manifest


def paired_grid(tracer_rows, official_rows, cases, seeds, tracer_policies, official_policies):
    expected_tracer = {(policy, seed, case) for policy in tracer_policies for seed in seeds for case in cases}
    expected_official = {(policy, case) for policy in official_policies for case in cases}
    tracer = {}
    official = {}
    for row in tracer_rows:
        key = row["policy"], int(row["seed"]), int(row["case"])
        if key in tracer:
            raise ValueError(f"Duplicate TRACER episode: {key}")
        tracer[key] = row
    for row in official_rows:
        key = row["policy"], int(row["case"])
        if key in official:
            raise ValueError(f"Duplicate official episode: {key}")
        official[key] = row
    if set(tracer) != expected_tracer or set(official) != expected_official:
        raise ValueError("Missing or unexpected policy/seed/case rows")
    for case in cases:
        reference = official[official_policies[0], case]
        selected = [row for (_, _, episode_case), row in tracer.items() if episode_case == case]
        selected += [row for (_, episode_case), row in official.items() if episode_case == case]
        for row in selected:
            if row["initial_scene_hash"] != reference["initial_scene_hash"]:
                raise ValueError(f"Initial scene mismatch: case {case}, {row['policy']}")
            fields = ["environment_dt"]
            if row["robot_kinematics"] == reference["robot_kinematics"]:
                fields.append("robot_initial_heading")
            for field in fields:
                if not math.isclose(float(row[field]), float(reference[field]), rel_tol=0, abs_tol=1e-10):
                    raise ValueError(f"{field} mismatch: case {case}, {row['policy']}")
            for field in METRICS:
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"Nonfinite {field}: case {case}, {row['policy']}")
            expected_success = float(row["success"]) * (1 - float(row["collision"])) * (1 - float(row["wall_collision"]))
            if float(row["collision_free_success"]) != expected_success:
                raise ValueError(f"Collision-free outcome mismatch: case {case}, {row['policy']}")
    return tracer, official


def bootstrap(values, seed=849731, samples=10000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap needs finite independent-scene values")
    generator = np.random.default_rng(seed)
    draws = values[generator.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()), "case_count": len(values),
        "case_bootstrap_ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
    }


def summarize(output, official_dir):
    output, official_dir = Path(output), Path(official_dir)
    tracer_dir = output / "tracer"
    verify_artifacts(official_dir)
    verify_artifacts(tracer_dir)
    tracer_meta = json.loads((tracer_dir / "metadata.json").read_text(encoding="utf-8"))
    official_meta = json.loads((official_dir / "metadata.json").read_text(encoding="utf-8"))
    if tracer_meta["external_repository"]["commit"] != official_meta["external_repository"]["commit"]:
        raise ValueError("External simulator commits differ")
    for field in ("max_steps", "case_start"):
        if tracer_meta["parameters"][field] != official_meta["parameters"][field]:
            raise ValueError(f"Episode protocol mismatch: {field}")
    tracer_rows = read_rows(tracer_dir / "episode_metrics.csv")
    official_rows = read_rows(official_dir / "episode_metrics.csv")
    tracer, official = paired_grid(tracer_rows, official_rows, CASES, SEEDS, TRACER_POLICIES, OFFICIAL_POLICIES)
    policies = {}
    scene_values = {}
    for policy in (*TRACER_POLICIES, *OFFICIAL_POLICIES):
        is_tracer = policy in TRACER_POLICIES
        selected = [row for row in (tracer_rows if is_tracer else official_rows) if row["policy"] == policy]
        values = {}
        for metric in METRICS:
            values[metric] = [
                float(np.mean([float(tracer[policy, seed, case][metric]) for seed in SEEDS]))
                if is_tracer else float(official[policy, case][metric])
                for case in CASES
            ]
        scene_values[policy] = values
        kinematics = sorted({row["robot_kinematics"] for row in selected})
        policies[policy] = {
            "records": len(selected), "independent_scenes": len(CASES), "robot_kinematics": kinematics,
            "comparison_role": "holonomic reference" if kinematics == ["holonomic"] else (
                "privileged-information unicycle reference" if policy == "sicnav-p" else "unicycle controller"
            ),
            "metrics": {metric: bootstrap(values[metric]) for metric in METRICS},
            "per_model_seed": {
                str(seed): {metric: float(np.mean([float(tracer[policy, seed, case][metric]) for case in CASES])) for metric in METRICS}
                for seed in SEEDS
            } if is_tracer else {},
        }
    contrasts = {
        policy: {metric: bootstrap(np.asarray(scene_values["tracer-persistent"][metric]) - scene_values[policy][metric]) for metric in METRICS}
        for policy in OFFICIAL_POLICIES
    }
    summary = {
        "status": "COMPLETED", "completed_at_utc": timestamp(),
        "scope": "24 previously inspected development cases; frozen recovery snapshot, not a blind confirmation or noninferiority proof",
        "tracer_records": len(tracer_rows), "official_records": len(official_rows),
        "independent_scenes": len(CASES), "model_seeds": list(SEEDS),
        "scene_hash_and_clock_pairing": "PASS",
        "uncertainty": "Paired scene bootstrap after averaging model seeds within scene; conditional on three fitted models",
        "policies": policies, "persistent_minus_official": contrasts,
        "input_hashes": {str(path): digest(path) for path in (
            tracer_dir / "manifest.json", official_dir / "manifest.json",
            tracer_dir / "episode_metrics.csv", official_dir / "episode_metrics.csv",
        )},
    }
    write_json(output / "summary.json", summary)
    columns = ["source", "policy", "seed", "case", "initial_scene_hash", "robot_kinematics", "environment_dt", *METRICS]
    with (output / "episode_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for source, rows in (("tracer", tracer_rows), ("official", official_rows)):
            for row in rows:
                writer.writerow({field: source if field == "source" else row.get(field, "") for field in columns})
    lines = [
        "# External baseline comparison — recovered frozen development run", "",
        f"Completed: {summary['completed_at_utc']}", "",
        "24 previously inspected cases (88–111), 0.25 s steps, 60-step budget. Three fitted response seeds; not 72 independent scenes.",
        "The original launch mixed older planner arguments with the static-regret checkpoint. This run restores the archived calibrated full controller and the source snapshot used by the later parent confirmation; no new parameter search is performed.",
        "Source is frozen but originated in a dirty worktree. No clean/blind confirmation or noninferiority claim is made.", "",
        "## Results", "",
        "| Policy | Role | Collision-free success | Success | Collision | Wall collision | Timeout | Progress | Time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, values in policies.items():
        metrics = values["metrics"]
        percentages = " | ".join(f"{metrics[metric]['mean']:.2%}" for metric in METRICS[:5])
        lines.append(f"| {policy} | {values['comparison_role']} | {percentages} | {metrics['task_progress']['mean']:.6f} | {metrics['total_time']['mean']:.6f} |")
    lines += ["", "## Persistent minus official: paired scene contrasts", "", "| Baseline | Collision-free success difference (pp), 95% CI | Time difference (s), 95% CI |", "|---|---:|---:|"]
    for policy, metrics in contrasts.items():
        success, elapsed = metrics["collision_free_success"], metrics["total_time"]
        success_low, success_high = success["case_bootstrap_ci95"]
        time_low, time_high = elapsed["case_bootstrap_ci95"]
        lines.append(f"| {policy} | {100 * success['mean']:+.2f} [{100 * success_low:+.2f}, {100 * success_high:+.2f}] | {elapsed['mean']:+.4f} [{time_low:+.4f}, {time_high:+.4f}] |")
    lines += [
        "", "## Boundaries", "",
        "- ORCA and Linear retain native holonomic kinematics; they are not matched-unicycle comparisons.",
        "- SICNav-P uses privileged information and is reported separately from nonprivileged controllers.",
        "- Controller candidate sets differ. This compares the same initial scenes and budgets, not identical candidate support.",
        "- HSL/MA57 was unavailable during the official run; its runtime warning reported fallback to the default IPOPT linear solver.",
        "- Aggregate time includes timeouts; success, collision and wall collision are reported separately.",
        "- The value-policy candidate that failed its separate confirmation is not silently promoted here.",
        "- Human-response burden is not reported as a common baseline outcome because the official run did not archive that measurement.",
        "- One robot in CrowdSimPlus: no multi-robot or real-human causal conclusion.", "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    write_json(output / "manifest.json", {
        "status": "COMPLETED", "artifacts": {name: digest(output / name) for name in (
            "summary.json", "report.md", "episode_metrics.csv", "selection.json",
        )},
    })
    return summary


def controller_parameters(metadata, output):
    parameters = dict(metadata["parameters"])
    parameters.update(
        allow_dirty=True, output_dir=str(output / "tracer"), cases=len(CASES), case_start=min(CASES),
        seeds=list(SEEDS), policies=list(TRACER_POLICIES), belief_update_stride=1,
        belief_cost_leverage=1.0, prefix_diversity=False, belief_vertex_audit=False,
    )
    return parameters


def run(args):
    workspace, snapshot, official, output = (Path(value).resolve() for value in (
        args.workspace, args.snapshot, args.official, args.output,
    ))
    output.mkdir(parents=True, exist_ok=True)
    controller_metadata = Path(args.controller_metadata).resolve()
    parameters = controller_parameters(json.loads(controller_metadata.read_text(encoding="utf-8")), output)
    protected = {str(snapshot / "tracer" / name): expected for name, expected in json.loads((snapshot.parent / "source_hashes.json").read_text(encoding="utf-8")).items()}
    official_meta = json.loads((official / "metadata.json").read_text(encoding="utf-8"))
    for name, expected in official_meta["source_hashes"].items():
        path = snapshot / name if name.startswith("tracer/") else workspace / name
        protected[str(path)] = expected
    for seed in SEEDS:
        path = workspace / f"experiments/tracer_live_belief_static_regret_20260906/seed{seed}/response.pt"
        protected[str(path)] = digest(path)
        calibration = workspace / str(parameters["belief_calibration_path"]).format(seed=seed)
        protected[str(calibration)] = digest(calibration)
    sage_checkpoint = workspace / "experiments/pilot/h3_sage_generator_p0_seed2027/synthetic_sage_epoch20.pt"
    protected[str(sage_checkpoint)] = digest(sage_checkpoint)
    protected[str(Path(__file__).resolve())] = digest(__file__)
    protected[str(controller_metadata)] = digest(controller_metadata)
    selection_path = output / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection["protected_inputs"] != protected:
            raise ValueError("Recovery inputs differ from the frozen selection")
    else:
        selection = {
            "frozen_at_utc": timestamp(), "workspace": str(workspace), "snapshot": str(snapshot),
            "official": str(official), "protected_inputs": protected,
            "controller_metadata": str(controller_metadata), "parameters": parameters, "cases": list(CASES),
            "seeds": list(SEEDS), "policies": list(TRACER_POLICIES),
            "execution": {"torch_threads": 1, "initial_rng_seed": 20260908},
            "boundary": "Previously inspected development cases. Restore archived calibrated controller, not the erroneous hybrid launch. No parameter search or value-policy promotion.",
        }
        write_json(selection_path, selection)
    for name, expected in protected.items():
        if digest(name) != expected:
            raise ValueError(f"Protected input changed: {name}")
    verify_artifacts(official)
    if (output / "tracer/manifest.json").exists():
        return summarize(output, official)
    os.chdir(workspace)
    sys.path.insert(0, str(snapshot))
    runner = importlib.import_module("tracer.experiments.run_sicnav_tracer_loop")
    if not Path(runner.__file__).resolve().is_relative_to(snapshot):
        raise RuntimeError("Runner was not imported from the frozen snapshot")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260908)
    np.random.seed(20260908)
    random.seed(20260908)
    original_episode = runner._run_episode
    recovery = output / "episodes"
    recovery.mkdir(exist_ok=True)
    completed = 0
    started = time.monotonic()

    def durable_episode(*episode_args, **episode_kwargs):
        nonlocal completed
        policy, case, seed = episode_args[2], episode_args[4], episode_args[7]
        ranking_rows, window_rows = episode_args[8], episode_args[9]
        path = recovery / f"seed{seed}_case{case}_{policy}.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["selection_sha256"] != digest(selection_path):
                raise ValueError(f"Recovery selection mismatch: {path}")
            ranking_rows.extend(saved["ranking_rows"])
            window_rows.extend(saved["window_rows"])
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
            np.random.set_state(saved["numpy_rng"])
            random.setstate(saved["python_rng"])
            record = saved["record"]
        else:
            ranking_start, window_start = len(ranking_rows), len(window_rows)
            record = original_episode(*episode_args, **episode_kwargs)
            saved = {
                "record": record, "ranking_rows": ranking_rows[ranking_start:], "window_rows": window_rows[window_start:],
                "selection_sha256": digest(selection_path), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            }
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                torch.save(saved, handle)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        completed += 1
        status = {
            "status": "RUNNING", "completed_episodes": completed, "expected_episodes": len(SEEDS) * len(CASES) * len(TRACER_POLICIES),
            "last_episode": {"seed": seed, "case": case, "policy": policy},
            "updated_at_utc": timestamp(), "elapsed_seconds": time.monotonic() - started,
        }
        write_json(output / "status.json", status)
        print(json.dumps(status), flush=True)
        return record

    runner._run_episode = durable_episode
    runner.run(argparse.Namespace(**selection["parameters"]))
    for name, expected in protected.items():
        if digest(name) != expected:
            raise ValueError(f"Protected input changed during evaluation: {name}")
    result = summarize(output, official)
    write_json(output / "status.json", {"status": "COMPLETED", "completed_episodes": completed, "updated_at_utc": timestamp()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--official", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--controller-metadata", required=True)
    args = parser.parse_args()
    try:
        result = run(args)
        print(json.dumps({key: result[key] for key in ("status", "tracer_records", "official_records")}), flush=True)
    except Exception as error:
        output = Path(args.output)
        if output.exists():
            write_json(output / "failure.json", {"status": "FAILED", "error": repr(error), "updated_at_utc": timestamp()})
        raise


if __name__ == "__main__":
    main()
