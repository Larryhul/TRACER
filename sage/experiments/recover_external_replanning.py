from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from sage.experiments.collect_external_planner_supervision import file_hash
from sage.experiments.eval_persistent_closed_loop import _load_model
from sage.experiments.external_information_value import simulator_snapshot
from sage.experiments.run_sicnav_tracer_loop import _run_episode
from sage.response.sage_candidate_adapter import SAGECandidateAdapter, SAGECandidateAdapterConfig


def require_equal(actual, expected, label, tolerance=2e-6):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=tolerance, rtol=0, equal_nan=True):
        raise ValueError(f"recovered factual replay differs: {label}")


def verify_completed_episode(original, replay, continuations, terminal_position, goal, dt, execute_steps):
    for name in ("case", "seed", "initial_scene_hash", "policy", "robot_kinematics"):
        if str(original[name]) != str(replay[name]):
            raise ValueError(f"recovered episode differs: {name}")
    for name in ("steps", "windows", "total_time", "success", "collision_free_success", "collision",
                 "wall_collision", "timeout", "environment_timeout", "task_progress"):
        require_equal(float(replay[name]), float(original[name]), name, 2e-5)
    terminal_distance = np.linalg.norm(np.asarray(terminal_position) - goal)
    for group in continuations:
        parents = [branch for branch in group["branches"] if branch["candidate"] == group["parent_selected"]]
        if len(parents) != 1 or not all(branch["first_window_verified"] for branch in group["branches"]):
            raise ValueError("recovered continuation requires one verified factual parent")
        parent = parents[0]
        elapsed_steps = int(replay["steps"]) - group["window"] * execute_steps
        require_equal(parent["elapsed_steps"], elapsed_steps, "parent remaining steps")
        require_equal(parent["elapsed_time"], elapsed_steps * dt, "parent remaining time")
        require_equal(parent["terminal_goal_distance"], terminal_distance, "parent terminal distance", 2e-5)
        require_equal(parent["success"], float(replay["success"]), "parent terminal success")


class RecoveryReplay:
    def __init__(self, root, records):
        self.root = Path(root)
        self.records = {(int(row["case"]), int(row["window"])): row for row in records}
        self.verified = set()
        self.pending = None
        self.env = None
        self.goal = None
        self.terminal_position = None
        self.input_hashes = {}

    def observe(self, **values):
        self.env = values["env"]
        self.goal = np.asarray(self.env.robot.get_goal_position())
        if self.pending is not None:
            raise ValueError("recovery skipped a factual execution check")
        key = (int(values["case"]), int(values["window"]))
        if key not in self.records:
            return
        filename = f"seed{values['seed']}_case{key[0]}_window{key[1]}.npz"
        input_path, audit_path = self.root / "inputs" / filename, self.root / "audit" / filename
        self.input_hashes.update({str(path): file_hash(path) for path in (input_path, audit_path)})
        with np.load(input_path, allow_pickle=False) as inputs, np.load(audit_path, allow_pickle=False) as audit:
            for name in ("histories", "current"):
                actual = values[name]
                if isinstance(actual, torch.Tensor):
                    actual = actual.detach().cpu().numpy()
                require_equal(actual, inputs[name], name)
            require_equal(values["beliefs"].detach().cpu().numpy(), inputs["beliefs"], "belief")
            require_equal(values["global_candidates"], audit["planned_candidates"], "candidate pool")
            require_equal(values["headings"], audit["planned_headings"], "headings")
            require_equal(values["decision_scores"].cpu().numpy(), audit["score_persistent"], "scores")
            for saved, actual in zip(("teacher_means", "teacher_log_stds", "teacher_references"), values["cached_predictions"]):
                require_equal(actual.detach().cpu().numpy(), inputs[saved], saved)
            continuation = json.loads((self.root / "continuations" / Path(filename).with_suffix(".json")).read_text())
            if values["selected"] != continuation["parent_selected"]:
                raise ValueError("recovery changed the factual selected action")
            self.pending = (key, audit["snapshots"][values["selected"], :values["args"].execute_steps].copy())

    def verify_execution(self, env, steps):
        self.terminal_position = np.asarray(env.robot.get_position())
        if self.pending is None:
            return
        key, snapshots = self.pending
        require_equal(simulator_snapshot(env), snapshots[steps - 1], "executed prefix", 2e-5)
        self.verified.add(key)
        self.pending = None


def main():
    parser = argparse.ArgumentParser(description="Recover only checkpointed complete cases after exact frozen factual replay.")
    parser.add_argument("--interrupted-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-stride", type=int, required=True)
    args = parser.parse_args()
    root, output = Path(args.interrupted_dir).resolve(), Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite a recovered shard")
    protocol = json.loads((root / "protocol.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "COLLECTING" or not protocol.get("continuation_windows") or protocol.get("behavior_value_model"):
        raise ValueError("recovery is limited to interrupted frozen-base complete-continuation collection")
    hashes = dict(protocol["source_hashes"])
    for path, expected in hashes.items():
        if not Path(path).exists() or file_hash(path) != expected:
            raise ValueError(f"interrupted collection source changed: {path}")
    episodes = list(csv.DictReader((root / "episodes.csv").open()))
    cases = [int(row["case"]) for row in episodes]
    if not cases or len(set(cases)) != len(cases) or cases != protocol["cases"][:len(cases)]:
        raise ValueError("recovery requires a unique complete-case prefix checkpoint")
    records = [json.loads(line) for line in (root / "audit/windows.jsonl").read_text().splitlines()]
    records = [row for row in records if int(row["case"]) in cases]
    if args.audit_stride < 1:
        raise ValueError("recovery requires the original positive collection stride")
    expected_keys = {(int(row["case"]), window) for row in episodes
                     for window in range(0, int(row["windows"]), args.audit_stride)}
    if len(records) != len(expected_keys) or {(int(row["case"]), int(row["window"])) for row in records} != expected_keys:
        raise ValueError("recovery cannot omit or duplicate an original audited window")
    if not all(row.get("live_prefix_verified") for row in records):
        raise ValueError("recovery cannot include unverified original execution prefixes")
    for name in ("protocol.json", "manifest.json", "episodes.csv", "audit/windows.jsonl"):
        hashes[str(root / name)] = file_hash(root / name)
    hashes[str(Path(__file__).resolve())] = file_hash(__file__)
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps({"status": "REPLAYING_INTERRUPTED_CASES"}) + "\n")
    runtime = argparse.Namespace(**protocol["parameters"])
    runtime.output_dir = str(output / "replay")
    torch.set_num_threads(2)
    model = _load_model(Path(protocol["checkpoint"]), runtime.device, expected_dt=protocol["dt"])
    adapter = SAGECandidateAdapter.from_checkpoint(
        Path(runtime.sage_checkpoint).resolve(), runtime.device,
        SAGECandidateAdapterConfig(
            num_candidates=runtime.num_candidates, guided=True, step_stride=runtime.sage_step_stride,
            goal_speed=runtime.sage_goal_speed, oversample_factor=runtime.sage_oversample_factor,
            task_prefilter=True, task_prefix_steps=runtime.execute_steps, goal_direction="task_goal",
            include_task_anchor=True, task_anchor_speed_scales=(.45,), batch_size=runtime.num_candidates,
            guidance_step_clip_norm=.1, diversity_weight=getattr(runtime, "sage_diversity_weight", 0.),
        ),
    )
    replay_audit = RecoveryReplay(root, records)
    groups = []
    for original in episodes:
        case = int(original["case"])
        case_records = [row for row in records if int(row["case"]) == case]
        continuations = []
        for row in case_records:
            path = root / "continuations" / f"seed{row['seed']}_case{case}_window{row['window']}.json"
            hashes[str(path)] = file_hash(path)
            continuations.append(json.loads(path.read_text()))
        try:
            replay = _run_episode(model, adapter, "tracer-persistent", Path(runtime.external_root).resolve(),
                                  case, Path(runtime.env_config).resolve(), runtime, protocol["seed"], decision_audit=replay_audit)
            verify_completed_episode(original, replay, continuations, replay_audit.terminal_position,
                                     replay_audit.goal, protocol["dt"], protocol["execute_steps"])
        finally:
            if replay_audit.env is not None:
                replay_audit.env.close()
        groups.extend(continuations)
        print(f"recovered case={case} groups={len(continuations)} exact_terminal_and_factual_prefixes=True", flush=True)
    if replay_audit.pending is not None or replay_audit.verified != set(replay_audit.records):
        raise ValueError("recovery did not verify every persisted factual group")
    hashes.update(replay_audit.input_hashes)
    changed = [path for path, expected in hashes.items() if file_hash(path) != expected]
    if changed:
        raise ValueError(f"recovery inputs changed during replay: {changed}")
    for row in records:
        filename = f"seed{row['seed']}_case{row['case']}_window{row['window']}.npz"
        for directory in ("inputs", "audit", "continuations"):
            name = Path(filename).with_suffix(".json") if directory == "continuations" else filename
            (output / directory).mkdir(exist_ok=True)
            (output / directory / name).symlink_to((root / directory / name).resolve())
    (output / "audit/windows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (output / "episodes.csv").write_bytes((root / "episodes.csv").read_bytes())
    protocol.update({"cases": cases, "recovered_from": str(root), "recovery_input_hashes": hashes,
                     "recovery_boundary": "Only complete checkpointed episodes. All saved factual inputs, candidate pools, predictions, selected executed prefixes, terminal returns and original episode outcomes were replay-verified. Interrupted next-case groups remain excluded; original files are unchanged."})
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({
        "status": "COMPLETED", "cases": cases, "changed_sources": [], "groups": len(records),
        "verified_groups": len(replay_audit.verified), "continuation_groups": len(groups),
        "continuation_branches": sum(len(group["branches"]) for group in groups),
        "factual_continuation_checks": len(replay_audit.verified), "replayed_complete_cases": len(cases),
        "verification_scope": "New exact factual-prefix and parent terminal-return replay; original intermediate continuation checks are not fabricated or recounted.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
