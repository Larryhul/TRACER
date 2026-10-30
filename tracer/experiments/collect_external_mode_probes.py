from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tracer.experiments.add_live_static_context import relative_static_segments
from tracer.experiments.train_rolling_external_belief import recover_training_labels, sha256, speed_thresholds
from tracer.experiments.train_rolling_external_response import branch_response, collect


def main():
    parser = argparse.ArgumentParser(description="Collect training-only same-state preferred-speed interventions without changing native validation or live episodes.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--num-modes", type=int, choices=(2, 3), default=3)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    args = parser.parse_args()
    source, output = Path(args.data_dir), Path(args.output_dir)
    if output.exists() or args.stride < 1:
        raise ValueError("mode probes require a fresh directory and positive stride")
    manifest = json.loads((source / "data_manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != "rolling-prefix-aligned-v2":
        raise ValueError("mode probes require replayable rolling collection data")
    collection = argparse.Namespace(**manifest["parameters"])
    with np.load(source / "train.npz") as loaded:
        training = dict(loaded)
    with np.load(source / f"{args.split}.npz") as loaded:
        native = dict(loaded)
    external_root = Path(collection.external_root).resolve()
    speeds = recover_training_labels({"train": training}, external_root)
    thresholds = speed_thresholds(training, speeds, args.num_modes)
    training_speeds = np.asarray([speeds[identity] for identity in sorted(speeds)])
    centers = [float(np.median(training_speeds[np.searchsorted(thresholds, training_speeds) == mode]))
               for mode in range(args.num_modes)]
    fields = ("history", "current", "candidates", "neighbors", "neighbor_mask", "target", "case", "window",
              "human", "branch", "reference_branch", "mode", "preferred_speed", "static_segments")
    records = {name: [] for name in fields}
    replayed_states = []

    def record_state(env, candidates, headings, histories, selected, reference, action_type, case, window, obstacles):
        if window % args.stride:
            return
        native_rows = np.flatnonzero((native["case"] == case) & (native["window"] == window))
        if not len(native_rows):
            raise ValueError("mode probe replay reached a state outside native training collection")
        robot_position, robot_velocity = np.asarray(env.robot.get_position()), np.asarray(env.robot.get_velocity())
        original_speeds = [human.v_pref for human in env.humans]
        for human_index, human in enumerate(env.humans):
            native_row = native_rows[np.flatnonzero(native["human"][native_rows] == human_index)[0]]
            if not np.allclose(histories[human_index], native["history"][native_row], atol=2e-5, rtol=0):
                raise ValueError("mode probe replay differs from native observed history")
            origin = np.asarray(human.get_position())
            current = np.concatenate([robot_position - origin, robot_velocity])[None]
            if not np.allclose(current, native["current"][native_row], atol=2e-5, rtol=0):
                raise ValueError("mode probe replay differs from native robot state")
            closest = int(np.linalg.norm(candidates[:, -1, :2] - origin, axis=-1).argmin())
            branches = sorted({selected, reference, closest})
            for mode, speed in enumerate(centers):
                for branch in branches:
                    positions, actions, targets = branch_response(
                        env, candidates[branch, :, 2:], headings[branch], action_type,
                        preferred_speeds={human_index: speed},
                    )
                    values = {"history": histories[human_index], "current": current,
                              "candidates": np.concatenate([positions - origin, actions], -1)[None],
                              "neighbors": histories, "neighbor_mask": np.arange(len(env.humans)) != human_index,
                              "target": targets[:, human_index], "case": case, "window": window, "human": human_index,
                              "branch": branch, "reference_branch": reference, "mode": mode, "preferred_speed": speed,
                              "static_segments": relative_static_segments(obstacles, origin[None])[0]}
                    for name in fields:
                        records[name].append(values[name])
        if original_speeds != [human.v_pref for human in env.humans]:
            raise ValueError("training mode interventions mutated the native live episode")
        replayed_states.append((int(case), int(window)))

    collect(collection, sorted(set(map(int, native["case"]))), state_callback=record_state,
            collect_native_responses=False)
    data = {name: np.asarray(values) for name, values in records.items()}
    if not len(data["target"]):
        raise ValueError("mode probes produced no training records")
    output.mkdir(parents=True)
    np.savez_compressed(output / f"{args.split}.npz", **data)
    evidence = {"phase": "train" if args.split == "train" else "validation", "split": args.split,
                "source_native_train_sha256": sha256(source / "train.npz"),
                "source_native_split_sha256": sha256(source / f"{args.split}.npz"),
                "source_collection_manifest": manifest, "case_ids": sorted(set(map(int, data["case"]))),
                "mode_speed_values": centers, "speed_thresholds_train_only": thresholds.tolist(),
                "state_replay_audit": "observed histories and robot states match native training collection",
                "replayed_states": replayed_states, "rows": len(data["target"]),
                "source_code_sha256": {name: sha256(Path(__file__).resolve().parents[1] / name)
                                       for name in ("experiments/collect_external_mode_probes.py", "experiments/train_rolling_external_response.py")},
                "boundary": "hypothetical speed intervention on one cloned human at a time; training only for train split, held-out diagnostics only for val split; never factual filter evidence or native mode relabeling"}
    (output / "manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"output": str(output), "rows": len(data["target"]), "mode_speed_values": centers}), flush=True)


if __name__ == "__main__":
    main()
