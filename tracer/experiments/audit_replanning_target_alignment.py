from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tracer.experiments.collect_external_planner_supervision import file_hash
from tracer.experiments.train_executable_value import load_continuation_groups


def prefix_cost_alignment(groups, directory, execute_steps, seed):
    rows = []
    for group in groups:
        if not group["rerank_allowed"]:
            continue
        name = f"seed{seed}_case{group['case']}_window{group['window']}.npz"
        with np.load(Path(directory) / "audit" / name, allow_pickle=False) as audit:
            prefixes = audit["snapshots"][:, :execute_steps, 0, :5].reshape(len(audit["snapshots"]), -1)
            oracle = audit["factual_total"]
            eligible = audit["eligible"]
            prefix_costs = []
            for index in group["candidate_indices"]:
                same = np.isclose(prefixes, prefixes[index], atol=1e-6, rtol=0).all(1) & eligible
                prefix_costs.append(float(oracle[same].min()) if same.any() else float("nan"))
        prefix_costs = np.asarray(prefix_costs)
        if not np.isfinite(prefix_costs[group["parent"]]):
            continue
        for index, original in enumerate(group["candidate_indices"]):
            if index == group["parent"] or not np.isfinite(prefix_costs[index]):
                continue
            difference = group["targets"][index] - group["targets"][group["parent"]]
            rows.append({
                "case": group["case"], "window": group["window"], "candidate": original,
                "open_loop_prefix_cost_difference": float(prefix_costs[index] - prefix_costs[group["parent"]]),
                "terminal_time_difference": float(difference[0]),
                "failure_difference": float(difference[1]), "collision_difference": float(difference[2]),
            })
    open_loop = np.asarray([row["open_loop_prefix_cost_difference"] for row in rows])
    navigation = np.asarray([row["terminal_time_difference"] for row in rows])
    safe = np.asarray([row["failure_difference"] <= 0 and row["collision_difference"] <= 0 for row in rows])
    local_better = open_loop < -1e-6
    genuinely_faster = (navigation < -1e-6) & safe
    return {
        "paired_alternatives": len(rows), "cases": len({row["case"] for row in rows}),
        "open_loop_better_alternatives": int(local_better.sum()),
        "open_loop_better_but_no_safe_time_gain": int((local_better & ~genuinely_faster).sum()),
        "safe_faster_alternatives": int(genuinely_faster.sum()),
        "safe_faster_not_recognized_by_open_loop": int((genuinely_faster & ~local_better).sum()),
        "delta_correlation": float(np.corrcoef(open_loop, navigation)[0, 1]) if len(rows) > 1 and np.std(open_loop) > 0 and np.std(navigation) > 0 else None,
        "records": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit whether finite-horizon prefix oracle improvement predicts terminal replanning improvement.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    groups, protocol, hashes = load_continuation_groups(args.data_dir, args.split)
    report = prefix_cost_alignment(groups, args.data_dir, protocol["execute_steps"], protocol["seed"])
    changed = [path for path, digest in hashes.items() if not Path(path).exists() or file_hash(path) != digest]
    report.update({"status": "SOURCE_CHANGED" if changed else "COMPLETED", "changed_inputs": changed,
                   "input_hashes": hashes, "split": args.split,
                   "boundary": "Same actual executed prefix, same observed parent state; compares an optimistic finite-pool open-loop diagnostic to one intervention followed by the frozen parent controller. Descriptive, not a deployable oracle or independent confirmation."})
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("target-alignment audit already exists")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("records", "input_hashes")}, indent=2))


if __name__ == "__main__":
    main()
