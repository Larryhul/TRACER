from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tracer.experiments.collect_external_planner_supervision import collection_cases, file_hash
from tracer.experiments.calibrate_external_belief_cadence import _validate_splits
from tracer.filesystem import link_or_copy


def merge(output, shards, splits, split, case_subset=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite merged planner data")
    cases = collection_cases(_validate_splits(splits), split)
    if case_subset is not None:
        if not case_subset or len(set(case_subset)) != len(case_subset) or not set(case_subset).issubset(cases):
            raise ValueError("merge subset must contain unique cases in the original declared split")
        cases = sorted(case_subset)
    records, inputs, seen_cases = [], {}, set()
    reference = None
    pending_files = []
    continuation_counts = {name: 0 for name in ("continuation_groups", "continuation_branches", "factual_continuation_checks", "factual_terminal_checks")}
    for shard in map(Path, shards):
        manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        protocol = json.loads((shard / "protocol.json").read_text(encoding="utf-8"))
        if manifest["status"] != "COMPLETED" or protocol["split"] != split:
            raise ValueError("only completed same-split shards may be merged")
        if manifest.get("changed_sources"):
            raise ValueError("changed collection sources cannot be merged")
        shard_cases = set(protocol["cases"])
        if not shard_cases.issubset(cases) or shard_cases & seen_cases:
            raise ValueError("shards must be non-overlapping subsets of the requested split")
        seen_cases.update(shard_cases)
        if reference is None:
            reference = protocol
        else:
            for name in ("planning", "dt", "execute_steps", "checkpoint_sha256", "seed"):
                if protocol[name] != reference[name]:
                    raise ValueError(f"shard protocol mismatch: {name}")
            for name in ("continuation_windows", "continuation_branches"):
                if protocol.get(name, 0) != reference.get(name, 0):
                    raise ValueError(f"shard continuation mismatch: {name}")
            if protocol.get("behavior_value_sha256") != reference.get("behavior_value_sha256"):
                raise ValueError("shards use different continuation behavior value policies")
            if protocol.get("continuation_outcome_schema") != reference.get("continuation_outcome_schema"):
                raise ValueError("shards use different terminal outcome contracts")
            allowed = {"output_dir", "case_start", "cases", "policies", "belief_cadence_source"}
            for name in set(protocol["parameters"]) | set(reference["parameters"]):
                if name == "belief_calibration_path":
                    first = Path(protocol["parameters"][name].format(seed=protocol["seed"])).resolve()
                    second = Path(reference["parameters"][name].format(seed=reference["seed"])).resolve()
                    if first != second or not first.is_file():
                        raise ValueError("shard calibration mismatch")
                    continue
                if name not in allowed and protocol["parameters"].get(name) != reference["parameters"].get(name):
                    raise ValueError(f"shard controller mismatch: {name}")
        for name in ("protocol.json", "manifest.json", "audit/windows.jsonl"):
            inputs[str((shard / name).resolve())] = file_hash(shard / name)
        for line in (shard / "audit/windows.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["case"] not in shard_cases or not row.get("live_prefix_verified"):
                raise ValueError("shard has out-of-split or unverified observations")
            filename = f"seed{row['seed']}_case{row['case']}_window{row['window']}.npz"
            for directory in ("audit", "inputs"):
                source = (shard / directory / filename).resolve()
                if not source.is_file():
                    raise ValueError("missing planner input or oracle snapshot")
                pending_files.append((source, output / directory / filename))
            if protocol.get("continuation_windows", 0):
                source = (shard / "continuations" / Path(filename).with_suffix(".json")).resolve()
                if not source.is_file():
                    raise ValueError("missing verified continuation group")
                pending_files.append((source, output / "continuations" / source.name))
            records.append(row)
        if protocol.get("continuation_windows", 0):
            for name in continuation_counts:
                continuation_counts[name] += manifest.get(name, 0) if name == "factual_terminal_checks" else manifest[name]
    if seen_cases != set(cases):
        raise ValueError("merged supervision does not cover the entire declared split")
    keys = {(row["seed"], row["case"], row["window"]) for row in records}
    if len(keys) != len(records):
        raise ValueError("duplicate decision groups")
    output.mkdir(parents=True)
    for source, destination in pending_files:
        destination.parent.mkdir(exist_ok=True)
        link_or_copy(source, destination)
    records.sort(key=lambda row: (row["case"], row["window"]))
    (output / "audit/windows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    protocol = {**reference, "version": "planner-supervision-merged-v1", "case_splits": splits, "cases": cases,
                "source_shards": [str(Path(shard).resolve()) for shard in shards], "shard_input_hashes": inputs,
                "boundary": "Disjoint immutable completed shards; per-shard controller and collection stride remain disclosed. No historical test or failed confirmation case enters train/validation."}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if reference.get("continuation_windows", 0) and continuation_counts["continuation_groups"] != len(records):
        raise ValueError("merged continuation count differs from the verified decision groups")
    (output / "manifest.json").write_text(json.dumps({"status": "COMPLETED", "cases": cases,
                                                     "groups": len(records), "verified_groups": len(records),
                                                     **(continuation_counts if reference.get("continuation_windows", 0) else {})}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Merge immutable disjoint planner-supervision shards.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shards", required=True, nargs="+")
    parser.add_argument("--case-splits", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--case-subset", type=int, nargs="+")
    args = parser.parse_args()
    merge(args.output, args.shards, json.loads(Path(args.case_splits).read_text(encoding="utf-8")), args.split, args.case_subset)


if __name__ == "__main__":
    main()
