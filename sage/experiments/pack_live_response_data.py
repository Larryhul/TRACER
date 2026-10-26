from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def partition_records(records, train_cases, val_cases):
    if set(train_cases) & set(val_cases) or not train_cases or not val_cases:
        raise ValueError("training and validation cases must be nonempty and disjoint")
    all_cases = set(train_cases) | set(val_cases)
    if set(np.unique(records["case"])) != all_cases:
        raise ValueError("response records must cover exactly the disclosed train/val cases")
    datasets = {split: {name: value[np.isin(records["case"], cases)] for name, value in records.items()}
                for split, cases in (("train", train_cases), ("val", val_cases))}
    for split, data in datasets.items():
        states = set(zip(data["case"], data["human"], data["window"]))
        selected = set(zip(data["case"][data["selected_branch"]], data["human"][data["selected_branch"]],
                           data["window"][data["selected_branch"]]))
        if states != selected:
            raise ValueError(f"{split} does not include the factual selected branch for every state")
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Pack post-selection live response audits into disjoint training/calibration data.")
    parser.add_argument("--collection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-start", type=int, default=112)
    parser.add_argument("--train-cases", type=int, default=24)
    parser.add_argument("--val-cases", type=int, default=8)
    args = parser.parse_args()
    collection, output = Path(args.collection_dir), Path(args.output_dir)
    if output.exists():
        raise ValueError("packed response data must use a fresh output directory")
    metadata = json.loads((collection / "metadata.json").read_text())
    parameters = metadata["parameters"]
    if len(parameters["seeds"]) != 1 or len(parameters["policies"]) != 1:
        raise ValueError("response packing requires one frozen collection controller")
    if parameters["response_audit_stride"] < 1 or parameters["execute_steps"] != 1:
        raise ValueError("this packing protocol requires one executed step and positive audit spacing")
    files = sorted((collection / "response_audit").glob("*.npz"))
    if not files:
        raise ValueError("no completed live response audits found")
    records = []
    for path in files:
        with np.load(path) as loaded:
            records.append(dict(loaded))
    data = {name: np.concatenate([record[name] for record in records]) for name in records[0]}
    if "human_current_velocity" in data:
        observed_velocity = (data["history"][:, -1] - data["history"][:, -2]) / parameters["env_dt"]
        if not np.allclose(observed_velocity, data["human_current_velocity"], atol=2e-5):
            raise ValueError("observed history velocity disagrees with the factual simulator velocity")
        if not np.allclose(data["current"][..., 2:], data["previous_executed_robot_velocity"], atol=1e-6):
            raise ValueError("delayed response must use the previous executed robot velocity")
    train = list(range(args.train_start, args.train_start + args.train_cases))
    val = list(range(train[-1] + 1, train[-1] + 1 + args.val_cases))
    datasets = partition_records(data, train, val)
    manifest = {"version": "tracer-live-prefix-aligned-v2" if "human_current_velocity" in data else "tracer-live-prefix-aligned-v1", "parameters": {
        "train_start": args.train_start, "train_cases": args.train_cases, "val_cases": args.val_cases,
        "history": int(data["history"].shape[1]), "horizon": int(data["target"].shape[1]),
        "windows": parameters["windows"], "collection_stride": parameters["response_audit_stride"],
        "data_seed": parameters["candidate_seed"], "dt": parameters["env_dt"],
    }, "collection_controller_parameters": parameters,
        "collection_metadata_sha256": hashlib.sha256((collection / "metadata.json").read_bytes()).hexdigest(),
        "source_audit_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "case_splits": {"train": train, "val": val},
        "rows": {split: len(records["target"]) for split, records in datasets.items()},
        "claim_boundary": "states visited by one frozen TRACER controller; branched futures are counterfactual training labels, not factual belief updates; not on-policy for every future model",
        "data_seed_semantics": "base candidate RNG seed of the disclosed frozen collection controller"}
    output.mkdir(parents=True)
    for split, records in datasets.items():
        np.savez_compressed(output / f"{split}.npz", **records)
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["rows"]))


if __name__ == "__main__":
    main()
