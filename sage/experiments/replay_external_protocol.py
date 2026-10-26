from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from sage.experiments.run_sicnav_tracer_loop import run


def replay_parameters(metadata, checkpoint_dir, output_dir, *, seeds=None, cases=None, case_start=None, policies=None, belief_cost_leverage=None):
    source = metadata["parameters"]
    parameters = copy.deepcopy(source)
    parameters.update(checkpoint_dir=checkpoint_dir, output_dir=output_dir)
    for name, value in (("seeds", seeds), ("cases", cases), ("case_start", case_start), ("policies", policies)):
        if value is not None:
            parameters[name] = value
    if belief_cost_leverage is not None:
        if not math.isfinite(belief_cost_leverage) or belief_cost_leverage <= 0:
            raise ValueError("belief-cost-leverage must be finite and positive")
        parameters["belief_cost_leverage"] = float(belief_cost_leverage)
    if not parameters["seeds"] or not set(parameters["seeds"]) <= set(source["seeds"]):
        raise ValueError("protocol replay seeds must be a nonempty subset of the reference")
    if not parameters["policies"] or not set(parameters["policies"]) <= set(source["policies"]):
        raise ValueError("protocol replay policies must be a nonempty subset of the reference")
    selected_cases = set(range(parameters["case_start"], parameters["case_start"] + parameters["cases"]))
    reference_cases = set(range(source["case_start"], source["case_start"] + source["cases"]))
    if not selected_cases or not selected_cases <= reference_cases:
        raise ValueError("protocol replay cases must be a nonempty subset of the reference")
    return parameters


def main():
    parser = argparse.ArgumentParser(description="Replay every frozen external controller parameter, changing only the response checkpoint and evaluation subset.")
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--cases", type=int)
    parser.add_argument("--case-start", type=int)
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--belief-vertex-audit", action="store_true")
    parser.add_argument("--belief-calibration-path")
    parser.add_argument("--belief-cost-leverage", type=float)
    args = parser.parse_args()
    source = Path(args.source_metadata)
    metadata = json.loads(source.read_text())
    parameters = replay_parameters(metadata, args.checkpoint_dir, args.output_dir,
                                   seeds=args.seeds, cases=args.cases, case_start=args.case_start,
                                   policies=args.policies, belief_cost_leverage=args.belief_cost_leverage)
    if args.belief_vertex_audit:
        parameters["belief_vertex_audit"] = True
    if args.belief_calibration_path:
        parameters["belief_calibration_path"] = args.belief_calibration_path
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("protocol replay requires a fresh output directory")
    output.mkdir(parents=True)
    overrides = {name: {"reference": metadata["parameters"].get(name), "replay": value}
                 for name, value in parameters.items() if value != metadata["parameters"].get(name)}
    (output / "protocol_replay.json").write_text(json.dumps({
        "source_metadata": str(source.resolve()), "source_metadata_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "overrides": overrides,
        "controller_parameters_changed": (
            parameters.get("belief_cost_leverage", 1.0) != metadata["parameters"].get("belief_cost_leverage", 1.0)
        ),
        "belief_calibration_replaced": bool(args.belief_calibration_path),
        "boundary": "reuses disclosed reference cases; a diagnostic replay, not a new untouched test set",
    }, indent=2) + "\n")
    run(argparse.Namespace(**parameters))


if __name__ == "__main__":
    main()
