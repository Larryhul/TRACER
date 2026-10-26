"""Audit memory accumulation separately from identity re-entry in saved H2 rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sage.experiments.formal_manifest import sha256_file


METRICS = ("predictive_trajectory_nll", "trajectory_ade", "trajectory_fde")


def memory_strata(active):
    observed = 0
    result = []
    for window, present in enumerate(active):
        labels = []
        if present and window > 0:
            labels.append("all_decisions")
            if observed == 0:
                labels.append("no_previous_observation")
            elif active[window - 1]:
                labels.append("one_previous_observation" if observed == 1 else "accumulation_without_reentry")
            else:
                labels.append("identity_reentry")
        result.append(labels)
        observed += int(present)
    return result


def summarize_rows(path):
    rows = {}
    tracks = defaultdict(dict)
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            if row["method"] not in {"persistent", "reset", "shuffled-id", "recurrent"}:
                continue
            key = (row["split"], int(row["episode_id"]), int(row["entity_id"]), int(row["window"]))
            method_key = (*key, row["method"])
            if method_key in rows:
                raise ValueError("duplicate H2 prediction row")
            rows[method_key] = row
            if row["method"] == "persistent":
                tracks[key[:3]][key[3]] = bool(int(row["active"]))
    episodes = defaultdict(lambda: defaultdict(list))
    for track, active_by_window in tracks.items():
        if sorted(active_by_window) != list(range(len(active_by_window))):
            raise ValueError("track windows must be complete and contiguous")
        labels = memory_strata([active_by_window[window] for window in range(len(active_by_window))])
        for window, strata in enumerate(labels):
            persistent = rows[(*track, window, "persistent")]
            for baseline in ("reset", "shuffled-id", "recurrent"):
                other = rows[(*track, window, baseline)]
                for key in ("mode", "active", "main_evaluation", "high_interaction"):
                    if persistent[key] != other[key]:
                        raise ValueError(f"methods use different H2 factual targets: {key}")
                for stratum in strata:
                    for metric in METRICS:
                        difference = float(persistent[metric]) - float(other[metric])
                        if not np.isfinite(difference):
                            raise ValueError("nonfinite paired H2 metric")
                        episodes[(track[0], stratum, baseline, metric)][track[1]].append(difference)
    return {
        key: {episode: float(np.mean(values)) for episode, values in groups.items()}
        for key, groups in episodes.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029, 2030, 2031])
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("do not overwrite a prior H2 audit")
    files = [Path(args.checkpoint_dir) / f"seed{seed}" / "raw_metrics.csv" for seed in args.seeds]
    hashes = {str(path): sha256_file(path) for path in files}
    seed_results = [summarize_rows(path) for path in files]
    records = []
    for key in sorted(seed_results[0]):
        episodes = sorted(seed_results[0][key])
        if any(set(seed[key]) != set(episodes) for seed in seed_results):
            raise ValueError("H2 seed/scene coverage differs")
        values = np.asarray([[seed[key][episode] for episode in episodes] for seed in seed_results])
        case_values = values.mean(axis=0)
        generator = np.random.default_rng(924621)
        bootstrap = generator.integers(0, len(episodes), size=(10000, len(episodes)))
        records.append({
            "split": key[0], "stratum": key[1], "contrast": "persistent_minus_" + key[2],
            "metric": key[3], "independent_episodes": len(episodes),
            "mean_difference": float(case_values.mean()),
            "case_bootstrap_ci95": np.quantile(case_values[bootstrap].mean(axis=1), [0.025, 0.975]).tolist(),
            "per_model_seed": dict(zip(args.seeds, values.mean(axis=1).tolist())),
        })
    changed = [path for path, digest in hashes.items() if sha256_file(Path(path)) != digest]
    result = {
        "status": "COMPLETED" if not changed else "INVALIDATED", "input_hashes": hashes,
        "changed_inputs": changed, "results": records,
        "scope": "post-hoc availability-based strata of frozen H2; predictive memory evidence, not navigation",
        "unit": "episode after averaging its windows/tracks, then fixed model seeds",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if changed:
        raise RuntimeError("H2 source results changed during audit")


if __name__ == "__main__":
    main()
