"""Aggregate multi-seed real-recording H1 response benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_metrics(root: Path, seed: int, split: str) -> dict:
    path = root / f"seed{seed}" / f"{split}_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(dataset: str, roots: list[Path], seeds: list[int], output: Path) -> dict:
    if not roots or len(roots) != len(seeds):
        raise ValueError("benchmark roots and seeds must be nonempty and have equal length")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds cannot count as independent runs")
    methods = ("history-only", "no-belief", "reset", "persistent", "recurrent")
    summary = {
        "dataset": dataset,
        "seeds": seeds,
        "benchmark_roots": [str(root) for root in roots],
        "splits": {},
    }
    markdown = [
        f"# Real H1 {dataset} Multi-Seed Summary",
        "",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`.",
        "",
        "H1 action ablation: compare no-belief against history-only. Comparing",
        "persistent against history-only changes both action conditioning and belief.",
        "Recorded robot futures are observational inputs, not exogenous interventions.",
        "",
    ]
    for split in ("val", "test"):
        split_summary = {}
        markdown.extend(
            [
                f"## {split}",
                "",
                "| Method | Response NLL | ADE | FDE |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in methods:
            values = []
            for root, seed in zip(roots, seeds):
                metrics = _load_metrics(root, seed, split)[method]
                values.append(
                    [
                        float(metrics["predictive_trajectory_nll"]),
                        float(metrics["trajectory_ade"]),
                        float(metrics["trajectory_fde"]),
                    ]
                )
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite metrics for {split}/{method}")
            mean = array.mean(axis=0)
            std = array.std(axis=0, ddof=1) if len(array) > 1 else np.zeros(3)
            split_summary[method] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "per_seed": array.tolist(),
            }
            markdown.append(
                f"| {method} | {mean[0]:.4f} +/- {std[0]:.4f} | "
                f"{mean[1]:.4f} +/- {std[1]:.4f} | "
                f"{mean[2]:.4f} +/- {std[2]:.4f} |"
            )
        history = np.asarray(split_summary["history-only"]["mean"])
        conditioned = np.asarray(split_summary["no-belief"]["mean"])
        persistent = np.asarray(split_summary["persistent"]["mean"])
        markdown.extend(
            [
                "",
                f"No-belief minus History-only: NLL `{conditioned[0] - history[0]:.4f}`, "
                f"ADE `{conditioned[1] - history[1]:.4f}`, "
                f"FDE `{conditioned[2] - history[2]:.4f}`.",
                "",
                f"Persistent minus History-only: NLL `{persistent[0] - history[0]:.4f}`, "
                f"ADE `{persistent[1] - history[1]:.4f}`, "
                f"FDE `{persistent[2] - history[2]:.4f}`.",
                "",
            ]
        )
        summary["splits"][split] = split_summary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True, nargs="+")
    parser.add_argument("--seed", type=int, required=True, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.benchmark_root) != len(args.seed):
        raise ValueError("--benchmark-root and --seed must have equal length")
    print(json.dumps(summarize(args.dataset, args.benchmark_root, args.seed, args.output), indent=2))


if __name__ == "__main__":
    main()
