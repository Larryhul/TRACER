"""Re-evaluate mode-switch attribution with scene-paired frozen checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tracer.experiments.closed_loop_protocol import prepare_manifest
from tracer.experiments.eval_component_factorial import _local_sources
from tracer.experiments.eval_mode_switch_benchmark import _evaluate_split
from tracer.experiments.formal_manifest import sha256_file
from tracer.response.persistent_trajectory import PersistentTrajectorySpec


def aggregate_switch_contrasts(rows, resamples=4000):
    indexed = {}
    for row in rows:
        key = (row["model_seed"], row["episode"], row["phase"], row["method"])
        if key in indexed:
            raise ValueError("duplicate mode-switch scene record")
        indexed[key] = row
    seeds = sorted({row["model_seed"] for row in rows})
    episodes = sorted({row["episode"] for row in rows})
    if not seeds or not episodes:
        raise ValueError("mode-switch attribution requires nonempty paired scenes")
    results = []
    for phase in ("all", "pre_switch", "post_switch"):
        for comparator in ("reset", "recurrent", "no-belief"):
            for metric in ("trajectory_ade", "trajectory_fde", "predictive_nll", "decision_brier", "posterior_brier", "decision_mode_accuracy", "posterior_mode_accuracy", "adaptation_delay_windows"):
                if metric == "adaptation_delay_windows" and phase != "post_switch":
                    continue
                try:
                    values = np.asarray([[float(indexed[seed, episode, phase, "persistent"][metric]) - float(indexed[seed, episode, phase, comparator][metric]) for episode in episodes] for seed in seeds])
                except KeyError as error:
                    raise ValueError("mode-switch seeds/methods/scenes must align") from error
                if not np.isfinite(values).all():
                    raise ValueError("mode-switch contrasts must be finite")
                scene_means = values.mean(0)
                generator = np.random.default_rng(924641)
                bootstrap = scene_means[generator.integers(0, len(episodes), size=(resamples, len(episodes)))].mean(1)
                results.append({"phase": phase, "contrast": f"persistent_minus_{comparator}", "metric": metric,
                                "mean_difference": float(scene_means.mean()), "scene_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
                                "independent_episodes": len(episodes), "model_seeds": seeds,
                                "per_model_seed": {str(seed): float(values[index].mean()) for index, seed in enumerate(seeds)}})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--data-seed", type=int, default=418731)
    parser.add_argument("--likelihood-scale", type=float, default=.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.episodes < 1 or len(set(args.seeds)) != len(args.seeds) or not np.isfinite(args.likelihood_scale) or args.likelihood_scale <= 0:
        raise ValueError("positive episodes/scale and unique training seeds are required")
    torch.set_num_threads(2)
    root, output = Path(args.checkpoint_root).resolve(), Path(args.output_dir)
    checkpoints = [root / f"seed{seed}" / name for seed in args.seeds for name in ("response.pt", "recurrent.pt")]
    manifest = prepare_manifest(output, {"arguments": vars(args), "scope": "historical mode-switch simulator protocol with the already selected common likelihood scale; descriptive frozen-baseline comparison, not new confirmation or independently retuned baselines", "switch_window": 4, "decision_evidence": "executed prefix only, no current response in decision prior"}, checkpoints, _local_sources(), False)
    results = {}
    for split in ("id", "ood"):
        all_rows, method_rows = [], []
        for seed in args.seeds:
            spec = PersistentTrajectorySpec(split=split, num_episodes=args.episodes, windows=8, num_robots=3, seed=args.data_seed,
                                           action_policy="iid" if split == "id" else "correlated", scene_family="radial", response_family="pincer", cache_samples=False)
            episode_rows = []
            summary = _evaluate_split(root / f"seed{seed}", spec, args.episodes, 4, args.device, args.likelihood_scale, episode_records=episode_rows)
            all_rows.extend({**row, "model_seed": seed} for row in episode_rows)
            method_rows.extend({**row, "model_seed": seed} for row in summary)
            print(split, seed, {row["method"]: row["trajectory_ade"] for row in summary}, flush=True)
        (output / f"{split}_episodes.json").write_text(json.dumps(all_rows, allow_nan=False) + "\n")
        (output / f"{split}_methods.json").write_text(json.dumps(method_rows, indent=2, allow_nan=False) + "\n")
        results[split] = aggregate_switch_contrasts(all_rows)
    changed = [path for group in ("sources", "checkpoints") for path, digest in manifest[group].items() if sha256_file(Path(path)) != digest]
    summary = {"status": "INVALIDATED" if changed else "COMPLETED", "changed_inputs": changed, "results": results,
               "unit": "base episode after averaging fixed training seeds", "scope": manifest["config"]["scope"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if changed:
        raise RuntimeError("mode-switch audit inputs changed")


if __name__ == "__main__":
    main()
