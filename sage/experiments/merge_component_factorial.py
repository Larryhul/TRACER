import argparse
import copy
import json
from pathlib import Path
import shutil

from sage.experiments.audit_component_attribution import audit_directory
from sage.experiments.closed_loop_protocol import prepare_manifest
from sage.experiments.eval_component_factorial import summarize_factorial
from sage.experiments.formal_manifest import sha256_file


def compatible_shard_configuration(first, second):
    first, second = copy.deepcopy(first), copy.deepcopy(second)
    for configuration in (first, second):
        for key in ("seeds", "splits", "output_dir"):
            configuration["arguments"].pop(key, None)
    if first != second:
        raise ValueError("factorial shards have incompatible frozen configurations")


def merge_factorial_shards(directories, output):
    if not directories or len(set(Path(path).resolve() for path in directories)) != len(directories):
        raise ValueError("nonempty unique shard directories required")
    output = Path(output).resolve()
    merged = {"same_state": {}, "closed_loop": {}}
    expected_configuration, settings, audits, hashes, files = None, {}, {}, {}, {}
    for directory in directories:
        audited, runs = audit_directory(directory)
        configuration = audited["run_configuration"]
        if expected_configuration is None:
            expected_configuration = copy.deepcopy(configuration)
        else:
            compatible_shard_configuration(expected_configuration, configuration)
        for path, digest in audited["evidence_sha256"].items():
            if path in hashes and hashes[path] != digest:
                raise ValueError("shards disagree about an input digest")
            hashes[path] = digest
        for seed, setting in audited["effective_settings"].items():
            if seed in settings and settings[seed] != setting:
                raise ValueError("domain shards changed effective model settings")
            settings[seed] = setting
        for mode, seed_runs in runs.items():
            for seed, splits in seed_runs.items():
                for split, policies in splits.items():
                    target = merged[mode].setdefault(seed, {})
                    if split in target:
                        raise ValueError("duplicate seed/domain/evaluation shard")
                    target[split] = policies
                    filename = f"seed{seed}_{split}_{mode}.json"
                    files[filename] = Path(directory).resolve() / filename
        audits.update(audited["audits"])
    seeds = sorted(merged["same_state"], key=int)
    splits = sorted({split for domains in merged["same_state"].values() for split in domains})
    for mode in merged:
        if set(merged[mode]) != set(seeds) or any(set(domains) != set(splits) for domains in merged[mode].values()):
            raise ValueError("complete seed-by-domain factorial coverage required")
        merged[mode] = {seed: {split: merged[mode][seed][split] for split in splits} for seed in seeds}
    arguments = expected_configuration["arguments"]
    arguments.update(seeds=[int(seed) for seed in seeds], splits=splits, output_dir=str(output))
    expected_configuration["aggregation"] = {
        "shards": [str(Path(path).resolve()) for path in directories],
        "scope": "Exact episode-file merge of completed same-protocol seed/domain shards; no outcome-dependent filtering or averaging shard confidence intervals.",
    }
    manifest = prepare_manifest(output, expected_configuration, [Path(path) for path in sorted(hashes)],
                                [Path(__file__).resolve()], False)
    for filename, source in files.items():
        shutil.copyfile(source, output / filename)
        if sha256_file(output / filename) != hashes[str(source)]:
            raise ValueError("merged episode bytes differ from the frozen shard")
    additional = ("terminal_goal_distance", "terminal_goal_success") if expected_configuration["config"]["simulation"].get("task_type") == "fixed_goal" else ()
    results = {mode: summarize_factorial(runs, arguments["bootstrap_resamples"], additional_metrics=additional) for mode, runs in merged.items()}
    changed = [path for path, digest in hashes.items() if sha256_file(Path(path)) != digest]
    changed_sources = [path for path, digest in manifest["sources"].items() if sha256_file(Path(path)) != digest]
    summary = {
        "status": "INVALIDATED" if changed or changed_sources else "COMPLETED", "phase": arguments["phase"],
        "changed_sources": changed_sources, "changed_checkpoints": changed,
        "effective_settings": settings, "audits": audits, "results": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if changed or changed_sources:
        raise ValueError("shard evidence changed during merge")
    audit_directory(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Merge independently audited factorial shards without pseudoreplication")
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = merge_factorial_shards(args.shards, args.output_dir)
    print(json.dumps({"status": result["status"], "output_dir": str(Path(args.output_dir).resolve())}))


if __name__ == "__main__":
    main()
