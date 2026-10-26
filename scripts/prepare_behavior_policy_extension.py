from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-start", type=int, default=1308)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--val-start", type=int, default=1340)
    parser.add_argument("--val-count", type=int, default=32)
    args = parser.parse_args()

    base_path = Path(args.base).resolve()
    output_path = Path(args.output).resolve()
    splits = json.loads(base_path.read_text(encoding="utf-8"))
    additions = {
        "train": list(range(args.train_start, args.train_start + args.train_count)),
        "val": list(range(args.val_start, args.val_start + args.val_count)),
    }
    for name, values in additions.items():
        if set(values) & set(splits[name]):
            raise ValueError(f"new {name} cases overlap the existing split")
        if set(values) & set(splits["test"]):
            raise ValueError(f"new {name} cases overlap the frozen test split")
        splits[name] = sorted(set(splits[name]) | set(values))
    all_cases = [case for values in splits.values() for case in values]
    if len(all_cases) != len(set(all_cases)):
        raise ValueError("expanded case splits are not disjoint")
    output_path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: len(values) for name, values in splits.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
