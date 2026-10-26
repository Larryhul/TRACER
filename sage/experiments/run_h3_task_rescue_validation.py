from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ANCHOR_FAMILIES = (
    (0.4,),
    (0.4, 0.7),
    (0.4, 0.7, 1.0),
    (0.4, 0.7, 1.0, 1.3),
)


def _selection_divergence(runs: dict[str, Any], split: str) -> float:
    seed_run = next(iter(runs.values()))[split]
    persistent = seed_run["persistent-joint"]["selected_candidates"]
    reset = seed_run["reset-joint"]["selected_candidates"]
    changed = [
        first != second
        for first, second in zip(persistent, reset, strict=True)
    ]
    return sum(changed) / len(changed)


def _review_setting(output_dir: Path) -> dict[str, Any]:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    runs = json.loads((output_dir / "runs.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    values: dict[str, Any] = {}
    for split in ("id", "ood"):
        persistent = summary[split]["persistent-joint"]
        reset = summary[split]["reset-joint"]
        task_only = summary[split]["task-only"]
        persistent_task = float(persistent["task_progress"]["mean"])
        reset_task = float(reset["task_progress"]["mean"])
        persistent_collision = float(persistent["collision_rate"]["mean"])
        reset_collision = float(reset["collision_rate"]["mean"])
        divergence = _selection_divergence(runs, split)
        values[split] = {
            "persistent_task_progress": persistent_task,
            "reset_task_progress": reset_task,
            "task_only_task_progress": float(task_only["task_progress"]["mean"]),
            "persistent_collision_rate": persistent_collision,
            "reset_collision_rate": reset_collision,
            "persistent_reset_rollout_divergence_rate": divergence,
        }
        checks[f"{split}_persistent_task_positive"] = persistent_task > 0.0
        checks[f"{split}_task_only_task_positive"] = values[split]["task_only_task_progress"] > 0.0
        checks[f"{split}_persistent_task_noninferior_to_reset_2pct"] = (
            persistent_task >= 0.98 * reset_task
        )
        checks[f"{split}_persistent_collision_not_worse_by_0_02"] = (
            persistent_collision <= reset_collision + 0.02
        )
        checks[f"{split}_persistent_changes_rollout_vs_reset"] = divergence > 0.0
    return {"passed": all(checks.values()), "checks": checks, "values": values}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only task-anchor selection for the corrected H3 rescue."
    )
    parser.add_argument(
        "--base-config", default="configs/persistent_closed_loop_formal_corrected.yaml"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--model-seed", type=int, default=2027)
    parser.add_argument("--data-seed", type=int, default=528731)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    base.pop("formal_protocol", None)
    base["simulation"]["episodes"] = args.episodes
    base["simulation"]["data_seed"] = args.data_seed
    base["suite"]["seeds"] = [args.model_seed]
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    reviews: list[dict[str, Any]] = []
    for index, scales in enumerate(ANCHOR_FAMILIES, start=1):
        raw = json.loads(json.dumps(base))
        raw["candidate_source"]["include_task_anchor"] = True
        raw["candidate_source"]["task_anchor_speed_scales"] = list(scales)
        setting_dir = root / f"setting{index}"
        setting_dir.mkdir(parents=True, exist_ok=True)
        config_path = setting_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "sage.experiments.eval_persistent_closed_loop",
                "--config",
                str(config_path),
                "--device",
                args.device,
                "--seeds",
                str(args.model_seed),
                "--episodes",
                str(args.episodes),
                "--output-dir",
                str(setting_dir),
            ],
            check=True,
        )
        review = {
            "setting": index,
            "task_anchor_speed_scales": list(scales),
            **_review_setting(setting_dir),
        }
        reviews.append(review)
        print(json.dumps(review, sort_keys=True), flush=True)

    feasible = [review for review in reviews if review["passed"]]
    selected = feasible[0] if feasible else None
    payload = {
        "selection_scope": "validation_only_post_failure_rescue",
        "base_config": args.base_config,
        "episodes": args.episodes,
        "model_seed": args.model_seed,
        "data_seed": args.data_seed,
        "selection_rule": "first feasible anchor family in ascending intervention size",
        "settings": reviews,
        "selected": selected,
        "passed": selected is not None,
    }
    (root / "selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": payload["passed"], "selected": selected}, sort_keys=True))
    raise SystemExit(0 if payload["passed"] else 2)


if __name__ == "__main__":
    main()
