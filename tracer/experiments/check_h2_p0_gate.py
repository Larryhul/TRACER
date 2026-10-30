from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _contrast(
    analysis: dict[str, Any], split: str, contrast: str, metric: str
) -> dict[str, Any]:
    for record in analysis["contrasts"]:
        if (
            record["scope"] == "all-post-initial"
            and record["split"] == split
            and record["contrast"] == contrast
            and record["metric"] == metric
        ):
            return record
    raise KeyError((split, contrast, metric))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the post-pilot H2 expansion gate.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    ood_metrics = json.loads(
        (run_dir / "ood_metrics.json").read_text(encoding="utf-8")
    )
    id_reset = _contrast(
        analysis, "id", "reset-minus-persistent", "trajectory_ade"
    )
    id_recurrent = _contrast(
        analysis, "id", "recurrent-minus-persistent", "trajectory_ade"
    )
    id_shuffled = _contrast(
        analysis, "id", "shuffled-id-minus-persistent", "trajectory_ade"
    )
    ood_reset = _contrast(
        analysis, "ood", "reset-minus-persistent", "trajectory_ade"
    )
    id_neutral = metrics["persistent"]["likelihood_mode_accuracy_by_true_mode"][
        "neutral"
    ]
    ood_neutral = ood_metrics["persistent"][
        "likelihood_mode_accuracy_by_true_mode"
    ]["neutral"]
    checks = {
        "window0_identical": analysis["window0_sanity"][
            "max_episode_sum_nll_difference"
        ]
        <= 1e-6,
        "id_reset_ci_positive": id_reset["bootstrap_ci95_low"] > 0.0,
        "id_shuffled_ci_positive": id_shuffled["bootstrap_ci95_low"] > 0.0,
        "id_matched_recurrent_not_better": id_recurrent["mean"] >= 0.0,
        "ood_reset_direction_positive": ood_reset["mean"] > 0.0,
        "id_neutral_above_chance": id_neutral > (1.0 / 3.0),
        "ood_neutral_above_chance": ood_neutral > (1.0 / 3.0),
        "id_oracle_ade_better_than_uniform": (
            metrics["oracle"]["trajectory_ade"]
            < metrics["no-belief"]["trajectory_ade"]
        ),
        "expected_main_count": metrics["persistent"]["main_evaluation_count"]
        == 64 * 3 * 7,
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "rule_status": "post-pilot operational expansion rule; not preregistered",
        "checks": checks,
        "diagnostics": {
            "id_reset_minus_persistent_ade": id_reset,
            "id_recurrent_minus_persistent_ade": id_recurrent,
            "id_shuffled_minus_persistent_ade": id_shuffled,
            "ood_reset_minus_persistent_ade": ood_reset,
            "id_neutral_likelihood_accuracy": id_neutral,
            "ood_neutral_likelihood_accuracy": ood_neutral,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"h2_p0_gate={payload['status']} output={output}")
    raise SystemExit(0 if payload["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
