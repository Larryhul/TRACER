from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def paired_contrast(first, second, metric):
    first_rows = {row["episode"]: row for row in first["episode_records"]}
    second_rows = {row["episode"]: row for row in second["episode_records"]}
    if first_rows.keys() != second_rows.keys():
        raise ValueError("paired episode IDs do not match")
    delta = np.array([second_rows[index][metric] - first_rows[index][metric] for index in sorted(first_rows)])
    if not np.isfinite(delta).all():
        raise ValueError("paired contrast requires finite episode outcomes")
    generator = np.random.default_rng(958731)
    draws = delta[generator.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {"mean_after_minus_before": float(delta.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(), "episodes": len(delta)}


def main():
    h2_root = Path("experiments/h2_h1_transfer_validation_20260905")
    h3_root = Path("experiments/h3_transfer_validation_20260905")
    goal_root = Path("experiments/h3_goal_support_validation_20260905")
    records = {"phase": "development_validation", "model_seeds": [2027], "h2": {}, "h3": {}, "goal_support": {}}
    lines = ["# H2/H3 correctness repair and validation — 2026-09-05", "",
             "## Boundaries", "",
             "These are development results, not a new formal test or a SICNav non-inferiority claim.",
             "Old cross-policy shared-pool H3 results are invalid as closed-loop performance controls.",
             "The frozen old H2 control is rerun under the state-consistent protocol.",
             "All H2 variants train for 8 epochs on 512 new episodes (data seed 718731), with 128 ID-validation and 128 OOD-development episodes.",
             "H3 uses 16 episodes per distribution (data seed 658731), eight windows, and model seed 2027.",
             "H1 transfer copies the encoder only, not its response decoder or factorial training objective; it also uses additional H1 pretraining data/compute.",
             "The validation-selected temperature is 0.5 for all three new H2 variants; the legacy control retains 0.05.", "",
             "## H2 calibrated validation", "",
             "| Model | Split | Predictive NLL | ADE | Decision-mode NLL | Final mode accuracy |",
             "|---|---|---:|---:|---:|---:|"]
    for variant in ("legacy_scratch", "enhanced_scratch", "h1_encoder"):
        records["h2"][variant] = {}
        for split, filename in (("id", "metrics.json"), ("ood", "ood_metrics.json")):
            payload = json.loads((h2_root / variant / "evaluation_calibrated" / filename).read_text(encoding="utf-8"))
            row = payload["persistent"]
            records["h2"][variant][split] = payload
            lines.append(f"| {variant} | {split} | {row['predictive_trajectory_nll']:.5f} | {row['trajectory_ade']:.5f} | {row['decision_mode_nll']:.5f} | {row['final_mode_accuracy']:.4f} |")
    lines += ["", "Persistent belief improves mode accuracy and ADE over reset here, but does not beat reset on predictive NLL; retain that negative result.", "",
              "## Independent closed-loop development", "",
              "| Variant | Split | Cost (lower) | Progress | Collision/window | Belief flip | Same-state uniform-minus-selected gain |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for variant in ("frozen_control", "legacy_scratch", "legacy_scratch_calibrated", "enhanced_scratch", "enhanced_scratch_calibrated", "h1_encoder", "h1_encoder_calibrated", "h1_encoder_prefix"):
        runs = json.loads((h3_root / variant / "runs.json").read_text(encoding="utf-8"))["2027"]
        audit = json.loads((h3_root / variant / "candidate_protocol.json").read_text(encoding="utf-8"))
        records["h3"][variant] = {"audit": audit, "splits": {}}
        for split in ("id", "ood"):
            row = runs[split]["persistent-joint"]
            records["h3"][variant]["splits"][split] = {key: value for key, value in row.items() if isinstance(value, dict) and "mean" in value}
            values = [row[key]["mean"] for key in ("realized_total_cost", "task_progress", "collision_rate", "belief_selection_flip", "belief_selection_cost_delta")]
            lines.append(f"| {variant} | {split} | " + " | ".join(f"{value:.5f}" for value in values) + " |")
    lines += ["", "Calibration increases useful selection changes, but average closed-loop cost is mixed across distributions.",
              "Shortening all response costs/safety to the executed prefix increases progress but worsens cost and introduces a collision in OOD; it is not selected as a performance fix.", "",
              "## Fixed-goal candidate support", "",
              "The support variant enables goal-distance prefiltering, stopping at the goal, and anchor speed scales [0.4, 0.8, 1.0].",
              "Candidate budget stays 16. This is a combined candidate-support ablation, not an isolated stopping-effect estimate or a belief-only improvement.", "",
              "| Model | Split | Old success | Support success | Old distance | Support distance | Support collision/window |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for variant, source in (("control", "fixed_goal_control"), ("transfer", "fixed_goal_transfer")):
        before = json.loads((h3_root / source / "runs.json").read_text(encoding="utf-8"))["2027"]
        after = json.loads((goal_root / variant / "runs.json").read_text(encoding="utf-8"))["2027"]
        records["goal_support"][variant] = {}
        for split in ("id", "ood"):
            first, second = before[split]["persistent-joint"], after[split]["persistent-joint"]
            records["goal_support"][variant][split] = {
                metric: paired_contrast(first, second, metric)
                for metric in ("terminal_goal_success", "terminal_goal_distance", "realized_total_cost", "episode_collision")
            }
            values = [first["terminal_goal_success"]["mean"], second["terminal_goal_success"]["mean"], first["terminal_goal_distance"]["mean"], second["terminal_goal_distance"]["mean"], second["collision_rate"]["mean"]]
            lines.append(f"| {variant} | {split} | " + " | ".join(f"{value:.5f}" for value in values) + " |")
    lines += ["", "## Remaining requirements", "",
              "- The external paired FRM duplicates one distribution into neutral belief slots; it cannot establish external persistent-belief benefit.",
              "- SICNav-NP uses a unicycle with speed/rotation/acceleration constraints; native TRACER currently uses holonomic velocity controls. Initial scenes match in a reset probe, but this does not make the control protocols equivalent.",
              "- External budget exhaustion and collision-free success need consistent reporting; the navigation-contract patch addresses these fields in new runs only.",
              "- A final comparison requires frozen dynamics, observations, budgets, baseline settings and a scientifically justified non-inferiority margin, plus paired uncertainty on fresh cases.",
              "- Do not replace these remaining requirements with synthetic task cost, a favourable single seed, or reduced fallback frequency."]
    report = Path("reports/state_consistent_repair_validation_20260905")
    report.with_suffix(".json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    report.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
