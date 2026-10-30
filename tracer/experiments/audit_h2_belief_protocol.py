"""Independent audit of the H2 belief implementation and evaluation protocol.

This module is deliberately read-only with respect to the existing H2 runner.
It consumes archived raw rows and summaries, separates decision-time belief
from the posterior after the current observation, and reports protocol risks
that can otherwise be hidden by aggregate predictive metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in archived row")
    return value


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return None if not values else float(statistics.fmean(values))


def _ece(confidence: list[float], correct: list[float], bins: int = 10) -> float | None:
    if not confidence:
        return None
    if len(confidence) != len(correct):
        raise ValueError("confidence and correct must have the same length")
    total = len(confidence)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            position
            for position, value in enumerate(confidence)
            if (value >= lower and value < upper)
            or (index == bins - 1 and value <= upper and value >= lower)
        ]
        if not selected:
            continue
        mean_confidence = statistics.fmean(confidence[position] for position in selected)
        mean_accuracy = statistics.fmean(correct[position] for position in selected)
        error += len(selected) / total * abs(mean_confidence - mean_accuracy)
    return float(error)


def _row_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    active = [row for row in rows if int(row["active"])]
    post_initial = [row for row in active if int(row["window"]) > 0]

    def metric(items: list[dict[str, str]], key: str) -> float | None:
        return _mean(_float(row, key) for row in items)

    def nll(items: list[dict[str, str]], key: str) -> float | None:
        return _mean(-math.log(max(_float(row, key), 1e-12)) for row in items)

    def ece(items: list[dict[str, str]], confidence_key: str, correct_key: str) -> float | None:
        return _ece(
            [_float(row, confidence_key) for row in items],
            [_float(row, correct_key) for row in items],
        )

    summary: dict[str, Any] = {
        "rows": len(rows),
        "active_rows": len(active),
        "post_initial_rows": len(post_initial),
        "decision_accuracy_all_active": metric(active, "decision_mode_correct"),
        "posterior_accuracy_all_active": metric(active, "posterior_mode_correct"),
        "decision_accuracy_post_initial": metric(post_initial, "decision_mode_correct"),
        "posterior_accuracy_post_initial": metric(post_initial, "posterior_mode_correct"),
        "decision_nll_post_initial": nll(post_initial, "decision_true_probability"),
        "posterior_nll_post_initial": nll(post_initial, "posterior_true_probability"),
        "predictive_trajectory_nll_post_initial": metric(
            post_initial, "predictive_trajectory_nll"
        ),
        "decision_confidence_post_initial": metric(post_initial, "decision_confidence"),
        "posterior_confidence_post_initial": metric(post_initial, "posterior_confidence"),
        "decision_ece_post_initial": ece(
            post_initial, "decision_confidence", "decision_mode_correct"
        ),
        "posterior_ece_post_initial": ece(
            post_initial, "posterior_confidence", "posterior_mode_correct"
        ),
    }
    decision_nll = summary["decision_nll_post_initial"]
    posterior_nll = summary["posterior_nll_post_initial"]
    decision_accuracy = summary["decision_accuracy_post_initial"]
    posterior_accuracy = summary["posterior_accuracy_post_initial"]
    summary["posterior_minus_decision_nll"] = (
        None if decision_nll is None or posterior_nll is None else posterior_nll - decision_nll
    )
    summary["posterior_minus_decision_accuracy"] = (
        None
        if decision_accuracy is None or posterior_accuracy is None
        else posterior_accuracy - decision_accuracy
    )
    summary["window_metrics"] = _window_summary(active)
    return summary


def _window_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["window"])].append(row)
    output = []
    for window in sorted(grouped):
        items = grouped[window]
        output.append(
            {
                "window": window,
                "count": len(items),
                "decision_accuracy": _mean(
                    _float(row, "decision_mode_correct") for row in items
                ),
                "posterior_accuracy": _mean(
                    _float(row, "posterior_mode_correct") for row in items
                ),
                "decision_confidence": _mean(
                    _float(row, "decision_confidence") for row in items
                ),
                "posterior_confidence": _mean(
                    _float(row, "posterior_confidence") for row in items
                ),
                "predictive_trajectory_nll": _mean(
                    _float(row, "predictive_trajectory_nll") for row in items
                ),
            }
        )
    return output


def audit_raw_metrics(root: Path) -> dict[str, Any]:
    seed_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for seed_dir in sorted(root.glob("seed*")):
        if not seed_dir.is_dir():
            continue
        path = seed_dir / "raw_metrics.csv"
        if not path.exists():
            continue
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                grouped[(row["split"], row["method"])].append(row)
        seed_summaries[seed_dir.name] = {
            f"{split}/{method}": _row_summary(rows)
            for (split, method), rows in sorted(grouped.items())
        }

    aggregate: dict[str, dict[str, Any]] = {}
    for key in sorted({key for values in seed_summaries.values() for key in values}):
        values = [seed[key] for seed in seed_summaries.values() if key in seed]
        aggregate[key] = {
            "seed_count": len(values),
            "metrics": {
                metric: {
                    "mean": _mean(
                        value[metric]
                        for value in values
                        if isinstance(value.get(metric), (int, float))
                    ),
                    "sd": _sd(
                        value[metric]
                        for value in values
                        if isinstance(value.get(metric), (int, float))
                    ),
                }
                for metric in (
                    "decision_accuracy_post_initial",
                    "posterior_accuracy_post_initial",
                    "decision_nll_post_initial",
                    "posterior_nll_post_initial",
                    "predictive_trajectory_nll_post_initial",
                    "decision_confidence_post_initial",
                    "posterior_confidence_post_initial",
                    "decision_ece_post_initial",
                    "posterior_ece_post_initial",
                    "posterior_minus_decision_nll",
                    "posterior_minus_decision_accuracy",
                )
            },
        }
    return {"seeds": seed_summaries, "aggregate": aggregate}


def _sd(values: Iterable[float]) -> float | None:
    values = list(values)
    return None if len(values) < 2 else float(statistics.stdev(values))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_sensitivity(path: Path, fields: tuple[str, ...]) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    rows = _read_csv(path)
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in fields)
        value = float(row.get("nll_mean", row.get("mean", "nan")))
        if math.isfinite(value):
            grouped[key].append(value)
    table = []
    for key, values in sorted(grouped.items()):
        table.append({**dict(zip(fields, key)), "values": values})
    return {"rows": table}


def audit_switch(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    persistent = summary.get("persistent", {})
    reset = summary.get("reset", {})
    recurrent = summary.get("recurrent", {})
    return {
        "persistent": persistent,
        "reset": reset,
        "recurrent": recurrent,
        "persistent_minus_reset_adaptation_delay": _difference(
            persistent.get("adaptation_delay_windows"), reset.get("adaptation_delay_windows")
        ),
        "persistent_minus_reset_success_rate": _difference(
            persistent.get("adaptation_success_rate"), reset.get("adaptation_success_rate")
        ),
    }


def _difference(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return float(left - right)


def _lookup(aggregate: dict[str, Any], key: str, metric: str) -> float | None:
    value = aggregate.get(key, {}).get("metrics", {}).get(metric, {}).get("mean")
    return None if value is None else float(value)


def build_audit(root: Path, repo_root: Path) -> dict[str, Any]:
    raw = audit_raw_metrics(root)
    scale = audit_sensitivity(
        repo_root / "reports/h2_prefix_scale_sensitivity_2026-08-23/summary.csv",
        ("prefix", "scale", "split", "contrast"),
    )
    prefix = audit_sensitivity(
        repo_root / "reports/h2_prefix_sensitivity_2026-08-21/prefix_summary.csv",
        ("prefix", "split", "contrast"),
    )
    calibration_path = repo_root / "reports/h2_calibration_2026-08-20/calibration.csv"
    calibration = _read_csv(calibration_path) if calibration_path.exists() else []
    switch = audit_switch(
        repo_root / "experiments/mode_switch_benchmark_20260824_id/summary.json"
    )

    aggregate = raw["aggregate"]
    persistent_id = aggregate.get("id/persistent", {}).get("metrics", {})
    persistent_ood = aggregate.get("ood/persistent", {}).get("metrics", {})
    reset_id = aggregate.get("id/reset", {}).get("metrics", {})
    findings = {
        "calibration_failure": {
            "persistent_id_ece": _lookup(aggregate, "id/persistent", "posterior_ece_post_initial"),
            "persistent_ood_ece": _lookup(aggregate, "ood/persistent", "posterior_ece_post_initial"),
            "recurrent_id_ece": _lookup(aggregate, "id/recurrent", "posterior_ece_post_initial"),
            "persistent_id_confidence_minus_accuracy": _difference(
                _lookup(aggregate, "id/persistent", "posterior_confidence_post_initial"),
                _lookup(aggregate, "id/persistent", "posterior_accuracy_post_initial"),
            ),
        },
        "posterior_is_not_decision": {
            "persistent_id_posterior_minus_decision_nll": _lookup(
                aggregate, "id/persistent", "posterior_minus_decision_nll"
            ),
            "persistent_id_posterior_minus_decision_accuracy": _lookup(
                aggregate, "id/persistent", "posterior_minus_decision_accuracy"
            ),
            "reset_id_posterior_minus_decision_nll": _lookup(
                aggregate, "id/reset", "posterior_minus_decision_nll"
            ),
        },
        "persistent_gain_is_small": {
            "reset_minus_persistent_predictive_nll": _difference(
                _lookup(aggregate, "id/reset", "predictive_trajectory_nll_post_initial"),
                _lookup(aggregate, "id/persistent", "predictive_trajectory_nll_post_initial"),
            ),
            "persistent_vs_reset_decision_nll": _difference(
                _lookup(aggregate, "id/persistent", "decision_nll_post_initial"),
                _lookup(aggregate, "id/reset", "decision_nll_post_initial"),
            ),
        },
        "switch_failure": {
            "persistent_minus_reset_adaptation_delay": switch.get(
                "persistent_minus_reset_adaptation_delay"
            ),
            "persistent_minus_reset_success_rate": switch.get(
                "persistent_minus_reset_success_rate"
            ),
        },
        "scale_or_prefix_sensitivity": {
            "scale_rows": scale.get("rows", []),
            "prefix_rows": prefix.get("rows", []),
        },
    }
    return {
        "protocol": {
            "formal_root": str(root),
            "forbidden_files_modified": [
                "tracer/belief/filter.py",
                "tracer/experiments/run_sicnav_tracer_loop.py",
                "tracer/experiments/eval_persistent_closed_loop.py",
            ],
            "raw_rows_are_audit_evidence_not_new_confirmatory_results": True,
        },
        "artifacts": {
            "raw_metrics": raw,
            "calibration_archived": calibration,
            "scale_sensitivity": scale,
            "prefix_sensitivity": prefix,
            "mode_switch": switch,
        },
        "findings": findings,
        "repair_plan": [
            {
                "priority": "P0",
                "action": "Use decision-time belief for every H3 action and predictive metric; keep post-observation posterior only for filtering diagnostics.",
                "reason": "posterior metrics consume the current window observation and are not the belief available before choosing that window's action.",
            },
            {
                "priority": "P0",
                "action": "Calibrate likelihood temperature on validation only and archive raw prefix log-likelihoods, per-coordinate normalization, and the selected temperature.",
                "reason": "the fixed 0.05 scale changes the filter but predictive NLL is evaluated with unscaled likelihoods, so the reported decision objective and update objective are not the same probabilistic model.",
            },
            {
                "priority": "P1",
                "action": "Evaluate stationary identity, re-entry, and explicit mode-switch episodes as separate H2 axes; select hazard/forgetting only on switch validation episodes.",
                "reason": "zero-hazard persistence is appropriate for stationary identity continuity but is expected to adapt poorly after a real behavior change.",
            },
            {
                "priority": "P1",
                "action": "Require a belief-attribution gate in H3: same state, same candidate pool, and report selected-candidate/action change plus paired regret before claiming a belief gain.",
                "reason": "the external audit currently shows near-identical actions across belief policies, so H3 gains cannot be attributed to Persistent belief.",
            },
        ],
        "source_map": {
            "filter": "tracer/belief/filter.py:21-35,62-71,96-157",
            "formal_evaluation": "tracer/experiments/train_persistent_trajectory_response.py:420-588",
            "trajectory_likelihood": "tracer/response/persistent_models.py:970-985",
            "metrics": "tracer/experiments/train_persistent_trajectory_response.py:170-221",
            "mode_switch": "tracer/experiments/eval_mode_switch_benchmark.py:1-7,261-344",
            "calibration_report": "reports/h2_calibration_2026-08-20/calibration.md",
            "unified_report": "reports/h2_unified_filter_ablation_20260904.md",
            "hazard_report": "reports/h2_bayesian_hazard_audit_20260905/summary.md",
        },
    }


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    findings = audit["findings"]
    calibration = findings["calibration_failure"]
    timing = findings["posterior_is_not_decision"]
    gain = findings["persistent_gain_is_small"]
    switch = findings["switch_failure"]
    lines = [
        "# Independent H2 Persistent Belief Audit",
        "",
        "Date: 2026-09-06. This is an independent protocol audit over archived H2 artifacts; it does not modify or rerun the three protected core files.",
        "",
        "## Executive verdict",
        "",
        "Persistent belief is useful as a stationary identity-memory baseline, but the current H2 evidence does not validate it as a calibrated or switch-robust belief module, and it does not establish that the H3 gain came from belief.",
        "",
        "## Findings",
        "",
        f"- **Calibration:** Persistent posterior ECE is {_format(calibration['persistent_id_ece'])} (ID) and {_format(calibration['persistent_ood_ece'])} (OOD), versus recurrent {_format(calibration['recurrent_id_ece'])}; posterior confidence minus accuracy is {_format(calibration['persistent_id_confidence_minus_accuracy'])}.",
        f"- **Timing:** Persistent posterior-minus-decision NLL is {_format(timing['persistent_id_posterior_minus_decision_nll'])}; posterior-minus-decision accuracy is {_format(timing['persistent_id_posterior_minus_decision_accuracy'])}. The after-update posterior is not the belief available before the current action.",
        f"- **Incremental value:** Reset-minus-Persistent predictive NLL is only {_format(gain['reset_minus_persistent_predictive_nll'])} in this row-weighted raw audit; the external H3 audit must therefore show a policy/action change before attributing any closed-loop gain to belief.",
        f"- **Switch adaptation:** Persistent minus Reset adaptation delay is {_format(switch['persistent_minus_reset_adaptation_delay'])} windows and success-rate difference is {_format(switch['persistent_minus_reset_success_rate'])}; the archived mode-switch benchmark directly contradicts a claim that zero-hazard persistence is robust to behavior changes.",
        "- **Scale/prefix sensitivity:** the archived sensitivity tables cross from positive to negative Reset-minus-Persistent contrasts as scale/prefix changes, so `0.05` is a protocol choice, not a demonstrated calibrated likelihood temperature.",
        "",
        "## Root causes",
        "",
        "1. The filter performs a mathematically valid log-space Bayes update, but its inputs are learned trajectory log-likelihoods whose calibration is not validated. The fixed `likelihood_scale` is applied before the filter, while predictive response NLL is computed from the unscaled likelihood; update and scoring therefore use different effective models.",
        "2. Persistent is defined as zero-hazard accumulation across all windows. That tests stationary identity continuity, not general adaptive belief. It intentionally retains stale mass after a latent mode switch.",
        "3. The evaluation reports both pre-update `decision_*` and post-update `posterior_*`, but the headline belief metrics are primarily posterior metrics. This makes the filter look stronger after consuming the current observation, while H3 planning must use the decision belief from the previous window.",
        "4. The current H2 trajectory dataset has a stable latent mode. The mode-switch benchmark is separate and frozen, so the main Persistent score cannot support a claim about switch adaptation or nonstationarity.",
        "5. The archived external H3 evidence uses duplicated/weak external response modes in some settings and shows little policy separation. Better navigation in such a run is not evidence for Persistent belief.",
        "",
        "## Today’s landing plan",
        "",
        "- **P0:** keep the protected runners unchanged; use this audit to label decision-time versus posterior-time metrics explicitly in the experiment outputs and paper tables.",
        "- **P0:** run validation-only temperature/scale selection over the existing prefix sensitivity grid, with raw likelihoods normalized per observed coordinate; freeze the selected value before test/OOD.",
        "- **P1:** run a same-checkpoint matrix: zero-hazard Persistent, Reset, fixed-hazard, and adaptive Persistent on stationary, re-entry, and mode-switch episodes. Report decision NLL/Brier/ECE, adaptation delay, and post-switch recovery.",
        "- **P1:** for H3, require same-state candidate/action attribution: selected candidate changed, executed prefix changed, response-cost regret, and collision/CFS. If these remain zero, report H3 as controller/safety improvement rather than belief improvement.",
        "",
        "## Evidence map",
        "",
        "- `reports/h2_unified_filter_ablation_20260904.md` reports the small stationary Persistent gain and explicitly records poor calibration.",
        "- `reports/h2_bayesian_hazard_audit_20260905/summary.md` records the fixed-hazard audit boundary and missing full probability vectors in the archived snapshot.",
        "- `reports/h2_calibration_2026-08-20/calibration.md` reports Persistent ECE `0.3812/0.3815` versus recurrent `0.0232/0.0264`.",
        "- `reports/h2_prefix_scale_sensitivity_2026-08-23/summary.md` shows scale-dependent sign changes in Reset-minus-Persistent contrasts.",
        "- `experiments/mode_switch_benchmark_20260824_id/summary.json` reports Persistent adaptation success `0.3945` versus Reset `0.7234`.",
        "- `reports/h2_h3_external_repair_audit_20260905.md` states that external mode differences were weak and that H3 improvements cannot currently be attributed to belief.",
        "",
        "## Protected files",
        "",
        "This audit does not modify `tracer/belief/filter.py`, `tracer/experiments/run_sicnav_tracer_loop.py`, or `tracer/experiments/eval_persistent_closed_loop.py`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        default="experiments/formal_v2/clean_chain_submission_20260904/h2",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/h2_independent_audit_20260906",
    )
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    audit = build_audit(Path(args.formal_root), Path(args.repo_root))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(output / "audit.md", audit)
    print(f"saved_json={output / 'audit.json'}")
    print(f"saved_report={output / 'audit.md'}")


if __name__ == "__main__":
    main()
