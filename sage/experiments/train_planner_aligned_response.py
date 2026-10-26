from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sage.experiments.collect_external_planner_supervision import collection_cases, file_hash
from sage.experiments.eval_persistent_closed_loop import _load_model
from sage.experiments.run_sicnav_tracer_loop import _differentiable_response_predictions
from sage.response.closed_loop import PlanningCostConfig
from sage.response.planner_supervision import (
    add_action_conditioned_uncertainty, configure_planner_finetuning, entity_risk_calibration_loss,
    expected_prefix_regret_loss, group_regret, planner_scores, prefix_equivalence,
    planner_gap_calibration_loss, response_supervision, top_prefix_ranking_loss,
)


def load_groups(directory, expected_split):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "COMPLETED" or protocol["split"] != expected_split:
        raise ValueError("training requires completed, correctly labelled supervision")
    cases = collection_cases(protocol["case_splits"], expected_split)
    if protocol["cases"] != cases or manifest["cases"] != cases:
        raise ValueError("collected cases must exactly match the declared split")
    groups = []
    hashes = {str((directory / name).resolve()): file_hash(directory / name)
              for name in ("protocol.json", "manifest.json", "audit/windows.jsonl")}
    for line in (directory / "audit/windows.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["case"] not in cases:
            raise ValueError("supervision contains an out-of-split case")
        if row["status"] != "matched" or not row.get("live_prefix_verified"):
            continue
        if not all(row.get(f"{name}_supported") for name in ("uniform", "persistent", "true_mode")):
            continue
        name = f"seed{row['seed']}_case{row['case']}_window{row['window']}.npz"
        input_path, audit_path = directory / "inputs" / name, directory / "audit" / name
        hashes[str(input_path.resolve())] = file_hash(input_path)
        hashes[str(audit_path.resolve())] = file_hash(audit_path)
        with np.load(input_path, allow_pickle=False) as values:
            group = {key: values[key] for key in values.files}
        with np.load(audit_path, allow_pickle=False) as audit:
            snapshots = audit["snapshots"]
            distance = np.linalg.norm(snapshots[:, :, 0:1, :2] - snapshots[:, :, 1:, :2], axis=-1)
            active = audit["active"].copy()
            if protocol["parameters"].get("goal_terminal_response"):
                goal = audit["initial_snapshot"][0, 5:7]
                reached = np.linalg.norm(snapshots[:, :, 0, :2] - goal, axis=-1) < float(audit["robot_radius"])
                reached_before = np.concatenate([np.zeros((len(snapshots), 1), dtype=bool),
                                                  np.maximum.accumulate(reached, axis=1)[:, :-1]], axis=1)
                active &= ~reached_before
            risk = ((distance < protocol["planning"]["collision_margin"]) & active[:, :, None]).any(axis=1).T
            group.update(
                target=snapshots[:, :, 1:, 2:4].transpose(2, 0, 1, 3),
                reference_target=audit["reference_snapshots"][:, 1:, 2:4].transpose(1, 0, 2),
                target_active=audit["active"],
                reference_active=(audit["reference_active"] if "reference_active" in audit
                                  else audit["active"].any(axis=0)),
                factual=audit["factual_total"], eligible=audit["eligible"],
                factual_entity_risk=risk,
                same_prefix=prefix_equivalence(snapshots, protocol["execute_steps"]),
                labels=np.asarray(row["true_modes_offline_only"]),
                case=row["case"], window=row["window"], dt=protocol["dt"],
                teacher_scores={name: audit[f"score_{name}"] for name in ("uniform", "persistent", "true_mode")},
            )
        groups.append(group)
    if {group["case"] for group in groups} != set(cases):
        raise ValueError("every scene must contribute verified comparable supervision")
    return groups, protocol, hashes


def weights_for(group, means, name):
    humans, _, modes = means.shape[:3]
    if name == "uniform":
        return torch.full((humans, modes), 1 / modes, device=means.device, dtype=means.dtype)
    if name == "persistent":
        return torch.as_tensor(group["beliefs"], device=means.device, dtype=means.dtype).exp()
    if name == "true_mode":
        return torch.nn.functional.one_hot(
            torch.as_tensor(group["labels"], device=means.device), modes,
        ).to(means.dtype)
    raise ValueError("unknown planner belief intervention")


def predict_group(model, group, device):
    return _differentiable_response_predictions(
        model, group["histories"], group["current"], group["candidates"], device, group["static_obstacles"],
    )


@torch.no_grad()
def evaluate(model, groups, planning, device, verify_teacher=False):
    model.eval()
    rows = []
    score_error = 0.0
    for group in groups:
        prediction = predict_group(model, group, device)
        means = prediction[0]
        factual = torch.as_tensor(group["factual"], device=device, dtype=means.dtype)
        eligible = torch.as_tensor(group["eligible"], device=device, dtype=torch.bool)
        same = torch.as_tensor(group["same_prefix"], device=device, dtype=torch.bool)
        row = {"case": group["case"], "window": group["window"]}
        row.update({name: float(value) for name, value in response_supervision(prediction, group).items()})
        for name in ("uniform", "persistent", "true_mode"):
            scores = planner_scores(prediction, group, planning, group["dt"], weights_for(group, means, name))
            row.update({f"{name}_{key}": value for key, value in group_regret(scores, factual, eligible, same).items()})
            if verify_teacher:
                teacher = torch.as_tensor(group["teacher_scores"][name], device=device, dtype=means.dtype)
                difference = float((scores[eligible] - teacher[eligible]).abs().max())
                score_error = max(score_error, difference)
                if difference > 1e-5:
                    raise ValueError(f"offline/deployed planner score mismatch: {difference}")
            original = int(np.argmin(group["teacher_scores"][name]))
            row[f"{name}_action_changed"] = int(not group["same_prefix"][original, row[f"{name}_selected"]])
        rows.append(row)
    cases = sorted({row["case"] for row in rows})
    metrics = [key for key in rows[0] if key not in {"case", "window"} and not key.endswith("_selected")]
    per_case = {str(case): {name: float(np.mean([row[name] for row in rows if row["case"] == case]))
                           for name in metrics} for case in cases}
    result = {name: float(np.mean([values[name] for values in per_case.values()])) for name in metrics}
    result.update(per_case=per_case, groups=len(rows), independent_cases=len(cases), deployed_score_max_error=score_error)
    return result, rows


def paired_case_changes(baseline, candidate):
    cases = sorted(baseline["per_case"])
    generator = np.random.default_rng(734193)
    indices = generator.integers(0, len(cases), (10000, len(cases)))
    comparisons = {}
    for name in ("uniform_regret", "persistent_regret", "true_mode_regret", "persistent_prefix_regret", "nll", "ade"):
        differences = np.asarray([candidate["per_case"][case][name] - baseline["per_case"][case][name] for case in cases])
        comparisons[name] = {"candidate_minus_baseline": float(differences.mean()),
                             "case_bootstrap_ci95": np.quantile(differences[indices].mean(1), [.025, .975]).tolist()}
    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Fine-tune response predictions against the unchanged deployed planner objective.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--ranking-weight", type=float, default=.25)
    parser.add_argument("--effect-weight", type=float, default=1.0)
    parser.add_argument("--risk-gradient-temperature", type=float, default=.1)
    parser.add_argument("--ranking-temperature", type=float, default=.05)
    parser.add_argument("--top-candidates", type=int, default=8)
    parser.add_argument("--accumulate-groups", type=int, default=4)
    parser.add_argument("--max-nll-degradation", type=float, default=.05)
    parser.add_argument("--max-ade-relative-degradation", type=float, default=.05)
    parser.add_argument("--freeze-factual-prefix", action="store_true")
    parser.add_argument("--scene-balanced", action="store_true")
    parser.add_argument("--gap-calibration-weight", type=float, default=0.)
    parser.add_argument("--action-conditioned-uncertainty", action="store_true")
    parser.add_argument("--risk-calibration-weight", type=float, default=0.)
    parser.add_argument("--expected-regret-weight", type=float, default=0.)
    args = parser.parse_args()
    if args.epochs < 1 or args.accumulate_groups < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("epochs, accumulation and learning rate must be positive")
    for name in ("ranking_weight", "effect_weight", "gap_calibration_weight", "risk_calibration_weight", "expected_regret_weight", "max_nll_degradation", "max_ade_relative_degradation"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    torch.set_num_threads(2)
    checkpoint_path = Path(args.checkpoint).resolve()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = int(saved["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    train, train_protocol, train_hashes = load_groups(Path(args.data_dir) / "train", "train")
    val, val_protocol, val_hashes = load_groups(Path(args.data_dir) / "val", "val")
    for name in ("planning", "dt", "execute_steps", "case_splits", "checkpoint_sha256"):
        if train_protocol[name] != val_protocol[name]:
            raise ValueError(f"train/validation protocol mismatch: {name}")
    if file_hash(checkpoint_path) != train_protocol["checkpoint_sha256"]:
        raise ValueError("pilot must begin from the frozen collection checkpoint")
    planning = PlanningCostConfig(**train_protocol["planning"])
    if planning.violation_probability_alpha is not None:
        raise ValueError("planner fine-tuning requires belief-independent hard masks")
    model = _load_model(checkpoint_path, args.device, expected_dt=train_protocol["dt"])
    if args.action_conditioned_uncertainty:
        model = add_action_conditioned_uncertainty(model)
    trainable_names = configure_planner_finetuning(model, args.freeze_factual_prefix)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                 lr=args.learning_rate, weight_decay=1e-4)
    source_root = Path(__file__).resolve().parents[1]
    hashes = {**train_hashes, **val_hashes, str(checkpoint_path): file_hash(checkpoint_path)}
    for path in source_root.rglob("*.py"):
        hashes[str(path.resolve())] = file_hash(path)
        destination = output / "source_snapshot" / "sage" / path.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    (output / "protocol.json").write_text(json.dumps({
        "parameters": vars(args), "input_and_source_hashes": hashes, "case_splits": train_protocol["case_splits"],
        "planning": train_protocol["planning"], "training_groups": len(train), "validation_groups": len(val),
        "trainable_parameters": trainable_names,
        "risk_gradient": "straight-through sigmoid of sampled minimum clearance; forward risk and all deployed safety masks remain exact",
        "legacy_reference_mask": "For older stationary-reference snapshots without an active mask, restrict reference supervision to steps supported by at least one factual candidate; CrowdSimPlus terminates only on robot goal or timeout.",
        "selection": "minimum fixed-state Persistent prefix regret subject to response-quality guards; ties by full regret; validation only",
        "boundary": "Candidate future/hidden labels are supervised training targets, never deployed inputs. Logged beliefs are frozen for paired selection, not a recalibrated closed-loop result.",
    }, indent=2) + "\n")
    baseline, baseline_rows = evaluate(model, val, planning, args.device, verify_teacher=True)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    (output / "baseline_windows.json").write_text(json.dumps(baseline_rows) + "\n")
    print(f"baseline {json.dumps({name: value for name, value in baseline.items() if name != 'per_case'})}", flush=True)
    best_key = (baseline["persistent_prefix_regret"], baseline["persistent_regret"])
    best_epoch = 0
    best_metrics = baseline
    history = []
    generator = np.random.default_rng(seed)
    scene_counts = {case: sum(group["case"] == case for group in train) for case in {group["case"] for group in train}}
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for position, index in enumerate(generator.permutation(len(train))):
            group = train[int(index)]
            prediction = predict_group(model, group, args.device)
            means = prediction[0]
            factual = torch.as_tensor(group["factual"], device=args.device, dtype=means.dtype)
            eligible = torch.as_tensor(group["eligible"], device=args.device, dtype=torch.bool)
            same = torch.as_tensor(group["same_prefix"], device=args.device, dtype=torch.bool)
            response = response_supervision(prediction, group)
            ranks, gaps, risks, expected_regrets = [], [], [], []
            for name in ("uniform", "persistent", "true_mode"):
                scores, costs = planner_scores(prediction, group, planning, group["dt"], weights_for(group, means, name),
                                               args.risk_gradient_temperature, return_components=True)
                ranks.append(top_prefix_ranking_loss(scores, factual, eligible, same,
                                                    args.ranking_temperature, args.top_candidates))
                if args.gap_calibration_weight:
                    gaps.append(planner_gap_calibration_loss(scores, factual, eligible, same, args.top_candidates))
                if args.risk_calibration_weight:
                    risks.append(entity_risk_calibration_loss(costs, group["factual_entity_risk"], eligible))
                if args.expected_regret_weight:
                    expected_regrets.append(expected_prefix_regret_loss(scores, factual, eligible, same, args.ranking_temperature))
            ranking = torch.stack(ranks).mean()
            loss = response["nll"] + args.effect_weight * (response["effect_mse"] + response["reference_mse"]) + args.ranking_weight * ranking
            if gaps:
                loss = loss + args.gap_calibration_weight * torch.stack(gaps).mean()
            if risks:
                loss = loss + args.risk_calibration_weight * torch.stack(risks).mean()
            if expected_regrets:
                loss = loss + args.expected_regret_weight * torch.stack(expected_regrets).mean()
            if args.scene_balanced:
                loss = loss * len(train) / (len(scene_counts) * scene_counts[group["case"]])
            if not torch.isfinite(loss):
                raise ValueError("non-finite planner training loss")
            accumulation_start = position - position % args.accumulate_groups
            accumulation_size = min(args.accumulate_groups, len(train) - accumulation_start)
            (loss / accumulation_size).backward()
            losses.append({"loss": float(loss.detach()), "ranking": float(ranking.detach())})
            if (position + 1) % args.accumulate_groups == 0 or position + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (position + 1) % 200 == 0:
                print(f"epoch={epoch} training_groups={position + 1}/{len(train)}", flush=True)
        metrics, rows = evaluate(model, val, planning, args.device)
        admissible = (metrics["nll"] <= baseline["nll"] + args.max_nll_degradation
                      and metrics["ade"] <= baseline["ade"] * (1 + args.max_ade_relative_degradation)
                      and metrics["persistent_regret"] <= baseline["persistent_regret"] + 1e-9)
        key = (metrics["persistent_prefix_regret"], metrics["persistent_regret"])
        metrics.update(epoch=epoch, quality_guard_pass=admissible, paired_changes=paired_case_changes(baseline, metrics),
                       training_loss=float(np.mean([row["loss"] for row in losses])),
                       ranking_loss=float(np.mean([row["ranking"] for row in losses])))
        history.append(metrics)
        (output / "training.json").write_text(json.dumps(history, indent=2) + "\n")
        (output / f"epoch{epoch}_windows.json").write_text(json.dumps(rows) + "\n")
        if admissible and key < best_key:
            best_key, best_epoch, best_metrics = key, epoch, metrics
            promoted = dict(saved)
            promoted.pop("belief_calibration", None)
            promoted.update(model=model.state_dict(), model_config=vars(model.config), best_epoch=epoch, best_metrics=metrics, selection_split="val")
            promoted["provenance"] = {**saved.get("provenance", {}), "case_splits": train_protocol["case_splits"], "planner_alignment": {
                "parent_checkpoint": str(checkpoint_path), "parent_sha256": file_hash(checkpoint_path),
                "parameters": vars(args), "case_splits": train_protocol["case_splits"],
                "training_protocol_sha256": file_hash(output / "protocol.json"),
                "calibration_required": True,
            }}
            checkpoint_output = output / f"seed{seed}" / "response.pt"
            checkpoint_output.parent.mkdir(exist_ok=True)
            torch.save(promoted, checkpoint_output)
        print(f"epoch={epoch} best={best_epoch} {json.dumps({name: value for name, value in metrics.items() if name not in {'per_case', 'paired_changes'}})}", flush=True)
    changed = [path for path, digest in hashes.items() if not Path(path).exists() or file_hash(path) != digest]
    summary = {"status": "SOURCE_CHANGED" if changed else "COMPLETED", "changed_inputs_or_sources": changed,
               "best_epoch": best_epoch, "improved_with_quality_guard": best_epoch > 0,
               "baseline": baseline, "selected": best_metrics,
               "paired_changes": paired_case_changes(baseline, best_metrics)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if changed:
        raise RuntimeError("training sources or data changed")


if __name__ == "__main__":
    main()
