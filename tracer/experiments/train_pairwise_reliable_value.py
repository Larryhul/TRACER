from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tracer.experiments.collect_external_planner_supervision import file_hash
from tracer.experiments.train_executable_value import load_continuation_groups
from tracer.response.executable_value import FEATURE_VERSION
from tracer.response.reliable_executable_value import (
    PAIRWISE_RELIABILITY_OBJECTIVE, RELIABILITY_TARGETS,
)


class PairwiseReliabilityNetwork(nn.Module):
    def __init__(self, inputs, width=96):
        super().__init__()
        self.input = nn.Linear(inputs, width)
        self.normalization = nn.LayerNorm(width)
        self.hidden = nn.Linear(width, 48)
        self.output = nn.Linear(48, 2)

    def forward(self, values):
        values = nn.functional.silu(self.normalization(self.input(values)))
        return self.output(nn.functional.silu(self.hidden(values)))


def basic_feature_count(names):
    if not names or f"signed_log_{names[0]}" not in names:
        raise ValueError("pairwise value requires the expanded observable feature contract")
    count = names.index(f"signed_log_{names[0]}")
    if count < 1 or len(names) != count * 10:
        raise ValueError("pairwise value feature expansion differs from training")
    return count


def pairwise_dataset(groups, minimum_human_gain):
    count = basic_feature_count(groups[0]["names"])
    rows, rows_by_group = [], {}
    for group_index, group in enumerate(groups):
        if not group["rerank_allowed"]:
            continue
        parent = group["parent"]
        basic = group["features"][:, :count]
        for candidate in range(len(basic)):
            if candidate == parent:
                continue
            difference = group["targets"][candidate] - group["targets"][parent]
            row = {
                "case": int(group["case"]), "group": group_index, "candidate": candidate,
                "features": np.concatenate([basic[parent], basic[candidate], basic[candidate] - basic[parent]]),
                "human_improvement": float(difference[7]) < -minimum_human_gain,
                "safety_or_task_harm": bool(np.any(difference[:7] > 1e-10)),
                "response_risk_support": bool(group["response_risk_support"][candidate]),
            }
            rows_by_group.setdefault(group_index, []).append(len(rows))
            rows.append(row)
    if not rows:
        raise ValueError("pairwise value training requires matched alternative continuations")
    features = np.stack([row["features"] for row in rows]).astype(np.float32)
    labels = np.asarray([
        [row["human_improvement"], row["safety_or_task_harm"]] for row in rows
    ], dtype=np.float32)
    cases = np.asarray([row["case"] for row in rows], dtype=int)
    return features, labels, cases, rows, rows_by_group, count


def _scene_weights(cases, multiplicity=None):
    counts = collections.Counter(map(int, cases))
    multiplicity = collections.Counter() if multiplicity is None else multiplicity
    weights = np.asarray([
        float(multiplicity.get(int(case), 1)) / counts[int(case)] for case in cases
    ], dtype=np.float32)
    return weights * (len(weights) / weights.sum())


def _member_record(model, center, scale):
    return {
        "center": center.tolist(), "scale": scale.tolist(),
        "input_weight": model.input.weight.detach().cpu().numpy().tolist(),
        "input_bias": model.input.bias.detach().cpu().numpy().tolist(),
        "layer_norm_weight": model.normalization.weight.detach().cpu().numpy().tolist(),
        "layer_norm_bias": model.normalization.bias.detach().cpu().numpy().tolist(),
        "layer_norm_epsilon": float(model.normalization.eps),
        "hidden_weight": model.hidden.weight.detach().cpu().numpy().tolist(),
        "hidden_bias": model.hidden.bias.detach().cpu().numpy().tolist(),
        "output_weight": model.output.weight.detach().cpu().numpy().tolist(),
        "output_bias": model.output.bias.detach().cpu().numpy().tolist(),
    }


def _member_probability(member, features):
    from tracer.response.reliable_executable_value import _pairwise_member_probability

    return _pairwise_member_probability(member, features)


def fit_member(features, labels, cases, training_cases, validation_cases, seed, epochs, width):
    generator = np.random.default_rng(seed)
    sampled_cases = generator.choice(training_cases, size=len(training_cases), replace=True)
    multiplicity = collections.Counter(map(int, sampled_cases))
    train_mask = np.isin(cases, list(multiplicity))
    validation_mask = np.isin(cases, validation_cases)
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("pairwise member needs disjoint training and early-stop scenes")
    train_features, train_labels = features[train_mask], labels[train_mask]
    weights = _scene_weights(cases[train_mask], multiplicity)
    normalized_weights = weights / weights.sum()
    center = (train_features * normalized_weights[:, None]).sum(axis=0)
    scale = np.sqrt((np.square(train_features - center) * normalized_weights[:, None]).sum(axis=0))
    scale = np.maximum(scale, 1e-3).astype(np.float32)
    train_tensor = torch.from_numpy((train_features - center) / scale)
    label_tensor = torch.from_numpy(train_labels)
    weight_tensor = torch.from_numpy(weights)
    validation_tensor = torch.from_numpy((features[validation_mask] - center) / scale)
    validation_labels = torch.from_numpy(labels[validation_mask])
    validation_weights = torch.from_numpy(_scene_weights(cases[validation_mask]))
    prevalence = (train_labels * normalized_weights[:, None]).sum(axis=0)
    positive_weight = torch.from_numpy(np.clip((1. - prevalence) / np.maximum(prevalence, 1e-5), 1., 30.).astype(np.float32))
    torch.manual_seed(seed)
    model = PairwiseReliabilityNetwork(features.shape[1], width=width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    batch_generator = torch.Generator().manual_seed(seed + 1)
    best, best_loss, patience = None, float("inf"), 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_tensor), generator=batch_generator)
        for start in range(0, len(order), 512):
            indices = order[start:start + 512]
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(train_tensor[indices]), label_tensor[indices],
                pos_weight=positive_weight, reduction="none",
            )
            loss = (loss * weight_tensor[indices, None]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = nn.functional.binary_cross_entropy_with_logits(
                model(validation_tensor), validation_labels,
                pos_weight=positive_weight, reduction="none",
            )
            validation_loss = float((validation_loss * validation_weights[:, None]).mean())
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best = {name: value.detach().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break
    model.load_state_dict(best)
    return _member_record(model, center, scale), epoch + 1


def fit_ensemble(features, labels, cases, available_cases, members, seed, epochs, width):
    fitted, epoch_counts = [], []
    for member_index in range(members):
        generator = np.random.default_rng(seed + member_index * 101)
        shuffled = np.asarray(available_cases).copy()
        generator.shuffle(shuffled)
        validation_count = max(1, round(.15 * len(shuffled)))
        validation_cases = shuffled[:validation_count]
        training_cases = shuffled[validation_count:]
        member, count = fit_member(
            features, labels, cases, training_cases, validation_cases,
            seed + member_index * 101 + 17, epochs, width,
        )
        fitted.append(member)
        epoch_counts.append(count)
    return fitted, epoch_counts


def ensemble_probability(members, features):
    values = np.asarray([_member_probability(member, features) for member in members])
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def fit_probability_calibration(probabilities, labels, cases):
    calibrations = []
    weights = torch.from_numpy(_scene_weights(cases))
    for target in range(2):
        probability = np.clip(probabilities[:, target], 1e-5, 1. - 1e-5)
        logits = torch.from_numpy(np.log(probability / (1. - probability)).astype(np.float32))
        expected = torch.from_numpy(labels[:, target])
        log_slope = torch.tensor(0., requires_grad=True)
        intercept = torch.tensor(0., requires_grad=True)
        optimizer = torch.optim.LBFGS([log_slope, intercept], max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            slope = nn.functional.softplus(log_slope) + 1e-3
            loss = nn.functional.binary_cross_entropy_with_logits(
                slope * logits + intercept, expected, reduction="none",
            )
            loss = (loss * weights).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        calibrations.append({
            "slope": float((nn.functional.softplus(log_slope) + 1e-3).detach()),
            "intercept": float(intercept.detach()),
            "scope": "Scene-held-out train-only OOF Platt calibration; not a safety probability guarantee.",
        })
    return calibrations


def calibrate_probability(probability, calibration):
    probability = np.clip(probability, 1e-6, 1. - 1e-6)
    logits = np.log(probability / (1. - probability))
    return 1. / (1. + np.exp(-(calibration["slope"] * logits + calibration["intercept"])))


def build_training_support(features, cases, allowed_cases, seed, anchors=512):
    mask = np.isin(cases, allowed_cases)
    selected_features, selected_cases = features[mask], cases[mask]
    weights = _scene_weights(selected_cases)
    normalized_weights = weights / weights.sum()
    center = (selected_features * normalized_weights[:, None]).sum(axis=0)
    scale = np.sqrt((np.square(selected_features - center) * normalized_weights[:, None]).sum(axis=0))
    scale = np.maximum(scale, 1e-3)
    normalized = (selected_features - center) / scale
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(len(normalized), min(anchors, len(normalized)), replace=False))
    anchor_values, anchor_cases = normalized[indices], selected_cases[indices]
    nearest = []
    for value, case in zip(anchor_values, anchor_cases):
        alternatives = anchor_values[anchor_cases != case]
        nearest.append(float(np.sqrt(np.square(alternatives - value).mean(axis=1)).min()))
    return {
        "center": center.tolist(), "scale": scale.tolist(), "anchors": anchor_values.tolist(),
        "maximum_distance": float(np.quantile(nearest, .99)),
        "scope": "Train-only cross-scene pair-feature nearest-neighbour abstention; not a safety guarantee.",
    }


def domain_support(domain, features):
    normalized = (features - np.asarray(domain["center"])) / np.asarray(domain["scale"])
    anchors = np.asarray(domain["anchors"])
    distance = np.sqrt(np.square(normalized[:, None] - anchors[None]).mean(axis=-1)).min(axis=1)
    return distance <= domain["maximum_distance"]


def _wilson_lower(successes, total):
    if not total:
        return 0.
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1. + z * z / total
    center = proportion + z * z / (2. * total)
    margin = z * math.sqrt(proportion * (1. - proportion) / total + z * z / (4. * total * total))
    return (center - margin) / denominator


def evaluate_policy(groups, rows, rows_by_group, lower_human, upper_harm, in_domain,
                    human_threshold, harm_threshold, deployment_stride=4, maximum_interventions=1,
                    minimum_human_gain=.02):
    records, intervention_counts = [], {}
    for group_index, group in sorted(enumerate(groups), key=lambda item: (item[1]["case"], item[1]["window"])):
        case, parent = int(group["case"]), group["parent"]
        selected = parent
        count = intervention_counts.get(case, 0)
        if group["rerank_allowed"] and group["window"] % deployment_stride == 0 and count < maximum_interventions:
            eligible = []
            for row_index in rows_by_group.get(group_index, []):
                row = rows[row_index]
                if (row["response_risk_support"] and in_domain[row_index]
                        and lower_human[row_index] >= human_threshold
                        and upper_harm[row_index] <= harm_threshold):
                    eligible.append((-lower_human[row_index], upper_harm[row_index], row["candidate"]))
            if eligible:
                selected = min(eligible)[2]
        changed = selected != parent
        intervention_counts[case] = count + int(changed)
        difference = group["targets"][selected] - group["targets"][parent]
        records.append({"case": case, "window": int(group["window"]), "changed": changed,
                        "difference": difference})
    changed = [record for record in records if record["changed"]]
    differences = np.asarray([record["difference"] for record in changed]) if changed else np.zeros((0, len(RELIABILITY_TARGETS)))
    cases = sorted({record["case"] for record in records})
    all_case_mean = np.mean([
        np.mean([record["difference"] for record in records if record["case"] == case], axis=0)
        for case in cases
    ], axis=0)
    improved = int((differences[:, 7] < -1e-10).sum())
    strongly_improved = int((differences[:, 7] < -minimum_human_gain).sum())
    worsened = int((differences[:, 7] > 1e-10).sum())
    return {
        "changed_cases": len({record["case"] for record in changed}),
        "changed_groups": len(changed), "human_improved": improved,
        "human_strongly_improved": strongly_improved, "human_worsened": worsened,
        "human_win_wilson_lower95": _wilson_lower(improved, len(changed)),
        "harmful_changes": int(np.any(differences[:, :7] > 1e-10, axis=1).sum()) if len(changed) else 0,
        "changed_mean_human": float(differences[:, 7].mean()) if len(changed) else 0.,
        "changed_mean_time": float(differences[:, 8].mean()) if len(changed) else 0.,
        "all_case_mean_differences": dict(zip(RELIABILITY_TARGETS, all_case_mean.tolist())),
        "changed_records": [{"case": record["case"], "window": record["window"],
                             "differences": dict(zip(RELIABILITY_TARGETS, record["difference"].tolist()))}
                            for record in changed],
    }


def binary_metrics(labels, probabilities, cases):
    weights = _scene_weights(cases)
    brier = float(np.average(np.square(probabilities - labels), weights=weights))
    bins = np.linspace(0., 1., 11)
    calibration_error = 0.
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = (probabilities >= lower) & (probabilities < upper if upper < 1. else probabilities <= upper)
        if selected.any():
            bin_weight = float(weights[selected].sum() / weights.sum())
            calibration_error += bin_weight * abs(float(np.average(probabilities[selected], weights=weights[selected]))
                                                  - float(np.average(labels[selected], weights=weights[selected])))
    positive, negative = labels > .5, labels <= .5
    comparisons = probabilities[positive, None] > probabilities[negative][None]
    ties = probabilities[positive, None] == probabilities[negative][None]
    auc = float((comparisons.sum() + .5 * ties.sum()) / comparisons.size)
    return {"auc": auc, "scene_weighted_brier": brier, "scene_weighted_ece10": calibration_error,
            "positive_rate": float(np.average(labels, weights=weights))}


def main():
    parser = argparse.ArgumentParser(description="Train a conservative pairwise executable-action reliability head.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-members", type=int, default=4)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--minimum-human-gain", type=float, default=.2)
    parser.add_argument("--deployment-stride", type=int, default=4)
    parser.add_argument("--maximum-episode-interventions", type=int, default=1)
    parser.add_argument("--width", type=int, default=96)
    args = parser.parse_args()
    if (args.folds < 5 or args.fold_members < 2 or args.members < 8 or args.epochs < 20
            or args.minimum_human_gain <= 0 or args.deployment_stride < 1
            or args.maximum_episode_interventions < 1 or args.width < 4):
        raise ValueError("pairwise reliability requires conservative cross-fit and deployment settings")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    groups, protocol, hashes = load_continuation_groups(args.train_dir, "train", "reliability")
    behavior_value_model = None
    if protocol.get("behavior_value_model"):
        behavior_value_model = json.loads(Path(protocol["behavior_value_model"]).read_text(encoding="utf-8"))
    source_paths = [Path(__file__).resolve(), Path(__file__).with_name("train_executable_value.py"),
                    Path(__file__).with_name("external_value_policy.py"),
                    Path(__file__).resolve().parents[1] / "response" / "executable_value.py",
                    Path(__file__).resolve().parents[1] / "response" / "reliable_executable_value.py"]
    hashes.update({str(path): file_hash(path) for path in source_paths})
    features, labels, cases, rows, rows_by_group, count = pairwise_dataset(groups, args.minimum_human_gain)
    unique_cases = np.unique(cases)
    generator = np.random.default_rng(970912)
    shuffled = unique_cases.copy()
    generator.shuffle(shuffled)
    case_folds = np.array_split(shuffled, args.folds)
    oof_mean = np.full((len(rows), 2), np.nan)
    oof_deviation = np.full((len(rows), 2), np.nan)
    oof_domain = np.zeros(len(rows), dtype=bool)
    fold_records = []
    for fold, heldout_cases in enumerate(case_folds):
        available_cases = unique_cases[~np.isin(unique_cases, heldout_cases)]
        members, epochs = fit_ensemble(
            features, labels, cases, available_cases, args.fold_members,
            971000 + fold * 1000, args.epochs, args.width,
        )
        mask = np.isin(cases, heldout_cases)
        oof_mean[mask], oof_deviation[mask] = ensemble_probability(members, features[mask])
        domain = build_training_support(features, cases, available_cases, 972000 + fold)
        oof_domain[mask] = domain_support(domain, features[mask])
        fold_records.append({"fold": fold, "heldout_cases": heldout_cases.tolist(),
                             "member_epochs": epochs, "in_domain_pairs": int(oof_domain[mask].sum()),
                             "pairs": int(mask.sum())})
        print(f"pairwise cross-fit {fold + 1}/{args.folds}", flush=True)
    if not np.isfinite(oof_mean).all() or not np.isfinite(oof_deviation).all():
        raise ValueError("pairwise cross-fit did not cover every training pair")
    calibration = fit_probability_calibration(oof_mean, labels, cases)
    calibrated_mean = np.column_stack([
        calibrate_probability(oof_mean[:, target], calibration[target]) for target in range(2)
    ])
    metrics = {
        "human_improvement": binary_metrics(labels[:, 0], calibrated_mean[:, 0], cases),
        "safety_or_task_harm": binary_metrics(labels[:, 1], calibrated_mean[:, 1], cases),
    }
    trials = []
    for uncertainty in (.1, .25, .5):
        lower_human = calibrate_probability(
            np.clip(oof_mean[:, 0] - uncertainty * oof_deviation[:, 0], 0., 1.), calibration[0]
        )
        upper_harm = calibrate_probability(
            np.clip(oof_mean[:, 1] + uncertainty * oof_deviation[:, 1], 0., 1.), calibration[1]
        )
        for human_threshold in (.2, .25, .3, .35, .4, .45, .5, .55, .6):
            for harm_threshold in (.01, .025, .05, .1):
                evaluation = evaluate_policy(
                    groups, rows, rows_by_group, lower_human, upper_harm, oof_domain,
                    human_threshold, harm_threshold, args.deployment_stride,
                    args.maximum_episode_interventions, args.minimum_human_gain,
                )
                changed_folds = len({
                    fold for fold, fold_cases in enumerate(case_folds)
                    if set(fold_cases) & {record["case"] for record in evaluation["changed_records"]}
                })
                eligible = (
                    evaluation["changed_cases"] >= 5
                    and evaluation["human_improved"] >= 5
                    and evaluation["human_strongly_improved"] >= 3
                    and evaluation["human_improved"] > evaluation["human_worsened"]
                    and evaluation["harmful_changes"] == 0
                    and evaluation["changed_mean_human"] < 0.
                    and changed_folds >= 3
                )
                trials.append({
                    "uncertainty_multiplier": uncertainty,
                    "human_score_threshold": human_threshold,
                    "harm_score_threshold": harm_threshold,
                    "changed_folds": changed_folds, "eligible": eligible, "evaluation": evaluation,
                })
    eligible = [trial for trial in trials if trial["eligible"]]
    selected = max(
        eligible,
        key=lambda trial: (
            trial["evaluation"]["human_win_wilson_lower95"],
            -trial["evaluation"]["harmful_changes"],
            -trial["evaluation"]["changed_mean_human"],
            trial["human_score_threshold"], -trial["harm_score_threshold"],
        ),
    ) if eligible else None
    final_members, final_epochs = fit_ensemble(
        features, labels, cases, unique_cases, args.members, 973000, args.epochs, args.width,
    )
    training_support = build_training_support(features, cases, unique_cases, 974000)
    changed = [path for path, digest in hashes.items() if not Path(path).exists() or file_hash(path) != digest]
    summary = {
        "status": "SOURCE_CHANGED" if changed else "COMPLETED",
        "changed_inputs": changed, "train_cases": protocol["cases"], "groups": len(groups),
        "matched_groups": sum(group["rerank_allowed"] for group in groups), "pairs": len(rows),
        "basic_feature_count": count, "minimum_human_gain": args.minimum_human_gain,
        "network_width": args.width,
        "crossfit_metrics": metrics, "folds": fold_records, "trials": trials,
        "eligible_model": selected is not None, "selected": selected,
        "final_member_epochs": final_epochs,
        "boundary": "Train-only scene-held-out model selection. Continuation outcomes are one-intervention supervision; actual repeated-replanning development navigation remains required before any confirmation or deployment.",
    }
    if selected is not None and not changed:
        model = {
            "feature_version": FEATURE_VERSION,
            "selection_objective": PAIRWISE_RELIABILITY_OBJECTIVE,
            "target_names": ["human_improvement", "safety_or_task_harm"],
            "basic_feature_count": count, "ensemble": final_members,
            "probability_calibration": calibration,
            "uncertainty_multiplier": selected["uncertainty_multiplier"],
            "human_score_threshold": selected["human_score_threshold"],
            "harm_score_threshold": selected["harm_score_threshold"],
            "training_support": training_support, "response_risk_guard": True,
            "feature_names": groups[0]["names"], "minimum_gain": 0.,
            "branch_count": protocol["continuation_branches"],
            "response_checkpoint_sha256": protocol["checkpoint_sha256"],
            "case_splits": protocol["case_splits"], "parent_parameters": protocol["parameters"],
            "parent_artifact_hashes": protocol["parent_artifact_hashes"],
            "training_behavior_value_sha256": protocol.get("behavior_value_sha256"),
            "reference_value_model": behavior_value_model,
            "reference_policy_role": "frozen_collection_behavior" if behavior_value_model is not None else None,
            "reference_scope": (
                "Frozen collection behavior policy; pairwise head only reranks its supported executable-prefix shortlist."
                if behavior_value_model is not None else
                "Original constrained controller; pairwise head only reranks its supported executable-prefix shortlist."
            ),
            "deployment_stride": args.deployment_stride,
            "maximum_episode_interventions": args.maximum_episode_interventions,
            "source_and_input_hashes": hashes, "deployment_status": "DEVELOPMENT_ONLY",
            "crossfit_selection": selected,
        }
        (output / "candidate_policy.json").write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "eligible_model": summary["eligible_model"],
                      "crossfit_metrics": metrics, "selected": selected}, indent=2, allow_nan=False))
    if changed:
        raise RuntimeError("pairwise reliability inputs changed")


if __name__ == "__main__":
    main()
