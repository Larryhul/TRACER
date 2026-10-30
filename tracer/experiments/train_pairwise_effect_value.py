from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tracer.experiments.collect_external_planner_supervision import file_hash
from tracer.experiments.train_executable_value import load_continuation_groups
from tracer.experiments.train_pairwise_reliable_value import (
    PairwiseReliabilityNetwork, build_training_support, calibrate_probability,
    domain_support, evaluate_policy, fit_probability_calibration, pairwise_dataset,
)
from tracer.response.executable_value import FEATURE_VERSION
from tracer.response.reliable_executable_value import PAIRWISE_EFFECT_OBJECTIVE


def scene_weights(cases, multiplicity=None):
    counts = collections.Counter(map(int, cases))
    multiplicity = collections.Counter() if multiplicity is None else multiplicity
    weights = np.asarray([
        float(multiplicity.get(int(case), 1)) / counts[int(case)] for case in cases
    ], dtype=np.float32)
    return weights * (len(weights) / weights.sum())


def member_record(model, center, scale):
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


def fit_effect_member(features, human_effect, harm, cases, training_cases,
                      validation_cases, seed, epochs, effect_scale):
    generator = np.random.default_rng(seed)
    sampled_cases = generator.choice(training_cases, size=len(training_cases), replace=True)
    multiplicity = collections.Counter(map(int, sampled_cases))
    train_mask = np.isin(cases, list(multiplicity))
    validation_mask = np.isin(cases, validation_cases)
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("pairwise effect member requires disjoint fit and early-stop scenes")
    train_features = features[train_mask]
    weights = scene_weights(cases[train_mask], multiplicity)
    normalized_weights = weights / weights.sum()
    center = (train_features * normalized_weights[:, None]).sum(axis=0)
    scale = np.sqrt((np.square(train_features - center) * normalized_weights[:, None]).sum(axis=0))
    scale = np.maximum(scale, 1e-3).astype(np.float32)
    train_tensor = torch.from_numpy((train_features - center) / scale)
    validation_tensor = torch.from_numpy((features[validation_mask] - center) / scale)
    train_effect = torch.from_numpy(np.arcsinh(human_effect[train_mask] / effect_scale))
    validation_effect = torch.from_numpy(np.arcsinh(human_effect[validation_mask] / effect_scale))
    train_harm = torch.from_numpy(harm[train_mask])
    validation_harm = torch.from_numpy(harm[validation_mask])
    weight_tensor = torch.from_numpy(weights)
    prevalence = float(np.average(harm[train_mask], weights=weights))
    positive_weight = torch.tensor(
        np.clip((1. - prevalence) / max(prevalence, 1e-5), 1., 30.), dtype=torch.float32
    )
    torch.manual_seed(seed)
    model = PairwiseReliabilityNetwork(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-4)
    batch_generator = torch.Generator().manual_seed(seed + 1)
    best, best_loss, patience = None, float("inf"), 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_tensor), generator=batch_generator)
        for start in range(0, len(order), 512):
            indices = order[start:start + 512]
            predictions = model(train_tensor[indices])
            effect_loss = nn.functional.smooth_l1_loss(
                predictions[:, 0], train_effect[indices], beta=.5, reduction="none"
            )
            harm_loss = nn.functional.binary_cross_entropy_with_logits(
                predictions[:, 1], train_harm[indices],
                pos_weight=positive_weight, reduction="none",
            )
            loss = ((effect_loss + harm_loss) * weight_tensor[indices]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(validation_tensor)
            validation_loss = nn.functional.smooth_l1_loss(
                predictions[:, 0], validation_effect, beta=.5, reduction="none"
            ) + nn.functional.binary_cross_entropy_with_logits(
                predictions[:, 1], validation_harm,
                pos_weight=positive_weight, reduction="none",
            )
            validation_loss = float(validation_loss.mean())
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best = {name: value.detach().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break
    model.load_state_dict(best)
    return member_record(model, center, scale), epoch + 1


def fit_effect_ensemble(features, human_effect, harm, cases, available_cases,
                        members, seed, epochs, effect_scale):
    fitted, epoch_counts = [], []
    for member_index in range(members):
        generator = np.random.default_rng(seed + member_index * 101)
        shuffled = np.asarray(available_cases).copy()
        generator.shuffle(shuffled)
        validation_count = max(1, round(.15 * len(shuffled)))
        member, count = fit_effect_member(
            features, human_effect, harm, cases,
            shuffled[validation_count:], shuffled[:validation_count],
            seed + member_index * 101 + 17, epochs, effect_scale,
        )
        fitted.append(member)
        epoch_counts.append(count)
    return fitted, epoch_counts


def member_effect_predictions(member, features, effect_scale):
    from tracer.response.reliable_executable_value import _pairwise_member_logits, _sigmoid

    raw = _pairwise_member_logits(member, features)
    return np.column_stack([effect_scale * np.sinh(raw[:, 0]), _sigmoid(raw[:, 1])])


def ensemble_effect_predictions(members, features, effect_scale):
    values = np.asarray([
        member_effect_predictions(member, features, effect_scale) for member in members
    ])
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def main():
    parser = argparse.ArgumentParser(description="Train an effect-size pairwise action head with harm veto.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-members", type=int, default=4)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--effect-transform-scale", type=float, default=.2)
    parser.add_argument("--deployment-stride", type=int, default=4)
    parser.add_argument("--maximum-episode-interventions", type=int, default=1)
    args = parser.parse_args()
    if (args.folds < 5 or args.fold_members < 4 or args.members < 8 or args.epochs < 20
            or args.effect_transform_scale <= 0 or args.deployment_stride < 1
            or args.maximum_episode_interventions < 1):
        raise ValueError("pairwise effect training requires conservative cross-fit and deployment settings")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    groups, protocol, hashes = load_continuation_groups(args.train_dir, "train", "reliability")
    behavior_value_model = None
    if protocol.get("behavior_value_model"):
        behavior_value_model = json.loads(Path(protocol["behavior_value_model"]).read_text(encoding="utf-8"))
    source_paths = [
        Path(__file__).resolve(), Path(__file__).with_name("train_pairwise_reliable_value.py"),
        Path(__file__).with_name("train_executable_value.py"),
        Path(__file__).with_name("external_value_policy.py"),
        Path(__file__).resolve().parents[1] / "response" / "executable_value.py",
        Path(__file__).resolve().parents[1] / "response" / "reliable_executable_value.py",
    ]
    hashes.update({str(path): file_hash(path) for path in source_paths})
    features, _, cases, rows, rows_by_group, count = pairwise_dataset(groups, .02)
    human_effect = np.asarray([
        groups[row["group"]]["targets"][row["candidate"], 7]
        - groups[row["group"]]["targets"][groups[row["group"]]["parent"], 7]
        for row in rows
    ], dtype=np.float32)
    harm = np.asarray([row["safety_or_task_harm"] for row in rows], dtype=np.float32)
    unique_cases = np.unique(cases)
    generator = np.random.default_rng(991500)
    shuffled = unique_cases.copy()
    generator.shuffle(shuffled)
    case_folds = np.array_split(shuffled, args.folds)
    oof_mean = np.full((len(rows), 2), np.nan)
    oof_deviation = np.full((len(rows), 2), np.nan)
    oof_domain = np.zeros(len(rows), dtype=bool)
    fold_records = []
    for fold, heldout_cases in enumerate(case_folds):
        available_cases = unique_cases[~np.isin(unique_cases, heldout_cases)]
        members, epochs = fit_effect_ensemble(
            features, human_effect, harm, cases, available_cases, args.fold_members,
            992000 + fold * 100, args.epochs, args.effect_transform_scale,
        )
        mask = np.isin(cases, heldout_cases)
        oof_mean[mask], oof_deviation[mask] = ensemble_effect_predictions(
            members, features[mask], args.effect_transform_scale
        )
        domain = build_training_support(features, cases, available_cases, 995000 + fold)
        oof_domain[mask] = domain_support(domain, features[mask])
        fold_records.append({
            "fold": fold, "heldout_cases": heldout_cases.tolist(), "member_epochs": epochs,
            "pairs": int(mask.sum()), "in_domain_pairs": int(oof_domain[mask].sum()),
        })
        print(f"pairwise effect cross-fit {fold + 1}/{args.folds}", flush=True)
    if not np.isfinite(oof_mean).all() or not np.isfinite(oof_deviation).all():
        raise ValueError("pairwise effect cross-fit did not cover every training pair")
    calibration = fit_probability_calibration(
        np.column_stack([np.full(len(rows), .5), oof_mean[:, 1]]),
        np.column_stack([np.zeros(len(rows)), harm]), cases,
    )[1]
    trials = []
    for uncertainty in (.1, .25, .5, 1.):
        upper_effect = oof_mean[:, 0] + uncertainty * oof_deviation[:, 0]
        upper_harm = calibrate_probability(
            np.clip(oof_mean[:, 1] + uncertainty * oof_deviation[:, 1], 0., 1.), calibration
        )
        for minimum_gain in (.01, .025, .05):
            for harm_threshold in (.01, .025, .05, .1):
                evaluation = evaluate_policy(
                    groups, rows, rows_by_group, -upper_effect, upper_harm, oof_domain,
                    minimum_gain, harm_threshold, args.deployment_stride,
                    args.maximum_episode_interventions, .2,
                )
                changed_cases = {record["case"] for record in evaluation["changed_records"]}
                changed_folds = sum(bool(set(fold_cases) & changed_cases) for fold_cases in case_folds)
                eligible = (
                    evaluation["changed_cases"] >= 12
                    and evaluation["human_improved"] >= 7
                    and evaluation["human_strongly_improved"] >= 3
                    and evaluation["human_improved"] > evaluation["human_worsened"]
                    and evaluation["harmful_changes"] == 0
                    and evaluation["changed_mean_human"] < 0.
                    and changed_folds >= 4
                )
                trials.append({
                    "uncertainty_multiplier": uncertainty,
                    "minimum_human_gain": minimum_gain,
                    "harm_score_threshold": harm_threshold,
                    "changed_folds": changed_folds, "eligible": eligible,
                    "evaluation": evaluation,
                })
    eligible = [trial for trial in trials if trial["eligible"]]
    selected = max(
        eligible,
        key=lambda trial: (
            trial["evaluation"]["human_win_wilson_lower95"],
            -trial["evaluation"]["human_worsened"],
            -trial["evaluation"]["changed_mean_human"],
            trial["minimum_human_gain"], -trial["harm_score_threshold"],
        ),
    ) if eligible else None
    final_members, final_epochs = fit_effect_ensemble(
        features, human_effect, harm, cases, unique_cases, args.members,
        996000, args.epochs, args.effect_transform_scale,
    )
    training_support = build_training_support(features, cases, unique_cases, 997000)
    changed = [path for path, digest in hashes.items() if not Path(path).exists() or file_hash(path) != digest]
    correlation = float(np.corrcoef(human_effect, oof_mean[:, 0])[0, 1])
    summary = {
        "status": "SOURCE_CHANGED" if changed else "COMPLETED", "changed_inputs": changed,
        "train_cases": protocol["cases"], "groups": len(groups),
        "matched_groups": sum(group["rerank_allowed"] for group in groups), "pairs": len(rows),
        "basic_feature_count": count, "effect_transform_scale": args.effect_transform_scale,
        "oof_human_effect_correlation": correlation, "folds": fold_records,
        "trials": trials, "eligible_model": selected is not None, "selected": selected,
        "final_member_epochs": final_epochs,
        "boundary": "Train-only scene-held-out effect-size selection. Safety/task harm is an empirical veto, not a guarantee. Actual repeated-replanning development navigation is required before confirmation.",
    }
    if selected is not None and not changed:
        model = {
            "feature_version": FEATURE_VERSION, "selection_objective": PAIRWISE_EFFECT_OBJECTIVE,
            "target_names": ["human_velocity_variation_difference", "safety_or_task_harm_probability"],
            "basic_feature_count": count, "ensemble": final_members,
            "effect_transform_scale": args.effect_transform_scale,
            "harm_probability_calibration": calibration,
            "uncertainty_multiplier": selected["uncertainty_multiplier"],
            "minimum_human_gain": selected["minimum_human_gain"],
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
                "Frozen collection behavior policy; effect head only reranks its supported executable-prefix shortlist."
                if behavior_value_model is not None else
                "Original constrained controller; effect head only reranks its supported executable-prefix shortlist."
            ),
            "deployment_stride": args.deployment_stride,
            "maximum_episode_interventions": args.maximum_episode_interventions,
            "source_and_input_hashes": hashes, "deployment_status": "DEVELOPMENT_ONLY",
            "crossfit_selection": selected,
        }
        (output / "candidate_policy.json").write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": summary["status"], "eligible_model": summary["eligible_model"],
        "oof_human_effect_correlation": correlation, "selected": selected,
    }, indent=2, allow_nan=False))
    if changed:
        raise RuntimeError("pairwise effect inputs changed")


if __name__ == "__main__":
    main()
