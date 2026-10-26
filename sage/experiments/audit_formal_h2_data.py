from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from sage.experiments.formal_manifest import sha256_file
from sage.experiments.train_persistent_trajectory_response import _make_dataset


AUDITED_TENSOR_FIELDS = (
    "mode",
    "entity_history",
    "robot_current",
    "robot_candidates",
    "target_future_vel",
    "activities",
    "presence_mask",
    "active_mask",
)


def _sample_fingerprint(sample: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for field in AUDITED_TENSOR_FIELDS:
        tensor = sample[field]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{field} must be a tensor")
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(field.encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _mean_off_diagonal_correlation(values: np.ndarray) -> float:
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("activities must have shape (observations, robots)")
    correlation = np.corrcoef(values, rowvar=False)
    mask = ~np.eye(correlation.shape[0], dtype=bool)
    return float(correlation[mask].mean())


def _expected_shapes(cfg: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    entities = int(cfg["num_entities"])
    windows = int(cfg["windows"])
    history = int(cfg["history"])
    horizon = int(cfg["horizon"])
    robots = int(cfg["num_robots"])
    return {
        "mode": (entities,),
        "entity_history": (entities, windows, history, 2),
        "robot_current": (entities, windows, robots, 4),
        "robot_candidates": (entities, windows, robots, horizon, 4),
        "target_future_vel": (entities, windows, horizon, 2),
        "activities": (entities, windows, robots),
        "presence_mask": (entities, windows),
        "active_mask": (entities, windows),
    }


def audit_h2_data(
    raw: dict[str, Any],
    episodes_per_split: int,
) -> dict[str, Any]:
    if episodes_per_split < 1:
        raise ValueError("episodes_per_split must be positive")
    cfg = raw["dataset"]
    data_seed = int(cfg["data_seed"])
    expected_shapes = _expected_shapes(cfg)
    active_entities = int(cfg["active_entities"])
    num_modes = 3

    split_reports: dict[str, Any] = {}
    split_fingerprints: dict[str, set[str]] = {}
    split_entity_ids: dict[str, set[int]] = {}
    configured_entity_id_ranges: dict[str, tuple[int, int]] = {}
    split_activities: dict[str, np.ndarray] = {}
    all_modes: list[int] = []
    all_fingerprints: list[str] = []
    structural_checks: list[bool] = []

    for split in ("train", "val", "test", "ood"):
        action_policy = (
            str(cfg.get("ood_action_policy", "correlated"))
            if split == "ood"
            else "iid"
        )
        dataset = _make_dataset(raw, split, data_seed, action_policy=action_policy)
        sample_count = min(len(dataset), episodes_per_split)
        fingerprints: list[str] = []
        entity_ids: set[int] = set()
        modes: list[int] = []
        activities: list[np.ndarray] = []
        max_shared_action_error = 0.0
        finite = True
        shapes_match = True
        active_count_match = True
        active_subset_of_presence = True
        reentry_pattern_match = True

        for episode in range(sample_count):
            sample = dataset[episode]
            fingerprint = _sample_fingerprint(sample)
            fingerprints.append(fingerprint)
            all_fingerprints.append(fingerprint)

            for field, shape in expected_shapes.items():
                tensor = sample[field]
                shapes_match &= tuple(tensor.shape) == shape
                finite &= bool(torch.isfinite(tensor).all())

            episode_entity_ids = [int(value) for value in sample["entity_id"]]
            structural_checks.append(len(episode_entity_ids) == len(set(episode_entity_ids)))
            structural_checks.append(int(sample["episode_id"]) == episode)
            entity_ids.update(episode_entity_ids)
            episode_modes = [int(value) for value in sample["mode"]]
            modes.extend(episode_modes)
            all_modes.extend(episode_modes)

            presence = sample["presence_mask"].bool()
            active = sample["active_mask"].bool()
            active_count_match &= bool((active.sum(dim=0) == active_entities).all())
            active_subset_of_presence &= not bool((active & ~presence).any())
            reentry_pattern_match &= (
                not bool(presence[0, 2:4].any())
                and bool(presence[0, 4])
            )

            current_velocity = sample["robot_current"][..., 2:]
            candidate_velocity = sample["robot_candidates"][..., 2:]
            episode_activities = sample["activities"]
            for tensor in (current_velocity, candidate_velocity, episode_activities):
                reference = tensor[0:1].expand_as(tensor)
                max_shared_action_error = max(
                    max_shared_action_error,
                    float((tensor - reference).abs().max()),
                )
            activities.append(episode_activities[0].numpy())

        repeated_fingerprint = _sample_fingerprint(dataset[0])
        deterministic = repeated_fingerprint == fingerprints[0]
        first_entity_ids = [int(value) for value in dataset[0]["entity_id"]]
        last_entity_ids = [int(value) for value in dataset[len(dataset) - 1]["entity_id"]]
        configured_entity_id_ranges[split] = (
            min(first_entity_ids),
            max(last_entity_ids),
        )
        activity_array = np.concatenate(activities, axis=0)
        split_activities[split] = activity_array
        split_fingerprints[split] = set(fingerprints)
        split_entity_ids[split] = entity_ids
        mode_counts = Counter(modes)
        split_reports[split] = {
            "configured_episodes": len(dataset),
            "audited_episodes": sample_count,
            "action_policy": action_policy,
            "dataset_sha256": hashlib.sha256(
                "".join(fingerprints).encode("ascii")
            ).hexdigest(),
            "unique_content_fingerprints": len(set(fingerprints)),
            "unique_entity_ids": len(entity_ids),
            "entity_id_min": min(entity_ids),
            "entity_id_max": max(entity_ids),
            "configured_entity_id_min": configured_entity_id_ranges[split][0],
            "configured_entity_id_max": configured_entity_id_ranges[split][1],
            "mode_counts": {str(mode): mode_counts[mode] for mode in range(num_modes)},
            "deterministic_repeated_access": deterministic,
            "all_tensors_finite": finite,
            "all_shapes_match_config": shapes_match,
            "active_count_matches_top_k": active_count_match,
            "active_is_subset_of_presence": active_subset_of_presence,
            "entity0_exit_reentry_pattern": reentry_pattern_match,
            "max_shared_team_action_error": max_shared_action_error,
            "activity_min": float(activity_array.min()),
            "activity_max": float(activity_array.max()),
            "mean_within_team_activity_sd": float(
                activity_array.std(axis=1).mean()
            ),
            "mean_robot_activity_correlation": _mean_off_diagonal_correlation(
                activity_array
            ),
        }
        structural_checks.extend(
            [
                deterministic,
                finite,
                shapes_match,
                active_count_match,
                active_subset_of_presence,
                reentry_pattern_match,
                max_shared_action_error <= 1e-6,
                len(set(fingerprints)) == sample_count,
                len(entity_ids) == sample_count * int(cfg["num_entities"]),
            ]
        )

    split_names = tuple(split_reports)
    content_overlap: dict[str, int] = {}
    entity_id_overlap: dict[str, int] = {}
    for first_index, first in enumerate(split_names):
        for second in split_names[first_index + 1 :]:
            key = f"{first}:{second}"
            content_overlap[key] = len(
                split_fingerprints[first] & split_fingerprints[second]
            )
            entity_id_overlap[key] = len(
                split_entity_ids[first] & split_entity_ids[second]
            )
    configured_id_ranges_disjoint = all(
        configured_entity_id_ranges[first][1]
        < configured_entity_id_ranges[second][0]
        or configured_entity_id_ranges[second][1]
        < configured_entity_id_ranges[first][0]
        for first_index, first in enumerate(split_names)
        for second in split_names[first_index + 1 :]
    )

    mode_counts = Counter(all_modes)
    total_modes = len(all_modes)
    mode_fractions = {
        str(mode): mode_counts[mode] / total_modes for mode in range(num_modes)
    }
    expected_fraction = 1.0 / num_modes
    tolerance = max(
        0.05,
        3.0 * math.sqrt(expected_fraction * (1.0 - expected_fraction) / total_modes),
    )
    mode_balance_ok = max(
        abs(fraction - expected_fraction) for fraction in mode_fractions.values()
    ) <= tolerance

    id_activity = split_activities["test"]
    ood_activity = split_activities["ood"]
    id_dispersion = float(id_activity.std(axis=1).mean())
    ood_dispersion = float(ood_activity.std(axis=1).mean())
    id_correlation = _mean_off_diagonal_correlation(id_activity)
    ood_correlation = _mean_off_diagonal_correlation(ood_activity)
    ood_shift_ok = (
        ood_dispersion < 0.35 * id_dispersion
        and ood_correlation > id_correlation + 0.5
    )
    no_split_overlap = (
        not any(content_overlap.values())
        and not any(entity_id_overlap.values())
        and configured_id_ranges_disjoint
    )
    checks = {
        "fixed_data_seed_declared": data_seed == 418731,
        "all_structural_checks_pass": all(structural_checks),
        "no_audited_content_overlap_and_full_id_ranges_disjoint": no_split_overlap,
        "mode_balance_within_tolerance": mode_balance_ok,
        "ood_action_correlation_shift_detected": ood_shift_ok,
        "activity_support_preserved": all(
            report["activity_min"] >= 0.18 - 1e-6
            and report["activity_max"] <= 1.0 + 1e-6
            for report in split_reports.values()
        ),
    }
    return {
        "audit_status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "Formal-v1 H2 synthetic data preflight; no model outcomes inspected",
        "data_seed": data_seed,
        "episodes_per_split_limit": episodes_per_split,
        "split_reports": split_reports,
        "cross_split_content_overlap": content_overlap,
        "cross_split_entity_id_overlap": entity_id_overlap,
        "global_content_sha256": hashlib.sha256(
            "".join(all_fingerprints).encode("ascii")
        ).hexdigest(),
        "mode_fractions": mode_fractions,
        "mode_balance_tolerance": tolerance,
        "ood_shift": {
            "id_mean_within_team_activity_sd": id_dispersion,
            "ood_mean_within_team_activity_sd": ood_dispersion,
            "id_mean_robot_activity_correlation": id_correlation,
            "ood_mean_robot_activity_correlation": ood_correlation,
        },
        "checks": checks,
        "gate_created": False,
    }


def _write_markdown(path: Path, payload: dict[str, Any], config_path: Path) -> None:
    lines = [
        "# Formal-v1 H2 Data Audit",
        "",
        f"- Status: `{payload['audit_status']}`",
        f"- Config: `{config_path}`",
        f"- Config SHA256: `{sha256_file(config_path)}`",
        f"- Data seed: `{payload['data_seed']}`",
        "- Boundary: structural preflight only; no model outcomes or formal gate were inspected or created.",
        "",
        "## Checks",
        "",
    ]
    for name, passed in payload["checks"].items():
        lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Split Summary",
            "",
            "| Split | Audited/configured | Policy | Unique IDs | Mode counts | Team activity SD | Robot correlation |",
            "|---|---:|---|---:|---|---:|---:|",
        ]
    )
    for split, report in payload["split_reports"].items():
        lines.append(
            f"| {split} | {report['audited_episodes']}/{report['configured_episodes']} | "
            f"{report['action_policy']} | {report['unique_entity_ids']} | "
            f"{report['mode_counts']} | {report['mean_within_team_activity_sd']:.4f} | "
            f"{report['mean_robot_activity_correlation']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Formal-v1 H2 generated data.")
    parser.add_argument(
        "--config", default="configs/persistent_trajectory_formal.yaml"
    )
    parser.add_argument("--episodes-per-split", type=int, default=256)
    parser.add_argument(
        "--output-dir", default="reports/formal_v1_h2_data_audit_2026-08-11"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    payload = audit_h2_data(raw, args.episodes_per_split)
    payload["config"] = {
        "path": str(config_path),
        "sha256": sha256_file(config_path),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_dir / "audit.md", payload, config_path)
    print(
        json.dumps(
            {"status": payload["audit_status"], "output_dir": str(output_dir)},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
