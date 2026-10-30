"""Aggregate paired SocialGym2 TRACER seed artifacts without rerunning ROS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from tracer.experiments.socialgym2_physics_contract import verify_physics_audits
from tracer.experiments.socialgym2_policy_roles import (
    EXTERNAL_POLICIES,
    POLICY_ROLE_SCHEMA,
    socialgym2_policy_roles,
)


METRICS = (
    "success",
    "team_success",
    "robot_success_fraction",
    "collision_free_team_success",
    "initial_collision",
    "collision",
    "task_progress",
    "mean_reward",
    "min_distance",
    "fallback_rate",
    "final_belief_entropy",
    "action_change_rate",
    "belief_selection_audit_windows",
    "belief_selection_change_rate",
    "belief_action_change_rate",
    "belief_score_l1",
    "belief_selected_regret",
)


def _wilson_interval(values: list[float]) -> dict[str, float]:
    if not values or any(value not in (0.0, 1.0) for value in values):
        raise ValueError("Wilson intervals require nonempty binary observations")
    count = len(values)
    proportion = sum(values) / count
    quantile = 1.959963984540054
    denominator = 1 + quantile**2 / count
    center = (proportion + quantile**2 / (2 * count)) / denominator
    radius = quantile * math.sqrt(proportion * (1 - proportion) / count + quantile**2 / (4 * count**2)) / denominator
    return {"mean": proportion, "ci95_low": max(0.0, center - radius), "ci95_high": min(1.0, center + radius)}


def _holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    adjusted = {}
    previous = 0.0
    for index, (name, probability) in enumerate(sorted(pvalues.items(), key=lambda item: item[1])):
        previous = max(previous, min(1.0, probability * (len(pvalues) - index)))
        adjusted[name] = previous
    return adjusted


def _bootstrap(values: list[float], seed: int, samples: int = 4000) -> dict[str, Any]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "count": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "count": int(len(array)),
    }


def _cluster_bootstrap(
    values_by_cluster: dict[str, list[float]],
    seed: int,
    samples: int = 4000,
) -> dict[str, Any]:
    """Bootstrap cluster means so episodes from one seed are not iid samples."""

    clusters = [
        np.asarray(values, dtype=np.float64)
        for _, values in sorted(values_by_cluster.items())
        if values
    ]
    if not clusters:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "count": 0}
    cluster_means = np.asarray([cluster.mean() for cluster in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = cluster_means[rng.integers(0, len(cluster_means), size=(samples, len(cluster_means)))].mean(axis=1)
    return {
        "mean": float(cluster_means.mean()),
        "std": float(cluster_means.std(ddof=1)) if len(cluster_means) > 1 else 0.0,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "count": int(sum(len(cluster) for cluster in clusters)),
        "clusters": int(len(cluster_means)),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (str(row["seed"]), str(row.get("scenario", "")), str(row["episode"]))


def _two_sided_sign_test(values: list[float]) -> dict[str, Any]:
    nonzero = [value for value in values if value != 0.0]
    positive = sum(value > 0.0 for value in nonzero)
    negative = sum(value < 0.0 for value in nonzero)
    count = len(nonzero)
    if count == 0:
        return {"count": 0, "positive": 0, "negative": 0, "p_two_sided": 1.0}
    tail = sum(math.comb(count, index) for index in range(min(positive, negative) + 1)) / (2.0 ** count)
    return {
        "count": count,
        "positive": int(positive),
        "negative": int(negative),
        "p_two_sided": float(min(1.0, 2.0 * tail)),
    }


def factual_memory_contrasts(rows: list[dict[str, str]], scale: float) -> dict[str, Any]:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("factual-memory replay requires the calibrated positive scale")
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["policy"] == "tracer-uniform":
            key = (*_record_key(row), row["track_id"])
            groups.setdefault(key, []).append(row)
    scene_values: dict[str, dict[str, list[float]]] = {}
    evidence_count = 0
    for key, group in sorted(groups.items()):
        if len({row["step"] for row in group}) != len(group):
            raise ValueError("duplicate factual evidence within a tracked person")
        uniform = np.full(3, -math.log(3.0))
        persistent = uniform.copy()
        reset = uniform.copy()
        for row in sorted(group, key=lambda item: int(item["step"])):
            likelihood = np.asarray(json.loads(row["factual_mode_log_likelihood"]), dtype=float)
            if likelihood.shape != (3,) or not np.isfinite(likelihood).all():
                raise ValueError("invalid factual likelihood in memory replay")
            if int(row["sim_steps_executed"]) != 16:
                raise ValueError("partial terminal observation cannot enter factual memory replay")
            for field in ("decision_log_belief", "posterior_log_belief"):
                recorded = np.asarray(json.loads(row[field]), dtype=float)
                if recorded.shape != (3,) or not np.allclose(recorded, uniform, atol=1e-5):
                    raise ValueError("factual replay source is not the declared Uniform policy")
            scores = {name: -float(np.logaddexp.reduce(prior + likelihood))
                      for name, prior in (("uniform", uniform), ("persistent", persistent), ("reset", reset))}
            scene = f"{row['scenario']}::{row['initial_state_hash']}"
            values = scene_values.setdefault(scene, {"uniform-minus-persistent": [], "reset-minus-persistent": []})
            values["uniform-minus-persistent"].append(scores["uniform"] - scores["persistent"])
            values["reset-minus-persistent"].append(scores["reset"] - scores["persistent"])
            persistent = persistent + scale * likelihood
            persistent -= np.logaddexp.reduce(persistent)
            reset = uniform + scale * likelihood
            reset -= np.logaddexp.reduce(reset)
            evidence_count += 1
    return {
        "status": "AVAILABLE" if scene_values else "NO_OBSERVED_UNIFORM_EVIDENCE",
        "independent_observed_scenes": len(scene_values), "evidence_rows": evidence_count,
        "claim_boundary": "same Uniform factual stream, decision-time mixture NLL; not counterfactual navigation value",
        "contrasts": {name: _cluster_bootstrap({scene: values[name] for scene, values in scene_values.items()}, 901723 + index)
                      for index, name in enumerate(("uniform-minus-persistent", "reset-minus-persistent"))},
    }


def aggregate(
    inputs: list[Path],
    output: Path,
    *,
    expected_policies: list[str] | None = None,
    expected_seeds: list[str] | None = None,
    expected_scenarios: list[str] | None = None,
    expected_episodes: int | None = None,
    physics_audits: list[Path] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SocialGym2 aggregate: {output}")
    records: list[dict[str, Any]] = []
    evidence_rows = []
    source_metadata = []
    legacy_policy_role_sources = 0
    for path in inputs:
        metrics_path = path / "episode_metrics.csv"
        metadata_path = path / "metadata.json"
        manifest_path = path / "manifest.json"
        if not metrics_path.is_file() or not metadata_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"incomplete SocialGym2 artifact: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETED":
            raise ValueError(f"SocialGym2 artifact is not complete: {path}")
        for name in ("metadata.json", "episode_metrics.csv", "belief_evidence.csv"):
            artifact_path = path / name
            if not artifact_path.is_file() or manifest.get("artifacts", {}).get(name) != _sha256(artifact_path):
                raise ValueError(f"SocialGym2 artifact hash mismatch: {artifact_path}")
        source_trace = path / "belief_trace.csv"
        if source_trace.is_file() and manifest.get("artifacts", {}).get("belief_trace.csv") != _sha256(source_trace):
            raise ValueError(f"SocialGym2 artifact hash mismatch: {source_trace}")
        source = json.loads(metadata_path.read_text(encoding="utf-8"))
        if source.get("protocol") != "tracer-socialgym2-joint-belief-synchronous-v5":
            raise ValueError(
                f"artifact {path} uses unsupported protocol {source.get('protocol')!r}"
            )
        contract = source.get("joint_response_contract", {})
        required_contract = {
            "controllable_robots": 3,
            "response_checkpoint_robots": 3,
            "pair_interactions": True,
        }
        for field, expected in required_contract.items():
            if contract.get(field) != expected:
                raise ValueError(
                    f"artifact {path} is not a corrected joint-belief run: "
                    f"{field}={contract.get(field)!r}, expected {expected!r}"
                )
        declared_policies = source.get("parameters", {}).get("policies", [])
        declared_external_policies = sorted(set(declared_policies) & set(EXTERNAL_POLICIES))
        if declared_external_policies:
            controller = source.get("external_controller", {})
            if controller.get("policy") != declared_external_policies[0]:
                raise ValueError(f"artifact {path} has mismatched external controller provenance")
            if not controller.get("source_hashes"):
                raise ValueError(f"artifact {path} lacks official external-controller source hashes")
        policy_pair_contract = contract.get("pair_interactions_by_policy")
        if not isinstance(policy_pair_contract, dict):
            raise ValueError(f"artifact {path} does not declare pair interaction per policy")
        policy_role_schema = contract.get("policy_role_schema")
        if policy_role_schema is None:
            legacy_policy_role_sources += 1
            if contract.get("same_state_uniform_counterfactual") is not True:
                raise ValueError(
                    f"artifact {path} lacks the legacy same-state Uniform audit"
                )
            for policy in declared_policies:
                if policy_pair_contract.get(policy) is not (policy != "tracer-no-pair"):
                    raise ValueError(f"artifact {path} has an invalid legacy pair contract for {policy}")
        elif policy_role_schema == POLICY_ROLE_SCHEMA:
            expected_roles = socialgym2_policy_roles(declared_policies, True)
            actual_roles = contract.get("policy_roles_by_policy")
            if not isinstance(actual_roles, dict):
                raise ValueError(f"artifact {path} has invalid policy execution roles")
            for policy, expected_role in expected_roles.items():
                actual_role = actual_roles.get(policy)
                if not isinstance(actual_role, dict):
                    raise ValueError(f"artifact {path} has invalid policy execution roles")
                required_role = {
                    field: value
                    for field, value in expected_role.items()
                    if field != "external_source"
                }
                if any(actual_role.get(field) != value for field, value in required_role.items()):
                    raise ValueError(f"artifact {path} has invalid policy execution roles")
                if (
                    "external_source" in actual_role
                    and actual_role["external_source"] != expected_role.get("external_source")
                ):
                    raise ValueError(f"artifact {path} has invalid external policy source role")
            for policy in declared_policies:
                expected_pair = expected_roles[policy]["pair_interactions"]
                if policy_pair_contract.get(policy) is not expected_pair:
                    raise ValueError(f"artifact {path} has an invalid pair contract for {policy}")
            expected_counterfactual = any(
                role["same_state_uniform_counterfactual_evaluated"]
                for role in expected_roles.values()
            )
            if expected_counterfactual and contract.get("same_state_uniform_counterfactual") is not True:
                raise ValueError(
                    f"artifact {path} has an invalid same-state Uniform audit contract"
                )
        else:
            raise ValueError(f"artifact {path} has unsupported policy role schema {policy_role_schema!r}")
        if (source.get("runtime_contract") or {}).get("protocol") != "socialgym2-joint-synchronous-runtime-v5":
            raise ValueError(f"artifact {path} lacks the frozen synchronous runtime v5 contract")
        if not source.get("algorithm_sources") or not source.get("checkpoint_hashes"):
            raise ValueError(f"artifact {path} lacks source/checkpoint provenance")
        parameters = source.get("parameters", {})
        required_parameters = {
            "num_agents": 3,
            "num_humans": 2,
            "partial_observation": True,
        }
        for field, expected in required_parameters.items():
            if parameters.get(field) != expected:
                raise ValueError(
                    f"artifact {path} is not the declared multi-entity protocol: "
                    f"{field}={parameters.get(field)!r}, expected {expected!r}"
                )
        clock = source.get("clock_contract", {})
        if not math.isclose(float(clock.get("simulator_delta_t", 0.0)), 0.025, abs_tol=1e-7):
            raise ValueError(f"artifact {path} does not use the official 0.025 s simulator clock")
        if clock.get("simulator_steps_per_planning_window") != 16:
            raise ValueError(f"artifact {path} does not execute 16 simulator ticks per planning window")
        if not math.isclose(float(clock.get("planning_window_dt", 0.0)), 0.4, abs_tol=1e-7):
            raise ValueError(f"artifact {path} does not use the 0.4 s response-model clock")
        if clock.get("observed_transition_used_for_belief_update") is not True:
            raise ValueError(f"artifact {path} does not update belief from the observed transition")
        for field in ("zero_step_reset", "post_step_observation", "clock_verified_every_window", "pedestrian_rng_seeded_per_episode", "all_controllable_robots_exposed_to_pedestrians"):
            if clock.get(field) is not True:
                raise ValueError(f"artifact {path} lacks synchronous runtime invariant {field}")
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row.setdefault("scenario", source.get("parameters", {}).get("scenario", ""))
                if not row.get("scenario"):
                    raise ValueError(f"artifact {path} has no scenario field")
                clock_error = float(row.get("clock_max_abs_error", "nan"))
                if not math.isfinite(clock_error) or clock_error < 0 or clock_error > 2e-5:
                    raise ValueError(f"artifact {path} contains a clock-contract violation: {row}")
                final_ticks = int(row.get("last_window_sim_ticks", "0"))
                if not 1 <= final_ticks <= 16 or int(row["executed_sim_ticks"]) != 16 * (int(row["steps"]) - 1) + final_ticks:
                    raise ValueError(f"artifact {path} contains inconsistent execution ticks")
                if final_ticks < 16 and row.get("terminal_reached") != "True":
                    raise ValueError(f"artifact {path} truncates a nonterminal planning window")
                if row["policy"] not in source["parameters"].get("policies", []):
                    raise ValueError(f"artifact {path} contains an undeclared policy")
                policy_role = socialgym2_policy_roles([row["policy"]], True)[row["policy"]]
                if (
                    not policy_role["same_state_uniform_counterfactual_evaluated"]
                    and float(row.get("belief_selection_audit_windows", 0.0)) != 0.0
                ):
                    raise ValueError(
                        f"artifact {path} records a Uniform audit for external policy "
                        f"{row['policy']}"
                    )
                records.append(row)
        source_metadata.append(source)
        with (path / "belief_evidence.csv").open(newline="", encoding="utf-8") as handle:
            evidence_rows.extend(dict(row) for row in csv.DictReader(handle))
    if not records:
        raise ValueError("no SocialGym2 records to aggregate")
    comparable_fields = (
        "response_checkpoint", "sage_checkpoint", "max_steps", "model_dt", "env_dt",
        "human_cost_weight", "safety_weight", "collision_margin", "risk_alpha",
        "risk_samples", "hard_risk_filter", "risk_weight", "smoothness_weight",
        "likelihood_scale", "calibration_artifact", "belief_cost_leverage",
    )
    reference_parameters = source_metadata[0]["parameters"]
    reference_algorithm_sources = source_metadata[0]["algorithm_sources"]
    for source in source_metadata[1:]:
        for field in comparable_fields:
            if source["parameters"].get(field) != reference_parameters.get(field):
                raise ValueError(f"SocialGym2 artifacts mix incompatible {field} settings")
        if source.get("calibration_contract") != source_metadata[0].get("calibration_contract"):
            raise ValueError("SocialGym2 artifacts mix calibration contracts")
        for field in ("runtime_contract", "checkpoint_hashes"):
            if source.get(field) != source_metadata[0].get(field):
                raise ValueError(f"SocialGym2 artifacts mix incompatible {field} provenance")
        source_policies = source.get("parameters", {}).get("policies", [])
        if not set(source_policies).issubset(EXTERNAL_POLICIES):
            if source.get("algorithm_sources") != reference_algorithm_sources:
                raise ValueError(
                    "SocialGym2 non-external artifacts mix incompatible "
                    "algorithm_sources provenance"
                )

    policies = sorted({row["policy"] for row in records})
    scenarios = sorted({row["scenario"] for row in records})
    native_external_policies = sorted(set(policies) & set(EXTERNAL_POLICIES))
    external_controller_provenance: dict[str, dict[str, Any]] = {}
    for source in source_metadata:
        controller = source.get("external_controller", {})
        policy = controller.get("policy")
        if policy not in native_external_policies:
            continue
        hashes = controller.get("source_hashes")
        previous = external_controller_provenance.get(policy)
        if previous is not None and previous != hashes:
            raise ValueError(f"SocialGym2 artifacts mix source revisions for external policy {policy}")
        external_controller_provenance[policy] = hashes
    policy_keys = {
        policy: {_record_key(row) for row in records if row["policy"] == policy}
        for policy in policies
    }
    duplicate_policies = []
    for policy, keys in policy_keys.items():
        row_count = sum(row["policy"] == policy for row in records)
        if len(keys) != row_count:
            duplicate_policies.append(policy)
    if duplicate_policies:
        raise ValueError(f"duplicate paired rows for policies: {duplicate_policies}")
    common_keys = set.intersection(*(keys for keys in policy_keys.values())) if policy_keys else set()
    if not common_keys:
        raise ValueError("no common seed/scenario/episode keys across policies")
    incomplete_policies = {
        policy: sorted(keys ^ common_keys)
        for policy, keys in policy_keys.items()
        if keys != common_keys
    }
    if incomplete_policies:
        raise ValueError(
            "paired SocialGym2 aggregate requires identical seed/scenario/episode keys "
            f"for every policy: {incomplete_policies}"
        )
    if expected_policies is not None and policies != sorted(expected_policies):
        raise ValueError(f"formal SocialGym2 run has policies {policies}, expected {sorted(expected_policies)}")
    if expected_seeds is not None and sorted({row["seed"] for row in records}) != sorted(expected_seeds):
        raise ValueError("formal SocialGym2 run has an incomplete seed lattice")
    if expected_scenarios is not None and scenarios != sorted(expected_scenarios):
        raise ValueError("formal SocialGym2 run has an incomplete scenario lattice")
    expected_keys = None
    if expected_seeds is not None and expected_scenarios is not None and expected_episodes is not None:
        expected_keys = {
            (str(seed), scenario, str(episode))
            for seed in expected_seeds
            for scenario in expected_scenarios
            for episode in range(expected_episodes)
        }
        if common_keys != expected_keys:
            missing = sorted(expected_keys - common_keys)
            extra = sorted(common_keys - expected_keys)
            raise ValueError(f"formal SocialGym2 lattice mismatch: missing={missing[:8]}, extra={extra[:8]}")
        if len(records) != len(expected_keys) * len(policies):
            raise ValueError("formal SocialGym2 row count does not match the complete policy lattice")
    pairing_fields = (
        "episode_seed",
        "initial_state_hash",
        "initial_scene_snapshot",
        "initial_collision",
    )
    reference_rows = {
        _record_key(row): row
        for row in records
        if row["policy"] == "tracer-persistent"
    }
    if len(reference_rows) != len(common_keys):
        raise ValueError("Persistent reference rows are incomplete for pairing validation")
    expected_seed_mismatches = []
    for key, row in reference_rows.items():
        expected_seed = int(row["seed"]) * 1000 + int(row["episode"])
        if int(row.get("episode_seed", -1)) != expected_seed:
            expected_seed_mismatches.append((key, row.get("episode_seed"), expected_seed))
    if expected_seed_mismatches:
        raise ValueError(
            "paired SocialGym2 artifacts use overlapping or non-canonical episode seeds: "
            f"{expected_seed_mismatches[:8]}"
        )
    pairing_mismatches = []
    raw_scenario_hash_mismatches = []
    for policy in policies:
        for row in records:
            if row["policy"] != policy:
                continue
            key = _record_key(row)
            reference = reference_rows[key]
            for field in pairing_fields:
                observed = row.get(field)
                expected = reference.get(field)
                if field == "initial_scene_snapshot" and observed and expected:
                    try:
                        matched = json.loads(observed) == json.loads(expected)
                    except json.JSONDecodeError:
                        matched = False
                else:
                    matched = bool(observed) and observed == expected
                if not matched:
                    pairing_mismatches.append(
                        {
                            "policy": policy,
                            "key": key,
                            "field": field,
                            "observed": observed,
                            "reference": expected,
                        }
                    )
            if row.get("scenario_hash") != reference.get("scenario_hash"):
                raw_scenario_hash_mismatches.append({"policy": policy, "key": key})
    if pairing_mismatches:
        raise ValueError(
            "paired SocialGym2 artifacts do not share deterministic episode states: "
            f"{pairing_mismatches[:8]}"
        )
    scene_keys: dict[tuple[str, str], tuple[str, str, str]] = {}
    duplicate_scenes = []
    for key, row in reference_rows.items():
        scene_key = (row["scenario"], row["initial_state_hash"])
        previous = scene_keys.get(scene_key)
        if previous is not None and previous != key:
            duplicate_scenes.append((scene_key, previous, key))
        scene_keys[scene_key] = key
    if duplicate_scenes:
        raise ValueError(
            "paired SocialGym2 aggregate contains duplicate initial scenes across keys: "
            f"{duplicate_scenes[:8]}"
        )
    formal_requested = any(value is not None for value in (
        expected_policies, expected_seeds, expected_scenarios, expected_episodes,
    ))
    physical_evidence = []
    if formal_requested:
        if any(value is None for value in (
            expected_policies, expected_seeds, expected_scenarios, expected_episodes,
        )):
            raise ValueError("formal aggregation requires the complete prespecified lattice")
        if not source_metadata[0].get("calibration_contract", {}).get("artifact_sha256"):
            raise ValueError("formal aggregation requires frozen validation calibration")
        physical_evidence = verify_physics_audits(
            physics_audits or [], source_metadata[0]["runtime_contract"], scenarios,
            minimum_ticks=int(reference_parameters["max_steps"]) * 16,
        )
    output.mkdir(parents=True)
    metrics_path = output / "episode_metrics.csv"
    metric_fields = list(records[0])
    for row in records[1:]:
        metric_fields.extend(field for field in row if field not in metric_fields)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(records)
    summary: dict[str, Any] = {
        "policies": {},
        "records": len(records),
        "seeds": sorted({row["seed"] for row in records}),
        "scenarios": scenarios,
        "paired_keys": len(common_keys),
        "pairing_fields": list(pairing_fields),
        "raw_scenario_hash_mismatch_count": len(raw_scenario_hash_mismatches),
        "raw_scenario_hash_boundary": (
            "Generated scenario files embed worker-specific absolute paths, so raw hashes are "
            "retained but not compared across isolated worker roots. Pairing instead requires "
            "canonical episode seeds, semantically identical full reset snapshots, reset-state "
            "hashes, and reset collision labels."
        ),
        "run_class": "frozen_complete_lattice" if formal_requested else "development_partial_lattice",
    }
    for index, policy in enumerate(policies):
        selected = [row for row in records if row["policy"] == policy]
        item = {}
        for metric_index, metric in enumerate(METRICS):
            available = [row for row in selected if metric in row and row[metric] != ""]
            if not available:
                continue
            item[metric] = _bootstrap(
                [float(row[metric]) for row in available],
                849731 + index * 31 + metric_index,
            )
            by_seed: dict[str, list[float]] = {}
            for row in available:
                by_seed.setdefault(str(row["seed"]), []).append(float(row[metric]))
            item[f"{metric}_seed_cluster"] = _cluster_bootstrap(
                by_seed,
                859731 + index * 31 + metric_index,
            )
            by_seed_scenario: dict[str, list[float]] = {}
            for row in available:
                by_seed_scenario.setdefault(
                    f"{row['seed']}::{row['scenario']}", []
                ).append(float(row[metric]))
            item[f"{metric}_seed_scenario_cluster"] = _cluster_bootstrap(
                by_seed_scenario,
                869731 + index * 31 + metric_index,
            )
            by_scene: dict[str, list[float]] = {}
            for row in available:
                by_scene.setdefault(
                    f"{row['scenario']}::{row['initial_state_hash']}", []
                ).append(float(row[metric]))
            item[f"{metric}_scene_cluster"] = _cluster_bootstrap(
                by_scene,
                879731 + index * 31 + metric_index,
            )
            if metric in {"success", "collision", "team_success", "collision_free_team_success", "initial_collision"}:
                item[f"{metric}_wilson"] = _wilson_interval([float(row[metric]) for row in available])
        item["count"] = len(selected)
        item["by_scenario"] = {}
        for scenario_index, scenario in enumerate(scenarios):
            scenario_rows = [row for row in selected if row["scenario"] == scenario]
            scenario_item = {"count": len(scenario_rows)}
            for metric_index, metric in enumerate(METRICS):
                available = [
                    row for row in scenario_rows
                    if metric in row and row[metric] != ""
                ]
                if available:
                    scenario_item[metric] = _bootstrap(
                        [float(row[metric]) for row in available],
                        879731 + index * 31 + scenario_index * 17 + metric_index,
                    )
            item["by_scenario"][scenario] = scenario_item
        if policy in native_external_policies:
            action_counts = np.zeros(4, dtype=np.int64)
            for row in selected:
                raw_counts = json.loads(row.get("external_raw_action_counts", "[]"))
                if len(raw_counts) != 4 or any(int(value) < 0 for value in raw_counts):
                    raise ValueError(f"invalid native action counts for {policy}: {raw_counts}")
                action_counts += np.asarray(raw_counts, dtype=np.int64)
            total_actions = int(action_counts.sum())
            item["native_action_usage"] = {
                name: {
                    "count": int(action_counts[action]),
                    "rate": (
                        float(action_counts[action] / total_actions)
                        if total_actions else 0.0
                    ),
                }
                for action, name in enumerate(("GoAlone", "Halt", "Follow", "Pass"))
            }
            item["native_action_count"] = total_actions
        summary["policies"][policy] = item

    persistent = {
        _record_key(row): row
        for row in records
        if row["policy"] == "tracer-persistent"
    }
    contrasts = {}
    for policy in policies:
        if policy == "tracer-persistent":
            continue
        compared = {
            _record_key(row): row
            for row in records
            if row["policy"] == policy
        }
        keys = sorted(persistent.keys() & compared.keys())
        effect_metrics = (
            "task_progress",
            "collision",
            "success",
            "team_success",
            "robot_success_fraction",
            "collision_free_team_success",
            "min_distance",
            "belief_selection_change_rate",
            "belief_action_change_rate",
            "belief_score_l1",
            "belief_selected_regret",
        )
        for metric in effect_metrics:
            if not keys or any(metric not in compared[key] or metric not in persistent[key] for key in keys):
                continue
            deltas = {
                key: float(compared[key][metric]) - float(persistent[key][metric])
                for key in keys
            }
            episode_result = _bootstrap(
                list(deltas.values()),
                900001 + len(contrasts),
            )
            by_seed: dict[str, list[float]] = {}
            for (seed, _scenario, _episode), delta in deltas.items():
                by_seed.setdefault(str(seed), []).append(delta)
            by_seed_scenario: dict[str, list[float]] = {}
            for (seed, scenario, _episode), delta in deltas.items():
                by_seed_scenario.setdefault(f"{seed}::{scenario}", []).append(delta)
            seed_means = [float(np.mean(values)) for values in by_seed.values()]
            seed_scenario_means = [
                float(np.mean(values)) for values in by_seed_scenario.values()
            ]
            by_scene: dict[str, list[float]] = {}
            for (seed, scenario, episode), delta in deltas.items():
                reference = persistent[(seed, scenario, episode)]
                scene_key = f"{scenario}::{reference['initial_state_hash']}"
                by_scene.setdefault(scene_key, []).append(delta)
            scene_means = [float(np.mean(values)) for values in by_scene.values()]
            contrasts[f"{policy}-persistent:{metric}"] = {
                "episode_bootstrap": episode_result,
                "scene_cluster_bootstrap": _cluster_bootstrap(
                    by_scene,
                    905001 + len(contrasts),
                ),
                "seed_cluster_bootstrap": _cluster_bootstrap(
                    by_seed,
                    910001 + len(contrasts),
                ),
                "seed_scenario_cluster_bootstrap": _cluster_bootstrap(
                    by_seed_scenario,
                    920001 + len(contrasts),
                ),
                "per_seed_mean": {
                    seed: float(np.mean(values)) for seed, values in sorted(by_seed.items())
                },
                "per_scenario_mean": {
                    scenario: float(np.mean([
                        delta for (_seed, row_scenario, _episode), delta in deltas.items()
                        if row_scenario == scenario
                    ]))
                    for scenario in scenarios
                    if any(
                        row_scenario == scenario
                        for (_seed, row_scenario, _episode) in deltas
                    )
                },
                "sign_test": _two_sided_sign_test(scene_means),
                "sign_test_seed_cluster": _two_sided_sign_test(seed_means),
                "sign_test_seed_scenario_cluster": _two_sided_sign_test(seed_scenario_means),
                "sign_test_episode_descriptive": _two_sided_sign_test(list(deltas.values())),
            }
    summary["paired_contrasts"] = contrasts
    summary["factual_memory_replay"] = factual_memory_contrasts(
        evidence_rows, float(reference_parameters.get("likelihood_scale", 0.05)),
    )
    clean_keys = {key for key in common_keys if float(persistent[key]["initial_collision"]) == 0}
    summary["initial_condition_strata"] = {
        "primary": "all paired scenes, including initial collisions",
        "initial_collision_free_scene_count": len(clean_keys),
        "initial_collision_scene_count": len(common_keys) - len(clean_keys),
        "secondary_initial_collision_free_contrasts": {},
    }
    for policy in policies:
        if policy == "tracer-persistent":
            continue
        compared = {_record_key(row): row for row in records if row["policy"] == policy}
        for metric in ("task_progress", "collision_free_team_success", "team_success", "collision"):
            differences = [float(compared[key][metric]) - float(persistent[key][metric]) for key in sorted(clean_keys)]
            summary["initial_condition_strata"]["secondary_initial_collision_free_contrasts"][f"{policy}-persistent:{metric}"] = _bootstrap(differences, 917231)
    navigation_family = {
        name: result["sign_test"]["p_two_sided"]
        for name, result in contrasts.items()
        if name.rsplit(":", 1)[-1] in {"task_progress", "collision", "team_success", "collision_free_team_success"}
    }
    for name, probability in _holm_adjust(navigation_family).items():
        contrasts[name]["sign_test"]["p_holm_navigation_family"] = probability
    summary["inference_boundary"] = {
        "noninferiority_established": False,
        "noninferiority_margin": None,
        "reason": "no prespecified noninferiority margin; nonsignificance is not equivalence",
        "sign_test_target": "direction consistency of paired scene differences, not a test of mean differences",
        "navigation_sign_test_family_size": len(navigation_family),
        "confidence_intervals": "pointwise; binary proportions additionally report Wilson intervals",
        "generalization": "sampled scenes in the evaluated maps with one fixed training checkpoint",
    }

    metadata = {
        "protocol": "tracer-socialgym2-joint-belief-h3-clustered-aggregate-v5",
        "run_class": summary["run_class"],
        "aggregation_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "physics_audits": physical_evidence,
        "frozen_inputs": {field: source_metadata[0][field] for field in (
            "algorithm_sources", "runtime_contract", "checkpoint_hashes",
        )},
        "algorithm_provenance_by_policy": {
            policy: sorted({
                json.dumps(source["algorithm_sources"], sort_keys=True)
                for source in source_metadata
                if policy in source.get("parameters", {}).get("policies", [])
            })
            for policy in policies
        },
        "claim_boundary": (
            "Aggregated paired multi-robot SocialGym2 runs with native official "
            "SocialGym2 external-controller rows and frozen binary TRACER rows; "
            "external rows use GoAlone/Halt/Follow/Pass while TRACER rows use "
            "GoAlone/Halt; behavior-response/navigation integration evidence only, "
            "not real-human causality or continuous-action H3 equivalence"
            if native_external_policies
            else
            "Aggregated paired multi-robot SocialGym2 runs; joint response checkpoint "
            "with pair interactions, official discrete GoAlone/Halt action projection, "
            "and ID-free spatial track association; behavior-response/navigation "
            "integration evidence only, not real-human causality or continuous-action "
            "H3 equivalence"
        ),
        "source_artifacts": [
            {
                "path": str(path),
                "metadata_sha256": _sha256(path / "metadata.json"),
                "metrics_sha256": _sha256(path / "episode_metrics.csv"),
                "seed": source.get("parameters", {}).get("seed"),
            }
            for path, source in zip(inputs, source_metadata)
        ],
        "policies": policies,
        "external_controller_provenance": external_controller_provenance,
        "policy_role_schema": POLICY_ROLE_SCHEMA,
        "policy_roles": socialgym2_policy_roles(policies, True),
        "source_policy_role_metadata": {
            "legacy_sources": legacy_policy_role_sources,
            "corrected_sources": len(source_metadata) - legacy_policy_role_sources,
            "legacy_boundary": (
                "legacy v5 sources mislabeled official controls as pair-interaction policies; "
                "use a hash-bound policy-role annotation"
                if legacy_policy_role_sources else None
            ),
        },
        "pairing_audit": {
            "verified_fields": list(pairing_fields),
            "raw_scenario_hash_mismatch_count": len(raw_scenario_hash_mismatches),
            "raw_scenario_hash_boundary": summary["raw_scenario_hash_boundary"],
        },
        "scenarios": scenarios,
        "records": len(records),
        "seeds": sorted({row["seed"] for row in records}),
        "independent_scene_clusters": len({
            (row["scenario"], row["initial_state_hash"])
            for row in reference_rows.values()
        }),
        "paired_keys": len(common_keys),
        "episodes_per_seed_scenario": {
            f"{seed}::{scenario}": sum(
                1
                for row in records
                if row["policy"] == "tracer-persistent"
                and row["seed"] == seed
                and row["scenario"] == scenario
            )
            for seed in sorted({row["seed"] for row in records})
            for scenario in scenarios
        },
        "cluster_bootstrap": {
            "primary_cluster_unit": "unique initial scene (scenario + initial_state_hash)",
            "secondary_cluster_unit": "environment seed block x scenario",
            "model_seed": "fixed response checkpoint seed2027; not a sampled nuisance cluster",
            "reason": "the corrected seed stream makes initial scenes disjoint; source seed blocks remain a conservative secondary cluster",
        },
    }
    metadata_path = output / "metadata.json"
    summary_path = output / "summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    evidence_path = output / "belief_evidence.csv"
    with evidence_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(evidence_rows[0]) if evidence_rows else ["policy", "step"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evidence_rows)
    trace_path = output / "belief_trace.csv"
    trace_fields: list[str] = []
    for path in inputs:
        source_trace = path / "belief_trace.csv"
        if not source_trace.is_file():
            continue
        with source_trace.open(newline="", encoding="utf-8") as source_handle:
            reader = csv.DictReader(source_handle)
            trace_fields.extend(
                field for field in (reader.fieldnames or []) if field not in trace_fields
            )
    if "scenario" not in trace_fields:
        trace_fields.append("scenario")
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trace_fields)
        writer.writeheader()
        for path, source in zip(inputs, source_metadata):
            source_trace = path / "belief_trace.csv"
            if not source_trace.is_file():
                continue
            with source_trace.open(newline="", encoding="utf-8") as source_handle:
                reader = csv.DictReader(source_handle)
                for row in reader:
                    source_scenario = source.get("parameters", {}).get("scenario", "")
                    if row.get("scenario") and row["scenario"] != source_scenario:
                        raise ValueError(f"artifact {path} contains mixed trace scenarios")
                    row["scenario"] = source_scenario
                    writer.writerow(row)
    report_lines = [
        "# TRACER on a shared patched SocialGym2 runtime: joint H3 paired aggregate",
        "",
        f"Seeds: `{', '.join(metadata['seeds'])}`; scenarios: `{', '.join(scenarios)}`; records: `{len(records)}`; paired keys: `{len(common_keys)}`.",
        (
            "The official SocialGym2 simulator supplies ROS reset/step, graph navigation, "
            "and pedsim humans. Native PIPS rows execute GoAlone/Halt/Follow/Pass, while "
            "TRACER rows use the frozen GoAlone/Halt interface."
            if native_external_policies
            else
            "The official SocialGym2 simulator supplies ROS reset/step, graph navigation, "
            "pedsim humans, and the binary GoAlone/Halt action space."
        ),
        "Applicable TRACER policies use a joint multi-robot response model, SAGE proposal ordering, per-track belief updates, and risk-aware branch selection; external controls bypass response and belief evaluation.",
        "All policies share opt-in reset, simulation-clock, seeded RNG, stable force-summation order, laser-state cleanup, lifecycle, and multi-robot state-transport patches. This is not an unmodified official reproduction.",
        "",
        "## Results",
        "",
    ]
    for policy in policies:
        item = summary["policies"][policy]
        report_lines.append(
            f"- `{policy}` (n={item['count']}): team success "
            f"`{item['team_success']['mean']:.3f}`, collision-free team success "
            f"`{item['collision_free_team_success']['mean']:.3f}`, collision "
            f"`{item['collision']['mean']:.3f}`, progress "
            f"`{item['task_progress']['mean']:.5f}`, min distance "
            f"`{item['min_distance']['mean']:.3f}`."
        )
        if "native_action_usage" in item:
            usage = item["native_action_usage"]
            report_lines.append(
                "  Native actions: "
                + ", ".join(
                    f"{name} `{usage[name]['rate']:.3f}`"
                    for name in ("GoAlone", "Halt", "Follow", "Pass")
                )
                + "."
            )
    for name, contrast in summary["paired_contrasts"].items():
        if name.rsplit(":", 1)[-1] in {
            "task_progress",
            "collision",
            "team_success",
            "collision_free_team_success",
        }:
            primary = contrast["scene_cluster_bootstrap"]
            report_lines.append(
                f"- paired `{name}`: mean `{primary['mean']:.5f}`, scene-cluster 95% CI `[{primary['ci95_low']:.5f}, {primary['ci95_high']:.5f}]`, sign-test p `{contrast['sign_test']['p_two_sided']:.4g}`."
            )
    report_lines.extend([
        "",
        "Paired contrasts use exact seed/scenario/episode keys. Results include descriptive episode bootstrap, primary unique-scene cluster bootstrap, secondary environment-seed and seed×scenario cluster bootstraps, per-seed/per-scenario effects, and cluster-level two-sided sign tests; the scene-level sign test is primary.",
        "Pairing is additionally verified by episode seed, full reset snapshot, rounded initial robot/human state hash, and reset collision label; incomplete or mismatched policy artifacts are rejected. Raw generated-scenario hashes are retained but not compared across isolated worker roots because generated files embed worker-specific absolute paths.",
        "The joint response contract requires three controllable robots, a matching three-robot response checkpoint, and same-state Uniform-belief counterfactual auditing over the complete binary action lattice for applicable TRACER policies. Per-policy execution roles distinguish pair/no-pair TRACER variants from official controls.",
        (
            "The native external-controller action interface and ID-free track association "
            "are explicit boundaries; this aggregate is not presented as continuous-action "
            "H3 replication."
            if native_external_policies
            else
            "The binary action projection and ID-free track association are explicit "
            "external-adapter boundaries; this aggregate is not presented as continuous-action "
            "H3 replication."
        ),
        f"Initial-collision-free secondary stratum: {summary['initial_condition_strata']['initial_collision_free_scene_count']} scenes. All paired scenes remain in the primary analysis.",
        f"Same-Uniform-stream memory replay: {summary['factual_memory_replay']['status']}; {summary['factual_memory_replay']['independent_observed_scenes']} observed scenes. Its NLL contrasts are prediction evidence, not counterfactual navigation value.",
        "",
    ])
    report_path = output / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    artifacts = {
        name: _sha256(output / name)
        for name in ("metadata.json", "summary.json", "episode_metrics.csv", "belief_trace.csv", "belief_evidence.csv", "report.md")
    }
    manifest = {
        "status": "COMPLETED",
        "protocol": metadata["protocol"],
        "records": len(records),
        "seeds": metadata["seeds"],
        "artifacts": artifacts,
        "claim_boundary": metadata["claim_boundary"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"metadata": metadata, "summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-policies", nargs="+")
    parser.add_argument("--expected-seeds", nargs="+")
    parser.add_argument("--expected-scenarios", nargs="+")
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--physics-audits", nargs="+", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    result = aggregate(
        args.inputs,
        Path(args.output_dir),
        expected_policies=args.expected_policies,
        expected_seeds=args.expected_seeds,
        expected_scenarios=args.expected_scenarios,
        expected_episodes=args.expected_episodes,
        physics_audits=args.physics_audits,
    )
    print(json.dumps({"status": result["manifest"]["status"], "records": result["manifest"]["records"], "seeds": result["manifest"]["seeds"]}, indent=2))


if __name__ == "__main__":
    main()
