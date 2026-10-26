"""Calibrate dense external evidence on factual, uniform-controller validation tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_SCALES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _case_list(values: Any, name: str) -> list[int]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must disclose nonempty case IDs")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{name} must contain nonnegative integer case IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate cases")
    return sorted(values)


def _validate_splits(case_splits: dict) -> dict[str, list[int]]:
    splits = {name: _case_list(case_splits.get(name), name) for name in ("train", "val", "test")}
    for first, second in (("train", "val"), ("val", "test"), ("train", "test")):
        if set(splits[first]) & set(splits[second]):
            raise ValueError(f"{first}/{second} case overlap")
    return splits


def validation_case_splits(checkpoint: dict, parameters: dict, cases=None) -> dict:
    """Derive collection cases from the checkpoint, never from navigation results."""
    splits = dict(checkpoint.get("provenance", {}).get("case_splits", {}))
    start, count = parameters.get("case_start"), parameters.get("cases")
    if type(start) is not int or start < 0 or type(count) is not int or count < 1:
        raise ValueError("controller metadata must disclose the excluded test case range")
    held_out = set(range(start, start + count))
    if "test" in splits:
        held_out.update(_case_list(splits["test"], "checkpoint test"))
    splits["test"] = sorted(held_out)
    splits = _validate_splits(splits)
    if cases is not None and _case_list(cases, "requested cases") != splits["val"]:
        raise ValueError("collection requires exactly checkpoint val cases; train/test cases are forbidden")
    return splits


def validate_evidence(rows: list[dict], episodes: list[dict], case_splits: dict, seed: int) -> list[dict]:
    """Require one complete factual prefix per tick; never silently deduplicate."""
    splits = _validate_splits(case_splits)
    episode_lookup = {}
    for episode in episodes:
        case = episode.get("case")
        if case not in splits["val"] or episode.get("seed") != seed:
            raise ValueError("episodes must use the checkpoint seed and validation cases only")
        if case in episode_lookup:
            raise ValueError("duplicate episode evidence")
        if episode.get("policy") != "tracer-uniform":
            raise ValueError("collection controller must be frozen uniform")
        windows = episode.get("windows")
        if type(windows) is not int or windows < 1:
            raise ValueError("episode must disclose its factual window count")
        if episode.get("steps") != windows or episode.get("belief_update_windows") != windows:
            raise ValueError("stride1 requires one executed step and one evidence update per window")
        if episode.get("environment_dt") != 0.25 or episode.get("robot_kinematics") != "unicycle":
            raise ValueError("episode motion/dt contract does not match the frozen controller")
        human_count = episode.get("human_num")
        if type(human_count) is not int or human_count < 1:
            raise ValueError("episode must disclose a positive human count")
        episode_lookup[case] = episode
    if sorted(episode_lookup) != splits["val"]:
        raise ValueError("missing validation episodes")
    grouped = {case: {} for case in splits["val"]}
    for row in rows:
        case, window = row.get("case"), row.get("window")
        if case not in grouped or row.get("seed") != seed:
            raise ValueError("raw evidence must come from validation cases only, never train/test")
        if type(window) is not int or window < 0:
            raise ValueError("evidence window must be a nonnegative integer")
        if window in grouped[case]:
            raise ValueError("duplicate factual evidence window")
        if row.get("policy") != "tracer-uniform" or any(name in row for name in ("branch", "selected_branch")):
            raise ValueError("only factual uniform-controller evidence is allowed, not cloned branches")
        if row.get("factual_response_steps") != 1 or row.get("belief_updated_this_window") != 1:
            raise ValueError("missing prefix1/stride1 factual evidence")
        expected = (episode_lookup[case]["human_num"], 3)
        for name in ("factual_mode_log_likelihood", "decision_log_beliefs", "posterior_log_beliefs"):
            try:
                values = np.asarray(row[name], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"missing or malformed {name}; enable belief_evidence_audit") from error
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite human×mode evidence with shape {expected}")
            if name != "factual_mode_log_likelihood" and not np.allclose(values, -math.log(3), rtol=0, atol=1e-6):
                raise ValueError("uniform collection must not condition decisions on inferred or hidden labels")
        grouped[case][window] = row
    ordered = []
    for case, episode in episode_lookup.items():
        if sorted(grouped[case]) != list(range(episode["windows"])):
            raise ValueError(f"missing windows or out-of-range evidence in case {case}")
        ordered.extend(grouped[case][window] for window in range(episode["windows"]))
    return ordered


def build_track_tensors(rows: list[dict], labels: list[dict]):
    tracks = {}
    for row in rows:
        for human, likelihood in enumerate(row["factual_mode_log_likelihood"]):
            track = tracks.setdefault((row["case"], human), {"windows": [], "likelihoods": []})
            track["windows"].append(row["window"])
            track["likelihoods"].append(likelihood)
    label_lookup = {}
    for label in labels:
        key = (label["case"], label["human"])
        if key in label_lookup or type(label["mode"]) is not int or not 0 <= label["mode"] < 3:
            raise ValueError("invalid or duplicate offline mode labels")
        label_lookup[key] = label["mode"]
    if set(tracks) != set(label_lookup):
        raise ValueError("offline labels must match every factual validation track exactly")
    longest = max(len(track["windows"]) for track in tracks.values())
    likelihoods = torch.zeros(len(tracks), longest, 3)
    active = torch.zeros(len(tracks), longest, dtype=torch.bool)
    modes = torch.zeros(len(tracks), dtype=torch.long)
    sequences = []
    for index, ((case, human), track) in enumerate(sorted(tracks.items())):
        length = len(track["windows"])
        if track["windows"] != list(range(length)):
            raise ValueError("missing windows or duplicate evidence in a validation track")
        likelihoods[index, :length] = torch.tensor(track["likelihoods"])
        active[index, :length] = True
        modes[index] = label_lookup[case, human]
        sequences.append({"case": case, "human": human, "offline_mode": label_lookup[case, human], **track})
    return likelihoods, modes, active, sequences


def select_validation_scale(likelihoods, modes, active, *, phase, split, cases, case_splits, scales=DEFAULT_SCALES,
                            selection_metric="decision_mode_nll"):
    """Select by a disclosed pre-observation objective on factual validation tracks."""
    if phase != "validation" or split != "val":
        raise ValueError("scale selection requires phase=validation and split=val")
    splits = _validate_splits(case_splits)
    if _case_list(cases, "evidence cases") != splits["val"]:
        raise ValueError("scale selection requires exactly checkpoint validation cases")
    if active.ndim != 2 or not active[:, 0].all() or (active[:, 1:] & ~active[:, :-1]).any():
        raise ValueError("active evidence must be contiguous from window zero")
    from sage.experiments.calibrate_persistent_belief import calibration_scores
    from sage.belief.filter import run_categorical_filter

    if selection_metric not in {"decision_mode_nll", "decision_response_nll"}:
        raise ValueError("unsupported validation selection metric")
    records = calibration_scores(likelihoods, modes, active, list(scales))
    decision_mask = active.clone()
    decision_mask[:, 0] = False
    for record in records:
        decision_belief, _ = run_categorical_filter(likelihoods * record["likelihood_scale"], observation_mask=active)
        response_nll = -torch.logsumexp(decision_belief + likelihoods, -1) / 2
        record["decision_response_nll"] = float(response_nll[decision_mask].mean())
    selected = min(records, key=lambda record: (record[selection_metric], record["likelihood_scale"]))
    before, _ = run_categorical_filter(likelihoods * selected["likelihood_scale"], observation_mask=active)
    uniform = torch.full_like(likelihoods, -math.log(3))
    reset = uniform.clone()
    reset[:, 1:] = (likelihoods[:, :-1] * selected["likelihood_scale"]).log_softmax(-1)
    decision_mask = active.clone()
    decision_mask[:, 0] = False
    targets = modes[:, None, None].expand(-1, likelihoods.shape[1], 1)
    controls = {}
    for policy, weights in (("persistent", before), ("reset", reset), ("uniform", uniform)):
        controls[policy] = {
            "decision_mode_nll": float(-weights.gather(-1, targets).squeeze(-1)[decision_mask].mean()),
            "decision_mode_accuracy": float((weights.argmax(-1) == modes[:, None])[decision_mask].float().mean()),
            "factual_prefix_mixture_nll_per_coordinate": float((-torch.logsumexp(weights + likelihoods, -1) / 2)[active].mean()),
        }
    return {"selection_metric": selection_metric, "selected": selected, "records": records,
            "uniform_mode_nll": math.log(3), "same_factual_stream_controls": controls}


def _controller_args(parameters, checkpoint_path, output_dir, splits, seed, device):
    for name, value in (("execute_steps", 1), ("env_dt", 0.25), ("model_dt", 0.25),
                        ("motion_profile", "sicnav-unicycle"), ("response_backend", "crowdsimplus-persistent")):
        if parameters.get(name) != value:
            raise ValueError(f"reference controller must use {name}={value}")
    if not parameters.get("static_braking_guard"):
        raise ValueError("reference controller must preserve its static braking guard")
    if not isinstance(parameters.get("geometric_safety_prefix_steps"), int) or parameters["geometric_safety_prefix_steps"] < 1:
        raise ValueError("reference controller must guard the executed prefix")
    values = dict(parameters)
    values.update(
        checkpoint_dir=str(checkpoint_path.parent.parent),
        external_response_checkpoint_dir=str(checkpoint_path.parent.parent),
        output_dir=str(output_dir), device=device, seeds=[seed], policies=["tracer-uniform"],
        cases=len(splits["val"]), case_start=min(splits["val"]),
        belief_update_stride=1, belief_evidence_audit=True, likelihood_scale=1.0,
        belief_cadence_source="validation_factual_collection", allow_belief_cadence_override=False,
        belief_selection_audit=False, response_audit_stride=0, candidate_debug_windows=0, ranking_audit_windows=0,
        allow_temporal_transfer=False,
    )
    return argparse.Namespace(**values)


def recover_validation_labels(episodes, splits, parameters, thresholds):
    """Read hidden speed only after collection, on separate hash-matched resets."""
    from sage.experiments.navigation_contract import initial_navigation_state
    from sage.experiments.run_sicnav_tracer_loop import _new_episode

    labels = []
    for episode in episodes:
        if episode["case"] not in splits["val"]:
            raise ValueError("offline recovery may read validation cases only")
        env, robot, _, obstacles, *_ = _new_episode(
            Path(parameters.external_root).resolve(), episode["case"], Path(parameters.env_config).resolve(),
            parameters.human_num, parameters.motion_profile,
        )
        try:
            observed = initial_navigation_state(env, robot, obstacles)
            if observed["initial_scene_hash"] != episode["initial_scene_hash"] or len(env.humans) != episode["human_num"]:
                raise ValueError("offline label reset does not match the factual validation scene")
            for human_index, human in enumerate(env.humans):
                preferred_speed = float(human.v_pref)
                if not math.isfinite(preferred_speed):
                    raise ValueError("offline preferred speed must be finite")
                labels.append({"case": episode["case"], "human": human_index,
                               "mode": int(np.searchsorted(thresholds, preferred_speed, side="right")),
                               "preferred_speed_offline_only": preferred_speed,
                               "initial_scene_hash": observed["initial_scene_hash"]})
        finally:
            env.close()
    return labels


@torch.no_grad()
def calibrate_external_cadence(checkpoint_path, controller_metadata_path, output_dir, *, device="cpu", seed=None,
                               cases=None, phase="validation", split="val", selection_metric="decision_mode_nll"):
    if phase != "validation" or split != "val":
        raise ValueError("collection and selection require phase=validation, split=val")
    checkpoint_path, controller_metadata_path, output_dir = (
        Path(path).resolve() for path in (checkpoint_path, controller_metadata_path, output_dir)
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    metadata_hash = sha256_file(controller_metadata_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parameters = json.loads(controller_metadata_path.read_text())["parameters"]
    splits = validation_case_splits(checkpoint, parameters, cases)
    if seed is None:
        seed = checkpoint["seed"]
    if seed != checkpoint.get("seed"):
        raise ValueError("collection seed must match the response checkpoint seed")
    config = checkpoint["model_config"]
    if (config.get("dt"), config.get("num_robots"), config.get("num_modes"), config.get("response_delay_steps")) != (0.25, 1, 3, 1):
        raise ValueError("checkpoint requires dt=.25, one robot, three modes and response delay one")
    if config.get("evidence_conditioning") or not checkpoint.get("mode_semantics"):
        raise ValueError("checkpoint must disclose modes and use no untracked evidence inputs")
    thresholds = np.asarray(checkpoint.get("speed_thresholds_train_only"), dtype=np.float64)
    if thresholds.shape != (2,) or not np.isfinite(thresholds).all() or not thresholds[0] < thresholds[1]:
        raise ValueError("checkpoint must contain two ordered training-only speed thresholds")
    args = _controller_args(parameters, checkpoint_path, output_dir, splits, seed, device)
    if output_dir.exists():
        raise ValueError("calibration output exists; use a new directory")
    output_dir.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    source_hashes = {}
    for source in sorted(root.rglob("*.py")):
        relative = source.relative_to(root)
        content = source.read_bytes()
        snapshot = output_dir / "source_snapshot" / "sage" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
        source_hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    _write_json(output_dir / "source_hashes.json", source_hashes)
    _write_json(output_dir / "controller_parameters.json", vars(args))
    input_hashes = {str(path): sha256_file(path) for path in (
        checkpoint_path, controller_metadata_path, Path(args.sage_checkpoint).resolve(),
        Path(args.env_config).resolve(), Path(args.motion_policy_config).resolve(),
    )}
    contract = {
        "protocol": "external-factual-prefix-cadence-v1", "phase": phase, "split": split,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "seed": seed, "execute_steps": 1, "observation_stride": 1, "factual_response_steps": 1,
        "dt": 0.25, "num_robots": 1, "num_modes": 3, "case_splits": splits,
        "validation_cases": splits["val"], "controller_policy": "tracer-uniform",
        "controller_parameters": vars(args),
        "controller_parameters_sha256": sha256_file(output_dir / "controller_parameters.json"),
        "controller_metadata_path": str(controller_metadata_path), "controller_metadata_sha256": metadata_hash,
        "controller_source_sha256": source_hashes["experiments/run_sicnav_tracer_loop.py"],
        "source_hashes": source_hashes, "input_hashes": input_hashes,
        "speed_thresholds_train_only": thresholds.tolist(), "mode_semantics": checkpoint["mode_semantics"],
        "previous_checkpoint_calibration": checkpoint.get("belief_calibration"),
        "boundary": "Validation-only frozen-uniform factual first-step evidence; labels recovered offline after all rollouts; no navigation superiority claim.",
    }
    _write_json(output_dir / "metadata.json", contract)
    _write_json(output_dir / "manifest.json", {"status": "COLLECTING", "phase": phase, "split": split})
    from sage.experiments.eval_persistent_closed_loop import _load_model
    from sage.experiments.navigation_contract import validate_response_clock
    from sage.experiments.run_sicnav_tracer_loop import _run_episode
    from sage.response.sage_candidate_adapter import SAGECandidateAdapter, SAGECandidateAdapterConfig

    model = _load_model(checkpoint_path, device, expected_dt=0.25)
    validate_response_clock(model, 0.25, False)
    adapter = None
    if args.candidate_profile in {"sage", "sage-hybrid", "sage-adaptive", "sage-recovery"}:
        adapter = SAGECandidateAdapter.from_checkpoint(
            Path(args.sage_checkpoint).resolve(), device,
            SAGECandidateAdapterConfig(
                num_candidates=args.num_candidates, guided=True, step_stride=args.sage_step_stride,
                goal_speed=args.sage_goal_speed, oversample_factor=args.sage_oversample_factor,
                task_prefilter=True, task_prefix_steps=args.execute_steps, goal_direction="task_goal",
                include_task_anchor=True, task_anchor_speed_scales=(0.45,), batch_size=args.num_candidates,
                guidance_step_clip_norm=0.1, diversity_weight=getattr(args, "sage_diversity_weight", 0.0),
            ),
        )
    rows, episodes = [], []
    with (output_dir / "raw_evidence.jsonl").open("x") as handle:
        for case in splits["val"]:
            case_rows = []
            print(f"validation_factual_uniform seed={seed} case={case} stride=1 prefix=1", flush=True)
            episode = _run_episode(
                model, adapter, "tracer-uniform", Path(args.external_root).resolve(), case,
                Path(args.env_config).resolve(), args, seed, window_audit_rows=case_rows,
            )
            episodes.append(episode)
            rows.extend(case_rows)
            for row in case_rows:
                handle.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            _write_json(output_dir / "episodes.json", episodes)
    rows = validate_evidence(rows, episodes, splits, seed)
    labels = recover_validation_labels(episodes, splits, args, thresholds)
    _write_json(output_dir / "offline_labels.json", labels)
    likelihoods, modes, active, sequences = build_track_tensors(rows, labels)
    _write_json(output_dir / "likelihood_sequences.json", sequences)
    selection = select_validation_scale(
        likelihoods, modes, active, phase=phase, split=split, cases=splits["val"], case_splits=splits,
        selection_metric=selection_metric,
    )
    for path, digest in input_hashes.items():
        if sha256_file(path) != digest:
            raise ValueError(f"frozen input changed during calibration: {path}")
    for relative, digest in source_hashes.items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"source changed during collection; raw evidence retained but calibration not certified: {relative}")
    artifacts = {name: sha256_file(output_dir / name) for name in (
        "raw_evidence.jsonl", "episodes.json", "offline_labels.json", "likelihood_sequences.json",
        "metadata.json", "source_hashes.json", "controller_parameters.json",
    )}
    artifact = {**contract, **selection, "tracks": len(sequences), "factual_windows": len(rows),
                "unique_observations": int(active.sum()), "artifacts": artifacts,
                "raw_likelihood_sequences": "likelihood_sequences.json", "source_unchanged": True}
    _write_json(output_dir / "calibration.json", artifact)
    _write_json(output_dir / "manifest.json", {
        "status": "COMPLETED", "phase": phase, "split": split,
        "artifacts": {**artifacts, "calibration.json": sha256_file(output_dir / "calibration.json")},
    })
    print(json.dumps({"output": str(output_dir), "selected": selection["selected"],
                      "factual_windows": len(rows), "tracks": len(sequences)}, indent=2), flush=True)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="experiments/tracer_live_belief_discriminative_20260905/seed2027/response.pt")
    parser.add_argument("--controller-metadata", default="experiments/h3_external_belief_causal_final_20260906/metadata.json")
    parser.add_argument("--output-dir", default="experiments/h2_external_dense_calibration_20260906")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cases", type=int, nargs="+", default=None)
    parser.add_argument("--phase", choices=("validation",), default="validation")
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--selection-metric", choices=("decision_mode_nll", "decision_response_nll"), default="decision_mode_nll")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    calibrate_external_cadence(args.checkpoint, args.controller_metadata, args.output_dir, device=args.device,
                               seed=args.seed, cases=args.cases, phase=args.phase, split=args.split,
                               selection_metric=args.selection_metric)


if __name__ == "__main__":
    main()
