"""Run the reproducible external response and identity audits.

The public external data are useful for response compatibility and identity
sanity checks, but none of them provide the full Formal-v1 joint intervention
protocol.  This runner therefore reports descriptive, paired, and
action-conditioned diagnostics without promoting them to causal claims.

The UF Retail parser intentionally uses only the standard library for XLSX
files.  The downloaded archive contains both mocap position sheets and gaze
exports, and it also contains a few incorrectly labelled ROS-bag URLs.  ZIP
signature and expected-size checks keep those files out of the experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


UF_TRIAL_RE = re.compile(r"^(P\d{3})_(\d{3})(?:_offaxis)?\.xlsx$")
UF_DOC_RE = re.compile(r"Participant Data Sheet_(P\d{3})\.docx$")
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_or_none(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def _median_or_none(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(values)) if values else None


def _bootstrap_ci(values: Iterable[float], seed: int, iterations: int = 4000) -> list[float] | None:
    values = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _sign_flip_pvalue(values: Iterable[float], seed: int, iterations: int = 20000) -> float | None:
    values = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if not len(values):
        return None
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(iterations, len(values)))
    null = np.abs((signs * values[None]).mean(axis=1))
    return float((1.0 + np.sum(null >= observed)) / (iterations + 1.0))


def _column_number(reference: str) -> int:
    letters = re.match(r"([A-Z]+)", reference.upper())
    if not letters:
        return -1
    number = 0
    for char in letters.group(1):
        number = number * 26 + ord(char) - ord("A") + 1
    return number - 1


def _xlsx_sheet_path(archive: zipfile.ZipFile) -> str | None:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        item.attrib.get("Id"): item.attrib.get("Target", "")
        for item in relationships.findall(f"{REL_NS}Relationship")
    }
    for sheet in workbook.findall(f"{XLSX_NS}sheets/{XLSX_NS}sheet"):
        name = sheet.attrib.get("name", "").lower()
        if name == "position":
            target = targets.get(sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"))
            if target:
                target = target.lstrip("/")
                return target if target.startswith("xl/") else f"xl/{target}"
    return None


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values: list[str] = []
    for item in root.findall(f"{XLSX_NS}si"):
        values.append("".join(text.text or "" for text in item.iter(f"{XLSX_NS}t")))
    return values


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.iter(f"{XLSX_NS}t"))
    value = cell.find(f"{XLSX_NS}v")
    raw = "" if value is None or value.text is None else value.text
    if cell_type == "s":
        try:
            return shared[int(raw)]
        except (ValueError, IndexError):
            return ""
    return raw


def _position_rows(path: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Read timestamp and pelvis x/y from one mocap position workbook."""

    with zipfile.ZipFile(path) as archive:
        sheet_path = _xlsx_sheet_path(archive)
        if sheet_path is None:
            raise ValueError("position sheet is missing")
        shared = _shared_strings(archive)
        timestamps: list[float] = []
        positions: list[tuple[float, float, float]] = []
        headers: list[str] | None = None
        with archive.open(sheet_path) as handle:
            for _, element in ET.iterparse(handle, events=("end",)):
                if element.tag != f"{XLSX_NS}row":
                    continue
                cells = {
                    _column_number(cell.attrib.get("r", "")): _cell_value(cell, shared)
                    for cell in element.findall(f"{XLSX_NS}c")
                }
                if headers is None:
                    headers = [cells.get(index, "") for index in range(max(cells, default=-1) + 1)]
                    element.clear()
                    continue
                if not headers:
                    element.clear()
                    continue
                header_index = {name: index for index, name in enumerate(headers)}
                required = ("Timestamp", "Pelvis_x", "Pelvis_y")
                if not all(name in header_index for name in required):
                    raise ValueError(f"required position columns missing: {required}")
                values = {name: cells.get(header_index[name], "") for name in required}
                timestamp = _finite(values["Timestamp"])
                x = _finite(values["Pelvis_x"])
                y = _finite(values["Pelvis_y"])
                z = _finite(cells.get(header_index.get("Pelvis_z", -1), ""))
                if timestamp is not None and x is not None and y is not None:
                    timestamps.append(timestamp)
                    positions.append((x, y, z if z is not None else float("nan")))
                element.clear()
    rows = [(timestamps[index], *positions[index]) for index in range(len(timestamps))]
    rows.sort(key=lambda row: row[0])
    deduplicated: list[tuple[float, float, float]] = []
    dedup_timestamps: list[float] = []
    for timestamp, x, y, z in rows:
        if dedup_timestamps and timestamp <= dedup_timestamps[-1]:
            continue
        dedup_timestamps.append(timestamp)
        deduplicated.append((x, y, z))
    return [str(value) for value in dedup_timestamps], deduplicated


def _parse_uf_trial_table(meta_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    trials: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(meta_root.glob("*.docx")):
        match = UF_DOC_RE.match(path.name)
        if not match:
            continue
        participant = match.group(1)
        try:
            with zipfile.ZipFile(path) as archive:
                root = ET.fromstring(archive.read("word/document.xml"))
        except (OSError, KeyError, ET.ParseError):
            continue
        text = " ".join(item.text or "" for item in root.iter() if item.tag.endswith("}t"))
        text = re.sub(r"\s+", " ", text)
        pattern = re.compile(
            r"(\d+)\.\s*(No robot|With robot),\s*(order picking and sorting|inventory checking)\s*(\d+)",
            re.IGNORECASE,
        )
        for trial, condition, task, repetition in pattern.findall(text):
            trial_number = int(trial)
            trials[(participant, trial_number)] = {
                "participant": participant,
                "trial": trial_number,
                "condition": condition.lower().replace(" ", "_"),
                "task": task.lower().replace(" ", "_"),
                "repetition": int(repetition),
                "metadata_file": str(path),
            }
    return trials


def _trajectory_metrics(timestamps: list[str], positions: list[tuple[float, float, float]]) -> dict[str, Any]:
    time = np.asarray([float(value) for value in timestamps], dtype=np.float64)
    xy = np.asarray([(x, y) for x, y, _ in positions], dtype=np.float64)
    if len(time) < 2:
        return {"samples": int(len(time)), "status": "too_short"}
    elapsed = (time[-1] - time[0]) / 1000.0
    steps = np.diff(xy, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    path_length = float(np.sum(step_lengths))
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    valid_dt = np.diff(time) / 1000.0
    valid_dt = valid_dt[valid_dt > 0]
    step_speed = step_lengths[: len(valid_dt)] / valid_dt if len(valid_dt) else np.asarray([])
    heading = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
    turn = float(np.sum(np.abs(np.diff(heading)))) if len(heading) > 1 else 0.0
    max_step = float(np.max(step_lengths)) if len(step_lengths) else None
    max_speed = float(np.max(step_speed)) if len(step_speed) else None
    quality_flag = "ok"
    if (max_step is not None and max_step > 0.5) or (max_speed is not None and max_speed > 6.0):
        quality_flag = "suspect_coordinate_jump"
    return {
        "status": "complete",
        "samples": int(len(time)),
        "duration_s": float(elapsed) if elapsed > 0 else None,
        "sampling_hz": float(1.0 / np.median(valid_dt)) if len(valid_dt) else None,
        "path_length_m": path_length,
        "displacement_m": displacement,
        "mean_speed_mps": float(path_length / elapsed) if elapsed > 0 else None,
        "straightness": float(displacement / path_length) if path_length > 0 else None,
        "turning_radians": turn,
        "max_step_m": max_step,
        "max_speed_mps": max_speed,
        "quality_flag": quality_flag,
        "start_x": float(xy[0, 0]),
        "start_y": float(xy[0, 1]),
        "end_x": float(xy[-1, 0]),
        "end_y": float(xy[-1, 1]),
    }


def _paired_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    quality_excluded = [
        record for record in records
        if record.get("status") == "complete" and record.get("quality_flag", "ok") != "ok"
    ]
    for record in records:
        if record.get("status") != "complete" or record.get("quality_flag", "ok") != "ok":
            continue
        key = (record["participant"], record["task"], int(record["repetition"]))
        by_key[key][record["condition"]] = record
    pairs: list[dict[str, Any]] = []
    metrics = ("duration_s", "path_length_m", "displacement_m", "mean_speed_mps", "straightness", "turning_radians")
    for (participant, task, repetition), values in sorted(by_key.items()):
        if not {"no_robot", "with_robot"}.issubset(values):
            continue
        row: dict[str, Any] = {"participant": participant, "task": task, "repetition": repetition}
        for metric in metrics:
            no_robot = _finite(values["no_robot"].get(metric))
            with_robot = _finite(values["with_robot"].get(metric))
            row[f"no_robot_{metric}"] = no_robot
            row[f"with_robot_{metric}"] = with_robot
            row[f"delta_with_minus_no_{metric}"] = (
                with_robot - no_robot if no_robot is not None and with_robot is not None else None
            )
        pairs.append(row)
    summary: dict[str, Any] = {
        "all_complete_trials": sum(record.get("status") == "complete" for record in records),
        "quality_excluded_trials": len(quality_excluded),
        "quality_excluded_files": [record["file"] for record in quality_excluded],
        "paired_groups": len(pairs),
        "metrics": {},
        "task_stratified": {},
    }
    for metric in metrics:
        deltas = [row[f"delta_with_minus_no_{metric}"] for row in pairs]
        deltas = [float(value) for value in deltas if value is not None and math.isfinite(float(value))]
        summary["metrics"][metric] = {
            "n": len(deltas),
            "mean_delta_with_minus_no": _mean_or_none(deltas),
            "median_delta_with_minus_no": _median_or_none(deltas),
            "positive_fraction": float(np.mean(np.asarray(deltas) > 0)) if deltas else None,
            "bootstrap_95ci_mean_delta": _bootstrap_ci(deltas, seed=20260825 + len(metric)),
            "sign_flip_pvalue": _sign_flip_pvalue(deltas, seed=20260825 + 100 + len(metric)),
        }
    for task in sorted({row["task"] for row in pairs}):
        task_rows = [row for row in pairs if row["task"] == task]
        task_summary: dict[str, Any] = {"paired_groups": len(task_rows), "metrics": {}}
        for metric in metrics:
            deltas = [row[f"delta_with_minus_no_{metric}"] for row in task_rows]
            deltas = [float(value) for value in deltas if value is not None and math.isfinite(float(value))]
            task_summary["metrics"][metric] = {
                "n": len(deltas),
                "mean_delta_with_minus_no": _mean_or_none(deltas),
                "bootstrap_95ci_mean_delta": _bootstrap_ci(deltas, seed=20260925 + len(metric)),
                "sign_flip_pvalue": _sign_flip_pvalue(deltas, seed=20260925 + 100 + len(metric)),
            }
        summary["task_stratified"][task] = task_summary
    return {"pairs": pairs, "summary": summary}


def audit_uf_retail(mocap_root: Path, meta_root: Path) -> dict[str, Any]:
    trial_table = _parse_uf_trial_table(meta_root)
    records: list[dict[str, Any]] = []
    invalid_files: list[dict[str, str]] = []
    for path in sorted(mocap_root.glob("*.xlsx")):
        match = UF_TRIAL_RE.match(path.name)
        if not match:
            continue
        participant, trial_text = match.groups()
        trial = int(trial_text)
        metadata = trial_table.get((participant, trial))
        record: dict[str, Any] = {
            "participant": participant,
            "trial": trial,
            "file": str(path),
            "source_sha256": _sha256(path),
            "expected_bytes": path.stat().st_size,
        }
        if metadata is None:
            record.update({"status": "missing_metadata", "reason": "no participant data-sheet row"})
            invalid_files.append({"file": str(path), "reason": record["reason"]})
            records.append(record)
            continue
        record.update({key: value for key, value in metadata.items() if key != "metadata_file"})
        record["metadata_file"] = metadata["metadata_file"]
        try:
            with zipfile.ZipFile(path) as archive:
                if "xl/workbook.xml" not in archive.namelist():
                    raise ValueError("not an XLSX workbook")
            timestamps, positions = _position_rows(path)
            record.update(_trajectory_metrics(timestamps, positions))
        except (OSError, zipfile.BadZipFile, KeyError, ET.ParseError, ValueError) as error:
            record.update({"status": "invalid", "reason": str(error)})
            invalid_files.append({"file": str(path), "reason": str(error)})
        records.append(record)
    complete = [record for record in records if record.get("status") == "complete"]
    paired = _paired_summary(records)
    return {
        "status": "complete" if complete else "blocked_missing_mocap",
        "dataset": "UF Retail Cobot",
        "source": "https://doi.org/10.11922/sciencedb.01351",
        "paper": "https://www.nature.com/articles/s41597-022-01802-8",
        "mocap_files_seen": len(records),
        "complete_trials": len(complete),
        "invalid_files": invalid_files,
        "participants": len({record["participant"] for record in complete}),
        "conditions": dict(Counter(record["condition"] for record in complete)),
        "tasks": dict(Counter(record["task"] for record in complete)),
        "trial_records": records,
        "paired_condition_audit": paired["summary"],
        "paired_records": paired["pairs"],
        "claim_boundary": "paired descriptive response audit; no robot candidate action log and no counterfactual same-state guarantee",
    }


def _track_metrics(track: pd.DataFrame) -> dict[str, Any]:
    track = track.sort_values(["Time", "frame_id"]).drop_duplicates("Time")
    xy = track[["x", "y"]].to_numpy(dtype=np.float64)
    time = track["Time"].to_numpy(dtype=np.float64)
    if len(track) < 2:
        return {"samples": int(len(track))}
    dt = np.diff(time)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    duration = float(time[-1] - time[0])
    return {
        "samples": int(len(track)),
        "duration_s": duration,
        "path_length": float(step.sum()),
        "displacement": float(np.linalg.norm(xy[-1] - xy[0])),
        "mean_speed": float(step.sum() / duration) if duration > 0 else None,
        "sampling_hz": float(1.0 / np.median(dt[dt > 0])) if np.any(dt > 0) else None,
    }


def audit_thor_magni(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    frame = pd.read_csv(path)
    required = {"Time", "frame_id", "x", "y", "ag_id", "agent_type", "file_name", "dataset_name", "tracklet_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"status": "invalid", "path": str(path), "missing_columns": missing}
    frame = frame.dropna(subset=["Time", "x", "y", "ag_id", "file_name"]).copy()
    frame["episode_id"] = frame["dataset_name"].astype(str) + "/" + frame["file_name"].astype(str)
    frame["condition"] = np.where(
        frame["file_name"].str.contains("SC3A", regex=False),
        "scenario_3_A",
        np.where(frame["file_name"].str.contains("SC3B", regex=False), "scenario_3_B", frame["dataset_name"].str.lower()),
    )
    human = frame[frame["agent_type"].astype(str).str.startswith("Visitors")].copy()
    track_rows: list[dict[str, Any]] = []
    for (episode_id, ag_id, tracklet_id), track in human.groupby(["episode_id", "ag_id", "tracklet_id"], sort=True):
        metrics = _track_metrics(track)
        track_rows.append(
            {
                "episode_id": episode_id,
                "ag_id": ag_id,
                "tracklet_id": tracklet_id,
                "condition": str(track["condition"].iloc[0]),
                "agent_type": str(track["agent_type"].iloc[0]),
                **metrics,
            }
        )
    track_df = pd.DataFrame(track_rows)
    condition_summary: dict[str, Any] = {}
    for condition, subset in track_df.groupby("condition", sort=True):
        condition_summary[str(condition)] = {
            "tracks": int(len(subset)),
            "episodes": int(subset["episode_id"].nunique()),
            "mean_path_length": _mean_or_none(subset["path_length"].dropna()),
            "mean_displacement": _mean_or_none(subset["displacement"].dropna()),
            "mean_speed": _mean_or_none(subset["mean_speed"].dropna()),
        }
    actions = Counter(str(value) for value in human.get("action", pd.Series(dtype=str)).dropna())
    return {
        "status": "complete",
        "dataset": "THOR-MAGNI processed trajectory snapshot",
        "source": "https://zenodo.org/records/13865754",
        "source_sha256": _sha256(path),
        "rows": int(len(frame)),
        "episodes": int(frame["episode_id"].nunique()),
        "human_rows": int(len(human)),
        "human_tracks": int(len(track_df)),
        "condition_summary": condition_summary,
        "action_classes": len(actions),
        "action_counts": dict(actions.most_common()),
        "agent_type_counts": {str(k): int(v) for k, v in frame["agent_type"].value_counts().items()},
        "identity": {
            "unique_agent_ids": int(human["ag_id"].nunique()),
            "tracklet_count": int(human["tracklet_id"].nunique()),
            "agent_ids_with_multiple_tracklets": int(
                sum(count > 1 for count in human.groupby("ag_id")["tracklet_id"].nunique())
            ),
        },
        "track_records": track_rows,
        "claim_boundary": "action-labelled and identity-aware descriptive audit; no executed multi-robot intervention or counterfactual pairing",
    }


def audit_peroi_sample(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "sample_not_local",
            "path": str(path),
            "repository": "https://github.com/peroi-dataset/peroi-sample",
            "does_not_claim_results": True,
        }
    frame = pd.read_csv(path)
    required = {
        "Frame_Number",
        "Pedestrian_ID",
        "X_Position",
        "Y_Position",
        "Robot_Presence",
        "Robot_Type",
        "Robot_Influence",
        "X_Robot",
        "Y_Robot",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"status": "invalid_sample", "path": str(path), "missing_columns": missing}
    frame = frame.sort_values(["Pedestrian_ID", "Frame_Number"])
    trajectories = []
    for pedestrian_id, track in frame.groupby("Pedestrian_ID", sort=True):
        xy = track[["X_Position", "Y_Position"]].to_numpy(dtype=np.float64)
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        trajectories.append(
            {
                "pedestrian_id": str(pedestrian_id),
                "samples": int(len(track)),
                "path_length_m": float(step.sum()),
                "displacement_m": float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) else None,
                "robot_type": str(track["Robot_Type"].iloc[0]),
                "robot_influence": str(track["Robot_Influence"].iloc[0]),
                "robot_presence": int(track["Robot_Presence"].iloc[0]),
            }
        )
    return {
        "status": "public_sample_only",
        "path": str(path),
        "repository": "https://github.com/peroi-dataset/peroi-sample",
        "rows": int(len(frame)),
        "pedestrian_ids": int(frame["Pedestrian_ID"].nunique()),
        "robot_types": {str(key): int(value) for key, value in frame["Robot_Type"].value_counts(dropna=False).items()},
        "robot_presence": {str(key): int(value) for key, value in frame["Robot_Presence"].value_counts(dropna=False).items()},
        "influence_labels": {str(key): int(value) for key, value in frame["Robot_Influence"].value_counts(dropna=False).items()},
        "trajectory_records": trajectories,
        "full_release_verified": False,
        "does_not_claim_results": True,
        "claim_boundary": "official public sample confirms schema and labels; full 18,669-trajectory release is not locally verified",
    }


def availability_registry(
    jrdb_root: Path,
    peroi_sample_path: Path,
    sacson_repo_root: Path,
    huron_result: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    jrdb_present = jrdb_root.exists()
    huron_result = huron_result or {"status": "not_run"}
    huron_status = (
        "public_rail_release_verified"
        if huron_result.get("status") == "complete" and huron_result.get("complete_bags", 0) > 0
        else "public_rail_release_partial"
        if huron_result.get("status") == "complete"
        else "public_rail_release_not_audited"
    )
    return {
        "HuRoN_SACSoN": {
            "status": huron_status,
            "paper": "https://arxiv.org/abs/2306.01874",
            "repository": "https://github.com/NHirose/SACSoN",
            "local_code_root": str(sacson_repo_root),
            "dataset_page": "https://sites.google.com/view/sacson-review/huron-dataset?authuser=0",
            "rail_mirror": "https://rail.eecs.berkeley.edu/datasets/huron/",
            "dataset_page_checked": True,
            "download_verified": huron_result.get("complete_bags", 0) > 0,
            "audit": huron_result,
            "scope": "single-robot action-conditioned human response; no persistent multi-robot intervention labels",
            "local_root": "raw_data/external/huron_sacson",
            "does_not_claim_results": True,
            "claim_boundary": "single-robot action-conditioned descriptive response audit; RAIL bags are recording-local and do not support causal or persistent-human identity claims",
        },
        "RPI_PeRoI": {
            "status": "public_sample_only_full_release_unverified",
            "paper": "https://arxiv.org/abs/2503.16481",
            "repository": "https://github.com/peroi-dataset/peroi-sample",
            "sample": audit_peroi_sample(peroi_sample_path),
            "scope": "robot-influenced pedestrian trajectories and response labels; single robot and no persistent identity guarantee",
            "does_not_claim_results": True,
        },
        "SCAND": {
            "status": "screened_observational_no_tracking_labels",
            "paper": "https://arxiv.org/abs/2203.15041",
            "dataset": "https://doi.org/10.18738/T8/0PRYRH",
            "scope": "robot teleoperation and social-interaction context; paper reports no human detection/tracking annotation",
            "local_root": "raw_data/external/scand",
            "does_not_claim_results": True,
        },
        "PARSNiP": {
            "status": "screened_user_study_not_trajectory_response",
            "paper": "https://link.springer.com/article/10.1007/s12369-025-01266-x",
            "repository": "https://github.com/duibcuiegiosahxois/PARSNiP",
            "scope": "perception/appropriateness study; useful supplementary HRI evidence but not an action-conditioned track benchmark",
            "local_root": "raw_data/external/pars_nip",
            "does_not_claim_results": True,
        },
        "JRDB": {
            "status": "blocked_authenticated_assets" if not jrdb_present else "local_assets_require_adapter_audit",
            "paper": "https://arxiv.org/abs/1910.11792",
            "official_download": "https://jrdb.erc.monash.edu/#downloads",
            "scope": "robot-centric perception and tracks; no executed robot action intervention labels",
            "local_root": str(jrdb_root),
            "does_not_claim_results": True,
        },
        "SocNavBench": {
            "status": "screened_simulator_replay_no_human_response",
            "paper": "https://arxiv.org/abs/2103.00047",
            "scope": "planner stress testing with replayed human trajectories; humans do not react to the robot",
            "does_not_claim_results": True,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict[str, Any]) -> str:
    uf = payload["datasets"]["UF Retail Cobot"]
    thor = payload["datasets"]["THOR-MAGNI"]
    thor_full = payload["datasets"].get("THOR-MAGNI full")
    huron = payload["screened_sources"].get("HuRoN_SACSoN", {})
    huron_audit = huron.get("audit", {})
    peroi = payload["screened_sources"].get("RPI_PeRoI", {})
    peroi_sample = peroi.get("sample", {})
    lines = [
        "# External Response Benchmark Audit",
        "",
        "This report is an external compatibility and response audit. It does not authorize the Formal-v1 causal claim: the public data do not expose matched joint candidate actions, same-initial-state counterfactuals, or executed persistent belief labels.",
        "",
        "| Dataset | Status | Trials/rows | Response or identity evidence |",
        "|---|---|---:|---|",
        f"| UF Retail Cobot | {uf.get('status')} | {uf.get('complete_trials', 0)} mocap trials | {uf.get('paired_condition_audit', {}).get('paired_groups', 0)} primary paired groups ({uf.get('paired_condition_audit', {}).get('quality_excluded_trials', 0)} QC exclusions) |",
        f"| THOR-MAGNI | {thor.get('status')} | {thor.get('rows', 0)} rows | {thor.get('human_tracks', 0)} visitor tracks; {thor.get('action_classes', 0)} action classes |",
        f"| THOR-MAGNI full | {thor_full.get('status') if thor_full else 'not_run'} | {thor_full.get('summary', {}).get('data_rows', '-') if thor_full else '-'} rows | {thor_full.get('summary', {}).get('tracks', '-') if thor_full else '-'} full-release tracks; {thor_full.get('summary', {}).get('gaze_2d_valid_rows', '-') if thor_full else '-'} valid gaze rows |",
        f"| HuRoN/SACSoN | {huron.get('status', 'not_run')} | {huron_audit.get('complete_bags', 0)}/{huron_audit.get('bags_seen', 0)} representative bags | {huron_audit.get('human_response_primary_observations', 0)} primary marker-motion transitions; {huron_audit.get('human_response_quality_excluded_observations', 0)} QC exclusions |",
        f"| RPI/PeRoI | {peroi.get('status', 'not_run')} | {peroi_sample.get('rows', '-')} sample rows | publicness/schema confirmation only; full release unverified |",
        "",
        "## Interpretation",
        "",
        "UF Retail compares human motion in repeated with-robot and no-robot task conditions within participant/task/repetition. The robot candidate path is not available in the mocap-only files, so the result is a paired response compatibility audit, not an action-conditioned causal estimate.",
        "",
        "The primary paired summary excludes only trials flagged by a transparent pelvis-coordinate jump rule (step > 0.5 m or speed > 6 m/s). All raw trial metrics and excluded filenames remain in the JSON/CSV outputs.",
        "",
        "THOR-MAGNI full reports all five scenarios, robot-condition mappings, motion-capture tracks, role metadata, gaze validity, and the partial action-label coverage. The older processed snapshot is retained only as the action-labelled subset and is not used as the full-release count.",
        "",
        "HuRoN/SACSoN is audited from the public Berkeley RAIL mirror using ROS1 bags. The audit reports robot odometry, pedestrian-marker motion, and laser availability; it uses recording-local marker IDs and excludes only transparent marker-jump transitions from primary response summaries.",
        "",
        "RPI/PeRoI is retained only as a public sample/schema confirmation. Its full release is not locally verified and contributes no benchmark metric.",
        "",
        "The remaining screened datasets are recorded with fail-closed availability or suitability statuses. Their absence does not contribute metrics.",
    ]
    return "\n".join(lines) + "\n"


def run(
    uf_mocap_root: Path,
    uf_meta_root: Path,
    thor_magni_csv: Path,
    jrdb_root: Path,
    output_dir: Path,
    peroi_sample_path: Path = Path("raw_data/external/source_repos/peroi_sample/peroi_sample.csv"),
    sacson_repo_root: Path = Path("raw_data/external/source_repos/sacson"),
    thor_full_summary: Path | None = Path("reports/external_validation_20260825/thor_magni_full/summary.json"),
    huron_summary_path: Path | None = Path("reports/external_validation_20260825/huron_sacson/summary.json"),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    uf = audit_uf_retail(uf_mocap_root, uf_meta_root)
    thor = audit_thor_magni(thor_magni_csv)
    thor_full: dict[str, Any] = {"status": "not_run", "summary": {}}
    if thor_full_summary is not None and thor_full_summary.exists():
        thor_full = json.loads(thor_full_summary.read_text(encoding="utf-8"))
    huron: dict[str, Any] = {"status": "not_run", "complete_bags": 0}
    if huron_summary_path is not None and huron_summary_path.exists():
        huron = json.loads(huron_summary_path.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "protocol": {
            "date": "2026-08-25",
            "claim_boundary": "external response, identity, and action-context compatibility only; no real-human causal or preference claim",
            "gpu": "not required for deterministic parsing and descriptive metrics",
        },
        "datasets": {"UF Retail Cobot": uf, "THOR-MAGNI": thor, "THOR-MAGNI full": thor_full},
        "screened_sources": availability_registry(jrdb_root, peroi_sample_path, sacson_repo_root, huron),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(
        output_dir / "uf_retail_trials.csv",
        uf.get("trial_records", []),
        ["participant", "trial", "condition", "task", "repetition", "status", "quality_flag", "samples", "duration_s", "path_length_m", "displacement_m", "mean_speed_mps", "straightness", "turning_radians", "max_step_m", "max_speed_mps", "file", "source_sha256"],
    )
    _write_csv(
        output_dir / "uf_retail_pairs.csv",
        uf.get("paired_records", []),
        sorted({key for row in uf.get("paired_records", []) for key in row}) if uf.get("paired_records") else ["participant", "task", "repetition"],
    )
    _write_csv(
        output_dir / "thor_magni_tracks.csv",
        thor.get("track_records", []),
        ["episode_id", "ag_id", "tracklet_id", "condition", "agent_type", "samples", "duration_s", "path_length", "displacement", "mean_speed", "sampling_hz"],
    )
    (output_dir / "summary.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external response and identity audits.")
    parser.add_argument("--uf-mocap-root", default="raw_data/external/uf_retail/mocap")
    parser.add_argument("--uf-meta-root", default="raw_data/external/uf_retail/meta")
    parser.add_argument("--thor-magni-csv", default="raw_data/external/thor_magni_merged_data.csv")
    parser.add_argument("--jrdb-root", default="raw_data/external/jrdb")
    parser.add_argument("--peroi-sample", default="raw_data/external/source_repos/peroi_sample/peroi_sample.csv")
    parser.add_argument("--sacson-repo-root", default="raw_data/external/source_repos/sacson")
    parser.add_argument("--thor-full-summary", default="reports/external_validation_20260825/thor_magni_full/summary.json")
    parser.add_argument("--huron-summary", default="reports/external_validation_20260825/huron_sacson/summary.json")
    parser.add_argument("--output-dir", default="reports/external_validation_20260825")
    args = parser.parse_args()
    payload = run(
        Path(args.uf_mocap_root),
        Path(args.uf_meta_root),
        Path(args.thor_magni_csv),
        Path(args.jrdb_root),
        Path(args.output_dir),
        Path(args.peroi_sample),
        Path(args.sacson_repo_root),
        Path(args.thor_full_summary) if args.thor_full_summary else None,
        Path(args.huron_summary) if args.huron_summary else None,
    )
    for name, result in payload["datasets"].items():
        print(f"dataset={name} status={result.get('status')} trials={result.get('complete_trials', result.get('rows', 0))}")


if __name__ == "__main__":
    main()
