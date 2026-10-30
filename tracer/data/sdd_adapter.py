from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from tracer.data.heterogeneous_scene import AgentRole, HeterogeneousSceneBatch
from tracer.data.mid_adapter import infer_entity_roles
from tracer.data.perception import DEFAULT_ROLE_RADIUS, effective_radius

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - exercised in user env when pandas exists.
    pd = None

try:
    import dill as pickle_backend
except ModuleNotFoundError:
    import pickle as pickle_backend


SDD_LABEL_COLUMNS = (
    "label",
    "agent_type",
    "agentType",
    "type",
    "class",
    "category",
    "node_type",
)

SDD_ROLE_LABELS = {
    int(AgentRole.PHA): {
        "pedestrian",
        "person",
        "people",
        "walker",
        "ped",
    },
    int(AgentRole.HNAA): {
        "biker",
        "bicycle",
        "bike",
        "cyclist",
        "skater",
        "skateboarder",
        "cart",
    },
    int(AgentRole.SSAA): {
        "car",
        "vehicle",
        "bus",
        "truck",
        "van",
    },
}

ROLE_DEFAULT_GEOMETRY = {
    int(AgentRole.ROBOT): (0.7, 0.7),
    int(AgentRole.PHA): (0.5, 0.5),
    int(AgentRole.HNAA): (1.4, 0.7),
    int(AgentRole.SSAA): (2.4, 1.2),
    int(AgentRole.UNKNOWN_ENTITY): (0.8, 0.8),
}

LABEL_GEOMETRY = {
    "pedestrian": (0.5, 0.5),
    "person": (0.5, 0.5),
    "biker": (1.5, 0.7),
    "bicycle": (1.5, 0.7),
    "cyclist": (1.5, 0.7),
    "skater": (1.0, 0.6),
    "skateboarder": (1.0, 0.6),
    "cart": (1.2, 0.8),
    "car": (2.4, 1.2),
    "vehicle": (2.4, 1.2),
    "bus": (4.0, 1.5),
    "truck": (3.5, 1.5),
    "van": (2.8, 1.3),
}


@dataclass(frozen=True)
class SDDTrack:
    scene_id: object
    track_id: object
    frames: np.ndarray
    positions: np.ndarray
    label: str | None

    @property
    def frame_to_index(self) -> dict[int, int]:
        return {int(frame): idx for idx, frame in enumerate(self.frames)}


@dataclass(frozen=True)
class SDDScene:
    scene_id: object
    name: str
    tracks: tuple[SDDTrack, ...]


@dataclass(frozen=True)
class SDDSampleIndex:
    scene_idx: int
    track_idx: int
    timestep: int


def normalize_label(label: object | None) -> str | None:
    if label is None:
        return None
    value = str(label).strip().strip('\"\'').lower()
    return value or None


def role_from_label(label: object | None) -> int | None:
    value = normalize_label(label)
    if value is None:
        return None
    for role_id, labels in SDD_ROLE_LABELS.items():
        if value in labels:
            return role_id
    return None


def geometry_from_role(role_id: int) -> tuple[float, float]:
    return ROLE_DEFAULT_GEOMETRY.get(int(role_id), ROLE_DEFAULT_GEOMETRY[int(AgentRole.UNKNOWN_ENTITY)])


def geometry_from_label(label: object | None, role_id: int | None = None) -> tuple[float, float]:
    value = normalize_label(label)
    if value in LABEL_GEOMETRY:
        return LABEL_GEOMETRY[value]
    return geometry_from_role(int(role_id) if role_id is not None else int(AgentRole.UNKNOWN_ENTITY))


def _mode_or_none(values: Sequence[object]) -> str | None:
    cleaned = [normalize_label(v) for v in values]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    counts: dict[str, int] = {}
    for value in cleaned:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _label_column(columns: Iterable[object]) -> str | None:
    columns = [str(c) for c in columns]
    lower_to_original = {c.lower(): c for c in columns}
    for candidate in SDD_LABEL_COLUMNS:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _count_non_null_labels(df, label_column: str | None) -> int:
    if label_column is None or label_column not in df.columns:
        return 0
    labels = df[label_column].dropna().astype(str).str.strip()
    labels = labels[labels != ""]
    return int(labels.shape[0])


def parse_sdd_scene_id(scene_id: str) -> tuple[str, str]:
    if "_" not in scene_id:
        raise ValueError(f"Cannot parse SDD sceneId={scene_id!r}; expected e.g. bookstore_0")
    scene_name, video_id = scene_id.rsplit("_", 1)
    return scene_name, video_id


def candidate_sdd_annotation_paths(root: Path, scene_id: str) -> list[Path]:
    scene_name, video_id = parse_sdd_scene_id(scene_id)
    video_names = [video_id, f"video{video_id}", f"video_{video_id}"]
    candidates: list[Path] = []
    for video_name in video_names:
        candidates.extend(
            [
                root / "annotations" / scene_name / video_name / "annotations.txt",
                root / scene_name / video_name / "annotations.txt",
                root / "annotations" / scene_name / video_name / "annotation.txt",
                root / scene_name / video_name / "annotation.txt",
            ]
        )
    return candidates


def find_sdd_annotation_file(root: Path, scene_id: str) -> Path | None:
    for candidate in candidate_sdd_annotation_paths(root, scene_id):
        if candidate.exists():
            return candidate
    return None


def load_trajnet_scene_ids(pkl_path: str | Path) -> set[str]:
    path = Path(pkl_path)
    with path.open("rb") as f:
        try:
            obj = pickle_backend.load(f, encoding="latin1")
        except TypeError:
            f.seek(0)
            obj = pickle_backend.load(f)
    if not hasattr(obj, "columns") or "sceneId" not in obj.columns:
        raise ValueError(f"{path} must contain a pandas DataFrame with sceneId")
    return {str(value) for value in obj["sceneId"].dropna().unique()}


def read_official_sdd_annotations(
    root: str | Path,
    scene_ids: Iterable[str],
    coordinate: str = "center",
) -> tuple[object, list[str]]:
    if pd is None:
        raise ModuleNotFoundError("pandas is required to read official SDD annotations")
    root = Path(root)
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    coordinate = coordinate.lower()
    if coordinate not in {"center", "top_left", "bottom_center"}:
        raise ValueError("coordinate must be one of: center, top_left, bottom_center")

    for scene_id in sorted({str(s) for s in scene_ids}):
        annotation_path = find_sdd_annotation_file(root, scene_id)
        if annotation_path is None:
            missing.append(scene_id)
            continue
        with annotation_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                track_id, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label = parts[:10]
                try:
                    xmin_f, ymin_f, xmax_f, ymax_f = float(xmin), float(ymin), float(xmax), float(ymax)
                    frame_i = int(float(frame))
                except ValueError:
                    continue
                if coordinate == "center":
                    x = 0.5 * (xmin_f + xmax_f)
                    y = 0.5 * (ymin_f + ymax_f)
                elif coordinate == "bottom_center":
                    x = 0.5 * (xmin_f + xmax_f)
                    y = ymax_f
                else:
                    x = xmin_f
                    y = ymin_f
                rows.append(
                    {
                        "frame": frame_i,
                        "trackId": track_id,
                        "x": x,
                        "y": y,
                        "sceneId": scene_id,
                        "metaId": f"{scene_id}_{track_id}",
                        "label": normalize_label(label),
                        "lost": int(float(lost)) if str(lost).replace(".", "", 1).isdigit() else lost,
                        "occluded": int(float(occluded)) if str(occluded).replace(".", "", 1).isdigit() else occluded,
                        "generated": int(float(generated)) if str(generated).replace(".", "", 1).isdigit() else generated,
                    }
                )
    if not rows:
        raise FileNotFoundError(
            f"No SDD annotation rows were loaded from {root}. Missing scenes: {missing[:10]}"
        )
    return pd.DataFrame(rows), missing


class SDDRawSceneDataset(Dataset):
    """Build SAGE samples from Stanford Drone Dataset trajnet-style pickles.

    The Y-Net/MID trajnet pickle shipped in this repository only stores
    trajectories. If a semantic label column is present, roles are mapped from
    SDD labels; otherwise roles fall back to speed-based proxy labels while the
    dataset reports ``role_source='speed_proxy'``.
    """

    def __init__(
        self,
        pkl_path: str | Path,
        history: int = 8,
        horizon: int = 12,
        max_entities: int = 12,
        num_robot_proxies: int = 1,
        dt: float = 0.4,
        frame_stride: int = 12,
        coordinate_scale: float = 50.0,
        center_scene: bool = True,
        include_entities_without_future: bool = False,
        role_strategy: str = "semantic_or_speed",
        role_slow_speed: float = 0.15,
        role_fast_speed: float = 1.8,
        role_slow_quantile: float = 0.33,
        role_fast_quantile: float = 0.67,
        limit_scenes: int = 0,
        label_map_path: str | Path | None = None,
        perception_mode: str = "nearest",
        perception_base_radius: float = 3.6,
        perception_max_radius: float = 8.0,
        perception_speed_scale: float = 0.8,
        perception_geometry_scale: float = 0.5,
        perception_role_radius: dict[int, float] | None = None,
        perception_fallback_to_nearest: bool = True,
    ) -> None:
        if pd is None:
            raise ModuleNotFoundError(
                "pandas is required to load raw_data/stanford/*_trajnet.pkl. "
                "Activate the SAGE environment or install pandas."
            )
        self.pkl_path = Path(pkl_path)
        self.history = int(history)
        self.horizon = int(horizon)
        self.max_entities = int(max_entities)
        self.num_robot_proxies = max(1, int(num_robot_proxies))
        self.dt = float(dt)
        self.frame_stride = int(frame_stride)
        self.coordinate_scale = float(coordinate_scale)
        self.center_scene = bool(center_scene)
        self.include_entities_without_future = include_entities_without_future
        self.role_strategy = role_strategy
        self.role_slow_speed = float(role_slow_speed)
        self.role_fast_speed = float(role_fast_speed)
        self.role_slow_quantile = float(role_slow_quantile)
        self.role_fast_quantile = float(role_fast_quantile)
        self.perception_mode = str(perception_mode).lower()
        self.perception_base_radius = float(perception_base_radius)
        self.perception_max_radius = float(perception_max_radius)
        self.perception_speed_scale = float(perception_speed_scale)
        self.perception_geometry_scale = float(perception_geometry_scale)
        self.perception_role_radius = dict(DEFAULT_ROLE_RADIUS if perception_role_radius is None else perception_role_radius)
        self.perception_fallback_to_nearest = bool(perception_fallback_to_nearest)

        self.raw_df = self._load_dataframe(self.pkl_path)
        self.raw_df = self._merge_label_map(self.raw_df, label_map_path)
        self.label_column = _label_column(self.raw_df.columns)
        self.label_non_null_rows = _count_non_null_labels(self.raw_df, self.label_column)
        self.role_source = "semantic" if self.label_non_null_rows > 0 else "speed_proxy"
        self.scenes = self._build_scenes(limit_scenes=limit_scenes)
        self.indices = self._build_index()

    def _load_dataframe(self, path: Path):
        with path.open("rb") as f:
            try:
                obj = pickle_backend.load(f, encoding="latin1")
            except TypeError:
                f.seek(0)
                obj = pickle_backend.load(f)
        if not hasattr(obj, "columns"):
            raise TypeError(f"Expected a pandas DataFrame in {path}, got {type(obj)!r}")
        required = {"sceneId", "trackId", "frame", "x", "y"}
        missing = sorted(required - set(map(str, obj.columns)))
        if missing:
            raise ValueError(f"{path} is missing required SDD columns: {missing}")
        return obj.copy()


    def _merge_label_map(self, df, label_map_path: str | Path | None):
        if label_map_path is None:
            return df
        map_path = Path(label_map_path)
        if not map_path.exists():
            raise FileNotFoundError(f"label map does not exist: {map_path}")
        if pd is None:
            raise ModuleNotFoundError("pandas is required to merge SDD label maps")
        label_df = pd.read_csv(map_path)
        label_column = _label_column(label_df.columns)
        if label_column is None:
            raise ValueError(
                f"{map_path} must contain one of the label columns {SDD_LABEL_COLUMNS}; "
                f"found {list(label_df.columns)}"
            )
        required = {"sceneId", "trackId"}
        missing = sorted(required - set(map(str, label_df.columns)))
        if missing:
            raise ValueError(f"{map_path} is missing required columns: {missing}")
        label_df = label_df[["sceneId", "trackId", label_column]].drop_duplicates(["sceneId", "trackId"])
        merged = df.merge(label_df, on=["sceneId", "trackId"], how="left", suffixes=("", "_labelmap"))
        if label_column not in merged.columns and f"{label_column}_labelmap" in merged.columns:
            merged[label_column] = merged[f"{label_column}_labelmap"]
        return merged

    def _build_scenes(self, limit_scenes: int = 0) -> list[SDDScene]:
        df = self.raw_df.copy()
        df["frame"] = pd.to_numeric(df["frame"], downcast="integer")
        try:
            df["trackId"] = pd.to_numeric(df["trackId"], downcast="integer")
        except (TypeError, ValueError):
            pass
        if self.frame_stride > 1:
            df["frame"] = df["frame"] // self.frame_stride
        agg = {"x": "mean", "y": "mean"}
        if self.label_column is not None:
            agg[self.label_column] = "first"
        if "metaId" in df.columns:
            agg["metaId"] = "first"
        df = df.groupby(["sceneId", "trackId", "frame"], as_index=False).agg(agg)

        scenes: list[SDDScene] = []
        scene_items = list(df.groupby("sceneId", sort=True))
        if limit_scenes > 0:
            scene_items = scene_items[:limit_scenes]
        for scene_id, scene_df in scene_items:
            scene_df = scene_df.copy()
            scene_df["frame"] = scene_df["frame"] - scene_df["frame"].min()
            scene_df["x"] = scene_df["x"] / self.coordinate_scale
            scene_df["y"] = scene_df["y"] / self.coordinate_scale
            if self.center_scene:
                scene_df["x"] = scene_df["x"] - scene_df["x"].mean()
                scene_df["y"] = scene_df["y"] - scene_df["y"].mean()

            tracks: list[SDDTrack] = []
            for track_id, track_df in scene_df.groupby("trackId", sort=True):
                track_df = track_df.sort_values("frame")
                frames = track_df["frame"].to_numpy(dtype=np.int64)
                positions = track_df[["x", "y"]].to_numpy(dtype=np.float32)
                if len(frames) < self.history + self.horizon:
                    continue
                label = _mode_or_none(track_df[self.label_column].tolist()) if self.label_column is not None else None
                tracks.append(
                    SDDTrack(
                        scene_id=scene_id,
                        track_id=track_id,
                        frames=frames,
                        positions=positions,
                        label=label,
                    )
                )
            if tracks:
                scenes.append(SDDScene(scene_id=scene_id, name=f"sdd_{scene_id}", tracks=tuple(tracks)))
        return scenes

    def _has_window(self, track: SDDTrack, start: int, end: int) -> bool:
        frame_to_index = track.frame_to_index
        return all(frame in frame_to_index for frame in range(int(start), int(end) + 1))

    def _position(self, track: SDDTrack, frame: int) -> np.ndarray:
        return track.positions[track.frame_to_index[int(frame)]]

    def _instant_speed(self, track: SDDTrack, timestep: int) -> float:
        frame_to_index = track.frame_to_index
        if int(timestep) not in frame_to_index:
            return 0.0
        idx = frame_to_index[int(timestep)]
        if idx > 0:
            prev = track.positions[idx - 1]
            cur = track.positions[idx]
            dt_frames = max(1, int(track.frames[idx] - track.frames[idx - 1]))
            return float(np.linalg.norm(cur - prev) / (self.dt * dt_frames))
        if idx + 1 < len(track.positions):
            cur = track.positions[idx]
            nxt = track.positions[idx + 1]
            dt_frames = max(1, int(track.frames[idx + 1] - track.frames[idx]))
            return float(np.linalg.norm(nxt - cur) / (self.dt * dt_frames))
        return 0.0

    def _proxy_role_from_speed(self, speed: float) -> int:
        if self.role_strategy == "all_pha":
            return int(AgentRole.PHA)
        if self.role_strategy == "all_unknown":
            return int(AgentRole.UNKNOWN_ENTITY)
        if speed < self.role_slow_speed:
            return int(AgentRole.SSAA)
        if speed < self.role_fast_speed:
            return int(AgentRole.PHA)
        return int(AgentRole.HNAA)

    def _track_role_for_perception(self, track: SDDTrack, timestep: int) -> int:
        role = role_from_label(track.label)
        if role is not None and self.role_strategy == "semantic_or_speed":
            return int(role)
        return self._proxy_role_from_speed(self._instant_speed(track, timestep))

    def _history(self, track: SDDTrack, timestep: int) -> np.ndarray:
        frames = range(timestep - self.history + 1, timestep + 1)
        return np.stack([self._position(track, frame) for frame in frames]).astype(np.float32)

    def _future_velocity(self, track: SDDTrack, timestep: int, valid_required: bool) -> tuple[np.ndarray, bool]:
        has_future = self._has_window(track, timestep, timestep + self.horizon)
        if valid_required and not has_future:
            raise ValueError("target track does not have a full future")
        if not has_future:
            return np.zeros((self.horizon, 2), dtype=np.float32), False
        velocities = []
        prev = self._position(track, timestep)
        for frame in range(timestep + 1, timestep + self.horizon + 1):
            cur = self._position(track, frame)
            velocities.append((cur - prev) / self.dt)
            prev = cur
        return np.stack(velocities).astype(np.float32), True

    def _build_index(self) -> list[SDDSampleIndex]:
        out: list[SDDSampleIndex] = []
        for scene_idx, scene in enumerate(self.scenes):
            for track_idx, track in enumerate(scene.tracks):
                min_t = int(track.frames.min()) + self.history - 1
                max_t = int(track.frames.max()) - self.horizon
                for timestep in range(min_t, max_t + 1):
                    if self._has_window(track, timestep - self.history + 1, timestep + self.horizon):
                        out.append(SDDSampleIndex(scene_idx=scene_idx, track_idx=track_idx, timestep=timestep))
        return out

    def __len__(self) -> int:
        return len(self.indices)

    def _neighbors(
        self,
        scene: SDDScene,
        target: SDDTrack,
        timestep: int,
        exclude_tracks: set[int] | None = None,
    ) -> list[SDDTrack]:
        exclude_tracks = exclude_tracks or set()
        target_pos = self._position(target, timestep)
        candidates = []
        in_radius = []
        use_radius = self.perception_mode in {"role_radius", "heterogeneous", "radius"}
        for track in scene.tracks:
            if track is target or id(track) in exclude_tracks:
                continue
            if not self._has_window(track, timestep - self.history + 1, timestep):
                continue
            has_future = self._has_window(track, timestep, timestep + self.horizon)
            if not has_future and not self.include_entities_without_future:
                continue
            dist = float(np.linalg.norm(self._position(track, timestep) - target_pos))
            candidates.append((dist, track))
            if use_radius:
                role_id = self._track_role_for_perception(track, timestep)
                radius = effective_radius(
                    role_id,
                    self._instant_speed(track, timestep),
                    geometry_from_label(track.label, role_id),
                    base_radius=self.perception_base_radius,
                    max_radius=self.perception_max_radius,
                    speed_scale=self.perception_speed_scale,
                    geometry_scale=self.perception_geometry_scale,
                    role_radius=self.perception_role_radius,
                )
                if dist <= radius:
                    in_radius.append((dist, track))
        candidates.sort(key=lambda item: item[0])
        in_radius.sort(key=lambda item: item[0])
        if use_radius and (in_radius or not self.perception_fallback_to_nearest):
            return [track for _, track in in_radius[: self.max_entities]]
        return [track for _, track in candidates[: self.max_entities]]

    def _robot_proxy_tracks(self, scene: SDDScene, target: SDDTrack, timestep: int) -> list[SDDTrack]:
        robots = [target]
        if self.num_robot_proxies <= 1:
            return robots
        target_pos = self._position(target, timestep)
        candidates = []
        for track in scene.tracks:
            if track is target:
                continue
            if not self._has_window(track, timestep - self.history + 1, timestep + self.horizon):
                continue
            role = role_from_label(track.label)
            if role == int(AgentRole.SSAA):
                continue
            dist = float(np.linalg.norm(self._position(track, timestep) - target_pos))
            candidates.append((dist, track))
        candidates.sort(key=lambda item: item[0])
        robots.extend(track for _, track in candidates[: self.num_robot_proxies - 1])
        return robots

    def _semantic_roles(self, labels: list[str | None], fallback_roles: torch.Tensor) -> torch.Tensor:
        roles = []
        for idx, label in enumerate(labels):
            role = role_from_label(label)
            roles.append(int(role) if role is not None else int(fallback_roles[idx].item()))
        return torch.tensor(roles, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict:
        item = self.indices[idx]
        scene = self.scenes[item.scene_idx]
        target = scene.tracks[item.track_idx]
        timestep = item.timestep

        robot_tracks = self._robot_proxy_tracks(scene, target, timestep)
        histories = [self._history(track, timestep) for track in robot_tracks]
        futures = [self._future_velocity(track, timestep, valid_required=True)[0] for track in robot_tracks]
        labels: list[str | None] = [track.label for track in robot_tracks]
        agent_mask = [True for _ in robot_tracks]
        loss_mask = [True for _ in robot_tracks]
        robot_count = len(robot_tracks)
        excluded = {id(track) for track in robot_tracks}

        for neighbor in self._neighbors(scene, target, timestep, exclude_tracks=excluded):
            future, valid_future = self._future_velocity(neighbor, timestep, valid_required=False)
            histories.append(self._history(neighbor, timestep))
            futures.append(future)
            labels.append(neighbor.label)
            agent_mask.append(True)
            loss_mask.append(valid_future)

        while len(histories) < self.max_entities + self.num_robot_proxies:
            histories.append(np.zeros((self.history, 2), dtype=np.float32))
            futures.append(np.zeros((self.horizon, 2), dtype=np.float32))
            labels.append(None)
            agent_mask.append(False)
            loss_mask.append(False)

        histories = histories[: self.max_entities + self.num_robot_proxies]
        futures = futures[: self.max_entities + self.num_robot_proxies]
        labels = labels[: self.max_entities + self.num_robot_proxies]
        agent_mask = agent_mask[: self.max_entities + self.num_robot_proxies]
        loss_mask = loss_mask[: self.max_entities + self.num_robot_proxies]
        robot_count = min(robot_count, len(histories))

        history_t = torch.tensor(np.stack(histories), dtype=torch.float32)
        future_t = torch.tensor(np.stack(futures), dtype=torch.float32)
        agent_mask_t = torch.tensor(agent_mask, dtype=torch.bool)
        loss_mask_t = torch.tensor(loss_mask, dtype=torch.bool)
        robot_mask_t = torch.zeros_like(agent_mask_t)
        robot_mask_t[:robot_count] = True
        entity_mask_t = agent_mask_t & ~robot_mask_t
        goals_t = history_t[:, -1, :2] + future_t.sum(dim=1) * self.dt
        goal_mask_t = robot_mask_t & agent_mask_t

        fallback_roles = infer_entity_roles(
            future_t[None],
            entity_mask_t[None],
            strategy="speed_threshold" if self.role_strategy == "semantic_or_speed" else self.role_strategy,
            slow_speed=self.role_slow_speed,
            fast_speed=self.role_fast_speed,
            slow_quantile=self.role_slow_quantile,
            fast_quantile=self.role_fast_quantile,
        )[0]
        if self.role_strategy == "semantic_or_speed" and self.label_column is not None:
            roles = self._semantic_roles(labels, fallback_roles)
        else:
            roles = fallback_roles
        roles[:robot_count] = int(AgentRole.ROBOT)
        roles = torch.where(agent_mask_t, roles, torch.full_like(roles, int(AgentRole.ROBOT)))

        geometry = []
        for i, role in enumerate(roles.tolist()):
            if i < robot_count:
                geometry.append(geometry_from_role(int(AgentRole.ROBOT)))
            else:
                geometry.append(geometry_from_label(labels[i], int(role)))
        geometry_t = torch.tensor(geometry, dtype=torch.float32)
        geometry_t = geometry_t * agent_mask_t[:, None].to(torch.float32)

        return {
            "history": history_t,
            "future": future_t,
            "current_pos": history_t[:, -1, :2],
            "goals": goals_t,
            "goal_mask": goal_mask_t,
            "roles": roles,
            "geometry": geometry_t,
            "agent_mask": agent_mask_t,
            "robot_mask": robot_mask_t,
            "entity_mask": entity_mask_t,
            "loss_mask": loss_mask_t,
            "dt": self.dt,
        }


class SDDOfficialSceneDataset(SDDRawSceneDataset):
    """Build SAGE samples directly from official SDD annotations.txt files."""

    def __init__(
        self,
        sdd_root: str | Path,
        split_reference_pkl: str | Path,
        history: int = 8,
        horizon: int = 12,
        max_entities: int = 12,
        num_robot_proxies: int = 1,
        dt: float = 0.4,
        frame_stride: int = 12,
        coordinate_scale: float = 50.0,
        center_scene: bool = True,
        include_entities_without_future: bool = False,
        role_strategy: str = "semantic_or_speed",
        role_slow_speed: float = 0.15,
        role_fast_speed: float = 1.8,
        role_slow_quantile: float = 0.33,
        role_fast_quantile: float = 0.67,
        limit_scenes: int = 0,
        coordinate: str = "center",
        perception_mode: str = "nearest",
        perception_base_radius: float = 3.6,
        perception_max_radius: float = 8.0,
        perception_speed_scale: float = 0.8,
        perception_geometry_scale: float = 0.5,
        perception_role_radius: dict[int, float] | None = None,
        perception_fallback_to_nearest: bool = True,
    ) -> None:
        if pd is None:
            raise ModuleNotFoundError("pandas is required to load official SDD annotations")
        self.pkl_path = Path(split_reference_pkl)
        self.sdd_root = Path(sdd_root)
        self.history = int(history)
        self.horizon = int(horizon)
        self.max_entities = int(max_entities)
        self.num_robot_proxies = max(1, int(num_robot_proxies))
        self.dt = float(dt)
        self.frame_stride = int(frame_stride)
        self.coordinate_scale = float(coordinate_scale)
        self.center_scene = bool(center_scene)
        self.include_entities_without_future = include_entities_without_future
        self.role_strategy = role_strategy
        self.role_slow_speed = float(role_slow_speed)
        self.role_fast_speed = float(role_fast_speed)
        self.role_slow_quantile = float(role_slow_quantile)
        self.role_fast_quantile = float(role_fast_quantile)
        self.perception_mode = str(perception_mode).lower()
        self.perception_base_radius = float(perception_base_radius)
        self.perception_max_radius = float(perception_max_radius)
        self.perception_speed_scale = float(perception_speed_scale)
        self.perception_geometry_scale = float(perception_geometry_scale)
        self.perception_role_radius = dict(DEFAULT_ROLE_RADIUS if perception_role_radius is None else perception_role_radius)
        self.perception_fallback_to_nearest = bool(perception_fallback_to_nearest)

        scene_ids = sorted(load_trajnet_scene_ids(split_reference_pkl))
        self.raw_df, self.missing_scene_ids = read_official_sdd_annotations(
            self.sdd_root,
            scene_ids=scene_ids,
            coordinate=coordinate,
        )
        self.label_column = "label"
        self.label_non_null_rows = _count_non_null_labels(self.raw_df, self.label_column)
        self.role_source = "semantic" if self.label_non_null_rows > 0 else "speed_proxy"
        self.scenes = self._build_scenes(limit_scenes=limit_scenes)
        self.indices = self._build_index()


def collate_sdd_scene_batch(samples: Sequence[dict]) -> HeterogeneousSceneBatch:
    dt = float(samples[0]["dt"])
    batch = HeterogeneousSceneBatch(
        history=torch.stack([s["history"] for s in samples], dim=0),
        future=torch.stack([s["future"] for s in samples], dim=0),
        current_pos=torch.stack([s["current_pos"] for s in samples], dim=0),
        goals=torch.stack([s["goals"] for s in samples], dim=0),
        goal_mask=torch.stack([s["goal_mask"] for s in samples], dim=0),
        roles=torch.stack([s["roles"] for s in samples], dim=0),
        agent_mask=torch.stack([s["agent_mask"] for s in samples], dim=0),
        robot_mask=torch.stack([s["robot_mask"] for s in samples], dim=0),
        entity_mask=torch.stack([s["entity_mask"] for s in samples], dim=0),
        loss_mask=torch.stack([s["loss_mask"] for s in samples], dim=0),
        geometry=torch.stack([s["geometry"] for s in samples], dim=0),
        dt=dt,
    )
    batch.validate()
    return batch
