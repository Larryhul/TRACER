from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from tracer.data.heterogeneous_scene import AgentRole, HeterogeneousSceneBatch
from tracer.data.mid_adapter import infer_entity_roles
from tracer.data.perception import DEFAULT_ROLE_RADIUS, effective_radius
from tracer.compat import load_legacy_pickle
from tracer.legacy.utils.trajectron_hypers import get_traj_hypers

@dataclass(frozen=True)
class SceneIndex:
    scene_idx: int
    timestep: int
    node_idx: int


class MIDRealSceneDataset(Dataset):
    """Build SAGE samples directly from MID processed Trajectron environments."""

    def __init__(
        self,
        pkl_path: str | Path,
        history: int = 8,
        horizon: int = 12,
        max_entities: int = 8,
        num_robot_proxies: int = 1,
        node_type: str = "PEDESTRIAN",
        include_entities_without_future: bool = False,
        role_strategy: str = "speed_threshold",
        role_slow_speed: float = 0.15,
        role_fast_speed: float = 1.8,
        role_slow_quantile: float = 0.33,
        role_fast_quantile: float = 0.67,
        perception_mode: str = "nearest",
        perception_base_radius: float = 3.6,
        perception_max_radius: float = 8.0,
        perception_speed_scale: float = 0.8,
        perception_geometry_scale: float = 0.0,
        perception_role_radius: dict[int, float] | None = None,
        perception_fallback_to_nearest: bool = True,
    ) -> None:
        self.pkl_path = Path(pkl_path)
        with self.pkl_path.open("rb") as f:
            try:
                self.env = load_legacy_pickle(f, encoding="latin1")
            except TypeError:
                f.seek(0)
                self.env = load_legacy_pickle(f)
        self.history = int(history)
        self.horizon = int(horizon)
        self.max_entities = int(max_entities)
        self.num_robot_proxies = max(1, int(num_robot_proxies))
        self.node_type = node_type
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
        self.hyperparams = get_traj_hypers()
        self.indices = self._build_index()

    def _node_type(self):
        for node_type in self.env.NodeType:
            if str(node_type) == self.node_type or getattr(node_type, "name", None) == self.node_type:
                return node_type
        return self.node_type

    def _build_index(self) -> List[SceneIndex]:
        out: List[SceneIndex] = []
        node_type = self._node_type()
        for scene_idx, scene in enumerate(self.env.scenes):
            present = scene.present_nodes(
                np.arange(0, scene.timesteps),
                type=node_type,
                min_history_timesteps=self.history - 1,
                min_future_timesteps=self.horizon,
                return_robot=True,
            )
            for timestep, nodes in present.items():
                for node in nodes:
                    out.append(SceneIndex(scene_idx, int(timestep), scene.nodes.index(node)))
        return out

    def __len__(self) -> int:
        return len(self.indices)

    def _state(self, node) -> dict:
        return self.hyperparams["state"][node.type]

    def _pred_state(self, node) -> dict:
        return self.hyperparams["pred_state"][node.type]

    def _history(self, node, timestep: int) -> np.ndarray:
        arr = node.get(np.array([timestep - self.history + 1, timestep]), self._state(node), padding=0.0)
        return arr[:, :2].astype(np.float32)

    def _future(self, node, timestep: int, valid_required: bool) -> tuple[np.ndarray, bool]:
        has_future = node.first_timestep <= timestep + 1 and timestep + self.horizon <= node.last_timestep
        if valid_required and not has_future:
            raise ValueError("target node does not have a full future")
        arr = node.get(np.array([timestep + 1, timestep + self.horizon]), self._pred_state(node), padding=0.0)
        return arr[:, :2].astype(np.float32), bool(has_future)

    def _instant_speed(self, node, timestep: int, dt: float) -> float:
        arr = node.get(np.array([timestep - 1, timestep]), self._state(node), padding=0.0)
        if arr.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(arr[-1, :2] - arr[-2, :2]) / max(float(dt), 1e-6))

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

    def _neighbors(self, scene, target, timestep: int) -> Sequence:
        try:
            graph = scene.get_scene_graph(
                timestep,
                self.env.attention_radius,
                self.hyperparams["edge_addition_filter"],
                self.hyperparams["edge_removal_filter"],
            )
            neighbors = []
            for edge_type in self.env.get_edge_types():
                if edge_type[0] != target.type:
                    continue
                neighbors.extend(graph.get_neighbors(target, edge_type[1]))
        except Exception:
            present = scene.present_nodes(
                np.array([timestep]),
                min_history_timesteps=self.history - 1,
                min_future_timesteps=0,
                return_robot=True,
            )
            neighbors = [node for node in present.get(timestep, []) if node is not target]

        candidates = []
        in_radius = []
        seen = set()
        target_pos = target.get(np.array([timestep]), {"position": ["x", "y"]})[-1]
        use_radius = self.perception_mode in {"role_radius", "heterogeneous", "radius"}
        for node in neighbors:
            if id(node) in seen or node is target:
                continue
            seen.add(id(node))
            node_pos = node.get(np.array([timestep]), {"position": ["x", "y"]}, padding=0.0)[-1]
            dist = float(np.linalg.norm(node_pos - target_pos))
            candidates.append((dist, node))
            if use_radius:
                speed = self._instant_speed(node, timestep, float(scene.dt))
                role_id = self._proxy_role_from_speed(speed)
                radius = effective_radius(
                    role_id,
                    speed,
                    None,
                    base_radius=self.perception_base_radius,
                    max_radius=self.perception_max_radius,
                    speed_scale=self.perception_speed_scale,
                    geometry_scale=self.perception_geometry_scale,
                    role_radius=self.perception_role_radius,
                )
                if dist <= radius:
                    in_radius.append((dist, node))
        candidates.sort(key=lambda item: item[0])
        in_radius.sort(key=lambda item: item[0])
        if use_radius and (in_radius or not self.perception_fallback_to_nearest):
            return [node for _, node in in_radius[: self.max_entities]]
        return [node for _, node in candidates[: self.max_entities]]

    def _robot_proxy_nodes(self, scene, target, timestep: int) -> list:
        robots = [target]
        if self.num_robot_proxies <= 1:
            return robots
        present = scene.present_nodes(
            np.array([timestep]),
            min_history_timesteps=self.history - 1,
            min_future_timesteps=self.horizon,
            return_robot=True,
        )
        target_pos = target.get(np.array([timestep]), {"position": ["x", "y"]}, padding=0.0)[-1]
        candidates = []
        for node in present.get(timestep, []):
            if node is target:
                continue
            try:
                self._future(node, timestep, valid_required=True)
            except ValueError:
                continue
            node_pos = node.get(np.array([timestep]), {"position": ["x", "y"]}, padding=0.0)[-1]
            dist = float(np.linalg.norm(node_pos - target_pos))
            candidates.append((dist, node))
        candidates.sort(key=lambda item: item[0])
        robots.extend(node for _, node in candidates[: self.num_robot_proxies - 1])
        return robots

    def __getitem__(self, idx: int) -> dict:
        item = self.indices[idx]
        scene = self.env.scenes[item.scene_idx]
        target = scene.nodes[item.node_idx]
        timestep = item.timestep

        robot_nodes = self._robot_proxy_nodes(scene, target, timestep)
        histories = [self._history(node, timestep) for node in robot_nodes]
        futures = [self._future(node, timestep, valid_required=True)[0] for node in robot_nodes]
        agent_mask = [True for _ in robot_nodes]
        loss_mask = [True for _ in robot_nodes]
        robot_count = len(robot_nodes)
        excluded = {id(node) for node in robot_nodes}

        for neighbor in self._neighbors(scene, target, timestep):
            if id(neighbor) in excluded:
                continue
            fut, valid_future = self._future(neighbor, timestep, valid_required=False)
            if not valid_future and not self.include_entities_without_future:
                continue
            histories.append(self._history(neighbor, timestep))
            futures.append(fut)
            agent_mask.append(True)
            loss_mask.append(valid_future)

        while len(histories) < self.max_entities + self.num_robot_proxies:
            histories.append(np.zeros((self.history, 2), dtype=np.float32))
            futures.append(np.zeros((self.horizon, 2), dtype=np.float32))
            agent_mask.append(False)
            loss_mask.append(False)

        histories = histories[: self.max_entities + self.num_robot_proxies]
        futures = futures[: self.max_entities + self.num_robot_proxies]
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
        goals_t = history_t[:, -1, :2] + future_t.sum(dim=1) * float(scene.dt)
        goal_mask_t = robot_mask_t & agent_mask_t
        roles = infer_entity_roles(
            future_t[None],
            entity_mask_t[None],
            strategy=self.role_strategy,
            slow_speed=self.role_slow_speed,
            fast_speed=self.role_fast_speed,
            slow_quantile=self.role_slow_quantile,
            fast_quantile=self.role_fast_quantile,
        )[0]
        roles[:robot_count] = int(AgentRole.ROBOT)

        return {
            "history": history_t,
            "future": future_t,
            "current_pos": history_t[:, -1, :2],
            "goals": goals_t,
            "goal_mask": goal_mask_t,
            "roles": roles,
            "agent_mask": agent_mask_t,
            "robot_mask": robot_mask_t,
            "entity_mask": entity_mask_t,
            "loss_mask": loss_mask_t,
            "dt": float(scene.dt),
        }


def collate_sage_scene_batch(samples: Sequence[dict]) -> HeterogeneousSceneBatch:
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
        dt=dt,
    )
    batch.validate()
    return batch
