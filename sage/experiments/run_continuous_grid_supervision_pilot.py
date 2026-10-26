"""Train continuous TRACER on counterfactual parameter-grid responses.

The ordinary continuous pilot supervises only the realized parameter.  This
pilot holds a simulator state and joint candidate fixed, changes one entity's
continuous Social-Force coordinate over a grid, and supplies the resulting
response surface to the model.  It is a diagnostic extension of the frozen
experiment, not a replacement for its records.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sage.experiments.run_continuous_ranking_pilot import _set_seed
from sage.experiments.run_continuous_tracer_pilot import _closed_loop
from sage.experiments.run_independent_multientity_suite import _build_spec
from sage.response.closed_loop import PlanningCostConfig
from sage.response.independent_social_force import (
    MultiEntityIndependentSocialForceSimulator,
)
from sage.response.persistent_models import (
    BeliefConditionedTeamResponseModel,
    PersistentTrajectoryModelConfig,
    continuous_coordinate_nll,
    continuous_trajectory_nll,
)


class ConditionalGridDataset(Dataset):
    def __init__(self, args: argparse.Namespace, episodes: int) -> None:
        self.args = args
        self.episodes = int(episodes)
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return self.episodes

    def _generate(self, index: int) -> dict[str, torch.Tensor]:
        spec = _build_spec(self.args, self.args.train_split, self.args.num_robots)
        simulator = MultiEntityIndependentSocialForceSimulator(
            spec,
            index,
            self.args.num_entities,
            self.args.active_entities,
            continuous_parameters=True,
            entity_interaction_strength=self.args.entity_interaction_strength,
            entity_interaction_decay=self.args.entity_interaction_decay,
            entity_interaction_radius=self.args.entity_interaction_radius,
        )
        planning = PlanningCostConfig(task_steps=self.args.execute_steps)
        grid = np.linspace(
            self.args.coordinate_min,
            self.args.coordinate_max,
            self.args.grid_size,
            dtype=np.float32,
        )
        histories = np.zeros(
            (self.args.windows, self.args.num_entities, self.args.history, 2),
            dtype=np.float32,
        )
        currents = np.zeros(
            (self.args.windows, self.args.num_entities, self.args.num_robots, 4),
            dtype=np.float32,
        )
        candidates = np.zeros(
            (
                self.args.windows,
                self.args.num_entities,
                self.args.num_robots,
                self.args.horizon,
                4,
            ),
            dtype=np.float32,
        )
        responses = np.zeros(
            (
                self.args.windows,
                self.args.num_entities,
                self.args.grid_size,
                self.args.horizon,
                2,
            ),
            dtype=np.float32,
        )
        active = np.zeros((self.args.windows, self.args.num_entities), dtype=np.bool_)
        rng = np.random.default_rng(self.args.data_seed + index * 9176)
        for window in range(self.args.windows):
            pool = simulator.candidate_pool()
            selected = int(rng.integers(self.args.num_candidates))
            snapshot = copy.deepcopy(simulator)
            histories[window] = simulator.histories
            currents[window] = pool.robot_current[selected].numpy()
            candidates[window] = pool.robot_candidates[selected].numpy()
            active[window] = simulator.active_mask()
            for entity in range(self.args.num_entities):
                for grid_index, coordinate in enumerate(grid):
                    counterfactual = copy.deepcopy(snapshot)
                    coordinates = counterfactual.parameter_coordinates.copy()
                    coordinates[entity] = float(coordinate)
                    counterfactual.set_parameter_coordinates(coordinates)
                    outcome = counterfactual.counterfactuals(pool, planning)[selected]
                    responses[window, entity, grid_index] = outcome.response[entity].numpy()
            simulator.step(pool, selected, planning)
        return {
            "history": torch.from_numpy(histories),
            "current": torch.from_numpy(currents),
            "candidates": torch.from_numpy(candidates),
            "responses": torch.from_numpy(responses),
            "active": torch.from_numpy(active),
            "coordinate": torch.from_numpy(simulator.parameter_coordinates.copy()),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.episodes:
            raise IndexError(index)
        if index not in self._cache:
            self._cache[index] = self._generate(index)
        return self._cache[index]


def _make_model(args: argparse.Namespace) -> BeliefConditionedTeamResponseModel:
    return BeliefConditionedTeamResponseModel(
        PersistentTrajectoryModelConfig(
            history=args.history,
            horizon=args.horizon,
            num_robots=args.num_robots,
            num_modes=1,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            dt=args.dt,
            architecture="tracer",
            num_heads=4,
            graph_layers=2,
            prefix_loss_weight=args.prefix_loss_weight,
            conditioning="continuous",
            coordinate_min=args.coordinate_min,
            coordinate_max=args.coordinate_max,
        )
    ).to(args.device)


def _train(model: BeliefConditionedTeamResponseModel, loader: DataLoader, args: argparse.Namespace) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        batches = 0
        for raw in loader:
            batch_size, windows, entities = raw["history"].shape[:3]
            grid_count = raw["responses"].shape[3]
            target_history = raw["history"].to(args.device).permute(0, 2, 1, 3, 4)
            target_current = raw["current"].to(args.device).permute(0, 2, 1, 3, 4)
            target_candidates = raw["candidates"].to(args.device).permute(0, 2, 1, 3, 4, 5)
            target_response = raw["responses"].to(args.device).permute(
                0, 2, 3, 1, 4, 5
            )
            target_active = raw["active"].to(args.device).permute(0, 2, 1)
            flat_history = target_history[:, :, None].expand(
                batch_size, entities, grid_count, windows, args.history, 2
            ).reshape(batch_size * entities * grid_count, windows, args.history, 2)
            flat_current = target_current[:, :, None].expand(
                batch_size, entities, grid_count, windows, args.num_robots, 4
            ).reshape(batch_size * entities * grid_count, windows, args.num_robots, 4)
            flat_candidates = target_candidates[:, :, None].expand(
                batch_size, entities, grid_count, windows, args.num_robots,
                args.horizon, 4
            ).reshape(
                batch_size * entities * grid_count, windows, args.num_robots,
                args.horizon, 4
            )
            neighbors = target_history[:, :, None, None].expand(
                batch_size, entities, grid_count, entities, windows, args.history, 2
            ).permute(0, 1, 2, 4, 3, 5, 6).reshape(
                batch_size * entities * grid_count, windows, entities, args.history, 2
            )
            neighbor_mask = target_active[:, :, None].expand(
                batch_size, entities, grid_count, windows
            )[:, :, :, :, None].expand(
                batch_size, entities, grid_count, windows, entities
            )
            neighbor_mask = neighbor_mask & ~torch.eye(
                entities, device=args.device, dtype=torch.bool
            )[None, :, None, None, :]
            neighbor_mask = neighbor_mask.reshape(
                batch_size * entities * grid_count, windows, entities
            )
            coordinates = torch.linspace(
                args.coordinate_min, args.coordinate_max, grid_count, device=args.device
            )[None, None, :, None].expand(
                batch_size, entities, grid_count, windows
            ).reshape(batch_size * entities * grid_count, windows, 1)
            output = model(
                flat_history,
                flat_current,
                flat_candidates,
                neighbor_history=neighbors,
                neighbor_mask=neighbor_mask,
                condition_coordinates=coordinates,
            )
            target = target_response.reshape(
                batch_size * entities * grid_count, windows, args.horizon, 2
            )
            active_flat = target_active[:, :, None].expand(
                batch_size, entities, grid_count, windows
            ).reshape(batch_size * entities * grid_count, windows)
            loss = continuous_trajectory_nll(
                output["mean"], output["log_std"], target, active_flat
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        value = total / max(batches, 1)
        history.append(value)
        print(f"grid-response epoch={epoch + 1}/{args.epochs} loss={value:.6f}", flush=True)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/continuous_grid_supervision_pilot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--data-seed", type=int, default=618731)
    parser.add_argument("--train-split", choices=["train", "ood"], default="ood")
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--num-entities", type=int, default=6)
    parser.add_argument("--active-entities", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.3)
    parser.add_argument("--response-noise", type=float, default=0.08)
    parser.add_argument("--entity-max-speed", type=float, default=2.2)
    parser.add_argument("--entity-interaction-strength", type=float, default=0.8)
    parser.add_argument("--entity-interaction-decay", type=float, default=0.72)
    parser.add_argument("--entity-interaction-radius", type=float, default=0.85)
    parser.add_argument("--scene-family", default="crossing")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prefix-loss-weight", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--coordinate-min", type=float, default=0.0)
    parser.add_argument("--coordinate-max", type=float, default=2.0)
    parser.add_argument("--likelihood-scale", type=float, default=0.5)
    parser.add_argument("--collision-margin", type=float, default=0.65)
    parser.add_argument("--risk-alpha", type=float, default=0.3)
    parser.add_argument("--risk-samples", type=int, default=32)
    parser.add_argument("--risk-aggregation", choices=["union", "max", "mean"], default="mean")
    parser.add_argument("--fallback-risk-weight", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.num_threads))
    _set_seed(args.seed)
    dataset = ConditionalGridDataset(args, args.train_episodes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = _make_model(args)
    train_loss = _train(model, loader, args)
    persistent = _closed_loop(model, args, persistent=True, uncertainty_scale=1.0)
    reset = _closed_loop(model, args, persistent=False, uncertainty_scale=1.0)
    payload = {
        "config": vars(args),
        "train_loss": train_loss,
        "persistent": persistent,
        "reset": reset,
        "reset_minus_persistent": reset["cost"] - persistent["cost"],
        "model_config": vars(model.config),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": vars(model.config)}, output_dir / "model.pt")
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
