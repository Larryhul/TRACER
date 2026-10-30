"""CPU closed-loop illustration with a small synthetic response model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
from torch.utils.data import DataLoader

from tracer.belief.filter import PersistentBeliefStore
from tracer.response.closed_loop import ClosedLoopSpec, PlanningCostConfig, SyntheticClosedLoopSimulator, expected_candidate_costs
from tracer.response.persistent_models import BeliefConditionedTeamResponseModel, PersistentTrajectoryModelConfig, executed_prefix_mode_log_likelihood, stationary_reference_candidates, trajectory_mode_log_likelihood
from tracer.response.persistent_trajectory import PersistentTrajectoryEpisodeDataset, PersistentTrajectorySpec


def make_model(seed: int, checkpoint: Path | None = None):
    torch.manual_seed(seed)
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = BeliefConditionedTeamResponseModel(PersistentTrajectoryModelConfig(**payload["config"]))
        model.load_state_dict(payload["model"])
        return model.eval()
    config = PersistentTrajectoryModelConfig(history=4, horizon=8, hidden_dim=24, num_heads=2, graph_layers=1, dropout=0.0, architecture="tracer")
    model = BeliefConditionedTeamResponseModel(config)
    dataset = PersistentTrajectoryEpisodeDataset(PersistentTrajectorySpec(num_episodes=24, windows=3, history=4, horizon=8, seed=seed, cache_samples=True))
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(4):
        for batch in loader:
            prediction = model(batch["entity_history"], batch["robot_current"], batch["robot_candidates"])
            likelihood = trajectory_mode_log_likelihood(prediction["mean"], prediction["log_std"], batch["target_future_vel"])
            loss = -likelihood.gather(-1, batch["mode"][:, None, None].expand(-1, likelihood.shape[1], 1)).mean() / 16
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


def run(output: Path, seed: int = 2027, checkpoint: Path | None = None) -> dict:
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    model = make_model(seed, checkpoint)
    config = model.config
    spec = ClosedLoopSpec(windows=7, history=config.history, horizon=config.horizon, num_robots=config.num_robots, num_candidates=12, execute_steps=2, dt=config.dt, scene_family="crossing", task_type="fixed_goal", seed=seed)
    simulator = SyntheticClosedLoopSimulator(spec, episode_index=0)
    beliefs = PersistentBeliefStore(config.num_modes)
    cost = PlanningCostConfig(task_weight=1.0, safety_weight=3.0, human_weight=0.6, task_steps=2)
    robot_frames = [simulator.robot_positions.copy()]
    entity_frames = [simulator.entity_position.copy()]
    belief_frames = [beliefs.get("entity-0").numpy()]
    decisions = []
    for window in range(spec.windows):
        pool = simulator.candidate_pool()
        count = pool.robot_current.shape[0]
        history = torch.from_numpy(simulator.history)[None, None].expand(count, 1, -1, -1)
        current = pool.robot_current[:, None]
        candidates = pool.robot_candidates[:, None]
        prior = beliefs.get("entity-0")
        with torch.no_grad():
            prediction = model(history, current, candidates)
            reference = model(history, current, stationary_reference_candidates(current, config.horizon))
            means, stds = prediction["mean"][:, 0], prediction["log_std"][:, 0]
            scores = expected_candidate_costs(means, prior, pool.robot_current, pool.robot_candidates, reference["mean"][:, 0], spec.dt, cost, mode_log_stds=stds, task_goal_offsets=torch.from_numpy(simulator.robot_goals - simulator.entity_position))
        selected = int(scores["total"].argmin())
        start_robots, start_entity = simulator.robot_positions.copy(), simulator.entity_position.copy()
        realized = simulator.step(pool, selected, cost)
        likelihood = executed_prefix_mode_log_likelihood(means[selected], stds[selected], realized.response, spec.execute_steps)
        posterior = beliefs.update("entity-0", likelihood)
        for step in range(spec.execute_steps):
            start_robots = start_robots + pool.robot_candidates[selected, :, step, 2:].numpy() * spec.dt
            start_entity = start_entity + realized.response[step].numpy() * spec.dt
            robot_frames.append(start_robots.copy())
            entity_frames.append(start_entity.copy())
            belief_frames.append((posterior if step + 1 == spec.execute_steps else prior).numpy())
        decisions.append({"window": window, "selected_candidate": selected, "prior": prior.tolist(), "posterior": posterior.tolist(), "observed_prefix_steps": spec.execute_steps})
        np.testing.assert_allclose(start_robots, simulator.robot_positions, atol=1e-5)
    robots, entity, belief = map(np.asarray, (robot_frames, entity_frames, belief_frames))
    summary = {"example": "synthetic-closed-loop", "seed": seed, "scope": "Small generated-data response model and one illustrative episode.", "robot_positions": robots.tolist(), "entity_positions": entity.tolist(), "beliefs": belief.tolist(), "decisions": decisions}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": asdict(config), "purpose": "synthetic-demo", "seed": seed}, output / "tracer_demo.pt")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, belief_ax) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.65, 1]})
    fig.patch.set_facecolor("#f8fafc")
    colors = ["#0d9488", "#f97316"]
    lines = [ax.plot([], [], color=colors[i % 2], lw=2.5, label=f"Robot {i + 1}")[0] for i in range(spec.num_robots)]
    dots = [ax.plot([], [], "o", color=colors[i % 2], ms=7)[0] for i in range(spec.num_robots)]
    human, = ax.plot([], [], color="#6366f1", lw=2, label="Pedestrian")
    human_dot, = ax.plot([], [], "o", color="#6366f1", ms=8)
    ax.scatter(simulator.robot_goals[:, 0], simulator.robot_goals[:, 1], marker="*", s=120, color=colors[:spec.num_robots], label="Robot goals")
    points = np.concatenate([robots.reshape(-1, 2), entity, simulator.robot_goals])
    low, high = points.min(axis=0) - 0.7, points.max(axis=0) + 0.7
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8, loc="upper left")
    bars = belief_ax.bar(["Yielding", "Neutral", "Assertive"], belief[0], color=["#0d9488", "#64748b", "#f97316"])
    belief_ax.set(ylim=(0, 1), ylabel="Belief probability", title="Persistent belief · entity-0")
    title = fig.suptitle("TRACER | predict, observe and replan", fontweight="bold")
    fig.text(0.5, 0.015, "Synthetic demonstration · executed-prefix observations · stable entity identity", ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))

    def frame(index):
        for robot_index in range(spec.num_robots):
            lines[robot_index].set_data(robots[:index + 1, robot_index, 0], robots[:index + 1, robot_index, 1])
            dots[robot_index].set_data([robots[index, robot_index, 0]], [robots[index, robot_index, 1]])
        human.set_data(entity[:index + 1, 0], entity[:index + 1, 1])
        human_dot.set_data([entity[index, 0]], [entity[index, 1]])
        for bar, probability in zip(bars, belief[index]):
            bar.set_height(float(probability))
        title.set_text(f"TRACER | closed-loop step {index}/{len(robots) - 1}")
        return *lines, *dots, human, human_dot, *bars

    animation = FuncAnimation(fig, frame, frames=len(robots), interval=180)
    animation.save(output / "tracer_demo.gif", writer=PillowWriter(fps=6))
    frame(len(robots) - 1)
    fig.savefig(output / "tracer_demo.png", dpi=150)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    result = run(args.output, args.seed, args.checkpoint)
    print(f"Executed {len(result['decisions'])} decision windows on CPU.")
    print(f"Animation, checkpoint and summary: {args.output.resolve()}")


if __name__ == "__main__":
    main()
