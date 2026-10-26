from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    "robot_ade",
    "robot_fde",
    "entity_ade",
    "entity_fde",
    "collision_rate",
    "min_distance",
    "energy",
    "pir",
    "svr",
    "speed_violation_rate",
]
STRATEGY_ORDER = ["speed_threshold", "speed_quantile", "all_pha"]
ROLE_ORDER = ["pha", "hnaa", "ssaa"]


def to_value(value: str):
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return float(value)
    except ValueError:
        return value


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{key: to_value(value) for key, value in row.items()} for row in csv.DictReader(f)]


def mean(values: list[object]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate_strategy(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, bool], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("role_strategy") in STRATEGY_ORDER and row.get("guided") in (True, False):
            grouped[(str(row["role_strategy"]), bool(row["guided"]))].append(row)

    out: list[dict[str, object]] = []
    for strategy in STRATEGY_ORDER:
        for guided in (False, True):
            subset = grouped.get((strategy, guided), [])
            if not subset:
                continue
            item: dict[str, object] = {
                "role_strategy": strategy,
                "method": "guided" if guided else "unguided",
                "num_datasets": len({row.get("dataset") for row in subset}),
            }
            for metric in METRICS:
                item[metric] = mean([row.get(metric) for row in subset])
            out.append(item)
    return out


def aggregate_guidance_effect(strategy_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_strategy: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in strategy_rows:
        by_strategy[str(row["role_strategy"])][str(row["method"])] = row

    out: list[dict[str, object]] = []
    for strategy in STRATEGY_ORDER:
        guided = by_strategy.get(strategy, {}).get("guided")
        unguided = by_strategy.get(strategy, {}).get("unguided")
        if not guided or not unguided:
            continue
        item: dict[str, object] = {"role_strategy": strategy}
        item["collision_reduction_pct"] = 100.0 * (float(unguided["collision_rate"]) - float(guided["collision_rate"])) / float(unguided["collision_rate"])
        item["energy_reduction_pct"] = 100.0 * (float(unguided["energy"]) - float(guided["energy"])) / float(unguided["energy"])
        if unguided.get("pir") is not None and guided.get("pir") is not None and float(unguided["pir"]) > 0:
            item["pir_reduction_pct"] = 100.0 * (float(unguided["pir"]) - float(guided["pir"])) / float(unguided["pir"])
        if unguided.get("svr") is not None and guided.get("svr") is not None and float(unguided["svr"]) > 0:
            item["svr_reduction_pct"] = 100.0 * (float(unguided["svr"]) - float(guided["svr"])) / float(unguided["svr"])
        item["min_distance_multiplier"] = float(guided["min_distance"]) / float(unguided["min_distance"])
        item["robot_ade_change_pct"] = 100.0 * (float(guided["robot_ade"]) - float(unguided["robot_ade"])) / float(unguided["robot_ade"])
        item["robot_fde_change_pct"] = 100.0 * (float(guided["robot_fde"]) - float(unguided["robot_fde"])) / float(unguided["robot_fde"])
        out.append(item)
    return out


def aggregate_rolewise_speed_threshold(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for guided in (False, True):
        subset = [row for row in rows if row.get("role_strategy") == "speed_threshold" and row.get("guided") is guided]
        for role in ROLE_ORDER:
            item: dict[str, object] = {
                "method": "guided" if guided else "unguided",
                "role": role,
                "num_datasets": len({row.get("dataset") for row in subset}),
            }
            for suffix, metric_name in (
                ("ade", "ade"),
                ("fde", "fde"),
                ("collision_rate", "collision_rate"),
                ("min_distance", "min_distance"),
                ("pir", "pir"),
                ("svr", "svr"),
                ("count", "count"),
            ):
                item[metric_name] = mean([row.get(f"role_{role}_{suffix}") for row in subset])
            out.append(item)
    return out


def plot_strategy_safety(strategy_rows: list[dict[str, object]], output: Path) -> None:
    values = {(row["role_strategy"], row["method"]): row for row in strategy_rows}
    metrics = [
        ("collision_rate", "Collision Rate", "lower"),
        ("pir", "PIR", "lower"),
        ("svr", "SVR", "lower"),
        ("min_distance", "Minimum Distance", "higher"),
        ("energy", "Energy", "lower"),
    ]
    labels = ["Speed threshold", "Speed quantile", "All PHA"]
    x = list(range(len(STRATEGY_ORDER)))
    width = 0.34
    colors = {"unguided": "#9AA6B2", "guided": "#2F6F73"}

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.5), dpi=180)
    for ax, (metric, title, direction) in zip(axes, metrics):
        if any(values[(s, method)].get(metric) is None for s in STRATEGY_ORDER for method in ("unguided", "guided")):
            ax.set_visible(False)
            continue
        unguided = [float(values[(s, "unguided")][metric]) for s in STRATEGY_ORDER]
        guided = [float(values[(s, "guided")][metric]) for s in STRATEGY_ORDER]
        ax.bar([i - width / 2 for i in x], unguided, width=width, label="Unguided", color=colors["unguided"])
        ax.bar([i + width / 2 for i in x], guided, width=width, label="Guided", color=colors["guided"])
        ax.set_title(f"{title} ({direction})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if metric == "collision_rate":
            ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_tradeoff(strategy_rows: list[dict[str, object]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=180)
    markers = {"speed_threshold": "o", "speed_quantile": "s", "all_pha": "^"}
    colors = {"unguided": "#9AA6B2", "guided": "#2F6F73"}
    for row in strategy_rows:
        strategy = str(row["role_strategy"])
        method = str(row["method"])
        x = float(row["robot_ade"])
        y = float(row["collision_rate"])
        ax.scatter(x, y, s=85, marker=markers[strategy], color=colors[method], edgecolor="white", linewidth=0.8)
        label = strategy.replace("speed_", "spd_").replace("_", "-")
        ax.annotate(label if method == "guided" else "", (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Robot ADE")
    ax.set_ylabel("Collision Rate")
    ax.set_title("Safety-Accuracy Trade-off by Role Strategy")
    ax.grid(alpha=0.25)
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="Guided", markerfacecolor=colors["guided"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Unguided", markerfacecolor=colors["unguided"], markersize=8),
        Line2D([0], [0], marker="o", color="black", label="speed_threshold", linestyle="None", markersize=7),
        Line2D([0], [0], marker="s", color="black", label="speed_quantile", linestyle="None", markersize=7),
        Line2D([0], [0], marker="^", color="black", label="all_pha", linestyle="None", markersize=7),
    ]
    ax.legend(handles=legend_items, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_rolewise_safety(role_rows: list[dict[str, object]], output: Path) -> None:
    values = {(row["role"], row["method"]): row for row in role_rows}
    x = list(range(len(ROLE_ORDER)))
    width = 0.34
    labels = [role.upper() for role in ROLE_ORDER]
    colors = {"unguided": "#9AA6B2", "guided": "#2F6F73"}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), dpi=180)
    for ax, metric, title in zip(axes, ["collision_rate", "min_distance", "pir", "svr"], ["Role-wise Collision Rate", "Role-wise Minimum Distance", "Role-wise PIR", "Role-wise SVR"]):
        if any(values[(role, method)].get(metric) is None for role in ROLE_ORDER for method in ("unguided", "guided")):
            ax.set_visible(False)
            continue
        unguided = [float(values[(role, "unguided")][metric]) for role in ROLE_ORDER]
        guided = [float(values[(role, "guided")][metric]) for role in ROLE_ORDER]
        ax.bar([i - width / 2 for i in x], unguided, width=width, color=colors["unguided"], label="Unguided")
        ax.bar([i + width / 2 for i in x], guided, width=width, color=colors["guided"], label="Guided")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot role ablation results for SAGE.")
    parser.add_argument("--csv", default="results/sage_role_ablation.csv")
    parser.add_argument("--table-dir", default="results/tables")
    parser.add_argument("--figure-dir", default="results/figures/paper")
    args = parser.parse_args()

    rows = read_rows(Path(args.csv))
    table_dir = Path(args.table_dir)
    figure_dir = Path(args.figure_dir)

    strategy_rows = aggregate_strategy(rows)
    effect_rows = aggregate_guidance_effect(strategy_rows)
    role_rows = aggregate_rolewise_speed_threshold(rows)

    write_csv(table_dir / "role_strategy_ablation.csv", strategy_rows, ["role_strategy", "method", "num_datasets", *METRICS])
    write_csv(
        table_dir / "role_strategy_guidance_effect.csv",
        effect_rows,
        ["role_strategy", "collision_reduction_pct", "energy_reduction_pct", "pir_reduction_pct", "svr_reduction_pct", "min_distance_multiplier", "robot_ade_change_pct", "robot_fde_change_pct"],
    )
    write_csv(table_dir / "rolewise_speed_threshold.csv", role_rows, ["method", "role", "num_datasets", "ade", "fde", "collision_rate", "min_distance", "pir", "svr", "count"])

    plot_strategy_safety(strategy_rows, figure_dir / "role_strategy_safety.png")
    plot_strategy_tradeoff(strategy_rows, figure_dir / "role_strategy_tradeoff.png")
    plot_rolewise_safety(role_rows, figure_dir / "rolewise_speed_threshold_safety.png")

    print(f"saved_table={table_dir / 'role_strategy_ablation.csv'}")
    print(f"saved_table={table_dir / 'role_strategy_guidance_effect.csv'}")
    print(f"saved_table={table_dir / 'rolewise_speed_threshold.csv'}")
    print(f"saved_figure={figure_dir / 'role_strategy_safety.png'}")
    print(f"saved_figure={figure_dir / 'role_strategy_tradeoff.png'}")
    print(f"saved_figure={figure_dir / 'rolewise_speed_threshold_safety.png'}")


if __name__ == "__main__":
    main()
