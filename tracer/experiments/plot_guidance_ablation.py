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
    "pir",
    "svr",
    "energy",
    "mean_robot_speed",
    "max_robot_speed_observed",
    "speed_violation_rate",
]


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


def aggregate(rows: list[dict[str, object]], comfort_profile: str) -> list[dict[str, object]]:
    filtered = [row for row in rows if row.get("comfort_profile") == comfort_profile and row.get("guided") is True]
    grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in filtered:
        grouped[float(row["guidance_scale"])].append(row)

    out: list[dict[str, object]] = []
    for scale in sorted(grouped):
        subset = grouped[scale]
        item: dict[str, object] = {
            "guidance_scale": scale,
            "num_datasets": len({row.get("dataset") for row in subset}),
            "comfort_profile": comfort_profile,
        }
        for metric in METRICS:
            item[metric] = mean([row.get(metric) for row in subset])
        out.append(item)
    return out


def aggregate_effect(scale_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    base = next((row for row in scale_rows if float(row["guidance_scale"]) == 0.0), None)
    if base is None:
        return []
    out: list[dict[str, object]] = []
    for row in scale_rows:
        scale = float(row["guidance_scale"])
        if scale == 0.0:
            continue
        item: dict[str, object] = {"guidance_scale": scale}
        for metric in ("collision_rate", "pir", "svr", "energy"):
            item[f"{metric}_reduction_pct"] = 100.0 * (float(base[metric]) - float(row[metric])) / float(base[metric])
        item["min_distance_multiplier"] = float(row["min_distance"]) / float(base["min_distance"])
        item["robot_ade_change_pct"] = 100.0 * (float(row["robot_ade"]) - float(base["robot_ade"])) / float(base["robot_ade"])
        item["robot_fde_change_pct"] = 100.0 * (float(row["robot_fde"]) - float(base["robot_fde"])) / float(base["robot_fde"])
        out.append(item)
    return out


def plot_safety_curves(rows: list[dict[str, object]], output: Path) -> None:
    x = [float(row["guidance_scale"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), dpi=220)
    axes_flat = axes.ravel()
    specs = [
        ("collision_rate", "Collision Rate", "#2F6F73"),
        ("pir", "PIR", "#B35C44"),
        ("svr", "SVR", "#7D5BA6"),
        ("energy", "Energy", "#4F6D7A"),
    ]
    for ax, (metric, title, color) in zip(axes_flat, specs):
        y = [float(row[metric]) for row in rows]
        ax.plot(x, y, marker="o", markersize=4.2, color=color, linewidth=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Guidance scale", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(alpha=0.25)
    axes_flat[0].set_ylabel("Metric value", fontsize=9)
    axes_flat[2].set_ylabel("Metric value", fontsize=9)
    fig.tight_layout(pad=0.8, w_pad=0.8, h_pad=0.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(rows: list[dict[str, object]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=180)
    x = [float(row["robot_ade"]) for row in rows]
    y = [float(row["collision_rate"]) for row in rows]
    scales = [float(row["guidance_scale"]) for row in rows]
    sizes = [65 + 500 * scale for scale in scales]
    ax.plot(x, y, color="#A7B0B8", linewidth=1.5, zorder=1)
    scatter = ax.scatter(x, y, s=sizes, c=scales, cmap="viridis", edgecolor="white", linewidth=0.8, zorder=2)
    for xi, yi, scale in zip(x, y, scales):
        ax.annotate(f"{scale:.2f}", (xi, yi), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Robot ADE")
    ax.set_ylabel("Collision Rate")
    ax.set_title("Guidance Scale Safety-Accuracy Trade-off")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Guidance scale")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SAGE guidance-scale ablation results.")
    parser.add_argument("--csv", default="results/sage_guidance_ablation.csv")
    parser.add_argument("--comfort-profile", default="role")
    parser.add_argument("--table-dir", default="results/tables")
    parser.add_argument("--figure-dir", default="results/figures/paper")
    args = parser.parse_args()

    rows = read_rows(Path(args.csv))
    scale_rows = aggregate(rows, comfort_profile=args.comfort_profile)
    effect_rows = aggregate_effect(scale_rows)

    table_dir = Path(args.table_dir)
    figure_dir = Path(args.figure_dir)
    write_csv(table_dir / "guidance_scale_ablation.csv", scale_rows, ["guidance_scale", "num_datasets", "comfort_profile", *METRICS])
    write_csv(
        table_dir / "guidance_scale_effect.csv",
        effect_rows,
        [
            "guidance_scale",
            "collision_rate_reduction_pct",
            "pir_reduction_pct",
            "svr_reduction_pct",
            "energy_reduction_pct",
            "min_distance_multiplier",
            "robot_ade_change_pct",
            "robot_fde_change_pct",
        ],
    )
    plot_safety_curves(scale_rows, figure_dir / "guidance_scale_safety_curves.png")
    plot_tradeoff(scale_rows, figure_dir / "guidance_scale_tradeoff.png")

    print(f"saved_table={table_dir / 'guidance_scale_ablation.csv'}")
    print(f"saved_table={table_dir / 'guidance_scale_effect.csv'}")
    print(f"saved_figure={figure_dir / 'guidance_scale_safety_curves.png'}")
    print(f"saved_figure={figure_dir / 'guidance_scale_tradeoff.png'}")


if __name__ == "__main__":
    main()
