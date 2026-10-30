"""Run the official AToM ILQGames numerical core without ROS.

The upstream AToM corridor model is a three-agent game.  Its numerical
sources do not call ROS from the model/solver path, so this runner uses a
small local ros/ros.h compatibility header and the unchanged upstream C++
sources.  This is a solver-core execution audit, not an AToM ROS episode or a
TRACER closed-loop comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = (2027, 2031, 2039, 2053, 2063)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _git_state(root: Path) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain")
    return {"commit": _git(root, "rev-parse", "HEAD"), "dirty": bool(status)}


def _external_state(path: Path) -> dict[str, Any]:
    try:
        commit = _git(path, "-C", str(path), "rev-parse", "HEAD")
        status = _git(path, "-C", str(path), "status", "--porcelain")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"available": False}
    return {"available": True, "commit": commit, "dirty": bool(status)}


def _run_one(
    root: Path, binary: Path, seed: int, count: int, output: Path
) -> tuple[list[dict[str, str]], str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = {"LD_LIBRARY_PATH": str(root / ".conda/envs/lucidgames-julia16/lib")}
    command = [
        str(binary),
        "--seed",
        str(seed),
        "--cases",
        str(count),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, **environment},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return list(csv.DictReader(output.open(newline=""))), completed.stdout


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = (root / args.binary).resolve()
    if not binary.is_file():
        raise FileNotFoundError(
            f"{binary} is missing; run tools/build_atom_headless.sh first"
        )

    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    rows: list[dict[str, str]] = []
    logs: dict[str, str] = {}
    for seed in seeds:
        seed_output = output_dir / f"results_seed_{seed}.csv"
        seed_rows, stdout = _run_one(root, binary, seed, args.cases, seed_output)
        for row in seed_rows:
            row["seed"] = str(seed)
        rows.extend(seed_rows)
        logs[str(seed)] = stdout
        (output_dir / f"solver_stdout_seed_{seed}.txt").write_text(stdout)

    rows_path = output_dir / "results.csv"
    fieldnames = ["seed"] + [
        name for name in rows[0] if name != "seed"
    ]
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def fraction(name: str) -> float:
        return sum(row[name] == "1" for row in rows) / len(rows)

    def mean(name: str) -> float:
        return sum(float(row[name]) for row in rows) / len(rows)

    atom_root = root / "raw_data/external/source_repos/atom_official"
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "atom-official-ilqgames-corridor3-headless-v1",
        "claim_boundary": (
            "official AToM three-agent numerical solver-core execution audit; "
            "not ROS runtime, not a TRACER closed-loop comparison, and not a "
            "real-human result"
        ),
        "main_git": _git_state(root),
        "atom_git": _external_state(atom_root),
        "binary": str(binary),
        "binary_sha256": _sha256(binary),
        "seeds": list(seeds),
        "cases_per_seed": args.cases,
        "rows": len(rows),
        "finite_fraction": fraction("finite"),
        "solver_success_fraction": fraction("solver_success"),
        "converged_fraction": fraction("converged"),
        "mean_final_min_distance": mean("final_min_distance"),
        "mean_final_goal_error": mean("final_goal_error"),
        "source_files": [
            "tools/atom_headless_smoke.cpp",
            "tools/atom_headless_compat/ros/ros.h",
            "raw_data/external/source_repos/atom_official/src/ilq_solver/src/corridor_3_agent.cpp",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = [
        "# Official AToM headless numerical-core audit",
        "",
        f"Main commit `{metadata['main_git']['commit']}`; clean `{not metadata['main_git']['dirty']}`.",
        f"Official AToM commit `{metadata['atom_git'].get('commit', 'unavailable')}`.",
        "",
        "This run executes the unchanged upstream AToM Corridor3Agent model and "
        "ILQGames/AugmentedLagrangianSolver sources through a local empty ROS "
        "header. It is deliberately reported as a solver-core audit: no ROS "
        "topics, robot endpoint, human observation stream, or TRACER belief "
        "loop is present.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {len(rows)} |",
        f"| Finite final states | {metadata['finite_fraction']:.3f} |",
        f"| Solver success | {metadata['solver_success_fraction']:.3f} |",
        f"| Solver converged | {metadata['converged_fraction']:.3f} |",
        f"| Mean final minimum pair distance | {metadata['mean_final_min_distance']:.4f} |",
        f"| Mean final goal error | {metadata['mean_final_goal_error']:.4f} |",
        "",
        "The finite-state result confirms that the official numerical core can "
        "execute headlessly in this workspace. Any non-converged rows remain "
        "visible in `results.csv` and are not silently counted as successful "
        "navigation episodes.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir", default="experiments/atom_headless_20260903"
    )
    parser.add_argument("--binary", default=".conda/atom_headless_smoke")
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--seeds", nargs="*", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
