"""Formal H1 外部响应基线的多 seed 配对评估入口。

每个 model seed 使用同一份冻结 Formal test 配置和四分支干预协议，先在
seed 内计算配对 contrast，再汇总 scene-level records。这样不会把不同 seed
中重复的 base_scene_id 错当成同一个统计单位。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from sage.experiments.eval_h1_unified_baselines import (
    KEDIA_METHOD_NAME,
    KEDIA_MODEL_VARIANT,
    METHODS,
    _contrasts,
    _evaluate_target_family,
    _load_models,
    _summary,
    _write_csv,
)
from sage.response.causal_synthetic import make_causal_response_dataset_from_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_markdown(path: Path, protocol: dict[str, Any], summary: list[dict[str, Any]], contrasts: list[dict[str, Any]]) -> None:
    kedia_note = (
        "- `Kedia-style` 是 latent-hypothesis game-theoretic method-level 适配，不是官方 Kedia 复现。"
        if KEDIA_METHOD_NAME in protocol["methods"]
        else "- Kedia-style 未纳入本次评估。"
    )
    lines = [
        "# Formal H1 外部响应基线多 seed 评估",
        "",
        "本报告使用统一的 EntityResponseBatch 接口和四分支配对干预，在 Formal causal response test split 上评估五个冻结 model seed。训练/验证/测试场景、数据 seed、方法集合和 checkpoint 选择规则在评估前固定；测试结果不参与模型或 seed 选择。",
        "",
        f"- model seeds: `{', '.join(str(v) for v in protocol['model_seeds'])}`；test base scenes/seed: `{protocol['num_test_scenes']}`；R=`{protocol['num_robots']}`。",
        f"- 数据 seed: `{protocol['data_seed']}`；每个 scene 的 paired interventions: `{protocol['interventions_per_scene']}`。",
        "- `ScePT-style` 是当前 `team-concat` 的场景上下文 proxy，并非官方 ScePT 复现；`eiffert2019-style` 是方法级重实现。",
        kedia_note,
        "",
        "| Test family | Method | N | NLL | ADE | FDE | Action-effect error | Pair RMSE | Sign Brier | Sign ECE | 1-sigma | 2-sigma |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['target_family']} | {row['method']} | {row['n_scenes']} | "
            f"{row['nll_mean']:.4f} +/- {row['nll_sd']:.4f} | {row['ade_mean']:.4f} | {row['fde_mean']:.4f} | "
            f"{row['action_effect_error_mean']:.4f} | {row['pair_rmse_mean']:.4f} | "
            f"{row['velocity_sign_brier_mean']:.4f} | {row['velocity_sign_ece_mean']:.4f} | "
            f"{row['coverage_1sigma_mean']:.4f} | {row['coverage_2sigma_mean']:.4f} |"
        )
    lines.extend([
        "",
        "## Seed 内 paired contrasts",
        "",
        "contrast 先在每个 model seed 的 base scene 内计算，再写入 `contrasts.csv`；因此不同 seed 的重复 scene id 不会被合并。",
        "",
        "| Model seed | Test family | Contrast | Mean | 95% CI | N |",
        "|---:|---|---|---:|---|---:|",
    ])
    for row in contrasts:
        lines.append(
            f"| {row['model_seed']} | {row['target_family']} | {row['contrast']} | {row['mean']:.5f} | "
            f"[{row['ci95_low']:.5f}, {row['ci95_high']:.5f}] | {row['n_scenes']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    formal_dir = root / args.formal_dir
    seed_dirs = {seed: formal_dir / "checkpoints" / f"seed{seed}" for seed in args.model_seeds}
    config = _load_yaml(seed_dirs[args.model_seeds[0]] / "config.yaml")
    causal = config.get("dataset", {}).get("causal_response", {})
    data_seed = int(args.data_seed)
    target_family = str(causal.get("response_family", "pincer"))
    include_kedia = bool(args.include_kedia)
    checkpoint_variants = (
        "history",
        "nearest",
        "additive",
        "matched-additive",
        "joint",
        "eiffert2019-style",
        "team-concat",
    ) + ((KEDIA_MODEL_VARIANT,) if include_kedia else ())
    methods = METHODS + ((KEDIA_METHOD_NAME,) if include_kedia else ())
    all_rows: list[dict[str, Any]] = []
    all_contrasts: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, dict[str, str]] = {}
    model_metadata: dict[str, Any] = {}
    for model_seed in args.model_seeds:
        checkpoint_dir = seed_dirs[model_seed]
        models = _load_models(checkpoint_dir, args.device, include_kedia=include_kedia)
        checkpoint_hashes[str(model_seed)] = {
            variant: _sha256(checkpoint_dir / f"{variant}.pt")
            for variant in checkpoint_variants
        }
        if not model_metadata:
            from sage.response.external_baselines import external_method_metadata

            model_metadata = {name: external_method_metadata(model, name) for name, model in models.items()}
        raw = dict(config)
        raw.setdefault("dataset", {}).setdefault("causal_response", {})["response_family"] = target_family
        raw["dataset"]["causal_response"]["seed"] = data_seed
        dataset = make_causal_response_dataset_from_config(
            raw,
            split="test",
            effect_mode="nonlinear",
            seed=data_seed,
            cache_samples=True,
        )
        rows = _evaluate_target_family(
            dataset,
            models,
            args.device,
            target_family,
            target_family,
            data_seed,
            args.test_chunk,
            methods=methods,
        )
        for row in rows:
            row["model_seed"] = model_seed
        all_rows.extend(rows)
        seed_contrasts = _contrasts(rows, bootstrap_resamples=args.bootstrap_resamples)
        for row in seed_contrasts:
            row["model_seed"] = model_seed
        all_contrasts.extend(seed_contrasts)
        print(f"evaluated model_seed={model_seed} scenes={dataset.num_base_scenes} rows={len(rows)}", flush=True)

    summary = _summary(all_rows)
    protocol = {
        "status": "completed",
        "formal_dir": args.formal_dir,
        "model_seeds": [int(seed) for seed in args.model_seeds],
        "train_family": target_family,
        "target_families": [target_family],
        "data_seed": data_seed,
        "num_test_scenes": int(causal.get("test_base_scenes", 256)),
        "num_robots": 2,
        "interventions_per_scene": 4,
        "methods": list(methods),
        "metrics": [
            "nll",
            "ade",
            "fde",
            "action_effect_error",
            "pair_rmse",
            "brier",
            "ece",
            "coverage_1sigma",
            "coverage_2sigma",
        ],
        "shared_contract": {
            "input": "EntityResponseBatch: entity_history, robot_current, candidate_robot_trajectory, fixed padded robot_mask",
            "output": "Gaussian trajectory mean and log_std",
            "paired_unit": "base_scene with four interventions evaluated together",
        },
        "selection": "validation-only best_val_nll; all model seeds and test protocol frozen before evaluation",
        "checkpoint_hashes": checkpoint_hashes,
        "model_metadata": model_metadata,
        "checkpoint_rule": str(config.get("training", {}).get("checkpoint_rule", "best_val_nll")),
        "include_kedia": include_kedia,
    }
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "records.csv", all_rows)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "contrasts.csv", all_contrasts)
    _write_markdown(output_dir / "summary.md", protocol, summary, all_contrasts)
    return {"protocol": protocol, "summary": summary, "contrasts": all_contrasts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--formal-dir", default="experiments/h1_external_response_baselines_20260905/formal")
    parser.add_argument("--output-dir", default="experiments/h1_external_response_baselines_20260905/formal/evaluation")
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--data-seed", type=int, default=318731)
    parser.add_argument("--test-chunk", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--include-kedia",
        action="store_true",
        help="Evaluate the method-level Kedia-style latent-hypothesis adapter.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
