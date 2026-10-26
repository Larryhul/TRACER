"""运行 H1 外部动作条件响应基线统一协议（中文报告）。

该入口只复用 ``train_response_baselines``、``eval_h1_unified_baselines`` 和
``EntityResponseBatch``，不创建第二套数据预处理或测试脚本。Eiffert 与
ScePT 均按方法级实现/代理标注，不声称官方复现。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from sage.experiments.eval_h1_unified_baselines import run as evaluate_h1
from sage.response.external_baselines import METHOD_TAXONOMY


def _git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    return {"commit": commit, "dirty": dirty}


def _write_chinese_report(output: Path, protocol: dict[str, Any]) -> None:
    lines = [
        "# H1 外部动作条件响应基线",
        "",
        "本报告采用统一的 paired intervention 协议、相同的候选机器人轨迹、相同的初始状态、相同的 train/validation/test split 和相同的预测目标。模型选择仅使用 validation NLL，test 场景未用于调参或选择。",
        "",
        "方法分类：",
        "- `history-only`：动作无关下界；",
        "- `nearest-robot`：单机器人局部性基线；",
        "- `additive` / `matched-additive`：内部主效应及容量匹配控制；",
        "- `eiffert2019-style`：Eiffert & Sukkarieh (2019) 方法级重实现，不是官方代码复现；",
        "- `ScePT-style`：现有 `team-concat` 的场景上下文代理，不是官方 ScePT 复现；",
        "- `joint`：TRACER 响应模型；",
        "- `shuffled-joint`：打乱 action-response 配对的负控制。",
        "",
        "ScePT 原始模型面向场景一致的条件联合轨迹预测，并使用其原始数据与训练目标；本项目 proxy 仅在统一 `EntityResponseBatch` 上拼接固定 padding 的所有机器人未来轨迹，并用 active mask，再以 Gaussian 响应目标训练。因此这里的比较是 method-level proxy 比较。",
        "",
        "报告中的正面表述边界：TRACER 与先前动作条件或场景上下文响应预测器的 method-level reimplementation/proxy 在共享 paired-intervention 协议下进行了比较；不声称官方复现或普遍优越。",
        "",
        f"协议状态：`{protocol.get('status', 'completed')}`；Formal synthetic H1 的输出见 `summary.csv`、`records.csv` 和 `protocol.json`。",
    ]
    (output / "summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    config_path = (root / args.config).resolve()
    output = (root / args.output_dir).resolve()
    checkpoint_dir = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    variants = tuple(config["training"]["variants"])
    missing = [name for name in variants if not (checkpoint_dir / f"{name}.pt").exists()]
    if missing and not args.skip_training:
        command = [
            sys.executable, "-m", "sage.experiments.train_response_baselines",
            "--config", str(config_path), "--output-dir", str(checkpoint_dir),
            "--device", args.device, "--variants", *variants,
        ]
        subprocess.run(command, cwd=root, check=True)
    missing = [name for name in variants if not (checkpoint_dir / f"{name}.pt").exists()]
    if missing:
        raise FileNotFoundError(f"缺少 checkpoint，无法进行统一评估: {missing}")
    eval_args = argparse.Namespace(
        root=str(root), checkpoint_dir=str(checkpoint_dir), output_dir=str(output),
        train_family="pincer", target_families=list(args.target_families),
        target_data_seed_start=args.target_data_seed_start, model_seed=args.model_seed,
        test_chunk=args.test_chunk, device=args.device,
    )
    result = evaluate_h1(eval_args)
    protocol = dict(result["protocol"])
    protocol.update({
        "protocol_name": "h1-external-response-baselines-v1",
        "datasets": ["Formal synthetic H1"],
        "method_taxonomy": METHOD_TAXONOMY,
        "claim_boundary_zh": "方法级重实现/代理比较，不是官方复现；不声称普遍优越。",
        "git": _git_state(root),
    })
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "protocol": protocol,
        "models": {
        name: {
            "implementation": METHOD_TAXONOMY.get(name, {}).get("implementation", name),
            "type": METHOD_TAXONOMY.get(name, {}).get("type", "internal baseline"),
            "checkpoint": str(checkpoint_dir / f"{name}.pt"),
        }
        for name in variants
        },
        "normalization": "shared EntityResponseBatch preprocessing",
        "checkpoint_selection": "validation-only; no test-driven selection",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    _write_chinese_report(output, protocol)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/h1_external_response_baselines.yaml")
    parser.add_argument("--output-dir", default="experiments/h1_external_response_baselines_20260905")
    parser.add_argument("--target-families", nargs="+", default=["pincer"])
    parser.add_argument("--target-data-seed-start", type=int, default=910000)
    parser.add_argument("--model-seed", type=int, default=2027)
    parser.add_argument("--test-chunk", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-training", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
