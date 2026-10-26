from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def compare_closed_loop(baseline: Path, improved: Path) -> dict:
    def read(directory):
        with (directory / "episode_metrics.csv").open() as handle:
            return {(int(row["seed"]), int(row["case"])): row for row in csv.DictReader(handle)}

    before, after = read(baseline), read(improved)
    if set(before) != set(after):
        raise ValueError("paired runs must contain identical seed/case keys")
    keys = sorted(before)
    cases = sorted({case for _, case in keys})
    metrics = ("success", "collision_free_success", "collision", "wall_collision", "task_progress", "min_distance", "fallback_rate", "executor_projection_events")
    result = {"episodes": len(keys), "cases": len(cases), "seeds": sorted({seed for seed, _ in keys}), "metrics": {}}
    rng = np.random.default_rng(923751)
    resamples = rng.integers(0, len(cases), size=(10000, len(cases)))
    for metric in metrics:
        baseline_values = np.array([float(before[key][metric]) for key in keys])
        improved_values = np.array([float(after[key][metric]) for key in keys])
        delta = improved_values - baseline_values
        case_deltas = np.array([np.mean([delta[index] for index, key in enumerate(keys) if key[1] == case]) for case in cases])
        interval = np.quantile(case_deltas[resamples].mean(axis=1), [0.025, 0.975])
        result["metrics"][metric] = {
            "baseline": float(baseline_values.mean()), "improved": float(improved_values.mean()),
            "delta": float(delta.mean()), "paired_case_cluster_ci95": interval.tolist(),
        }
    with (improved / "window_metrics.csv").open() as handle:
        origins = Counter(row["selected_origin"] for row in csv.DictReader(handle))
    result["selected_origins"] = dict(origins)
    for label, values in (("baseline", before), ("improved", after)):
        result[f"{label}_seconds_per_window"] = sum(float(row["wall_time_seconds"]) for row in values.values()) / sum(int(row["windows"]) for row in values.values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired repair experiments without hiding regressions.")
    parser.add_argument("--root", default="experiments/repair_20260905")
    parser.add_argument("--improved", default="robust_test")
    parser.add_argument("--output", default="reports/closed_loop_repairs_20260905")
    args = parser.parse_args()
    root = Path(args.root)
    comparison = compare_closed_loop(root / "baseline_test", root / args.improved)
    payload = {"closed_loop": comparison, "response": {}, "response_multiseed": {}, "validation_profiles": {}}
    for split, filename in (("validation", "metrics.json"), ("ood", "ood_metrics.json")):
        before = json.loads((root / "response_baseline" / filename).read_text())["persistent"]
        after = json.loads((root / "response_enhanced" / filename).read_text())["persistent"]
        payload["response"][split] = {metric: {"baseline": before[metric], "enhanced": after[metric]} for metric in ("predictive_trajectory_nll", "trajectory_ade", "trajectory_fde", "final_mode_accuracy")}
        seed_rows = [payload["response"][split]]
        seeds = [2027]
        for seed in (2028, 2029):
            before_path = root / f"response_baseline_seed{seed}" / filename
            after_path = root / f"response_enhanced_seed{seed}" / filename
            if before_path.exists() and after_path.exists():
                before = json.loads(before_path.read_text())["persistent"]
                after = json.loads(after_path.read_text())["persistent"]
                seed_rows.append({metric: {"baseline": before[metric], "enhanced": after[metric]} for metric in payload["response"][split]})
                seeds.append(seed)
        payload["response_multiseed"][split] = {
            "seeds": seeds,
            "metrics": {metric: {label: float(np.mean([row[metric][label] for row in seed_rows])) for label in ("baseline", "enhanced")} for metric in payload["response"][split]},
            "per_seed": seed_rows,
        }
    for directory in sorted(root.glob("*_validation")):
        summary = directory / "summary.json"
        if summary.exists():
            payload["validation_profiles"][directory.name] = json.loads(summary.read_text())["policies"]["tracer-persistent"]
    lines = [
        "# 闭环修复与配对实验（2026-09-05）", "",
        "## 结论", "",
        "安全指标显著改善，但任务完成率和进度下降；不能声称全部问题解决、全面提升或已经超过 SICNav。",
        "结果属于开发验证：保留未提交源代码快照，不覆盖原论文的冻结实验。", "",
        "## 闭环配对结果", "",
        f"{comparison['episodes']} episodes；{comparison['cases']} cases；seeds={comparison['seeds']}。相同外部环境、FRM、SAGE checkpoint、dt=0.25、60-step budget。",
        "参数只在 cases 40–47 选择；cases 48–63 用于本轮冻结评估，但这些场景在历史实验中已经被使用，不称为全新盲测。",
        "改进方案为 SAGE 加连续恢复候选、多人风险聚合、多样性筛选和 swept guard，不是纯 SAGE 相同候选池消融。", "",
        "|指标|修改前|修改后|配对差值（后−前）|按场景聚类 bootstrap 95% CI|",
        "|---|---:|---:|---:|---|",
    ]
    for metric, values in comparison["metrics"].items():
        lower, upper = values["paired_case_cluster_ci95"]
        lines.append(f"|{metric}|{values['baseline']:.4f}|{values['improved']:.4f}|{values['delta']:+.4f}|[{lower:+.4f}, {upper:+.4f}]|")
    lines.extend([
        "", f"选中候选来源：{comparison['selected_origins']}。收益不能归因于 belief；外部 domain FRM 的两个 mode 槽是相同分布的中性复制。",
        f"实测总运行时间/执行窗口：修改前 {comparison['baseline_seconds_per_window']:.4f}s，修改后 {comparison['improved_seconds_per_window']:.4f}s；包含环境开销且测试存在并行负载，不作为严格 latency benchmark。",
        "", "## 响应预测 pilot", "",
        "相同 512 个训练 episode、128 个验证/128 个 OOD episode、8 个 epoch、seed 2027；仅 pilot，未替换已冻结生产 checkpoint。",
        "|split|metric|baseline|enhanced|", "|---|---|---:|---:|",
    ])
    for split, metrics in payload["response"].items():
        for metric, values in metrics.items():
            lines.append(f"|{split}|{metric}|{values['baseline']:.5f}|{values['enhanced']:.5f}|")
    lines.extend(["", "### 补充多 seed pilot", "", "|split|seeds|metric|baseline mean|enhanced mean|", "|---|---|---|---:|---:|"])
    for split, block in payload["response_multiseed"].items():
        for metric, values in block["metrics"].items():
            lines.append(f"|{split}|{block['seeds']}|{metric}|{values['baseline']:.5f}|{values['enhanced']:.5f}|")
    belief_path = root / "belief" / "records.json"
    if belief_path.exists():
        belief = json.loads(belief_path.read_text())["records"]
        payload["belief_repair"] = {}
        lines.extend(["", "## Belief 温度修复审计", "", "独立 toy response filter、5 seeds、ID/OOD，scale=0.05；不是完整导航收益。", "|condition|method|final accuracy|mode NLL|mean hazard|", "|---|---|---:|---:|---:|"])
        for condition in ("stationary", "switch"):
            for method in ("persistent", "legacy-scaled-surprise", "raw-surprise"):
                rows = [row for row in belief if row["condition"] == condition and row["method"] == method]
                values = {name: float(np.mean([row[name] for row in rows])) for name in ("final_accuracy", "final_mode_nll", "mean_hazard")}
                payload["belief_repair"][f"{condition}:{method}"] = values
                lines.append(f"|{condition}|{method}|{values['final_accuracy']:.5f}|{values['final_mode_nll']:.5f}|{values['mean_hazard']:.6f}|")
    lines.extend([
        "", "## 已实现", "",
        "- raw likelihood 用于 surprise，tempered likelihood 只用于 posterior 更新；外部 change-point 同步修复。",
        "- CrowdNav fallback 原先最小化距离（选最危险分支），已改为在官方可执行支撑中最大化间距；修正执行 horizon 被 XY 维度截断的问题。",
        "- 修正 evidence GRU 隐藏维误索引及 padding 污染；attention evidence 只读执行前缀。",
        "- diagonal Gaussian 风险采样使用独立时间维 Sobol 噪声，保留 shared legacy 复现选项。",
        "- 多行人风险提供 max/union-bound 聚合，避免危险行人被安全行人平均稀释。",
        "- 加入 swept body check、可行恢复候选、多样性筛选和可观察状态风险项。",
        "- 执行器投影动作后重新预测用于 belief 更新；guard 必须覆盖完整执行前缀。",
        "- 响应模型可选空间几何、时序 attention、candidate-conditioned uncertainty、三机器人以上 set residual；旧 checkpoint 默认结构不变。",
        "- 修正 ranking dataset 的 entity/time 维切片；增加真实 prefix evidence 与可选 on-policy rollout，已通过训练冒烟测试。",
        "- 共用 stationary reference 每行人只计算一次；风险噪声缓存；评分和 realized ranking audit 的已知代价项一致。",
        "", "## 未解决及限制", "",
        "- 外部高密度拥挤环境仍有超时和停滞；安全收益伴随进度退化。",
        "- 长时间外部 on-policy 重训、显式静态障碍物输入 SAGE、真实身份关联、全时序协方差尚未实现完整验证。",
        "- 新模型仍为小规模 pilot；ranking/on-policy 只有 smoke，不声称其闭环性能提升。",
        "- adaptive raw surprise 恢复触发，但阈值和 likelihood scale 仍需验证集校准；不能将触发率当作真实变化检测准确率。",
        "- goal-return/terminal-goal 候选验证未通过，默认关闭，不用它的结果挑选测试案例。",
        "- 尚未用本轮最终配置重跑官方 SICNav；不声称优于官方控制器。",
        "", "## 复现", "",
        "```bash", "bash scripts/run_repaired_sicnav.sh --allow-dirty", 
        "python -m sage.experiments.eval_repaired_belief",
        "python -m sage.experiments.summarize_repair_results", "```", "",
        f"主结果：experiments/repair_20260905/baseline_test 和 {args.improved}。当前集成检查：current_integration。所有被否决的 validation profile 同样保留。",
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
