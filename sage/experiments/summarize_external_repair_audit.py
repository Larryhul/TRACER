from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from sage.experiments.compare_navigation_validation import compare


DEVELOPMENT_RUNS = (
    ("tracer_feedback_clearance05", "旧 FRM＋可执行路线候选"),
    ("tracer_incumbent", "旧 FRM＋上一计划候选"),
    ("tracer_rolling_frm", "rolling FRM"),
    ("tracer_delay", "rolling FRM＋首步延迟"),
    ("tracer_delay_effect", "延迟＋响应差分损失"),
    ("tracer_real_belief_cadence", "真实三模式＋校准更新频率"),
    ("tracer_native_task_scale", "增强任务进度权重"),
    ("tracer_terminal_goal_v2", "旧 FRM＋到达终止边界"),
)


def summary_rows(directory, label):
    payload = json.loads((directory / "summary.json").read_text())
    return [{"variant": label, "directory": str(directory), "policy": policy,
             **{metric: values[metric]["mean"] for metric in (
                 "success", "collision_free_success", "collision", "total_time", "final_belief_entropy")}}
            for policy, values in payload["policies"].items()]


def main():
    parser = argparse.ArgumentParser(description="Summarize external repair experiments without selecting a final test winner.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="reports/h2_h3_external_repair_audit_20260905")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    development = root / "experiments/sicnav_state_contract_validation_20260905"
    extension = root / "experiments/sicnav_state_contract_extension_20260905"
    belief_root = root / "experiments/rolling_external_belief_validation_20260905/seed2027"
    records, sources = [], []
    for name, label in DEVELOPMENT_RUNS:
        directory = development / name
        if not (directory / "summary.json").exists():
            continue
        records.extend(summary_rows(directory, label))
        sources.extend([directory / "summary.json", directory / "metadata.json"])
    comparisons = {}
    executor_overrides = {}
    for name in ("tracer_frozen_route", "tracer_terminal_goal"):
        directory = extension / name
        if (directory / "summary.json").exists():
            comparisons[name] = compare(extension / "official_native", directory)
            sources.extend([directory / "episode_metrics.csv", directory / "metadata.json"])
            with (directory / "window_metrics.csv").open() as handle:
                executor_overrides[name] = [
                    {key: row[key] for key in ("seed", "case", "window", "fallback_tier", "executor_projection", "wall_collision")}
                    for row in csv.DictReader(handle) if int(row["executor_projection"])
                ]
            sources.append(directory / "window_metrics.csv")
    calibration = json.loads((belief_root / "calibration.json").read_text())
    mixture = json.loads((belief_root / "response_mixture_audit.json").read_text())
    window_path = extension / "tracer_frozen_route/window_metrics.csv"
    with window_path.open() as handle:
        windows = list(csv.DictReader(handle))
    human_cost = np.asarray([float(row["selected_predicted_human"]) for row in windows])
    diagnostics = {
        "windows": len(windows),
        "selected_risk_saturation_fraction": float(np.mean([float(row["selected_predicted_risk"]) >= .999 for row in windows])),
        "selected_human_cost_median": float(np.median(human_cost)),
        "selected_human_cost_max": float(human_cost.max()),
        "boundary": "selected-candidate diagnostics only; not a complete candidate ranking or calibration audit",
    }
    sources.extend([window_path, belief_root / "calibration.json", belief_root / "response_mixture_audit.json",
                    extension / "official_native/episode_metrics.csv", extension / "official_native/metadata.json",
                    development / "official_native/episode_metrics.csv", development / "official_native/metadata.json"])
    result = {"date": "2026-09-05", "phase": "development_validation", "noninferiority_established": False,
              "development": records, "extension": comparisons, "executor_overrides": executor_overrides,
              "external_belief_calibration": calibration,
              "external_belief_response_audit": mixture, "selected_candidate_diagnostics": diagnostics,
              "sources_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    lines = [
        "# H2/H3 外部闭环修复审计 — 2026-09-05", "",
        "## 结论与边界", "",
        "当前不能宣称 TRACER 已达到 SICNav 的非劣性能。正确性修复有效，但预测误差下降没有稳定转化为导航提升。",
        "本报告更新早期 state_consistent_repair_validation 报告的外部部分；所有数据都是开发验证，不是最终盲测。",
        "case 80–87 是反复使用的开发集；88–111 是扩展开发集。FRM 训练使用 112–135，校准使用 136–143。",
        "不得把同一 case 的不同模型种子当作独立场景，也不得混合 persistent/reset/uniform 后计算一个方法均值。", "",
        "## 已修正的正确性问题", "",
        "- 真闭环在各策略自己的实际状态上重生成候选；同状态诊断不再冒充独立闭环。",
        "- belief 只接收真实执行前缀的观测；评分与执行坐标、积分和时间步接受契约检查。",
        "- 候选和发出的原始控制采用 SICNav 原生单车运动限制、初始状态、dt 和执行预算；没有改动官方基线源码。执行器紧急投影仍单独审计，不宣称全部实际执行零干预。",
        "- 到达、碰撞、碰撞后到达、超时分别计数；主导航指标是 collision-free success，而不是只看 success。",
        "- 旧外部适配器仍只有一个响应分布，复制为两槽不代表真实 belief。新增三模式后端是独立实验，不静默替换旧模型。",
        "- 新外部模型第一步响应不依赖尚未执行的新动作；belief 校准和运行都每六个执行步接收一次单步事实证据。",
        "- 新终止边界可选项在到达目标的那一步仍计风险，之后不再对不存在的任务后缀惩罚；旧默认保持不变。", "",
        "## H2：真实外部 belief 仍很弱", "",
        "新模型用训练集身份级首选速度分位数定义三个稳定模式。模拟器隐藏速度只用于训练标签和验证，不输入运行时规划。",
        "这是行为参数代理，不是现实人类意图真值；它也不等价于论文中所有合成行为模式。",
        f"24 条验证行人轨迹、{calibration['unique_observations']} 条去重事实观测；温度 {calibration['selected']['likelihood_scale']:.2f} 仅在验证集选择。",
        f"观测前模式 NLL={calibration['selected']['decision_mode_nll']:.5f}，uniform={calibration['uniform_mode_nll']:.5f}；模式准确率={calibration['selected']['decision_mode_accuracy']:.2%}。",
        "不能把给定真实模式的 conditional ADE/NLL 当作在线 belief 的指标；后者必须使用观测前推断的混合权重。", "",
        "| 观测前权重 | 混合响应 NLL↓ | 混合响应 ADE↓ |",
        "|---|---:|---:|",
    ]
    for policy, metrics in mixture["metrics"].items():
        lines.append(f"| {policy} | {metrics['mixture_nll']:.6f} | {metrics['mixture_ade']:.6f} |")
    lines.extend([
        "", "Persistent 没有赢得混合响应 NLL，ADE 差值很小。加模式头解决结构缺口，不代表解决了模式可辨识性。",
        "合成 H2 中 H1 encoder 迁移改善预测及模式识别的结果仍有效，但不能据此外推外部 belief 有收益。", "",
        "## H3：8 个开发场景", "",
        "以下均为 seed 2027、原生单车运动约束、相同场景和 60 步预算；旧 FRM 路线对照使用相同候选支持。",
        "SICNav-NP 在这 8 个场景的无碰撞成功率为 100%；这不是最终统计结论。", "",
        "| 变体 | belief 策略 | 无碰撞成功率 | 到达率 | 碰撞率 | 平均仿真时间 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in records:
        lines.append(f"| {row['variant']} | {row['policy']} | {row['collision_free_success']:.1%} | {row['success']:.1%} | {row['collision']:.1%} | {row['total_time']:.3f} |")
    lines.extend(["", "## H3：24 个扩展场景", "",
                  "每个场景先平均三个模型种子，再进行配对场景 bootstrap。置信区间仅是描述性验证，不构成非劣检验。", "",
                  "| 变体 | TRACER 无碰撞成功率 | SICNav 无碰撞成功率 | 差值 | 差值 95% CI | 仿真时间差 |",
                  "|---|---:|---:|---:|---|---:|"])
    for name, comparison in comparisons.items():
        metrics = comparison["metrics"]
        cfs = metrics["collision_free_success"]
        lower, upper = cfs["descriptive_case_bootstrap_ci95"]
        lines.append(f"| {name} | {cfs['tracer']:.2%} | {cfs['official']:.2%} | {cfs['tracer_minus_official']:+.2%} | [{lower:+.2%}, {upper:+.2%}] | {metrics['total_time']['tracer_minus_official']:+.3f} |")
    lines.extend(["", "执行器投影审计："])
    for name, overrides in executor_overrides.items():
        lines.append(f"- {name}：{len(overrides)} 个投影窗口。")
        for override in overrides:
            lines.append(f"  seed {override['seed']} / case {override['case']} / window {override['window']}：fallback={override['fallback_tier']}，wall_collision={override['wall_collision']}。")
    lines.append("终止边界版本虽修复一个到达失败，但仍出现墙边无可执行候选后的投影。初始运动协议一致不等价于每一步实际执行都未被环境改写；这一缺口尚未修复。")
    lines.extend([
        "", "## 为什么闭环还上不去", "",
        "1. **H1→H2→H3 模型不是同一个自动升级链。** Encoder 迁移没有复制响应解码器和差分训练目标；旧外部模型更不是新 H1 的完整下游。",
        "2. **模式预测差异不足。** 真三模式在当前外部验证中的可辨识性接近均匀，推断出的 belief 难以稳定改善候选排序。",
        f"3. **风险代价容易饱和。** 旧 FRM 扩展运行的 {diagnostics['windows']} 个已选窗口中，{diagnostics['selected_risk_saturation_fraction']:.1%} 的风险分数达到 1。",
        f"4. **响应干预项接近失效。** 同一运行已选候选的人类响应代价中位数仅 {diagnostics['selected_human_cost_median']:.3g}，最大值 {diagnostics['selected_human_cost_max']:.3g}。这是线索，不能代替全候选差分/排序审计。",
        "5. **终端行为和死锁仍存在。** 已见门口停滞、侧向贴墙、到目标附近无法收敛、以及超过目标后返回不及时；候选包含一条路径并不保证闭环持续跟随。",
        "6. **训练分布并非 TRACER 实际访问分布。** rolling 数据由混合探索控制器采集，不能称为当前 TRACER on-policy 数据；困难死锁状态可能覆盖不足。",
        "7. **离线校准与决策指标不同。** 更好的 ADE/NLL、修正因果首步、增强任务权重，都已经出现闭环下降，不能继续盲调一个权重就期待解决。", "",
        "新增事实前缀诊断记录每个窗口执行后的混合 NLL、速度 MSE、模式预测差异和动作条件响应差分；这些值使用更新前 belief，不泄漏本窗口观测。",
        "tracer_factual_prefix_diagnostic 是 case 80 的单场景集成检查，不用于成功率选型或正式机制优势结论。", "",
        "## 下一步优先级", "",
        "- P0：冻结并保持状态/执行/时间/证据契约；标注旧共享池闭环结论失效、外部单模式适配器的边界。",
        "- P0-H3：补充制动可达安全检查和显式制动候选，避免前一步进入下一步全部候选均不可执行的墙边状态；不能靠违反减速度约束的瞬间停车伪装修复。",
        "- P1-H2：在独立训练场景收集当前控制器实际经过的状态，验证模式条件响应差分、事实前缀 likelihood、混合校准；不把真实模式标签送给规划器。",
        "- P1-H2：稳定身份模式与真实模式切换分开验证；固定 Persistent 在模式变化后可能锁死，需要独立校准的 hazard/forgetting 或变化检测，不能拿稳定速度代理证明切换适应能力。",
        "- P1-H3：在相同状态和相同可执行候选下记录 uniform/reset/persistent 的排序变化、执行前缀误差及近身风险校准，定位被错误淘汰的可行路径。",
        "- P1-H3：将到达终止、可执行转向/制动、递推计划一致性作为控制器问题处理，不将候选工程的收益归因于 belief。",
        "- P2：保留正式合成多机器人主实验，外部单机器人只是迁移/系统验证；不以单机器人成功率代替多机器人联合效应。",
        "- P2：冻结选模规则、超参数和有科学依据的非劣界限，再用未使用场景比较 SICNav；现在不能承诺必定达到非劣。", "",
        "## 失效或重复诊断", "",
        "- rolling_external_frm_validation_20260905 的第一版历史末帧滞后一拍，已由 aligned 版本替代，保留原始数据但不作为训练证据。",
        "- tracer_real_belief 第一版每步更新、每六步校准，不作为正确时序结果；使用 tracer_real_belief_cadence。",
        "- tracer_prefix_guard 与此前对照的有效 prefix 设置相同，只是重复验证，不是新的消融。",
        "- tracer_terminal_goal 第一版因生成器/响应 horizon 长度检查失败，没有完整结果；使用 tracer_terminal_goal_v2。",
        "- 所有修改都在 TRACER/评估适配层；原始结果、官方源码和其他并行 H1 改动不覆盖。", "",
    ])
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text("\n".join(lines))
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
