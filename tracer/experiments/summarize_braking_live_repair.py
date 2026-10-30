from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from tracer.experiments.compare_navigation_validation import compare


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def navigation_rows(path, label):
    summary = read_json(path / "summary.json")
    metadata = read_json(path / "metadata.json")
    with (path / "window_metrics.csv").open() as handle:
        windows = list(csv.DictReader(handle))
    return [{"label": label, "policy": policy, "path": str(path),
             "response_backend": metadata["response_contract"]["backend"],
             "metrics": {name: metrics[name]["mean"] for name in (
                 "collision_free_success", "success", "collision", "wall_collision", "total_time", "final_belief_entropy")},
             "executor_projection_windows": sum(int(row["executor_projection"]) for row in windows if row["policy"] == policy)}
            for policy, metrics in summary["policies"].items()]


def main():
    parser = argparse.ArgumentParser(description="Summarize braking safety, actual-controller response training, and H2 transfer evidence.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="reports/braking_live_response_repair_20260905")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    experiments = root / "experiments"
    runs = experiments / "sicnav_braking_validation_20260905"
    development = []
    sources = []
    for name, label in (("development", "旧 FRM＋制动安全"), ("live_frm_control", "实际控制器状态训练 FRM"),
                        ("margin02_dev", "实际控制器状态训练 FRM＋0.02 m 裕度"),
                        ("live_belief_control", "三模式：仅 encoder 迁移"), ("live_static_belief", "三模式：encoder＋地图"),
                        ("live_full_transfer_belief", "三模式：完整预测头迁移")):
        path = runs / name
        if (path / "summary.json").exists():
            development.extend(navigation_rows(path, label))
            sources.extend([path / filename for filename in ("summary.json", "metadata.json", "window_metrics.csv")])
    extension = {}
    for name in ("extension", "live_frm_extension", "corrected_heading_extension", "margin02_extension",
                 "live_static_belief_extension", "orca_augmented_extension_v2", "belief_discriminative_extension",
                 "belief_discriminative_margin02_extension"):
        path = runs / name
        if (path / "summary.json").exists():
            extension[name] = compare(experiments / "sicnav_state_contract_extension_20260905/official_native", path)
            sources.extend([path / filename for filename in ("episode_metrics.csv", "metadata.json", "window_metrics.csv")])
            with (path / "window_metrics.csv").open() as handle:
                windows = list(csv.DictReader(handle))
            extension[name]["executor_projection_windows"] = sum(int(row["executor_projection"]) for row in windows)
    h2 = {}
    for name in ("tracer_live_belief_control_20260905", "tracer_live_static_belief_20260905", "tracer_live_full_transfer_belief_20260905", "tracer_live_belief_discriminative_20260905"):
        path = experiments / name / "seed2027"
        if (path / "calibration.json").exists():
            h2[name] = {"calibration": read_json(path / "calibration.json"),
                        "mixture": read_json(path / "response_mixture_audit.json")}
            sources.extend([path / "calibration.json", path / "response_mixture_audit.json"])
    audit_path = experiments / "tracer_live_response_data_20260905"
    response_audits = [read_json(audit_path / name) for name in ("existing_model_audit.json", "live_model_audit.json")]
    sources.extend([audit_path / name for name in ("existing_model_audit.json", "live_model_audit.json", "data_manifest.json")])
    data_manifest = read_json(audit_path / "data_manifest.json")
    belief_selection = None
    selection_path = runs / "live_static_belief_selection_audit/window_metrics.csv"
    if selection_path.exists():
        with selection_path.open() as handle:
            selections = list(csv.DictReader(handle))
        belief_selection = {"windows": len(selections),
                           "candidate_changes": sum(int(row["same_state_belief_selection_changed"]) for row in selections),
                           "executed_action_changes": sum(int(row["same_state_belief_action_changed"]) for row in selections),
                           "scope": "seed2027/cases80-87; uniform rescores identical model predictions and identical feasible candidate set"}
        sources.append(selection_path)
    h2_navigation_audit = None
    h2_navigation_path = runs / "belief_discriminative_margin02_policy_audit"
    if (h2_navigation_path / "episode_metrics.csv").exists() and (h2_navigation_path / "window_metrics.csv").exists():
        with (h2_navigation_path / "episode_metrics.csv").open() as handle:
            navigation = list(csv.DictReader(handle))
        with (h2_navigation_path / "window_metrics.csv").open() as handle:
            navigation_windows = list(csv.DictReader(handle))
        h2_navigation_audit = {
            "path": str(h2_navigation_path),
            "policies": {
                policy: {
                    "collision_free_success": sum(float(row["collision_free_success"]) for row in navigation if row["policy"] == policy) / sum(row["policy"] == policy for row in navigation),
                    "success": sum(float(row["success"]) for row in navigation if row["policy"] == policy) / sum(row["policy"] == policy for row in navigation),
                    "collision": sum(float(row["collision"]) for row in navigation if row["policy"] == policy) / sum(row["policy"] == policy for row in navigation),
                    "task_progress": sum(float(row["task_progress"]) for row in navigation if row["policy"] == policy) / sum(row["policy"] == policy for row in navigation),
                }
                for policy in ("tracer-persistent", "tracer-reset", "tracer-uniform")
            },
            "same_state_selection_changes": sum(int(row.get("same_state_belief_selection_changed", 0)) for row in navigation_windows if row["policy"] == "tracer-persistent"),
            "same_state_action_changes": sum(int(row.get("same_state_belief_action_changed", 0)) for row in navigation_windows if row["policy"] == "tracer-persistent"),
            "windows": sum(row["policy"] == "tracer-persistent" for row in navigation_windows),
            "scope": "unseen cases88-95; three discriminative persistent checkpoints; same candidate set and state for uniform rescore",
        }
        sources.extend([h2_navigation_path / name for name in ("episode_metrics.csv", "window_metrics.csv", "metadata.json")])
    sources.extend([experiments / "sicnav_state_contract_extension_20260905/official_native" / name
                    for name in ("episode_metrics.csv", "metadata.json")])
    payload = {"date": "2026-09-05", "phase": "correctness_repair_validation", "noninferiority_established": False,
               "development": development, "extension": extension, "h2": h2,
               "h2_navigation_policy_audit": h2_navigation_audit,
               "same_state_belief_selection_audit": belief_selection,
               "live_data": data_manifest, "response_audits": response_audits,
               "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    lines = ["# 制动安全与实际闭环响应修复 — 2026-09-05", "",
             "## 状态与边界", "",
             "这是正确性修复后的验证记录；扩展集仍是描述性比较，不宣称正式非劣。",
             "单分布 FRM 后端只能作为 H1/H3 控制对照；真实三模式后端保留 persistent/reset/uniform 对照。",
             "本轮不改官方 SICNav 源码、不改变 60 步预算、不降低原生单车速度/角速度/加减速约束、不覆盖历史结果。", "",
             "## 本轮修正", "",
             "- 新增一个显式减速停车候选，并要求每个候选执行后仍存在静态无碰撞的直线制动路径。候选上限由 35 增至 36；这是候选＋安全检查的组合修复。",
             "- 静态制动条件不会被 fallback 绕过。若可行集合为空或环境再次改写制动安全动作，运行直接报错，不能伪装成满足运动约束。",
             "- 机器人实际执行速度现在在 env.step 之后读取，并与位置积分核对；此前环境内部第二次约束可能使事前估计动作不同于实际动作。",
             "- 已知 seed2027/case107 由墙碰撞＋投影变为零墙碰撞、零投影，但仍超时；不是以安全检查已经解决到达能力。",
             "- 新增后选择的分支响应采集：复制当前环境，保持随机状态，验证已选分支前缀与真实执行一致。分支结果不输入控制器，不作额外 belief 证据。",
             "- case107 开关审计后 60 个窗口的选择和机器人位置完全一致；第一版审计因内部二次约束未通过积分检查，使用 failure_case_audit_v2，不使用失败版本。",
             "- H2 新增可选墙体上下文，输入仅为相对当前行人的可观测线段，具有平移和端点顺序不变性。缺少墙体信息是模型输入缺口，不等于已证明它解释全部性能差距。",
             "- H2 新增可选首步 mode-discrimination loss；旧 0.05 m 裕度配置为 88.89%，本轮 0.02 m 裕度为 98.61%，因此只能说明裕度修复后恢复闭环，不能把增益单独归因于 mode loss。",
             "- H3 新增可审计的 fallback-progress 死锁恢复开关；低权重版本在失败场景触发人碰撞，故保持默认关闭。",
             "- ORCA-augmented 作为候选支持对照达到与 live FRM 相同的扩展 CFS，但未消除超时，不能把它解释为 belief 收益。",
             "- 单车候选的执行器安全检查现在显式传递每条候选的 heading 和 signed speed；不再用世界速度方向错误重建 ActionRot。",
             "- 额外人体裕度从 0.05 m 调整为 0.02 m；0.05 m 在窄场景制造大量 executor fallback 和死锁，0.02 m 在开发/扩展集保持物理安全边界并恢复通过率。",
             "- 新增单机器人完整 H1→H2 预测头迁移：包括 encoder、均值头、动作响应头和候选相关标准差头。迁移初始化时每个模式与源模型数值一致，随后模式条件参数可学习。",
             "- 完整预测头迁移只适用于当前单机器人 joint 模型，不冒充多机器人 pair decoder 的完整迁移。旧配置仍走原有路径。", "",
             "## 实际控制器状态数据", "",
             f"冻结控制器在训练 case112–135 采集 {data_manifest['rows']['train']} 条分支/行人样本，在校准 case136–143 采集 {data_manifest['rows']['val']} 条。",
             "每六个窗口采集一次，包含已选分支和五个独立抽样候选；分支被环境改写时记录实际控制，并在响应差分审计中排除被改写分支。",
             "数据是该冻结控制器访问的状态，不保证对后续新模型全部 on-policy。不同 case 切分、初始状态、地图来源及源数据哈希均记录。", "",
             "## H1→H3 响应诊断", "",
             "同一验证状态及同一分支集合；下表不使用隐藏模式标签。响应差分只统计没有被环境投影的分支对。", "",
             "| checkpoint 家族 | NLL↓ | ADE↓ | 真实差分能量 | 预测差分能量 | 差分误差 MSE↓ | 后延迟差分相关 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for audit in response_audits:
        for model in audit["models"]:
            metrics, effects = model["metrics"]["all"], model["response_effects"]
            lines.append(f"| {Path(model['checkpoint']).parts[-3]} | {metrics['nll']:.5f} | {metrics['ade']:.5f} | {effects['true_effect_mse']:.6f} | {effects['predicted_effect_mse']:.6f} | {effects['effect_error_mse']:.6f} | {effects['post_delay_effect_correlation']:.4f} |")
    lines.extend(["", "旧模型严重低估候选改变造成的响应差异。实际状态数据改善总体误差；成组差分训练有小幅差分改善，但尚未充分恢复交互效应。", "",
                  "## H2：观测前校准", "",
                  "模式仍是训练身份首选速度的三分位代理；标签只用于训练和验证，不作为规划输入。", "",
                  "| 模型 | 选定 likelihood scale | 观测前模式 NLL↓ | 准确率 | persistent 混合 NLL↓ | uniform 混合 NLL↓ |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, record in h2.items():
        selected, mixture = record["calibration"]["selected"], record["mixture"]["metrics"]
        lines.append(f"| {name} | {selected['likelihood_scale']:.2f} | {selected['decision_mode_nll']:.6f} | {selected['decision_mode_accuracy']:.2%} | {mixture['persistent']['mixture_nll']:.6f} | {mixture['uniform']['mixture_nll']:.6f} |")
    if belief_selection is not None:
        lines.extend(["", f"同状态、同预测、同候选支持的 uniform 重评分：{belief_selection['windows']} 个窗口中，候选索引改变 {belief_selection['candidate_changes']} 次，实际执行前缀改变 {belief_selection['executed_action_changes']} 次。未执行任何诊断动作。"])
    if h2_navigation_audit is not None:
        lines.extend(["", f"三模式 H2 同状态策略审计（{h2_navigation_audit['scope']}）：persistent/reset/uniform 均报告 {h2_navigation_audit['policies']['tracer-persistent']['collision_free_success']:.2%}/{h2_navigation_audit['policies']['tracer-reset']['collision_free_success']:.2%}/{h2_navigation_audit['policies']['tracer-uniform']['collision_free_success']:.2%} 无碰撞成功；persistent 相对 uniform 重评分改变候选索引 {h2_navigation_audit['same_state_selection_changes']} 次、实际执行前缀 {h2_navigation_audit['same_state_action_changes']} 次。"])
    lines.extend(["", "均匀模式 NLL 为 1.098612；接近或差于它的结果不能叫模式推断优势。低条件 ADE 不等于 belief 的贡献。", "",
                  "## H3：8 场景开发验证", "",
                  "| 变体 | 策略 | 无碰撞成功率 | 到达率 | 人碰撞率 | 墙碰撞率 | 平均仿真时间 | 执行投影窗口 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in development:
        metrics = row["metrics"]
        lines.append(f"| {row['label']} | {row['policy']} | {metrics['collision_free_success']:.2%} | {metrics['success']:.2%} | {metrics['collision']:.2%} | {metrics['wall_collision']:.2%} | {metrics['total_time']:.3f} | {row['executor_projection_windows']} |")
    lines.extend(["", "## H3：24 场景×3 种子扩展", "",
                  "同一 case 先平均模型种子，再计算配对场景置信区间；不是把 72 条 rollout 当成 72 个独立场景。", "",
                  "| 变体 | TRACER 无碰撞成功率 | SICNav-NP | 差值 | 描述性 95% CI | 仿真时间差 | 墙碰撞率 | 投影窗口 |",
                  "|---|---:|---:|---:|---|---:|---:|---:|"])
    for name, record in extension.items():
        metrics = record["metrics"]
        success = metrics["collision_free_success"]
        lower, upper = success["descriptive_case_bootstrap_ci95"]
        lines.append(f"| {name} | {success['tracer']:.2%} | {success['official']:.2%} | {success['tracer_minus_official']:+.2%} | [{lower:+.2%}, {upper:+.2%}] | {metrics['total_time']['tracer_minus_official']:+.3f} | {metrics['wall_collision']['tracer']:.2%} | {record['executor_projection_windows']} |")
    lines.extend(["", "## 尚未完成", "",
                  "- 同一开发集达到 100% 不证明总体与 SICNav 非劣；更不能用它替代未使用场景的冻结检验。",
                  "- 若真实 belief 后端与 uniform/reset 轨迹一致，闭环提升只能归于数据、预测器或控制支持，不能归于在线 belief。",
                  "- 到达时间、动态人碰撞、死锁、模型种子稳定性仍必须一并审查，不只报告最好的成功率。",
                  "- 地图条件化和完整预测头迁移均是可选、可回溯实现；只有通过对应验证才能进入最终配置。",
                  "- 多机器人 H1/H2/H3 主实验仍需保留；这里的单机器人外部验证不能证明多机器人联合非加性。", ""])
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text("\n".join(lines))
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
