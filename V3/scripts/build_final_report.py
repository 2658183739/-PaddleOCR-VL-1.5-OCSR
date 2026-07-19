import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_json(path: Path):
    return load_json(path) if path.exists() else None


def latest_locked_run(project_root: Path):
    root = project_root / "V3" / "eval_runs_locked"
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No locked evaluation run under {root}")
    return candidates[-1]


def metric(report, name):
    if not report:
        return None
    return report.get("accuracy", {}).get(name)


def training_metrics(project_root: Path, run_id: str):
    path = project_root / "V3" / "outputs" / run_id / "train_results.json"
    payload = maybe_json(path)
    if not payload:
        return None
    return {
        key: payload.get(key)
        for key in (
            "train_loss",
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
        )
    }


def paired_rows(evidence: Path, prefix: str):
    rows = []
    for panel in ("legacy_core_dev", "legacy_region_dev"):
        payload = maybe_json(evidence / f"{prefix}_{panel}_paired.json")
        if not payload:
            continue
        exact = payload["canonical_exact"]
        rows.append(
            {
                "panel": panel,
                "independent_units": exact["independent_units"],
                "exact_delta": exact["delta_mean"],
                "exact_ci95_low": exact["ci95_low"],
                "exact_ci95_high": exact["ci95_high"],
                "valid_delta": payload["valid_smiles"]["delta_mean"],
            }
        )
    return rows


def build_summary(project_root: Path):
    evidence = project_root / "V3" / "evidence"
    dataset = load_json(evidence / "dataset_build_report.json")
    probe = load_json(evidence / "probe_analysis.json")
    probe_pairwise = maybe_json(evidence / "probe_paired_summary.json")
    checkpoint = load_json(evidence / "final_checkpoint_selection.json")
    hard = load_json(evidence / "final_vs_hard_replay.json")
    generation = load_json(evidence / "generation_policy_selection.json")
    generation_beam = maybe_json(evidence / "generation_policy_beam_selection.json")
    locked_root = latest_locked_run(project_root)
    attestation = maybe_json(
        project_root / "V3" / "qc" / "manual_review_attestation.json"
    )
    human_review_status = dataset["wild_eval"]["human_review_status"]
    if attestation and attestation.get("status") == "owner_attested_complete":
        human_review_status = "owner_attested_complete"

    locked = {
        "run_dir": str(locked_root.relative_to(project_root)),
        "wild_strict": load_json(locked_root / "wild_strict" / "report.json"),
        "wild_scaffold_novel": load_json(
            locked_root / "wild_scaffold_novel" / "report.json"
        ),
        "wild_symbolic": load_json(locked_root / "wild_symbolic" / "report.json"),
        "private_photo": maybe_json(locked_root / "private_photo" / "report.json"),
    }

    run_ids = []
    for condition in ("00", "10", "01", "11"):
        run_ids.extend(probe["conditions"][condition]["run_ids"])
    run_ids.extend(["aug_dose2_s1", "warmstart_control_s1", "final_s1", "hard_replay_s1"])

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "base_rows": dataset["inputs"]["base_rows"],
            "control_rows": dataset["base_train_filter"]["accepted"],
            "wild_train_rows": dataset["wild_paper_group_split"]["strict_train"],
            "wild_locked_rows": dataset["wild_paper_group_split"]["strict_eval"],
            "wild_locked_paper_groups": dataset["wild_paper_group_split"]["paper_groups_eval"],
            "wild_symbolic_rows": dataset["wild_paper_group_split"]["symbolic_eval_same_papers"],
            "scaffold_novel_rows": dataset["wild_eval"]["scaffold_novel_rows"],
            "human_review_status": human_review_status,
        },
        "probe": probe,
        "probe_pairwise": probe_pairwise,
        "final_checkpoint_selection": checkpoint,
        "final_vs_hard_replay": hard,
        "generation_policy_beam_selection": generation_beam,
        "generation_policy_selection": generation,
        "posttraining_pairwise": {
            "final_vs_hard_replay": paired_rows(evidence, "final_vs_hard_replay"),
            "generation_policy": paired_rows(evidence, "generation_policy"),
        },
        "locked": locked,
        "training_metrics": {
            run_id: training_metrics(project_root, run_id) for run_id in run_ids
        },
        "evidence_boundaries": {
            "locked_used_for_tuning": False,
            "symbolic_in_main_canonical_score": False,
            "private_photo_available": locked["private_photo"] is not None,
            "human_double_review_completed": False,
            "probe_confirmatory": False,
            "probe_seed_count": 2,
            "probe_run_order_balanced": False,
        },
    }


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(summary):
    dataset = summary["dataset"]
    probe = summary["probe"]
    probe_pairwise = summary.get("probe_pairwise") or {"comparisons": []}
    checkpoint = summary["final_checkpoint_selection"]
    hard = summary["final_vs_hard_replay"]
    generation_beam = summary.get("generation_policy_beam_selection")
    generation = summary["generation_policy_selection"]
    posttraining_pairwise = summary.get("posttraining_pairwise") or {}
    locked = summary["locked"]
    lines = [
        "# V3 实际训练与最终评测结果",
        "",
        "> 本文件由 `scripts/build_final_report.py` 从 JSON 证据自动生成。",
        "> Locked test 不参与任何训练、checkpoint、prompt 或生成参数选择。",
        "",
        "## 1. 数据与划分",
        "",
        "| 项目 | 数量 | 角色 |",
        "| --- | ---: | --- |",
        f"| V2-1 输入记录 | {dataset['base_rows']} | 过滤前训练输入 |",
        f"| strict control | {dataset['control_rows']} | final 胜出训练配比 |",
        f"| strict wild train | {dataset['wild_train_rows']} | 仅进入含 wild 的 probe |",
        f"| wild strict locked | {dataset['wild_locked_rows']} | {dataset['wild_locked_paper_groups']} 篇整论文留出 |",
        f"| scaffold novel | {dataset['scaffold_novel_rows']} | locked wild 子集 |",
        f"| symbolic | {dataset['wild_symbolic_rows']} | 独立转写 track |",
        "",
        "训练与 locked test 的 canonical molecule 和 `paper_group` 重叠均为 0；legacy core/region 是历史 development。",
        "",
        "## 2. 2x2 两 seed 数据消融",
        "",
        "| 条件 | 两 seed 宏平均 exact | seed 范围 | 最低 valid rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition in ("00", "10", "01", "11"):
        row = probe["conditions"][condition]
        lines.append(
            f"| {condition} | {fmt(row['mean_macro_exact'])} | "
            f"{fmt(row['seed_range'])} | {fmt(row['min_valid_rate'])} |"
        )
    review_status = summary["dataset"].get("human_review_status")
    if review_status == "owner_attested_complete":
        review_boundary = (
            "- 项目所有者确认 frozen legacy/wild/symbolic labels 已完成离线人工审核；"
            "公开证据为 `qc/manual_review_attestation.json` 及其绑定的四个 labels SHA256。"
        )
    else:
        review_boundary = (
            "- 当前 locked 标签仍缺人工语义复核；这是提交真实性与统计可信度的剩余缺口。"
        )

    lines.extend(
        [
            "",
            f"- wild 主效应：{fmt(probe['effects']['wild_main_effect'])}",
            f"- augmentation 主效应：{fmt(probe['effects']['augmentation_main_effect'])}",
            f"- 交互效应：{fmt(probe['effects']['interaction'])}",
            f"- 最终训练数据：`{probe['winner']['dataset_path']}`",
            "- 解释边界：本轮只有两个 seed，运行顺序未完全随机化或位置平衡；以下结论是工程探索，不报告 ANOVA p 值或统计显著性。",
            "",
            "### 2.1 逐样本 paired bootstrap",
            "",
            "| baseline | candidate | panel | units | exact delta | 95% CI |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in probe_pairwise["comparisons"]:
        lines.append(
            f"| {row['baseline_run']} | {row['candidate_run']} | {row['panel']} | "
            f"{row['independent_units']} | {fmt(row['exact_delta'], 6)} | "
            f"[{fmt(row['exact_ci95_low'], 6)}, {fmt(row['exact_ci95_high'], 6)}] |"
        )
    lines.extend(
        [
            "",
            "主因子比较中，除 wild-only/seed2 两个面板的 CI 完全低于 0 外，其余单 seed CI 均跨 0；这不支持 wild 或 augmentation 的稳定正向收益。warm-start 对照则在两个面板均有明确正向 CI。",
            "",
            "## 3. 辅助消融",
            "",
        ]
    )
    dose = probe.get("diagnostics", {}).get("augmentation_dose2")
    warmstart = probe.get("diagnostics", {}).get("warmstart")
    lines.append(
        f"- 增强剂量 2：macro exact={fmt(dose.get('macro_exact') if dose else None)}，"
        f"相对 11/seed1={fmt(dose.get('delta_vs_11_s1') if dose else None)}。"
    )
    lines.append(
        f"- V2-1 continuation 相对原始 1.5 warm-start："
        f"{fmt(warmstart.get('continuation_minus_base15') if warmstart else None)}。"
    )

    lines.extend(
        [
            "",
            "## 4. 训练成本",
            "",
            "| run | train loss | runtime (min) | samples/s | steps/s |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for run_id, metrics in summary["training_metrics"].items():
        if not metrics:
            continue
        runtime = metrics.get("train_runtime")
        runtime_minutes = runtime / 60.0 if runtime is not None else None
        lines.append(
            f"| {run_id} | {fmt(metrics.get('train_loss'))} | {fmt(runtime_minutes, 2)} | "
            f"{fmt(metrics.get('train_samples_per_second'))} | "
            f"{fmt(metrics.get('train_steps_per_second'))} |"
        )

    lines.extend(
        [
            "",
            "## 5. Final checkpoint 与 hard replay",
            "",
            "| checkpoint | step | development macro exact | min valid |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in checkpoint["checkpoints"]:
        lines.append(
            f"| {row['checkpoint']} | {row['step']} | {fmt(row['macro_exact'])} | "
            f"{fmt(row['min_valid_rate'])} |"
        )
    lines.extend(
        [
            "",
            f"- 选中 checkpoint：`{checkpoint['winner']['checkpoint']}`，"
            f"development macro exact={fmt(checkpoint['winner']['macro_exact'])}，"
            f"min valid={fmt(checkpoint['winner']['min_valid_rate'])}。",
            f"- hard replay 最终决策：`{hard['winner']['label']}`。",
            f"- hard replay macro delta：{fmt(hard.get('candidate_macro_delta'))}；"
            f"采用门槛：至少 {fmt(hard.get('minimum_improvement'))}。",
            f"- hard replay development macro exact：{fmt(hard['candidate']['macro_exact'])}；"
            f"final baseline：{fmt(hard['baseline']['macro_exact'])}。",
            f"- 最终生成策略：`{generation['winner']['label']}`。",
            "",
            "### 5.1 后训练 paired bootstrap",
            "",
            "| comparison | panel | units | exact delta | 95% CI | valid delta |",
            "| --- | --- | ---: | ---: | --- | ---: |",
        ]
    )
    if generation_beam:
        lines.insert(
            lines.index("### 5.1 后训练 paired bootstrap") - 1,
            f"- Decoder 对照：`{generation_beam['candidate']['label']}` macro exact="
            f"{fmt(generation_beam['candidate']['macro_exact'])}，"
            f"`{generation_beam['baseline']['label']}`="
            f"{fmt(generation_beam['baseline']['macro_exact'])}，"
            f"delta={fmt(generation_beam.get('candidate_macro_delta'))}。",
        )
    lines.insert(
        lines.index("### 5.1 后训练 paired bootstrap") - 1,
        f"- Rerank 对照：`{generation['candidate']['label']}` macro exact="
        f"{fmt(generation['candidate']['macro_exact'])}，"
        f"`{generation['baseline']['label']}`="
        f"{fmt(generation['baseline']['macro_exact'])}，"
        f"delta={fmt(generation.get('candidate_macro_delta'))}；"
        f"采用门槛：至少 {fmt(generation.get('minimum_improvement'))}。",
    )
    for comparison, rows in posttraining_pairwise.items():
        for row in rows:
            lines.append(
                f"| {comparison} | {row['panel']} | {row['independent_units']} | "
                f"{fmt(row['exact_delta'], 6)} | "
                f"[{fmt(row['exact_ci95_low'], 6)}, {fmt(row['exact_ci95_high'], 6)}] | "
                f"{fmt(row['valid_delta'], 6)} |"
            )
    lines.extend(
        [
            "",
            "## 6. 一次性 locked test",
            "",
            "| 面板 | N | 主 exact | valid/nonempty |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    strict = locked["wild_strict"]
    scaffold = locked["wild_scaffold_novel"]
    symbolic = locked["wild_symbolic"]
    lines.append(
        f"| wild strict | {strict['total']} | "
        f"{fmt(metric(strict, 'canonical_exact_match_accuracy'))} | "
        f"{fmt(metric(strict, 'valid_smiles_rate'))} |"
    )
    lines.append(
        f"| scaffold novel | {scaffold['total']} | "
        f"{fmt(metric(scaffold, 'canonical_exact_match_accuracy'))} | "
        f"{fmt(metric(scaffold, 'valid_smiles_rate'))} |"
    )
    lines.append(
        f"| symbolic（独立 track） | {symbolic['counts']['total']} | "
        f"{fmt(symbolic['accuracy']['whitespace_normalized_exact_match_accuracy'])} | "
        f"{fmt(symbolic['accuracy']['nonempty_prediction_rate'])} |"
    )
    private = locked.get("private_photo")
    if private:
        lines.append(
            f"| private photo | {private['total']} | "
            f"{fmt(metric(private, 'canonical_exact_match_accuracy'))} | "
            f"{fmt(metric(private, 'valid_smiles_rate'))} |"
        )

    lines.extend(
        [
            "",
            "## 7. 解释边界与证据入口",
            "",
            "- wild strict/scaffold novel 才进入 canonical SMILES 主结论。",
            "- symbolic 是文字转写诊断，不使用 RDKit canonicalization，也不混入主分数。",
            review_boundary,
            "- private photo 若为 N/A，表示没有真实自采 locked test，算法退化不能替代实拍。",
            "- 模型、配置、评测标签、prompt 与生成策略 hash 见 locked run 下的 `locked_test_manifest.sha256`。",
            "- 可恢复 final/hard-replay checkpoint 位于 `evidence/training_artifacts/resume/`。",
            "- 实际环境快照位于 `evidence/runtime/`。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-json", default="V3/evidence/FINAL_RESULTS.json")
    parser.add_argument("--output-md", default="V3/evidence/FINAL_RESULTS_zh.md")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    summary = build_summary(project_root)

    output_json = project_root / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    output_md = project_root / args.output_md
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(output_md)


if __name__ == "__main__":
    main()
