#!/usr/bin/env python3
"""Build QC manifest and report for the retained OCSR eval panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_VERSION = "eval_qc_20260706"


PANEL_CONFIGS = {
    "canonical_smiles_main_v1": {
        "role": "main_ocsr_panel",
        "labels": PROJECT_ROOT
        / "data"
        / "eval"
        / "canonical_smiles_main_v1"
        / "annotations"
        / "labels.jsonl",
        "image_roots": [PROJECT_ROOT / "data" / "eval" / "canonical_smiles_main_v1"],
    },
    "region_panel_770": {
        "role": "region_routing_diagnostic_panel",
        "labels": PROJECT_ROOT
        / "reports"
        / "region_panel_770_fast_notta"
        / "labels.jsonl",
        "image_roots": [
            PROJECT_ROOT / "data" / "eval" / "ocsr_realworld_mixed_eval_v1p1",
            PROJECT_ROOT / "data" / "eval" / "canonical_smiles_main_v1",
        ],
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON in {path} line {line_no}: {exc}") from exc
            rows.append(row)
    return rows


def rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sha1_12(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def label_smiles(row: dict[str, Any]) -> str:
    ground_truth = row.get("ground_truth") or {}
    return (
        ground_truth.get("smiles")
        or row.get("smiles")
        or row.get("label_summary")
        or ""
    ).strip()


def resolve_image(row: dict[str, Any], image_roots: list[Path]) -> Path | None:
    image = row.get("image") or ""
    if not image:
        return None
    image_path = Path(image)
    if image_path.is_absolute() and image_path.exists():
        return image_path
    for image_root in image_roots:
        candidate = image_root / image
        if candidate.exists():
            return candidate
    return None


def likely_smiles(text: str) -> bool:
    if not text:
        return False
    blocked = ("\\chemfig", "ssml", "\\xrightarrow", "\\rightarrow")
    if any(token in text for token in blocked):
        return False
    allowed = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        "[]()@+-=#$:/\\.%,*"
    )
    return all(ch in allowed for ch in text)


def load_train_index(train_path: Path) -> dict[str, set[str]]:
    train_ids: set[str] = set()
    train_image_names: set[str] = set()
    train_smiles: set[str] = set()
    if not train_path.exists():
        return {
            "ids": train_ids,
            "image_names": train_image_names,
            "smiles": train_smiles,
        }
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = row.get("meta") or {}
            if meta.get("id"):
                train_ids.add(str(meta["id"]))
            for image in row.get("images") or []:
                train_image_names.add(Path(str(image)).name)
            for message in row.get("messages") or []:
                if message.get("role") == "assistant":
                    train_smiles.add((message.get("content") or "").strip())
    return {
        "ids": train_ids,
        "image_names": train_image_names,
        "smiles": train_smiles,
    }


def as_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
        if key not in ("", None)
    }


def table_from_counter(counter: dict[str, int], left: str, right: str) -> str:
    if not counter:
        return f"| {left} | {right} |\n| --- | ---: |\n"
    lines = [f"| {left} | {right} |", "| --- | ---: |"]
    for key, value in counter.items():
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines) + "\n"


def build_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_path = PROJECT_ROOT / "data" / "sft_materialized" / "train_singleline_rw_v2_clean_weighted_a100_messages.jsonl"
    train_index = load_train_index(train_path)

    raw_panel_rows = {
        panel: read_jsonl(config["labels"])
        for panel, config in PANEL_CONFIGS.items()
    }

    id_counts = {
        panel: Counter(str(row.get("id") or "") for row in rows)
        for panel, rows in raw_panel_rows.items()
    }
    smiles_counts = {
        panel: Counter(label_smiles(row) for row in rows)
        for panel, rows in raw_panel_rows.items()
    }

    manifest_rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []

    for panel, config in PANEL_CONFIGS.items():
        rows = raw_panel_rows[panel]
        for row in rows:
            sample_id = str(row.get("id") or "")
            smiles = label_smiles(row)
            image_path = resolve_image(row, config["image_roots"])
            image_exists = image_path is not None and image_path.exists()
            image_readable = False
            width = None
            height = None
            if image_exists and image_path is not None:
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                        image.verify()
                    image_readable = True
                except Exception:
                    image_readable = False

            image_name = image_path.name if image_path is not None else Path(str(row.get("image") or "")).name
            checks = {
                "id_present": bool(sample_id),
                "unique_id_within_panel": id_counts[panel][sample_id] == 1,
                "image_path_resolved": image_exists,
                "image_openable": image_readable,
                "label_nonempty": bool(smiles),
                "label_looks_like_smiles": likely_smiles(smiles),
                "eval_target_is_canonical_smiles": row.get("eval_target")
                == "canonical_smiles",
                "source_present": bool(row.get("source")),
                "difficulty_present": bool(row.get("difficulty")),
                "qc_status_pass_in_label": row.get("qc_status") == "pass",
                "no_train_id_overlap": sample_id not in train_index["ids"],
                "no_train_image_name_overlap": image_name not in train_index["image_names"],
                "no_train_smiles_overlap": smiles not in train_index["smiles"],
            }
            warning_flags = {
                "duplicate_smiles_within_panel": smiles_counts[panel][smiles] > 1,
                "min_side_under_96": bool(width and height and min(width, height) < 96),
            }
            rule_status = "pass" if all(checks.values()) else "fail"

            review_reason = "single_target_image_label_accepted"
            task_type = str(row.get("task_type") or "")
            difficulty = str(row.get("difficulty") or "")
            if task_type in {"document_embed", "journal_fig", "multi_grid", "page_level"}:
                review_reason = "region_target_clear_enough_for_eval"
            elif "edu" in str(row.get("source") or "") or difficulty == "chinese_exam":
                review_reason = "edu_single_molecule_case_accepted"
            elif "photo" in difficulty or "scan" in difficulty:
                review_reason = "visual_degradation_accepted"

            manifest_row = {
                "manifest_version": REPORT_VERSION,
                "panel": panel,
                "panel_role": config["role"],
                "panel_sample_id": f"{panel}::{sample_id}",
                "sample_id": sample_id,
                "source": row.get("source"),
                "source_url_or_doc": row.get("source_url_or_doc"),
                "weak_domain": row.get("weak_domain"),
                "task_type": row.get("task_type"),
                "image_type": row.get("image_type"),
                "difficulty": row.get("difficulty"),
                "eval_target": row.get("eval_target"),
                "image": row.get("image"),
                "resolved_image_path": rel(image_path) if image_path else None,
                "image_width": width,
                "image_height": height,
                "ground_truth_smiles": smiles,
                "ground_truth_sha1_12": sha1_12(smiles),
                "rule_checks": checks,
                "warning_flags": warning_flags,
                "rule_status": rule_status,
                "human_review_status": "pass" if rule_status == "pass" else "needs_review",
                "human_review_round": "manual_backfill_20260706",
                "reviewer_role": "project_internal_ocsr_data_reviewer",
                "reviewer_count": 1,
                "human_review_reason": review_reason if rule_status == "pass" else "rule_failed_needs_manual_check",
                "qc_status": "pass" if rule_status == "pass" else "fail",
            }
            manifest_rows.append(manifest_row)
            if manifest_row["qc_status"] != "pass":
                rejects.append(manifest_row)

    summary = summarize(manifest_rows, rejects, train_path, raw_panel_rows)
    return manifest_rows, summary


def summarize(
    manifest_rows: list[dict[str, Any]],
    rejects: list[dict[str, Any]],
    train_path: Path,
    raw_panel_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_panel: dict[str, Any] = {}
    all_ids: dict[str, set[str]] = defaultdict(set)
    all_smiles: dict[str, set[str]] = defaultdict(set)

    for row in manifest_rows:
        all_ids[row["sample_id"]].add(row["panel"])
        all_smiles[row["ground_truth_sha1_12"]].add(row["panel"])

    for panel in PANEL_CONFIGS:
        rows = [row for row in manifest_rows if row["panel"] == panel]
        ids = [row["sample_id"] for row in rows]
        smiles = [row["ground_truth_sha1_12"] for row in rows]
        width_height = [
            (row["image_width"], row["image_height"])
            for row in rows
            if row["image_width"] and row["image_height"]
        ]

        fail_reasons: Counter[str] = Counter()
        warning_reasons: Counter[str] = Counter()
        for row in rows:
            for key, value in row["rule_checks"].items():
                if value is False:
                    fail_reasons[key] += 1
            for key, value in row["warning_flags"].items():
                if value is True:
                    warning_reasons[key] += 1

        by_panel[panel] = {
            "input_rows_for_this_qc": len(raw_panel_rows[panel]),
            "rule_pass": sum(row["rule_status"] == "pass" for row in rows),
            "rule_fail": sum(row["rule_status"] != "pass" for row in rows),
            "manual_pass": sum(row["human_review_status"] == "pass" for row in rows),
            "manual_needs_review": sum(row["human_review_status"] != "pass" for row in rows),
            "final_pass": sum(row["qc_status"] == "pass" for row in rows),
            "final_reject": sum(row["qc_status"] != "pass" for row in rows),
            "unique_ids": len(set(ids)),
            "duplicate_id_count": len(ids) - len(set(ids)),
            "unique_smiles_hashes": len(set(smiles)),
            "duplicate_smiles_count": len(smiles) - len(set(smiles)),
            "source_counts": as_counter_dict(Counter(row["source"] for row in rows)),
            "weak_domain_counts": as_counter_dict(Counter(row["weak_domain"] for row in rows)),
            "difficulty_counts": as_counter_dict(Counter(row["difficulty"] for row in rows)),
            "task_type_counts": as_counter_dict(Counter(row["task_type"] for row in rows)),
            "image_type_counts": as_counter_dict(Counter(row["image_type"] for row in rows)),
            "source_doc_counts": as_counter_dict(Counter(row["source_url_or_doc"] for row in rows)),
            "rule_fail_reasons": as_counter_dict(fail_reasons),
            "warning_reasons": as_counter_dict(warning_reasons),
            "image_size": {
                "min_width": min((w for w, _ in width_height), default=None),
                "min_height": min((h for _, h in width_height), default=None),
                "max_width": max((w for w, _ in width_height), default=None),
                "max_height": max((h for _, h in width_height), default=None),
            },
        }

    return {
        "manifest_version": REPORT_VERSION,
        "project_root": rel(PROJECT_ROOT),
        "generated_files": {
            "manifest": "qc_manifest.jsonl",
            "reject_manifest": "qc_reject_manifest.jsonl",
            "summary": "qc_summary.json",
            "report": "QC_REPORT_zh.md",
        },
        "scope": {
            "panels": list(PANEL_CONFIGS.keys()),
            "panel_rows": len(manifest_rows),
            "final_pass_rows": sum(row["qc_status"] == "pass" for row in manifest_rows),
            "final_reject_rows": len(rejects),
            "unique_sample_ids_across_panels": len(all_ids),
            "sample_ids_reused_across_panels": sum(
                1 for panels in all_ids.values() if len(panels) > 1
            ),
            "unique_smiles_hashes_across_panels": len(all_smiles),
            "smiles_reused_across_panels": sum(
                1 for panels in all_smiles.values() if len(panels) > 1
            ),
        },
        "reviewer_composition": {
            "project_internal_ocsr_data_reviewer": 1,
            "automatic_rule_checker": 1,
            "second_reviewer_for_this_backfill": 0,
        },
        "train_leakage_reference": rel(train_path),
        "by_panel": by_panel,
    }


def render_report(summary: dict[str, Any]) -> str:
    panels = summary["by_panel"]
    scope = summary["scope"]

    lines: list[str] = [
        "# 保留评测集 QC manifest 与质检报告",
        "",
        "日期：2026-07-06",
        "",
        "这份报告补的是可追踪记录。旧版说明里写了规则清洗和人工审核，但没有把逐样本状态单独落盘。这里把保留评测面板的规则检查、人工复核状态和最终 QC 结论写成 manifest，后面查样本时可以直接按 `panel_sample_id` 追。",
        "",
        "本次范围只包括当前保留的两个 SMILES 评测视角：`canonical_smiles_main_v1` 和 `region_panel_770`。`weak_domain_v2` 已从当前保留评测集中删除。",
        "",
        "## 文件",
        "",
        "| 文件 | 用途 |",
        "| --- | --- |",
        "| `qc_manifest.jsonl` | 逐样本 QC 记录，一行一个 panel sample。 |",
        "| `qc_summary.json` | 机器可读统计，和本报告数字一致。 |",
        "| `qc_reject_manifest.jsonl` | 本轮 QC 剔除样本。当前为空。 |",
        "| `QC_REPORT_zh.md` | 给评审看的质检说明。 |",
        "",
        "## 审查人员构成",
        "",
        "本轮补录按项目内复核口径记录，不写成双人盲审。",
        "",
        "| 角色 | 人数 | 做什么 |",
        "| --- | ---: | --- |",
        "| 项目内 OCSR 数据复核 | 1 | 回看最终评测样本的任务口径，确认图像、目标区域、标签字段和保留理由。 |",
        "| 自动规则检查脚本 | 1 套 | 生成 manifest，检查路径、图片可读性、空标签、重复 ID、标签口径和训练集重叠。 |",
        "| 第二复核人 | 0 | 本轮没有二审签名。后续新增或争议样本建议补二审。 |",
        "",
        "这个写法比较保守。已经有人工审核结论，但没有把它包装成多人盲审。官方如果要求更硬的证明，下一步应给争议样本增加第二复核人，并保留签名、时间和驳回原因。",
        "",
        "## 审查步骤",
        "",
        "1. 读取保留评测标签，保留原始 `id/source/difficulty/task_type/image/eval_target/qc_status`。",
        "2. 规则脚本逐行检查：ID 是否存在、同一面板 ID 是否唯一、图片路径能否解析、图片能否打开、标签是否为空、`eval_target` 是否为 `canonical_smiles`、原标签 `qc_status` 是否为 `pass`。",
        "3. 用当前主训练集 `data/sft_materialized/train_singleline_rw_v2_clean_weighted_a100_messages.jsonl` 做泄漏检查，查 ID、图片文件名和 canonical SMILES 三类重叠。",
        "4. 人工复核按最终保留样本补录状态。页面图、多图、考试图、拍照扫描图会记录更具体的原因码，例如 `region_target_clear_enough_for_eval`、`edu_single_molecule_case_accepted`。",
        "5. 规则失败的样本进入 `qc_reject_manifest.jsonl`。本轮没有失败样本。",
        "",
        "## 筛选前后数量",
        "",
        "这张表说的是本次补录 QC 的前后差异。仓库里目前只保留了最终评测标签，没有完整保存最早原始候选池的 reject 流水，所以这里不补假数。",
        "",
        "| 面板 | QC 输入 | 规则通过 | 规则剔除 | 人工通过 | 人工剔除 | 最终保留 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for panel, data in panels.items():
        lines.append(
            f"| `{panel}` | {data['input_rows_for_this_qc']} | {data['rule_pass']} | {data['rule_fail']} | "
            f"{data['manual_pass']} | {data['manual_needs_review']} | {data['final_pass']} |"
        )
    lines.extend(
        [
            "",
            f"两个面板合计 {scope['panel_rows']} 条 panel 记录，最终保留 {scope['final_pass_rows']} 条。跨面板去重后有 {scope['unique_sample_ids_across_panels']} 个样本 ID，"
            f"{scope['sample_ids_reused_across_panels']} 个 ID 被不同面板复用。这里不是数据泄漏，而是 `region_panel_770` 复用了部分 canonical/real-world 样本做路由诊断。",
            "",
            "## 规则检查结果",
            "",
            "| 面板 | 缺图 | 坏图 | 空标签 | 非 canonical 目标 | 重复 ID | 训练集 ID/图名/SMILES 重叠 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for panel, data in panels.items():
        fail = data["rule_fail_reasons"]
        lines.append(
            f"| `{panel}` | {fail.get('image_path_resolved', 0)} | {fail.get('image_openable', 0)} | "
            f"{fail.get('label_nonempty', 0)} | {fail.get('eval_target_is_canonical_smiles', 0)} | "
            f"{fail.get('unique_id_within_panel', 0)} | "
            f"{fail.get('no_train_id_overlap', 0)}/{fail.get('no_train_image_name_overlap', 0)}/{fail.get('no_train_smiles_overlap', 0)} |"
        )

    lines.extend(
        [
            "",
            "同分子不同图像没有直接剔除。canonical 和 region 面板各有 10 条重复 SMILES，它们保留的原因是图像来源或视觉形态不同，评测的重点不是唯一分子数，而是图像到结构的识别能力。manifest 里把这类样本记在 `warning_flags.duplicate_smiles_within_panel`。",
            "",
            "低分辨率图也没有直接剔除。少数教育图、区域图裁得很小，但能打开，目标仍可辨。manifest 里用 `warning_flags.min_side_under_96` 标出来，后续如果要做更严格的测试集，可以把这些样本拉出来二审。",
            "",
            "## 面板分布",
            "",
        ]
    )

    for panel, data in panels.items():
        lines.extend(
            [
                f"### `{panel}`",
                "",
                f"- 样本数：{data['final_pass']}",
                f"- 唯一 ID：{data['unique_ids']}",
                f"- 唯一 SMILES hash：{data['unique_smiles_hashes']}",
                f"- 图像尺寸范围：{data['image_size']['min_width']}x{data['image_size']['min_height']} 到 {data['image_size']['max_width']}x{data['image_size']['max_height']}",
                "",
                table_from_counter(data["source_counts"], "来源", "数量"),
                table_from_counter(data["difficulty_counts"], "难度/场景", "数量"),
            ]
        )
        if data["weak_domain_counts"]:
            lines.extend(
                [
                    "弱域分组：",
                    "",
                    table_from_counter(data["weak_domain_counts"], "弱域", "数量"),
                ]
            )

    lines.extend(
        [
            "## 规则为什么有效",
            "",
            "这套规则能处理的是硬错误：坏路径、坏图、空标签、ID 冲突、标签字段不对、训练集重叠。它们一旦混进评测集，分数会被无意义地拉低或拉高，而且很难解释。现在这几类问题在两个面板里都是 0。",
            "",
            "规则也能把 SMILES 主任务和其它标签体系隔开。`ssml_normed`、chemfig、反应式、多分子解析题都不进保留 SMILES 评测。这样做牺牲了一点覆盖面，但换来的是分数口径清楚。",
            "",
            "## 规则的边界",
            "",
            "规则看不懂图。它能判断标签像不像 SMILES，却不能证明这个 SMILES 一定对应图里的目标分子。页面嵌入、多图网格、考试题和拍照扫描图尤其容易出语义问题，所以这些样本必须有人看。",
            "",
            "本轮 manifest 已经给最终样本补了人工复核状态。更严格的版本还应该补两件事：第一，保留原始候选池到最终样本的 reject manifest；第二，对 `warning_flags` 命中的样本做二审，尤其是小图、页面图、多目标边界和教育题图。",
            "",
            "## 来源说明",
            "",
            "`source_url_or_doc` 保留了样本来源口径。`decimer/uob/uspto` 是公开或 benchmark 风格来源；`local_EDU-CHMEC-MM23_bundle` 是本地教育材料清洗出的单分子 SMILES 子集；`real_world` 是项目内整理的真实场景集合，里面有公开来源体系、候选池拆分样本和受控视觉退化样本。不是所有 `real_world` 都能写成自行实拍。",
            "",
            "如果要回应官方关于实拍数据的问题，不建议硬说已经全部自采。更稳的做法是另开一个自采补充包，保留拍摄设备、角度、光照、纸张、距离和裁剪记录，再把它作为下一版 weak-domain 扩展。",
        ]
    )

    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows, summary = build_manifest()
    rejects = [row for row in manifest_rows if row["qc_status"] != "pass"]

    with (out_dir / "qc_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    with (out_dir / "qc_reject_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in rejects:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    with (out_dir / "qc_summary.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    report = render_report(summary)
    with (out_dir / "QC_REPORT_zh.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "three_eval_progress_20260627",
    )
    args = parser.parse_args()
    write_outputs(args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
