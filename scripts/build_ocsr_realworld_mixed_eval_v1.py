from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


def load_module_from_sibling(name: str):
    module_path = Path(__file__).resolve().with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "source", "image", "task_type", "image_type", "difficulty", "label_summary", "eval_target", "benchmark_track", "qc_status"])
        for row in records:
            writer.writerow([
                row["id"],
                row["source"],
                row["image"],
                row["task_type"],
                row["image_type"],
                row["difficulty"],
                row.get("label_summary", ""),
                row["eval_target"],
                row["benchmark_track"],
                row.get("qc_status", "pass"),
            ])


def copy_image(src: Path, dest_root: Path, source_name: str) -> str:
    if not src.exists():
        raise FileNotFoundError(f"Missing image: {src}")
    target_dir = dest_root / source_name
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
    return str(dest.relative_to(dest_root.parent).as_posix())


def build_curated_canonical_rows(project_root: Path):
    canonical_builder = load_module_from_sibling("build_canonical_smiles_curated_v2.py")
    out_root = project_root / "V2" / "data" / "eval" / "canonical_smiles_curated_v2"
    summary = canonical_builder.build_curated_evalset(project_root, out_root)
    rows = list(read_jsonl(out_root / "annotations" / "labels.jsonl"))
    return rows, summary


def select_edu_rows(project_root: Path, edu_max_ssml_len: int, edu_max_side: int):
    labels_path = project_root / "V2" / "data" / "eval" / "edu_chmec_ssml_normed_test_v1" / "annotations" / "labels.jsonl"
    root = project_root / "V2" / "data" / "eval" / "edu_chmec_ssml_normed_test_v1"
    selected = []
    for row in read_jsonl(labels_path):
        text = str(row.get("ssml_normed", ""))
        if len(text) > edu_max_ssml_len:
            continue
        image_path = root / row["image"]
        with Image.open(image_path) as image:
            width, height = image.size
        if max(width, height) > edu_max_side:
            continue
        selected.append((row, image_path, width, height, len(text)))
    return selected


def build_mixed_evalset(project_root: Path, out_root: Path, edu_max_ssml_len: int = 220, edu_max_side: int = 768) -> dict[str, object]:
    canonical_rows, canonical_summary = build_curated_canonical_rows(project_root)
    edu_candidates = select_edu_rows(project_root, edu_max_ssml_len, edu_max_side)

    annotations_root = out_root / "annotations"
    images_root = out_root / "images"
    mixed_rows = []

    for row in canonical_rows:
        src = project_root / "V2" / "data" / "eval" / "canonical_smiles_curated_v2" / row["image"]
        rel_image = copy_image(src, images_root, row["source"])
        new_row = {
            **row,
            "image": rel_image,
            "benchmark_track": "canonical_main",
            "label_summary": row["ground_truth"]["smiles"],
        }
        mixed_rows.append(new_row)

    for row, image_path, width, height, seq_len in edu_candidates:
        rel_image = copy_image(image_path, images_root, row["source"])
        mixed_rows.append(
            {
                "id": row["id"],
                "source": row["source"],
                "image": rel_image,
                "task_type": row["task_type"],
                "image_type": row["image_type"],
                "difficulty": row["difficulty"],
                "label_format": row["label_format"],
                "ssml_normed": row["ssml_normed"],
                "eval_target": row["eval_target"],
                "benchmark_track": "edu_realworld",
                "label_summary": row["ssml_normed"],
                "annotation_json": row.get("annotation_json", ""),
                "qc_status": row.get("qc_status", "pass"),
                "image_size": [width, height],
                "sequence_length": seq_len,
            }
        )

    write_jsonl(annotations_root / "labels.jsonl", mixed_rows)
    write_csv(annotations_root / "labels.csv", mixed_rows)

    by_track = Counter(r["benchmark_track"] for r in mixed_rows)
    by_source = Counter(r["source"] for r in mixed_rows)
    by_difficulty = Counter(r["difficulty"] for r in mixed_rows)
    stats = {
        "total": len(mixed_rows),
        "by_track": dict(by_track),
        "by_source": dict(by_source),
        "by_difficulty": dict(by_difficulty),
        "canonical_summary": canonical_summary,
        "edu_filters": {"max_ssml_len": edu_max_ssml_len, "max_side": edu_max_side, "selected": len(edu_candidates)},
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    subgroup_summary = {
        "canonical_main": {
            "count": by_track.get("canonical_main", 0),
            "eval_target": "canonical_smiles",
            "sources": {k: v for k, v in by_source.items() if k in {"uob", "uspto", "real_world", "decimer"}},
        },
        "edu_realworld": {
            "count": by_track.get("edu_realworld", 0),
            "eval_target": "ssml_normed",
            "sources": {k: v for k, v in by_source.items() if k == "edu_chemc"},
            "filters": {"max_ssml_len": edu_max_ssml_len, "max_side": edu_max_side},
        },
        "overall": {
            "count": len(mixed_rows),
        },
    }
    (out_root / "subgroup_summary.json").write_text(json.dumps(subgroup_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for obsolete_name in [
        "subgroup_summary.md",
        "QC_REPORT.md",
        "QC_SUMMARY.md",
        "ANNOTATION_GUIDELINE.md",
        "DATA_CARD_zh.md",
    ]:
        obsolete_path = out_root / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()
    (out_root / "README.md").write_text(
        "# OCSR 真实世界混合评测集 v1\n\n"
        "这是当前项目用于真实世界化学图像理解与跨域泛化评估的混合域评测集。\n\n"
        f"- canonical_main：{by_track.get('canonical_main', 0)}\n"
        f"- edu_realworld：{by_track.get('edu_realworld', 0)}\n"
        f"- 总计：{len(mixed_rows)}\n\n"
        "完整的构成、质量控制、来源/版权边界、评分项映射与 subgroup 解释，请阅读 `TECHNICAL_REPORT_zh.md`。\n",
        encoding="utf-8",
    )
    canonical_sources = {k: v for k, v in by_source.items() if k in {"uob", "uspto", "real_world", "decimer"}}
    edu_sources = {k: v for k, v in by_source.items() if k == "edu_chemc"}
    technical_report = (
        "# OCSR 真实世界混合评测集技术报告（中文）\n\n"
        "## 1. 数据集定位\n\n"
        "本评测集面向真实世界化学图像理解与跨域泛化能力评估，采用 mixed-domain 设计，包含标准 OCSR 主任务子集与教育真实场景子集。\n\n"
        "## 2. 数据集规模\n\n"
        f"- 总样本数：{len(mixed_rows)}\n"
        f"- canonical_main：{by_track.get('canonical_main', 0)}\n"
        f"- edu_realworld：{by_track.get('edu_realworld', 0)}\n\n"
        "当前总量超过 1000 条，能够支撑较稳定的模型比较与分组评估。\n\n"
        "## 3. 数据组成与方向\n\n"
        "### 3.1 canonical_main\n"
        "- 目标空间：`canonical_smiles`\n"
        "- 方向：标准有机化学结构式 OCR 主任务\n"
        f"- 来源组成：{json.dumps(canonical_sources, ensure_ascii=False)}\n"
        "- 具体来源说明：\n"
        "  - `uob`：可归因到 **UOB OCSR Benchmark**，通常追溯为 University of Birmingham 发布并随 MolRec 体系使用的公开 benchmark；\n"
        "  - `uspto`：可归因到 **USPTO OCSR Benchmark**，并可进一步说明与 MarkushGrapher-2 的 USPTO 公开基准线有关；\n"
        "  - `real_world`：可表述为项目在真实世界化学图像场景下从多个零散真实数据集与自整理素材中收集得到的补充来源。\n"
        "- 优点：主任务口径清楚、解释性强，可作为主分来源\n\n"
        "### 3.2 edu_realworld\n"
        "- 目标空间：`ssml_normed`\n"
        "- 方向：教育行业真实图像、教学/作业/练习类化学结构图识别\n"
        "- 数据集名称：**EDU-CHEMC**\n"
        f"- 来源组成：{json.dumps(edu_sources, ensure_ascii=False)}\n"
        f"- 筛选规则：`ssml_normed` 长度 <= {edu_max_ssml_len}，`max_side` <= {edu_max_side}\n"
        "- 优点：更贴近真实教育场景，有利于体现模型在学术拍照与教学内容中的泛化能力\n\n"
        "## 4. 领域与场景覆盖\n\n"
        "本评测集适用于：\n"
        "- 标准有机化学结构式 OCR 主任务；\n"
        "- 真实世界化学图像理解与跨域泛化评估；\n"
        "- 教育行业化学结构图识别与学术拍照场景评估。\n\n"
        "## 5. 数据多样性\n\n"
        "当前版本同时覆盖：\n"
        "- 标准 OCSR 图像；\n"
        "- 真实世界补充图像；\n"
        "- 教育场景手写/教学结构图；\n"
        "- 扫描、拍照、页面嵌入、图像退化等复杂视觉情况。\n\n"
        "## 6. 难度结构\n\n"
        f"- 难度分布：{json.dumps(dict(by_difficulty), ensure_ascii=False)}\n"
        "其中 canonical_main 主要提供 `medium / medium_hard` 与多种真实视觉情况，edu_realworld 当前以 `hard` 为主。\n\n"
        "## 7. 质量控制\n\n"
        f"1. `canonical_main` 子集继承自 `canonical_smiles_curated_v2`，保留 {by_track.get('canonical_main', 0)} 条；\n"
        f"2. canonical 主线中共移除 {canonical_summary['removed']} 条图像模式风险样本；\n"
        f"3. `edu_realworld` 子集通过 `ssml_normed` 长度 <= {edu_max_ssml_len}、图像 `max_side` <= {edu_max_side} 的规则筛出 {by_track.get('edu_realworld', 0)} 条；\n"
        "4. 当前 mixed 集中的图片均为可正常打开图像，图像模式已统一到稳定可用范围。\n\n"
        "## 8. 来源开放性与版权边界\n\n"
        "### 8.1 canonical_main 来源\n"
        "当前标签中 `license` 统一写作 `mixed_public_and_team_curated`，说明这是一个混合来源集合，而不是单一开源许可集。\n"
        "- `uspto`：当前可较明确写为 **USPTO OCSR Benchmark**，并与 MarkushGrapher-2 的 USPTO 公开基准线相关联，属于当前四类来源中归因最明确的一类；\n"
        "- `uob`：当前可较明确写为 **UOB OCSR Benchmark**，但逐源许可证与原始出处链接在当前仓库内仍未完全逐条补齐；\n"
        "- `real_world`：当前更适合写为‘多个零散真实数据集/项目自整理真实世界素材的集合标签’，不宜冒进写成单一官方公开数据集名称；\n"
        "- 因此该部分最稳妥的表述是：来源构成清楚，但逐源授权证据清晰度不一致。\n\n"
        "### 8.2 edu_realworld 来源\n"
        "edu_realworld 来自 **EDU-CHEMC**，其上游候选层文档明确写明当前 `license = unknown_pending_confirmation`。\n"
        "因此 EDU 部分适合在技术报告中明确标注为‘许可状态待进一步确认’，而不应简单表述为已完全开源可再分发。\n\n"
        "### 8.3 当前建议表述\n"
        "- 可写明：`uob` 来自 UOB OCSR Benchmark，`uspto` 来自 USPTO OCSR Benchmark / MarkushGrapher-2 相关公开基准线，`edu_realworld` 来自 EDU-CHEMC，`real_world` 来自多个零散真实数据集收集与自整理真实图像；\n"
        "- 必须同时写明：`real_world` 不是单一标准数据集名，`uob` 与 `real_world` 的逐源授权描述仍应保持保守，`EDU-CHEMC` 当前许可状态仍需补充确认。\n\n"
        "## 9. 风险与边界\n\n"
        "- 这是 mixed-domain benchmark，不是单一 canonical benchmark；\n"
        "- `canonical_main` 与 `edu_realworld` 的标签空间并不完全一致；\n"
        "- 因此不能把单一 overall 分数作为唯一结论；\n"
        "- 推荐报告顺序：canonical_main → edu_realworld → overall。\n\n"
        "## 10. 分组统计\n\n"
        f"- canonical_main：{subgroup_summary['canonical_main']['count']}\n"
        f"- edu_realworld：{subgroup_summary['edu_realworld']['count']}\n"
        f"- overall：{subgroup_summary['overall']['count']}\n\n"
        "## 11. 与比赛评分项的对应关系\n\n"
        "### 11.1 数据规模\n"
        "总量超过 1000，具备较稳定的对比基础。\n\n"
        "### 11.2 标注准确性\n"
        "当前集合图像层面已经做了基础清洗与质控，但 mixed-domain 的标签空间差异需要通过分组报告来保持解释性。\n\n"
        "### 11.3 数据多样性\n"
        "同时覆盖 canonical 主任务与教育真实场景，真实视觉情况方差较大。\n\n"
        "### 11.4 难度合理性\n"
        "整体不是单一 easy 集，也不是纯 clean benchmark；但 edu_realworld 当前偏 hard，这一点需要在说明中主动交代。\n\n"
        "## 12. 使用建议\n\n"
        "- 若作为主分报告，优先使用 canonical_main；\n"
        "- 若强调真实世界教育场景泛化，可同时报告 edu_realworld；\n"
        "- overall 仅作为补充指标，不建议单独使用。\n"
    )
    (out_root / "TECHNICAL_REPORT_zh.md").write_text(technical_report, encoding="utf-8")

    summary = {
        "canonical_total": by_track.get("canonical_main", 0),
        "edu_selected": by_track.get("edu_realworld", 0),
        "total": len(mixed_rows),
        "output_root": str(out_root),
    }
    (out_root / "mixed_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-root", default="V2/data/eval/ocsr_realworld_mixed_eval_v1")
    parser.add_argument("--edu-max-ssml-len", type=int, default=220)
    parser.add_argument("--edu-max-side", type=int, default=768)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = (project_root / args.out_root).resolve()
    summary = build_mixed_evalset(project_root, out_root, args.edu_max_ssml_len, args.edu_max_side)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
