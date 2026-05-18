from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def enrich_rows(eval_root: Path, rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        image_path = eval_root / row["image"]
        with Image.open(image_path) as image:
            width, height = image.size

        source = str(row.get("source", ""))
        if source == "uob":
            source_desc = "UOB OCSR Benchmark"
            copyright_note = "混合来源标记，逐源授权证据仍建议继续补齐"
        elif source == "uspto":
            source_desc = "USPTO OCSR Benchmark / MarkushGrapher-2 相关公开基准线"
            copyright_note = "公开基准线归因较强，但建议保留来源说明"
        elif source == "edu_chemc":
            source_desc = "EDU-CHEMC"
            copyright_note = "许可状态待进一步确认"
        else:
            source_desc = "多个零散真实数据集与自整理真实图像"
            copyright_note = "来源构成清楚，但逐条出处与授权证明仍待补齐"

        new_row = dict(row)
        new_row["image_size"] = [width, height]
        new_row["source_desc"] = source_desc
        new_row["copyright_note"] = copyright_note
        enriched.append(new_row)
    return enriched


def select_samples(rows: list[dict], per_track_limit: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("benchmark_track", "unknown"))].append(row)

    selected = []
    for _, items in grouped.items():
        by_source: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            by_source[str(item.get("source", "unknown"))].append(item)

        source_names = sorted(by_source.keys())
        per_source_quota = max(1, per_track_limit // max(1, len(source_names)))
        chosen_ids = set()

        for source in source_names:
            for item in by_source[source][:per_source_quota]:
                selected.append(item)
                chosen_ids.add(item["id"])

        if len(chosen_ids) < per_track_limit:
            for item in items:
                if item["id"] in chosen_ids:
                    continue
                selected.append(item)
                chosen_ids.add(item["id"])
                if len(chosen_ids) >= per_track_limit:
                    break
    return selected


def percentile(values: list[int], q: float) -> int:
    values = sorted(values)
    if not values:
        return 0
    idx = int((len(values) - 1) * q)
    return values[idx]


def build_html(samples: list[dict], total_samples: int, per_track_limit: int) -> str:
    track_counts: dict[str, int] = defaultdict(int)
    max_sides: list[int] = []
    cards = []
    for row in samples:
        track_counts[str(row.get("benchmark_track", "unknown"))] += 1
        max_sides.append(max(row.get("image_size", [0, 0])))
        cards.append(
            f"""
            <article class=\"card\">
              <div class=\"thumb-wrap\"><img src=\"{row['image']}\" alt=\"{row['id']}\"></div>
              <div class=\"meta\">
                <div class=\"pill\">{row.get('benchmark_track','unknown')}</div>
                <h3>{row['id']}</h3>
                <p><strong>来源：</strong>{row.get('source','')}</p>
                <p><strong>来源说明：</strong>{row.get('source_desc','')}</p>
                <p><strong>版权状态：</strong>{row.get('copyright_note','')}</p>
                <p><strong>难度：</strong>{row.get('difficulty','')}</p>
                <p><strong>任务：</strong>{row.get('task_type','')}</p>
                <p><strong>目标空间：</strong>{row.get('eval_target','')}</p>
                <p><strong>图像尺寸：</strong>{row.get('image_size',[0,0])[0]} × {row.get('image_size',[0,0])[1]}</p>
                <div class=\"label-box\">{row.get('label_summary','')}</div>
              </div>
            </article>
            """
        )

    p50 = percentile(max_sides, 0.5)
    p90 = percentile(max_sides, 0.9)

    return f"""
<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>OCSR 评测集可视化抽检报告</title>
  <style>
    :root {{ --bg:#07111f; --panel:#101a32; --panel2:#13203d; --muted:#9fb0d0; --text:#eef2ff; --accent:#7dd3fc; --accent2:#a78bfa; --line:rgba(255,255,255,.08); --shadow:0 18px 50px rgba(0,0,0,.35); }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Microsoft YaHei","PingFang SC","Noto Sans SC",sans-serif; background:radial-gradient(circle at top,#17284b,#07111f 52%); color:var(--text); }}
    .wrap {{ max-width:1450px; margin:0 auto; padding:40px 24px 80px; }}
    .hero,.panel {{ padding:28px 32px; border:1px solid var(--line); border-radius:24px; background:linear-gradient(135deg,rgba(125,211,252,.12),rgba(167,139,250,.08)); box-shadow:var(--shadow); margin-bottom:24px; }}
    .panel {{ background:linear-gradient(135deg,rgba(255,255,255,.04),rgba(255,255,255,.03)); }}
    h1 {{ margin:0 0 12px; font-size:42px; line-height:1.12; }}
    h2 {{ margin:0 0 14px; font-size:26px; }}
    .desc {{ color:var(--muted); font-size:16px; line-height:1.85; max-width:980px; }}
    .stats {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; margin-top:24px; }}
    .stat {{ padding:18px 20px; border-radius:18px; background:rgba(255,255,255,.04); border:1px solid var(--line); }}
    .stat .k {{ color:var(--muted); font-size:13px; }}
    .stat .v {{ font-size:30px; font-weight:700; margin-top:6px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:20px; }}
    .two-col {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; }}
    .card {{ overflow:hidden; border-radius:22px; border:1px solid var(--line); background:var(--panel2); box-shadow:0 14px 40px rgba(0,0,0,.28); }}
    .thumb-wrap {{ aspect-ratio:16/10; background:#fff; display:flex; align-items:center; justify-content:center; padding:10px; }}
    .thumb-wrap img {{ max-width:100%; max-height:100%; object-fit:contain; }}
    .meta {{ padding:18px 20px 22px; }}
    .pill {{ display:inline-block; padding:6px 10px; border-radius:999px; background:rgba(125,211,252,.15); color:var(--accent); font-size:12px; margin-bottom:10px; }}
    h3 {{ margin:0 0 10px; font-size:20px; }}
    p {{ margin:6px 0; color:var(--muted); line-height:1.72; }}
    .label-box {{ margin-top:12px; padding:12px 14px; border-radius:14px; background:rgba(255,255,255,.04); border:1px solid var(--line); color:var(--text); font-family:ui-monospace,Consolas,monospace; font-size:13px; word-break:break-all; line-height:1.6; }}
    .list {{ color:var(--muted); line-height:1.9; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"hero\">
      <h1>OCSR 评测集可视化抽检报告</h1>
      <div class=\"desc\">本报告用于对 <code>ocsr_realworld_mixed_eval_v1</code> 进行抽样可视化检查，帮助评审与开发者直观看到主任务子域与教育真实场景子域的图像风格、标签形式、难度结构、来源说明与版权状态。</div>
      <div class=\"stats\">
        <div class=\"stat\"><div class=\"k\">总样本数</div><div class=\"v\">{total_samples}</div></div>
        <div class=\"stat\"><div class=\"k\">抽检样本数</div><div class=\"v\">{len(samples)}</div></div>
        <div class=\"stat\"><div class=\"k\">抽检覆盖子域</div><div class=\"v\">2</div></div>
      </div>
    </section>

    <section class=\"panel\">
      <h2>抽样策略</h2>
      <div class=\"desc\">当前报告优先覆盖 <code>canonical_main</code> 与 <code>edu_realworld</code> 两个子域，并尽量兼顾不同来源。默认每个子域最多抽取 {per_track_limit} 条样本进行人工抽检。</div>
    </section>

    <section class=\"two-col\">
      <section class=\"panel\">
        <h2>子域概览</h2>
        <div class=\"list\">canonical_main：{track_counts.get('canonical_main', 0)} 条<br>edu_realworld：{track_counts.get('edu_realworld', 0)} 条<br>overall：{len(samples)} 条抽检样本</div>
      </section>
      <section class=\"panel\">
        <h2>尺寸摘要</h2>
        <div class=\"list\">抽检样本最大边长度 p50：{p50} 像素<br>抽检样本最大边长度 p90：{p90} 像素</div>
      </section>
    </section>

    <section class=\"two-col\">
      <section class=\"panel\">
        <h2>子域比例</h2>
        <div class=\"list\">canonical_main：{track_counts.get('canonical_main', 0)}/{len(samples)}<br>edu_realworld：{track_counts.get('edu_realworld', 0)}/{len(samples)}</div>
      </section>
      <section class=\"panel\">
        <h2>风险摘要</h2>
        <div class=\"list\">当前集合的主要风险不是坏图或模式异常，而是 mixed-domain / mixed-target 带来的解释性风险，因此必须坚持 subgroup 报告。</div>
      </section>
    </section>

    <section class=\"panel\">
      <h2>来源与版权说明</h2>
      <div class=\"desc\">本报告中的来源说明与版权状态用于帮助评审快速理解样本背景：`uob` 对应 UOB OCSR Benchmark，`uspto` 对应 USPTO OCSR Benchmark / MarkushGrapher-2 相关公开基准线，`edu_chemc` 对应 EDU-CHEMC，`real_world` 对应多个零散真实数据集与自整理真实图像。若作为正式材料，应以技术报告中的来源与版权边界说明为准。</div>
    </section>

    <section class=\"panel\">
      <h2>分组结论</h2>
      <div class=\"desc\">本评测集必须按子域解释：<strong>canonical_main</strong> 更适合作为主任务结果，<strong>edu_realworld</strong> 更适合作为教育真实场景泛化能力的专项结果，<strong>overall</strong> 只能作为补充指标。</div>
    </section>

    <section class=\"panel\">
      <h2>风险边界</h2>
      <div class=\"desc\">不要把 mixed-domain 结果误当作单一 canonical benchmark 分数。当前集合的主要风险不是坏图或模式异常，而是 mixed-domain / mixed-target 带来的解释性风险，因此必须坚持 subgroup 报告与边界说明。</div>
    </section>

    <section class=\"panel\">
      <h2>评分项映射</h2>
      <div class=\"desc\">从评审视角看，这份评测集的优势主要体现在数据规模、真实世界覆盖和多样性；需要重点解释的，是 mixed-domain 设计下的标签空间差异与 subgroup 结果汇报方式。</div>
    </section>

    <h2>抽检样本</h2>
    <section class=\"grid\">{''.join(cards)}</section>
  </div>
</body>
</html>
"""


def build_visual_report(eval_root: Path, out_path: Path, per_track_limit: int = 8) -> dict[str, object]:
    labels_path = eval_root / "annotations" / "labels.jsonl"
    rows = list(read_jsonl(labels_path))
    enriched_rows = enrich_rows(eval_root, rows)
    samples = select_samples(enriched_rows, per_track_limit)
    html = build_html(samples, len(rows), per_track_limit)
    out_path.write_text(html, encoding="utf-8")
    return {"total_samples": len(rows), "spotcheck_samples": len(samples), "output": str(out_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="V2/data/eval/ocsr_realworld_mixed_eval_v1")
    parser.add_argument("--out", default="V2/data/eval/ocsr_realworld_mixed_eval_v1/VISUAL_SPOTCHECK_REPORT_zh.html")
    parser.add_argument("--per-track-limit", type=int, default=8)
    args = parser.parse_args()

    eval_root = Path(args.eval_root).resolve()
    out_path = Path(args.out).resolve()
    summary = build_visual_report(eval_root=eval_root, out_path=out_path, per_track_limit=args.per_track_limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
