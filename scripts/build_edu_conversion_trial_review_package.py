from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
        writer.writerow([
            "id",
            "image",
            "ssml_len",
            "image_width",
            "image_height",
            "heuristic_bucket",
            "predicted_smiles",
            "rdkit_valid",
            "manual_status",
            "notes",
        ])
        for row in records:
            writer.writerow([
                row["id"],
                row["image"],
                row.get("ssml_len", ""),
                row.get("image_size", ["", ""])[0],
                row.get("image_size", ["", ""])[1],
                row.get("heuristic_bucket", ""),
                row.get("predicted_smiles", ""),
                row.get("rdkit_valid", ""),
                row.get("manual_status", ""),
                row.get("notes", ""),
            ])


def build_review_package(trial_root: Path, out_root: Path) -> dict[str, object]:
    labels_path = trial_root / "annotations" / "labels.jsonl"
    rows = list(read_jsonl(labels_path))
    out_root.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_root / "labels.jsonl", rows)

    review_rows = []
    for row in rows:
        review_rows.append(
            {
                "id": row["id"],
                "image": row["image"],
                "ssml_len": row.get("ssml_len", ""),
                "image_size": row.get("image_size", ["", ""]),
                "heuristic_bucket": row.get("heuristic_bucket", ""),
                "predicted_smiles": "",
                "rdkit_valid": "",
                "manual_status": "",
                "notes": "",
            }
        )

    write_jsonl(out_root / "review_template.jsonl", review_rows)
    write_csv(out_root / "review_template.csv", review_rows)

    (out_root / "README_zh.md").write_text(
        "# EDU 可转化试点评估包\n\n"
        "本目录用于对 `edu_chemc_convertibility_trial_v1` 中的高信心候选样本做 conversion 试点复核。\n\n"
        "包含内容：\n"
        "- `labels.jsonl`：候选样本原始清单\n"
        "- `review_template.jsonl`：用于填写自动转换结果与人工复核结论\n"
        "- `review_template.csv`：便于人工查看与批注的表格版本\n\n"
        "建议记录字段：\n"
        "- predicted_smiles\n"
        "- rdkit_valid\n"
        "- manual_status（pass / review / fail）\n"
        "- notes\n",
        encoding="utf-8",
    )

    return {"total_candidates": len(rows), "output_root": str(out_root)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-root", default="V2/data/eval/edu_chemc_convertibility_trial_v1")
    parser.add_argument("--out-root", default="V2/data/eval/edu_chemc_convertibility_trial_v1/review_package")
    args = parser.parse_args()

    trial_root = Path(args.trial_root).resolve()
    out_root = Path(args.out_root).resolve()
    summary = build_review_package(trial_root=trial_root, out_root=out_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
