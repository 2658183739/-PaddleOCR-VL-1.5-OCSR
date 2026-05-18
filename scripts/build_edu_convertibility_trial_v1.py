from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


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


def is_likely_single_structure(text: str) -> bool:
    has_arrow = ("xrightarrow" in text) or ("\\rightarrow" in text) or ("->" in text)
    plus_count = text.count(" + ") + text.count("+ ") + text.count(" +")
    chemfig_count = text.count("\\chemfig")
    return not (has_arrow or plus_count >= 1 or chemfig_count > 1)


def select_candidates(labels_path: Path, max_ssml_len: int, max_side: int):
    root = labels_path.parent.parent
    selected = []
    total = 0
    for row in read_jsonl(labels_path):
        total += 1
        ssml = str(row.get("ssml_normed", ""))
        chemfig = str(row.get("chemfig", ssml))
        text = ssml if ssml else chemfig
        if not is_likely_single_structure(text):
            continue
        if len(ssml) > max_ssml_len:
            continue
        image_path = root / row["image"]
        with Image.open(image_path) as image:
            width, height = image.size
        if max(width, height) > max_side:
            continue
        selected.append(
            {
                **row,
                "image_size": [width, height],
                "ssml_len": len(ssml),
                "heuristic_bucket": "high_confidence_single_structure",
            }
        )
    return total, selected


def build_trial(project_root: Path, out_root: Path, max_ssml_len: int = 220, max_side: int = 768) -> dict[str, object]:
    labels_path = project_root / "V2" / "data" / "eval" / "edu_chmec_ssml_normed_test_v1" / "annotations" / "labels.jsonl"
    total, selected = select_candidates(labels_path, max_ssml_len, max_side)

    out_images = out_root / "images"
    out_annotations = out_root / "annotations"
    copied_rows = []
    source_root = labels_path.parent.parent
    for row in selected:
        src = source_root / row["image"]
        out_images.mkdir(parents=True, exist_ok=True)
        dest = out_images / Path(row["image"]).name
        if not dest.exists():
            shutil.copy2(src, dest)
        new_row = dict(row)
        new_row["image"] = f"images/{dest.name}"
        copied_rows.append(new_row)

    write_jsonl(out_annotations / "labels.jsonl", copied_rows)
    summary = {
        "total_input": total,
        "selected": len(copied_rows),
        "max_ssml_len": max_ssml_len,
        "max_side": max_side,
        "output_root": str(out_root),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-root", default="V2/data/eval/edu_chemc_convertibility_trial_v1")
    parser.add_argument("--max-ssml-len", type=int, default=220)
    parser.add_argument("--max-side", type=int, default=768)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = (project_root / args.out_root).resolve()
    summary = build_trial(project_root, out_root, args.max_ssml_len, args.max_side)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
