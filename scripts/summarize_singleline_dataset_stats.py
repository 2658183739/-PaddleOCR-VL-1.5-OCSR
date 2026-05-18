import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def assistant_text(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def percentile(values, ratio: float):
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return ordered[index]


def resolve_image(jsonl_path: Path, image_value: str) -> Path:
    raw = Path(str(image_value))
    if raw.is_absolute():
        return raw
    return (jsonl_path.parent / raw).resolve()


def summarize(path: Path):
    widths = []
    heights = []
    areas = []
    smiles_lengths = []
    aspect_bins = Counter()
    image_size_cache = {}

    for record in read_jsonl(path):
        smiles_lengths.append(len(assistant_text(record)))
        image_path = resolve_image(path, record["images"][0])
        if image_path not in image_size_cache:
            with Image.open(image_path) as image:
                image_size_cache[image_path] = image.size
        width, height = image_size_cache[image_path]
        widths.append(width)
        heights.append(height)
        areas.append(width * height)
        ratio = width / max(1, height)
        if ratio >= 3:
            aspect_bins["very_wide"] += 1
        elif ratio >= 1.5:
            aspect_bins["wide"] += 1
        elif ratio >= 0.67:
            aspect_bins["balanced"] += 1
        else:
            aspect_bins["tall"] += 1

    return {
        "path": str(path),
        "total": len(smiles_lengths),
        "unique_images": len(image_size_cache),
        "smiles_length": {
            "min": min(smiles_lengths) if smiles_lengths else 0,
            "p50": percentile(smiles_lengths, 0.50),
            "p90": percentile(smiles_lengths, 0.90),
            "p95": percentile(smiles_lengths, 0.95),
            "p99": percentile(smiles_lengths, 0.99),
            "max": max(smiles_lengths) if smiles_lengths else 0,
        },
        "image_width": {
            "min": min(widths) if widths else 0,
            "p50": percentile(widths, 0.50),
            "p90": percentile(widths, 0.90),
            "p95": percentile(widths, 0.95),
            "p99": percentile(widths, 0.99),
            "max": max(widths) if widths else 0,
        },
        "image_height": {
            "min": min(heights) if heights else 0,
            "p50": percentile(heights, 0.50),
            "p90": percentile(heights, 0.90),
            "p95": percentile(heights, 0.95),
            "p99": percentile(heights, 0.99),
            "max": max(heights) if heights else 0,
        },
        "image_area": {
            "min": min(areas) if areas else 0,
            "p50": percentile(areas, 0.50),
            "p90": percentile(areas, 0.90),
            "p95": percentile(areas, 0.95),
            "p99": percentile(areas, 0.99),
            "max": max(areas) if areas else 0,
        },
        "aspect_bins": dict(aspect_bins),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--input", default="V2/data/sft_materialized/train_singleline_rw_messages.jsonl")
    parser.add_argument("--report", default="V2/reports/singleline_rw_dataset_stats.json")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_path = (project_root / args.input).resolve()
    report_path = (project_root / args.report).resolve()
    report = summarize(input_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
