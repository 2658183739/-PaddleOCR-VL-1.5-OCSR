import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


EXAM_Q1_BOX = (0.13, 0.09, 0.385, 0.235)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_image_ref(row: dict) -> str:
    if str(row.get("image", "")).strip():
        return str(row["image"])
    if str(row.get("image_path", "")).strip():
        return str(row["image_path"])
    raise KeyError(f"missing image field for row {row.get('id')}")


def resolve_image_path(raw_ref: str, project_root: Path, labels_path: Path) -> Path:
    raw = Path(raw_ref)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                project_root / raw,
                labels_path.parent / raw,
                labels_path.parent.parent / raw,
                labels_path.parent.parent.parent / raw,
                raw,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(raw_ref)


def scaled_box(image: Image.Image, rel_box):
    left, top, right, bottom = rel_box
    return (
        max(0, int(round(left * image.width))),
        max(0, int(round(top * image.height))),
        min(image.width, int(round(right * image.width))),
        min(image.height, int(round(bottom * image.height))),
    )


def add_border(image: Image.Image, border: int) -> Image.Image:
    return ImageOps.expand(image.convert("RGB"), border=border, fill=(255, 255, 255))


def content_bbox(image: Image.Image, threshold: int, pad: int):
    gray = ImageOps.grayscale(image)
    mask = gray.point(lambda value: 255 if value < threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return (0, 0, image.width, image.height)
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(image.width, right + pad)
    bottom = min(image.height, bottom + pad)
    if right <= left or bottom <= top:
        return (0, 0, image.width, image.height)
    return (left, top, right, bottom)


def is_chinese_exam(row: dict) -> bool:
    keys = [
        row.get("id", ""),
        row.get("task_type", ""),
        row.get("difficulty", ""),
        row.get("image_type", ""),
    ]
    return any("chinese_exam" in str(value) for value in keys)


def crop_original_or_exam_q1(image: Image.Image, row: dict, border: int) -> Image.Image:
    if is_chinese_exam(row):
        return add_border(image.crop(scaled_box(image, EXAM_Q1_BOX)), border=border)
    return add_border(image, border=border)


def crop_trimmed(image: Image.Image, threshold: int, pad: int, border: int) -> Image.Image:
    return add_border(image.crop(content_bbox(image, threshold=threshold, pad=pad)), border=border)


def variant_exam_q1_panel(image: Image.Image, row: dict, threshold: int, pad: int, border: int) -> Image.Image:
    return crop_original_or_exam_q1(image, row=row, border=border)


def variant_exam_q1_trim(image: Image.Image, row: dict, threshold: int, pad: int, border: int) -> Image.Image:
    panel = crop_original_or_exam_q1(image, row=row, border=0)
    return crop_trimmed(panel, threshold=threshold, pad=pad, border=border)


def variant_exam_q1_trim_gray(image: Image.Image, row: dict, threshold: int, pad: int, border: int) -> Image.Image:
    trimmed = variant_exam_q1_trim(image, row=row, threshold=threshold, pad=pad, border=border)
    gray = ImageOps.autocontrast(ImageOps.grayscale(trimmed))
    out = ImageEnhance.Contrast(gray).enhance(1.35)
    out = ImageEnhance.Sharpness(out).enhance(1.25)
    return out.convert("RGB")


def variant_exam_q1_trim_else_original(
    image: Image.Image, row: dict, threshold: int, pad: int, border: int
) -> Image.Image:
    if not is_chinese_exam(row):
        return image.convert("RGB")
    return variant_exam_q1_trim(image, row=row, threshold=threshold, pad=pad, border=border)


VARIANTS = {
    "exam_q1_panel": variant_exam_q1_panel,
    "exam_q1_trim": variant_exam_q1_trim,
    "exam_q1_trim_gray": variant_exam_q1_trim_gray,
    "exam_q1_trim_else_original": variant_exam_q1_trim_else_original,
}


def portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def summarize(rows):
    return {
        "total": len(rows),
        "source": dict(Counter(str(row.get("source", "")) for row in rows)),
        "difficulty": dict(Counter(str(row.get("difficulty", "")) for row in rows)),
        "task_type": dict(Counter(str(row.get("task_type", "")) for row in rows)),
        "preprocess_variant": dict(Counter(str(row.get("preprocess_variant", "")) for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--output-root", default="V2-1/reports/realworld_region_crop_probe_v1")
    parser.add_argument("--variants", default="exam_q1_panel,exam_q1_trim,exam_q1_trim_gray")
    parser.add_argument("--threshold", type=int, default=245)
    parser.add_argument("--pad", type=int, default=18)
    parser.add_argument("--border", type=int, default=48)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    labels_path = (project_root / args.labels_jsonl).resolve()
    output_root = project_root / args.output_root
    selected_variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in selected_variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available: {sorted(VARIANTS)}")

    rows = list(read_jsonl(labels_path))
    if args.limit > 0:
        rows = rows[: args.limit]

    manifest = {
        "source_labels": str(labels_path),
        "output_root": portable_path(output_root, project_root),
        "exam_q1_box_relative": EXAM_Q1_BOX,
        "variants": {},
    }

    for variant_name in selected_variants:
        transform = VARIANTS[variant_name]
        variant_root = output_root / variant_name
        out_rows = []
        for row in rows:
            row_id = str(row.get("id"))
            image_path = resolve_image_path(get_image_ref(row), project_root, labels_path)
            image = Image.open(image_path).convert("RGB")
            out_image = transform(
                image,
                row=row,
                threshold=args.threshold,
                pad=args.pad,
                border=args.border,
            )
            out_rel = Path(args.output_root) / variant_name / "images" / f"{row_id}.png"
            out_abs = project_root / out_rel
            out_abs.parent.mkdir(parents=True, exist_ok=True)
            out_image.save(out_abs)
            new_row = dict(row)
            new_row["image"] = out_rel.as_posix()
            new_row.pop("image_path", None)
            new_row["preprocess_variant"] = variant_name
            if is_chinese_exam(row):
                new_row["preprocess_note"] = "crop first question region from Chinese exam page"
            out_rows.append(new_row)

        labels_out = variant_root / "annotations" / "labels.jsonl"
        write_jsonl(labels_out, out_rows)
        manifest["variants"][variant_name] = {
            "labels": portable_path(labels_out, project_root),
            "summary": summarize(out_rows),
        }

    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
