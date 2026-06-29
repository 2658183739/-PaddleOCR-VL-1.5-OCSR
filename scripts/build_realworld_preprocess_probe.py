import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


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


def add_border(image: Image.Image, border: int) -> Image.Image:
    return ImageOps.expand(image.convert("RGB"), border=border, fill=(255, 255, 255))


def crop_image(image: Image.Image, threshold: int, pad: int, border: int) -> Image.Image:
    bbox = content_bbox(image, threshold=threshold, pad=pad)
    return add_border(image.crop(bbox), border=border)


def variant_crop(image: Image.Image, threshold: int, pad: int, border: int) -> Image.Image:
    return crop_image(image, threshold=threshold, pad=pad, border=border)


def variant_crop_gray_auto(image: Image.Image, threshold: int, pad: int, border: int) -> Image.Image:
    cropped = crop_image(image, threshold=threshold, pad=pad, border=border)
    gray = ImageOps.grayscale(cropped)
    return ImageOps.autocontrast(gray).convert("RGB")


def variant_crop_gray_sharp(image: Image.Image, threshold: int, pad: int, border: int) -> Image.Image:
    gray_auto = variant_crop_gray_auto(image, threshold=threshold, pad=pad, border=border)
    out = ImageEnhance.Contrast(gray_auto).enhance(1.65)
    out = ImageEnhance.Sharpness(out).enhance(1.55)
    return out.convert("RGB")


def variant_crop_bw_thicken(image: Image.Image, threshold: int, pad: int, border: int) -> Image.Image:
    gray_auto = ImageOps.grayscale(variant_crop_gray_auto(image, threshold=threshold, pad=pad, border=border))
    bw = gray_auto.point(lambda value: 0 if value < 215 else 255)
    thick = bw.filter(ImageFilter.MinFilter(3))
    return thick.convert("RGB")


VARIANTS = {
    "crop": variant_crop,
    "crop_gray_auto": variant_crop_gray_auto,
    "crop_gray_sharp": variant_crop_gray_sharp,
    "crop_bw_thicken": variant_crop_bw_thicken,
}


def summarize(rows):
    return {
        "total": len(rows),
        "source": dict(Counter(str(row.get("source", "")) for row in rows)),
        "difficulty": dict(Counter(str(row.get("difficulty", "")) for row in rows)),
        "task_type": dict(Counter(str(row.get("task_type", "")) for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--output-root", default="V2-1/reports/preprocess_probe_v1")
    parser.add_argument("--variants", default="crop,crop_gray_auto,crop_gray_sharp,crop_bw_thicken")
    parser.add_argument("--threshold", type=int, default=245)
    parser.add_argument("--pad", type=int, default=12)
    parser.add_argument("--border", type=int, default=24)
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
        "output_root": output_root.relative_to(project_root).as_posix(),
        "variants": {},
    }

    for variant_name in selected_variants:
        transform = VARIANTS[variant_name]
        variant_root = output_root / variant_name
        image_dir = variant_root / "images"
        out_rows = []
        for row in rows:
            row_id = str(row.get("id"))
            image_path = resolve_image_path(get_image_ref(row), project_root, labels_path)
            image = Image.open(image_path).convert("RGB")
            out_image = transform(image, args.threshold, args.pad, args.border)
            out_rel = Path(args.output_root) / variant_name / "images" / f"{row_id}.png"
            out_abs = project_root / out_rel
            out_abs.parent.mkdir(parents=True, exist_ok=True)
            out_image.save(out_abs)
            new_row = dict(row)
            new_row["image"] = out_rel.as_posix()
            new_row.pop("image_path", None)
            new_row["preprocess_variant"] = variant_name
            out_rows.append(new_row)

        labels_out = variant_root / "annotations" / "labels.jsonl"
        write_jsonl(labels_out, out_rows)
        manifest["variants"][variant_name] = {
            "labels": labels_out.relative_to(project_root).as_posix(),
            "summary": summarize(out_rows),
        }

    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
