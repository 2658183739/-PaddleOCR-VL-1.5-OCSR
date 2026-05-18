from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


PROMPT = "OCR: Output only the canonical SMILES string for the molecule shown in the image."
VALID_BUCKETS = {
    "auto_photo_scan",
    "auto_document_context",
    "auto_exam_context",
    "auto_handdrawn_like",
    "auto_long_stereo",
}


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


def try_load_rdkit():
    try:
        from rdkit import Chem

        return Chem
    except Exception:
        return None


def canonicalize(Chem, smiles: str) -> str | None:
    text = str(smiles or "").strip()
    if not text:
        return None
    if Chem is None:
        return text
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def assistant_text(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def resolve_image(jsonl_path: Path, image_value: str) -> Path:
    raw = Path(str(image_value))
    if raw.is_absolute():
        return raw
    return (jsonl_path.parent / raw).resolve()


def load_eval_smiles(paths: list[Path], Chem) -> set[str]:
    result = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            gt = row.get("ground_truth")
            raw = ""
            if isinstance(gt, dict):
                raw = gt.get("smiles") or ""
            raw = raw or row.get("canonical_smiles") or row.get("smiles") or row.get("label_summary") or ""
            canonical = canonicalize(Chem, raw)
            if canonical:
                result.add(canonical)
    return result


def stable_seed(text: str, base_seed: int) -> int:
    total = base_seed
    for index, char in enumerate(text):
        total += (index + 1) * ord(char)
    return total


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=max(20, min(95, quality)))
    buf.seek(0)
    with Image.open(buf) as reopened:
        return reopened.convert("RGB")


def add_noise(image: Image.Image, rng: random.Random, amount: float) -> Image.Image:
    noise = Image.effect_noise(image.size, sigma=max(2.0, amount * 40.0)).convert("L")
    noise = ImageEnhance.Contrast(noise).enhance(1.2)
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    return Image.blend(image, noise_rgb, max(0.03, min(0.22, amount)))


def add_shadow(image: Image.Image, rng: random.Random) -> Image.Image:
    overlay = Image.new("L", image.size, 255)
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    x0 = rng.randint(-width // 3, width // 2)
    y0 = rng.randint(-height // 4, height // 2)
    x1 = x0 + rng.randint(width // 3, width)
    y1 = y0 + rng.randint(height // 4, height)
    shade = rng.randint(120, 210)
    draw.ellipse((x0, y0, x1, y1), fill=shade)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(12, min(width, height) // 10)))
    shadow = Image.merge("RGB", (overlay, overlay, overlay))
    return ImageChops.multiply(image, shadow)


def rotate_expand(image: Image.Image, rng: random.Random, degrees: float, fill: tuple[int, int, int]) -> Image.Image:
    angle = rng.uniform(-degrees, degrees)
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)


def pad_canvas(image: Image.Image, fill: tuple[int, int, int], pad_x: int, pad_y: int) -> Image.Image:
    canvas = Image.new("RGB", (image.width + 2 * pad_x, image.height + 2 * pad_y), fill)
    canvas.paste(image, (pad_x, pad_y))
    return canvas


def draw_context_lines(image: Image.Image, rng: random.Random, line_count: int, margin: int, gray_min: int = 80, gray_max: int = 180):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(line_count):
        y = rng.randint(margin, max(margin, height - margin))
        x0 = rng.randint(margin // 2, max(margin // 2, width // 6))
        x1 = rng.randint(max(x0 + 20, width // 2), max(x0 + 20, width - margin // 2))
        color = rng.randint(gray_min, gray_max)
        thickness = rng.randint(1, 2)
        draw.line((x0, y, x1, y), fill=(color, color, color), width=thickness)


def document_context(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    base = ImageEnhance.Color(image).enhance(0.9)
    base = rotate_expand(base, rng, degrees=2.4, fill=(247, 245, 240))
    canvas_w = int(base.width * rng.uniform(1.2, 1.6))
    canvas_h = int(base.height * rng.uniform(1.2, 1.8))
    canvas = Image.new("RGB", (canvas_w, canvas_h), (249, 248, 243))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas_w - 1, canvas_h - 1), outline=(220, 218, 210), width=1)
    draw_context_lines(canvas, rng, line_count=rng.randint(12, 26), margin=18)
    x = rng.randint(20, max(20, canvas_w - base.width - 20))
    y = rng.randint(20, max(20, canvas_h - base.height - 20))
    canvas.paste(base, (x, y))
    if rng.random() < 0.65:
        draw.rectangle((x - 3, y - 3, x + base.width + 3, y + base.height + 3), outline=(150, 150, 150), width=1)
    canvas = add_noise(canvas, rng, amount=0.08)
    return ImageEnhance.Sharpness(canvas).enhance(0.9)


def exam_context(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    base = rotate_expand(image, rng, degrees=1.8, fill=(252, 250, 244))
    canvas_w = int(base.width * rng.uniform(1.25, 1.7))
    canvas_h = int(base.height * rng.uniform(1.3, 1.8))
    canvas = Image.new("RGB", (canvas_w, canvas_h), (252, 250, 244))
    draw = ImageDraw.Draw(canvas)
    top = rng.randint(18, 36)
    for idx in range(rng.randint(4, 8)):
        y = top + idx * rng.randint(14, 20)
        draw.line((20, y, canvas_w - 20, y), fill=(135, 135, 135), width=1)
    x = rng.randint(28, max(28, canvas_w - base.width - 28))
    y = rng.randint(canvas_h // 4, max(canvas_h // 4, canvas_h - base.height - 24))
    canvas.paste(base, (x, y))
    if rng.random() < 0.55:
        draw.rectangle((x - 4, y - 4, x + base.width + 4, y + base.height + 4), outline=(115, 115, 115), width=1)
    draw.text((18, 12), f"({rng.randint(1, 9)})", fill=(60, 60, 60))
    draw.text((20, canvas_h - 24), rng.choice(["A.", "B.", "C.", "D."]), fill=(70, 70, 70))
    canvas = add_noise(canvas, rng, amount=0.06)
    return ImageEnhance.Contrast(canvas).enhance(rng.uniform(0.92, 1.05))


def photo_scan_context(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    base = rotate_expand(image, rng, degrees=4.0, fill=(246, 246, 246))
    base = pad_canvas(base, (247, 247, 247), rng.randint(8, 32), rng.randint(8, 32))
    if rng.random() < 0.75:
        base = add_shadow(base, rng)
    if rng.random() < 0.7:
        base = jpeg_roundtrip(base, quality=rng.randint(28, 60))
    if rng.random() < 0.8:
        base = base.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.1)))
    base = ImageEnhance.Contrast(base).enhance(rng.uniform(0.75, 1.1))
    base = ImageEnhance.Brightness(base).enhance(rng.uniform(0.82, 1.05))
    base = add_noise(base, rng, amount=0.05)
    return base


def handwritten_like(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("L")
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.3))
    image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
    paper = Image.new("L", image.size, rng.randint(230, 248))
    image = ImageChops.multiply(ImageOps.invert(ImageChops.subtract(ImageOps.invert(paper), image)), paper)
    image = Image.merge("RGB", (image, image, image))
    image = rotate_expand(image, rng, degrees=3.0, fill=(245, 244, 240))
    image = add_noise(image, rng, amount=0.07)
    return image


def long_stereo_context(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = rng.uniform(1.08, 1.28)
    image = image.resize((max(64, int(image.width * scale)), max(64, int(image.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (int(image.width * rng.uniform(1.05, 1.25)), int(image.height * rng.uniform(1.15, 1.45))), (248, 248, 246))
    x = max(0, (canvas.width - image.width) // 2)
    y = max(0, (canvas.height - image.height) // 2)
    canvas.paste(image, (x, y))
    canvas = photo_scan_context(canvas, rng)
    return canvas


def build_image(bucket: str, src_path: Path, seed_value: int) -> Image.Image:
    rng = random.Random(seed_value)
    with Image.open(src_path) as image:
        if bucket == "auto_document_context":
            return document_context(image, rng)
        if bucket == "auto_exam_context":
            return exam_context(image, rng)
        if bucket == "auto_handdrawn_like":
            return handwritten_like(image, rng)
        if bucket == "auto_long_stereo":
            return long_stereo_context(image, rng)
        return photo_scan_context(image, rng)


def assign_bucket(meta: dict, smiles: str) -> str | None:
    source = str(meta.get("source", ""))
    difficulty = str(meta.get("difficulty", ""))
    if source == "decimer" or difficulty == "handwritten":
        return "auto_handdrawn_like"
    if difficulty in {"chinese_exam"}:
        return "auto_exam_context"
    if difficulty in {"document_embed", "page_level", "journal_fig", "multi_grid"}:
        return "auto_document_context"
    if len(smiles) >= 100 or "@" in smiles or "/" in smiles or "\\" in smiles:
        return "auto_long_stereo"
    if difficulty in {"photo", "scan", "degraded_scan"} or source == "real_world":
        return "auto_photo_scan"
    return None


def candidate_record(jsonl_path: Path, record: dict, eval_smiles: set[str], Chem):
    smiles = assistant_text(record)
    canonical = canonicalize(Chem, smiles)
    if not canonical:
        return None
    if canonical in eval_smiles:
        return None
    meta = record.get("meta", {})
    bucket = assign_bucket(meta, canonical)
    if bucket not in VALID_BUCKETS:
        return None
    images = record.get("images") or []
    if not images:
        return None
    src_path = resolve_image(jsonl_path, images[0])
    if not src_path.exists():
        return None
    return {
        "record": record,
        "bucket": bucket,
        "canonical_smiles": canonical,
        "src_path": src_path,
        "meta": meta,
    }


def build_candidates(train_jsonl: Path, eval_smiles: set[str], Chem):
    dedup = {}
    for record in read_jsonl(train_jsonl):
        candidate = candidate_record(train_jsonl, record, eval_smiles, Chem)
        if candidate is None:
            continue
        key = str(candidate["meta"].get("id") or candidate["src_path"].name)
        if key not in dedup:
            dedup[key] = candidate
    grouped = defaultdict(list)
    for candidate in dedup.values():
        grouped[candidate["bucket"]].append(candidate)
    return grouped


def rel_image(output_jsonl: Path, image_path: Path) -> str:
    return Path(os.path.relpath(image_path.resolve(), output_jsonl.parent.resolve())).as_posix()


def generate_records(
    grouped: dict[str, list[dict]],
    output_jsonl: Path,
    output_assets: Path,
    seed: int,
    per_bucket_caps: dict[str, int],
):
    rows = []
    stats = Counter()
    for bucket, candidates in grouped.items():
        cap = per_bucket_caps.get(bucket, 0)
        if cap <= 0:
            continue
        rng = random.Random(seed + len(bucket) * 17)
        rng.shuffle(candidates)
        selected = candidates[: min(cap, len(candidates))]
        for index, item in enumerate(selected):
            base_id = str(item["meta"].get("id") or item["src_path"].stem)
            out_id = f"{bucket}__{base_id}"
            bucket_dir = output_assets / bucket
            bucket_dir.mkdir(parents=True, exist_ok=True)
            out_path = bucket_dir / f"{out_id}.png"
            image = build_image(bucket, item["src_path"], stable_seed(out_id, seed))
            image.save(out_path, format="PNG", optimize=True)
            meta = dict(item["meta"])
            meta["source"] = f"auto_weak_{meta.get('source', 'unknown')}"
            meta["difficulty"] = meta.get("difficulty", "hard")
            meta["weak_domain"] = bucket
            meta["auto_aug_from_id"] = base_id
            meta["auto_aug_bucket"] = bucket
            meta["canonical_smiles_length"] = len(item["canonical_smiles"])
            meta["contains_stereo"] = "@" in item["canonical_smiles"]
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": f"<image>{PROMPT}"},
                        {"role": "assistant", "content": item["canonical_smiles"]},
                    ],
                    "images": [rel_image(output_jsonl, out_path)],
                    "meta": meta,
                }
            )
            stats[bucket] += 1
    return rows, dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train", default="V2-1/data/sft_materialized/train_singleline_rw_messages.jsonl")
    parser.add_argument("--output", default="V2-1/data/sft_materialized/train_weak_domain_auto_messages.jsonl")
    parser.add_argument("--assets-root", default="V2-1/data/assets/weak_domain_auto_v1")
    parser.add_argument("--report", default="V2-1/reports/weak_domain_auto_generation_report.json")
    parser.add_argument(
        "--eval-labels",
        nargs="*",
        default=[
            "V2-1/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl",
            "V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl",
            "V2-1/data/eval/weak_domain_v2/annotations/labels.jsonl",
        ],
    )
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--photo-scan-cap", type=int, default=320)
    parser.add_argument("--document-cap", type=int, default=220)
    parser.add_argument("--exam-cap", type=int, default=200)
    parser.add_argument("--handdrawn-cap", type=int, default=180)
    parser.add_argument("--long-stereo-cap", type=int, default=140)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    train_jsonl = (project_root / args.train).resolve()
    output_jsonl = (project_root / args.output).resolve()
    assets_root = (project_root / args.assets_root).resolve()
    report_path = (project_root / args.report).resolve()
    Chem = try_load_rdkit()
    eval_smiles = load_eval_smiles([(project_root / path).resolve() for path in args.eval_labels], Chem)
    grouped = build_candidates(train_jsonl, eval_smiles, Chem)
    caps = {
        "auto_photo_scan": args.photo_scan_cap,
        "auto_document_context": args.document_cap,
        "auto_exam_context": args.exam_cap,
        "auto_handdrawn_like": args.handdrawn_cap,
        "auto_long_stereo": args.long_stereo_cap,
    }
    rows, generated_counts = generate_records(
        grouped=grouped,
        output_jsonl=output_jsonl,
        output_assets=assets_root,
        seed=args.seed,
        per_bucket_caps=caps,
    )
    write_jsonl(output_jsonl, rows)
    candidate_counts = {bucket: len(items) for bucket, items in grouped.items()}
    report = {
        "train_jsonl": str(train_jsonl),
        "output_jsonl": str(output_jsonl),
        "assets_root": str(assets_root),
        "rdkit_available": Chem is not None,
        "eval_smiles_filter_count": len(eval_smiles),
        "candidate_counts": candidate_counts,
        "generated_counts": generated_counts,
        "total_generated": len(rows),
        "bucket_caps": caps,
        "notes": [
            "These are automatic weak-domain replay samples generated from existing training images.",
            "They are intended to reduce manual collection pressure while increasing domain variety.",
            "Use together with any manually collected weak-domain pool for V2-2.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
