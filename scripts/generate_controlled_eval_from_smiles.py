from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


VALID_STYLES = {
    "print_page",
    "exam_page",
    "photo_sim",
    "scan_sim",
}


def try_load_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        return Chem, Draw
    except Exception:
        return None, None


def canonicalize(Chem, smiles: str) -> str | None:
    text = str(smiles or "").strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def read_rows(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if line:
                    yield line_no, json.loads(line)
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            yield line_no, row


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_seed(text: str, base_seed: int) -> int:
    total = base_seed
    for index, ch in enumerate(text):
        total += (index + 1) * ord(ch)
    return total


def render_molecule(Draw, Chem, smiles: str, size: tuple[int, int]) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    img = Draw.MolToImage(mol, size=size)
    return img.convert("RGB")


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=max(20, min(95, quality)))
    buf.seek(0)
    with Image.open(buf) as reopened:
        return reopened.convert("RGB")


def add_noise(image: Image.Image, amount: float) -> Image.Image:
    noise = Image.effect_noise(image.size, sigma=max(2.0, amount * 40.0)).convert("L")
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    return Image.blend(image, noise_rgb, max(0.02, min(0.2, amount)))


def add_shadow(image: Image.Image, rng: random.Random) -> Image.Image:
    overlay = Image.new("L", image.size, 255)
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    x0 = rng.randint(-width // 3, width // 2)
    y0 = rng.randint(-height // 4, height // 2)
    x1 = x0 + rng.randint(width // 3, width)
    y1 = y0 + rng.randint(height // 4, height)
    draw.ellipse((x0, y0, x1, y1), fill=rng.randint(120, 215))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(12, min(width, height) // 10)))
    shadow = Image.merge("RGB", (overlay, overlay, overlay))
    return ImageChops.multiply(image, shadow)


def rotate_expand(image: Image.Image, rng: random.Random, degrees: float, fill: tuple[int, int, int]) -> Image.Image:
    return image.rotate(rng.uniform(-degrees, degrees), resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)


def draw_rule_lines(draw: ImageDraw.ImageDraw, width: int, height: int, rng: random.Random, count: int):
    for _ in range(count):
        y = rng.randint(15, max(15, height - 15))
        x0 = rng.randint(10, max(10, width // 5))
        x1 = rng.randint(max(x0 + 20, width // 2), max(x0 + 20, width - 15))
        gray = rng.randint(90, 175)
        draw.line((x0, y, x1, y), fill=(gray, gray, gray), width=rng.randint(1, 2))


def make_print_page(mol_image: Image.Image, rng: random.Random) -> Image.Image:
    mol_image = rotate_expand(mol_image, rng, 1.4, (250, 249, 245))
    canvas = Image.new("RGB", (int(mol_image.width * 1.35), int(mol_image.height * 1.45)), (250, 249, 245))
    draw = ImageDraw.Draw(canvas)
    draw_rule_lines(draw, canvas.width, canvas.height, rng, rng.randint(10, 18))
    x = (canvas.width - mol_image.width) // 2
    y = (canvas.height - mol_image.height) // 2
    canvas.paste(mol_image, (x, y))
    draw.rectangle((x - 3, y - 3, x + mol_image.width + 3, y + mol_image.height + 3), outline=(150, 150, 150), width=1)
    return add_noise(canvas, 0.05)


def make_exam_page(mol_image: Image.Image, rng: random.Random) -> Image.Image:
    mol_image = rotate_expand(mol_image, rng, 1.8, (252, 250, 244))
    canvas = Image.new("RGB", (int(mol_image.width * 1.45), int(mol_image.height * 1.6)), (252, 250, 244))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), f"({rng.randint(1,9)})", fill=(65, 65, 65))
    draw.text((18, canvas.height - 26), rng.choice(["A.", "B.", "C.", "D."]), fill=(75, 75, 75))
    draw_rule_lines(draw, canvas.width, canvas.height, rng, rng.randint(6, 12))
    x = rng.randint(24, max(24, canvas.width - mol_image.width - 24))
    y = rng.randint(canvas.height // 4, max(canvas.height // 4, canvas.height - mol_image.height - 24))
    canvas.paste(mol_image, (x, y))
    return add_noise(canvas, 0.06)


def make_photo_sim(mol_image: Image.Image, rng: random.Random) -> Image.Image:
    image = rotate_expand(mol_image, rng, 4.5, (247, 247, 247))
    bg = Image.new("RGB", (image.width + rng.randint(30, 80), image.height + rng.randint(30, 80)), (247, 247, 247))
    x = (bg.width - image.width) // 2
    y = (bg.height - image.height) // 2
    bg.paste(image, (x, y))
    bg = add_shadow(bg, rng)
    bg = ImageEnhance.Contrast(bg).enhance(rng.uniform(0.75, 1.05))
    bg = ImageEnhance.Brightness(bg).enhance(rng.uniform(0.82, 1.05))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.0)))
    bg = jpeg_roundtrip(bg, quality=rng.randint(28, 60))
    return add_noise(bg, 0.07)


def make_scan_sim(mol_image: Image.Image, rng: random.Random) -> Image.Image:
    image = rotate_expand(mol_image, rng, 1.2, (246, 246, 244))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.78, 0.96))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.05))
    image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))
    image = jpeg_roundtrip(image, quality=rng.randint(35, 68))
    return add_noise(image, 0.05)


def render_style(Draw, Chem, smiles: str, style: str, seed_value: int, size: tuple[int, int]) -> Image.Image:
    rng = random.Random(seed_value)
    mol_image = render_molecule(Draw, Chem, smiles, size)
    if style == "print_page":
        return make_print_page(mol_image, rng)
    if style == "exam_page":
        return make_exam_page(mol_image, rng)
    if style == "photo_sim":
        return make_photo_sim(mol_image, rng)
    if style == "scan_sim":
        return make_scan_sim(mol_image, rng)
    raise ValueError(f"Unsupported style: {style}")


def normalize_id(text: str) -> str:
    clean = []
    for ch in str(text):
        clean.append(ch if ch.isalnum() or ch in {"_", "-", "."} else "_")
    value = "".join(clean).strip("._-")
    return value or "sample"


def load_existing_smiles(paths: list[Path], Chem) -> set[str]:
    result = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_rows(path):
            _, item = row
            raw = ""
            gt = item.get("ground_truth") if isinstance(item, dict) else None
            if isinstance(gt, dict):
                raw = gt.get("smiles") or ""
            raw = raw or item.get("smiles") or item.get("canonical_smiles") or item.get("label_summary") or ""
            canonical = canonicalize(Chem, raw)
            if canonical:
                result.add(canonical)
    return result


def build_eval(
    input_path: Path,
    output_root: Path,
    styles: list[str],
    seed: int,
    width: int,
    height: int,
    filter_paths: list[Path],
):
    Chem, Draw = try_load_rdkit()
    if Chem is None or Draw is None:
        raise RuntimeError("RDKit is required for rendering molecule images from SMILES.")

    existing_smiles = load_existing_smiles(filter_paths, Chem)
    annotations = []
    counts = Counter()
    skipped = Counter()
    seen = set()

    for line_no, row in read_rows(input_path):
        raw_smiles = row.get("smiles") or row.get("canonical_smiles") or row.get("label_summary") or ""
        canonical = canonicalize(Chem, raw_smiles)
        if not canonical:
            skipped["invalid_smiles"] += 1
            continue
        if canonical in existing_smiles:
            skipped["existing_eval_or_train_overlap"] += 1
            continue
        base_id = normalize_id(row.get("id") or f"generated_{line_no:05d}")
        if canonical in seen:
            skipped["duplicate_input_smiles"] += 1
            continue
        seen.add(canonical)
        source_name = str(row.get("source") or "generated_controlled")
        base_task_type = str(row.get("task_type") or "molecule_structure_recognition")

        for style in styles:
            style_seed = stable_seed(f"{base_id}::{style}", seed)
            image = render_style(Draw, Chem, canonical, style, style_seed, (width, height))
            image_dir = output_root / "images" / style
            image_dir.mkdir(parents=True, exist_ok=True)
            record_id = f"{style}__{base_id}"
            image_path = image_dir / f"{record_id}.png"
            image.save(image_path, format="PNG", optimize=True)
            difficulty = {
                "print_page": "document_embed",
                "exam_page": "chinese_exam",
                "photo_sim": "photo",
                "scan_sim": "scan",
            }[style]
            annotations.append(
                {
                    "id": record_id,
                    "source": source_name,
                    "image": image_path.relative_to(output_root).as_posix(),
                    "task_type": base_task_type,
                    "image_type": style,
                    "difficulty": difficulty,
                    "ground_truth": {
                        "smiles": canonical,
                        "inchi": None,
                        "selfies": None,
                        "mol": None,
                    },
                    "eval_target": "canonical_smiles",
                    "license": row.get("license", "self_generated_internal"),
                    "source_url_or_doc": row.get("source_url_or_doc", str(input_path.name)),
                    "qc_status": "pass",
                    "benchmark_track": "generated_eval_v1",
                    "generator_style": style,
                }
            )
            counts[style] += 1

    annotations_path = output_root / "annotations" / "labels.jsonl"
    write_jsonl(annotations_path, annotations)
    write_csv(output_root / "annotations" / "labels.csv", annotations)
    stats = {
        "input": str(input_path),
        "output_root": str(output_root),
        "styles": styles,
        "total": len(annotations),
        "by_style": dict(counts),
        "skipped": dict(skipped),
        "width": width,
        "height": height,
        "filter_paths": [str(path) for path in filter_paths],
    }
    (output_root / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(readme_text(stats), encoding="utf-8")
    return stats


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "source",
        "image",
        "task_type",
        "image_type",
        "difficulty",
        "eval_target",
        "generator_style",
        "qc_status",
        "smiles",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "image": row["image"],
                    "task_type": row["task_type"],
                    "image_type": row["image_type"],
                    "difficulty": row["difficulty"],
                    "eval_target": row["eval_target"],
                    "generator_style": row["generator_style"],
                    "qc_status": row["qc_status"],
                    "smiles": row["ground_truth"]["smiles"],
                }
            )


def readme_text(stats: dict) -> str:
    return (
        "# Generated Controlled Evaluation Set v1\n\n"
        "This set is generated from known SMILES and rendered into multiple realistic page/image styles.\n\n"
        f"- Total samples: {stats['total']}\n"
        f"- Styles: {json.dumps(stats['by_style'], ensure_ascii=False)}\n\n"
        "Included styles:\n"
        "- `print_page`: printed document/page layout\n"
        "- `exam_page`: exam or worksheet style layout\n"
        "- `photo_sim`: phone-photo style distortions\n"
        "- `scan_sim`: scan/compression style distortions\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="V2-1/data/eval_generated/source_smiles/generated_eval_seed.csv")
    parser.add_argument("--output-root", default="V2-1/data/eval_generated/generated_eval_v1")
    parser.add_argument("--styles", nargs="*", default=["print_page", "exam_page", "photo_sim", "scan_sim"])
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=420)
    parser.add_argument(
        "--filter-paths",
        nargs="*",
        default=[
            "V2-1/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl",
            "V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl",
            "V2-1/data/eval/weak_domain_v2/annotations/labels.jsonl",
            "V2-1/data/sft_materialized/train_singleline_rw_messages.jsonl",
        ],
    )
    args = parser.parse_args()

    bad = [style for style in args.styles if style not in VALID_STYLES]
    if bad:
        raise ValueError(f"Unsupported styles: {bad}")

    input_path = Path(args.input).resolve()
    output_root = Path(args.output_root).resolve()
    filter_paths = [Path(path).resolve() for path in args.filter_paths]
    stats = build_eval(
        input_path=input_path,
        output_root=output_root,
        styles=args.styles,
        seed=args.seed,
        width=args.width,
        height=args.height,
        filter_paths=filter_paths,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
