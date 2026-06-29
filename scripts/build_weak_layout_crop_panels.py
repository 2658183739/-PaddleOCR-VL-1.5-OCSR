#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


WEAK_DOMAINS = {"document_embed", "journal_fig", "multi_grid"}

REL_BOXES = {
    "document_embed": {
        "layout_primary": (0.16, 0.31, 0.86, 0.98),
        "layout_target_tight": (0.22, 0.27, 0.92, 0.86),
        "layout_wide": (0.03, 0.22, 0.97, 1.00),
        "layout_top_left": (0.05, 0.24, 0.55, 0.72),
        "layout_full": (0.00, 0.00, 1.00, 1.00),
        "layout_center": (0.18, 0.12, 0.94, 0.84),
        "layout_right": (0.34, 0.10, 0.98, 0.90),
        "layout_lower": (0.08, 0.38, 0.98, 1.00),
        "layout_first_row": (0.00, 0.00, 1.00, 0.64),
        "layout_structure_band": (0.08, 0.22, 0.98, 0.80),
        "layout_top_left_tiny": (0.00, 0.18, 0.46, 0.62),
        "layout_focus_lower_right": (0.24, 0.55, 0.98, 0.96),
        "layout_focus_mid_right": (0.28, 0.42, 0.98, 0.82),
        "layout_captionless_lower": (0.30, 0.60, 0.98, 0.94),
        "layout_panel_a_zoom": (0.20, 0.42, 0.88, 0.88),
        "layout_panel_a_square": (0.18, 0.36, 0.82, 0.92),
        "layout_textless_right": (0.40, 0.38, 0.98, 0.86),
        "layout_structure_core": (0.20, 0.36, 0.96, 0.90),
        "layout_structure_core_tight": (0.30, 0.42, 0.96, 0.84),
        "layout_panel_a_core": (0.22, 0.38, 0.90, 0.92),
        "layout_right_structure_only": (0.48, 0.34, 0.98, 0.84),
        "layout_mid_structure_only": (0.36, 0.34, 0.92, 0.78),
        "layout_lower_structure_only": (0.36, 0.50, 0.98, 0.92),
        "layout_upper_structure_only": (0.34, 0.28, 0.98, 0.68),
    },
    "journal_fig": {
        "layout_primary": (0.02, 0.03, 0.52, 0.38),
        "layout_target_tight": (0.04, 0.08, 0.52, 0.32),
        "layout_wide": (0.00, 0.00, 0.78, 0.52),
        "layout_top_left": (0.00, 0.00, 0.42, 0.31),
        "layout_full": (0.00, 0.00, 1.00, 1.00),
        "layout_center": (0.10, 0.02, 0.86, 0.48),
        "layout_right": (0.38, 0.00, 0.98, 0.48),
        "layout_lower": (0.00, 0.18, 0.92, 0.68),
        "layout_first_row": (0.00, 0.00, 1.00, 0.38),
        "layout_structure_band": (0.00, 0.00, 1.00, 0.54),
        "layout_top_left_tiny": (0.00, 0.00, 0.34, 0.24),
        "layout_focus_lower_right": (0.28, 0.10, 0.92, 0.58),
        "layout_focus_mid_right": (0.18, 0.00, 0.72, 0.42),
        "layout_captionless_lower": (0.00, 0.02, 0.56, 0.36),
        "layout_panel_a_zoom": (0.00, 0.00, 0.40, 0.28),
        "layout_panel_a_square": (0.00, 0.00, 0.48, 0.36),
        "layout_textless_right": (0.02, 0.00, 0.58, 0.38),
        "layout_structure_core": (0.00, 0.00, 0.58, 0.40),
        "layout_structure_core_tight": (0.00, 0.00, 0.42, 0.30),
        "layout_panel_a_core": (0.00, 0.00, 0.50, 0.36),
        "layout_right_structure_only": (0.08, 0.00, 0.68, 0.42),
        "layout_mid_structure_only": (0.00, 0.00, 0.64, 0.44),
        "layout_lower_structure_only": (0.00, 0.06, 0.68, 0.50),
        "layout_upper_structure_only": (0.00, 0.00, 0.62, 0.34),
    },
    "multi_grid": {
        "layout_primary": (0.00, 0.04, 0.40, 0.48),
        "layout_target_tight": (0.00, 0.10, 0.38, 0.46),
        "layout_wide": (0.00, 0.00, 0.60, 0.66),
        "layout_top_left": (0.00, 0.00, 0.34, 0.36),
        "layout_full": (0.00, 0.00, 1.00, 1.00),
        "layout_center": (0.10, 0.02, 0.62, 0.54),
        "layout_right": (0.24, 0.00, 0.72, 0.52),
        "layout_lower": (0.00, 0.30, 0.62, 0.86),
        "layout_first_row": (0.00, 0.00, 1.00, 0.46),
        "layout_structure_band": (0.00, 0.00, 0.74, 0.58),
        "layout_top_left_tiny": (0.00, 0.00, 0.28, 0.30),
        "layout_focus_lower_right": (0.00, 0.18, 0.52, 0.68),
        "layout_focus_mid_right": (0.18, 0.00, 0.58, 0.42),
        "layout_captionless_lower": (0.00, 0.04, 0.46, 0.52),
        "layout_panel_a_zoom": (0.00, 0.00, 0.48, 0.42),
        "layout_panel_a_square": (0.00, 0.00, 0.52, 0.52),
        "layout_textless_right": (0.00, 0.00, 0.56, 0.50),
        "layout_structure_core": (0.00, 0.00, 0.56, 0.52),
        "layout_structure_core_tight": (0.00, 0.00, 0.40, 0.38),
        "layout_panel_a_core": (0.00, 0.00, 0.48, 0.48),
        "layout_right_structure_only": (0.08, 0.00, 0.62, 0.52),
        "layout_mid_structure_only": (0.00, 0.00, 0.56, 0.48),
        "layout_lower_structure_only": (0.00, 0.10, 0.56, 0.62),
        "layout_upper_structure_only": (0.00, 0.00, 0.54, 0.38),
    },
}

BASE_VARIANTS = {
    "layout_primary",
    "layout_target_tight",
    "layout_wide",
    "layout_top_left",
    "layout_full",
    "layout_center",
    "layout_right",
    "layout_lower",
    "layout_first_row",
    "layout_structure_band",
    "layout_top_left_tiny",
    "layout_focus_lower_right",
    "layout_focus_mid_right",
    "layout_captionless_lower",
    "layout_panel_a_zoom",
    "layout_panel_a_square",
    "layout_textless_right",
    "layout_structure_core",
    "layout_structure_core_tight",
    "layout_panel_a_core",
    "layout_right_structure_only",
    "layout_mid_structure_only",
    "layout_lower_structure_only",
    "layout_upper_structure_only",
}

TRIM_VARIANTS = {"layout_primary_trim", "layout_wide_trim", "layout_target_gray", "layout_primary_gray"}
GRAY_VARIANTS = {"layout_primary_gray", "layout_target_gray"}
AUTO_STRUCTURE_VARIANTS = {"layout_auto_structure", "layout_wide_auto_structure"}
CORE_STRUCTURE_VARIANTS = set()
DERIVED_VARIANT_BASE = {
    "layout_primary_trim": "layout_primary",
    "layout_primary_gray": "layout_primary",
    "layout_auto_structure": "layout_primary",
    "layout_wide_trim": "layout_wide",
    "layout_wide_auto_structure": "layout_wide",
    "layout_target_gray": "layout_target_tight",
}
VARIANTS = BASE_VARIANTS | set(DERIVED_VARIANT_BASE)
DEFAULT_VARIANTS = [
    "layout_primary",
    "layout_target_tight",
    "layout_primary_trim",
    "layout_primary_gray",
    "layout_auto_structure",
    "layout_wide",
    "layout_top_left",
    "layout_full",
    "layout_center",
    "layout_right",
    "layout_lower",
    "layout_first_row",
    "layout_structure_band",
    "layout_top_left_tiny",
    "layout_wide_trim",
    "layout_wide_auto_structure",
    "layout_target_gray",
    "layout_focus_lower_right",
    "layout_focus_mid_right",
    "layout_captionless_lower",
    "layout_panel_a_zoom",
    "layout_panel_a_square",
    "layout_textless_right",
    "layout_structure_core",
    "layout_structure_core_tight",
    "layout_panel_a_core",
    "layout_right_structure_only",
    "layout_mid_structure_only",
    "layout_lower_structure_only",
]

DOMAIN_PROMPTS = {
    "document_embed": [
        "Ignore all paragraph text and labels. Read only the molecule drawing embedded in the document figure. Output only canonical SMILES.",
        "The target is the chemical structure in this crop, not the surrounding text. Return only the SMILES string.",
        "Extract the primary molecular diagram from this document crop. Ignore captions, descriptions, and supporting information.",
        "OCR the molecule structure only. Do not copy words from the document. Output canonical SMILES only.",
    ],
    "journal_fig": [
        "Use only Compound 1 or the first visible molecule structure in this journal figure. Ignore other compounds and all text. Output only canonical SMILES.",
        "Read the top-left first-labeled chemical structure. Ignore captions, table headers, and later compounds. Return SMILES only.",
        "Identify Compound 1 in this cropped figure and provide its canonical SMILES. Do not include explanations.",
        "If multiple compounds are visible, choose the first target molecule only. Output only the SMILES string.",
    ],
    "multi_grid": [
        "Use only panel (a), the top-left molecule structure. Ignore panel (b), other panels, labels, and text. Output only canonical SMILES.",
        "Read the first/top-left structure in this multi-panel grid. Return only the SMILES string for panel (a).",
        "Extract the molecule in panel (a). Ignore adjacent molecules and grid text. Output canonical SMILES only.",
        "If more than one structure is visible, choose the top-left target structure. Do not include explanations.",
    ],
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


def portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def weak_domain(row: dict) -> str:
    values = [
        row.get("difficulty", ""),
        row.get("task_type", ""),
        row.get("id", ""),
        row.get("preprocess_domain", ""),
    ]
    for value in values:
        text = str(value)
        for domain in WEAK_DOMAINS:
            if domain in text:
                return domain
    return ""


def scaled_box(image: Image.Image, rel_box):
    left, top, right, bottom = rel_box
    return (
        max(0, int(round(left * image.width))),
        max(0, int(round(top * image.height))),
        min(image.width, int(round(right * image.width))),
        min(image.height, int(round(bottom * image.height))),
    )


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


def structure_component_bbox(
    image: Image.Image,
    threshold: int,
    pad: int,
    max_side: int = 360,
    include_score_ratio: float = 0.28,
    expand_ratio: float = 0.04,
):
    original_width, original_height = image.size
    scale = min(1.0, max_side / max(1, max(original_width, original_height)))
    if scale < 1.0:
        work_image = image.resize(
            (
                max(1, int(round(original_width * scale))),
                max(1, int(round(original_height * scale))),
            ),
            Image.Resampling.BILINEAR,
        )
        work_pad = max(2, int(round(pad * scale)))
    else:
        work_image = image
        work_pad = pad

    gray = ImageOps.grayscale(work_image)
    width, height = gray.size
    pixels = gray.load()
    visited = bytearray(width * height)
    components = []

    for start_y in range(height):
        for start_x in range(width):
            start_index = start_y * width + start_x
            if visited[start_index] or pixels[start_x, start_y] >= threshold:
                continue

            stack = [(start_x, start_y)]
            visited[start_index] = 1
            left = right = start_x
            top = bottom = start_y
            area = 0

            while stack:
                x, y = stack.pop()
                area += 1
                if x < left:
                    left = x
                if x > right:
                    right = x
                if y < top:
                    top = y
                if y > bottom:
                    bottom = y

                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    index = ny * width + nx
                    if visited[index] or pixels[nx, ny] >= threshold:
                        continue
                    visited[index] = 1
                    stack.append((nx, ny))

            box_width = right - left + 1
            box_height = bottom - top + 1
            if area < 12:
                continue
            if box_width < 18 and box_height < 18 and area < 80:
                continue
            score = area + 0.03 * box_width * box_height + 1.25 * max(box_width, box_height)
            components.append(
                {
                    "box": (left, top, right + 1, bottom + 1),
                    "area": area,
                    "width": box_width,
                    "height": box_height,
                    "score": score,
                }
            )

    if not components:
        return content_bbox(image, threshold=threshold, pad=pad)

    best = max(components, key=lambda item: item["score"])
    left, top, right, bottom = best["box"]
    expanded = (
        max(0, left - max(work_pad, int(width * expand_ratio))),
        max(0, top - max(work_pad, int(height * expand_ratio))),
        min(width, right + max(work_pad, int(width * expand_ratio))),
        min(height, bottom + max(work_pad, int(height * expand_ratio))),
    )
    union = [left, top, right, bottom]
    for component in components:
        c_left, c_top, c_right, c_bottom = component["box"]
        intersects = not (
            c_right < expanded[0] or c_left > expanded[2] or c_bottom < expanded[1] or c_top > expanded[3]
        )
        if intersects or component["score"] > best["score"] * include_score_ratio:
            union[0] = min(union[0], c_left)
            union[1] = min(union[1], c_top)
            union[2] = max(union[2], c_right)
            union[3] = max(union[3], c_bottom)

    left = max(0, union[0] - work_pad)
    top = max(0, union[1] - work_pad)
    right = min(width, union[2] + work_pad)
    bottom = min(height, union[3] + work_pad)
    if right <= left or bottom <= top:
        return content_bbox(image, threshold=threshold, pad=pad)
    if scale < 1.0:
        inverse = 1.0 / scale
        return (
            max(0, int(left * inverse) - pad),
            max(0, int(top * inverse) - pad),
            min(original_width, int(right * inverse) + pad),
            min(original_height, int(bottom * inverse) + pad),
        )
    return (left, top, right, bottom)


def add_border(image: Image.Image, border: int) -> Image.Image:
    return ImageOps.expand(image.convert("RGB"), border=border, fill=(255, 255, 255))


def crop_by_variant(image: Image.Image, domain: str, variant: str, threshold: int, pad: int, border: int) -> tuple[Image.Image, tuple]:
    base_variant = DERIVED_VARIANT_BASE.get(variant, variant)
    rel_box = REL_BOXES[domain][base_variant]
    cropped = image.crop(scaled_box(image, rel_box))
    if variant in TRIM_VARIANTS:
        cropped = cropped.crop(content_bbox(cropped, threshold=threshold, pad=pad))
    if variant in AUTO_STRUCTURE_VARIANTS:
        cropped = cropped.crop(structure_component_bbox(cropped, threshold=min(225, threshold), pad=pad))
    if variant in CORE_STRUCTURE_VARIANTS:
        cropped = cropped.crop(
            structure_component_bbox(
                cropped,
                threshold=min(225, threshold),
                pad=max(8, pad // 2),
                include_score_ratio=0.55,
                expand_ratio=0.025,
            )
        )
    if variant in GRAY_VARIANTS:
        gray = ImageOps.autocontrast(ImageOps.grayscale(cropped))
        gray = ImageEnhance.Contrast(gray).enhance(1.35)
        gray = ImageEnhance.Sharpness(gray).enhance(1.25)
        cropped = gray.convert("RGB")
    return add_border(cropped, border=border), rel_box


def safe_image_name(sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    if not stem:
        stem = "sample"
    return f"{stem[:120]}_{digest}.png"


def summarize(rows):
    return {
        "total": len(rows),
        "eval_panel": dict(Counter(str(row.get("eval_panel", "")) for row in rows)),
        "source": dict(Counter(str(row.get("source", "")) for row in rows)),
        "difficulty": dict(Counter(str(row.get("difficulty", "")) for row in rows)),
        "task_type": dict(Counter(str(row.get("task_type", "")) for row in rows)),
        "preprocess_domain": dict(Counter(str(row.get("preprocess_domain", "")) for row in rows)),
        "preprocess_variant": dict(Counter(str(row.get("preprocess_variant", "")) for row in rows)),
    }


def build_montage(paths: list[Path], out_path: Path, title: str, thumb_width: int = 260, columns: int = 4):
    if not paths:
        return
    font = ImageFont.load_default()
    tiles = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        scale = min(1.0, thumb_width / max(1, image.width))
        thumb = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        canvas = Image.new("RGB", (thumb_width, thumb.height + 20), "white")
        canvas.paste(thumb, ((thumb_width - thumb.width) // 2, 0))
        ImageDraw.Draw(canvas).text((4, thumb.height + 4), path.stem[:40], fill=(0, 0, 0), font=font)
        tiles.append(canvas)

    rows = (len(tiles) + columns - 1) // columns
    row_heights = []
    for row_index in range(rows):
        row_tiles = tiles[row_index * columns : (row_index + 1) * columns]
        row_heights.append(max(tile.height for tile in row_tiles))
    title_height = 28
    width = columns * thumb_width
    height = title_height + sum(row_heights)
    montage = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(montage)
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    y = title_height
    for row_index in range(rows):
        row_tiles = tiles[row_index * columns : (row_index + 1) * columns]
        for col_index, tile in enumerate(row_tiles):
            montage.paste(tile, (col_index * thumb_width, y))
        y += row_heights[row_index]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--labels-jsonl",
        default="V2-1/reports/main_eval_with_candidates_20260627_fast_notta/combined/labels.jsonl",
    )
    parser.add_argument("--output-root", default="V2-1/reports/weak_layout_crop_20260627")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
    )
    parser.add_argument("--domains", default="document_embed,journal_fig,multi_grid")
    parser.add_argument("--threshold", type=int, default=245)
    parser.add_argument("--pad", type=int, default=16)
    parser.add_argument("--border", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--spotcheck-count", type=int, default=16)
    parser.add_argument("--inject-domain-prompts", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    labels_path = (project_root / args.labels_jsonl).resolve()
    output_root = project_root / args.output_root
    selected_variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    selected_domains = {item.strip() for item in args.domains.split(",") if item.strip()}
    unknown_variants = [item for item in selected_variants if item not in VARIANTS]
    unknown_domains = sorted(selected_domains - WEAK_DOMAINS)
    if unknown_variants:
        raise ValueError(f"unknown variants: {unknown_variants}; available: {sorted(VARIANTS)}")
    if unknown_domains:
        raise ValueError(f"unknown domains: {unknown_domains}; available: {sorted(WEAK_DOMAINS)}")

    source_rows = []
    skipped = Counter()
    for row in read_jsonl(labels_path):
        domain = weak_domain(row)
        if not domain or domain not in selected_domains:
            skipped["not_selected_domain"] += 1
            continue
        out = dict(row)
        out["preprocess_domain"] = domain
        source_rows.append(out)
        if args.limit > 0 and len(source_rows) >= args.limit:
            break

    manifest = {
        "source_labels": portable_path(labels_path, project_root),
        "output_root": portable_path(output_root, project_root),
        "domains": sorted(selected_domains),
        "rel_boxes": REL_BOXES,
        "threshold": args.threshold,
        "pad": args.pad,
        "border": args.border,
        "source_summary": summarize(source_rows),
        "skipped": dict(skipped),
        "variants": {},
    }

    for variant_name in selected_variants:
        variant_root = output_root / variant_name
        rows = []
        sample_paths = []
        for row in source_rows:
            sample_id = str(row["id"])
            domain = row["preprocess_domain"]
            image_path = resolve_image_path(get_image_ref(row), project_root, labels_path)
            image = Image.open(image_path).convert("RGB")
            out_image, rel_box = crop_by_variant(
                image,
                domain=domain,
                variant=variant_name,
                threshold=args.threshold,
                pad=args.pad,
                border=args.border,
            )
            out_rel = Path(args.output_root) / variant_name / "images" / safe_image_name(sample_id)
            out_abs = project_root / out_rel
            out_abs.parent.mkdir(parents=True, exist_ok=True)
            out_image.save(out_abs)
            if len(sample_paths) < args.spotcheck_count:
                sample_paths.append(out_abs)

            new_row = dict(row)
            new_row["image"] = portable_path(out_abs, project_root)
            new_row.pop("image_path", None)
            new_row["original_image"] = portable_path(image_path, project_root)
            new_row["preprocess_variant"] = variant_name
            new_row["preprocess_box_rel"] = rel_box
            if args.inject_domain_prompts:
                new_row["prompt_profile"] = f"weak_layout_{domain}"
                new_row["prompt_list"] = DOMAIN_PROMPTS[domain]
            rows.append(new_row)

        labels_out = variant_root / "annotations" / "labels.jsonl"
        write_jsonl(labels_out, rows)
        spotcheck_path = output_root / "spotcheck" / f"{variant_name}.jpg"
        build_montage(sample_paths, spotcheck_path, title=variant_name)
        manifest["variants"][variant_name] = {
            "labels": portable_path(labels_out, project_root),
            "spotcheck": portable_path(spotcheck_path, project_root) if sample_paths else "",
            "summary": summarize(rows),
        }

    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
