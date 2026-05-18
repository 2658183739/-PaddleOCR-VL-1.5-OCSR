from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_id(value: str) -> str:
    value = SAFE_ID_PATTERN.sub("_", str(value).strip())
    return value.strip("._-") or "sample"


def read_table(path: Path):
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


def resolve_image(input_table: Path, image_value: str) -> Path:
    path = Path(str(image_value))
    if path.is_absolute():
        return path
    return (input_table.parent / path).resolve()


def normalize_image(src: Path, out_dir: Path, record_id: str, min_side: int, max_side: int, pad: int) -> tuple[Path, tuple[int, int], str]:
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)
        original_mode = image.mode
        image = image.convert("RGB")
        width, height = image.size
        if min(width, height) < min_side:
            scale = min_side / max(1, min(width, height))
            width = int(round(width * scale))
            height = int(round(height * scale))
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        if max(width, height) > max_side:
            scale = max_side / max(width, height)
            width = int(round(width * scale))
            height = int(round(height * scale))
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        if pad > 0:
            padded = Image.new("RGB", (image.width + 2 * pad, image.height + 2 * pad), "white")
            padded.paste(image, (pad, pad))
            image = padded
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{record_id}.png"
        image.save(dest, format="PNG", optimize=True)
        return dest, image.size, original_mode


def prepare_manifest(
    input_table: Path,
    output_manifest: Path,
    output_image_root: Path,
    default_source: str,
    default_difficulty: str,
    default_weak_domain: str,
    min_side: int,
    max_side: int,
    pad: int,
):
    Chem = try_load_rdkit()
    rows = []
    skipped = Counter()
    examples = {"missing_image": [], "bad_extension": [], "unreadable_image": [], "invalid_smiles": [], "duplicate_id": []}
    ids = Counter()
    modes = Counter()
    sources = Counter()
    difficulties = Counter()

    for line_no, raw in read_table(input_table):
        record_id = safe_id(raw.get("id") or f"weak_candidate_{line_no:06d}")
        ids[record_id] += 1
        if ids[record_id] > 1:
            skipped["duplicate_id"] += 1
            if len(examples["duplicate_id"]) < 20:
                examples["duplicate_id"].append({"line": line_no, "id": record_id})
            continue

        image_value = raw.get("image") or raw.get("image_path") or raw.get("file")
        if not image_value:
            skipped["missing_image"] += 1
            if len(examples["missing_image"]) < 20:
                examples["missing_image"].append({"line": line_no, "id": record_id, "image": ""})
            continue
        src = resolve_image(input_table, image_value)
        if not src.exists():
            skipped["missing_image"] += 1
            if len(examples["missing_image"]) < 20:
                examples["missing_image"].append({"line": line_no, "id": record_id, "image": str(src)})
            continue
        if src.suffix.lower() not in VALID_IMAGE_EXTS:
            skipped["bad_extension"] += 1
            if len(examples["bad_extension"]) < 20:
                examples["bad_extension"].append({"line": line_no, "id": record_id, "image": str(src)})
            continue

        raw_smiles = raw.get("smiles") or raw.get("canonical_smiles") or raw.get("label_summary")
        smiles = canonicalize(Chem, str(raw_smiles or ""))
        if not smiles:
            skipped["invalid_smiles"] += 1
            if len(examples["invalid_smiles"]) < 20:
                examples["invalid_smiles"].append({"line": line_no, "id": record_id, "smiles": raw_smiles})
            continue

        source = str(raw.get("source") or default_source)
        difficulty = str(raw.get("difficulty") or default_difficulty)
        weak_domain = str(raw.get("weak_domain") or default_weak_domain)
        try:
            normalized_image, size, mode = normalize_image(
                src=src,
                out_dir=output_image_root / source,
                record_id=record_id,
                min_side=min_side,
                max_side=max_side,
                pad=pad,
            )
        except Exception as exc:
            skipped["unreadable_image"] += 1
            if len(examples["unreadable_image"]) < 20:
                examples["unreadable_image"].append({"line": line_no, "id": record_id, "image": str(src), "reason": str(exc)})
            continue

        modes[mode] += 1
        sources[source] += 1
        difficulties[difficulty] += 1
        rows.append(
            {
                "id": record_id,
                "image": normalized_image.relative_to(output_manifest.parent).as_posix(),
                "smiles": smiles,
                "source": source,
                "difficulty": difficulty,
                "weak_domain": weak_domain,
                "task_type": raw.get("task_type") or "molecule_structure_recognition",
                "license": raw.get("license") or "",
                "source_url_or_doc": raw.get("source_url_or_doc") or "",
                "collector": raw.get("collector") or "",
                "notes": raw.get("notes") or "",
                "image_size": list(size),
            }
        )

    write_jsonl(output_manifest, rows)
    report = {
        "input_table": str(input_table),
        "output_manifest": str(output_manifest),
        "output_image_root": str(output_image_root),
        "total_records": len(rows),
        "rdkit_available": Chem is not None,
        "source_counts": dict(sources),
        "difficulty_counts": dict(difficulties),
        "original_image_modes": dict(modes),
        "skipped": dict(skipped),
        "examples": examples,
        "image_rules": {
            "format": "RGB PNG",
            "min_side": min_side,
            "max_side": max_side,
            "pad": pad,
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV or JSONL with id,image,smiles and optional metadata.")
    parser.add_argument("--output", default="V2-1/data/manifests/weak_domain_training_candidates.jsonl")
    parser.add_argument("--image-output-root", default="V2-1/data/incoming/weak_domain/normalized_images")
    parser.add_argument("--report", default="V2-1/reports/weak_domain_manifest_prepare_report.json")
    parser.add_argument("--default-source", default="private_weak_domain")
    parser.add_argument("--default-difficulty", default="hard")
    parser.add_argument("--default-weak-domain", default="real_world_photo_scan")
    parser.add_argument("--min-side", type=int, default=96)
    parser.add_argument("--max-side", type=int, default=1800)
    parser.add_argument("--pad", type=int, default=8)
    args = parser.parse_args()

    input_table = Path(args.input).resolve()
    output_manifest = Path(args.output).resolve()
    output_image_root = Path(args.image_output_root).resolve()
    report = prepare_manifest(
        input_table=input_table,
        output_manifest=output_manifest,
        output_image_root=output_image_root,
        default_source=args.default_source,
        default_difficulty=args.default_difficulty,
        default_weak_domain=args.default_weak_domain,
        min_side=args.min_side,
        max_side=args.max_side,
        pad=args.pad,
    )
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
