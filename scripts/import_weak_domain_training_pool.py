from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


PROMPT = "OCR: Output only the canonical SMILES string for the molecule shown in the image."


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


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


def load_eval_smiles(paths: list[Path], Chem) -> set[str]:
    result = set()
    for path in paths:
        if not path.exists():
            continue
        for _, row in read_jsonl(path):
            gt = row.get("ground_truth")
            raw = ""
            if isinstance(gt, dict):
                raw = gt.get("smiles") or ""
            raw = raw or row.get("canonical_smiles") or row.get("smiles") or row.get("label_summary") or ""
            canonical = canonicalize(Chem, raw)
            if canonical:
                result.add(canonical)
    return result


def source_image_path(manifest_path: Path, row: dict) -> Path:
    raw = row.get("image") or row.get("image_path")
    if not raw:
        return Path("")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def copy_image(src: Path, assets_root: Path, source: str, record_id: str) -> Path:
    suffix = src.suffix.lower() or ".png"
    target_dir = assets_root / source
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{record_id}{suffix}"
    if not target.exists():
        shutil.copy2(src, target)
    return target


def build_record(output_jsonl: Path, image_path: Path, smiles: str, meta: dict) -> dict:
    rel_image = Path(os.path.relpath(image_path.resolve(), output_jsonl.parent.resolve())).as_posix()
    return {
        "messages": [
            {"role": "user", "content": f"<image>{PROMPT}"},
            {"role": "assistant", "content": smiles},
        ],
        "images": [rel_image],
        "meta": meta,
    }


def import_pool(
    manifest_path: Path,
    output_jsonl: Path,
    assets_root: Path,
    eval_smiles: set[str],
    default_source: str,
    default_difficulty: str,
):
    Chem = try_load_rdkit()
    records = []
    source_counts = Counter()
    difficulty_counts = Counter()
    skipped = Counter()
    examples = {"invalid_smiles": [], "missing_image": [], "unreadable_image": [], "eval_overlap": []}

    for line_no, row in read_jsonl(manifest_path):
        record_id = str(row.get("id") or f"weak_train_{line_no:06d}")
        source = str(row.get("source") or default_source)
        difficulty = str(row.get("difficulty") or default_difficulty)
        raw_smiles = row.get("smiles") or row.get("canonical_smiles") or row.get("label_summary")
        if not raw_smiles and isinstance(row.get("ground_truth"), dict):
            raw_smiles = row["ground_truth"].get("smiles")
        smiles = canonicalize(Chem, str(raw_smiles or ""))
        if not smiles:
            skipped["invalid_smiles"] += 1
            if len(examples["invalid_smiles"]) < 20:
                examples["invalid_smiles"].append({"line": line_no, "id": record_id, "smiles": raw_smiles})
            continue
        if smiles in eval_smiles:
            skipped["eval_overlap"] += 1
            if len(examples["eval_overlap"]) < 20:
                examples["eval_overlap"].append({"line": line_no, "id": record_id, "smiles": smiles})
            continue

        src_image = source_image_path(manifest_path, row)
        if not src_image.exists():
            skipped["missing_image"] += 1
            if len(examples["missing_image"]) < 20:
                examples["missing_image"].append({"line": line_no, "id": record_id, "image": str(row.get("image") or row.get("image_path") or "")})
            continue
        try:
            with Image.open(src_image) as image:
                width, height = image.size
                image.verify()
        except Exception as exc:
            skipped["unreadable_image"] += 1
            if len(examples["unreadable_image"]) < 20:
                examples["unreadable_image"].append({"line": line_no, "id": record_id, "image": str(src_image), "reason": str(exc)})
            continue

        copied = copy_image(src_image, assets_root, source, record_id)
        meta = {
            "id": record_id,
            "source": source,
            "difficulty": difficulty,
            "task_type": row.get("task_type", "molecule_structure_recognition"),
            "weak_domain": row.get("weak_domain", source),
            "license": row.get("license", ""),
            "source_url_or_doc": row.get("source_url_or_doc", ""),
            "image_size": [width, height],
            "canonical_smiles_length": len(smiles),
            "contains_stereo": "@" in smiles,
        }
        records.append(build_record(output_jsonl, copied, smiles, meta))
        source_counts[source] += 1
        difficulty_counts[difficulty] += 1

    write_jsonl(output_jsonl, records)
    return {
        "manifest": str(manifest_path),
        "output": str(output_jsonl),
        "assets_root": str(assets_root),
        "total_records": len(records),
        "source_counts": dict(source_counts),
        "difficulty_counts": dict(difficulty_counts),
        "skipped": dict(skipped),
        "rdkit_available": Chem is not None,
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", required=True, help="JSONL with id,image,smiles/source/difficulty fields.")
    parser.add_argument("--output", default="V2-1/data/sft_materialized/train_weak_domain_pool_messages.jsonl")
    parser.add_argument("--assets-root", default="V2-1/data/assets/weak_domain_pool")
    parser.add_argument(
        "--eval-labels",
        nargs="*",
        default=[
            "V2-1/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl",
            "V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl",
            "V2-1/data/eval/weak_domain_v2/annotations/labels.jsonl",
        ],
    )
    parser.add_argument("--default-source", default="private_weak_domain")
    parser.add_argument("--default-difficulty", default="hard")
    parser.add_argument("--report", default="V2-1/reports/weak_domain_training_pool_import.json")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    Chem = try_load_rdkit()
    eval_smiles = load_eval_smiles([(project_root / path).resolve() for path in args.eval_labels], Chem)
    report = import_pool(
        manifest_path=(project_root / args.manifest).resolve(),
        output_jsonl=(project_root / args.output).resolve(),
        assets_root=(project_root / args.assets_root).resolve(),
        eval_smiles=eval_smiles,
        default_source=args.default_source,
        default_difficulty=args.default_difficulty,
    )
    report["eval_smiles_filter_count"] = len(eval_smiles)
    report_path = (project_root / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
