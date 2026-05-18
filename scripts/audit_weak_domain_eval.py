from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def get_smiles(row: dict) -> str:
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, dict) and ground_truth.get("smiles"):
        return str(ground_truth["smiles"]).strip()
    for key in ("canonical_smiles", "smiles", "label_summary"):
        if row.get(key):
            return str(row[key]).strip()
    return ""


def try_load_rdkit():
    try:
        from rdkit import Chem

        return Chem
    except Exception:
        return None


def canonicalize(Chem, smiles: str) -> str | None:
    if Chem is None:
        return smiles or None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def resolve_image(eval_root: Path, row: dict) -> Path:
    image_value = row.get("image") or row.get("image_path")
    if not image_value:
        return Path("")
    path = Path(str(image_value))
    if path.is_absolute():
        return path
    return eval_root / path


def load_train_keys(train_jsonl: Path, Chem):
    smiles = Counter()
    image_names = Counter()
    ids = Counter()
    if not train_jsonl.exists():
        return ids, image_names, smiles

    for _, row in read_jsonl(train_jsonl):
        meta = row.get("meta", {})
        record_id = str(meta.get("id", ""))
        if record_id:
            ids[record_id] += 1
        for image in row.get("images") or []:
            image_names[Path(str(image)).name.lower()] += 1
        assistant = ""
        for message in row.get("messages", []):
            if message.get("role") == "assistant":
                assistant = str(message.get("content", "")).strip()
                break
        canonical = canonicalize(Chem, assistant)
        if canonical:
            smiles[canonical] += 1
    return ids, image_names, smiles


def audit(eval_root: Path, train_jsonl: Path, limit_examples: int):
    Chem = try_load_rdkit()
    labels_path = eval_root / "annotations" / "labels.jsonl"
    train_ids, train_image_names, train_smiles = load_train_keys(train_jsonl, Chem)

    total = 0
    by_source = Counter()
    by_domain = Counter()
    by_difficulty = Counter()
    ids = Counter()
    canonical_smiles = Counter()
    missing_images = []
    unreadable_images = []
    invalid_smiles = []
    train_id_overlap = []
    train_image_overlap = []
    train_smiles_overlap = []

    for line_no, row in read_jsonl(labels_path):
        total += 1
        record_id = str(row.get("id", ""))
        ids[record_id] += 1
        by_source[str(row.get("source", "unknown"))] += 1
        by_domain[str(row.get("weak_domain", "unknown"))] += 1
        by_difficulty[str(row.get("difficulty", "unknown"))] += 1

        smiles = get_smiles(row)
        canonical = canonicalize(Chem, smiles)
        if canonical is None:
            invalid_smiles.append({"line": line_no, "id": record_id, "smiles": smiles})
        else:
            canonical_smiles[canonical] += 1

        image_path = resolve_image(eval_root, row)
        if not image_path.exists():
            missing_images.append({"line": line_no, "id": record_id, "image": str(row.get("image", ""))})
        else:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                unreadable_images.append({"line": line_no, "id": record_id, "image": str(row.get("image", "")), "reason": str(exc)})
            if image_path.name.lower() in train_image_names and len(train_image_overlap) < limit_examples:
                train_image_overlap.append({"line": line_no, "id": record_id, "image_name": image_path.name})

        original_id = str(row.get("original_id", ""))
        if (record_id in train_ids or original_id in train_ids) and len(train_id_overlap) < limit_examples:
            train_id_overlap.append({"line": line_no, "id": record_id, "original_id": original_id})
        if canonical is not None and canonical in train_smiles and len(train_smiles_overlap) < limit_examples:
            train_smiles_overlap.append({"line": line_no, "id": record_id, "canonical_smiles": canonical})

    duplicate_ids = {key: value for key, value in ids.items() if value > 1}
    duplicate_smiles = {key: value for key, value in canonical_smiles.items() if value > 1}
    return {
        "eval_root": str(eval_root),
        "labels_path": str(labels_path),
        "train_jsonl": str(train_jsonl),
        "total": total,
        "rdkit_available": Chem is not None,
        "by_source": dict(by_source),
        "by_weak_domain": dict(by_domain),
        "by_difficulty": dict(by_difficulty),
        "unique_ids": len(ids),
        "duplicate_id_count": len(duplicate_ids),
        "unique_canonical_smiles": len(canonical_smiles),
        "duplicate_canonical_smiles_count": len(duplicate_smiles),
        "missing_images_count": len(missing_images),
        "unreadable_images_count": len(unreadable_images),
        "invalid_smiles_count": len(invalid_smiles),
        "train_id_overlap_count": len(train_id_overlap),
        "train_image_overlap_count": len(train_image_overlap),
        "train_smiles_overlap_count": len(train_smiles_overlap),
        "examples": {
            "missing_images": missing_images[:limit_examples],
            "unreadable_images": unreadable_images[:limit_examples],
            "invalid_smiles": invalid_smiles[:limit_examples],
            "train_id_overlap": train_id_overlap[:limit_examples],
            "train_image_overlap": train_image_overlap[:limit_examples],
            "train_smiles_overlap": train_smiles_overlap[:limit_examples],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--eval-root", default="V2-1/data/eval/weak_domain_v2")
    parser.add_argument("--train", default="V2-1/data/sft_materialized/train_singleline_rw_messages.jsonl")
    parser.add_argument("--report", default="V2-1/reports/weak_domain_v2_audit.json")
    parser.add_argument("--limit-examples", type=int, default=20)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    report = audit(
        eval_root=(project_root / args.eval_root).resolve(),
        train_jsonl=(project_root / args.train).resolve(),
        limit_examples=args.limit_examples,
    )
    report_path = (project_root / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
