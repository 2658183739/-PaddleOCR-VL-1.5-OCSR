#!/usr/bin/env python3
"""Validate and import self-collected photo/handwritten OCSR samples into V3.

The input CSV is deliberately strict so the provenance and human review trail
cannot be silently omitted. Images are split by structure_id, never by photo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
from rdkit import Chem


REQUIRED_COLUMNS = {
    "sample_id",
    "structure_id",
    "image_path",
    "canonical_smiles",
    "capture_device",
    "capture_condition",
    "angle_deg",
    "lighting",
    "collector",
    "capture_time",
    "reviewer_1",
    "reviewer_1_decision",
    "reviewer_2",
    "reviewer_2_decision",
    "license_consent",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def canonicalize(value: str) -> str:
    current = re.sub(r"\s+", "", value)
    if not current or "." in current:
        raise ValueError(f"Invalid, empty, or multi-fragment SMILES: {value}")
    seen: set[str] = set()
    for _ in range(16):
        if current in seen:
            raise ValueError(f"SMILES canonicalization did not converge: {value}")
        seen.add(current)
        mol = Chem.MolFromSmiles(current)
        if mol is None or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            raise ValueError(f"Invalid or symbolic SMILES: {value}")
        normalized = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if normalized == current:
            return normalized
        current = normalized
    raise ValueError(f"SMILES canonicalization did not converge: {value}")


def assistant_text(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages", [])):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def ground_truth_text(row: dict[str, Any]) -> str:
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, dict):
        return str(ground_truth.get("smiles", ""))
    return str(row.get("smiles", ""))


def split_for_structure(structure_id: str, eval_fraction: float) -> str:
    value = int(hashlib.sha256(f"20260717|{structure_id}".encode()).hexdigest()[:8], 16)
    return "eval" if value / 0xFFFFFFFF < eval_fraction else "train"


def requested_structure_split(
    structure_id: str, eval_fraction: float, requested: str = ""
) -> str:
    split = requested.strip().lower()
    if split:
        if split not in {"train", "eval"}:
            raise ValueError(f"split must be train or eval, got: {requested}")
        return split
    return split_for_structure(structure_id, eval_fraction)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--eval-fraction", type=float, default=0.30)
    args = parser.parse_args()

    root = args.project_root.resolve()
    v3 = root / "V3"
    if not 0.2 <= args.eval_fraction <= 0.5:
        raise ValueError("--eval-fraction must be between 0.2 and 0.5")

    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        source_rows = list(reader)

    existing_train = read_jsonl(v3 / "data/sft_materialized/train_v3_b_recommended.jsonl")
    train_smiles = {canonicalize(assistant_text(row)) for row in existing_train}
    existing_eval_paths = [
        v3 / "data/eval/dev_legacy_core_strict/labels.jsonl",
        v3 / "data/eval/dev_legacy_region_strict/labels.jsonl",
        v3 / "data/eval/wild_strict_v3/labels.jsonl",
    ]
    eval_smiles = {
        canonicalize(ground_truth_text(row))
        for path in existing_eval_paths
        for row in read_jsonl(path)
    }
    seen_ids: set[str] = set()
    structure_split: dict[str, str] = {}
    structure_canonical: dict[str, str] = {}
    private_train_smiles: set[str] = set()
    private_eval_smiles: set[str] = set()
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for row_number, row in enumerate(source_rows, 2):
        sample_id = row["sample_id"].strip()
        structure_id = row["structure_id"].strip()
        if not sample_id or not structure_id:
            raise ValueError(f"Row {row_number}: sample_id and structure_id are required")
        if sample_id in seen_ids:
            raise ValueError(f"Row {row_number}: duplicate sample_id {sample_id}")
        seen_ids.add(sample_id)
        if row["license_consent"].strip().lower() not in {"yes", "true", "1"}:
            raise ValueError(f"Row {row_number}: license_consent must be yes")
        if row["reviewer_1_decision"].strip().lower() != "pass":
            raise ValueError(f"Row {row_number}: reviewer_1_decision must be pass")
        if row["reviewer_2_decision"].strip().lower() != "pass":
            raise ValueError(f"Row {row_number}: reviewer_2_decision must be pass")

        source_image = Path(row["image_path"])
        if not source_image.is_absolute():
            source_image = (args.csv.parent / source_image).resolve()
        if not source_image.exists():
            raise FileNotFoundError(f"Row {row_number}: image not found: {source_image}")
        with Image.open(source_image) as image:
            image.verify()

        canonical = canonicalize(row["canonical_smiles"])
        previous_canonical = structure_canonical.setdefault(structure_id, canonical)
        if previous_canonical != canonical:
            raise ValueError(
                f"Row {row_number}: structure_id {structure_id} maps to multiple molecules"
            )
        proposed_split = requested_structure_split(
            structure_id, args.eval_fraction, row.get("split", "")
        )
        split = structure_split.setdefault(structure_id, proposed_split)
        if split != proposed_split:
            raise ValueError(
                f"Row {row_number}: structure_id {structure_id} appears in both train and eval"
            )
        if split == "eval":
            if canonical in train_smiles or canonical in private_train_smiles:
                raise ValueError(
                    f"Row {row_number}: eval molecule overlaps train: {structure_id}"
                )
            private_eval_smiles.add(canonical)
        else:
            if canonical in eval_smiles or canonical in private_eval_smiles:
                raise ValueError(
                    f"Row {row_number}: train molecule overlaps evaluation: {structure_id}"
                )
            private_train_smiles.add(canonical)
        suffix = source_image.suffix.lower() or ".jpg"
        if split == "eval":
            destination = v3 / "data/eval/private_photo_v3/images" / f"{sample_id}{suffix}"
        else:
            destination = v3 / "data/assets/private_photo_train_v3" / f"{sample_id}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, destination)

        provenance = {
            "structure_id": structure_id,
            "capture_device": row["capture_device"].strip(),
            "capture_condition": row["capture_condition"].strip(),
            "angle_deg": row["angle_deg"].strip(),
            "lighting": row["lighting"].strip(),
            "collector": row["collector"].strip(),
            "capture_time": row["capture_time"].strip(),
            "reviewer_1": row["reviewer_1"].strip(),
            "reviewer_2": row["reviewer_2"].strip(),
            "license_consent": True,
        }
        if split == "eval":
            eval_rows.append(
                {
                    "id": sample_id,
                    "source": "team_self_collected",
                    "structure_id": structure_id,
                    "paper_group": "team_self_collected",
                    "image": f"V3/data/eval/private_photo_v3/images/{destination.name}",
                    "task_type": "molecule_structure_recognition",
                    "image_type": "self_collected_photo_or_handwritten",
                    "difficulty": row["capture_condition"].strip(),
                    "ground_truth": {"smiles": canonical},
                    "eval_target": "canonical_smiles",
                    "license": "team_collected_with_consent",
                    "source_url_or_doc": "V3/qc/private_photo_collection.csv",
                    "qc_status": "double_review_pass",
                    "benchmark_role": "locked_final_test",
                    "provenance": provenance,
                }
            )
        else:
            train_rows.append(
                {
                    "images": [f"../assets/private_photo_train_v3/{destination.name}"],
                    "messages": [
                        {
                            "role": "user",
                            "content": "<image>OCR: Output only the canonical SMILES string for the molecule shown in the image.",
                        },
                        {"role": "assistant", "content": canonical},
                    ],
                    "meta": {
                        "id": sample_id,
                        "source": "team_self_collected",
                        "difficulty": row["capture_condition"].strip(),
                        "task_type": "molecule_structure_recognition",
                        "eval_target": "canonical_smiles",
                        "provenance": provenance,
                    },
                }
            )
        counters[split] += 1
        counters[f"condition:{row['capture_condition'].strip()}"] += 1

    write_jsonl(v3 / "data/sft_materialized/train_v3_private_photo_addon.jsonl", train_rows)
    write_jsonl(v3 / "data/eval/private_photo_v3/labels.jsonl", eval_rows)
    report = {
        "input_rows": len(source_rows),
        "unique_structures": len(structure_split),
        "unique_train_molecules": len(private_train_smiles),
        "unique_eval_molecules": len(private_eval_smiles),
        "counts": dict(sorted(counters.items())),
        "split_policy": "explicit CSV split when provided, otherwise SHA256 seeded group split by structure_id",
        "eval_fraction": args.eval_fraction,
        "qc": "two named reviewers required to pass before import",
    }
    (v3 / "evidence/private_photo_import_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
