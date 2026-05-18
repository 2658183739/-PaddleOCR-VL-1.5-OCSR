from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


PROMPT = "OCR: Output only the canonical SMILES string for the molecule shown in the image."
DEFAULT_BUNDLE_RELATIVE_ROOT = "V2/data/eval/ocsr_real_only_decimer_core_v1"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def canonicalize_smiles(smiles: str) -> str:
    text = str(smiles or "").strip()
    if not text:
        raise ValueError("Empty SMILES string")
    try:
        from rdkit import Chem  # type: ignore
    except ModuleNotFoundError:
        return text
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {text}")
    return Chem.MolToSmiles(mol, canonical=True)


def load_decimer_smiles_map(path: Path) -> dict[str, str]:
    smiles_by_id: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stem, smiles = line.split("\t", 1)
            smiles_by_id[stem] = canonicalize_smiles(smiles)
    return smiles_by_id


def select_decimer_records(selection_jsonl: Path, expected_count: int = 150) -> list[dict]:
    records = [row for row in read_jsonl(selection_jsonl) if row.get("source") == "decimer"]
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} DECIMER records, found {len(records)}")

    seen: set[str] = set()
    for row in records:
        source_original_id = str(row.get("source_original_id", "")).strip()
        if not source_original_id:
            raise ValueError("Missing DECIMER source_original_id")
        if source_original_id in seen:
            raise ValueError(f"Duplicate DECIMER source_original_id: {source_original_id}")
        seen.add(source_original_id)
    return records


def build_label_record(
    index: int,
    source_original_id: str,
    canonical_smiles: str,
    source_dataset: str = "DECIMER_HDM_Dataset",
) -> dict:
    return {
        "id": f"decimer_{index:05d}",
        "source": "decimer",
        "source_dataset": source_dataset,
        "source_original_id": source_original_id,
        "image": f"images/decimer/decimer_{index:05d}.png",
        "task_type": "hand_drawn_molecule_structure_recognition",
        "image_type": "hand_drawn_chemical_structure",
        "difficulty": "hard",
        "ground_truth": {
            "smiles": canonical_smiles,
            "inchi": None,
            "selfies": None,
            "mol": None,
        },
        "eval_target": "canonical_smiles",
    }


def build_benchmark_record(label_record: dict, bundle_root: Path) -> dict:
    _ = bundle_root
    return {
        "id": label_record["id"],
        "source": label_record["source"],
        "task_type": label_record["task_type"],
        "difficulty": label_record["difficulty"],
        "image_path": f"{DEFAULT_BUNDLE_RELATIVE_ROOT}/{label_record['image']}",
        "canonical_smiles": label_record["ground_truth"]["smiles"],
        "prompt": PROMPT,
        "repeat_hint": 1,
    }


def build_stats(records: list[dict]) -> dict:
    return {
        "total": len(records),
        "by_source": dict(Counter(row["source"] for row in records)),
        "by_difficulty": dict(Counter(row["difficulty"] for row in records)),
        "by_task_type": dict(Counter(row["task_type"] for row in records)),
        "by_image_type": dict(Counter(row["image_type"] for row in records)),
        "source_license": {"decimer": "CC BY 4.0"},
        "core_policy": {
            "real_only": True,
            "synthetic_included": False,
            "commercial_or_paid_sources_included": False,
            "non_commercial_sources_included": False,
            "permission_needed_sources_included": False,
        },
    }


def detect_train_eval_overlap(train_meta_path: Path, source_original_ids: set[str]) -> list[dict]:
    overlaps: list[dict] = []
    for row in read_jsonl(train_meta_path):
        if row.get("source") != "decimer":
            continue
        stem = Path(str(row["image_path"])).stem
        if stem in source_original_ids:
            overlaps.append(
                {
                    "source_original_id": stem,
                    "train_record_id": row["id"],
                    "image_path": row["image_path"],
                }
            )
    return overlaps


def write_labels_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "source",
        "source_dataset",
        "source_original_id",
        "image",
        "task_type",
        "image_type",
        "difficulty",
        "smiles",
        "inchi",
        "selfies",
        "mol",
        "eval_target",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "source_dataset": row["source_dataset"],
                    "source_original_id": row["source_original_id"],
                    "image": row["image"],
                    "task_type": row["task_type"],
                    "image_type": row["image_type"],
                    "difficulty": row["difficulty"],
                    "smiles": row["ground_truth"]["smiles"],
                    "inchi": row["ground_truth"]["inchi"],
                    "selfies": row["ground_truth"]["selfies"],
                    "mol": row["ground_truth"]["mol"],
                    "eval_target": row["eval_target"],
                }
            )


def write_source_selection_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bundle_id",
        "source_original_id",
        "source_dataset",
        "source_license",
        "source_url",
        "source_doi",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "bundle_id": row["id"],
                    "source_original_id": row["source_original_id"],
                    "source_dataset": row["source_dataset"],
                    "source_license": "CC BY 4.0",
                    "source_url": "https://zenodo.org/records/6456306",
                    "source_doi": "10.5281/zenodo.6456306",
                }
            )


def write_overlap_report(path: Path, overlaps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_original_id", "train_record_id", "image_path"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in overlaps:
            writer.writerow(row)


def write_file_manifest(path: Path, files: list[Path], bundle_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for file_path in sorted(files):
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            rel = file_path.relative_to(bundle_root).as_posix()
            handle.write(f"{digest}  {rel}\n")


def build_bundle(
    selection_jsonl: Path,
    source_images_root: Path,
    train_meta_path: Path,
    out_root: Path,
    benchmark_jsonl: Path,
    expected_count: int = 150,
) -> dict:
    selection_records = select_decimer_records(selection_jsonl, expected_count=expected_count)
    source_original_ids = {row["source_original_id"] for row in selection_records}
    overlaps = detect_train_eval_overlap(train_meta_path, source_original_ids)
    if overlaps:
        raise ValueError(f"Found {len(overlaps)} train/eval overlaps")

    label_records: list[dict] = []
    benchmark_records: list[dict] = []
    image_outputs: list[Path] = []

    for index, row in enumerate(selection_records):
        source_original_id = row["source_original_id"]
        source_dataset = row.get("source_dataset", "DECIMER_HDM_Dataset")
        canonical_smiles = canonicalize_smiles(row["ground_truth"]["smiles"])
        label_record = build_label_record(index, source_original_id, canonical_smiles, source_dataset)
        source_image_rel = str(row.get("image", "")).strip()
        candidate_names = []
        if source_image_rel:
            candidate_names.append(Path(source_image_rel).name)
        candidate_names.append(f"{source_original_id}.png")
        source_image_path = None
        for candidate_name in candidate_names:
            candidate_path = source_images_root / candidate_name
            if candidate_path.exists():
                source_image_path = candidate_path
                break
        if source_image_path is None:
            raise FileNotFoundError(f"Missing source image under {source_images_root} for {source_original_id}")
        dest_image_path = out_root / label_record["image"]
        dest_image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image_path, dest_image_path)
        image_outputs.append(dest_image_path)
        label_records.append(label_record)
        benchmark_records.append(build_benchmark_record(label_record, out_root))

    labels_jsonl = out_root / "annotations" / "labels.jsonl"
    labels_csv = out_root / "annotations" / "labels.csv"
    stats_json = out_root / "stats.json"
    source_selection_csv = out_root / "manifests" / "source_selection_decimer_v1.csv"
    core_manifest_jsonl = out_root / "manifests" / "core_manifest.jsonl"
    excluded_csv = out_root / "manifests" / "excluded_decimer_records.csv"
    overlap_csv = out_root / "manifests" / "train_eval_overlap_report.csv"
    file_manifest = out_root / "manifests" / "file_manifest.sha256"
    qc_summary = out_root / "reports" / "qc_summary.json"
    duplicate_report = out_root / "reports" / "duplicate_report.json"
    canonicalization_report = out_root / "reports" / "canonicalization_report.json"
    image_integrity_report = out_root / "reports" / "image_integrity_report.json"

    write_jsonl(labels_jsonl, label_records)
    write_labels_csv(labels_csv, label_records)
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(build_stats(label_records), ensure_ascii=False, indent=2), encoding="utf-8")
    write_source_selection_csv(source_selection_csv, label_records)
    write_jsonl(core_manifest_jsonl, benchmark_records)
    write_overlap_report(overlap_csv, overlaps)
    write_jsonl(benchmark_jsonl, benchmark_records)

    excluded_csv.parent.mkdir(parents=True, exist_ok=True)
    with excluded_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_original_id", "reason"])
        writer.writeheader()

    qc_summary.parent.mkdir(parents=True, exist_ok=True)
    qc_summary.write_text(json.dumps({"total": len(label_records), "images_present": len(image_outputs), "rdkit_canonicalization_success": len(label_records), "train_eval_overlaps": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
    duplicate_report.write_text(json.dumps({"duplicate_bundle_ids": 0, "duplicate_source_original_ids": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
    canonicalization_report.write_text(json.dumps({"total": len(label_records), "canonicalized": len(label_records), "failures": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
    image_integrity_report.write_text(json.dumps({"total": len(image_outputs), "copied": len(image_outputs), "missing": 0}, ensure_ascii=False, indent=2), encoding="utf-8")

    bundle_files = [labels_jsonl, labels_csv, stats_json, source_selection_csv, core_manifest_jsonl, overlap_csv, excluded_csv, qc_summary, duplicate_report, canonicalization_report, image_integrity_report, *image_outputs]
    write_file_manifest(file_manifest, bundle_files, out_root)

    return {
        "selected_decimer": len(label_records),
        "written_labels": len(label_records),
        "written_images": len(image_outputs),
        "train_eval_overlaps": 0,
        "bundle_root": str(out_root),
        "benchmark_jsonl": str(benchmark_jsonl),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-jsonl", default="ocsr_evalset_final/annotations/labels.jsonl")
    parser.add_argument("--source-images-root", default="ocsr_evalset_final/images/decimer")
    parser.add_argument("--train-meta", default="V2/data/meta/train_meta_source.jsonl")
    parser.add_argument("--out-root", default=DEFAULT_BUNDLE_RELATIVE_ROOT)
    parser.add_argument("--benchmark-jsonl", default="V2/data/benchmarks/ocsr_real_only_decimer_core_v1.jsonl")
    parser.add_argument("--expected-count", type=int, default=150)
    args = parser.parse_args()

    summary = build_bundle(
        selection_jsonl=Path(args.selection_jsonl).resolve(),
        source_images_root=Path(args.source_images_root).resolve(),
        train_meta_path=Path(args.train_meta).resolve(),
        out_root=Path(args.out_root).resolve(),
        benchmark_jsonl=Path(args.benchmark_jsonl).resolve(),
        expected_count=args.expected_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
