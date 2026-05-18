from __future__ import annotations

import argparse
import csv
import json
import tarfile
import zipfile
from pathlib import Path

from datasets import load_dataset


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_junk_filename(name: str) -> bool:
    return name.startswith("._") or name.startswith("__MACOSX") or name.endswith(".DS_Store")


CSV_HEADER = [
    "id",
    "source",
    "source_type",
    "image",
    "task_type",
    "image_type",
    "difficulty",
    "smiles",
    "eval_target",
    "license",
    "attribution",
    "collector",
    "collection_date",
    "source_url_or_doc",
    "qc_status",
]


def append_label_rows(bundle_root: Path, jsonl_rows: list[dict]) -> None:
    labels_out = bundle_root / "annotations" / "labels.jsonl"
    csv_out = bundle_root / "annotations" / "labels.csv"
    ensure_dir(labels_out.parent)

    existing_ids = set()
    if labels_out.exists():
        with labels_out.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line)["id"])

    rows_to_add = [row for row in jsonl_rows if row["id"] not in existing_ids]

    with labels_out.open("a", encoding="utf-8") as f:
        for row in rows_to_add:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_exists = csv_out.exists()
    with csv_out.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(CSV_HEADER)
        for row in rows_to_add:
            writer.writerow([
                row["id"],
                row["source"],
                row["source_type"],
                row["image"],
                row["task_type"],
                row["image_type"],
                row["difficulty"],
                row["ground_truth"]["smiles"],
                row["eval_target"],
                row["license"],
                row["attribution"],
                row["collector"],
                row["collection_date"],
                row["source_url_or_doc"],
                row["qc_status"],
            ])


def copy_zip_images(zip_path: Path, out_dir: Path, prefix: str = "") -> int:
    ensure_dir(out_dir)
    copied = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name
            if is_junk_filename(name):
                continue
            if prefix and not name.startswith(prefix):
                # for SMiCRM we keep everything; prefix optional for future datasets
                pass
            out_path = out_dir / name
            with zf.open(member) as src, out_path.open("wb") as dst:
                dst.write(src.read())
            copied += 1
    return copied


def import_smicrm(source_dir: Path, bundle_root: Path) -> dict:
    images_dir = bundle_root / "images" / "public_cc_by" / "smicrm"
    ensure_dir(images_dir)

    mechanism_zip = source_dir / "mechanism.zip"
    mechanism_csv = source_dir / "mechanism.csv"
    if not mechanism_zip.exists():
        raise FileNotFoundError(f"Missing SMiCRM archive: {mechanism_zip}")
    if not mechanism_csv.exists():
        raise FileNotFoundError(f"Missing SMiCRM csv: {mechanism_csv}")

    # Re-import should be idempotent for the SMiCRM directory.
    copied = copy_zip_images(mechanism_zip, images_dir)

    rows = []
    with mechanism_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows.append(row)

    jsonl_rows = []
    for idx, row in enumerate(rows):
        image_name = Path(row["file_path"]).name
        jsonl_rows.append(
            {
                "id": f"smicrm_{idx:05d}",
                "source": "smicrm",
                "source_type": "public_cc_by",
                "image": f"images/public_cc_by/smicrm/{image_name}",
                "task_type": "mechanistic_molecular_image_recognition",
                "image_type": "mechanistic_molecular_image",
                "difficulty": "hard",
                "ground_truth": {"smiles": row.get("SMILES", ""), "inchi": None, "selfies": None, "mol": None},
                "eval_target": "canonical_smiles",
                "license": "CC BY 4.0",
                "attribution": "SMiCRM: A Benchmark Dataset of Mechanistic Molecular Images",
                "collector": "import_supplement_source.py",
                "collection_date": "2026-05-09",
                "source_url_or_doc": "https://doi.org/10.5281/zenodo.11060696",
                "qc_status": "pass",
            }
        )

    append_label_rows(bundle_root, jsonl_rows)

    return {"source": "smicrm", "copied_files": copied, "labels": len(jsonl_rows)}


def import_markush_subset(subset_name: str, bundle_root: Path, hf_repo: str = "docling-project/MarkushGrapher-2-Datasets") -> dict:
    out_dir = bundle_root / "images" / "public_cc_by" / "markushgrapher2" / subset_name.replace("-", "_")
    ensure_dir(out_dir)

    ds = load_dataset(hf_repo, subset_name, split="test")
    jsonl_rows = []
    for idx, row in enumerate(ds):
        image_name = row.get("image_name") or f"{row['id']}.png"
        # page_image is a PIL image object in the HF dataset; save it to disk.
        image = row["page_image"]
        image_path = out_dir / image_name
        image.save(image_path)

        cxsmiles = row.get("cxsmiles") or row.get("cxsmiles_opt") or row.get("cxsmiles_dataset") or ""
        jsonl_rows.append(
            {
                "id": f"{subset_name.replace('-', '_')}_{idx:05d}",
                "source": subset_name,
                "source_type": "public_cc_by",
                "image": str(image_path.relative_to(bundle_root).as_posix()),
                "task_type": "markush_or_chemical_ocr_recognition",
                "image_type": "patent_markush_structure",
                "difficulty": "hard",
                "ground_truth": {"smiles": cxsmiles, "inchi": None, "selfies": None, "mol": None},
                "eval_target": "canonical_smiles",
                "license": "CC BY 4.0",
                "attribution": "MarkushGrapher-2 benchmark subset",
                "collector": "import_supplement_source.py",
                "collection_date": "2026-05-09",
                "source_url_or_doc": f"https://huggingface.co/datasets/{hf_repo}",
                "qc_status": "pass",
            }
        )

    append_label_rows(bundle_root, jsonl_rows)

    return {"source": subset_name, "labels": len(jsonl_rows), "downloaded": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["smicrm", "m2s", "uspto-markush", "ip5-markush"])
    parser.add_argument("--source-dir", default=".")
    parser.add_argument("--bundle-root", default="V2/data/eval/ocsr_supplement_sources_v1")
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    if args.source == "smicrm":
        result = import_smicrm(Path(args.source_dir).resolve(), bundle_root)
    else:
        result = import_markush_subset(args.source, bundle_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
