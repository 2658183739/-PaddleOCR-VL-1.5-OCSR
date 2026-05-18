from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path


def load_audit_module():
    module_path = Path(__file__).resolve().with_name("audit_current_evalsets.py")
    spec = importlib.util.spec_from_file_location("audit_current_evalsets", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load audit_current_evalsets from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "source", "image", "task_type", "image_type", "difficulty", "smiles", "eval_target", "qc_status"])
        for row in records:
            writer.writerow([
                row["id"],
                row["source"],
                row["image"],
                row["task_type"],
                row.get("image_type", row.get("source", "unknown")),
                row["difficulty"],
                row["ground_truth"]["smiles"],
                row["eval_target"],
                row.get("qc_status", "pass"),
            ])


def build_problem_lookup(report: dict) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in report["targets"]["evaluation"]["records"]:
        if row["problem"]:
            pairs.add((row["dataset"], row["id"]))
    return pairs


def copy_image(project_root: Path, source_rel: str, dest_root: Path, source_name: str) -> str:
    src = (project_root / "V2" / "data" / "eval" / "canonical_smiles_main_v1" / source_rel).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Missing eval image: {src}")
    source_dir = dest_root / source_name
    source_dir.mkdir(parents=True, exist_ok=True)
    dest = source_dir / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
    return str(dest.relative_to(dest_root.parent).as_posix())


def build_curated_evalset(project_root: Path, out_root: Path) -> dict[str, object]:
    audit_module = load_audit_module()
    audit_report = audit_module.run_audit(project_root, out_root.parent / "canonical_smiles_curated_v2_audit.json")
    problems = build_problem_lookup(audit_report)

    source_labels = project_root / "V2" / "data" / "eval" / "canonical_smiles_main_v1" / "annotations" / "labels.jsonl"
    records = list(read_jsonl(source_labels))
    kept = []
    removed = 0
    for row in records:
        rid = str(row.get("id", ""))
        if ("canonical_smiles_main_v1", rid) in problems:
            removed += 1
            continue
        kept.append(row)

    annotations_root = out_root / "annotations"
    images_root = out_root / "images"
    curated_records = []
    for row in kept:
        rel_image = copy_image(project_root, row["image"], images_root, row["source"])
        new_row = dict(row)
        new_row["image"] = rel_image
        curated_records.append(new_row)

    write_jsonl(annotations_root / "labels.jsonl", curated_records)
    write_csv(annotations_root / "labels.csv", curated_records)

    by_source = Counter(r["source"] for r in curated_records)
    by_diff = Counter(r["difficulty"] for r in curated_records)
    by_task = Counter(r["task_type"] for r in curated_records)
    by_img = Counter(r.get("image_type", r.get("source", "unknown")) for r in curated_records)
    stats = {
        "total": len(curated_records),
        "by_source": dict(by_source),
        "by_difficulty": dict(by_diff),
        "by_task_type": dict(by_task),
        "by_image_type": dict(by_img),
        "eval_target": "canonical_smiles",
        "source_benchmark": "canonical_smiles_main_v1",
        "removed_problem_images": removed,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_root / "README.md").write_text(
        f"# canonical_smiles_curated_v2\n\n"
        f"A cleaned unified canonical-SMILES benchmark derived from canonical_smiles_main_v1.\n\n"
        f"- before: {len(records)}\n"
        f"- removed problematic images: {removed}\n"
        f"- after: {len(curated_records)}\n",
        encoding="utf-8",
    )
    (out_root / "QC_REPORT.md").write_text(
        f"# QC REPORT\n\n"
        f"- Source benchmark: canonical_smiles_main_v1\n"
        f"- Removed image-mode risk samples: {removed}\n"
        f"- Final total: {len(curated_records)}\n"
        f"- Main target: canonical_smiles\n",
        encoding="utf-8",
    )
    (out_root / "ANNOTATION_GUIDELINE.md").write_text(
        "# Annotation Guideline\n\n"
        "- Output target: canonical SMILES\n"
        "- One image corresponds to one target structure string\n"
        "- Derived from canonical_smiles_main_v1 after image-quality filtering\n",
        encoding="utf-8",
    )

    summary = {
        "before": len(records),
        "removed": removed,
        "after": len(curated_records),
        "output_root": str(out_root),
    }
    (out_root / "curation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-root", default="V2/data/eval/canonical_smiles_curated_v2")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = (project_root / args.out_root).resolve()
    summary = build_curated_evalset(project_root, out_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
