from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


DEFAULT_SEED = 20260513


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


def get_smiles(row: dict) -> str:
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, dict) and ground_truth.get("smiles"):
        return str(ground_truth["smiles"]).strip()
    for key in ("canonical_smiles", "smiles", "label_summary"):
        if row.get(key):
            return str(row[key]).strip()
    return ""


def image_path_for_row(dataset_root: Path, row: dict) -> Path:
    image_value = row.get("image") or row.get("image_path")
    if not image_value:
        raise ValueError(f"Row {row.get('id', '<missing id>')} has no image field")
    image_path = Path(str(image_value))
    if image_path.is_absolute():
        return image_path
    return dataset_root / image_path


def verify_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def copy_image(src: Path, out_root: Path, source: str, new_id: str) -> str:
    suffix = src.suffix.lower() or ".png"
    target_dir = out_root / "images" / source
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{new_id}{suffix}"
    if not target.exists():
        shutil.copy2(src, target)
    return target.relative_to(out_root).as_posix()


def normalize_row(row: dict, dataset_root: Path, out_root: Path, domain: str, ordinal: int) -> dict:
    src_path = image_path_for_row(dataset_root, row)
    width, height = verify_image(src_path)
    source = str(row.get("source") or domain)
    new_source = {
        "decimer": "decimer_handdrawn",
        "real_world": "real_world_photo_scan",
        "edu_chemc": "edu_exam",
    }.get(source, source)
    new_id = f"weak_{new_source}_{ordinal:05d}"
    rel_image = copy_image(src_path, out_root, new_source, new_id)
    smiles = get_smiles(row)
    return {
        "id": new_id,
        "source": new_source,
        "original_id": row.get("id", ""),
        "original_source": row.get("source", ""),
        "image": rel_image,
        "task_type": "molecule_structure_recognition",
        "image_type": row.get("image_type", new_source),
        "difficulty": row.get("difficulty", domain),
        "ground_truth": {
            "smiles": smiles,
            "inchi": None,
            "selfies": None,
            "mol": None,
        },
        "eval_target": "canonical_smiles",
        "license": row.get("license", "mixed_public_and_team_curated"),
        "source_url_or_doc": row.get("source_url_or_doc", row.get("source", "")),
        "qc_status": row.get("qc_status", "pass"),
        "benchmark_track": "weak_domain_v2",
        "weak_domain": domain,
        "label_summary": smiles,
        "image_size": [width, height],
    }


def sample_rows(rows: list[dict], cap: int, seed: int) -> list[dict]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if cap <= 0 or cap >= len(rows):
        return rows
    return rows[:cap]


def load_source_rows(labels_path: Path, dataset_root: Path, wanted_sources: set[str]) -> list[tuple[dict, Path]]:
    loaded = []
    for row in read_jsonl(labels_path):
        if str(row.get("source", "")) not in wanted_sources:
            continue
        loaded.append((row, dataset_root))
    return loaded


def build_eval(project_root: Path, out_root: Path, seed: int, include_long_stereo: bool) -> dict:
    eval_root = project_root / "V2-1" / "data" / "eval"
    canonical_root = eval_root / "canonical_smiles_main_v1"
    mixed_root = eval_root / "ocsr_realworld_mixed_eval_v1p1"
    canonical_labels = canonical_root / "annotations" / "labels.jsonl"
    mixed_labels = mixed_root / "annotations" / "labels.jsonl"

    candidates: list[tuple[str, dict, Path]] = []
    for row, root in load_source_rows(canonical_labels, canonical_root, {"decimer"}):
        candidates.append(("decimer_handdrawn", row, root))
    for row, root in load_source_rows(canonical_labels, canonical_root, {"real_world"}):
        candidates.append(("real_world_photo_scan", row, root))
    for row, root in load_source_rows(mixed_labels, mixed_root, {"edu_chemc"}):
        candidates.append(("edu_exam", row, root))

    if include_long_stereo:
        stress_rows = []
        for row, root in load_source_rows(canonical_labels, canonical_root, {"uob", "uspto"}):
            smiles = get_smiles(row)
            if len(smiles) >= 100 or "@" in smiles or "/" in smiles or "\\" in smiles:
                stress_rows.append((row, root))
        for row, root in sample_rows(stress_rows, cap=100, seed=seed + 17):
            candidates.append(("long_or_stereo", row, root))

    rows = []
    seen_smiles = set()
    skipped_duplicate_smiles = 0
    for ordinal, (domain, row, root) in enumerate(candidates):
        smiles = get_smiles(row)
        if not smiles:
            continue
        if smiles in seen_smiles:
            skipped_duplicate_smiles += 1
            continue
        seen_smiles.add(smiles)
        rows.append(normalize_row(row, root, out_root, domain, ordinal))

    write_jsonl(out_root / "annotations" / "labels.jsonl", rows)
    write_csv(out_root / "annotations" / "labels.csv", rows)

    by_source = Counter(row["source"] for row in rows)
    by_domain = Counter(row["weak_domain"] for row in rows)
    by_difficulty = Counter(row["difficulty"] for row in rows)
    stats = {
        "name": "weak_domain_v2",
        "total": len(rows),
        "seed": seed,
        "include_long_stereo": include_long_stereo,
        "skipped_duplicate_smiles": skipped_duplicate_smiles,
        "by_source": dict(by_source),
        "by_weak_domain": dict(by_domain),
        "by_difficulty": dict(by_difficulty),
        "sources": {
            "canonical_smiles_main_v1": str(canonical_labels),
            "ocsr_realworld_mixed_eval_v1p1": str(mixed_labels),
        },
        "notes": [
            "This is a weak-domain evaluation seed assembled from existing held-out eval sources.",
            "Do not train on these images or canonical SMILES.",
            "Use this set to select V2-2 checkpoints after adding weak-domain training data.",
        ],
    }
    (out_root / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "README.md").write_text(readme_text(stats), encoding="utf-8")
    return stats


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "source",
        "original_id",
        "weak_domain",
        "image",
        "task_type",
        "difficulty",
        "label_summary",
        "eval_target",
        "qc_status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def readme_text(stats: dict) -> str:
    return (
        "# Weak Domain Evaluation Set v2\n\n"
        "This evaluation set focuses on the current V2-1 weak domains: hand-drawn DECIMER-style structures, "
        "real-world photos/scans, education/exam-style structures, and optional long/stereo stress cases.\n\n"
        "## Scale\n\n"
        f"- Total: {stats['total']}\n"
        f"- By weak domain: `{json.dumps(stats['by_weak_domain'], ensure_ascii=False)}`\n"
        f"- By source: `{json.dumps(stats['by_source'], ensure_ascii=False)}`\n\n"
        "## Files\n\n"
        "- `images/`: copied images grouped by source.\n"
        "- `annotations/labels.jsonl`: primary labels in the current OCSR schema.\n"
        "- `annotations/labels.csv`: lightweight review sheet.\n"
        "- `stats.json`: construction summary.\n\n"
        "## Guardrail\n\n"
        "Do not include any sample from this set in training. Filter training data by canonical SMILES, image filename, and ID.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="V2-1/data/eval/weak_domain_v2")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-long-stereo", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = (project_root / args.output).resolve()
    stats = build_eval(
        project_root=project_root,
        out_root=out_root,
        seed=args.seed,
        include_long_stereo=not args.no_long_stereo,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
