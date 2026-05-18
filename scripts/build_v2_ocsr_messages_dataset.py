import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROMPT = "OCR: Output only the canonical SMILES string for the molecule shown in the image."

CORE_REAL_SOURCES = {"uob", "uspto", "real_world"}
HARD_REAL_SOURCES = {"decimer"}
STRUCTURAL_SYNTHETIC_SOURCES = {"uspto30k_abbreviated", "uspto30k_large"}
SYNTHETIC_EASY_SOURCES = {"uspto30k_clean"}
VISUAL_ROBUSTNESS_SOURCES = {"molgrapher_synthetic"}


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


def make_message_record(meta):
    image_path = str(meta["image_path"]).replace("\\", "/")
    if not image_path.startswith("./") and not image_path.startswith("http"):
        image_path = f"./{image_path}"
    return {
        "messages": [
            {"role": "user", "content": f"<image>{PROMPT}"},
            {"role": "assistant", "content": meta["canonical_smiles"]},
        ],
        "images": [image_path],
        "meta": {
            "id": meta["id"],
            "source": meta["source"],
            "difficulty": meta.get("difficulty", "unknown"),
            "task_type": meta.get("task_type", "molecule_structure_recognition"),
            "canonical_smiles_length": len(meta["canonical_smiles"]),
            "contains_stereo": "@" in meta["canonical_smiles"],
        },
    }


def looks_structural_hard(meta):
    smiles = meta["canonical_smiles"]
    return (
        len(smiles) >= 80
        or "@" in smiles
        or "[" in smiles
        or meta.get("difficulty", "") in {"large", "abbreviated", "hard", "medium_hard"}
    )


def bucket_records(records):
    buckets = defaultdict(list)
    for meta in records:
        source = meta["source"]
        if source in CORE_REAL_SOURCES:
            buckets["core_real"].append(meta)
        elif source in HARD_REAL_SOURCES:
            buckets["hard_real"].append(meta)
        elif source in STRUCTURAL_SYNTHETIC_SOURCES:
            buckets["structural_synthetic"].append(meta)
        elif source in SYNTHETIC_EASY_SOURCES:
            buckets["synthetic_easy"].append(meta)
        elif source in VISUAL_ROBUSTNESS_SOURCES:
            buckets["visual_robustness"].append(meta)
        else:
            buckets["other"].append(meta)

        if looks_structural_hard(meta):
            buckets["structural_hard_all"].append(meta)
    return buckets


def deterministic_sample(records, limit, seed):
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    if limit <= 0 or limit >= len(items):
        return items
    return items[:limit]


def sort_records(records):
    return sorted(records, key=lambda item: (item["source"], item["id"]))


def build_phase_records(buckets):
    phase1 = []
    phase1.extend(sort_records(buckets["core_real"]))
    phase1.extend(sort_records(deterministic_sample(buckets["hard_real"], 800, 101)))
    phase1.extend(sort_records(deterministic_sample(buckets["synthetic_easy"], 3000, 102)))

    phase2 = []
    phase2.extend(sort_records(buckets["core_real"]))
    phase2.extend(sort_records(deterministic_sample(buckets["hard_real"], 1500, 201)))
    phase2.extend(sort_records(deterministic_sample(buckets["synthetic_easy"], 3000, 202)))
    phase2.extend(sort_records(deterministic_sample(buckets["structural_synthetic"], 5000, 203)))

    phase3 = []
    phase3.extend(sort_records(buckets["core_real"]))
    phase3.extend(sort_records(deterministic_sample(buckets["hard_real"], 2500, 301)))
    phase3.extend(sort_records(deterministic_sample(buckets["synthetic_easy"], 3000, 302)))
    phase3.extend(sort_records(deterministic_sample(buckets["structural_synthetic"], 5000, 303)))
    phase3.extend(sort_records(deterministic_sample(buckets["visual_robustness"], 2000, 304)))

    return {
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
    }


def count_sources(records):
    return dict(Counter(item["source"] for item in records))


def count_difficulties(records):
    return dict(Counter(item.get("difficulty", "unknown") for item in records))


def write_summary(path: Path, buckets, phases, val_records):
    summary = {
        "bucket_counts": {name: len(items) for name, items in buckets.items()},
        "bucket_source_counts": {name: count_sources(items) for name, items in buckets.items()},
        "phase_counts": {name: len(items) for name, items in phases.items()},
        "phase_source_counts": {name: count_sources(items) for name, items in phases.items()},
        "phase_difficulty_counts": {name: count_difficulties(items) for name, items in phases.items()},
        "val_count": len(val_records),
        "val_source_counts": count_sources(val_records),
        "prompt": PROMPT,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-meta", default="V2/data/meta/train_meta_source.jsonl")
    parser.add_argument("--val-meta", default="V2/data/meta/val_meta_source.jsonl")
    parser.add_argument("--out-dir", default="V2/data")
    parser.add_argument("--report-path", default="V2/reports/v2_dataset_summary.json")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    train_meta_path = (project_root / args.train_meta).resolve()
    val_meta_path = (project_root / args.val_meta).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    report_path = (project_root / args.report_path).resolve()

    train_records = list(read_jsonl(train_meta_path))
    val_records = list(read_jsonl(val_meta_path))

    buckets = bucket_records(train_records)
    phases = build_phase_records(buckets)

    manifests_dir = out_dir / "manifests"
    sft_dir = out_dir / "sft"

    for bucket_name, items in buckets.items():
        write_jsonl(manifests_dir / f"{bucket_name}.jsonl", sort_records(items))

    for phase_name, items in phases.items():
        write_jsonl(manifests_dir / f"{phase_name}_train_meta.jsonl", items)
        write_jsonl(sft_dir / f"train_{phase_name}_messages.jsonl", [make_message_record(item) for item in items])

    write_jsonl(manifests_dir / "val_meta_v2.jsonl", sort_records(val_records))
    write_jsonl(sft_dir / "val_messages.jsonl", [make_message_record(item) for item in sort_records(val_records)])
    write_summary(report_path, buckets, phases, val_records)

    print(f"train_records={len(train_records)}")
    print(f"val_records={len(val_records)}")
    for phase_name, items in phases.items():
        print(f"{phase_name}={len(items)}")
    print(f"summary={report_path}")


if __name__ == "__main__":
    main()
