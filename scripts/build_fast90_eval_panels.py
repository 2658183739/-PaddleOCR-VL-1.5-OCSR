import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


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


def rewrite_image_ref(row: dict, source_root_rel: Path) -> dict:
    new_row = dict(row)
    image_ref = row.get("image") or row.get("image_path")
    if image_ref and not Path(str(image_ref)).is_absolute():
        new_row["image"] = (source_root_rel / image_ref).as_posix()
        new_row.pop("image_path", None)
    return new_row


def take_by_source(rows, source: str, limit: int):
    out = []
    for row in rows:
        if row.get("source") == source:
            out.append(row)
            if len(out) >= limit:
                break
    return out


def take_round_robin(rows, sources: list[str], per_source: int):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row.get("source")].append(row)
    selected = []
    for source in sources:
        selected.extend(buckets[source][:per_source])
    return selected


def summarize(rows):
    return {
        "total": len(rows),
        "source": dict(Counter(row.get("source", "") for row in rows)),
        "difficulty": dict(Counter(row.get("difficulty", "") for row in rows)),
        "task_type": dict(Counter(row.get("task_type", "") for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mixed-labels", default="V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl")
    parser.add_argument("--canonical-labels", default="V2-1/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl")
    parser.add_argument("--output-root", default="V2-1/reports/fast90_panels_v1")
    parser.add_argument("--uob-limit", type=int, default=80)
    parser.add_argument("--uob-smoke-limit", type=int, default=20)
    parser.add_argument("--balanced-per-source", type=int, default=20)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    mixed_labels = project_root / args.mixed_labels
    canonical_labels = project_root / args.canonical_labels
    output_root = project_root / args.output_root
    mixed_root_rel = Path("V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1")
    canonical_root_rel = Path("V2-1/data/eval/canonical_smiles_main_v1")

    mixed_rows = [rewrite_image_ref(row, mixed_root_rel) for row in read_jsonl(mixed_labels)]
    canonical_rows = [rewrite_image_ref(row, canonical_root_rel) for row in read_jsonl(canonical_labels)]

    panels = {
        "uob_medium_smoke20": take_by_source(mixed_rows, "uob", args.uob_smoke_limit),
        "uob_medium_80": take_by_source(mixed_rows, "uob", args.uob_limit),
        "mixed_uob_uspto_realworld_60": take_round_robin(
            mixed_rows,
            ["uob", "uspto", "real_world"],
            args.balanced_per_source,
        ),
        "canonical_decimer_hard_20": take_by_source(canonical_rows, "decimer", 20),
    }

    manifest = {}
    for name, rows in panels.items():
        labels_path = output_root / name / "annotations" / "labels.jsonl"
        write_jsonl(labels_path, rows)
        manifest[name] = {
            "labels": labels_path.relative_to(project_root).as_posix(),
            "summary": summarize(rows),
        }

    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
