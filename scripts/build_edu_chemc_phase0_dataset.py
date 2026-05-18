import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


PROMPT = "OCR: Output only the normalized chemistry structure string (ssml_normed) for the molecule shown in the image."


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def infer_split(path: Path) -> str:
    lowered = [part.lower() for part in path.parts]
    if "val999" in lowered:
        return "val"
    if "validation" in lowered:
        return "train"
    if "train" in lowered:
        return "train"
    if "val" in lowered:
        return "val"
    if "test" in lowered:
        return "test"
    return "unknown"


def locate_image(annotation_path: Path) -> Path | None:
    stem = annotation_path.stem
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = annotation_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    # fallback for alternate image directory layouts
    split = infer_split(annotation_path)
    bundle_root = annotation_path.parent
    for parent in [bundle_root, *bundle_root.parents[:4]]:
        image_dir = parent / split
        for suffix in (".jpg", ".png", ".jpeg"):
            candidate = image_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def make_materialized_record(project_root: Path, split: str, ann_path: Path, payload: dict, image_path: Path) -> tuple[dict, dict]:
    rel_image = image_path.relative_to(project_root).as_posix()
    asset_dir = project_root / "V2" / "data" / "assets" / "phase0_edu" / split / "edu_chemc__hard"
    ensure_dir(asset_dir)
    dest_path = asset_dir / image_path.name
    if not dest_path.exists():
        shutil.copy2(image_path, dest_path)

    record = {
        "messages": [
            {"role": "user", "content": f"<image>{PROMPT}"},
            {"role": "assistant", "content": payload["ssml_normed"]},
        ],
        "images": [f"../assets/phase0_edu/{split}/edu_chemc__hard/{image_path.name}"],
        "meta": {
            "id": f"edu_chemc_{split}_{ann_path.stem}",
            "source": "edu_chemc",
            "difficulty": "hard",
            "task_type": "education_structure_recognition",
            "label_format": "ssml_normed",
            "annotation_json": ann_path.relative_to(project_root).as_posix(),
            "original_image": rel_image,
        },
    }
    meta = {
        "id": f"edu_chemc_{split}_{ann_path.stem}",
        "source": "edu_chemc",
        "split": split,
        "image_path": dest_path.relative_to(project_root).as_posix(),
        "label_format": "ssml_normed",
        "task_type": "education_structure_recognition",
        "difficulty": "hard",
        "ssml_normed": payload["ssml_normed"],
    }
    return record, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--raw-root", default="V2/data/EDU-CHMEC-MM23/raw_unpacked")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    raw_root = (project_root / args.raw_root).resolve()
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing raw unpacked EDU-CHEMC root: {raw_root}")

    train_records = []
    val_records = []
    train_meta = []
    val_meta = []

    json_paths = sorted(raw_root.rglob("*.json"))
    for ann_path in json_paths:
        with ann_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "ssml_normed" not in payload or not payload["ssml_normed"]:
            continue

        split = infer_split(ann_path)
        if split not in {"train", "val"}:
            continue

        image_path = locate_image(ann_path)
        if image_path is None:
            continue

        record, meta = make_materialized_record(project_root, split, ann_path, payload, image_path)
        if split == "train":
            train_records.append(record)
            train_meta.append(meta)
        else:
            val_records.append(record)
            val_meta.append(meta)

    sft_root = project_root / "V2" / "data" / "sft_materialized"
    manifests_root = project_root / "V2" / "data" / "manifests"
    reports_root = project_root / "V2" / "reports"

    write_jsonl(sft_root / "train_phase0_edu_messages.jsonl", train_records)
    write_jsonl(sft_root / "val_phase0_edu_messages.jsonl", val_records)
    write_jsonl(manifests_root / "edu_chemc_train_meta.jsonl", train_meta)
    write_jsonl(manifests_root / "edu_chemc_val_meta.jsonl", val_meta)

    summary = {
        "train": len(train_records),
        "val": len(val_records),
        "source": "edu_chemc",
        "label_format": "ssml_normed",
        "difficulty_counts": dict(Counter(item["difficulty"] for item in train_meta + val_meta)),
        "split_counts": {"train": len(train_records), "val": len(val_records)},
    }
    ensure_dir(reports_root)
    (reports_root / "edu_chemc_phase0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
