import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from PIL import Image


DEFAULT_PROMPT = (
    "OCR: Output only the canonical SMILES string for the molecule shown in the image."
)


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


def load_prompt(prompt_file: Path | None, prompt_text: str | None):
    if prompt_text:
        return prompt_text.strip()
    if prompt_file and prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT


def canonicalize_smiles(smiles: str):
    if smiles is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def resolve_image_path(raw_path: str, labels_path: Path, project_root: Path):
    raw = Path(raw_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(project_root / raw)
        candidates.append(labels_path.parent / raw)
        candidates.append(labels_path.parent.parent / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Cannot resolve image path: {raw_path}")


def validate_image_file(image_path: Path):
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as exc:  # Pillow raises several subclasses for bad files.
        return f"invalid_image_file:{type(exc).__name__}"
    return None


def to_image_url(image_path: Path, project_root: Path, image_path_mode: str):
    if image_path_mode == "absolute":
        return str(image_path)
    return str(image_path.relative_to(project_root)).replace("\\", "/")


def parse_crop_bbox(raw_value):
    text = str(raw_value or "").strip()
    if not text:
        return None

    def normalize(candidate):
        if isinstance(candidate, dict):
            candidate = candidate.get("bbox") or candidate.get("box")
        if not isinstance(candidate, (list, tuple)) or len(candidate) < 4:
            return None
        coords = []
        for value in candidate[:4]:
            try:
                coords.append(int(round(float(value))))
            except (TypeError, ValueError):
                return None
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        parts = [item.strip() for item in text.split(",") if item.strip()]
        if len(parts) >= 4:
            return normalize(parts[:4])
        return None

    if isinstance(payload, list):
        if payload and isinstance(payload[0], (list, tuple, dict)):
            for candidate in payload:
                bbox = normalize(candidate)
                if bbox:
                    return bbox
            return None
        return normalize(payload)
    if isinstance(payload, dict):
        return normalize(payload)
    return None


def make_crop_namespace(manifest_path: Path):
    parts = [manifest_path.parent.name, manifest_path.stem]
    raw = "_".join(part for part in parts if part)
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)


def materialize_manifest_crop(
    image_path: Path,
    bbox,
    derived_root: Path,
    manifest_path: Path,
    sample_id: str,
):
    x1, y1, x2, y2 = bbox
    namespace = make_crop_namespace(manifest_path)
    out_dir = derived_root / namespace
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id}.png"

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        pad = max(12, int(round(min(x2 - x1, y2 - y1) * 0.08)))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return image_path
        crop = img.crop((x1, y1, x2, y2))
        crop.save(out_path, format="PNG")

    return out_path.resolve()


def record_to_meta(record, labels_path: Path, project_root: Path, prompt: str, image_path_mode: str):
    gt = record.get("ground_truth", {})
    smiles = canonicalize_smiles(gt.get("smiles"))
    if not smiles:
        return None, "invalid_smiles"

    try:
        image_path = resolve_image_path(record["image"], labels_path, project_root)
    except FileNotFoundError:
        return None, "missing_image_file"
    image_reason = validate_image_file(image_path)
    if image_reason:
        return None, image_reason
    image_url = to_image_url(image_path, project_root, image_path_mode)

    return (
        {
            "id": record["id"],
            "source": record.get("source", "unknown"),
            "task_type": record.get("task_type", "molecule_structure_recognition"),
            "difficulty": record.get("difficulty", "unknown"),
            "image_path": image_url,
            "canonical_smiles": smiles,
            "prompt": prompt,
            "repeat_hint": 1,
        },
        None,
    )


def parse_repeat_hint(value):
    text = str(value or "").strip()
    if not text:
        return 1
    try:
        return max(1, int(text))
    except ValueError:
        return 1


def manifest_row_to_meta(
    row,
    manifest_path: Path,
    project_root: Path,
    derived_root: Path,
    prompt: str,
    image_path_mode: str,
    default_source: str,
    default_task_type: str,
    default_difficulty: str,
    manifest_bbox_crop_mode: str,
):
    sample_id = (row.get("sample_id") or "").strip()
    image_value = (row.get("image_path") or "").strip()
    smiles_value = canonicalize_smiles(row.get("canonical_smiles"))
    split = (row.get("split") or "train").strip().lower()
    if not sample_id or not image_value or not smiles_value:
        return None, "missing_required_fields_or_invalid_smiles"
    if split not in {"train", "val", "test"}:
        split = "train"

    try:
        image_path = resolve_image_path(image_value, manifest_path, project_root)
    except FileNotFoundError:
        return None, "missing_image_file"
    image_reason = validate_image_file(image_path)
    if image_reason:
        return None, image_reason
    used_bbox_crop = False
    if manifest_bbox_crop_mode != "off":
        should_crop = manifest_bbox_crop_mode == "all" or (
            manifest_bbox_crop_mode == "train_val" and split in {"train", "val"}
        )
        if should_crop:
            bbox = parse_crop_bbox(row.get("crop_bbox"))
            if bbox:
                image_path = materialize_manifest_crop(
                    image_path,
                    bbox,
                    derived_root,
                    manifest_path,
                    sample_id,
                )
                image_reason = validate_image_file(image_path)
                if image_reason:
                    return None, image_reason
                used_bbox_crop = True
    image_url = to_image_url(image_path, project_root, image_path_mode)
    source = (row.get("source") or default_source).strip() or default_source
    task_type = (
        row.get("task_type")
        or row.get("scene_type")
        or default_task_type
    )
    difficulty = (
        row.get("difficulty")
        or row.get("capture_type")
        or default_difficulty
    )
    repeat_hint = parse_repeat_hint(row.get("repeat_hint"))

    return (
        {
            "id": sample_id,
            "source": source,
            "task_type": str(task_type).strip() or default_task_type,
            "difficulty": str(difficulty).strip() or default_difficulty,
            "image_path": image_url,
            "canonical_smiles": smiles_value,
            "prompt": prompt,
            "split": split,
            "repeat_hint": repeat_hint,
            "used_bbox_crop": used_bbox_crop,
        },
        None,
    )


def make_sft_record(meta):
    return {
        "image_info": [
            {"matched_text_index": 0, "image_url": meta["image_path"]},
        ],
        "text_info": [
            {"text": meta["prompt"], "tag": "mask"},
            {"text": meta["canonical_smiles"], "tag": "no_mask"},
        ],
    }


def get_train_repeat(meta, args):
    repeat = 1
    smiles = meta["canonical_smiles"]
    repeat = max(repeat, int(meta.get("repeat_hint", 1)))

    if meta["source"] == "real_world":
        repeat = max(repeat, args.repeat_real_world_train)

    if len(smiles) >= args.long_smiles_threshold:
        repeat = max(repeat, args.repeat_long_smiles_train)

    if "@" in smiles:
        repeat = max(repeat, args.repeat_stereo_train)

    return max(1, int(repeat))


def expand_train_meta(train_meta, args):
    expanded = []
    repeat_stats = Counter()

    for meta in train_meta:
        repeat = get_train_repeat(meta, args)
        repeat_stats[f"repeat_{repeat}"] += 1
        for copy_index in range(repeat):
            item = dict(meta)
            if repeat > 1:
                item["repeat_group"] = meta["id"]
                item["repeat_index"] = copy_index
                item["id"] = f"{meta['id']}__rep{copy_index}"
            expanded.append(item)

    expanded.sort(key=lambda item: item["id"])
    return expanded, dict(repeat_stats)


def stratified_train_val_split(records, val_ratio: float, seed: int):
    groups = defaultdict(list)
    for record in records:
        groups[record["source"]].append(record)

    rng = random.Random(seed)
    train_records = []
    val_records = []

    for group_records in groups.values():
        items = list(group_records)
        rng.shuffle(items)
        if len(items) <= 10:
            val_count = 1 if len(items) >= 5 else 0
        else:
            val_count = max(1, int(round(len(items) * val_ratio)))

        val_records.extend(items[:val_count])
        train_records.extend(items[val_count:])

    train_records.sort(key=lambda item: item["id"])
    val_records.sort(key=lambda item: item["id"])
    return train_records, val_records


def load_manifest_records(
    manifest_path: Path,
    project_root: Path,
    derived_root: Path,
    prompt: str,
    image_path_mode: str,
    default_source: str,
    default_task_type: str,
    default_difficulty: str,
    manifest_bbox_crop_mode: str,
):
    if not manifest_path.exists():
        return [], []

    records = []
    skipped = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            meta, reason = manifest_row_to_meta(
                row,
                manifest_path,
                project_root,
                derived_root,
                prompt,
                image_path_mode,
                default_source,
                default_task_type,
                default_difficulty,
                manifest_bbox_crop_mode,
            )
            if meta:
                records.append(meta)
            else:
                skipped.append(
                    {
                        "id": row.get("sample_id", ""),
                        "source": default_source,
                        "reason": reason,
                    }
                )
    return records, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-labels", default="prepared/competition_data/public_train/labels.jsonl")
    parser.add_argument("--eval-labels", default="ocsr_evalset_final/annotations/labels.jsonl")
    parser.add_argument("--real-world-manifest", action="append", default=[])
    parser.add_argument("--extra-manifest", action="append", default=[])
    parser.add_argument("--out-dir", default="server_ready/paddleocr_vl_ocsr_a800/data")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--repeat-real-world-train", type=int, default=1)
    parser.add_argument("--repeat-long-smiles-train", type=int, default=1)
    parser.add_argument("--repeat-stereo-train", type=int, default=1)
    parser.add_argument("--long-smiles-threshold", type=int, default=80)
    parser.add_argument(
        "--image-path-mode",
        choices=["absolute", "relative"],
        default="absolute",
    )
    parser.add_argument(
        "--manifest-bbox-crop-mode",
        choices=["off", "train_val", "all"],
        default="train_val",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    train_labels = Path(args.train_labels).resolve()
    eval_labels = Path(args.eval_labels).resolve()
    real_world_manifest_paths = [Path(item).resolve() for item in args.real_world_manifest if item]
    extra_manifest_paths = [Path(item).resolve() for item in args.extra_manifest]
    out_dir = Path(args.out_dir).resolve()
    derived_root = out_dir / "derived_images"
    prompt_file = Path(args.prompt_file).resolve() if args.prompt_file else None
    prompt = load_prompt(prompt_file, args.prompt_text)

    public_train_meta = []
    skipped_public = []
    for record in read_jsonl(train_labels):
        meta, reason = record_to_meta(
            record,
            train_labels,
            project_root,
            prompt,
            args.image_path_mode,
        )
        if meta:
            public_train_meta.append(meta)
        else:
            skipped_public.append(
                {
                    "id": record.get("id", ""),
                    "source": record.get("source", "unknown"),
                    "reason": reason,
                }
            )

    train_public, val_public = stratified_train_val_split(public_train_meta, args.val_ratio, args.seed)

    real_world_records = []
    skipped_real_world = []
    for manifest_path in real_world_manifest_paths:
        records, skipped = load_manifest_records(
            manifest_path,
            project_root,
            derived_root,
            prompt,
            args.image_path_mode,
            "real_world",
            "real_world",
            "unknown",
            args.manifest_bbox_crop_mode,
        )
        real_world_records.extend(records)
        skipped_real_world.extend(skipped)

    real_world_train = [record for record in real_world_records if record["split"] == "train"]
    real_world_val = [record for record in real_world_records if record["split"] == "val"]
    real_world_test = [record for record in real_world_records if record["split"] == "test"]

    extra_public_records = []
    skipped_extra_public = []
    for extra_manifest_path in extra_manifest_paths:
        records, skipped = load_manifest_records(
            extra_manifest_path,
            project_root,
            derived_root,
            prompt,
            args.image_path_mode,
            "external_public",
            "molecule_structure_recognition",
            "unknown",
            args.manifest_bbox_crop_mode,
        )
        extra_public_records.extend(records)
        skipped_extra_public.extend(skipped)

    extra_public_train = [record for record in extra_public_records if record["split"] == "train"]
    extra_public_val = [record for record in extra_public_records if record["split"] == "val"]
    extra_public_test = [record for record in extra_public_records if record["split"] == "test"]

    train_meta = sorted(train_public + real_world_train + extra_public_train, key=lambda item: item["id"])
    val_meta = sorted(val_public + real_world_val + extra_public_val, key=lambda item: item["id"])
    expanded_train_meta, repeat_stats = expand_train_meta(train_meta, args)

    competition_eval_meta = []
    skipped_eval = []
    for record in read_jsonl(eval_labels):
        meta, reason = record_to_meta(
            record,
            eval_labels,
            project_root,
            prompt,
            args.image_path_mode,
        )
        if meta:
            competition_eval_meta.append(meta)
        else:
            skipped_eval.append(
                {
                    "id": record.get("id", ""),
                    "source": record.get("source", "unknown"),
                    "reason": reason,
                }
            )
    competition_eval_meta = sorted(competition_eval_meta, key=lambda item: item["id"])
    auxiliary_eval_meta = sorted(real_world_test + extra_public_test, key=lambda item: item["id"])
    combined_eval_meta = sorted(competition_eval_meta + auxiliary_eval_meta, key=lambda item: item["id"])

    write_jsonl(out_dir / "meta" / "train_meta.jsonl", train_meta)
    write_jsonl(out_dir / "meta" / "train_meta_expanded.jsonl", expanded_train_meta)
    write_jsonl(out_dir / "meta" / "val_meta.jsonl", val_meta)
    write_jsonl(out_dir / "benchmarks" / "competition_eval.jsonl", competition_eval_meta)
    write_jsonl(out_dir / "benchmarks" / "auxiliary_eval.jsonl", auxiliary_eval_meta)
    write_jsonl(out_dir / "benchmarks" / "combined_eval.jsonl", combined_eval_meta)
    write_jsonl(out_dir / "reports" / "skipped_public_invalid_smiles.jsonl", skipped_public)
    write_jsonl(out_dir / "reports" / "skipped_eval_invalid_smiles.jsonl", skipped_eval)
    write_jsonl(out_dir / "reports" / "skipped_real_world_rows.jsonl", skipped_real_world)
    write_jsonl(out_dir / "reports" / "skipped_extra_public_rows.jsonl", skipped_extra_public)

    write_jsonl(out_dir / "sft" / "train.jsonl", [make_sft_record(item) for item in expanded_train_meta])
    write_jsonl(out_dir / "sft" / "val.jsonl", [make_sft_record(item) for item in val_meta])

    summary = {
        "prompt": prompt,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "image_path_mode": args.image_path_mode,
        "manifest_bbox_crop_mode": args.manifest_bbox_crop_mode,
        "train_total_base": len(train_meta),
        "train_total_exported": len(expanded_train_meta),
        "val_total": len(val_meta),
        "competition_eval_total": len(competition_eval_meta),
        "auxiliary_eval_total": len(auxiliary_eval_meta),
        "combined_eval_total": len(combined_eval_meta),
        "skipped_public_invalid_smiles": len(skipped_public),
        "skipped_eval_invalid_smiles": len(skipped_eval),
        "skipped_real_world_rows": len(skipped_real_world),
        "skipped_extra_public_rows": len(skipped_extra_public),
        "train_by_source": dict(Counter(item["source"] for item in train_meta)),
        "train_bbox_crops_applied": sum(1 for item in train_meta if item.get("used_bbox_crop")),
        "train_repeat_stats": repeat_stats,
        "val_by_source": dict(Counter(item["source"] for item in val_meta)),
        "val_bbox_crops_applied": sum(1 for item in val_meta if item.get("used_bbox_crop")),
        "competition_eval_by_source": dict(Counter(item["source"] for item in competition_eval_meta)),
        "auxiliary_eval_by_source": dict(Counter(item["source"] for item in auxiliary_eval_meta)),
        "auxiliary_eval_bbox_crops_applied": sum(1 for item in auxiliary_eval_meta if item.get("used_bbox_crop")),
        "combined_eval_by_source": dict(Counter(item["source"] for item in combined_eval_meta)),
        "extra_public_total": len(extra_public_records),
        "repeat_real_world_train": args.repeat_real_world_train,
        "repeat_long_smiles_train": args.repeat_long_smiles_train,
        "repeat_stereo_train": args.repeat_stereo_train,
        "long_smiles_threshold": args.long_smiles_threshold,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Train samples (base):", len(train_meta))
    print("Train samples (exported):", len(expanded_train_meta))
    print("Val samples:", len(val_meta))
    print("Competition eval samples:", len(competition_eval_meta))
    print("Auxiliary eval samples:", len(auxiliary_eval_meta))
    print("Combined eval samples:", len(combined_eval_meta))
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
