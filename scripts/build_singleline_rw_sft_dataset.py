import argparse
import json
import os
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from PIL import Image


SOURCE_REPEAT = {
    "real_world": 5,
    "molgrapher_synthetic": 2,
    "uob": 1,
    "uspto": 1,
    "decimer": 2,
}

SOURCE_CAP = {
    "uspto30k_clean": 1500,
    "uspto30k_abbreviated": 1500,
    "uspto30k_large": 1500,
}

PROMPT = "OCR: Output only the canonical SMILES string for the molecule shown in the image."
CANONICAL_PROMPT_FRAGMENT = "canonical SMILES"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_eval_smiles(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    smiles = set()
    for row in read_jsonl(path):
        gt = row.get("ground_truth") or {}
        value = row.get("canonical_smiles") or row.get("smiles") or gt.get("smiles") or row.get("label_summary")
        if value:
            smiles.add(str(value).strip())
    return smiles


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_record_image(jsonl_path: Path, record: dict) -> Path | None:
    images = record.get("images") or []
    if not images:
        return None
    raw = Path(str(images[0]))
    if raw.is_absolute():
        return raw
    return (jsonl_path.parent / raw).resolve()


def is_readable_image(path: Path | None) -> bool:
    if path is None or not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def eval_smiles_from_row(row: dict) -> str:
    gt = row.get("ground_truth") or {}
    return str(row.get("canonical_smiles") or row.get("smiles") or gt.get("smiles") or row.get("label_summary") or "").strip()


def make_eval_message_record(row: dict, eval_root: Path, output_path: Path) -> dict:
    image_value = row.get("image") or row.get("image_path")
    image_path = Path(str(image_value))
    if not image_path.is_absolute():
        image_path = eval_root / image_path
    rel_image = Path(os.path.relpath(image_path.resolve(), output_path.parent.resolve())).as_posix()
    smiles = eval_smiles_from_row(row)
    return {
        "messages": [
            {"role": "user", "content": f"<image>{PROMPT}"},
            {"role": "assistant", "content": smiles},
        ],
        "images": [rel_image],
        "meta": {
            "id": row.get("id", ""),
            "source": row.get("source", "unknown"),
            "difficulty": row.get("difficulty", "unknown"),
            "task_type": row.get("task_type", "molecule_structure_recognition"),
            "benchmark_track": row.get("benchmark_track", ""),
            "canonical_smiles_length": len(smiles),
            "contains_stereo": "@" in smiles,
        },
    }


def assistant_text(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def user_text(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def is_canonical_smiles_record(record: dict) -> bool:
    text = assistant_text(record).strip()
    prompt = user_text(record)
    if not text:
        return False
    if text.startswith("\\chemfig") or "ssml_normed" in prompt:
        return False
    return CANONICAL_PROMPT_FRAGMENT in prompt


def deterministic_sample(items, cap: int, seed: int):
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    if cap <= 0 or cap >= len(items):
        return items
    return items[:cap]


def stable_source_seed(source: str) -> int:
    return sum((index + 1) * ord(ch) for index, ch in enumerate(source))


def repeated_record(record: dict, repeat_index: int, policy: str) -> dict:
    item = deepcopy(record)
    meta = dict(item.get("meta", {}))
    meta["singleline_policy"] = policy
    meta["repeat_index"] = repeat_index
    item["meta"] = meta
    return item


def build_records(records, seed: int, eval_smiles: set[str], input_path: Path):
    by_source = defaultdict(list)
    skipped = 0
    skipped_eval_smiles = 0
    skipped_unreadable_images = 0
    for record in records:
        if not is_canonical_smiles_record(record):
            skipped += 1
            continue
        if assistant_text(record).strip() in eval_smiles:
            skipped_eval_smiles += 1
            continue
        if not is_readable_image(resolve_record_image(input_path, record)):
            skipped_unreadable_images += 1
            continue
        source = record.get("meta", {}).get("source", "unknown")
        by_source[source].append(record)

    selected = []
    policy_counts = Counter()

    for source, items in sorted(by_source.items()):
        if source in SOURCE_CAP:
            base_items = deterministic_sample(items, SOURCE_CAP[source], seed + stable_source_seed(source))
            repeat = 1
            policy = f"cap_{SOURCE_CAP[source]}"
        else:
            base_items = list(items)
            repeat = SOURCE_REPEAT.get(source, 1)
            policy = f"repeat_{repeat}"

        for record in base_items:
            for repeat_index in range(repeat):
                selected.append(repeated_record(record, repeat_index, policy))
                policy_counts[f"{source}:{policy}"] += 1

    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected, skipped, skipped_eval_smiles, skipped_unreadable_images, dict(policy_counts)


def summarize(
    records,
    skipped: int,
    skipped_eval_smiles: int,
    skipped_unreadable_images: int,
    policy_counts: dict,
    input_path: Path,
    output_path: Path,
    eval_smiles_path: Path | None,
):
    source_counts = Counter()
    difficulty_counts = Counter()
    length_bins = Counter()
    for record in records:
        meta = record.get("meta", {})
        source_counts[meta.get("source", "unknown")] += 1
        difficulty_counts[meta.get("difficulty", "unknown")] += 1
        length = len(assistant_text(record))
        if length < 40:
            length_bins["lt_40"] += 1
        elif length < 80:
            length_bins["40_79"] += 1
        elif length < 120:
            length_bins["80_119"] += 1
        else:
            length_bins["ge_120"] += 1

    return {
        "strategy": "single_stage_real_weighted_lora_sft",
        "input": str(input_path),
        "output": str(output_path),
        "total": len(records),
        "skipped_non_canonical_records": skipped,
        "skipped_eval_smiles_records_before_weighting": skipped_eval_smiles,
        "skipped_unreadable_image_records_before_weighting": skipped_unreadable_images,
        "eval_smiles_filter": str(eval_smiles_path) if eval_smiles_path is not None else "",
        "source_counts": dict(source_counts),
        "difficulty_counts": dict(difficulty_counts),
        "smiles_length_bins": dict(length_bins),
        "policy_counts": policy_counts,
        "source_repeat": SOURCE_REPEAT,
        "source_cap": SOURCE_CAP,
        "notes": [
            "Only canonical-SMILES message records are used.",
            "EDU-CHEMC ssml_normed records are intentionally excluded from this one-line run.",
            "Real-world and visual-robustness samples are upweighted by deterministic repetition.",
            "DECIMER is configured for repeat_2 if present in the input dataset; it is absent from the current materialized phase3 file.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--input", default="V2/data/sft_materialized/train_phase3_messages.jsonl")
    parser.add_argument("--output", default="V2/data/sft_materialized/train_singleline_rw_messages.jsonl")
    parser.add_argument("--eval-output", default="V2/data/sft_materialized/val_singleline_v1p1_messages.jsonl")
    parser.add_argument("--report", default="V2/reports/singleline_rw_dataset_summary.json")
    parser.add_argument("--exclude-eval-smiles", default="V2/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl")
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_path = (project_root / args.input).resolve()
    output_path = (project_root / args.output).resolve()
    eval_output_path = (project_root / args.eval_output).resolve()
    report_path = (project_root / args.report).resolve()
    eval_smiles_path = (project_root / args.exclude_eval_smiles).resolve() if args.exclude_eval_smiles else None

    records = list(read_jsonl(input_path))
    eval_smiles = load_eval_smiles(eval_smiles_path)
    selected, skipped, skipped_eval_smiles, skipped_unreadable_images, policy_counts = build_records(
        records,
        seed=args.seed,
        eval_smiles=eval_smiles,
        input_path=input_path,
    )
    write_jsonl(output_path, selected)

    eval_count = 0
    if eval_smiles_path is not None and eval_smiles_path.exists():
        eval_root = eval_smiles_path.parent.parent
        eval_rows = [make_eval_message_record(row, eval_root, eval_output_path) for row in read_jsonl(eval_smiles_path)]
        write_jsonl(eval_output_path, eval_rows)
        eval_count = len(eval_rows)

    report = summarize(
        selected,
        skipped,
        skipped_eval_smiles,
        skipped_unreadable_images,
        policy_counts,
        input_path,
        output_path,
        eval_smiles_path,
    )
    report["eval_output"] = str(eval_output_path)
    report["eval_output_count"] = eval_count
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output_path), "report": str(report_path), "total": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
