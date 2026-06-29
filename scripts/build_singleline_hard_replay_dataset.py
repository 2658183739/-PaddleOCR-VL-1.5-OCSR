from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


SOURCE_CAPS = {
    "uob": 1200,
    "uspto": 1800,
    "uspto30k_abbreviated": 700,
    "uspto30k_large": 700,
    "uspto30k_clean": 450,
    "molgrapher_synthetic": 3200,
}

DIFFICULTY_REPEAT = {
    "handwritten": 8,
    "chinese_exam": 6,
    "photo": 6,
    "scan": 6,
    "degraded_scan": 6,
    "document_embed": 5,
    "journal_fig": 5,
    "multi_grid": 5,
    "page_level": 5,
    "hard": 3,
    "medium_hard": 2,
    "abbreviated": 2,
    "large": 2,
    "medium": 1,
    "easy": 1,
    "clean": 1,
}

AUTO_WEAK_REPEAT = {
    "auto_photo_scan": 14,
    "auto_document_context": 14,
    "auto_exam_context": 14,
    "auto_handdrawn_like": 14,
    "auto_long_stereo": 10,
}


def read_jsonl(path: Path):
    if not path.exists():
        return
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


def try_load_rdkit():
    try:
        from rdkit import Chem

        return Chem
    except Exception:
        return None


def canonicalize(Chem, smiles: str) -> str | None:
    text = str(smiles or "").strip()
    if not text:
        return None
    if Chem is None:
        return text
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def assistant_text(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def load_eval_smiles(paths: list[Path], Chem) -> set[str]:
    smiles = set()
    for path in paths:
        for row in read_jsonl(path) or []:
            ground_truth = row.get("ground_truth")
            raw = ""
            if isinstance(ground_truth, dict):
                raw = ground_truth.get("smiles") or ""
            raw = raw or row.get("canonical_smiles") or row.get("smiles") or row.get("label_summary") or ""
            canonical = canonicalize(Chem, raw)
            if canonical:
                smiles.add(canonical)
    return smiles


def record_key(record: dict) -> str:
    meta = record.get("meta", {})
    record_id = str(meta.get("id", "")).strip()
    images = "|".join(str(item) for item in record.get("images", []))
    target = assistant_text(record)
    return f"{record_id}\t{images}\t{target}"


def repeated(record: dict, repeat_index: int, policy: str) -> dict:
    item = deepcopy(record)
    meta = dict(item.get("meta", {}))
    meta["hard_replay_policy"] = policy
    meta["hard_replay_repeat_index"] = repeat_index
    item["meta"] = meta
    return item


def source_cap(source: str) -> int | None:
    return SOURCE_CAPS.get(source)


def base_repeat(record: dict) -> int:
    meta = record.get("meta", {})
    source = str(meta.get("source", "unknown"))
    difficulty = str(meta.get("difficulty", "unknown"))
    repeat = DIFFICULTY_REPEAT.get(difficulty, 1)

    if source == "real_world":
        return max(repeat, 4)
    if source == "molgrapher_synthetic":
        return min(max(repeat, 2), 3)
    if source in {"uspto", "uspto30k_abbreviated", "uspto30k_large"}:
        return min(max(repeat, 1), 2)
    if source in {"uob", "uspto30k_clean"}:
        return 1
    return repeat


def auto_repeat(record: dict) -> int:
    meta = record.get("meta", {})
    bucket = str(meta.get("weak_domain") or meta.get("auto_aug_bucket") or meta.get("difficulty") or "unknown")
    return AUTO_WEAK_REPEAT.get(bucket, 8)


def build_dataset(
    base_path: Path,
    auto_weak_path: Path,
    output_path: Path,
    report_path: Path,
    eval_smiles: set[str],
    seed: int,
):
    Chem = try_load_rdkit()
    rng = random.Random(seed)
    base_unique = {}
    skipped = Counter()
    source_candidates = defaultdict(list)

    for record in read_jsonl(base_path) or []:
        canonical = canonicalize(Chem, assistant_text(record))
        if canonical in eval_smiles:
            skipped["base_eval_smiles_overlap"] += 1
            continue
        key = record_key(record)
        if key not in base_unique:
            base_unique[key] = record

    for record in base_unique.values():
        source = str(record.get("meta", {}).get("source", "unknown"))
        source_candidates[source].append(record)

    capped_records = []
    cap_report = {}
    for source, records in sorted(source_candidates.items()):
        records = list(records)
        rng.shuffle(records)
        cap = source_cap(source)
        if cap is not None and len(records) > cap:
            cap_report[source] = {"before": len(records), "after": cap}
            skipped[f"{source}_cap"] += len(records) - cap
            records = records[:cap]
        else:
            cap_report[source] = {"before": len(records), "after": len(records)}
        capped_records.extend(records)

    output = []
    source_counts = Counter()
    difficulty_counts = Counter()
    policy_counts = Counter()
    weak_domain_counts = Counter()

    for record in capped_records:
        meta = record.get("meta", {})
        source = str(meta.get("source", "unknown"))
        difficulty = str(meta.get("difficulty", "unknown"))
        repeat = base_repeat(record)
        policy = f"base_{source}_{difficulty}:repeat_{repeat}"
        for repeat_index in range(repeat):
            output.append(repeated(record, repeat_index, policy))
            source_counts[source] += 1
            difficulty_counts[difficulty] += 1
            policy_counts[policy] += 1

    auto_unique = {}
    for record in read_jsonl(auto_weak_path) or []:
        canonical = canonicalize(Chem, assistant_text(record))
        if canonical in eval_smiles:
            skipped["auto_eval_smiles_overlap"] += 1
            continue
        auto_unique.setdefault(record_key(record), record)

    for record in auto_unique.values():
        meta = record.get("meta", {})
        source = str(meta.get("source", "unknown"))
        difficulty = str(meta.get("difficulty", "unknown"))
        bucket = str(meta.get("weak_domain") or meta.get("auto_aug_bucket") or difficulty)
        repeat = auto_repeat(record)
        policy = f"auto_{bucket}:repeat_{repeat}"
        for repeat_index in range(repeat):
            output.append(repeated(record, repeat_index, policy))
            source_counts[source] += 1
            difficulty_counts[difficulty] += 1
            policy_counts[policy] += 1
            weak_domain_counts[bucket] += 1

    rng.shuffle(output)
    write_jsonl(output_path, output)

    report = {
        "strategy": "hard_replay_from_v1_sft_low_lr",
        "base": str(base_path),
        "auto_weak_pool": str(auto_weak_path),
        "output": str(output_path),
        "total": len(output),
        "base_unique_after_eval_filter": len(base_unique),
        "auto_unique_after_eval_filter": len(auto_unique),
        "rdkit_available": Chem is not None,
        "eval_smiles_filter_count": len(eval_smiles),
        "skipped": dict(skipped),
        "source_caps": SOURCE_CAPS,
        "source_cap_report": cap_report,
        "difficulty_repeat": DIFFICULTY_REPEAT,
        "auto_weak_repeat": AUTO_WEAK_REPEAT,
        "source_counts": dict(source_counts),
        "difficulty_counts": dict(difficulty_counts),
        "weak_domain_weighted_counts": dict(weak_domain_counts),
        "policy_counts": dict(policy_counts),
        "notes": [
            "This dataset starts from unique V2-1 SFT records instead of preserving old repeat weights.",
            "It caps easy UOB/clean sources and aggressively replays weak visual domains.",
            "Use with low learning rate and fewer steps from V2-1/outputs/export to reduce forgetting.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--base", default="V2-1/data/sft_materialized/train_singleline_rw_messages.jsonl")
    parser.add_argument("--auto-weak-pool", default="V2-1/data/sft_materialized/train_weak_domain_auto_messages.jsonl")
    parser.add_argument("--output", default="V2-1/data/sft_materialized/train_singleline_hard_replay_messages.jsonl")
    parser.add_argument("--report", default="V2-1/reports/singleline_hard_replay_dataset_summary.json")
    parser.add_argument(
        "--eval-labels",
        nargs="*",
        default=[
            "V2-1/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl",
            "V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl",
            "V2-1/data/eval/weak_domain_v2/annotations/labels.jsonl",
        ],
    )
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    Chem = try_load_rdkit()
    eval_smiles = load_eval_smiles([(project_root / path).resolve() for path in args.eval_labels], Chem)
    report = build_dataset(
        base_path=(project_root / args.base).resolve(),
        auto_weak_path=(project_root / args.auto_weak_pool).resolve(),
        output_path=(project_root / args.output).resolve(),
        report_path=(project_root / args.report).resolve(),
        eval_smiles=eval_smiles,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
