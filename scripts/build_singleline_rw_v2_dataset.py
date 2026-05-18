from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path


WEAK_DOMAIN_REPEAT = {
    "decimer_handdrawn": 4,
    "real_world_photo_scan": 5,
    "edu_exam": 5,
    "document_page_context": 3,
    "long_or_stereo": 3,
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
            gt = row.get("ground_truth")
            raw = ""
            if isinstance(gt, dict):
                raw = gt.get("smiles") or ""
            raw = raw or row.get("canonical_smiles") or row.get("smiles") or row.get("label_summary") or ""
            canonical = canonicalize(Chem, raw)
            if canonical:
                smiles.add(canonical)
    return smiles


def repeated(record: dict, repeat_index: int, policy: str) -> dict:
    item = deepcopy(record)
    meta = dict(item.get("meta", {}))
    meta["singleline_v2_policy"] = policy
    meta["singleline_v2_repeat_index"] = repeat_index
    item["meta"] = meta
    return item


def build_dataset(base_path: Path, weak_path: Path, output_path: Path, report_path: Path, eval_smiles: set[str], seed: int):
    Chem = try_load_rdkit()
    output = []
    skipped = Counter()
    source_counts = Counter()
    weak_domain_counts = Counter()
    policy_counts = Counter()

    for record in read_jsonl(base_path) or []:
        canonical = canonicalize(Chem, assistant_text(record))
        if canonical in eval_smiles:
            skipped["base_eval_smiles_overlap"] += 1
            continue
        output.append(record)
        meta = record.get("meta", {})
        source_counts[str(meta.get("source", "unknown"))] += 1
        policy_counts["base_keep"] += 1

    for record in read_jsonl(weak_path) or []:
        canonical = canonicalize(Chem, assistant_text(record))
        if canonical in eval_smiles:
            skipped["weak_eval_smiles_overlap"] += 1
            continue
        meta = record.get("meta", {})
        weak_domain = str(meta.get("weak_domain") or meta.get("source") or "unknown")
        repeat = WEAK_DOMAIN_REPEAT.get(weak_domain, 3)
        policy = f"weak_{weak_domain}:repeat_{repeat}"
        for repeat_index in range(repeat):
            output.append(repeated(record, repeat_index, policy))
            policy_counts[policy] += 1
            source_counts[str(meta.get("source", "unknown"))] += 1
            weak_domain_counts[weak_domain] += 1

    rng = random.Random(seed)
    rng.shuffle(output)
    write_jsonl(output_path, output)

    report = {
        "strategy": "singleline_rw_v2_base_plus_weak_domain_replay",
        "base": str(base_path),
        "weak_pool": str(weak_path),
        "output": str(output_path),
        "total": len(output),
        "rdkit_available": Chem is not None,
        "eval_smiles_filter_count": len(eval_smiles),
        "skipped": dict(skipped),
        "source_counts": dict(source_counts),
        "weak_domain_weighted_counts": dict(weak_domain_counts),
        "policy_counts": dict(policy_counts),
        "weak_domain_repeat": WEAK_DOMAIN_REPEAT,
        "notes": [
            "This builder keeps the V2-1 base records and adds weighted weak-domain replay records.",
            "It filters any record whose canonical assistant SMILES appears in evaluation labels.",
            "Use this as the V2-2 dataset after importing public/private weak-domain candidates.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def merge_pool(
    output: list[dict],
    pool_path: Path,
    eval_smiles: set[str],
    Chem,
    skipped: Counter,
    source_counts: Counter,
    weak_domain_counts: Counter,
    policy_counts: Counter,
    default_repeat_map: dict[str, int],
    policy_prefix: str,
):
    for record in read_jsonl(pool_path) or []:
        canonical = canonicalize(Chem, assistant_text(record))
        if canonical in eval_smiles:
            skipped[f"{policy_prefix}_eval_smiles_overlap"] += 1
            continue
        meta = record.get("meta", {})
        weak_domain = str(meta.get("weak_domain") or meta.get("source") or "unknown")
        repeat = default_repeat_map.get(weak_domain, 3)
        policy = f"{policy_prefix}_{weak_domain}:repeat_{repeat}"
        for repeat_index in range(repeat):
            output.append(repeated(record, repeat_index, policy))
            policy_counts[policy] += 1
            source_counts[str(meta.get("source", "unknown"))] += 1
            weak_domain_counts[weak_domain] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--base", default="V2-1/data/sft_materialized/train_singleline_rw_messages.jsonl")
    parser.add_argument("--weak-pool", default="V2-1/data/sft_materialized/train_weak_domain_pool_messages.jsonl")
    parser.add_argument("--auto-weak-pool", default="V2-1/data/sft_materialized/train_weak_domain_auto_messages.jsonl")
    parser.add_argument("--output", default="V2-1/data/sft_materialized/train_singleline_rw_v2_messages.jsonl")
    parser.add_argument("--report", default="V2-1/reports/singleline_rw_v2_dataset_summary.json")
    parser.add_argument(
        "--eval-labels",
        nargs="*",
        default=[
            "V2-1/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl",
            "V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl",
            "V2-1/data/eval/weak_domain_v2/annotations/labels.jsonl",
        ],
    )
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    Chem = try_load_rdkit()
    eval_smiles = load_eval_smiles([(project_root / path).resolve() for path in args.eval_labels], Chem)
    base_path = (project_root / args.base).resolve()
    weak_path = (project_root / args.weak_pool).resolve()
    auto_weak_path = (project_root / args.auto_weak_pool).resolve()
    output_path = (project_root / args.output).resolve()
    report_path = (project_root / args.report).resolve()

    output = []
    skipped = Counter()
    source_counts = Counter()
    weak_domain_counts = Counter()
    policy_counts = Counter()

    for record in read_jsonl(base_path) or []:
        canonical = canonicalize(Chem, assistant_text(record))
        if canonical in eval_smiles:
            skipped["base_eval_smiles_overlap"] += 1
            continue
        output.append(record)
        meta = record.get("meta", {})
        source_counts[str(meta.get("source", "unknown"))] += 1
        policy_counts["base_keep"] += 1

    merge_pool(
        output=output,
        pool_path=weak_path,
        eval_smiles=eval_smiles,
        Chem=Chem,
        skipped=skipped,
        source_counts=source_counts,
        weak_domain_counts=weak_domain_counts,
        policy_counts=policy_counts,
        default_repeat_map=WEAK_DOMAIN_REPEAT,
        policy_prefix="manual",
    )
    merge_pool(
        output=output,
        pool_path=auto_weak_path,
        eval_smiles=eval_smiles,
        Chem=Chem,
        skipped=skipped,
        source_counts=source_counts,
        weak_domain_counts=weak_domain_counts,
        policy_counts=policy_counts,
        default_repeat_map={
            "auto_photo_scan": 2,
            "auto_document_context": 2,
            "auto_exam_context": 2,
            "auto_handdrawn_like": 2,
            "auto_long_stereo": 2,
        },
        policy_prefix="auto",
    )

    rng = random.Random(args.seed)
    rng.shuffle(output)
    write_jsonl(output_path, output)
    report = {
        "strategy": "singleline_rw_v2_base_plus_manual_and_auto_weak_replay",
        "base": str(base_path),
        "weak_pool": str(weak_path),
        "auto_weak_pool": str(auto_weak_path),
        "output": str(output_path),
        "total": len(output),
        "rdkit_available": Chem is not None,
        "eval_smiles_filter_count": len(eval_smiles),
        "skipped": dict(skipped),
        "source_counts": dict(source_counts),
        "weak_domain_weighted_counts": dict(weak_domain_counts),
        "policy_counts": dict(policy_counts),
        "manual_weak_domain_repeat": WEAK_DOMAIN_REPEAT,
        "auto_weak_domain_repeat": {
            "auto_photo_scan": 2,
            "auto_document_context": 2,
            "auto_exam_context": 2,
            "auto_handdrawn_like": 2,
            "auto_long_stereo": 2,
        },
        "notes": [
            "This builder keeps the V2-1 base training set and adds both manual weak-domain pool and auto-generated weak-domain replay.",
            "It filters any record whose canonical assistant SMILES appears in evaluation labels.",
            "Use this as the V2-2 dataset after generating auto replay and optionally importing public/private weak-domain candidates.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
