"""
Curriculum Learning: sort training data by difficulty (easy → hard).
Modifies train_meta_expanded.jsonl and regenerates train.jsonl in order.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


DIFFICULTY_RANK = {
    "clean": 0,
    "easy": 1,
    "medium": 2,
    "medium_hard": 3,
    "abbreviated": 4,
    "large": 5,
    "hard": 6,
    "photo": 7,
    "scan": 8,
    "page_level": 9,
    "handwritten": 10,
    "unknown": 5,
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_difficulty_rank(meta):
    diff = str(meta.get("difficulty", "unknown")).lower()
    return DIFFICULTY_RANK.get(diff, 5)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="server_ready/paddleocr_vl_ocsr_a800/data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-within-tier", action="store_true", default=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    meta_path = data_dir / "meta" / "train_meta_expanded.jsonl"
    sft_path = data_dir / "sft" / "train.jsonl"

    # Read expanded metadata
    print("Loading training metadata...", flush=True)
    metas = list(read_jsonl(meta_path))
    print(f"  Total records: {len(metas)}", flush=True)

    # Show current distribution
    by_diff = Counter(str(m.get("difficulty", "unknown")) for m in metas)
    print(f"  Difficulty distribution: {dict(by_diff)}", flush=True)

    # Sort by difficulty (easy → hard)
    print("\nSorting by difficulty (curriculum: easy → hard)...", flush=True)
    rng = random.Random(args.seed)

    # Group by difficulty tier
    tiers = {}
    for m in metas:
        rank = get_difficulty_rank(m)
        tiers.setdefault(rank, []).append(m)

    # Sort each tier (optionally shuffle within tier)
    sorted_metas = []
    for rank in sorted(tiers.keys()):
        tier_items = tiers[rank]
        if args.shuffle_within_tier:
            rng.shuffle(tier_items)
        sorted_metas.extend(tier_items)

    # Show the order
    print("  Curriculum order:")
    for rank in sorted(tiers.keys()):
        rank_name = [k for k, v in DIFFICULTY_RANK.items() if v == rank][0]
        print(f"    Tier {rank} ({rank_name}): {len(tiers[rank])} samples")

    # Verify counts match
    assert len(sorted_metas) == len(metas), f"Count mismatch: {len(sorted_metas)} vs {len(metas)}"

    # Save sorted metadata
    write_jsonl(meta_path, sorted_metas)

    # Regenerate SFT JSONL
    print("\nRegenerating SFT train.jsonl in curriculum order...", flush=True)
    sft_records = [make_sft_record(m) for m in sorted_metas]
    write_jsonl(sft_path, sft_records)

    print(f"  Saved: {meta_path} ({len(sorted_metas)} records)")
    print(f"  Saved: {sft_path} ({len(sft_records)} records)")
    print("\nCurriculum learning applied. Data is now ordered easy → hard.", flush=True)


if __name__ == "__main__":
    main()
