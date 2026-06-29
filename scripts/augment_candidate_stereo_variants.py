#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rerank_ocsr_candidates import canonicalize, read_jsonl  # noqa: E402


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def stereo_variant(candidate: dict, score_delta: float, penalty_delta: float):
    source_smiles = candidate.get("canonical_prediction") or candidate.get("prediction")
    canonical_iso = canonicalize(source_smiles, isomeric=True)
    canonical_noniso = canonicalize(source_smiles, isomeric=False)
    if not canonical_iso or not canonical_noniso:
        return None
    if canonical_iso == canonical_noniso:
        return None
    if not any(marker in canonical_iso for marker in ("/", "\\", "@")):
        return None

    out = dict(candidate)
    out["prediction"] = canonical_noniso
    out["canonical_prediction"] = canonical_noniso
    out["raw_text"] = candidate.get("raw_text", "")
    out["candidate_variant"] = "stereo_stripped"
    out["variant_parent_prediction"] = candidate.get("prediction", "")
    out["variant_parent_canonical"] = canonical_iso
    if candidate.get("generation_score") is not None:
        out["generation_score"] = safe_float(candidate.get("generation_score"), -1_000_000.0) + score_delta
    if candidate.get("smiles_structure_penalty") is not None:
        out["smiles_structure_penalty"] = safe_float(candidate.get("smiles_structure_penalty"), 0.0) + penalty_delta
    tta_name = str(candidate.get("tta_name", "") or "")
    out["tta_name"] = f"{tta_name}:stereo_stripped" if tta_name else "stereo_stripped"
    return out


def selected_as_candidate(row: dict):
    return {
        "prompt": row.get("prompt", ""),
        "prediction": row.get("prediction", ""),
        "canonical_prediction": row.get("canonical_prediction"),
        "generation_score": row.get("generation_score"),
        "smiles_structure_penalty": row.get("smiles_structure_penalty", 1000),
        "raw_text": row.get("raw_text", ""),
        "prompt_index": 0,
        "tta_name": "selected",
        "tta_index": 0,
        "generation_index": 0,
    }


def augment_row(row: dict, include_selected: bool, score_delta: float, penalty_delta: float):
    out = dict(row)
    candidates = [dict(item) for item in row.get("candidates") or []]
    existing = {
        canonicalize(item.get("canonical_prediction") or item.get("prediction"), isomeric=True)
        for item in candidates
    }
    existing.discard(None)

    sources = list(candidates)
    if include_selected and str(row.get("prediction", "") or "").strip():
        sources.insert(0, selected_as_candidate(row))

    added = 0
    for candidate in sources:
        variant = stereo_variant(candidate, score_delta, penalty_delta)
        if variant is None:
            continue
        canonical = canonicalize(variant.get("canonical_prediction") or variant.get("prediction"), isomeric=True)
        if not canonical or canonical in existing:
            continue
        candidates.append(variant)
        existing.add(canonical)
        added += 1

    out["candidates"] = candidates
    debug = dict(out.get("candidate_augmentation_debug") or {})
    debug["stereo_stripped_added"] = added
    debug["candidate_count_after_stereo_augment"] = len(candidates)
    out["candidate_augmentation_debug"] = debug
    return out, added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--include-selected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-delta", type=float, default=-0.15)
    parser.add_argument("--penalty-delta", type=float, default=0.0)
    args = parser.parse_args()

    rows = []
    total_added = 0
    rows_with_added = 0
    for row in read_jsonl(Path(args.input_jsonl)):
        out, added = augment_row(row, args.include_selected, args.score_delta, args.penalty_delta)
        rows.append(out)
        total_added += added
        rows_with_added += int(added > 0)

    write_jsonl(Path(args.output_jsonl), rows)
    print(
        json.dumps(
            {
                "input_jsonl": args.input_jsonl,
                "output_jsonl": args.output_jsonl,
                "rows": len(rows),
                "rows_with_added": rows_with_added,
                "stereo_stripped_candidates_added": total_added,
                "include_selected": args.include_selected,
                "score_delta": args.score_delta,
                "penalty_delta": args.penalty_delta,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
