from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from candidate_policy import load_policy, rank_candidates


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


def rerank_row(row: dict, policy: dict) -> dict:
    candidates = row.get("candidates") or []
    if not candidates:
        out = dict(row)
        out["selection_reason"] = out.get("selection_reason") or "no_candidates"
        return out

    ranked = rank_candidates(candidates, policy)
    best = ranked[0]
    out = dict(row)
    out["prompt"] = best.get("prompt", out.get("prompt", ""))
    out["prediction"] = best.get("prediction", "")
    out["canonical_prediction"] = best.get("canonical_prediction")
    out["raw_text"] = best.get("raw_text", "")
    out["selection_reason"] = "candidate_policy_linear"
    out["vote_count"] = best.get("vote_count", 0)
    out["policy_score"] = best.get("policy_score")
    out["policy_top_candidate"] = {
        "prompt_index": best.get("prompt_index"),
        "tta_index": best.get("tta_index"),
        "tta_name": best.get("tta_name"),
        "policy_score": best.get("policy_score"),
        "canonical_prediction": best.get("canonical_prediction"),
    }
    out["candidates"] = ranked
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.input_jsonl).resolve()))
    policy = load_policy(Path(args.policy_json).resolve())
    outputs = [rerank_row(row, policy) for row in rows]
    write_jsonl(Path(args.output_jsonl).resolve(), outputs)
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "output": str(Path(args.output_jsonl).resolve()),
                "policy": str(Path(args.policy_json).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
