#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-jsonl", required=True)
    parser.add_argument("--prediction-glob", required=True)
    parser.add_argument("--merged-jsonl", required=True)
    parser.add_argument("--remaining-jsonl", required=True)
    parser.add_argument("--status-json", default="")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_jsonl)
    labels = list(read_jsonl(benchmark_path))
    label_ids = [row["id"] for row in labels]
    label_id_set = set(label_ids)

    predictions = {}
    part_paths = sorted(Path(p) for p in glob.glob(args.prediction_glob))
    for part_path in part_paths:
        for row in read_jsonl(part_path):
            sample_id = row.get("id")
            if sample_id in label_id_set:
                predictions[sample_id] = row

    merged_rows = [predictions[sample_id] for sample_id in label_ids if sample_id in predictions]
    remaining_rows = [row for row in labels if row["id"] not in predictions]
    write_jsonl(Path(args.merged_jsonl), merged_rows)
    write_jsonl(Path(args.remaining_jsonl), remaining_rows)

    status = {
        "benchmark": str(benchmark_path),
        "parts": [str(path) for path in part_paths],
        "total": len(labels),
        "done": len(merged_rows),
        "remaining": len(remaining_rows),
        "merged_jsonl": args.merged_jsonl,
        "remaining_jsonl": args.remaining_jsonl,
    }
    if args.status_json:
        Path(args.status_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.status_json).write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
