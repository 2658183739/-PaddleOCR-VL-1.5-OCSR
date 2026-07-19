import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(benchmark_rows, prediction_rows):
    predictions = {str(row["id"]): row for row in prediction_rows}
    totals = defaultdict(int)
    details = []

    for target in benchmark_rows:
        sample_id = str(target["id"])
        expected = str(target.get("ground_truth", {}).get("smiles", "")).strip()
        prediction_row = predictions.get(sample_id)
        prediction = "" if prediction_row is None else str(prediction_row.get("prediction", "")).strip()

        raw_exact = prediction == expected
        normalized_exact = normalize_whitespace(prediction) == normalize_whitespace(expected)
        totals["total"] += 1
        totals["missing"] += prediction_row is None
        totals["nonempty"] += bool(prediction)
        totals["raw_exact"] += raw_exact
        totals["whitespace_normalized_exact"] += normalized_exact

        details.append(
            {
                "id": sample_id,
                "paper_group": target.get("paper_group", ""),
                "difficulty": target.get("difficulty", "unknown"),
                "expected": expected,
                "prediction": prediction,
                "raw_exact": raw_exact,
                "whitespace_normalized_exact": normalized_exact,
                "missing_prediction": prediction_row is None,
            }
        )

    report = {
        "track": "wild_symbolic_v3",
        "metric_policy": "literal symbolic transcription; no RDKit canonicalization",
        "counts": dict(totals),
        "accuracy": {
            "raw_exact_match_accuracy": safe_rate(totals["raw_exact"], totals["total"]),
            "whitespace_normalized_exact_match_accuracy": safe_rate(
                totals["whitespace_normalized_exact"], totals["total"]
            ),
            "nonempty_prediction_rate": safe_rate(totals["nonempty"], totals["total"]),
        },
    }
    return report, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-jsonl", required=True)
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--details-jsonl", required=True)
    args = parser.parse_args()

    report, details = evaluate(
        list(read_jsonl(Path(args.benchmark_jsonl).resolve())),
        list(read_jsonl(Path(args.prediction_jsonl).resolve())),
    )

    report_path = Path(args.report_json).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    details_path = Path(args.details_jsonl).resolve()
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
