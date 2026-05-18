import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def canonicalize_smiles(smiles: str):
    text = str(smiles or "").strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def safe_rate(numerator: int, denominator: int):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-jsonl", required=True)
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--report-json", default="")
    args = parser.parse_args()

    benchmark = {row["id"]: row for row in read_jsonl(Path(args.benchmark_jsonl).resolve())}
    predictions = {row["id"]: row for row in read_jsonl(Path(args.prediction_jsonl).resolve())}

    total = 0
    raw_exact = 0
    canonical_exact = 0
    valid_predictions = 0
    missing_predictions = 0
    by_source = defaultdict(Counter)

    for sample_id, target in benchmark.items():
        total += 1
        source = target.get("source", "unknown")
        gt_raw = str(target.get("canonical_smiles", "")).strip()
        gt_canonical = canonicalize_smiles(gt_raw)
        pred_row = predictions.get(sample_id)

        if pred_row is None:
            missing_predictions += 1
            by_source[source]["total"] += 1
            by_source[source]["missing"] += 1
            continue

        pred_raw = str(pred_row.get("prediction", "")).strip()
        pred_canonical = canonicalize_smiles(pred_raw)

        by_source[source]["total"] += 1

        if pred_raw == gt_raw:
            raw_exact += 1
            by_source[source]["raw_exact"] += 1

        if pred_canonical is not None:
            valid_predictions += 1
            by_source[source]["valid"] += 1

        if pred_canonical is not None and pred_canonical == gt_canonical:
            canonical_exact += 1
            by_source[source]["canonical_exact"] += 1

    report = {
        "total": total,
        "missing_predictions": missing_predictions,
        "raw_exact_match": {
            "count": raw_exact,
            "rate": safe_rate(raw_exact, total),
        },
        "canonical_exact_match": {
            "count": canonical_exact,
            "rate": safe_rate(canonical_exact, total),
        },
        "valid_smiles_rate": {
            "count": valid_predictions,
            "rate": safe_rate(valid_predictions, total),
        },
        "by_source": {},
    }

    for source, counter in sorted(by_source.items()):
        source_total = counter["total"]
        report["by_source"][source] = {
            "total": source_total,
            "missing_predictions": counter["missing"],
            "raw_exact_match_rate": safe_rate(counter["raw_exact"], source_total),
            "canonical_exact_match_rate": safe_rate(counter["canonical_exact"], source_total),
            "valid_smiles_rate": safe_rate(counter["valid"], source_total),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
