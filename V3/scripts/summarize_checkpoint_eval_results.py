import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_report_files(eval_root: Path):
    return sorted(eval_root.rglob("report.json"))


def infer_metadata(report_path: Path, eval_root: Path):
    relative = report_path.resolve().relative_to(eval_root.resolve())
    if len(relative.parts) < 4:
        return "unknown", "unknown", "unknown"
    return relative.parts[0], relative.parts[1], relative.parts[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="V2/eval_runs")
    parser.add_argument("--csv-out", default="V2/reports/checkpoint_eval_summary.csv")
    parser.add_argument("--md-out", default="V2/reports/checkpoint_eval_summary.md")
    args = parser.parse_args()

    root = Path(args.eval_root).resolve()
    report_files = find_report_files(root)
    rows = []

    for report_path in report_files:
        payload = load_json(report_path)
        phase, checkpoint, benchmark = infer_metadata(report_path, root)
        row = {
            "phase": phase,
            "checkpoint": checkpoint,
            "benchmark": benchmark,
            "raw_exact_match_accuracy": payload["accuracy"]["raw_exact_match_accuracy"],
            "canonical_exact_match_accuracy": payload["accuracy"].get("canonical_exact_match_accuracy", ""),
            "valid_smiles_rate": payload["accuracy"].get("valid_smiles_rate", ""),
            "smiles_token_micro_f1": payload["f1"].get("smiles_token_micro_f1", ""),
            "mean_normalized_edit_similarity": payload["similarity"].get("mean_normalized_edit_similarity", ""),
            "report_path": str(report_path),
        }
        rows.append(row)

    rows.sort(key=lambda item: (item["phase"], item["checkpoint"], item["benchmark"]))

    csv_path = Path(args.csv_out).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["phase", "checkpoint", "benchmark", "raw_exact_match_accuracy", "canonical_exact_match_accuracy", "valid_smiles_rate", "smiles_token_micro_f1", "mean_normalized_edit_similarity", "report_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = Path(args.md_out).resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Checkpoint Evaluation Summary\n\n")
        handle.write("| Phase | Checkpoint | Benchmark | Raw Exact | Canonical Exact | Valid Rate | Token Micro F1 | Edit Similarity |\n")
        handle.write("|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['phase']} | {row['checkpoint']} | {row['benchmark']} | {row['raw_exact_match_accuracy']} | {row['canonical_exact_match_accuracy']} | {row['valid_smiles_rate']} | {row['smiles_token_micro_f1']} | {row['mean_normalized_edit_similarity']} |\n"
            )

    print(f"csv={csv_path}")
    print(f"md={md_path}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
