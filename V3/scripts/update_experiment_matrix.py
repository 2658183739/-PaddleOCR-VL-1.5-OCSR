import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def completed_runs(analysis: dict) -> dict[str, dict]:
    runs = dict(analysis.get("runs", {}))
    diagnostics = analysis.get("diagnostics", {})
    for name in ("augmentation_dose2", "warmstart"):
        diagnostic = diagnostics.get(name)
        if diagnostic and diagnostic.get("run_id"):
            runs[diagnostic["run_id"]] = diagnostic
    return runs


def update_matrix(matrix_path: Path, analysis_path: Path):
    analysis = load_json(analysis_path)
    runs = completed_runs(analysis)

    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"Experiment matrix has no header: {matrix_path}")

    for row in rows:
        result = runs.get(row.get("run_id", ""))
        if not result:
            continue
        panels = result["panels"]
        row["dev_core_exact"] = f'{panels["legacy_core_dev"]["canonical_exact"]:.9f}'
        row["dev_region_exact"] = f'{panels["legacy_region_dev"]["canonical_exact"]:.9f}'
        row["dev_valid"] = f'{result["min_valid_rate"]:.9f}'
        row["selected_checkpoint"] = result["checkpoint"]
        row["status"] = "completed"

    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--analysis", required=True)
    args = parser.parse_args()
    update_matrix(Path(args.matrix).resolve(), Path(args.analysis).resolve())


if __name__ == "__main__":
    main()
