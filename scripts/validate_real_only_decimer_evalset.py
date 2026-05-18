from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_SOURCE = "decimer"
EXPECTED_TASK_TYPE = "hand_drawn_molecule_structure_recognition"
EXPECTED_IMAGE_TYPE = "hand_drawn_chemical_structure"
EXPECTED_DIFFICULTY = "hard"
EXPECTED_PROMPT = "OCR: Output only the canonical SMILES string for the molecule shown in the image."


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def validate_bundle(bundle_root: Path, benchmark_jsonl: Path) -> dict:
    labels_path = bundle_root / "annotations" / "labels.jsonl"
    stats_path = bundle_root / "stats.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing stats file: {stats_path}")
    if not benchmark_jsonl.exists():
        raise FileNotFoundError(f"Missing benchmark file: {benchmark_jsonl}")

    labels = list(read_jsonl(labels_path))
    benchmark_rows = list(read_jsonl(benchmark_jsonl))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    for row in labels:
        if row.get("source") != EXPECTED_SOURCE:
            raise ValueError("Non-DECIMER source found")
        if row.get("task_type") != EXPECTED_TASK_TYPE:
            raise ValueError("Unexpected task_type found")
        if row.get("image_type") != EXPECTED_IMAGE_TYPE:
            raise ValueError("Unexpected image_type found")
        if row.get("difficulty") != EXPECTED_DIFFICULTY:
            raise ValueError("Unexpected difficulty found")
        image_path = bundle_root / row["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing bundle image: {image_path}")

    if stats.get("total") != len(labels):
        raise ValueError("Stats total mismatch")

    if len(benchmark_rows) != len(labels):
        raise ValueError("Benchmark count mismatch")

    for row in benchmark_rows:
        if row.get("source") != EXPECTED_SOURCE:
            raise ValueError("Benchmark contains non-DECIMER source")
        if row.get("task_type") != EXPECTED_TASK_TYPE:
            raise ValueError("Benchmark task_type mismatch")
        if row.get("difficulty") != EXPECTED_DIFFICULTY:
            raise ValueError("Benchmark difficulty mismatch")
        if row.get("prompt") != EXPECTED_PROMPT:
            raise ValueError("Benchmark prompt mismatch")

    return {
        "validation_passed": True,
        "total": len(labels),
        "decimer": len(labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--benchmark-jsonl", required=True)
    args = parser.parse_args()

    result = validate_bundle(Path(args.bundle_root).resolve(), Path(args.benchmark_jsonl).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
