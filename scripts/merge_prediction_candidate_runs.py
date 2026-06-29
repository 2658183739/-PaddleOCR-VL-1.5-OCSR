#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rerank_ocsr_candidates import canonicalize  # noqa: E402


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


def parse_run_arg(value: str):
    if "=" not in value:
        raise ValueError(f"run must be NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty run name in: {value}")
    return name, Path(path)


def load_prediction_run(path: Path):
    rows = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("id", ""))
        if sample_id:
            rows[sample_id] = row
    return rows


def safe_int(value, default: int):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def normalize_candidate(run_name: str, row: dict, candidate: dict, source: str, index: int):
    prediction = str(candidate.get("prediction", "") or "").strip()
    canonical_prediction = candidate.get("canonical_prediction") or canonicalize(prediction)
    tta_name = str(candidate.get("tta_name", source) or source)
    out = dict(candidate)
    out["prediction"] = prediction
    out["canonical_prediction"] = canonical_prediction
    out["candidate_run"] = run_name
    out["candidate_source"] = source
    out["candidate_index"] = index
    out["prompt"] = candidate.get("prompt", row.get("prompt", ""))
    out["raw_text"] = candidate.get("raw_text", row.get("raw_text", ""))
    out["generation_score"] = candidate.get("generation_score")
    out["smiles_structure_penalty"] = candidate.get("smiles_structure_penalty", row.get("smiles_structure_penalty", 1000))
    out["prompt_index"] = safe_int(candidate.get("prompt_index"), 0 if source == "selected" else 99)
    out["tta_index"] = safe_int(candidate.get("tta_index"), 0 if source == "selected" else 99)
    out["generation_index"] = safe_int(candidate.get("generation_index"), 0 if source == "selected" else 99)
    out["tta_name"] = f"{run_name}:{tta_name}"
    return out


def selected_as_candidate(run_name: str, row: dict):
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


def collect_run_candidates(run_name: str, row: dict, include_selected: bool, max_candidates: int):
    candidates = []
    if include_selected and str(row.get("prediction", "")).strip():
        candidates.append(normalize_candidate(run_name, row, selected_as_candidate(run_name, row), "selected", 0))
    for index, candidate in enumerate(row.get("candidates") or [], start=1):
        if max_candidates > 0 and index > max_candidates:
            break
        candidates.append(normalize_candidate(run_name, row, candidate, "candidate", index))
    return candidates


def build_output_row(sample_id: str, reference_row: dict, label_row: dict, candidates: list[dict], per_run: dict):
    out = dict(reference_row) if reference_row else {}
    out["id"] = sample_id
    if label_row:
        out.setdefault("image_path", label_row.get("image") or label_row.get("image_path", ""))
        for key in ("source", "difficulty", "task_type", "eval_panel"):
            if key in label_row:
                out[key] = label_row[key]
    out["candidates"] = candidates
    out["selection_reason"] = "cross_run_candidate_pool_reference"
    out["candidate_pool_debug"] = {
        "candidate_count": len(candidates),
        "valid_candidate_count": sum(1 for item in candidates if item.get("canonical_prediction")),
        "run_count": len(per_run),
        "per_run_candidate_count": per_run,
        "unique_valid_canonical_count": len({item.get("canonical_prediction") for item in candidates if item.get("canonical_prediction")}),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-jsonl", default="")
    parser.add_argument("--run", action="append", required=True, help="Prediction run in NAME=PATH format. Pass multiple times.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--reference-run", default="", help="Run name whose selected prediction stays as the row prediction. Defaults to first run.")
    parser.add_argument("--include-selected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-candidates-per-run", type=int, default=0)
    parser.add_argument("--include-unlabeled-union", action="store_true")
    args = parser.parse_args()

    run_specs = [parse_run_arg(value) for value in args.run]
    runs = [(name, load_prediction_run(path)) for name, path in run_specs]
    reference_run = args.reference_run or runs[0][0]

    labels = {}
    ordered_ids = []
    if args.benchmark_jsonl:
        for row in read_jsonl(Path(args.benchmark_jsonl)):
            sample_id = str(row.get("id", ""))
            if sample_id:
                labels[sample_id] = row
                ordered_ids.append(sample_id)

    if args.include_unlabeled_union or not ordered_ids:
        seen = set(ordered_ids)
        for _, rows in runs:
            for sample_id in rows:
                if sample_id not in seen:
                    ordered_ids.append(sample_id)
                    seen.add(sample_id)

    output_rows = []
    missing_by_run = Counter()
    for sample_id in ordered_ids:
        all_candidates = []
        per_run = {}
        reference_row = None
        first_row = None
        for run_name, rows in runs:
            row = rows.get(sample_id)
            if row is None:
                missing_by_run[run_name] += 1
                continue
            if first_row is None:
                first_row = row
            if run_name == reference_run:
                reference_row = row
            run_candidates = collect_run_candidates(
                run_name,
                row,
                include_selected=args.include_selected,
                max_candidates=args.max_candidates_per_run,
            )
            per_run[run_name] = len(run_candidates)
            all_candidates.extend(run_candidates)
        if reference_row is None:
            reference_row = first_row
        if reference_row is None:
            reference_row = {"id": sample_id, "prediction": "", "canonical_prediction": None}
        output_rows.append(build_output_row(sample_id, reference_row, labels.get(sample_id, {}), all_candidates, per_run))

    write_jsonl(Path(args.output_jsonl), output_rows)
    summary = {
        "benchmark_jsonl": args.benchmark_jsonl,
        "output_jsonl": args.output_jsonl,
        "reference_run": reference_run,
        "include_selected": args.include_selected,
        "max_candidates_per_run": args.max_candidates_per_run,
        "rows": len(output_rows),
        "runs": {name: str(path) for name, path in run_specs},
        "missing_by_run": dict(missing_by_run),
        "candidate_count": sum(len(row.get("candidates", [])) for row in output_rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
