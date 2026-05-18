from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_audit_module():
    module_path = Path(__file__).resolve().with_name("audit_current_evalsets.py")
    spec = importlib.util.spec_from_file_location("audit_current_evalsets", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load audit_current_evalsets from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_problem_lookup(report: dict) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for target_payload in report["targets"].values():
        for row in target_payload["records"]:
            if row["problem"]:
                pairs.add((row["dataset"], row["id"]))
    return pairs


def filter_validation_dataset(jsonl_path: Path, problems: set[tuple[str, str]], out_path: Path) -> dict[str, object]:
    records = list(read_jsonl(jsonl_path))
    kept = []
    removed = 0
    for record in records:
        record_id = str((record.get("meta", {}) or {}).get("id", ""))
        if (jsonl_path.name, record_id) in problems:
            removed += 1
        else:
            kept.append(record)
    write_jsonl(out_path, kept)
    return {"source": str(jsonl_path), "output": str(out_path), "before": len(records), "removed": removed, "after": len(kept)}


def filter_eval_dataset(labels_path: Path, dataset_name: str, problems: set[tuple[str, str]], out_path: Path) -> dict[str, object]:
    records = list(read_jsonl(labels_path))
    kept = []
    removed = 0
    for record in records:
        record_id = str(record.get("id", ""))
        if (dataset_name, record_id) in problems:
            removed += 1
        else:
            kept.append(record)
    write_jsonl(out_path, kept)
    return {"source": str(labels_path), "output": str(out_path), "before": len(records), "removed": removed, "after": len(kept)}


def build_clean_evalsets(project_root: Path, out_root: Path) -> dict[str, object]:
    audit_module = load_audit_module()
    audit_report = audit_module.run_audit(project_root, out_root / "current_evalsets_audit_for_clean.json")
    problems = build_problem_lookup(audit_report)

    validation_dir = out_root / "validation"
    evaluation_dir = out_root / "evaluation"

    validation_summary = filter_validation_dataset(
        project_root / "V2" / "data" / "sft_materialized" / "val_messages.jsonl",
        problems,
        validation_dir / "val_messages_clean.jsonl",
    )
    eval_summary = filter_eval_dataset(
        project_root / "V2" / "data" / "eval" / "canonical_smiles_main_v1" / "annotations" / "labels.jsonl",
        "canonical_smiles_main_v1",
        problems,
        evaluation_dir / "canonical_smiles_main_v1_clean.jsonl",
    )

    summary = {
        "validation": validation_summary,
        "evaluation": eval_summary,
    }
    (out_root / "clean_evalsets_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-root", default="V2/reports/clean_evalsets")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = (project_root / args.out_root).resolve()
    summary = build_clean_evalsets(project_root, out_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
