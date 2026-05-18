from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


DEFAULT_DELETE_EXACT = {"P", "PA", "1", "I", "F"}
DEFAULT_DELETE_PREFIXES = ("I;16",)
REVIEW_MODES = {"RGBA", "LA", "L", "CMYK", "YCbCr"}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def classify_mode(mode: str) -> dict[str, object]:
    normalized = str(mode or "").strip()
    if normalized == "RGB":
        return {"bucket": "ok", "problem": False, "reason": "Standard RGB image."}
    if normalized in DEFAULT_DELETE_EXACT or any(normalized.startswith(prefix) for prefix in DEFAULT_DELETE_PREFIXES):
        return {"bucket": "default_delete", "problem": True, "reason": "Known inference-risk image mode."}
    if normalized in REVIEW_MODES:
        return {"bucket": "review", "problem": False, "reason": "Non-RGB but not default-delete."}
    return {"bucket": "review", "problem": False, "reason": "Unrecognized non-RGB mode; review manually."}


def resolve_message_image_path(messages_jsonl: Path, image_ref: str) -> Path:
    return (messages_jsonl.parent / image_ref).resolve()


def resolve_eval_image_path(labels_jsonl: Path, image_ref: str) -> Path:
    return (labels_jsonl.parent.parent / image_ref).resolve()


def inspect_image(image_path: Path) -> dict[str, object]:
    if not image_path.exists():
        return {"mode": "", "size": None, "problem": True, "bucket": "missing", "reason": f"Missing image: {image_path}"}
    try:
        with Image.open(image_path) as image:
            mode = image.mode
            size = list(image.size)
    except Exception as exc:  # pragma: no cover
        return {"mode": "", "size": None, "problem": True, "bucket": "error", "reason": f"{type(exc).__name__}: {exc}"}
    result = classify_mode(mode)
    return {"mode": mode, "size": size, **result}


def audit_validation(project_root: Path) -> dict[str, object]:
    targets = [
        project_root / "V2" / "data" / "sft_materialized" / "val_messages.jsonl",
        project_root / "V2" / "data" / "sft_materialized" / "val_phase0_edu_messages.jsonl",
    ]
    rows = []
    bucket_counts = Counter()
    for jsonl_path in targets:
        if not jsonl_path.exists():
            continue
        for record in read_jsonl(jsonl_path):
            image_ref = str((record.get("images") or [""])[0])
            image_path = resolve_message_image_path(jsonl_path, image_ref)
            inspection = inspect_image(image_path)
            bucket_counts[inspection["bucket"]] += 1
            rows.append(
                {
                    "dataset": jsonl_path.name,
                    "id": str((record.get("meta", {}) or {}).get("id", "")),
                    "image_ref": image_ref,
                    "resolved_image_path": str(image_path),
                    **inspection,
                }
            )
    return {
        "summary": {
            "total_records": len(rows),
            "problem_records": sum(1 for row in rows if row["problem"]),
            "bucket_counts": dict(bucket_counts),
        },
        "records": rows,
    }


def audit_evaluation(project_root: Path) -> dict[str, object]:
    targets = [
        project_root / "V2" / "data" / "eval" / "canonical_smiles_main_v1" / "annotations" / "labels.jsonl",
        project_root / "V2" / "data" / "eval" / "edu_chmec_ssml_normed_test_v1" / "annotations" / "labels.jsonl",
    ]
    rows = []
    bucket_counts = Counter()
    for jsonl_path in targets:
        if not jsonl_path.exists():
            continue
        for record in read_jsonl(jsonl_path):
            image_ref = str(record.get("image", record.get("image_path", "")))
            image_path = resolve_eval_image_path(jsonl_path, image_ref)
            inspection = inspect_image(image_path)
            bucket_counts[inspection["bucket"]] += 1
            rows.append(
                {
                    "dataset": str(jsonl_path.parent.parent.name),
                    "id": str(record.get("id", "")),
                    "image_ref": image_ref,
                    "resolved_image_path": str(image_path),
                    **inspection,
                }
            )
    return {
        "summary": {
            "total_records": len(rows),
            "problem_records": sum(1 for row in rows if row["problem"]),
            "bucket_counts": dict(bucket_counts),
        },
        "records": rows,
    }


def run_audit(project_root: Path, report_json: Path) -> dict[str, object]:
    report = {
        "project_root": str(project_root),
        "targets": {
            "validation": audit_validation(project_root),
            "evaluation": audit_evaluation(project_root),
        },
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--report-json", default="V2/reports/current_evalsets_audit.json")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    report = run_audit(project_root, (project_root / args.report_json).resolve())
    summary = {name: payload["summary"] for name, payload in report["targets"].items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
