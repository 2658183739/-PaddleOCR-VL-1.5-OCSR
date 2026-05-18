from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


PHASES = ("phase0_edu", "phase1", "phase2", "phase3")
EVAL_TARGET = "eval"
TARGET_SELECTORS = {
    "phase0_edu": ["phase0_edu"],
    "phase1": ["phase1"],
    "phase2": ["phase2"],
    "phase3": ["phase3"],
    "all_phases": ["phase0_edu", "phase1", "phase2", "phase3"],
    "eval": ["eval"],
    "all": ["phase0_edu", "phase1", "phase2", "phase3", "eval"],
}
DEFAULT_DELETE_EXACT = {"P", "PA", "1", "I", "F"}
DEFAULT_DELETE_PREFIXES = ("I;16",)
REVIEW_MODES = {"RGBA", "LA", "L", "CMYK", "YCbCr"}
CONFIRM_DELETE_TOKEN = "DELETE_PHASE_MODE_CANDIDATES"


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


def phase_messages_path(project_root: Path, phase: str) -> Path:
    return project_root / "V2" / "data" / "sft_materialized" / f"train_{phase}_messages.jsonl"


def phase_manifest_path(project_root: Path, phase: str) -> Path:
    if phase == "phase0_edu":
        return project_root / "V2" / "data" / "manifests" / "edu_chemc_train_meta.jsonl"
    return project_root / "V2" / "data" / "manifests" / f"{phase}_train_meta.jsonl"


def eval_label_paths(project_root: Path) -> list[Path]:
    return [
        project_root / "V2" / "data" / "eval" / "canonical_smiles_main_v1" / "annotations" / "labels.jsonl",
        project_root / "V2" / "data" / "eval" / "edu_chmec_ssml_normed_test_v1" / "annotations" / "labels.jsonl",
    ]


def classify_mode(mode: str) -> dict[str, object]:
    normalized = str(mode or "").strip()
    if normalized == "RGB":
        return {
            "bucket": "ok",
            "default_delete_candidate": False,
            "reason": "Standard RGB image; keep.",
        }

    if normalized in DEFAULT_DELETE_EXACT or any(normalized.startswith(prefix) for prefix in DEFAULT_DELETE_PREFIXES):
        return {
            "bucket": "default_delete",
            "default_delete_candidate": True,
            "reason": "Known incompatible mode for saturation-related PIL blend path.",
        }

    if normalized in REVIEW_MODES:
        return {
            "bucket": "review",
            "default_delete_candidate": False,
            "reason": "Non-RGB mode that should be reviewed, not auto-deleted.",
        }

    return {
        "bucket": "review",
        "default_delete_candidate": False,
        "reason": "Unrecognized non-RGB mode; review manually before deleting.",
    }


def resolve_message_image_path(messages_jsonl: Path, image_ref: str) -> Path:
    return (messages_jsonl.parent / image_ref).resolve()


def load_manifest_records(manifest_jsonl: Path) -> dict[str, dict]:
    manifest_map = {}
    for record in read_jsonl(manifest_jsonl):
        record_id = str(record.get("id", "")).strip()
        if record_id:
            manifest_map[record_id] = record
    return manifest_map


def resolve_eval_image_path(labels_jsonl: Path, image_ref: str) -> Path:
    return (labels_jsonl.parent.parent / image_ref).resolve()


def audit_phase(project_root: Path, phase: str) -> dict[str, object]:
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}")

    messages_jsonl = phase_messages_path(project_root, phase)
    manifest_jsonl = phase_manifest_path(project_root, phase)
    if not messages_jsonl.exists():
        raise FileNotFoundError(f"Missing messages jsonl: {messages_jsonl}")
    if not manifest_jsonl.exists():
        raise FileNotFoundError(f"Missing manifest jsonl: {manifest_jsonl}")

    manifest_map = load_manifest_records(manifest_jsonl)
    records = []
    bucket_counts = Counter()
    mode_counts = Counter()

    for record in read_jsonl(messages_jsonl):
        meta = record.get("meta", {}) or {}
        record_id = str(meta.get("id", "")).strip()
        image_ref = ""
        images = record.get("images") or []
        if images:
            image_ref = str(images[0])

        resolved_path = resolve_message_image_path(messages_jsonl, image_ref) if image_ref else None
        manifest_record = manifest_map.get(record_id, {})

        row = {
            "phase": phase,
            "id": record_id,
            "source": meta.get("source", ""),
            "difficulty": meta.get("difficulty", ""),
            "task_type": meta.get("task_type", ""),
            "raw_image_ref": image_ref,
            "resolved_image_path": str(resolved_path) if resolved_path else "",
            "manifest_image_path": str(manifest_record.get("image_path", "")),
            "messages_jsonl": str(messages_jsonl),
            "manifest_jsonl": str(manifest_jsonl),
        }

        if not resolved_path or not resolved_path.exists():
            row.update(
                {
                    "mode": "",
                    "size": None,
                    "bucket": "error",
                    "default_delete_candidate": False,
                    "reason": "Image file missing.",
                    "error": f"Missing image: {resolved_path}",
                }
            )
            bucket_counts["error"] += 1
            records.append(row)
            continue

        try:
            with Image.open(resolved_path) as image:
                mode = image.mode
                size = list(image.size)
        except Exception as exc:  # pragma: no cover - kept for runtime safety
            row.update(
                {
                    "mode": "",
                    "size": None,
                    "bucket": "error",
                    "default_delete_candidate": False,
                    "reason": "Failed to open image.",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            bucket_counts["error"] += 1
            records.append(row)
            continue

        classification = classify_mode(mode)
        row.update(
            {
                "mode": mode,
                "size": size,
                "bucket": classification["bucket"],
                "default_delete_candidate": classification["default_delete_candidate"],
                "reason": classification["reason"],
                "error": "",
            }
        )
        bucket_counts[row["bucket"]] += 1
        mode_counts[mode] += 1
        records.append(row)

    return {
        "target": phase,
        "phase": phase,
        "messages_jsonl": str(messages_jsonl),
        "manifest_jsonl": str(manifest_jsonl),
        "summary": {
            "total_records": len(records),
            "bucket_counts": dict(bucket_counts),
            "mode_counts": dict(mode_counts),
            "default_delete_candidates": sum(1 for row in records if row["default_delete_candidate"]),
        },
        "records": records,
    }


def audit_eval(project_root: Path) -> dict[str, object]:
    label_paths = eval_label_paths(project_root)
    records = []
    bucket_counts = Counter()
    mode_counts = Counter()

    for labels_jsonl in label_paths:
        if not labels_jsonl.exists():
            raise FileNotFoundError(f"Missing eval labels jsonl: {labels_jsonl}")

        for record in read_jsonl(labels_jsonl):
            record_id = str(record.get("id", "")).strip()
            image_ref = str(record.get("image", "")).strip()
            resolved_path = resolve_eval_image_path(labels_jsonl, image_ref) if image_ref else None
            row = {
                "target": EVAL_TARGET,
                "phase": EVAL_TARGET,
                "id": record_id,
                "source": record.get("source", ""),
                "difficulty": record.get("difficulty", ""),
                "task_type": record.get("task_type", ""),
                "raw_image_ref": image_ref,
                "resolved_image_path": str(resolved_path) if resolved_path else "",
                "messages_jsonl": "",
                "manifest_jsonl": str(labels_jsonl),
                "labels_jsonl": str(labels_jsonl),
            }

            if not resolved_path or not resolved_path.exists():
                row.update(
                    {
                        "mode": "",
                        "size": None,
                        "bucket": "error",
                        "default_delete_candidate": False,
                        "reason": "Image file missing.",
                        "error": f"Missing image: {resolved_path}",
                    }
                )
                bucket_counts["error"] += 1
                records.append(row)
                continue

            try:
                with Image.open(resolved_path) as image:
                    mode = image.mode
                    size = list(image.size)
            except Exception as exc:  # pragma: no cover
                row.update(
                    {
                        "mode": "",
                        "size": None,
                        "bucket": "error",
                        "default_delete_candidate": False,
                        "reason": "Failed to open image.",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                bucket_counts["error"] += 1
                records.append(row)
                continue

            classification = classify_mode(mode)
            row.update(
                {
                    "mode": mode,
                    "size": size,
                    "bucket": classification["bucket"],
                    "default_delete_candidate": classification["default_delete_candidate"],
                    "reason": classification["reason"],
                    "error": "",
                }
            )
            bucket_counts[row["bucket"]] += 1
            mode_counts[mode] += 1
            records.append(row)

    return {
        "target": EVAL_TARGET,
        "summary": {
            "total_records": len(records),
            "bucket_counts": dict(bucket_counts),
            "mode_counts": dict(mode_counts),
            "default_delete_candidates": sum(1 for row in records if row["default_delete_candidate"]),
        },
        "records": records,
    }


def filter_jsonl_by_ids(jsonl_path: Path, remove_ids: set[str], id_getter, dry_run: bool) -> tuple[int, int, int]:
    records = list(read_jsonl(jsonl_path))
    kept = []
    removed = 0
    for record in records:
        if id_getter(record) in remove_ids:
            removed += 1
        else:
            kept.append(record)
    if not dry_run:
        write_jsonl(jsonl_path, kept)
    return len(records), removed, len(kept)


def apply_candidate_deletions(
    project_root: Path,
    candidate_rows: list[dict],
    delete_assets: bool,
    dry_run: bool,
) -> dict[str, dict[str, object]]:
    rows_by_phase: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        if row.get("default_delete_candidate"):
            rows_by_phase[str(row.get("target") or row.get("phase", ""))].append(row)

    summaries: dict[str, dict[str, object]] = {}
    for phase, rows in rows_by_phase.items():
        if phase == EVAL_TARGET:
            labels_map: dict[Path, list[dict]] = defaultdict(list)
            for row in rows:
                labels_map[Path(str(row.get("labels_jsonl", "")))].append(row)

            total_labels_before = total_labels_removed = total_labels_after = 0
            deleted_paths = []
            for labels_jsonl, label_rows in labels_map.items():
                remove_ids = {str(row.get("id", "")).strip() for row in label_rows if row.get("id")}
                before, removed, after = filter_jsonl_by_ids(
                    labels_jsonl,
                    remove_ids,
                    lambda item: str(item.get("id", "")).strip(),
                    dry_run=dry_run,
                )
                total_labels_before += before
                total_labels_removed += removed
                total_labels_after += after

                if delete_assets:
                    for row in label_rows:
                        raw_path = str(row.get("resolved_image_path", "")).strip()
                        if not raw_path:
                            continue
                        path = Path(raw_path)
                        if path.exists():
                            deleted_paths.append(str(path))
                            if not dry_run:
                                path.unlink()

            summaries[phase] = {
                "labels_before": total_labels_before,
                "labels_removed": total_labels_removed,
                "labels_after": total_labels_after,
                "assets_deleted": len(deleted_paths),
                "deleted_asset_paths": deleted_paths,
            }
            continue

        remove_ids = {str(row.get("id", "")).strip() for row in rows if row.get("id")}
        messages_jsonl = phase_messages_path(project_root, phase)
        manifest_jsonl = phase_manifest_path(project_root, phase)

        msg_before, msg_removed, msg_after = filter_jsonl_by_ids(
            messages_jsonl,
            remove_ids,
            lambda item: str((item.get("meta", {}) or {}).get("id", "")).strip(),
            dry_run=dry_run,
        )
        manifest_before, manifest_removed, manifest_after = filter_jsonl_by_ids(
            manifest_jsonl,
            remove_ids,
            lambda item: str(item.get("id", "")).strip(),
            dry_run=dry_run,
        )

        deleted_paths = []
        if delete_assets:
            for row in rows:
                raw_path = str(row.get("resolved_image_path", "")).strip()
                if not raw_path:
                    continue
                path = Path(raw_path)
                if path.exists():
                    deleted_paths.append(str(path))
                    if not dry_run:
                        path.unlink()

        summaries[phase] = {
            "messages_before": msg_before,
            "messages_removed": msg_removed,
            "messages_after": msg_after,
            "manifests_before": manifest_before,
            "manifests_removed": manifest_removed,
            "manifests_after": manifest_after,
            "assets_deleted": len(deleted_paths),
            "deleted_asset_paths": deleted_paths,
            "remove_ids": sorted(remove_ids),
        }
    return summaries


def write_report_files(report: dict[str, object], report_json: Path, candidate_json: Path, candidate_txt: Path):
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    candidate_rows = [
        row
        for target_report in report["targets"]
        for row in target_report["records"]
        if row["default_delete_candidate"]
    ]
    candidate_payload = {
        "generated_from": str(report_json),
        "confirm_delete_token": CONFIRM_DELETE_TOKEN,
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
    }
    candidate_json.write_text(json.dumps(candidate_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    candidate_txt.parent.mkdir(parents=True, exist_ok=True)
    with candidate_txt.open("w", encoding="utf-8") as handle:
        for row in candidate_rows:
            handle.write(f"{row['phase']}\t{row['id']}\t{row['mode']}\t{row['resolved_image_path']}\n")


def run_audit(project_root: Path, targets: list[str], report_json: Path, candidate_json: Path, candidate_txt: Path) -> dict[str, object]:
    target_reports = []
    for target in targets:
        if target == EVAL_TARGET:
            target_reports.append(audit_eval(project_root))
        else:
            target_reports.append(audit_phase(project_root, target))

    report = {
        "project_root": str(project_root),
        "targets": target_reports,
        "summary": {
            "total_records": sum(target_report["summary"]["total_records"] for target_report in target_reports),
            "default_delete_candidates": sum(target_report["summary"]["default_delete_candidates"] for target_report in target_reports),
        },
    }
    write_report_files(report, report_json, candidate_json, candidate_txt)
    return report


def parse_targets(value: str) -> list[str]:
    if value not in TARGET_SELECTORS:
        raise ValueError(f"Unsupported target selector: {value}")
    return list(TARGET_SELECTORS[value])


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--project-root", default=".")
    audit_parser.add_argument("--target", choices=list(TARGET_SELECTORS.keys()), default="all_phases")
    audit_parser.add_argument("--report-json", default="V2/reports/phase_image_mode_audit.json")
    audit_parser.add_argument("--candidate-json", default="V2/reports/phase_image_mode_delete_candidates.json")
    audit_parser.add_argument("--candidate-txt", default="V2/reports/phase_image_mode_delete_candidates.txt")

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--project-root", default=".")
    apply_parser.add_argument("--candidate-json", required=True)
    apply_parser.add_argument("--delete-assets", action="store_true")
    apply_parser.add_argument("--dry-run", action="store_true")
    apply_parser.add_argument("--confirm-delete", default="")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "audit":
        project_root = Path(args.project_root).resolve()
        targets = parse_targets(args.target)
        report = run_audit(
            project_root=project_root,
            targets=targets,
            report_json=(project_root / args.report_json).resolve(),
            candidate_json=(project_root / args.candidate_json).resolve(),
            candidate_txt=(project_root / args.candidate_txt).resolve(),
        )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return

    project_root = Path(args.project_root).resolve()
    candidate_payload = json.loads(Path(args.candidate_json).resolve().read_text(encoding="utf-8"))
    candidate_rows = candidate_payload.get("candidates", [])

    if not args.dry_run and args.confirm_delete != CONFIRM_DELETE_TOKEN:
        raise ValueError(
            f"Refusing destructive delete without --confirm-delete {CONFIRM_DELETE_TOKEN}"
        )

    summary = apply_candidate_deletions(
        project_root=project_root,
        candidate_rows=candidate_rows,
        delete_assets=args.delete_assets,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
