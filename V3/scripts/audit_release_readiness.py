import argparse
import csv
import json
from collections import Counter
from pathlib import Path


MANIFESTS = {
    "final_train_control": "V3/data/sft_materialized/train_v3_a_control.jsonl",
    "dev_legacy_core": "V3/data/eval/dev_legacy_core_strict/labels.jsonl",
    "dev_legacy_region": "V3/data/eval/dev_legacy_region_strict/labels.jsonl",
    "wild_strict_locked": "V3/data/eval/wild_strict_v3/labels.jsonl",
    "wild_symbolic_locked": "V3/data/eval/wild_symbolic_v3/labels.jsonl",
}


def value(row, name):
    if row.get(name) not in (None, ""):
        return row[name]
    meta = row.get("meta") or {}
    return meta.get(name)


def audit_manifest(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    license_present = sum(bool(value(row, "license")) for row in rows)
    source_url_present = sum(bool(value(row, "source_url_or_doc")) for row in rows)
    structure_id_present = sum(bool(value(row, "structure_id")) for row in rows)
    qc_counts = Counter(str(value(row, "qc_status") or "missing") for row in rows)
    source_counts = Counter(str(value(row, "source") or "missing") for row in rows)
    return {
        "path": str(path),
        "rows": total,
        "license_present": license_present,
        "license_coverage": license_present / total if total else 0.0,
        "source_url_present": source_url_present,
        "source_url_coverage": source_url_present / total if total else 0.0,
        "structure_id_present": structure_id_present,
        "structure_id_coverage": structure_id_present / total if total else 0.0,
        "qc_status_counts": dict(qc_counts),
        "source_counts": dict(source_counts),
    }


def build_audit(project_root: Path):
    manifests = {
        name: audit_manifest(project_root / relative)
        for name, relative in MANIFESTS.items()
    }
    project_files = {
        "LICENSE": (project_root / "V3" / "LICENSE").is_file(),
        "NOTICE": (project_root / "V3" / "NOTICE").is_file(),
        "data_license_matrix": (project_root / "V3" / "DATA_LICENSES_AND_ATTRIBUTION_zh.md").is_file(),
        "CONTRIBUTING": (project_root / "V3" / "CONTRIBUTING.md").is_file(),
        "Dockerfile": (project_root / "V3" / "Dockerfile").is_file(),
        "model_card": (project_root / "V3" / "MODEL_CARD_zh.md").is_file(),
        "dataset_card": (project_root / "V3" / "DATASET_CARD_zh.md").is_file(),
        "reproduction_guide": (project_root / "V3" / "REPRODUCTION_GUIDE_zh.md").is_file(),
    }
    manual_review = project_root / "V3" / "qc" / "eval_manual_review.csv"
    review_rows = []
    if manual_review.exists():
        with manual_review.open("r", encoding="utf-8-sig", newline="") as handle:
            review_rows = list(csv.DictReader(handle))
    review_panel_counts = Counter(row.get("panel") or "missing" for row in review_rows)
    pending_review_rows = sum(
        (row.get("final_decision") or "").strip().lower() in {"", "pending"}
        for row in review_rows
    )
    private_labels = project_root / "V3" / "data" / "eval" / "private_photo_v3" / "labels.jsonl"
    private_photo_rows = (
        sum(bool(line.strip()) for line in private_labels.read_text(encoding="utf-8").splitlines())
        if private_labels.exists()
        else 0
    )
    blockers = []
    attestation_path = project_root / "V3" / "qc" / "manual_review_attestation.json"
    attestation = {}
    if attestation_path.is_file():
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attested_complete = attestation.get("status") == "owner_attested_complete"
    if manifests["final_train_control"]["license_coverage"] < 1.0:
        blockers.append("final training manifest lacks complete sample-level license and source URL fields")
    if not attested_complete and (pending_review_rows or manifests["wild_strict_locked"]["qc_status_counts"].get("pending_manual_review")):
        blockers.append("locked labels require two independent human reviews and adjudication or a project-owner attestation")
    if private_photo_rows == 0:
        blockers.append("private-photo evaluation set is empty")
    if not project_files["LICENSE"] or not project_files["NOTICE"] or not project_files["data_license_matrix"] or not project_files["CONTRIBUTING"]:
        blockers.append("project license, NOTICE, data license matrix, and CONTRIBUTING must be present")
    if not project_files["Dockerfile"]:
        blockers.append("no container image or Dockerfile has been independently reproduced")
    return {
        "manifests": manifests,
        "project_files": project_files,
        "manual_review_sheet_present": manual_review.is_file(),
        "manual_review_rows": len(review_rows),
        "manual_review_pending_rows": pending_review_rows,
        "manual_review_panel_counts": dict(review_panel_counts),
        "manual_review_attestation_present": attestation_path.is_file(),
        "manual_review_attested_complete": attested_complete,
        "manual_review_attestation_scope": attestation.get("scope", {}),
        "private_photo_rows": private_photo_rows,
        "release_blockers": blockers,
    }


def render_markdown(audit):
    lines = [
        "# V3 release-readiness audit",
        "",
        "| manifest | rows | license coverage | source URL coverage | structure ID coverage | QC status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in audit["manifests"].items():
        lines.append(
            f"| {name} | {row['rows']} | {row['license_coverage']:.1%} | "
            f"{row['source_url_coverage']:.1%} | {row['structure_id_coverage']:.1%} | "
            f"`{json.dumps(row['qc_status_counts'], ensure_ascii=False, sort_keys=True)}` |"
        )
    lines.extend(["", "## Project files", ""])
    for name, present in audit["project_files"].items():
        lines.append(f"- {name}: {'present' if present else 'missing'}")
    lines.extend(
        [
            "",
            "## Human and private evaluation status",
            "",
            f"- Manual-review sheet rows: {audit['manual_review_rows']}",
            f"- Pending final decisions: {audit['manual_review_pending_rows']}",
            f"- Review panels: `{json.dumps(audit['manual_review_panel_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- Owner attestation: {'complete' if audit.get('manual_review_attested_complete') else 'missing'}",
            f"- Private-photo rows: {audit['private_photo_rows']}",
        ]
    )
    lines.extend(["", "## Release blockers", ""])
    for blocker in audit["release_blockers"]:
        lines.append(f"- {blocker}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-json", default="V3/evidence/release_readiness_audit.json")
    parser.add_argument("--output-md", default="V3/evidence/release_readiness_audit.md")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    audit = build_audit(project_root)
    (project_root / args.output_json).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (project_root / args.output_md).write_text(render_markdown(audit), encoding="utf-8")
    print(project_root / args.output_md)


if __name__ == "__main__":
    main()
