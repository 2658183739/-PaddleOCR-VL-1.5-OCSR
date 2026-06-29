#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_ocsr_predictions_detailed import (  # noqa: E402
    canonicalize_smiles,
    get_ground_truth_smiles,
)
from sweep_candidate_reward_head_margin import evaluate_prediction_rows  # noqa: E402
from train_candidate_reward_head import read_jsonl, write_jsonl  # noqa: E402


def parse_run_arg(value: str):
    if "=" not in value:
        raise ValueError(f"run must be NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty run name in: {value}")
    return name, Path(path)


def load_run(path: Path):
    return {str(row["id"]): row for row in read_jsonl(path)}


def label_group_value(label: dict, field: str):
    parts = [part.strip() for part in field.split("+") if part.strip()]
    if len(parts) <= 1:
        return str(label.get(field, "unknown"))
    return "|".join(str(label.get(part, "unknown")) for part in parts)


def init_acc():
    return {
        "total": 0,
        "raw_exact": 0,
        "canonical_exact": 0,
        "valid": 0,
    }


def score_run_by_group(labels: dict, predictions: dict, group_field: str):
    groups = defaultdict(init_acc)
    for sample_id, label in labels.items():
        group = label_group_value(label, group_field)
        acc = groups[group]
        acc["total"] += 1
        target_raw = get_ground_truth_smiles(label)
        target_canonical = canonicalize_smiles(target_raw) or target_raw
        pred_row = predictions.get(sample_id)
        if pred_row is None:
            continue
        pred_raw = str(pred_row.get("prediction", "") or "").strip()
        pred_canonical = canonicalize_smiles(pred_raw)
        if pred_raw == target_raw:
            acc["raw_exact"] += 1
        if pred_canonical is not None:
            acc["valid"] += 1
        if pred_canonical is not None and pred_canonical == target_canonical:
            acc["canonical_exact"] += 1
    return groups


def group_key(acc: dict):
    total = acc.get("total", 0) or 1
    return (
        acc.get("canonical_exact", 0) / total,
        acc.get("raw_exact", 0) / total,
        acc.get("valid", 0) / total,
    )


def choose_run_by_group(labels: dict, runs: dict, group_field: str):
    run_group_scores = {
        run_name: score_run_by_group(labels, rows, group_field)
        for run_name, rows in runs.items()
    }
    group_names = sorted({label_group_value(label, group_field) for label in labels.values()})
    choices = {}
    table = {}
    for group_name in group_names:
        best_name = max(
            runs,
            key=lambda run_name: group_key(run_group_scores[run_name].get(group_name, init_acc())),
        )
        choices[group_name] = best_name
        acc = run_group_scores[best_name].get(group_name, init_acc())
        total = acc.get("total", 0) or 1
        table[group_name] = {
            "run": best_name,
            "total": acc.get("total", 0),
            "canonical_exact": acc.get("canonical_exact", 0) / total,
            "raw_exact": acc.get("raw_exact", 0) / total,
            "valid_smiles_rate": acc.get("valid", 0) / total,
        }
    return choices, table


def build_routed_rows(labels: dict, runs: dict, group_field: str, choices: dict, fallback_run: str):
    rows = []
    for sample_id, label in labels.items():
        group = label_group_value(label, group_field)
        run_name = choices.get(group, fallback_run)
        row = dict(runs[run_name].get(sample_id) or runs[fallback_run].get(sample_id) or {"id": sample_id})
        row["selection_reason"] = f"group_run_route:{group_field}:{run_name}"
        route_debug = dict(row.get("route_debug") or {})
        route_debug["group_field"] = group_field
        route_debug["group_value"] = group
        route_debug["selected_run"] = run_name
        row["route_debug"] = route_debug
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--run", action="append", required=True, help="NAME=prediction.jsonl. Pass multiple times.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-fields", required=True)
    parser.add_argument("--report-group-fields", default="source,difficulty,task_type,eval_panel")
    parser.add_argument("--fallback-run", default="")
    args = parser.parse_args()

    labels = {str(row["id"]): row for row in read_jsonl(Path(args.labels_jsonl))}
    run_specs = [parse_run_arg(item) for item in args.run]
    runs = {name: load_run(path) for name, path in run_specs}
    fallback_run = args.fallback_run or run_specs[0][0]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_group_fields = [field.strip() for field in args.report_group_fields.split(",") if field.strip()]

    summary = {
        "labels_jsonl": args.labels_jsonl,
        "runs": {name: str(path) for name, path in run_specs},
        "fallback_run": fallback_run,
        "group_results": [],
    }
    for group_field in [field.strip() for field in args.group_fields.split(",") if field.strip()]:
        choices, table = choose_run_by_group(labels, runs, group_field)
        output_rows = build_routed_rows(labels, runs, group_field, choices, fallback_run)
        slug = group_field.replace("+", "_").replace("/", "_").replace("\\", "_")
        pred_path = out_dir / f"pred_route_by_{slug}.jsonl"
        report_path = out_dir / f"report_route_by_{slug}.json"
        write_jsonl(pred_path, output_rows)
        report = evaluate_prediction_rows(labels, {str(row["id"]): row for row in output_rows}, report_group_fields)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        accuracy = report.get("accuracy", {})
        similarity = report.get("similarity", {})
        summary["group_results"].append(
            {
                "group_field": group_field,
                "prediction_jsonl": str(pred_path),
                "report_json": str(report_path),
                "overall": {
                    "canonical_exact": accuracy.get("canonical_exact_match_accuracy"),
                    "raw_exact": accuracy.get("raw_exact_match_accuracy"),
                    "valid_smiles_rate": accuracy.get("valid_smiles_rate"),
                    "mean_tanimoto": similarity.get("mean_fingerprint_tanimoto"),
                },
                "choice_table": table,
            }
        )

    best = max(
        summary["group_results"],
        key=lambda item: (
            item["overall"].get("canonical_exact") or 0.0,
            item["overall"].get("raw_exact") or 0.0,
            item["overall"].get("mean_tanimoto") or 0.0,
            item["overall"].get("valid_smiles_rate") or 0.0,
        ),
    )
    summary["best"] = best
    summary_path = out_dir / "route_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
