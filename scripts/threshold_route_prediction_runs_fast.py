#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
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
from sweep_candidate_reward_head_margin import evaluate_prediction_rows, label_group_value  # noqa: E402
from train_candidate_reward_head import read_jsonl, write_jsonl  # noqa: E402


NEG_INF = float("-inf")
POS_INF = float("inf")


def parse_run_arg(value: str):
    if "=" not in value:
        raise ValueError(f"run must be NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty run name in: {value}")
    return name, Path(path)


def stable_fold(sample_id: str, seed: int, folds: int):
    digest = hashlib.sha1(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % folds


def nested_get(row: dict | None, path: str):
    cur = row or {}
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def score_value(row: dict | None, score_path: str):
    value = nested_get(row, score_path)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return NEG_INF


def exact_tuple(label: dict, row: dict | None):
    if row is None:
        return (0, 0, 0)
    gt_raw = get_ground_truth_smiles(label)
    gt_canonical = canonicalize_smiles(gt_raw) or gt_raw
    pred_raw = str(row.get("prediction", "") or "").strip()
    pred_canonical = canonicalize_smiles(pred_raw)
    return (
        int(pred_canonical is not None and pred_canonical == gt_canonical),
        int(pred_raw == gt_raw),
        int(pred_canonical is not None),
    )


def metric_key(counts: tuple[int, int, int], total: int):
    denom = max(1, total)
    return (counts[0] / denom, counts[1] / denom, counts[2] / denom)


def add_tuple(left: tuple[int, int, int], right: tuple[int, int, int]):
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def sub_tuple(left: tuple[int, int, int], right: tuple[int, int, int]):
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def best_rule_for_records(records: list[dict]):
    total = len(records)
    main_counts = (0, 0, 0)
    for record in records:
        main_counts = add_tuple(main_counts, record["main_exact"])

    best = {
        "threshold": POS_INF,
        "direction": "ge",
        "train_total": total,
        "choice_count": 0,
        "train_key": metric_key(main_counts, total),
        "reason": "threshold_sweep",
    }

    def consider(counts, direction, threshold, choice_count):
        nonlocal best
        key = metric_key(counts, total)
        current = (key, -choice_count, direction, -float(threshold) if math.isfinite(threshold) else 0.0)
        prior = (
            tuple(best["train_key"]),
            -int(best["choice_count"]),
            best["direction"],
            -float(best["threshold"]) if math.isfinite(float(best["threshold"])) else 0.0,
        )
        if current > prior:
            best = {
                "threshold": threshold,
                "direction": direction,
                "train_total": total,
                "choice_count": choice_count,
                "train_key": key,
                "reason": "threshold_sweep",
            }

    finite_or_missing = sorted(records, key=lambda item: item["score"], reverse=True)
    counts = main_counts
    choice_count = 0
    index = 0
    while index < len(finite_or_missing):
        score = finite_or_missing[index]["score"]
        while index < len(finite_or_missing) and finite_or_missing[index]["score"] == score:
            record = finite_or_missing[index]
            counts = add_tuple(sub_tuple(counts, record["main_exact"]), record["choice_exact"])
            choice_count += 1
            index += 1
        consider(counts, "ge", score, choice_count)

    finite_or_missing = sorted(records, key=lambda item: item["score"])
    counts = main_counts
    choice_count = 0
    index = 0
    while index < len(finite_or_missing):
        score = finite_or_missing[index]["score"]
        while index < len(finite_or_missing) and finite_or_missing[index]["score"] == score:
            record = finite_or_missing[index]
            counts = add_tuple(sub_tuple(counts, record["main_exact"]), record["choice_exact"])
            choice_count += 1
            index += 1
        consider(counts, "le", score, choice_count)

    return best


def build_records(labels: dict, main_rows: dict, choice_rows: dict, score_path: str):
    records = {}
    for sample_id, label in labels.items():
        main_row = main_rows.get(sample_id)
        choice_row = choice_rows.get(sample_id)
        records[sample_id] = {
            "id": sample_id,
            "label": label,
            "score": score_value(choice_row, score_path),
            "main_exact": exact_tuple(label, main_row),
            "choice_exact": exact_tuple(label, choice_row),
        }
    return records


def fit_policy(train_records: dict, group_field: str, min_train_group_size: int):
    fallback = best_rule_for_records(list(train_records.values()))
    groups = defaultdict(list)
    for record in train_records.values():
        groups[label_group_value(record["label"], group_field)].append(record)
    rules = {}
    table = {}
    for group, group_records in sorted(groups.items()):
        if len(group_records) < min_train_group_size:
            rule = dict(fallback)
            rule["reason"] = "small_group_global"
        else:
            rule = best_rule_for_records(group_records)
            rule["reason"] = "group_threshold"
        rules[group] = rule
        table[group] = rule
    return {"group_field": group_field, "rules": rules}, fallback, table


def select_choice(score: float, rule: dict):
    threshold = float(rule["threshold"])
    if rule["direction"] == "ge":
        return score >= threshold
    if rule["direction"] == "le":
        return score <= threshold
    raise ValueError(f"unknown direction: {rule['direction']}")


def build_rows(labels: dict, records: dict, main_rows: dict, choice_rows: dict, policy: dict, fallback: dict, score_path: str):
    rows = []
    for sample_id, label in labels.items():
        record = records[sample_id]
        group = label_group_value(label, policy["group_field"])
        rule = policy["rules"].get(group) or fallback
        main_row = main_rows.get(sample_id)
        choice_row = choice_rows.get(sample_id)
        use_choice = select_choice(record["score"], rule) and choice_row is not None
        selected_run = "choice" if use_choice else "main"
        selected = choice_row if use_choice else main_row
        out = dict(selected or main_row or choice_row or {"id": sample_id})
        out["selection_reason"] = (
            f"threshold_route:{policy['group_field']}:{selected_run}:"
            f"{rule['direction']}:{rule['threshold']}"
        )
        route_debug = dict(out.get("route_debug") or {})
        route_debug.update(
            {
                "router": "threshold_route_fast",
                "group_field": policy["group_field"],
                "group_value": group,
                "selected_run": selected_run,
                "choice_score_path": score_path,
                "choice_score": record["score"],
                "threshold": rule["threshold"],
                "direction": rule["direction"],
                "train_total": rule.get("train_total"),
                "choice_count": rule.get("choice_count"),
                "rule_reason": rule.get("reason"),
            }
        )
        out["route_debug"] = route_debug
        rows.append(out)
    return rows


def metric_summary(report: dict):
    accuracy = report.get("accuracy", {})
    similarity = report.get("similarity", {})
    by_eval_panel = report.get("by_group", {}).get("eval_panel", {})
    return {
        "total": report.get("total"),
        "canonical_exact": accuracy.get("canonical_exact_match_accuracy"),
        "raw_exact": accuracy.get("raw_exact_match_accuracy"),
        "valid_smiles_rate": accuracy.get("valid_smiles_rate"),
        "mean_tanimoto": similarity.get("mean_fingerprint_tanimoto"),
        "by_eval_panel_exact": {
            key: value.get("canonical_exact_match_accuracy")
            for key, value in by_eval_panel.items()
        },
    }


def report_key(report: dict):
    accuracy = report.get("accuracy", {})
    similarity = report.get("similarity", {})
    return (
        accuracy.get("canonical_exact_match_accuracy") or 0.0,
        accuracy.get("raw_exact_match_accuracy") or 0.0,
        similarity.get("mean_fingerprint_tanimoto") or 0.0,
        accuracy.get("valid_smiles_rate") or 0.0,
    )


def slugify(text: str):
    return text.replace("+", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--main-run", required=True, help="main=prediction.jsonl")
    parser.add_argument("--choice-run", required=True, help="choice=prediction.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-fields", required=True)
    parser.add_argument("--score-path", default="reward_head_debug.reward_head_score")
    parser.add_argument("--report-group-fields", default="source,difficulty,task_type,eval_panel")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260633)
    parser.add_argument("--min-train-group-size", type=int, default=8)
    parser.add_argument("--write-full-label-route", action="store_true")
    args = parser.parse_args()

    main_name, main_path = parse_run_arg(args.main_run)
    choice_name, choice_path = parse_run_arg(args.choice_run)
    if main_name != "main" or choice_name != "choice":
        raise SystemExit("run names must be main=... and choice=...")

    labels = {str(row["id"]): row for row in read_jsonl(Path(args.labels_jsonl))}
    main_rows = {str(row["id"]): row for row in read_jsonl(main_path)}
    choice_rows = {str(row["id"]): row for row in read_jsonl(choice_path)}
    records = build_records(labels, main_rows, choice_rows, args.score_path)

    group_fields = [field.strip() for field in args.group_fields.split(",") if field.strip()]
    report_group_fields = [field.strip() for field in args.report_group_fields.split(",") if field.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_reports = {}
    for run_name, rows in (("main", main_rows), ("choice", choice_rows)):
        report = evaluate_prediction_rows(labels, rows, report_group_fields)
        report_path = out_dir / f"report_baseline_{run_name}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        baseline_reports[run_name] = {
            "report_json": str(report_path),
            "overall": metric_summary(report),
        }

    fold_ids = {
        fold: {
            sample_id
            for sample_id in labels
            if stable_fold(sample_id, args.seed, args.folds) == fold
        }
        for fold in range(args.folds)
    }

    group_results = []
    for group_field in group_fields:
        slug = slugify(group_field)
        cv_rows_by_id = {}
        fold_results = []
        for fold in range(args.folds):
            dev_ids = fold_ids[fold]
            train_records = {
                sample_id: record
                for sample_id, record in records.items()
                if sample_id not in dev_ids
            }
            dev_labels = {
                sample_id: label
                for sample_id, label in labels.items()
                if sample_id in dev_ids
            }
            policy, fallback, table = fit_policy(train_records, group_field, args.min_train_group_size)
            dev_rows = build_rows(dev_labels, records, main_rows, choice_rows, policy, fallback, args.score_path)
            cv_rows_by_id.update({str(row["id"]): row for row in dev_rows})
            dev_report = evaluate_prediction_rows(
                dev_labels,
                {str(row["id"]): row for row in dev_rows},
                report_group_fields,
            )
            fold_results.append(
                {
                    "fold": fold,
                    "train_total": len(train_records),
                    "dev_total": len(dev_labels),
                    "fallback_rule": fallback,
                    "choice_table": table,
                    "overall": metric_summary(dev_report),
                }
            )

        cv_rows = [cv_rows_by_id[sample_id] for sample_id in labels if sample_id in cv_rows_by_id]
        cv_pred_path = out_dir / f"pred_cv_threshold_route_by_{slug}.jsonl"
        cv_report_path = out_dir / f"report_cv_threshold_route_by_{slug}.json"
        write_jsonl(cv_pred_path, cv_rows)
        cv_report = evaluate_prediction_rows(labels, {str(row["id"]): row for row in cv_rows}, report_group_fields)
        cv_report_path.write_text(json.dumps(cv_report, ensure_ascii=False, indent=2), encoding="utf-8")

        result = {
            "group_field": group_field,
            "prediction_jsonl": str(cv_pred_path),
            "report_json": str(cv_report_path),
            "overall": metric_summary(cv_report),
            "fold_results": fold_results,
        }

        if args.write_full_label_route:
            policy, fallback, table = fit_policy(records, group_field, args.min_train_group_size)
            full_rows = build_rows(labels, records, main_rows, choice_rows, policy, fallback, args.score_path)
            full_pred_path = out_dir / f"pred_full_label_threshold_route_by_{slug}.jsonl"
            full_report_path = out_dir / f"report_full_label_threshold_route_by_{slug}.json"
            write_jsonl(full_pred_path, full_rows)
            full_report = evaluate_prediction_rows(
                labels,
                {str(row["id"]): row for row in full_rows},
                report_group_fields,
            )
            full_report_path.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
            result["full_label_route"] = {
                "prediction_jsonl": str(full_pred_path),
                "report_json": str(full_report_path),
                "overall": metric_summary(full_report),
                "fallback_rule": fallback,
                "choice_table": table,
                "note": "Uses all visible labels to fit thresholds; diagnostic only.",
            }

        group_results.append(result)

    best_cv = max(group_results, key=lambda item: report_key(json.loads(Path(item["report_json"]).read_text(encoding="utf-8"))))
    summary = {
        "labels_jsonl": args.labels_jsonl,
        "main_run": str(main_path),
        "choice_run": str(choice_path),
        "score_path": args.score_path,
        "folds": args.folds,
        "seed": args.seed,
        "min_train_group_size": args.min_train_group_size,
        "baseline_reports": baseline_reports,
        "group_results": group_results,
        "best_cv": {
            "group_field": best_cv["group_field"],
            "prediction_jsonl": best_cv["prediction_jsonl"],
            "report_json": best_cv["report_json"],
            "overall": best_cv["overall"],
        },
    }
    summary_path = out_dir / "threshold_route_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_json": str(summary_path),
                "baseline_reports": {key: value["overall"] for key, value in baseline_reports.items()},
                "best_cv": summary["best_cv"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
