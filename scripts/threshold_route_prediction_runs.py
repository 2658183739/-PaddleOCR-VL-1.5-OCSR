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


def exact_flags(label: dict, row: dict | None):
    if row is None:
        return {"raw": False, "canonical": False, "valid": False}
    gt_raw = get_ground_truth_smiles(label)
    gt_canonical = canonicalize_smiles(gt_raw) or gt_raw
    pred_raw = str(row.get("prediction", "") or "").strip()
    pred_canonical = canonicalize_smiles(pred_raw)
    return {
        "raw": pred_raw == gt_raw,
        "canonical": pred_canonical is not None and pred_canonical == gt_canonical,
        "valid": pred_canonical is not None,
    }


def metric_key(rows_by_id: dict, labels: dict):
    canonical = 0
    raw = 0
    valid = 0
    total = 0
    for sample_id, label in labels.items():
        flags = exact_flags(label, rows_by_id.get(sample_id))
        total += 1
        canonical += int(flags["canonical"])
        raw += int(flags["raw"])
        valid += int(flags["valid"])
    denom = max(1, total)
    return (canonical / denom, raw / denom, valid / denom)


def threshold_candidates(scores: list[float]):
    finite = sorted({score for score in scores if math.isfinite(score)})
    thresholds = [NEG_INF, POS_INF]
    thresholds.extend(finite)
    for left, right in zip(finite, finite[1:]):
        thresholds.append((left + right) / 2.0)
    return sorted(set(thresholds))


def choose_row(main_row: dict | None, choice_row: dict | None, score: float, threshold: float, direction: str):
    use_choice = score >= threshold if direction == "ge" else score <= threshold
    if use_choice and choice_row is not None:
        return "choice", choice_row
    return "main", main_row


def build_rows(labels: dict, main_rows: dict, choice_rows: dict, score_path: str, policy: dict, fallback_policy: dict):
    output = []
    for sample_id, label in labels.items():
        group = label_group_value(label, policy["group_field"])
        rule = policy["rules"].get(group) or fallback_policy
        main_row = main_rows.get(sample_id)
        choice_row = choice_rows.get(sample_id)
        score = score_value(choice_row, score_path)
        selected_run, selected = choose_row(
            main_row,
            choice_row,
            score,
            float(rule["threshold"]),
            rule["direction"],
        )
        out = dict(selected or main_row or choice_row or {"id": sample_id})
        out["selection_reason"] = (
            f"threshold_route:{policy['group_field']}:{selected_run}:"
            f"{rule['direction']}:{rule['threshold']}"
        )
        route_debug = dict(out.get("route_debug") or {})
        route_debug.update(
            {
                "router": "threshold_route",
                "group_field": policy["group_field"],
                "group_value": group,
                "selected_run": selected_run,
                "choice_score_path": score_path,
                "choice_score": score,
                "threshold": rule["threshold"],
                "direction": rule["direction"],
                "train_total": rule.get("train_total"),
            }
        )
        out["route_debug"] = route_debug
        output.append(out)
    return output


def score_policy(labels: dict, main_rows: dict, choice_rows: dict, score_path: str, group_field: str, rules: dict, fallback_rule: dict):
    rows = build_rows(
        labels,
        main_rows,
        choice_rows,
        score_path,
        {"group_field": group_field, "rules": rules},
        fallback_rule,
    )
    return metric_key({str(row["id"]): row for row in rows}, labels)


def fit_global_rule(labels: dict, main_rows: dict, choice_rows: dict, score_path: str):
    scores = [score_value(choice_rows.get(sample_id), score_path) for sample_id in labels]
    best = None
    for direction in ("ge", "le"):
        for threshold in threshold_candidates(scores):
            rules = {"__global__": {"threshold": threshold, "direction": direction, "train_total": len(labels)}}
            key = score_policy(
                labels,
                main_rows,
                choice_rows,
                score_path,
                "__global_field__",
                {},
                rules["__global__"],
            )
            item = (key, direction, threshold)
            if best is None or item > best:
                best = item
    assert best is not None
    key, direction, threshold = best
    return {"threshold": threshold, "direction": direction, "train_total": len(labels), "train_key": key}


def fit_group_policy(
    train_labels: dict,
    main_rows: dict,
    choice_rows: dict,
    score_path: str,
    group_field: str,
    min_train_group_size: int,
):
    fallback = fit_global_rule(train_labels, main_rows, choice_rows, score_path)
    groups = defaultdict(dict)
    for sample_id, label in train_labels.items():
        groups[label_group_value(label, group_field)][sample_id] = label
    rules = {}
    table = {}
    for group, group_labels in sorted(groups.items()):
        if len(group_labels) < min_train_group_size:
            rules[group] = dict(fallback, reason="small_group_global")
            table[group] = {"reason": "small_group_global", **rules[group]}
            continue
        group_scores = [score_value(choice_rows.get(sample_id), score_path) for sample_id in group_labels]
        best = None
        for direction in ("ge", "le"):
            for threshold in threshold_candidates(group_scores):
                rule = {"threshold": threshold, "direction": direction, "train_total": len(group_labels)}
                key = score_policy(
                    group_labels,
                    main_rows,
                    choice_rows,
                    score_path,
                    group_field,
                    {group: rule},
                    fallback,
                )
                item = (key, direction, threshold)
                if best is None or item > best:
                    best = item
        key, direction, threshold = best
        rules[group] = {
            "threshold": threshold,
            "direction": direction,
            "train_total": len(group_labels),
            "train_key": key,
            "reason": "group_threshold",
        }
        table[group] = dict(rules[group])
    return {"group_field": group_field, "rules": rules}, fallback, table


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
    parser.add_argument("--main-run", required=True, help="NAME=prediction.jsonl")
    parser.add_argument("--choice-run", required=True, help="NAME=prediction.jsonl")
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
        raise SystemExit("run names must be main=... and choice=... for now")

    labels = {str(row["id"]): row for row in read_jsonl(Path(args.labels_jsonl))}
    main_rows = {str(row["id"]): row for row in read_jsonl(main_path)}
    choice_rows = {str(row["id"]): row for row in read_jsonl(choice_path)}
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
            train_labels = {sample_id: label for sample_id, label in labels.items() if sample_id not in dev_ids}
            dev_labels = {sample_id: label for sample_id, label in labels.items() if sample_id in dev_ids}
            policy, fallback, table = fit_group_policy(
                train_labels,
                main_rows,
                choice_rows,
                args.score_path,
                group_field,
                args.min_train_group_size,
            )
            dev_rows = build_rows(dev_labels, main_rows, choice_rows, args.score_path, policy, fallback)
            cv_rows_by_id.update({str(row["id"]): row for row in dev_rows})
            dev_report = evaluate_prediction_rows(
                dev_labels,
                {str(row["id"]): row for row in dev_rows},
                report_group_fields,
            )
            fold_results.append(
                {
                    "fold": fold,
                    "train_total": len(train_labels),
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
            policy, fallback, table = fit_group_policy(
                labels,
                main_rows,
                choice_rows,
                args.score_path,
                group_field,
                args.min_train_group_size,
            )
            full_rows = build_rows(labels, main_rows, choice_rows, args.score_path, policy, fallback)
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
                "baseline_reports": {k: v["overall"] for k, v in baseline_reports.items()},
                "best_cv": summary["best_cv"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
