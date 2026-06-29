#!/usr/bin/env python3
import argparse
import hashlib
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
from sweep_candidate_reward_head_margin import evaluate_prediction_rows, label_group_value  # noqa: E402
from train_candidate_reward_head import read_jsonl, write_jsonl  # noqa: E402


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


def init_acc():
    return {
        "total": 0,
        "raw_exact": 0,
        "canonical_exact": 0,
        "valid": 0,
    }


def rate(numerator: float, denominator: float):
    return 0.0 if denominator == 0 else numerator / denominator


def add_prediction(acc: dict, label: dict, pred_row: dict | None):
    acc["total"] += 1
    if pred_row is None:
        return

    gt_raw = get_ground_truth_smiles(label)
    gt_canonical = canonicalize_smiles(gt_raw) or gt_raw
    pred_raw = str(pred_row.get("prediction", "") or "").strip()
    pred_canonical = canonicalize_smiles(pred_raw)

    if pred_raw == gt_raw:
        acc["raw_exact"] += 1
    if pred_canonical is not None:
        acc["valid"] += 1
    if pred_canonical is not None and pred_canonical == gt_canonical:
        acc["canonical_exact"] += 1


def score_run(labels: dict, predictions: dict):
    acc = init_acc()
    for sample_id, label in labels.items():
        add_prediction(acc, label, predictions.get(sample_id))
    return acc


def score_run_by_group(labels: dict, predictions: dict, group_field: str):
    groups = defaultdict(init_acc)
    for sample_id, label in labels.items():
        group = label_group_value(label, group_field)
        add_prediction(groups[group], label, predictions.get(sample_id))
    return groups


def summarize_acc(acc: dict):
    total = acc.get("total", 0)
    return {
        "total": total,
        "canonical_exact": rate(acc.get("canonical_exact", 0), total),
        "raw_exact": rate(acc.get("raw_exact", 0), total),
        "valid_smiles_rate": rate(acc.get("valid", 0), total),
    }


def smoothed_key(acc: dict, global_acc: dict, shrinkage: float):
    total = acc.get("total", 0)
    denom = total + max(0.0, shrinkage)
    if denom <= 0:
        denom = 1.0
    global_total = global_acc.get("total", 0)
    global_canonical = rate(global_acc.get("canonical_exact", 0), global_total)
    global_raw = rate(global_acc.get("raw_exact", 0), global_total)
    global_valid = rate(global_acc.get("valid", 0), global_total)
    smooth = max(0.0, shrinkage)
    return (
        (acc.get("canonical_exact", 0) + smooth * global_canonical) / denom,
        (acc.get("raw_exact", 0) + smooth * global_raw) / denom,
        (acc.get("valid", 0) + smooth * global_valid) / denom,
    )


def choose_global_best(train_labels: dict, runs: dict, shrinkage: float):
    # shrinkage is irrelevant when comparing whole-train totals, but keeping the
    # same key shape makes ties deterministic with the later run-name fallback.
    run_scores = {name: score_run(train_labels, rows) for name, rows in runs.items()}
    best_name = max(
        runs,
        key=lambda name: (
            smoothed_key(run_scores[name], run_scores[name], shrinkage),
            name,
        ),
    )
    return best_name, run_scores


def choose_run_by_group(
    train_labels: dict,
    runs: dict,
    group_field: str,
    fallback_run: str,
    min_train_group_size: int,
    shrinkage: float,
    small_group_policy: str,
):
    global_best, global_scores = choose_global_best(train_labels, runs, shrinkage)
    run_group_scores = {
        run_name: score_run_by_group(train_labels, rows, group_field)
        for run_name, rows in runs.items()
    }
    group_names = sorted({label_group_value(label, group_field) for label in train_labels.values()})
    choices = {}
    table = {}
    for group_name in group_names:
        group_total = max(
            (run_group_scores[run_name].get(group_name, init_acc()).get("total", 0) for run_name in runs),
            default=0,
        )
        if group_total < min_train_group_size:
            selected = fallback_run if small_group_policy == "fallback" else global_best
            reason = f"small_group_{small_group_policy}"
        else:
            selected = max(
                runs,
                key=lambda run_name: (
                    smoothed_key(
                        run_group_scores[run_name].get(group_name, init_acc()),
                        global_scores[run_name],
                        shrinkage,
                    ),
                    run_name,
                ),
            )
            reason = "group_train_score"
        choices[group_name] = selected
        table[group_name] = {
            "selected_run": selected,
            "reason": reason,
            "train_total": group_total,
            "runs": {
                run_name: summarize_acc(run_group_scores[run_name].get(group_name, init_acc()))
                for run_name in runs
            },
        }
    return choices, table, global_best, {key: summarize_acc(value) for key, value in global_scores.items()}


def routed_row(sample_id: str, label: dict, runs: dict, group_field: str, choices: dict, fallback_run: str):
    group = label_group_value(label, group_field)
    run_name = choices.get(group, fallback_run)
    row = dict(runs.get(run_name, {}).get(sample_id) or runs[fallback_run].get(sample_id) or {"id": sample_id})
    row["selection_reason"] = f"cv_group_run_route:{group_field}:{run_name}"
    route_debug = dict(row.get("route_debug") or {})
    route_debug["group_field"] = group_field
    route_debug["group_value"] = group
    route_debug["selected_run"] = run_name
    row["route_debug"] = route_debug
    return row


def build_routed_rows(labels: dict, runs: dict, group_field: str, choices: dict, fallback_run: str):
    return [
        routed_row(sample_id, label, runs, group_field, choices, fallback_run)
        for sample_id, label in labels.items()
    ]


def metric_key(report: dict):
    accuracy = report.get("accuracy", {})
    similarity = report.get("similarity", {})
    return (
        accuracy.get("canonical_exact_match_accuracy") or 0.0,
        accuracy.get("raw_exact_match_accuracy") or 0.0,
        similarity.get("mean_fingerprint_tanimoto") or 0.0,
        accuracy.get("valid_smiles_rate") or 0.0,
    )


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


def slugify(text: str):
    return text.replace("+", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--run", action="append", required=True, help="NAME=prediction.jsonl. Pass multiple times.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-fields", required=True)
    parser.add_argument("--report-group-fields", default="source,difficulty,task_type,eval_panel")
    parser.add_argument("--fallback-run", default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260632)
    parser.add_argument("--min-train-group-size", type=int, default=8)
    parser.add_argument("--shrinkage", type=float, default=4.0)
    parser.add_argument("--small-group-policy", choices=["fallback", "global_best"], default="fallback")
    parser.add_argument("--write-full-label-route", action="store_true")
    args = parser.parse_args()

    if args.folds < 2:
        raise SystemExit("--folds must be >= 2")

    labels = {str(row["id"]): row for row in read_jsonl(Path(args.labels_jsonl))}
    run_specs = [parse_run_arg(item) for item in args.run]
    runs = {name: {str(row["id"]): row for row in read_jsonl(path)} for name, path in run_specs}
    fallback_run = args.fallback_run or run_specs[0][0]
    if fallback_run not in runs:
        raise SystemExit(f"fallback run not found: {fallback_run}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_group_fields = [field.strip() for field in args.report_group_fields.split(",") if field.strip()]
    group_fields = [field.strip() for field in args.group_fields.split(",") if field.strip()]

    baseline_reports = {}
    for run_name, rows in runs.items():
        report = evaluate_prediction_rows(labels, rows, report_group_fields)
        report_path = out_dir / f"report_baseline_{slugify(run_name)}.json"
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
            train_labels = {
                sample_id: label
                for sample_id, label in labels.items()
                if sample_id not in dev_ids
            }
            dev_labels = {
                sample_id: label
                for sample_id, label in labels.items()
                if sample_id in dev_ids
            }
            choices, choice_table, global_best, global_scores = choose_run_by_group(
                train_labels,
                runs,
                group_field,
                fallback_run,
                args.min_train_group_size,
                args.shrinkage,
                args.small_group_policy,
            )
            dev_rows = build_routed_rows(dev_labels, runs, group_field, choices, fallback_run)
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
                    "global_best_train_run": global_best,
                    "global_train_scores": global_scores,
                    "overall": metric_summary(dev_report),
                    "choice_table": choice_table,
                }
            )

        cv_rows = [cv_rows_by_id[sample_id] for sample_id in labels if sample_id in cv_rows_by_id]
        pred_path = out_dir / f"pred_cv_route_by_{slug}.jsonl"
        report_path = out_dir / f"report_cv_route_by_{slug}.json"
        write_jsonl(pred_path, cv_rows)
        cv_report = evaluate_prediction_rows(labels, {str(row["id"]): row for row in cv_rows}, report_group_fields)
        report_path.write_text(json.dumps(cv_report, ensure_ascii=False, indent=2), encoding="utf-8")

        result = {
            "group_field": group_field,
            "prediction_jsonl": str(pred_path),
            "report_json": str(report_path),
            "overall": metric_summary(cv_report),
            "fold_results": fold_results,
        }

        if args.write_full_label_route:
            choices, choice_table, global_best, global_scores = choose_run_by_group(
                labels,
                runs,
                group_field,
                fallback_run,
                args.min_train_group_size,
                args.shrinkage,
                args.small_group_policy,
            )
            full_rows = build_routed_rows(labels, runs, group_field, choices, fallback_run)
            full_pred_path = out_dir / f"pred_full_label_route_by_{slug}.jsonl"
            full_report_path = out_dir / f"report_full_label_route_by_{slug}.json"
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
                "global_best_train_run": global_best,
                "global_train_scores": global_scores,
                "choice_table": choice_table,
                "note": "Uses all visible labels to choose group routing; diagnostic only.",
            }

        group_results.append(result)

    best_cv = max(group_results, key=lambda item: metric_key(json.loads(Path(item["report_json"]).read_text(encoding="utf-8"))))
    summary = {
        "labels_jsonl": args.labels_jsonl,
        "runs": {name: str(path) for name, path in run_specs},
        "fallback_run": fallback_run,
        "folds": args.folds,
        "seed": args.seed,
        "min_train_group_size": args.min_train_group_size,
        "shrinkage": args.shrinkage,
        "small_group_policy": args.small_group_policy,
        "baseline_reports": baseline_reports,
        "group_results": group_results,
        "best_cv": {
            "group_field": best_cv["group_field"],
            "prediction_jsonl": best_cv["prediction_jsonl"],
            "report_json": best_cv["report_json"],
            "overall": best_cv["overall"],
        },
    }
    summary_path = out_dir / "crossval_route_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {
        "summary_json": str(summary_path),
        "fallback_run": fallback_run,
        "baseline_reports": {
            run_name: item["overall"]
            for run_name, item in baseline_reports.items()
        },
        "best_cv": summary["best_cv"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
