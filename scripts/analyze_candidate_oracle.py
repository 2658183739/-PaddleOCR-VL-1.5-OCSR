#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rerank_ocsr_candidates import (  # noqa: E402
    aggregate_candidates,
    canonicalize,
    choose_chem_light,
    get_ground_truth_smiles,
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def add_metric(acc: Counter, unique_count: int, candidate_count: int, state: dict):
    acc["total"] += 1
    acc["unique_valid_candidates_sum"] += unique_count
    acc["raw_candidate_count_sum"] += candidate_count
    for key, value in state.items():
        acc[key] += int(bool(value))


def rate(acc: Counter, key: str):
    total = acc.get("total", 0)
    return acc.get(key, 0) / total if total else 0.0


def finalize(acc: Counter):
    total = acc.get("total", 0)
    out = dict(acc)
    out["selected_exact_rate"] = rate(acc, "selected_exact")
    out["chem_light_exact_rate"] = rate(acc, "chem_light_exact")
    out["oracle_exact_rate"] = rate(acc, "oracle_exact")
    out["oracle_nonisomeric_rate"] = rate(acc, "oracle_nonisomeric")
    out["recoverable_by_rerank_rate"] = rate(acc, "recoverable_by_rerank")
    out["missing_from_candidates_rate"] = rate(acc, "missing_from_candidates")
    out["missing_prediction_rate"] = rate(acc, "missing_prediction")
    out["missing_target_rate"] = rate(acc, "missing_target")
    out["avg_unique_valid_candidates"] = (
        acc.get("unique_valid_candidates_sum", 0) / total if total else 0.0
    )
    out["avg_raw_candidate_count"] = acc.get("raw_candidate_count_sum", 0) / total if total else 0.0
    out["headroom_over_selected_exact"] = out["oracle_exact_rate"] - out["selected_exact_rate"]
    out["headroom_over_chem_light_exact"] = out["oracle_exact_rate"] - out["chem_light_exact_rate"]
    return out


def group_value(label: dict, prediction: dict | None, field: str):
    if prediction and prediction.get(field):
        return str(prediction[field])
    if label.get(field):
        return str(label[field])
    if field == "eval_panel":
        return "unknown"
    return ""


def analyze(predictions: dict, labels: dict, example_limit: int):
    overall = Counter()
    groups = {
        "eval_panel": defaultdict(Counter),
        "source": defaultdict(Counter),
        "task_type": defaultdict(Counter),
        "difficulty": defaultdict(Counter),
    }
    unique_counts = []
    raw_counts = []
    examples = {
        "recoverable_by_rerank": [],
        "missing_from_candidates": [],
        "stereo_or_isomer_only": [],
    }

    for sample_id, label in labels.items():
        prediction = predictions.get(sample_id)
        target = canonicalize(get_ground_truth_smiles(label))
        target_noniso = canonicalize(target, isomeric=False) if target else None
        state = {
            "missing_prediction": prediction is None,
            "missing_target": target is None,
            "selected_exact": False,
            "chem_light_exact": False,
            "oracle_exact": False,
            "oracle_nonisomeric": False,
            "recoverable_by_rerank": False,
            "missing_from_candidates": False,
        }
        unique_count = 0
        raw_count = 0
        selected_canonical = None
        chem_canonical = None
        aggregates = []

        if prediction is not None and target is not None:
            raw_candidates = prediction.get("candidates", []) or []
            raw_count = len(raw_candidates)
            aggregates = aggregate_candidates(raw_candidates)
            unique_count = len(aggregates)
            candidate_set = {item.get("canonical") for item in aggregates}
            noniso_set = {item.get("nonisomeric") for item in aggregates}
            selected_canonical = canonicalize(
                prediction.get("canonical_prediction") or prediction.get("prediction")
            )
            chem = (
                choose_chem_light(
                    aggregates,
                    salt_bonus=2.0,
                    heavy_penalty=0.02,
                    stereo_min_ratio=0.25,
                )
                if aggregates
                else None
            )
            chem_canonical = chem.get("canonical") if chem else None
            state["selected_exact"] = selected_canonical == target
            state["chem_light_exact"] = chem_canonical == target
            state["oracle_exact"] = target in candidate_set
            state["oracle_nonisomeric"] = target_noniso in noniso_set if target_noniso else False
            state["recoverable_by_rerank"] = state["oracle_exact"] and not state["selected_exact"]
            state["missing_from_candidates"] = not state["oracle_exact"]

        unique_counts.append(unique_count)
        raw_counts.append(raw_count)
        add_metric(overall, unique_count, raw_count, state)
        for field, bucket in groups.items():
            add_metric(bucket[group_value(label, prediction, field)], unique_count, raw_count, state)

        if len(examples["recoverable_by_rerank"]) < example_limit and state["recoverable_by_rerank"]:
            examples["recoverable_by_rerank"].append(
                {
                    "id": sample_id,
                    "eval_panel": group_value(label, prediction, "eval_panel"),
                    "source": group_value(label, prediction, "source"),
                    "task_type": group_value(label, prediction, "task_type"),
                    "target": target,
                    "selected": selected_canonical,
                    "chem_light": chem_canonical,
                    "unique_valid_candidates": unique_count,
                    "image": label.get("image") or label.get("image_path") or prediction.get("image_path", ""),
                }
            )
        if len(examples["missing_from_candidates"]) < example_limit and state["missing_from_candidates"]:
            examples["missing_from_candidates"].append(
                {
                    "id": sample_id,
                    "eval_panel": group_value(label, prediction, "eval_panel"),
                    "source": group_value(label, prediction, "source"),
                    "task_type": group_value(label, prediction, "task_type"),
                    "target": target,
                    "selected": selected_canonical,
                    "chem_light": chem_canonical,
                    "unique_valid_candidates": unique_count,
                    "image": label.get("image") or label.get("image_path") or (prediction or {}).get("image_path", ""),
                }
            )
        if (
            len(examples["stereo_or_isomer_only"]) < example_limit
            and state["oracle_nonisomeric"]
            and not state["oracle_exact"]
        ):
            examples["stereo_or_isomer_only"].append(
                {
                    "id": sample_id,
                    "eval_panel": group_value(label, prediction, "eval_panel"),
                    "source": group_value(label, prediction, "source"),
                    "task_type": group_value(label, prediction, "task_type"),
                    "target": target,
                    "target_nonisomeric": target_noniso,
                    "selected": selected_canonical,
                    "chem_light": chem_canonical,
                    "unique_valid_candidates": unique_count,
                    "candidate_nonisomeric": sorted(
                        item.get("canonical")
                        for item in aggregates
                        if item.get("nonisomeric") == target_noniso
                    ),
                    "image": label.get("image") or label.get("image_path") or (prediction or {}).get("image_path", ""),
                }
            )

    def percentile(values, q):
        if not values:
            return 0
        return statistics.quantiles(values, n=100, method="inclusive")[q - 1]

    return {
        "overall": finalize(overall),
        "by_group": {
            field: {key: finalize(value) for key, value in sorted(bucket.items())}
            for field, bucket in groups.items()
        },
        "candidate_count_stats": {
            "unique_valid_p50": statistics.median(unique_counts) if unique_counts else 0,
            "unique_valid_p90": percentile(unique_counts, 90),
            "unique_valid_max": max(unique_counts) if unique_counts else 0,
            "raw_p50": statistics.median(raw_counts) if raw_counts else 0,
            "raw_p90": percentile(raw_counts, 90),
            "raw_max": max(raw_counts) if raw_counts else 0,
        },
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional report path. Use empty or '-' to only print a compact summary.",
    )
    parser.add_argument("--example-limit", type=int, default=30)
    parser.add_argument(
        "--print-group-summary",
        action="store_true",
        help="Print compact overall and group metrics instead of only overall metrics.",
    )
    args = parser.parse_args()

    predictions = {str(row["id"]): row for row in read_jsonl(Path(args.prediction_jsonl))}
    labels = {str(row["id"]): row for row in read_jsonl(Path(args.labels_jsonl))}
    report = {
        "prediction_jsonl": args.prediction_jsonl,
        "labels_jsonl": args.labels_jsonl,
        **analyze(predictions, labels, args.example_limit),
    }
    if args.output_json and args.output_json != "-":
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def compact_metrics(metrics: dict):
        return {
            "total": metrics.get("total", 0),
            "selected_exact_rate": metrics.get("selected_exact_rate", 0.0),
            "chem_light_exact_rate": metrics.get("chem_light_exact_rate", 0.0),
            "oracle_exact_rate": metrics.get("oracle_exact_rate", 0.0),
            "oracle_nonisomeric_rate": metrics.get("oracle_nonisomeric_rate", 0.0),
            "recoverable_by_rerank_rate": metrics.get("recoverable_by_rerank_rate", 0.0),
            "missing_from_candidates_rate": metrics.get("missing_from_candidates_rate", 0.0),
            "avg_unique_valid_candidates": metrics.get("avg_unique_valid_candidates", 0.0),
            "headroom_over_selected_exact": metrics.get("headroom_over_selected_exact", 0.0),
        }

    if args.print_group_summary:
        compact = {"overall": compact_metrics(report["overall"]), "by_group": {}}
        for field, groups in report.get("by_group", {}).items():
            compact["by_group"][field] = {
                key: compact_metrics(value)
                for key, value in groups.items()
            }
        print(json.dumps(compact, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
