#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_ocsr_predictions_detailed import (  # noqa: E402
    canonicalize_smiles,
    compute_overlap_metrics,
    finalize_group,
    fingerprint_tanimoto,
    get_ground_truth_smiles,
    init_group_accumulator,
    normalized_edit_similarity,
    safe_rate,
    tokenize_smiles,
)
from rerank_ocsr_candidates import aggregate_candidates, choose_chem_light  # noqa: E402
from reward_policy_reranker import bounded_score, candidate_feature_dict, vectorize  # noqa: E402
from train_candidate_reward_head import (  # noqa: E402
    RewardHead,
    choose_selected,
    read_jsonl,
    write_jsonl,
)


def parse_margin_grid(text: str):
    margins = []
    for item in text.split(","):
        item = item.strip()
        if item:
            margins.append(float(item))
    if not margins:
        raise SystemExit("empty margin grid")
    return margins


def margin_slug(margin: float):
    return f"m{margin:.2f}".replace("-", "n").replace(".", "p")


def load_reward_head(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args", {})
    feature_names = checkpoint.get("feature_names", [])
    model = RewardHead(
        len(feature_names),
        int(train_args.get("hidden_dim", 64)),
        float(train_args.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model


def score_aggregates(model, normalizer: dict, row: dict, aggregates: list[dict]):
    features = [candidate_feature_dict(row, aggregate, aggregates) for aggregate in aggregates]
    vectors = torch.tensor([vectorize(feature, normalizer) for feature in features], dtype=torch.float32)
    with torch.no_grad():
        scores = model(vectors).tolist()
    if not isinstance(scores, list):
        scores = [float(scores)]
    return [float(score) for score in scores]


def choose_policy_index(aggregates: list[dict], scores: list[float]):
    return max(
        range(len(aggregates)),
        key=lambda index: (
            scores[index],
            aggregates[index].get("count", 0),
            bounded_score(aggregates[index].get("max_score")),
            -aggregates[index].get("min_prompt_index", 99),
            -aggregates[index].get("min_generation_index", 99),
        ),
    )


def choose_fallback_index(row: dict, aggregates: list[dict], fallback_mode: str):
    if fallback_mode == "selected":
        fallback = choose_selected(row, aggregates)
    elif fallback_mode == "chem_light":
        fallback = choose_chem_light(
            aggregates,
            salt_bonus=2.0,
            heavy_penalty=0.02,
            stereo_min_ratio=0.25,
        )
    elif fallback_mode == "none":
        fallback = None
    else:
        raise ValueError(f"unknown fallback mode: {fallback_mode}")
    if fallback is None:
        return None
    return aggregates.index(fallback)


def build_selection_cache(model, normalizer: dict, prediction_rows: dict, fallback_mode: str):
    cache = []
    for sample_id, row in prediction_rows.items():
        aggregates = aggregate_candidates(row.get("candidates", []))
        if not aggregates:
            cache.append({"id": sample_id, "row": row, "aggregates": []})
            continue
        scores = score_aggregates(model, normalizer, row, aggregates)
        policy_index = choose_policy_index(aggregates, scores)
        fallback_index = choose_fallback_index(row, aggregates, fallback_mode)
        cache.append(
            {
                "id": sample_id,
                "row": row,
                "aggregates": aggregates,
                "scores": scores,
                "policy_index": policy_index,
                "fallback_index": fallback_index,
            }
        )
    return cache


def selected_output_row(cached: dict, chosen_index: int, score: float, reason: str, keep_candidates: bool):
    row = cached["row"]
    chosen = cached["aggregates"][chosen_index]
    rep = chosen.get("representative", {})
    out = dict(row)
    out["prediction"] = rep.get("prediction") or chosen.get("canonical")
    out["canonical_prediction"] = chosen.get("canonical")
    out["generation_score"] = rep.get("generation_score")
    out["smiles_structure_penalty"] = rep.get("smiles_structure_penalty")
    out["raw_text"] = rep.get("raw_text", "")
    out["prompt"] = rep.get("prompt", row.get("prompt", ""))
    out["selection_reason"] = reason
    out["vote_count"] = chosen.get("count")
    out["reward_head_debug"] = {
        "reward_head_score": score,
        "unique_valid_candidates": len(cached["aggregates"]),
        "policy_margin": None,
    }
    if not keep_candidates:
        out.pop("candidates", None)
    return out


def build_output_rows_from_cache(cache: list[dict], fallback_mode: str, policy_margin: float, keep_candidates: bool):
    output = []
    for cached in cache:
        row = cached["row"]
        aggregates = cached["aggregates"]
        if not aggregates:
            out = dict(row)
            out["selection_reason"] = "reward_head_no_valid_candidate"
            if not keep_candidates:
                out.pop("candidates", None)
            output.append(out)
            continue

        policy_index = cached["policy_index"]
        fallback_index = cached["fallback_index"]
        scores = cached["scores"]
        chosen_index = policy_index
        reason = "reward_head"

        if fallback_index is not None:
            policy = aggregates[policy_index]
            fallback = aggregates[fallback_index]
            if fallback.get("canonical") == policy.get("canonical"):
                chosen_index = fallback_index
                reason = f"reward_head_{fallback_mode}_agreement"
            elif scores[policy_index] - scores[fallback_index] >= policy_margin:
                reason = f"reward_head_override_{fallback_mode}"
            else:
                chosen_index = fallback_index
                reason = f"reward_head_fallback_{fallback_mode}"

        out = selected_output_row(cached, chosen_index, scores[chosen_index], reason, keep_candidates)
        out["reward_head_debug"]["policy_margin"] = policy_margin
        output.append(out)
    return output


def evaluate_prediction_rows(benchmark: dict, prediction_rows: dict, group_fields: list[str]):
    groups = {field: defaultdict(init_group_accumulator) for field in group_fields}
    overall = init_group_accumulator()

    for sample_id, target in benchmark.items():
        gt_raw = get_ground_truth_smiles(target)
        gt_canonical = canonicalize_smiles(gt_raw) or gt_raw
        pred_row = prediction_rows.get(sample_id)

        overall["total"] += 1
        for field in group_fields:
            groups[field][label_group_value(target, field)]["total"] += 1

        if pred_row is None:
            overall["missing_predictions"] += 1
            for field in group_fields:
                groups[field][label_group_value(target, field)]["missing_predictions"] += 1
            continue

        pred_raw = str(pred_row.get("prediction", "")).strip()
        pred_canonical = canonicalize_smiles(pred_raw)
        eval_gt = gt_canonical
        eval_pred = pred_canonical if pred_canonical is not None else pred_raw
        token_metrics = compute_overlap_metrics(tokenize_smiles(eval_gt), tokenize_smiles(eval_pred))
        edit_similarity = normalized_edit_similarity(eval_gt, eval_pred)
        tanimoto = fingerprint_tanimoto(gt_canonical, pred_canonical) if pred_canonical else None
        raw_exact = pred_raw == gt_raw
        canonical_exact = pred_canonical is not None and pred_canonical == gt_canonical
        valid_prediction = pred_canonical is not None

        if raw_exact:
            overall["raw_exact"] += 1
        if canonical_exact:
            overall["canonical_exact"] += 1
        if valid_prediction:
            overall["valid_predictions"] += 1
        overall["token_tp"] += token_metrics["overlap"]
        overall["token_pred"] += token_metrics["pred_count"]
        overall["token_gt"] += token_metrics["gt_count"]
        overall["token_macro_precision_sum"] += token_metrics["precision"]
        overall["token_macro_recall_sum"] += token_metrics["recall"]
        overall["token_macro_f1_sum"] += token_metrics["f1"]
        overall["edit_similarity_sum"] += edit_similarity
        if tanimoto is not None:
            overall["fingerprint_tanimoto_sum"] += tanimoto
            overall["fingerprint_tanimoto_count"] += 1

        for field in group_fields:
            acc = groups[field][label_group_value(target, field)]
            if raw_exact:
                acc["raw_exact"] += 1
            if canonical_exact:
                acc["canonical_exact"] += 1
            if valid_prediction:
                acc["valid_predictions"] += 1
            acc["token_tp"] += token_metrics["overlap"]
            acc["token_pred"] += token_metrics["pred_count"]
            acc["token_gt"] += token_metrics["gt_count"]
            acc["token_macro_precision_sum"] += token_metrics["precision"]
            acc["token_macro_recall_sum"] += token_metrics["recall"]
            acc["token_macro_f1_sum"] += token_metrics["f1"]
            acc["edit_similarity_sum"] += edit_similarity
            if tanimoto is not None:
                acc["fingerprint_tanimoto_sum"] += tanimoto
                acc["fingerprint_tanimoto_count"] += 1

    finalized = finalize_group(overall)
    report = {
        "total": overall["total"],
        "missing_predictions": overall["missing_predictions"],
        "accuracy": {
            "raw_exact_match_accuracy": safe_rate(overall["raw_exact"], overall["total"]),
            "canonical_exact_match_accuracy": safe_rate(overall["canonical_exact"], overall["total"]),
            "valid_smiles_rate": safe_rate(overall["valid_predictions"], overall["total"]),
        },
        "f1": {
            "smiles_token_micro_precision": finalized["token_micro_precision"],
            "smiles_token_micro_recall": finalized["token_micro_recall"],
            "smiles_token_micro_f1": finalized["token_micro_f1"],
            "smiles_token_macro_precision": finalized["token_macro_precision"],
            "smiles_token_macro_recall": finalized["token_macro_recall"],
            "smiles_token_macro_f1": finalized["token_macro_f1"],
        },
        "similarity": {
            "mean_normalized_edit_similarity": finalized["mean_normalized_edit_similarity"],
            "mean_fingerprint_tanimoto": finalized["mean_fingerprint_tanimoto"],
            "fingerprint_tanimoto_coverage": finalized["fingerprint_tanimoto_coverage"],
        },
        "by_group": {},
    }
    for field in group_fields:
        report["by_group"][field] = {
            group_name: finalize_group(acc)
            for group_name, acc in sorted(groups[field].items())
        }
    return report


def label_group_value(label: dict, field: str):
    parts = [part.strip() for part in field.split("+") if part.strip()]
    if len(parts) <= 1:
        return str(label.get(field, "unknown"))
    return "|".join(str(label.get(part, "unknown")) for part in parts)


def metric_row(margin: float, report: dict, pred_path: Path, report_path: Path):
    accuracy = report.get("accuracy", {})
    similarity = report.get("similarity", {})
    by_eval_panel = report.get("by_group", {}).get("eval_panel", {})
    return {
        "margin": margin,
        "prediction_jsonl": str(pred_path),
        "report_json": str(report_path),
        "canonical_exact": accuracy.get("canonical_exact_match_accuracy"),
        "raw_exact": accuracy.get("raw_exact_match_accuracy"),
        "valid_smiles_rate": accuracy.get("valid_smiles_rate"),
        "mean_tanimoto": similarity.get("mean_fingerprint_tanimoto"),
        "canonical_panel_exact": by_eval_panel.get("canonical_smiles_main_v1", {}).get(
            "canonical_exact_match_accuracy"
        ),
        "weak_panel_exact": by_eval_panel.get("weak_domain_v2", {}).get("canonical_exact_match_accuracy"),
    }


def metric_key_from_report(report: dict):
    accuracy = report.get("accuracy", {})
    similarity = report.get("similarity", {})
    return (
        accuracy.get("canonical_exact_match_accuracy") or 0.0,
        accuracy.get("raw_exact_match_accuracy") or 0.0,
        similarity.get("mean_fingerprint_tanimoto") or 0.0,
        accuracy.get("valid_smiles_rate") or 0.0,
    )


def metric_key_from_group(group_report: dict):
    return (
        group_report.get("canonical_exact_match_accuracy") or 0.0,
        group_report.get("raw_exact_match_accuracy") or 0.0,
        group_report.get("mean_fingerprint_tanimoto") or 0.0,
        group_report.get("valid_smiles_rate") or 0.0,
    )


def build_group_margin_output(
    field: str,
    labels: dict,
    rows_by_margin: dict[float, list[dict]],
    reports_by_margin: dict[float, dict],
    global_best_margin: float,
    out_dir: Path,
    group_fields: list[str],
):
    group_names = sorted(
        {
            label_group_value(row, field)
            for row in labels.values()
        }
    )
    best_margin_by_group = {}
    group_metric_table = {}
    for group_name in group_names:
        candidates = []
        for margin, report in reports_by_margin.items():
            group_report = report.get("by_group", {}).get(field, {}).get(group_name)
            if group_report is None:
                continue
            candidates.append((margin, group_report))
        if not candidates:
            best_margin_by_group[group_name] = global_best_margin
            continue
        margin, group_report = max(candidates, key=lambda item: metric_key_from_group(item[1]))
        best_margin_by_group[group_name] = margin
        group_metric_table[group_name] = {
            "margin": margin,
            "canonical_exact": group_report.get("canonical_exact_match_accuracy"),
            "raw_exact": group_report.get("raw_exact_match_accuracy"),
            "valid_smiles_rate": group_report.get("valid_smiles_rate"),
            "mean_tanimoto": group_report.get("mean_fingerprint_tanimoto"),
            "total": group_report.get("total"),
        }

    row_lookup_by_margin = {
        margin: {str(row["id"]): row for row in rows}
        for margin, rows in rows_by_margin.items()
    }
    output_rows = []
    for sample_id, label in labels.items():
        group_name = label_group_value(label, field)
        margin = best_margin_by_group.get(group_name, global_best_margin)
        output_rows.append(row_lookup_by_margin[margin][sample_id])

    slug = field.replace("/", "_").replace("\\", "_")
    pred_path = out_dir / f"pred_pair_reward_group_margin_{slug}.jsonl"
    report_path = out_dir / f"report_pair_reward_group_margin_{slug}.json"
    write_jsonl(pred_path, output_rows)
    report = evaluate_prediction_rows(labels, {str(row["id"]): row for row in output_rows}, group_fields)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "group_field": field,
        "prediction_jsonl": str(pred_path),
        "report_json": str(report_path),
        "best_margin_by_group": best_margin_by_group,
        "group_metric_table": group_metric_table,
        "overall": {
            "canonical_exact": report.get("accuracy", {}).get("canonical_exact_match_accuracy"),
            "raw_exact": report.get("accuracy", {}).get("raw_exact_match_accuracy"),
            "valid_smiles_rate": report.get("accuracy", {}).get("valid_smiles_rate"),
            "mean_tanimoto": report.get("similarity", {}).get("mean_fingerprint_tanimoto"),
        },
    }


def write_markdown_summary(path: Path, summary: dict):
    lines = [
        "# Candidate reward head margin sweep",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- prediction_jsonl: `{summary['prediction_jsonl']}`",
        f"- labels_jsonl: `{summary['labels_jsonl']}`",
        f"- fallback_mode: `{summary['fallback_mode']}`",
        f"- best_margin: `{summary['best']['margin']}`",
        f"- best_canonical_exact: `{summary['best']['canonical_exact']}`",
        "",
        "| margin | canonical exact | raw exact | mean Tanimoto | canonical panel | weak panel |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["results"]:
        lines.append(
            "| {margin:.2f} | {canonical_exact:.6f} | {raw_exact:.6f} | {mean_tanimoto:.6f} | "
            "{canonical_panel_exact:.6f} | {weak_panel_exact:.6f} |".format(
                margin=item["margin"],
                canonical_exact=item["canonical_exact"] or 0.0,
                raw_exact=item["raw_exact"] or 0.0,
                mean_tanimoto=item["mean_tanimoto"] or 0.0,
                canonical_panel_exact=item["canonical_panel_exact"] or 0.0,
                weak_panel_exact=item["weak_panel_exact"] or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "Best prediction:",
            "",
            f"`{summary['best']['prediction_jsonl']}`",
            "",
            "Best report:",
            "",
            f"`{summary['best']['report_json']}`",
            "",
        ]
    )
    for item in summary.get("group_margin_results", []):
        overall = item.get("overall", {})
        lines.extend(
            [
                "",
                f"Group margin by `{item['group_field']}`:",
                "",
                f"- canonical exact: `{overall.get('canonical_exact')}`",
                f"- raw exact: `{overall.get('raw_exact')}`",
                f"- mean Tanimoto: `{overall.get('mean_tanimoto')}`",
                f"- prediction: `{item.get('prediction_jsonl')}`",
                f"- report: `{item.get('report_json')}`",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fallback-mode", choices=["none", "selected", "chem_light"], default="")
    parser.add_argument("--margin-grid", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.75")
    parser.add_argument("--group-fields", default="source,difficulty,task_type,eval_panel")
    parser.add_argument(
        "--group-margin-fields",
        default="",
        help="Comma-separated label fields. For each field, select the best margin per group and emit an extra prediction file.",
    )
    parser.add_argument("--keep-candidates", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, model = load_reward_head(Path(args.checkpoint))
    train_args = checkpoint.get("args", {})
    fallback_mode = args.fallback_mode or train_args.get("fallback_mode", "chem_light")
    prediction_rows = {str(row["id"]): row for row in read_jsonl(Path(args.prediction_jsonl))}
    labels = {str(row["id"]): row for row in read_jsonl(Path(args.labels_jsonl))}
    group_fields = [field.strip() for field in args.group_fields.split(",") if field.strip()]
    group_margin_fields = [field.strip() for field in args.group_margin_fields.split(",") if field.strip()]
    for field in group_margin_fields:
        if field not in group_fields:
            group_fields.append(field)
    selection_cache = build_selection_cache(model, checkpoint["normalizer"], prediction_rows, fallback_mode)

    results = []
    rows_by_margin = {}
    reports_by_margin = {}
    for margin in parse_margin_grid(args.margin_grid):
        slug = margin_slug(margin)
        pred_path = out_dir / f"pred_pair_reward_{slug}.jsonl"
        report_path = out_dir / f"report_pair_reward_{slug}.json"
        output_rows = build_output_rows_from_cache(selection_cache, fallback_mode, margin, args.keep_candidates)
        write_jsonl(pred_path, output_rows)
        report = evaluate_prediction_rows(labels, {str(row["id"]): row for row in output_rows}, group_fields)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        rows_by_margin[margin] = output_rows
        reports_by_margin[margin] = report
        results.append(metric_row(margin, report, pred_path, report_path))

    best = max(
        results,
        key=lambda item: (
            item["canonical_exact"] or 0.0,
            item["raw_exact"] or 0.0,
            item["mean_tanimoto"] or 0.0,
            item["valid_smiles_rate"] or 0.0,
        ),
    )
    global_best_margin = float(best["margin"])
    group_margin_results = [
        build_group_margin_output(
            field,
            labels,
            rows_by_margin,
            reports_by_margin,
            global_best_margin,
            out_dir,
            group_fields,
        )
        for field in group_margin_fields
    ]
    summary = {
        "checkpoint": args.checkpoint,
        "prediction_jsonl": args.prediction_jsonl,
        "labels_jsonl": args.labels_jsonl,
        "output_dir": str(out_dir),
        "fallback_mode": fallback_mode,
        "margin_grid": parse_margin_grid(args.margin_grid),
        "best": best,
        "results": results,
        "group_margin_results": group_margin_results,
    }
    (out_dir / "margin_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(out_dir / "margin_sweep_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
