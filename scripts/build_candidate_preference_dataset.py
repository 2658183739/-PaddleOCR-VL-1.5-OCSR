#!/usr/bin/env python3
import argparse
import json
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
from reward_policy_reranker import (  # noqa: E402
    choose_final_candidate,
    choose_policy_candidate,
    load_policy,
    score_aggregate,
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def find_aggregate(aggregates, canonical):
    if not canonical:
        return None
    for aggregate in aggregates:
        if aggregate.get("canonical") == canonical:
            return aggregate
    return None


def aggregate_brief(aggregate, policy_score=None):
    if aggregate is None:
        return None
    rep = aggregate.get("representative", {})
    return {
        "smiles": rep.get("prediction") or aggregate.get("canonical"),
        "canonical": aggregate.get("canonical"),
        "count": aggregate.get("count"),
        "max_score": aggregate.get("max_score"),
        "mean_score": aggregate.get("mean_score"),
        "heavy_atoms": aggregate.get("heavy_atoms"),
        "rings": aggregate.get("rings"),
        "fragments": aggregate.get("fragments"),
        "prompt": rep.get("prompt", ""),
        "raw_text": rep.get("raw_text", ""),
        "policy_score": policy_score,
    }


def rank_key(row, aggregate, aggregates, weights=None, normalizer=None):
    policy_score = None
    if weights is not None and normalizer is not None:
        policy_score, _ = score_aggregate(row, aggregate, aggregates, weights, normalizer)
    return (
        policy_score if policy_score is not None else -1_000_000.0,
        aggregate.get("count", 0),
        safe_float(aggregate.get("max_score"), -1_000_000.0),
        -safe_float(aggregate.get("min_prompt_index"), 99.0),
        -safe_float(aggregate.get("min_generation_index"), 99.0),
    )


def add_pair(
    pairs,
    seen,
    label,
    prediction,
    target_canonical,
    chosen,
    rejected,
    pair_type,
    weights=None,
    normalizer=None,
    aggregates=None,
):
    if chosen is None or rejected is None:
        return False
    chosen_canonical = chosen.get("canonical")
    rejected_canonical = rejected.get("canonical")
    if not chosen_canonical or not rejected_canonical or chosen_canonical == rejected_canonical:
        return False

    sample_id = prediction["id"]
    key = (sample_id, chosen_canonical, rejected_canonical)
    if key in seen:
        return False
    seen.add(key)

    chosen_policy_score = None
    rejected_policy_score = None
    if weights is not None and normalizer is not None and aggregates is not None:
        chosen_policy_score, _ = score_aggregate(prediction, chosen, aggregates, weights, normalizer)
        rejected_policy_score, _ = score_aggregate(prediction, rejected, aggregates, weights, normalizer)

    row = {
        "id": f"{sample_id}::{len(pairs):06d}",
        "sample_id": sample_id,
        "eval_panel": prediction.get("eval_panel") or label.get("eval_panel", ""),
        "source": label.get("source", prediction.get("source", "")),
        "task_type": label.get("task_type", prediction.get("task_type", "")),
        "difficulty": label.get("difficulty", prediction.get("difficulty", "")),
        "image": label.get("image") or label.get("image_path") or prediction.get("image_path", ""),
        "pair_type": pair_type,
        "ground_truth": get_ground_truth_smiles(label),
        "ground_truth_canonical": target_canonical,
        "prompt": prediction.get("prompt", ""),
        "chosen": aggregate_brief(chosen, chosen_policy_score),
        "rejected": aggregate_brief(rejected, rejected_policy_score),
    }
    pairs.append(row)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", default="")
    parser.add_argument("--reward-policy-json", default="")
    parser.add_argument("--max-hard-negatives-per-sample", type=int, default=3)
    parser.add_argument("--policy-margin", type=float, default=1.5)
    args = parser.parse_args()

    labels = {str(row["id"]): row for row in read_jsonl(Path(args.labels_jsonl))}
    predictions = {str(row["id"]): row for row in read_jsonl(Path(args.prediction_jsonl))}

    weights = None
    normalizer = None
    if args.reward_policy_json:
        weights, normalizer, _ = load_policy(Path(args.reward_policy_json))

    pairs = []
    seen = set()
    counts = Counter()
    pair_counts = Counter()
    by_panel = defaultdict(Counter)
    by_source = defaultdict(Counter)

    for sample_id, label in labels.items():
        prediction = predictions.get(sample_id)
        counts["total"] += 1
        if prediction is None:
            counts["missing_prediction"] += 1
            continue

        target_canonical = canonicalize(get_ground_truth_smiles(label))
        if not target_canonical:
            counts["missing_target_canonical"] += 1
            continue

        aggregates = aggregate_candidates(prediction.get("candidates", []))
        if not aggregates:
            counts["no_valid_candidates"] += 1
            continue
        counts["with_valid_candidates"] += 1

        positive = find_aggregate(aggregates, target_canonical)
        if positive is None:
            counts["oracle_positive_missing"] += 1
            continue
        counts["oracle_positive_present"] += 1

        selected_canonical = canonicalize(prediction.get("canonical_prediction") or prediction.get("prediction"))
        selected = find_aggregate(aggregates, selected_canonical)
        chem_light = choose_chem_light(
            aggregates,
            salt_bonus=2.0,
            heavy_penalty=0.02,
            stereo_min_ratio=0.25,
        )
        policy = None
        if weights is not None and normalizer is not None:
            policy, _, _, _ = choose_final_candidate(
                prediction,
                aggregates,
                weights,
                normalizer,
                fallback_mode="chem_light",
                policy_margin=args.policy_margin,
            )

        negatives = [item for item in aggregates if item.get("canonical") != target_canonical]
        ranked_negatives = sorted(
            negatives,
            key=lambda item: rank_key(prediction, item, aggregates, weights, normalizer),
            reverse=True,
        )

        sample_pair_start = len(pairs)
        if selected is not None and selected.get("canonical") != target_canonical:
            add_pair(
                pairs,
                seen,
                label,
                prediction,
                target_canonical,
                positive,
                selected,
                "oracle_positive_vs_selected",
                weights,
                normalizer,
                aggregates,
            )
        if chem_light is not None and chem_light.get("canonical") != target_canonical:
            add_pair(
                pairs,
                seen,
                label,
                prediction,
                target_canonical,
                positive,
                chem_light,
                "oracle_positive_vs_chem_light",
                weights,
                normalizer,
                aggregates,
            )
        if policy is not None and policy.get("canonical") != target_canonical:
            add_pair(
                pairs,
                seen,
                label,
                prediction,
                target_canonical,
                positive,
                policy,
                "oracle_positive_vs_reward_policy",
                weights,
                normalizer,
                aggregates,
            )

        for negative in ranked_negatives[: args.max_hard_negatives_per_sample]:
            add_pair(
                pairs,
                seen,
                label,
                prediction,
                target_canonical,
                positive,
                negative,
                "oracle_positive_vs_hard_negative",
                weights,
                normalizer,
                aggregates,
            )

        if selected is not None and selected.get("canonical") == target_canonical:
            if chem_light is not None and chem_light.get("canonical") != target_canonical:
                add_pair(
                    pairs,
                    seen,
                    label,
                    prediction,
                    target_canonical,
                    selected,
                    chem_light,
                    "selected_correct_guard_vs_chem_light",
                    weights,
                    normalizer,
                    aggregates,
                )
            if policy is not None and policy.get("canonical") != target_canonical:
                add_pair(
                    pairs,
                    seen,
                    label,
                    prediction,
                    target_canonical,
                    selected,
                    policy,
                    "selected_correct_guard_vs_reward_policy",
                    weights,
                    normalizer,
                    aggregates,
                )

        if len(pairs) == sample_pair_start:
            counts["oracle_positive_no_pair"] += 1
        else:
            new_pairs = pairs[sample_pair_start:]
            counts["samples_with_pairs"] += 1
            panel = prediction.get("eval_panel") or label.get("eval_panel", "")
            source = label.get("source", prediction.get("source", ""))
            for pair in new_pairs:
                pair_type = pair["pair_type"]
                pair_counts[pair_type] += 1
                by_panel[panel][pair_type] += 1
                by_source[source][pair_type] += 1

    write_jsonl(Path(args.output_jsonl), pairs)
    report = {
        "prediction_jsonl": args.prediction_jsonl,
        "labels_jsonl": args.labels_jsonl,
        "output_jsonl": args.output_jsonl,
        "reward_policy_json": args.reward_policy_json,
        "policy_margin": args.policy_margin,
        "max_hard_negatives_per_sample": args.max_hard_negatives_per_sample,
        "counts": dict(counts),
        "pair_counts": dict(pair_counts),
        "by_panel_pair_counts": {key: dict(value) for key, value in by_panel.items()},
        "by_source_pair_counts": {key: dict(value) for key, value in by_source.items()},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report_json:
        path = Path(args.report_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
