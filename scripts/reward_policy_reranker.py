#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from rerank_ocsr_candidates import (
    aggregate_candidates,
    canonicalize,
    choose_chem_light,
    get_ground_truth_smiles,
)


FEATURE_NAMES = [
    "count_log",
    "vote_share",
    "count_margin",
    "score_max",
    "score_mean",
    "score_margin",
    "is_best_count",
    "is_best_score",
    "selected_match",
    "structure_penalty",
    "heavy_atoms_log",
    "hetero_atoms_log",
    "hetero_fraction",
    "rings_log",
    "fragments_log",
    "formal_charge_abs",
    "has_dot",
    "smiles_len_log",
    "min_prompt_index",
    "min_tta_index",
    "min_generation_index",
    "is_prompt0",
    "is_orig_tta",
    "is_generation0",
    "has_stereo_marker",
    "has_atom_bracket",
    "has_boron",
    "has_halogen",
]


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


def stable_fraction(sample_id: str, seed: int):
    digest = hashlib.sha1(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def safe_float(value, default=0.0):
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def bounded_score(value):
    score = safe_float(value, -50.0)
    if score < -50.0:
        return -50.0
    if score > 10.0:
        return 10.0
    return score


def candidate_feature_dict(row: dict, aggregate: dict, aggregates: list[dict]):
    selected_canonical = canonicalize(row.get("canonical_prediction") or row.get("prediction"))
    total_votes = sum(max(0, int(item.get("count", 0))) for item in aggregates) or 1
    best_count = max((int(item.get("count", 0)) for item in aggregates), default=0)
    best_score = max((bounded_score(item.get("max_score")) for item in aggregates), default=-50.0)

    other_counts = [
        int(item.get("count", 0))
        for item in aggregates
        if item.get("canonical") != aggregate.get("canonical")
    ]
    other_scores = [
        bounded_score(item.get("max_score"))
        for item in aggregates
        if item.get("canonical") != aggregate.get("canonical")
    ]
    max_other_count = max(other_counts) if other_counts else 0
    max_other_score = max(other_scores) if other_scores else -50.0

    canonical = str(aggregate.get("canonical") or "")
    heavy_atoms = max(0.0, safe_float(aggregate.get("heavy_atoms"), 0.0))
    hetero_atoms = max(0.0, safe_float(aggregate.get("hetero_atoms"), 0.0))
    count = max(0, int(aggregate.get("count", 0)))
    max_score = bounded_score(aggregate.get("max_score"))
    mean_score = bounded_score(aggregate.get("mean_score"))
    rep = aggregate.get("representative", {})

    return {
        "count_log": math.log1p(count),
        "vote_share": count / total_votes,
        "count_margin": (count - max_other_count) / total_votes,
        "score_max": max_score,
        "score_mean": mean_score,
        "score_margin": max_score - max_other_score,
        "is_best_count": 1.0 if count == best_count else 0.0,
        "is_best_score": 1.0 if max_score == best_score else 0.0,
        "selected_match": 1.0 if selected_canonical and selected_canonical == canonical else 0.0,
        "structure_penalty": safe_float(rep.get("smiles_structure_penalty"), 0.0),
        "heavy_atoms_log": math.log1p(heavy_atoms),
        "hetero_atoms_log": math.log1p(hetero_atoms),
        "hetero_fraction": hetero_atoms / heavy_atoms if heavy_atoms else 0.0,
        "rings_log": math.log1p(max(0.0, safe_float(aggregate.get("rings"), 0.0))),
        "fragments_log": math.log1p(max(0.0, safe_float(aggregate.get("fragments"), 0.0))),
        "formal_charge_abs": max(0.0, safe_float(aggregate.get("formal_charge_abs"), 0.0)),
        "has_dot": 1.0 if aggregate.get("has_dot") else 0.0,
        "smiles_len_log": math.log1p(len(canonical)),
        "min_prompt_index": safe_float(aggregate.get("min_prompt_index"), 99.0),
        "min_tta_index": safe_float(aggregate.get("min_tta_index"), 99.0),
        "min_generation_index": safe_float(aggregate.get("min_generation_index"), 99.0),
        "is_prompt0": 1.0 if aggregate.get("min_prompt_index") == 0 else 0.0,
        "is_orig_tta": 1.0 if aggregate.get("min_tta_index") == 0 else 0.0,
        "is_generation0": 1.0 if aggregate.get("min_generation_index") == 0 else 0.0,
        "has_stereo_marker": 1.0 if ("/" in canonical or "\\" in canonical or "@" in canonical) else 0.0,
        "has_atom_bracket": 1.0 if "[" in canonical else 0.0,
        "has_boron": 1.0 if "B" in canonical else 0.0,
        "has_halogen": 1.0 if ("Cl" in canonical or "Br" in canonical or "F" in canonical or "I" in canonical) else 0.0,
    }


def vectorize(feature_dict: dict, normalizer: dict):
    values = []
    means = normalizer.get("mean", {})
    stds = normalizer.get("std", {})
    for name in FEATURE_NAMES:
        value = safe_float(feature_dict.get(name), 0.0)
        values.append((value - means.get(name, 0.0)) / stds.get(name, 1.0))
    return values


def build_normalizer(samples):
    values = {name: [] for name in FEATURE_NAMES}
    for sample in samples:
        for features in sample["features"]:
            for name in FEATURE_NAMES:
                values[name].append(safe_float(features.get(name), 0.0))

    mean = {}
    std = {}
    for name, nums in values.items():
        if not nums:
            mean[name] = 0.0
            std[name] = 1.0
            continue
        avg = sum(nums) / len(nums)
        var = sum((item - avg) ** 2 for item in nums) / max(1, len(nums) - 1)
        scale = math.sqrt(var)
        mean[name] = avg
        std[name] = scale if scale > 1e-8 else 1.0
    return {"mean": mean, "std": std}


def dot(weights, vector):
    return sum(weight * value for weight, value in zip(weights, vector))


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def load_labeled_samples(prediction_rows: dict, benchmark: dict):
    samples = []
    for sample_id, row in prediction_rows.items():
        label_row = benchmark.get(sample_id)
        if not label_row:
            continue
        target = canonicalize(get_ground_truth_smiles(label_row))
        aggregates = aggregate_candidates(row.get("candidates", []))
        if not target or not aggregates:
            continue
        features = [candidate_feature_dict(row, item, aggregates) for item in aggregates]
        positive_indices = [
            index
            for index, item in enumerate(aggregates)
            if item.get("canonical") == target
        ]
        samples.append(
            {
                "id": sample_id,
                "row": row,
                "label": label_row,
                "target": target,
                "aggregates": aggregates,
                "features": features,
                "positive_indices": positive_indices,
            }
        )
    return samples


def split_train_dev(samples, train_fraction: float, seed: int):
    trainable = [
        sample
        for sample in samples
        if sample["positive_indices"] and len(sample["aggregates"]) > 1
    ]
    if train_fraction >= 1.0:
        return trainable, []

    train = []
    dev = []
    for sample in trainable:
        if stable_fraction(sample["id"], seed) < train_fraction:
            train.append(sample)
        else:
            dev.append(sample)

    if not train and trainable:
        train = trainable[:1]
        dev = trainable[1:]
    if not dev and len(trainable) > 1:
        dev = train[-max(1, len(trainable) // 5):]
        train = train[:-len(dev)] or trainable[:1]
    return train, dev


def build_training_pairs(samples, normalizer, max_negatives_per_positive: int):
    pairs = []
    for sample in samples:
        vectors = [vectorize(features, normalizer) for features in sample["features"]]
        positive_indices = set(sample["positive_indices"])
        negative_indices = [
            index
            for index in range(len(sample["aggregates"]))
            if index not in positive_indices
        ]
        negative_indices.sort(
            key=lambda index: (
                sample["features"][index].get("selected_match", 0.0),
                sample["aggregates"][index].get("count", 0),
                bounded_score(sample["aggregates"][index].get("max_score")),
            ),
            reverse=True,
        )
        if max_negatives_per_positive > 0:
            negative_indices = negative_indices[:max_negatives_per_positive]
        for positive_index in positive_indices:
            for negative_index in negative_indices:
                diff = [
                    pos_value - neg_value
                    for pos_value, neg_value in zip(vectors[positive_index], vectors[negative_index])
                ]
                pairs.append(diff)
    return pairs


def fit_pairwise_policy(samples, normalizer, args):
    rng = random.Random(args.seed)
    pairs = build_training_pairs(samples, normalizer, args.max_negatives_per_positive)
    weights = [0.0 for _ in FEATURE_NAMES]
    if not pairs:
        return weights, {"pair_count": 0, "final_pair_loss": None, "final_pair_accuracy": None}

    final_loss = 0.0
    for epoch in range(args.epochs):
        rng.shuffle(pairs)
        eta = args.lr / math.sqrt(1.0 + epoch * args.lr_decay)
        loss_sum = 0.0
        correct = 0
        for diff in pairs:
            margin = dot(weights, diff)
            prob = sigmoid(margin)
            loss_sum += -math.log(max(prob, 1e-12))
            correct += int(margin > 0.0)
            for index, value in enumerate(diff):
                weights[index] *= 1.0 - eta * args.l2
                weights[index] += eta * (1.0 - prob) * value
        final_loss = loss_sum / len(pairs)
        final_accuracy = correct / len(pairs)

    return weights, {
        "pair_count": len(pairs),
        "final_pair_loss": final_loss,
        "final_pair_accuracy": final_accuracy,
    }


def choose_policy_candidate(row: dict, aggregates: list[dict], weights: list[float], normalizer: dict):
    scored = []
    for aggregate in aggregates:
        features = candidate_feature_dict(row, aggregate, aggregates)
        vector = vectorize(features, normalizer)
        score = dot(weights, vector)
        scored.append((score, aggregate, features))
    score, aggregate, features = max(
        scored,
        key=lambda item: (
            item[0],
            item[1].get("count", 0),
            bounded_score(item[1].get("max_score")),
            -safe_float(item[1].get("min_prompt_index"), 99.0),
            -safe_float(item[1].get("min_generation_index"), 99.0),
        ),
    )
    return aggregate, score, features


def score_aggregate(row: dict, aggregate: dict, aggregates: list[dict], weights: list[float], normalizer: dict):
    features = candidate_feature_dict(row, aggregate, aggregates)
    return dot(weights, vectorize(features, normalizer)), features


def choose_selected_candidate(row: dict, aggregates: list[dict]):
    selected = canonicalize(row.get("canonical_prediction") or row.get("prediction"))
    if selected:
        for aggregate in aggregates:
            if aggregate.get("canonical") == selected:
                return aggregate
    if aggregates:
        return max(
            aggregates,
            key=lambda item: (
                item.get("count", 0),
                bounded_score(item.get("max_score")),
            ),
        )
    return None


def choose_fallback_candidate(row: dict, aggregates: list[dict], mode: str):
    if mode == "selected":
        return choose_selected_candidate(row, aggregates)
    if mode == "chem_light":
        return choose_chem_light(
            aggregates,
            salt_bonus=2.0,
            heavy_penalty=0.02,
            stereo_min_ratio=0.25,
        )
    if mode == "none":
        return None
    raise ValueError(f"unknown fallback mode: {mode}")


def choose_final_candidate(row: dict, aggregates: list[dict], weights: list[float], normalizer: dict, fallback_mode: str, policy_margin: float):
    policy_aggregate, policy_score, policy_features = choose_policy_candidate(row, aggregates, weights, normalizer)
    fallback = choose_fallback_candidate(row, aggregates, fallback_mode)
    if fallback is None:
        return policy_aggregate, policy_score, policy_features, "reward_policy_rerank"

    fallback_score, fallback_features = score_aggregate(row, fallback, aggregates, weights, normalizer)
    if policy_aggregate.get("canonical") == fallback.get("canonical"):
        return fallback, fallback_score, fallback_features, f"reward_policy_{fallback_mode}_agreement"
    if policy_score - fallback_score >= policy_margin:
        return policy_aggregate, policy_score, policy_features, f"reward_policy_override_{fallback_mode}"
    return fallback, fallback_score, fallback_features, f"reward_policy_fallback_{fallback_mode}"


def build_output_row(row: dict, aggregate: dict, score: float, features: dict, keep_candidates: bool):
    rep = aggregate.get("representative", {})
    out = dict(row)
    out["prediction"] = rep.get("prediction") or aggregate.get("canonical")
    out["canonical_prediction"] = aggregate.get("canonical")
    out["generation_score"] = rep.get("generation_score")
    out["smiles_structure_penalty"] = rep.get("smiles_structure_penalty")
    out["raw_text"] = rep.get("raw_text", "")
    out["prompt"] = rep.get("prompt", row.get("prompt", ""))
    out["selection_reason"] = "reward_policy_rerank"
    out["vote_count"] = aggregate.get("count")
    out["reward_policy_debug"] = {
        "policy_score": score,
        "unique_valid_candidates": None,
        "max_score": aggregate.get("max_score"),
        "mean_score": aggregate.get("mean_score"),
        "heavy_atoms": aggregate.get("heavy_atoms"),
        "fragments": aggregate.get("fragments"),
        "selected_match": features.get("selected_match"),
    }
    if not keep_candidates:
        out.pop("candidates", None)
    return out


def init_metric():
    return {
        "total": 0,
        "selected_exact": 0,
        "chem_light_exact": 0,
        "policy_exact": 0,
        "oracle_exact": 0,
        "selected_valid": 0,
        "policy_valid": 0,
        "policy_changed": 0,
        "policy_good_changes": 0,
        "policy_bad_changes": 0,
        "policy_changed_from_chem_light": 0,
        "policy_good_changes_from_chem_light": 0,
        "policy_bad_changes_from_chem_light": 0,
    }


def add_metric(acc, target, selected, chem_light, policy, oracle):
    acc["total"] += 1
    acc["selected_exact"] += int(bool(target and selected == target))
    acc["chem_light_exact"] += int(bool(target and chem_light == target))
    acc["policy_exact"] += int(bool(target and policy == target))
    acc["oracle_exact"] += int(oracle)
    acc["selected_valid"] += int(bool(selected))
    acc["policy_valid"] += int(bool(policy))
    if selected != policy:
        acc["policy_changed"] += 1
        acc["policy_good_changes"] += int(bool(target and policy == target and selected != target))
        acc["policy_bad_changes"] += int(bool(target and selected == target and policy != target))
    if chem_light != policy:
        acc["policy_changed_from_chem_light"] += 1
        acc["policy_good_changes_from_chem_light"] += int(bool(target and policy == target and chem_light != target))
        acc["policy_bad_changes_from_chem_light"] += int(bool(target and chem_light == target and policy != target))


def finalize_metric(acc):
    total = acc["total"]
    if total == 0:
        return {
            "total": 0,
            "selected_exact": 0.0,
            "chem_light_exact": 0.0,
            "policy_exact": 0.0,
            "oracle_exact": 0.0,
            "selected_valid_rate": 0.0,
            "policy_valid_rate": 0.0,
        "policy_changed": 0,
        "policy_good_changes": 0,
        "policy_bad_changes": 0,
        "policy_changed_from_chem_light": 0,
        "policy_good_changes_from_chem_light": 0,
        "policy_bad_changes_from_chem_light": 0,
        "policy_gain_over_selected": 0.0,
        "policy_gain_over_chem_light": 0.0,
    }
    return {
        "total": total,
        "selected_exact": acc["selected_exact"] / total,
        "chem_light_exact": acc["chem_light_exact"] / total,
        "policy_exact": acc["policy_exact"] / total,
        "oracle_exact": acc["oracle_exact"] / total,
        "selected_valid_rate": acc["selected_valid"] / total,
        "policy_valid_rate": acc["policy_valid"] / total,
        "policy_changed": acc["policy_changed"],
        "policy_good_changes": acc["policy_good_changes"],
        "policy_bad_changes": acc["policy_bad_changes"],
        "policy_changed_from_chem_light": acc["policy_changed_from_chem_light"],
        "policy_good_changes_from_chem_light": acc["policy_good_changes_from_chem_light"],
        "policy_bad_changes_from_chem_light": acc["policy_bad_changes_from_chem_light"],
        "policy_gain_over_selected": (acc["policy_exact"] - acc["selected_exact"]) / total,
        "policy_gain_over_chem_light": (acc["policy_exact"] - acc["chem_light_exact"]) / total,
    }


def evaluate_policy(samples, weights, normalizer, split_ids=None, fallback_mode="none", policy_margin=0.0):
    split_ids = set(split_ids) if split_ids is not None else None
    overall = init_metric()
    by_source = defaultdict(init_metric)
    by_task_type = defaultdict(init_metric)
    detail_rows = []

    for sample in samples:
        if split_ids is not None and sample["id"] not in split_ids:
            continue
        row = sample["row"]
        target = sample["target"]
        aggregates = sample["aggregates"]
        selected_aggregate = choose_selected_candidate(row, aggregates)
        selected = selected_aggregate.get("canonical") if selected_aggregate else canonicalize(row.get("prediction"))
        chem_light = None
        if aggregates:
            chem_light = choose_chem_light(
                aggregates,
                salt_bonus=2.0,
                heavy_penalty=0.02,
                stereo_min_ratio=0.25,
            ).get("canonical")
        policy_aggregate, policy_score, _, _ = choose_final_candidate(
            row,
            aggregates,
            weights,
            normalizer,
            fallback_mode,
            policy_margin,
        )
        policy = policy_aggregate.get("canonical")
        oracle = bool(target and any(item.get("canonical") == target for item in aggregates))

        add_metric(overall, target, selected, chem_light, policy, oracle)
        source = str(sample["label"].get("source", "unknown"))
        task_type = str(sample["label"].get("task_type", "unknown"))
        add_metric(by_source[source], target, selected, chem_light, policy, oracle)
        add_metric(by_task_type[task_type], target, selected, chem_light, policy, oracle)

        detail_rows.append(
            {
                "id": sample["id"],
                "source": source,
                "task_type": task_type,
                "target": target,
                "selected": selected,
                "chem_light": chem_light,
                "policy": policy,
                "policy_score": policy_score,
                "oracle": oracle,
                "selected_correct": bool(target and selected == target),
                "chem_light_correct": bool(target and chem_light == target),
                "policy_correct": bool(target and policy == target),
            }
        )

    return {
        "metrics": finalize_metric(overall),
        "by_source": {key: finalize_metric(value) for key, value in sorted(by_source.items())},
        "by_task_type": {key: finalize_metric(value) for key, value in sorted(by_task_type.items())},
        "details": detail_rows,
    }


def load_policy(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    weights_by_name = data.get("weights", {})
    weights = [safe_float(weights_by_name.get(name), 0.0) for name in FEATURE_NAMES]
    return weights, data.get("normalizer", {"mean": {}, "std": {}}), data


def write_policy(path: Path, weights, normalizer, train_info, report):
    weights_by_name = {
        name: weights[index]
        for index, name in enumerate(FEATURE_NAMES)
    }
    sorted_weights = sorted(weights_by_name.items(), key=lambda item: abs(item[1]), reverse=True)
    data = {
        "policy_type": "pairwise_logistic_candidate_reward_policy",
        "feature_names": FEATURE_NAMES,
        "weights": weights_by_name,
        "top_abs_weights": sorted_weights[:20],
        "normalizer": normalizer,
        "train_info": train_info,
        "report": report,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--labels-jsonl", default="")
    parser.add_argument("--policy-json", default="")
    parser.add_argument("--load-policy-json", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--details-jsonl", default="")
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--lr-decay", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--max-negatives-per-positive", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument(
        "--fallback-mode",
        choices=["none", "selected", "chem_light"],
        default="none",
        help="Optional conservative fallback. With chem_light, policy only overrides the current hand-built reranker when its score margin is large enough.",
    )
    parser.add_argument("--policy-margin", type=float, default=0.0)
    parser.add_argument("--keep-candidates", action="store_true")
    args = parser.parse_args()

    prediction_rows = {
        str(row["id"]): row
        for row in read_jsonl(Path(args.prediction_jsonl))
    }
    benchmark = {}
    if args.labels_jsonl:
        benchmark = {
            str(row["id"]): row
            for row in read_jsonl(Path(args.labels_jsonl))
        }

    if args.load_policy_json:
        weights, normalizer, loaded = load_policy(Path(args.load_policy_json))
        loaded_train_info = loaded.get("train_info") or {}
        train_info = {"loaded_policy_json": args.load_policy_json, "loaded_train_info": loaded_train_info}
        samples = load_labeled_samples(prediction_rows, benchmark) if benchmark else []
    else:
        if not benchmark:
            raise SystemExit("--labels-jsonl is required when training a policy")
        samples = load_labeled_samples(prediction_rows, benchmark)
        train_samples, dev_samples = split_train_dev(samples, args.train_fraction, args.seed)
        normalizer = build_normalizer(train_samples)
        weights, fit_info = fit_pairwise_policy(train_samples, normalizer, args)
        train_info = {
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "trainable_samples": len([sample for sample in samples if sample["positive_indices"]]),
            "train_samples": len(train_samples),
            "dev_samples": len(dev_samples),
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_decay": args.lr_decay,
            "l2": args.l2,
            "max_negatives_per_positive": args.max_negatives_per_positive,
            **fit_info,
        }

    output_rows = []
    for row in prediction_rows.values():
        aggregates = aggregate_candidates(row.get("candidates", []))
        if not aggregates:
            out = dict(row)
            out["selection_reason"] = "reward_policy_no_valid_candidate"
            if not args.keep_candidates:
                out.pop("candidates", None)
            output_rows.append(out)
            continue
        chosen, score, features, selection_reason = choose_final_candidate(
            row,
            aggregates,
            weights,
            normalizer,
            args.fallback_mode,
            args.policy_margin,
        )
        out = build_output_row(row, chosen, score, features, args.keep_candidates)
        out["selection_reason"] = selection_reason
        out["reward_policy_debug"]["unique_valid_candidates"] = len(aggregates)
        output_rows.append(out)
    write_jsonl(Path(args.output_jsonl), output_rows)

    report = {
        "prediction_jsonl": args.prediction_jsonl,
        "output_jsonl": args.output_jsonl,
        "train_info": train_info,
        "fallback_mode": args.fallback_mode,
        "policy_margin": args.policy_margin,
    }
    details = []
    if benchmark:
        all_eval = evaluate_policy(
            samples,
            weights,
            normalizer,
            fallback_mode=args.fallback_mode,
            policy_margin=args.policy_margin,
        )
        report["all"] = {key: value for key, value in all_eval.items() if key != "details"}
        details = all_eval["details"]

        split_fraction = args.train_fraction
        split_seed = args.seed
        if args.load_policy_json and isinstance(train_info.get("loaded_train_info"), dict):
            split_fraction = float(train_info["loaded_train_info"].get("train_fraction", split_fraction))
            split_seed = int(train_info["loaded_train_info"].get("seed", split_seed))

        if split_fraction < 1.0:
            train_ids = [
                sample["id"]
                for sample in samples
                if sample["positive_indices"] and stable_fraction(sample["id"], split_seed) < split_fraction
            ]
            dev_ids = [
                sample["id"]
                for sample in samples
                if sample["positive_indices"] and sample["id"] not in set(train_ids)
            ]
            if train_ids:
                report["train_oracle_subset"] = {
                    key: value
                    for key, value in evaluate_policy(
                        samples,
                        weights,
                        normalizer,
                        train_ids,
                        fallback_mode=args.fallback_mode,
                        policy_margin=args.policy_margin,
                    ).items()
                    if key != "details"
                }
            if dev_ids:
                report["dev_oracle_subset"] = {
                    key: value
                    for key, value in evaluate_policy(
                        samples,
                        weights,
                        normalizer,
                        dev_ids,
                        fallback_mode=args.fallback_mode,
                        policy_margin=args.policy_margin,
                    ).items()
                    if key != "details"
                }

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.details_jsonl and details:
        write_jsonl(Path(args.details_jsonl), details)

    if args.policy_json and not args.load_policy_json:
        write_policy(Path(args.policy_json), weights, normalizer, train_info, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
