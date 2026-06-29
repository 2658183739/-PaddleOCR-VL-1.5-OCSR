#!/usr/bin/env python3
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rerank_ocsr_candidates import aggregate_candidates, canonicalize  # noqa: E402
from reward_policy_reranker import (  # noqa: E402
    FEATURE_NAMES,
    build_normalizer,
    candidate_feature_dict,
    load_labeled_samples,
    stable_fraction,
    vectorize,
)
from train_candidate_reward_head import (  # noqa: E402
    RewardHead,
    build_output_rows,
    evaluate_samples,
    parse_margin_grid,
    read_jsonl,
    write_jsonl,
)


DEFAULT_PAIR_TYPE_WEIGHTS = {
    "oracle_positive_vs_hard_negative": 1.0,
    "oracle_positive_vs_selected": 2.0,
    "oracle_positive_vs_chem_light": 1.5,
    "oracle_positive_vs_reward_policy": 2.0,
    "selected_correct_guard_vs_chem_light": 3.0,
    "selected_correct_guard_vs_reward_policy": 3.0,
}


def parse_pair_type_weights(text: str):
    weights = dict(DEFAULT_PAIR_TYPE_WEIGHTS)
    if not text:
        return weights, 1.0

    default_weight = 1.0
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"bad pair weight item: {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        value = float(value.strip())
        if name == "default":
            default_weight = value
        else:
            weights[name] = value
    return weights, default_weight


def canonical_from_pair_side(side: dict):
    if not side:
        return None
    return canonicalize(side.get("canonical") or side.get("smiles"))


def build_candidate_cache(prediction_rows: dict):
    cache = {}
    for sample_id, row in prediction_rows.items():
        aggregates = aggregate_candidates(row.get("candidates", []))
        features = [candidate_feature_dict(row, aggregate, aggregates) for aggregate in aggregates]
        by_canonical = {}
        for index, aggregate in enumerate(aggregates):
            canonical = aggregate.get("canonical")
            if canonical and canonical not in by_canonical:
                by_canonical[canonical] = index
        cache[sample_id] = {
            "row": row,
            "aggregates": aggregates,
            "features": features,
            "by_canonical": by_canonical,
        }
    return cache


def load_pair_examples(pair_jsonl: Path, prediction_rows: dict, pair_weights: dict, default_weight: float):
    cache = build_candidate_cache(prediction_rows)
    examples = []
    counts = Counter()
    pair_counts = Counter()
    by_source = defaultdict(Counter)
    by_panel = defaultdict(Counter)

    for pair in read_jsonl(pair_jsonl):
        counts["total_pairs"] += 1
        sample_id = str(pair.get("sample_id", ""))
        cached = cache.get(sample_id)
        if cached is None:
            counts["missing_prediction"] += 1
            continue
        if not cached["aggregates"]:
            counts["no_valid_candidates"] += 1
            continue

        chosen_canonical = canonical_from_pair_side(pair.get("chosen", {}))
        rejected_canonical = canonical_from_pair_side(pair.get("rejected", {}))
        if not chosen_canonical or not rejected_canonical:
            counts["missing_pair_canonical"] += 1
            continue
        chosen_index = cached["by_canonical"].get(chosen_canonical)
        rejected_index = cached["by_canonical"].get(rejected_canonical)
        if chosen_index is None:
            counts["chosen_not_in_candidates"] += 1
            continue
        if rejected_index is None:
            counts["rejected_not_in_candidates"] += 1
            continue
        if chosen_index == rejected_index:
            counts["same_candidate_pair"] += 1
            continue

        pair_type = str(pair.get("pair_type", "unknown"))
        weight = float(pair_weights.get(pair_type, default_weight))
        if weight <= 0.0:
            counts["zero_weight_pairs"] += 1
            continue

        example = {
            "id": str(pair.get("id") or f"{sample_id}::{counts['accepted_pairs']:06d}"),
            "sample_id": sample_id,
            "pair_type": pair_type,
            "source": str(pair.get("source", "")),
            "eval_panel": str(pair.get("eval_panel", "")),
            "weight": weight,
            "chosen_features": cached["features"][chosen_index],
            "rejected_features": cached["features"][rejected_index],
        }
        examples.append(example)
        counts["accepted_pairs"] += 1
        pair_counts[pair_type] += 1
        by_source[example["source"]][pair_type] += 1
        by_panel[example["eval_panel"]][pair_type] += 1

    return examples, cache, {
        "counts": dict(counts),
        "pair_counts": dict(pair_counts),
        "by_source_pair_counts": {key: dict(value) for key, value in sorted(by_source.items())},
        "by_panel_pair_counts": {key: dict(value) for key, value in sorted(by_panel.items())},
    }


def split_pair_examples(examples, train_fraction: float, seed: int):
    if train_fraction >= 1.0:
        return list(examples), []

    train = []
    dev = []
    for example in examples:
        if stable_fraction(example["sample_id"], seed) < train_fraction:
            train.append(example)
        else:
            dev.append(example)

    if not train and examples:
        train = examples[:1]
        dev = examples[1:]
    if not dev and len(examples) > 1:
        holdout = max(1, len(examples) // 5)
        dev = train[-holdout:]
        train = train[:-holdout] or examples[:1]
    return train, dev


def normalizer_samples_from_pair_examples(examples, cache):
    sample_ids = sorted({example["sample_id"] for example in examples})
    samples = []
    for sample_id in sample_ids:
        cached = cache.get(sample_id)
        if cached and cached["features"]:
            samples.append({"features": cached["features"]})
    if samples:
        return samples
    return [
        {"features": [example["chosen_features"], example["rejected_features"]]}
        for example in examples
    ]


def build_pair_tensors(examples, normalizer):
    if not examples:
        empty = torch.empty((0, len(FEATURE_NAMES)), dtype=torch.float32)
        return empty, empty, torch.empty((0,), dtype=torch.float32)
    chosen = [vectorize(example["chosen_features"], normalizer) for example in examples]
    rejected = [vectorize(example["rejected_features"], normalizer) for example in examples]
    weights = [float(example["weight"]) for example in examples]
    return (
        torch.tensor(chosen, dtype=torch.float32),
        torch.tensor(rejected, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


def evaluate_pair_tensors(model: nn.Module, pos, neg, weights):
    if pos.numel() == 0:
        return {"pair_count": 0, "pair_loss": None, "pair_accuracy": None, "weighted_pair_accuracy": None}
    with torch.no_grad():
        margin = model(pos) - model(neg)
        losses = F.softplus(-margin)
        loss = float((losses * weights).sum().item() / max(1e-8, float(weights.sum().item())))
        correct = (margin > 0).float()
        weighted_acc = float((correct * weights).sum().item() / max(1e-8, float(weights.sum().item())))
        acc = float(correct.mean().item())
    return {
        "pair_count": int(pos.shape[0]),
        "pair_loss": loss,
        "pair_accuracy": acc,
        "weighted_pair_accuracy": weighted_acc,
    }


def train_model(args, train_examples, dev_examples, normalizer):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = RewardHead(len(FEATURE_NAMES), args.hidden_dim, args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_pos, train_neg, train_weights = build_pair_tensors(train_examples, normalizer)
    dev_pos, dev_neg, dev_weights = build_pair_tensors(dev_examples, normalizer)
    if train_pos.numel() == 0:
        raise SystemExit("no train pairs")

    best_state = None
    best_key = None
    history = []
    rng = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(train_pos.shape[0], generator=rng)
        loss_sum = 0.0
        weight_sum = 0.0
        correct_sum = 0.0
        total = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            pos = train_pos[idx]
            neg = train_neg[idx]
            weight = train_weights[idx]
            optimizer.zero_grad(set_to_none=True)
            margin = model(pos) - model(neg)
            loss_items = F.softplus(-margin)
            loss = (loss_items * weight).sum() / weight.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()

            batch_weight = float(weight.sum().item())
            loss_sum += float(loss.item()) * batch_weight
            weight_sum += batch_weight
            correct_sum += float(((margin.detach() > 0).float() * weight).sum().item())
            total += len(idx)

        model.eval()
        train_loss = loss_sum / max(1e-8, weight_sum)
        train_weighted_acc = correct_sum / max(1e-8, weight_sum)
        with torch.no_grad():
            train_acc = float(((model(train_pos) - model(train_neg)) > 0).float().mean().item())
        train_pair_eval = {
            "pair_loss": train_loss,
            "pair_accuracy": train_acc,
            "weighted_pair_accuracy": train_weighted_acc,
        }
        dev_pair_eval = evaluate_pair_tensors(model, dev_pos, dev_neg, dev_weights)
        dev_acc = dev_pair_eval.get("weighted_pair_accuracy")
        dev_loss = dev_pair_eval.get("pair_loss")
        key = (
            dev_acc if dev_acc is not None else train_weighted_acc,
            -(dev_loss if dev_loss is not None else train_loss),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            history.append(
                {
                    "epoch": epoch,
                    "train_pair_loss": train_pair_eval["pair_loss"],
                    "train_pair_accuracy": train_pair_eval["pair_accuracy"],
                    "train_weighted_pair_accuracy": train_pair_eval["weighted_pair_accuracy"],
                    "dev_pair_loss": dev_pair_eval.get("pair_loss"),
                    "dev_pair_accuracy": dev_pair_eval.get("pair_accuracy"),
                    "dev_weighted_pair_accuracy": dev_pair_eval.get("weighted_pair_accuracy"),
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, {
        "train_pair_count": int(train_pos.shape[0]),
        "dev_pair_count": int(dev_pos.shape[0]),
        "train_pair_eval": evaluate_pair_tensors(model, train_pos, train_neg, train_weights),
        "dev_pair_eval": evaluate_pair_tensors(model, dev_pos, dev_neg, dev_weights),
    }


def choose_best_margin(model, normalizer, samples, fallback_mode: str, margins, dev_sample_ids):
    if not samples:
        return 0.0, []
    eval_ids = dev_sample_ids or None
    margin_reports = []
    for margin in margins:
        report = evaluate_samples(model, normalizer, samples, fallback_mode, margin, eval_ids)
        metric = report["metrics"]
        margin_reports.append(
            {
                "margin": margin,
                "dev_policy_exact": metric["policy_exact"],
                "dev_gain_over_selected": metric.get("policy_gain_over_selected", 0.0),
                "dev_gain_over_chem_light": metric.get("policy_gain_over_chem_light", 0.0),
                "dev_policy_bad_changes": metric.get("policy_bad_changes", 0),
                "dev_policy_good_changes": metric.get("policy_good_changes", 0),
            }
        )
    best_margin = max(
        margin_reports,
        key=lambda item: (
            item["dev_policy_exact"],
            item["dev_gain_over_selected"],
            -item["dev_policy_bad_changes"],
            item["dev_gain_over_chem_light"],
        ),
    )["margin"]
    return best_margin, margin_reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-jsonl", required=True)
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--labels-jsonl", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--fallback-mode", choices=["none", "selected", "chem_light"], default="chem_light")
    parser.add_argument("--margin-grid", default="0,0.05,0.1,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--pair-type-weights", default="")
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = {str(row["id"]): row for row in read_jsonl(Path(args.prediction_jsonl))}
    labels = {}
    if args.labels_jsonl:
        labels = {str(row["id"]): row for row in read_jsonl(Path(args.labels_jsonl))}

    pair_weights, default_pair_weight = parse_pair_type_weights(args.pair_type_weights)
    examples, cache, pair_report = load_pair_examples(
        Path(args.pair_jsonl),
        prediction_rows,
        pair_weights,
        default_pair_weight,
    )
    train_examples, dev_examples = split_pair_examples(examples, args.train_fraction, args.seed)
    normalizer = build_normalizer(normalizer_samples_from_pair_examples(train_examples, cache))

    model, history, pair_info = train_model(args, train_examples, dev_examples, normalizer)

    samples = load_labeled_samples(prediction_rows, labels) if labels else []
    dev_sample_ids = sorted({example["sample_id"] for example in dev_examples})
    train_sample_ids = sorted({example["sample_id"] for example in train_examples})
    best_margin, margin_reports = choose_best_margin(
        model,
        normalizer,
        samples,
        args.fallback_mode,
        parse_margin_grid(args.margin_grid),
        dev_sample_ids,
    )

    output_rows = build_output_rows(
        model,
        normalizer,
        prediction_rows,
        labels,
        args.fallback_mode,
        best_margin,
        keep_candidates=args.keep_candidates,
    )
    pred_path = out_dir / "pred_pair_reward_head.jsonl"
    write_jsonl(pred_path, output_rows)

    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_names": FEATURE_NAMES,
        "normalizer": normalizer,
        "args": vars(args),
        "best_margin": best_margin,
        "pair_type_weights": pair_weights,
        "default_pair_weight": default_pair_weight,
    }
    checkpoint_path = out_dir / "reward_head.pt"
    torch.save(checkpoint, checkpoint_path)

    report = {
        "pair_jsonl": args.pair_jsonl,
        "prediction_jsonl": args.prediction_jsonl,
        "labels_jsonl": args.labels_jsonl,
        "output_dir": str(out_dir),
        "feature_names": FEATURE_NAMES,
        "pair_type_weights": pair_weights,
        "default_pair_weight": default_pair_weight,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "accepted_pair_count": len(examples),
        "train_pair_example_count": len(train_examples),
        "dev_pair_example_count": len(dev_examples),
        "train_sample_count": len(train_sample_ids),
        "dev_sample_count": len(dev_sample_ids),
        **pair_info,
        "pair_report": pair_report,
        "history": history,
        "margin_reports": margin_reports,
        "best_margin": best_margin,
        "train_eval": evaluate_samples(model, normalizer, samples, args.fallback_mode, best_margin, train_sample_ids) if samples else None,
        "dev_eval": evaluate_samples(model, normalizer, samples, args.fallback_mode, best_margin, dev_sample_ids) if samples else None,
        "all_eval": evaluate_samples(model, normalizer, samples, args.fallback_mode, best_margin) if samples else None,
        "prediction_output": str(pred_path),
        "checkpoint": str(checkpoint_path),
    }
    (out_dir / "pair_reward_head_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
