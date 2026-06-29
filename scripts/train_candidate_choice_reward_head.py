#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reward_policy_reranker import (  # noqa: E402
    FEATURE_NAMES,
    bounded_score,
    build_normalizer,
    load_labeled_samples,
    split_train_dev,
    vectorize,
)
from train_candidate_reward_head import (  # noqa: E402
    RewardHead,
    build_output_rows,
    choose_chem_light,
    choose_selected,
    evaluate_samples,
    parse_margin_grid,
    read_jsonl,
    write_jsonl,
)


def parse_weight_rules(text: str):
    default_weight = 1.0
    rules = []
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"bad sample weight rule: {item!r}")
        left, right = item.split("=", 1)
        weight = float(right.strip())
        left = left.strip()
        if left == "default":
            default_weight = weight
            continue
        if ":" not in left:
            raise SystemExit(f"bad sample weight rule: {item!r}")
        field, value = left.split(":", 1)
        rules.append((field.strip(), value.strip(), weight))
    return default_weight, rules


def sample_weight(sample: dict, default_weight: float, rules):
    label = sample.get("label", {})
    row = sample.get("row", {})
    weight = default_weight
    for field, value, rule_weight in rules:
        if str(label.get(field, row.get(field, ""))) == value:
            weight *= rule_weight
    return max(0.0, float(weight))


def candidate_rank_key(sample: dict, index: int):
    aggregate = sample["aggregates"][index]
    features = sample["features"][index]
    return (
        features.get("selected_match", 0.0),
        aggregate.get("count", 0),
        bounded_score(aggregate.get("max_score")),
        -aggregate.get("min_prompt_index", 99),
        -aggregate.get("min_generation_index", 99),
    )


def keep_candidate_indices(sample: dict, max_candidates: int):
    indices = list(range(len(sample["aggregates"])))
    if max_candidates <= 0 or len(indices) <= max_candidates:
        return indices
    positives = set(sample["positive_indices"])
    negatives = [index for index in indices if index not in positives]
    negatives.sort(key=lambda index: candidate_rank_key(sample, index), reverse=True)
    kept = sorted(positives)
    kept.extend(negatives[: max(0, max_candidates - len(kept))])
    return sorted(set(kept))


def build_choice_examples(samples, normalizer, default_weight: float, rules, max_candidates: int):
    examples = []
    stats = {
        "samples_seen": 0,
        "accepted": 0,
        "no_oracle_positive": 0,
        "single_candidate": 0,
        "zero_weight": 0,
    }
    for sample in samples:
        stats["samples_seen"] += 1
        if not sample["positive_indices"]:
            stats["no_oracle_positive"] += 1
            continue
        if len(sample["aggregates"]) <= 1:
            stats["single_candidate"] += 1
            continue
        weight = sample_weight(sample, default_weight, rules)
        if weight <= 0.0:
            stats["zero_weight"] += 1
            continue
        indices = keep_candidate_indices(sample, max_candidates)
        positive_set = set(sample["positive_indices"])
        mask = torch.tensor([index in positive_set for index in indices], dtype=torch.bool)
        if not bool(mask.any()):
            stats["no_oracle_positive"] += 1
            continue
        vectors = [vectorize(sample["features"][index], normalizer) for index in indices]
        examples.append(
            {
                "id": sample["id"],
                "features": torch.tensor(vectors, dtype=torch.float32),
                "positive_mask": mask,
                "weight": weight,
            }
        )
        stats["accepted"] += 1
    return examples, stats


def choice_loss(model, batch, hard_negative_loss_weight: float):
    losses = []
    weights = []
    for example in batch:
        scores = model(example["features"])
        pos_mask = example["positive_mask"]
        loss = torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[pos_mask], dim=0)
        neg_mask = ~pos_mask
        if hard_negative_loss_weight > 0 and bool(neg_mask.any()):
            hard_pos = scores[pos_mask].max()
            hard_neg = scores[neg_mask].max()
            loss = loss + hard_negative_loss_weight * F.softplus(-(hard_pos - hard_neg))
        losses.append(loss)
        weights.append(float(example["weight"]))
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=loss_tensor.device)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-8)


def evaluate_choice(model, examples):
    if not examples:
        return {"sample_count": 0, "choice_accuracy": None, "weighted_choice_accuracy": None, "choice_loss": None}
    correct = 0
    weighted_correct = 0.0
    weight_sum = 0.0
    loss_sum = 0.0
    with torch.no_grad():
        for example in examples:
            scores = model(example["features"])
            pos_mask = example["positive_mask"]
            is_correct = bool(pos_mask[int(torch.argmax(scores).item())].item())
            weight = float(example["weight"])
            correct += int(is_correct)
            weighted_correct += weight * int(is_correct)
            weight_sum += weight
            loss = torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[pos_mask], dim=0)
            loss_sum += float(loss.item()) * weight
    return {
        "sample_count": len(examples),
        "choice_accuracy": correct / len(examples),
        "weighted_choice_accuracy": weighted_correct / max(1e-8, weight_sum),
        "choice_loss": loss_sum / max(1e-8, weight_sum),
    }


def checkpoint_key_from_choice(train_eval: dict, dev_eval: dict):
    return (
        dev_eval["weighted_choice_accuracy"]
        if dev_eval["weighted_choice_accuracy"] is not None
        else train_eval["weighted_choice_accuracy"],
        dev_eval["choice_accuracy"] if dev_eval["choice_accuracy"] is not None else train_eval["choice_accuracy"],
        -(dev_eval["choice_loss"] if dev_eval["choice_loss"] is not None else train_eval["choice_loss"]),
    )


def checkpoint_key_from_policy(model, normalizer, samples, fallback_modes, margins, dev_ids, train_eval, dev_eval):
    best_policy, _ = choose_best_policy(model, normalizer, samples, fallback_modes, margins, dev_ids)
    return (
        best_policy["dev_policy_exact"],
        best_policy["dev_gain_over_selected"],
        best_policy["dev_gain_over_chem_light"],
        -best_policy["dev_policy_bad_changes"],
        best_policy["dev_policy_good_changes"],
        dev_eval["weighted_choice_accuracy"] if dev_eval["weighted_choice_accuracy"] is not None else 0.0,
        -(dev_eval["choice_loss"] if dev_eval["choice_loss"] is not None else 0.0),
    ), best_policy


def train_model(args, train_examples, dev_examples, normalizer, samples, fallback_modes, margins, dev_ids):
    if not train_examples:
        raise SystemExit("no train choice examples")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)
    model = RewardHead(len(FEATURE_NAMES), args.hidden_dim, args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_key = None
    best_epoch = None
    best_selection = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        loss_sum = 0.0
        weight_sum = 0
        for start in range(0, len(order), args.batch_size):
            batch = [train_examples[index] for index in order[start : start + args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = choice_loss(model, batch, args.hard_negative_loss_weight)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch)
            weight_sum += len(batch)

        model.eval()
        train_eval = evaluate_choice(model, train_examples)
        dev_eval = evaluate_choice(model, dev_examples)
        key = None
        selection = None
        if args.checkpoint_selection == "policy":
            if epoch == 1 or epoch == args.epochs or epoch % args.selection_every == 0:
                key, selection = checkpoint_key_from_policy(
                    model,
                    normalizer,
                    samples,
                    fallback_modes,
                    margins,
                    dev_ids,
                    train_eval,
                    dev_eval,
                )
        elif args.checkpoint_selection == "final":
            key = (epoch,)
            selection = {"checkpoint_selection": "final"}
        else:
            key = checkpoint_key_from_choice(train_eval, dev_eval)
            selection = {
                "checkpoint_selection": "choice",
                "dev_choice_accuracy": dev_eval["choice_accuracy"],
                "dev_weighted_choice_accuracy": dev_eval["weighted_choice_accuracy"],
                "dev_choice_loss": dev_eval["choice_loss"],
            }
        if key is not None and (best_key is None or key > best_key):
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch
            best_selection = selection
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            history.append(
                {
                    "epoch": epoch,
                    "train_epoch_loss": loss_sum / max(1, weight_sum),
                    "train_choice_accuracy": train_eval["choice_accuracy"],
                    "train_weighted_choice_accuracy": train_eval["weighted_choice_accuracy"],
                    "train_choice_loss": train_eval["choice_loss"],
                    "dev_choice_accuracy": dev_eval["choice_accuracy"],
                    "dev_weighted_choice_accuracy": dev_eval["weighted_choice_accuracy"],
                    "dev_choice_loss": dev_eval["choice_loss"],
                    "checkpoint_selection_key": list(key) if key is not None else None,
                    "checkpoint_selection": selection,
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "best_key": list(best_key or []), "best_selection": best_selection}


def aggregate_index(sample: dict, aggregate):
    if aggregate is None:
        return None
    for index, item in enumerate(sample["aggregates"]):
        if item is aggregate:
            return index
    canonical = aggregate.get("canonical")
    for index, item in enumerate(sample["aggregates"]):
        if item.get("canonical") == canonical:
            return index
    return None


def best_policy_index(sample: dict, scores):
    return max(
        range(len(sample["aggregates"])),
        key=lambda index: (
            float(scores[index]),
            sample["aggregates"][index].get("count", 0),
            bounded_score(sample["aggregates"][index].get("max_score")),
            -sample["aggregates"][index].get("min_prompt_index", 99),
            -sample["aggregates"][index].get("min_generation_index", 99),
        ),
    )


def precompute_policy_records(model, normalizer, samples, dev_ids):
    dev_id_set = set(dev_ids)
    records = []
    with torch.no_grad():
        for sample in samples:
            if sample["id"] not in dev_id_set:
                continue
            vectors = [vectorize(features, normalizer) for features in sample["features"]]
            score_tensor = model(torch.tensor(vectors, dtype=torch.float32))
            scores = [float(value) for value in score_tensor.detach().cpu().tolist()]
            policy_index = best_policy_index(sample, scores)
            selected_index = aggregate_index(sample, choose_selected(sample["row"], sample["aggregates"]))
            chem_index = aggregate_index(
                sample,
                choose_chem_light(
                    sample["aggregates"],
                    salt_bonus=2.0,
                    heavy_penalty=0.02,
                    stereo_min_ratio=0.25,
                ),
            )
            target = sample["target"]
            records.append(
                {
                    "sample": sample,
                    "target": target,
                    "scores": scores,
                    "policy_index": policy_index,
                    "selected_index": selected_index,
                    "chem_index": chem_index,
                    "selected_correct": bool(
                        selected_index is not None
                        and sample["aggregates"][selected_index].get("canonical") == target
                    ),
                    "chem_correct": bool(
                        chem_index is not None and sample["aggregates"][chem_index].get("canonical") == target
                    ),
                }
            )
    return records


def choose_record_index(record, fallback_mode: str, margin: float):
    policy_index = record["policy_index"]
    if fallback_mode == "none":
        return policy_index
    if fallback_mode == "selected":
        fallback_index = record["selected_index"]
    elif fallback_mode == "chem_light":
        fallback_index = record["chem_index"]
    else:
        raise ValueError(f"unknown fallback mode: {fallback_mode}")
    if fallback_index is None:
        return policy_index
    sample = record["sample"]
    policy = sample["aggregates"][policy_index]
    fallback = sample["aggregates"][fallback_index]
    if policy.get("canonical") == fallback.get("canonical"):
        return fallback_index
    if record["scores"][policy_index] - record["scores"][fallback_index] >= margin:
        return policy_index
    return fallback_index


def evaluate_policy_records(records, fallback_mode: str, margin: float):
    total = len(records)
    selected_exact = 0
    chem_exact = 0
    policy_exact = 0
    good_changes = 0
    bad_changes = 0
    for record in records:
        sample = record["sample"]
        selected_index = record["selected_index"]
        chosen_index = choose_record_index(record, fallback_mode, margin)
        target = record["target"]
        selected_correct = record["selected_correct"]
        chem_correct = record["chem_correct"]
        policy_correct = bool(sample["aggregates"][chosen_index].get("canonical") == target)
        selected_exact += int(selected_correct)
        chem_exact += int(chem_correct)
        policy_exact += int(policy_correct)
        selected_canonical = sample["aggregates"][selected_index].get("canonical") if selected_index is not None else None
        policy_canonical = sample["aggregates"][chosen_index].get("canonical")
        if selected_canonical != policy_canonical:
            good_changes += int(policy_correct and not selected_correct)
            bad_changes += int(selected_correct and not policy_correct)
    if total <= 0:
        return {
            "policy_exact": 0.0,
            "policy_gain_over_selected": 0.0,
            "policy_gain_over_chem_light": 0.0,
            "policy_bad_changes": 0,
            "policy_good_changes": 0,
        }
    return {
        "policy_exact": policy_exact / total,
        "policy_gain_over_selected": (policy_exact - selected_exact) / total,
        "policy_gain_over_chem_light": (policy_exact - chem_exact) / total,
        "policy_bad_changes": bad_changes,
        "policy_good_changes": good_changes,
    }


def choose_best_policy(model, normalizer, samples, fallback_modes, margins, dev_ids):
    reports = []
    records = precompute_policy_records(model, normalizer, samples, dev_ids)
    for fallback_mode in fallback_modes:
        for margin in margins:
            metric = evaluate_policy_records(records, fallback_mode, margin)
            reports.append(
                {
                    "fallback_mode": fallback_mode,
                    "margin": margin,
                    "dev_policy_exact": metric["policy_exact"],
                    "dev_gain_over_selected": metric.get("policy_gain_over_selected", 0.0),
                    "dev_gain_over_chem_light": metric.get("policy_gain_over_chem_light", 0.0),
                    "dev_policy_bad_changes": metric.get("policy_bad_changes", 0),
                    "dev_policy_good_changes": metric.get("policy_good_changes", 0),
                }
            )
    best = max(
        reports,
        key=lambda item: (
            item["dev_policy_exact"],
            item["dev_gain_over_selected"],
            item["dev_gain_over_chem_light"],
            -item["dev_policy_bad_changes"],
        ),
    )
    return best, reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--fallback-modes", default="chem_light,selected,none")
    parser.add_argument("--margin-grid", default="0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.35,0.5")
    parser.add_argument("--sample-weight-rules", default="")
    parser.add_argument("--max-candidates-per-sample", type=int, default=0)
    parser.add_argument("--hard-negative-loss-weight", type=float, default=0.15)
    parser.add_argument(
        "--selection-every",
        type=int,
        default=5,
        help="When checkpoint-selection=policy, evaluate dev rerank policy every N epochs plus first/final.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=["policy", "choice", "final"],
        default="policy",
        help="Select checkpoint by dev rerank policy, listwise choice accuracy, or final epoch.",
    )
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = {str(row["id"]): row for row in read_jsonl(Path(args.labels_jsonl))}
    prediction_rows = {str(row["id"]): row for row in read_jsonl(Path(args.prediction_jsonl))}
    samples = load_labeled_samples(prediction_rows, labels)
    train_samples, dev_samples = split_train_dev(samples, args.train_fraction, args.seed)
    normalizer = build_normalizer(train_samples)
    default_weight, rules = parse_weight_rules(args.sample_weight_rules)
    train_examples, train_stats = build_choice_examples(
        train_samples, normalizer, default_weight, rules, args.max_candidates_per_sample
    )
    dev_examples, dev_stats = build_choice_examples(
        dev_samples, normalizer, default_weight, rules, args.max_candidates_per_sample
    )
    dev_ids = [sample["id"] for sample in dev_samples]
    train_ids = [sample["id"] for sample in train_samples]
    fallback_modes = [item.strip() for item in args.fallback_modes.split(",") if item.strip()]
    margins = parse_margin_grid(args.margin_grid)
    model, history, checkpoint_selection = train_model(
        args,
        train_examples,
        dev_examples,
        normalizer,
        samples,
        fallback_modes,
        margins,
        dev_ids,
    )
    best_policy, policy_reports = choose_best_policy(
        model, normalizer, samples, fallback_modes, margins, dev_ids
    )
    output_rows = build_output_rows(
        model,
        normalizer,
        prediction_rows,
        labels,
        best_policy["fallback_mode"],
        best_policy["margin"],
        keep_candidates=args.keep_candidates,
    )
    pred_path = out_dir / "pred_choice_reward_head.jsonl"
    write_jsonl(pred_path, output_rows)
    checkpoint_path = out_dir / "reward_head.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "normalizer": normalizer,
            "args": {**vars(args), "fallback_mode": best_policy["fallback_mode"]},
            "best_margin": best_policy["margin"],
            "best_policy": best_policy,
            "checkpoint_selection": checkpoint_selection,
            "sample_weight_rules_parsed": rules,
            "default_sample_weight": default_weight,
        },
        checkpoint_path,
    )
    report = {
        "training_objective": "listwise_candidate_choice_softmax",
        "prediction_jsonl": args.prediction_jsonl,
        "labels_jsonl": args.labels_jsonl,
        "output_dir": str(out_dir),
        "sample_count": len(samples),
        "train_sample_count": len(train_samples),
        "dev_sample_count": len(dev_samples),
        "train_choice_stats": train_stats,
        "dev_choice_stats": dev_stats,
        "checkpoint_selection": checkpoint_selection,
        "train_choice_eval": evaluate_choice(model, train_examples),
        "dev_choice_eval": evaluate_choice(model, dev_examples),
        "history": history,
        "policy_reports": policy_reports,
        "best_policy": best_policy,
        "train_eval": evaluate_samples(
            model, normalizer, samples, best_policy["fallback_mode"], best_policy["margin"], train_ids
        ),
        "dev_eval": evaluate_samples(
            model, normalizer, samples, best_policy["fallback_mode"], best_policy["margin"], dev_ids
        ),
        "all_eval": evaluate_samples(model, normalizer, samples, best_policy["fallback_mode"], best_policy["margin"]),
        "prediction_output": str(pred_path),
        "checkpoint": str(checkpoint_path),
    }
    (out_dir / "choice_reward_head_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
