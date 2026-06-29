#!/usr/bin/env python3
import argparse
import json
import math
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

from rerank_ocsr_candidates import (  # noqa: E402
    aggregate_candidates,
    canonicalize,
    choose_chem_light,
    get_ground_truth_smiles,
)
from reward_policy_reranker import (  # noqa: E402
    FEATURE_NAMES,
    build_normalizer,
    bounded_score,
    candidate_feature_dict,
    load_labeled_samples,
    split_train_dev,
    stable_fraction,
    vectorize,
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


class RewardHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        mid_dim = max(8, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


def build_pair_tensors(samples, normalizer, max_negatives_per_positive: int):
    positives = []
    negatives = []
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
                positives.append(vectors[positive_index])
                negatives.append(vectors[negative_index])
    if not positives:
        empty = torch.empty((0, len(FEATURE_NAMES)), dtype=torch.float32)
        return empty, empty
    return (
        torch.tensor(positives, dtype=torch.float32),
        torch.tensor(negatives, dtype=torch.float32),
    )


def score_vector(model, normalizer, features):
    vector = torch.tensor([vectorize(features, normalizer)], dtype=torch.float32)
    with torch.no_grad():
        return float(model(vector).item())


def choose_selected(row: dict, aggregates: list[dict]):
    selected = canonicalize(row.get("canonical_prediction") or row.get("prediction"))
    if selected:
        for aggregate in aggregates:
            if aggregate.get("canonical") == selected:
                return aggregate
    return None


def choose_policy(model, normalizer, sample):
    scored = []
    for aggregate, features in zip(sample["aggregates"], sample["features"]):
        score = score_vector(model, normalizer, features)
        scored.append((score, aggregate))
    return max(
        scored,
        key=lambda item: (
            item[0],
            item[1].get("count", 0),
            bounded_score(item[1].get("max_score")),
            -item[1].get("min_prompt_index", 99),
            -item[1].get("min_generation_index", 99),
        ),
    )


def choose_final(model, normalizer, sample, fallback_mode: str, policy_margin: float):
    policy_score, policy = choose_policy(model, normalizer, sample)
    fallback = None
    if fallback_mode == "selected":
        fallback = choose_selected(sample["row"], sample["aggregates"])
    elif fallback_mode == "chem_light":
        fallback = choose_chem_light(
            sample["aggregates"],
            salt_bonus=2.0,
            heavy_penalty=0.02,
            stereo_min_ratio=0.25,
        )
    elif fallback_mode != "none":
        raise ValueError(f"unknown fallback mode: {fallback_mode}")

    if fallback is None:
        return policy, policy_score, "reward_head"

    fallback_index = sample["aggregates"].index(fallback)
    fallback_score = score_vector(model, normalizer, sample["features"][fallback_index])
    if fallback.get("canonical") == policy.get("canonical"):
        return fallback, fallback_score, f"reward_head_{fallback_mode}_agreement"
    if policy_score - fallback_score >= policy_margin:
        return policy, policy_score, f"reward_head_override_{fallback_mode}"
    return fallback, fallback_score, f"reward_head_fallback_{fallback_mode}"


def add_metric(acc, target, selected, chem_light, policy):
    acc["total"] += 1
    acc["selected_exact"] += int(bool(selected and selected == target))
    acc["chem_light_exact"] += int(bool(chem_light and chem_light == target))
    acc["policy_exact"] += int(bool(policy and policy == target))
    if selected != policy:
        acc["policy_changed"] += 1
        acc["policy_good_changes"] += int(bool(policy and policy == target and selected != target))
        acc["policy_bad_changes"] += int(bool(selected and selected == target and policy != target))
    if chem_light != policy:
        acc["policy_changed_from_chem_light"] += 1
        acc["policy_good_changes_from_chem_light"] += int(bool(policy and policy == target and chem_light != target))
        acc["policy_bad_changes_from_chem_light"] += int(bool(chem_light and chem_light == target and policy != target))


def finalize_metric(acc):
    total = acc.get("total", 0)
    if total == 0:
        return {
            "total": 0,
            "selected_exact": 0.0,
            "chem_light_exact": 0.0,
            "policy_exact": 0.0,
        }
    out = dict(acc)
    for key in ["selected_exact", "chem_light_exact", "policy_exact"]:
        out[key] = acc[key] / total
    out["policy_gain_over_selected"] = (acc["policy_exact"] - acc["selected_exact"]) / total
    out["policy_gain_over_chem_light"] = (acc["policy_exact"] - acc["chem_light_exact"]) / total
    return out


def evaluate_samples(model, normalizer, samples, fallback_mode: str, policy_margin: float, split_ids=None):
    split_id_set = set(split_ids) if split_ids is not None else None
    overall = Counter()
    by_panel = defaultdict(Counter)
    by_source = defaultdict(Counter)
    for sample in samples:
        if split_id_set is not None and sample["id"] not in split_id_set:
            continue
        selected = choose_selected(sample["row"], sample["aggregates"])
        chem = choose_chem_light(
            sample["aggregates"],
            salt_bonus=2.0,
            heavy_penalty=0.02,
            stereo_min_ratio=0.25,
        )
        policy, _, _ = choose_final(model, normalizer, sample, fallback_mode, policy_margin)
        target = sample["target"]
        selected_canonical = selected.get("canonical") if selected else None
        chem_canonical = chem.get("canonical") if chem else None
        policy_canonical = policy.get("canonical") if policy else None
        add_metric(overall, target, selected_canonical, chem_canonical, policy_canonical)
        panel = sample["row"].get("eval_panel") or sample["label"].get("eval_panel", "")
        source = sample["label"].get("source", "")
        add_metric(by_panel[panel], target, selected_canonical, chem_canonical, policy_canonical)
        add_metric(by_source[source], target, selected_canonical, chem_canonical, policy_canonical)
    return {
        "metrics": finalize_metric(overall),
        "by_panel": {key: finalize_metric(value) for key, value in sorted(by_panel.items())},
        "by_source": {key: finalize_metric(value) for key, value in sorted(by_source.items())},
    }


def build_output_rows(model, normalizer, prediction_rows, labels, fallback_mode, policy_margin, keep_candidates=False):
    output = []
    for sample_id, row in prediction_rows.items():
        aggregates = aggregate_candidates(row.get("candidates", []))
        label = labels.get(sample_id, {})
        if not aggregates:
            out = dict(row)
            out["selection_reason"] = "reward_head_no_valid_candidate"
            if not keep_candidates:
                out.pop("candidates", None)
            output.append(out)
            continue
        features = [candidate_feature_dict(row, aggregate, aggregates) for aggregate in aggregates]
        sample = {
            "id": sample_id,
            "row": row,
            "label": label,
            "target": canonicalize(get_ground_truth_smiles(label)) if label else None,
            "aggregates": aggregates,
            "features": features,
            "positive_indices": [],
        }
        chosen, score, reason = choose_final(model, normalizer, sample, fallback_mode, policy_margin)
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
            "unique_valid_candidates": len(aggregates),
        }
        if not keep_candidates:
            out.pop("candidates", None)
        output.append(out)
    return output


def parse_margin_grid(text: str):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def train_model(args, train_samples, dev_samples, normalizer):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = RewardHead(len(FEATURE_NAMES), args.hidden_dim, args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_pos, train_neg = build_pair_tensors(train_samples, normalizer, args.max_negatives_per_positive)
    dev_pos, dev_neg = build_pair_tensors(dev_samples, normalizer, args.max_negatives_per_positive)
    if train_pos.numel() == 0:
        raise SystemExit("no train pairs")

    best_state = None
    best_key = None
    history = []
    rng = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(train_pos.shape[0], generator=rng)
        losses = []
        correct = 0
        total = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            pos = train_pos[idx]
            neg = train_neg[idx]
            optimizer.zero_grad(set_to_none=True)
            margin = model(pos) - model(neg)
            loss = F.softplus(-margin).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()) * len(idx))
            correct += int((margin.detach() > 0).sum().item())
            total += len(idx)
        train_loss = sum(losses) / max(1, total)
        train_acc = correct / max(1, total)

        model.eval()
        dev_loss = None
        dev_acc = None
        if dev_pos.numel() > 0:
            with torch.no_grad():
                dev_margin = model(dev_pos) - model(dev_neg)
                dev_loss = float(F.softplus(-dev_margin).mean().item())
                dev_acc = float((dev_margin > 0).float().mean().item())
        key = (dev_acc if dev_acc is not None else train_acc, -(dev_loss if dev_loss is not None else train_loss))
        if best_key is None or key > best_key:
            best_key = key
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_pair_accuracy": train_acc,
                    "dev_loss": dev_loss,
                    "dev_pair_accuracy": dev_acc,
                }
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, {"train_pair_count": int(train_pos.shape[0]), "dev_pair_count": int(dev_pos.shape[0])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-negatives-per-positive", type=int, default=8)
    parser.add_argument("--fallback-mode", choices=["none", "selected", "chem_light"], default="chem_light")
    parser.add_argument("--margin-grid", default="0,0.1,0.25,0.5,0.75,1,1.5,2")
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

    model, history, pair_info = train_model(args, train_samples, dev_samples, normalizer)
    margins = parse_margin_grid(args.margin_grid)
    train_ids = [sample["id"] for sample in train_samples]
    dev_ids = [sample["id"] for sample in dev_samples]
    margin_reports = []
    for margin in margins:
        dev_report = evaluate_samples(model, normalizer, samples, args.fallback_mode, margin, dev_ids)
        metric = dev_report["metrics"]
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
    )["margin"] if margin_reports else 0.0

    output_rows = build_output_rows(
        model,
        normalizer,
        prediction_rows,
        labels,
        args.fallback_mode,
        best_margin,
        keep_candidates=args.keep_candidates,
    )
    pred_path = out_dir / "pred_reward_head.jsonl"
    write_jsonl(pred_path, output_rows)

    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_names": FEATURE_NAMES,
        "normalizer": normalizer,
        "args": vars(args),
        "best_margin": best_margin,
    }
    torch.save(checkpoint, out_dir / "reward_head.pt")

    report = {
        "prediction_jsonl": args.prediction_jsonl,
        "labels_jsonl": args.labels_jsonl,
        "output_dir": str(out_dir),
        "feature_names": FEATURE_NAMES,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "sample_count": len(samples),
        "train_sample_count": len(train_samples),
        "dev_sample_count": len(dev_samples),
        **pair_info,
        "history": history,
        "margin_reports": margin_reports,
        "best_margin": best_margin,
        "train_eval": evaluate_samples(model, normalizer, samples, args.fallback_mode, best_margin, train_ids),
        "dev_eval": evaluate_samples(model, normalizer, samples, args.fallback_mode, best_margin, dev_ids),
        "all_eval": evaluate_samples(model, normalizer, samples, args.fallback_mode, best_margin),
        "prediction_output": str(pred_path),
        "checkpoint": str(out_dir / "reward_head.pt"),
    }
    (out_dir / "reward_head_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
