#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rerank_ocsr_candidates import aggregate_candidates, choose_chem_light  # noqa: E402
from reward_policy_reranker import (  # noqa: E402
    FEATURE_NAMES,
    bounded_score,
    candidate_feature_dict,
    vectorize,
)
from train_candidate_reward_head import (  # noqa: E402
    RewardHead,
    choose_selected,
    read_jsonl,
    write_jsonl,
)


def parse_checkpoint_paths(text: str) -> list[Path]:
    paths = [Path(item.strip()) for item in text.split(",") if item.strip()]
    if not paths:
        raise SystemExit("no checkpoints provided")
    return paths


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    feature_names = checkpoint.get("feature_names", [])
    if feature_names != FEATURE_NAMES:
        raise SystemExit(
            f"feature mismatch for {path}: checkpoint has {feature_names}, current code has {FEATURE_NAMES}"
        )
    train_args = checkpoint.get("args", {})
    model = RewardHead(
        len(FEATURE_NAMES),
        int(train_args.get("hidden_dim", 64)),
        float(train_args.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "path": str(path),
        "checkpoint": checkpoint,
        "model": model,
        "normalizer": checkpoint["normalizer"],
        "best_margin": float(checkpoint.get("best_margin", 0.0)),
        "fallback_mode": train_args.get("fallback_mode", "chem_light"),
    }


def resolve_margin(models, explicit_margin, margin_mode: str) -> float:
    if explicit_margin is not None:
        return float(explicit_margin)
    margins = [item["best_margin"] for item in models]
    if margin_mode == "mean":
        return float(statistics.mean(margins))
    if margin_mode == "min":
        return float(min(margins))
    if margin_mode == "max":
        return float(max(margins))
    if margin_mode == "zero":
        return 0.0
    raise ValueError(f"unknown margin mode: {margin_mode}")


def score_one(model_info, features: dict) -> float:
    vector = torch.tensor([vectorize(features, model_info["normalizer"])], dtype=torch.float32)
    with torch.no_grad():
        return float(model_info["model"](vector).item())


def choose_fallback(row: dict, aggregates: list[dict], fallback_mode: str):
    if fallback_mode == "selected":
        return choose_selected(row, aggregates)
    if fallback_mode == "chem_light":
        return choose_chem_light(
            aggregates,
            salt_bonus=2.0,
            heavy_penalty=0.02,
            stereo_min_ratio=0.25,
        )
    if fallback_mode == "none":
        return None
    raise ValueError(f"unknown fallback mode: {fallback_mode}")


def choose_policy(models, features_list: list[dict], aggregates: list[dict]):
    scored = []
    for aggregate, features in zip(aggregates, features_list):
        component_scores = [score_one(model_info, features) for model_info in models]
        score = float(statistics.mean(component_scores))
        scored.append((score, component_scores, aggregate))
    return max(
        scored,
        key=lambda item: (
            item[0],
            item[2].get("count", 0),
            bounded_score(item[2].get("max_score")),
            -item[2].get("min_prompt_index", 99),
            -item[2].get("min_generation_index", 99),
        ),
    )


def build_output_rows(models, prediction_rows, fallback_mode: str, policy_margin: float, keep_candidates: bool):
    output = []
    for sample_id, row in prediction_rows.items():
        aggregates = aggregate_candidates(row.get("candidates", []))
        if not aggregates:
            out = dict(row)
            out["selection_reason"] = "reward_head_ensemble_no_valid_candidate"
            if not keep_candidates:
                out.pop("candidates", None)
            output.append(out)
            continue

        features_list = [candidate_feature_dict(row, aggregate, aggregates) for aggregate in aggregates]
        policy_score, policy_components, policy = choose_policy(models, features_list, aggregates)
        fallback = choose_fallback(row, aggregates, fallback_mode)
        chosen = policy
        chosen_score = policy_score
        chosen_components = policy_components
        reason = "reward_head_ensemble"

        if fallback is not None:
            fallback_index = aggregates.index(fallback)
            fallback_components = [score_one(model_info, features_list[fallback_index]) for model_info in models]
            fallback_score = float(statistics.mean(fallback_components))
            if fallback.get("canonical") == policy.get("canonical"):
                chosen = fallback
                chosen_score = fallback_score
                chosen_components = fallback_components
                reason = f"reward_head_ensemble_{fallback_mode}_agreement"
            elif policy_score - fallback_score >= policy_margin:
                reason = f"reward_head_ensemble_override_{fallback_mode}"
            else:
                chosen = fallback
                chosen_score = fallback_score
                chosen_components = fallback_components
                reason = f"reward_head_ensemble_fallback_{fallback_mode}"

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
            "reward_head_score": chosen_score,
            "reward_head_component_scores": chosen_components,
            "policy_margin": policy_margin,
            "ensemble_size": len(models),
            "unique_valid_candidates": len(aggregates),
        }
        if not keep_candidates:
            out.pop("candidates", None)
        output.append(out)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True, help="comma-separated reward_head.pt paths")
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--fallback-mode", choices=["none", "selected", "chem_light"], default="")
    parser.add_argument("--policy-margin", type=float, default=None)
    parser.add_argument("--margin-mode", choices=["mean", "min", "max", "zero"], default="mean")
    parser.add_argument("--keep-candidates", action="store_true")
    args = parser.parse_args()

    models = [load_checkpoint(path) for path in parse_checkpoint_paths(args.checkpoints)]
    fallback_mode = args.fallback_mode or models[0]["fallback_mode"]
    policy_margin = resolve_margin(models, args.policy_margin, args.margin_mode)
    prediction_rows = {str(row["id"]): row for row in read_jsonl(Path(args.prediction_jsonl))}
    output_rows = build_output_rows(
        models,
        prediction_rows,
        fallback_mode=fallback_mode,
        policy_margin=policy_margin,
        keep_candidates=args.keep_candidates,
    )
    write_jsonl(Path(args.output_jsonl), output_rows)
    print(
        json.dumps(
            {
                "checkpoints": [item["path"] for item in models],
                "best_margins": [item["best_margin"] for item in models],
                "prediction_jsonl": args.prediction_jsonl,
                "output_jsonl": args.output_jsonl,
                "fallback_mode": fallback_mode,
                "policy_margin": policy_margin,
                "margin_mode": args.margin_mode,
                "rows": len(output_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
