#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_candidate_reward_head import (  # noqa: E402
    RewardHead,
    build_output_rows,
    read_jsonl,
    write_jsonl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--labels-jsonl", default="")
    parser.add_argument("--fallback-mode", choices=["none", "selected", "chem_light"], default="")
    parser.add_argument("--policy-margin", type=float, default=None)
    parser.add_argument("--keep-candidates", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args", {})
    fallback_mode = args.fallback_mode or train_args.get("fallback_mode", "chem_light")
    policy_margin = args.policy_margin
    if policy_margin is None:
        policy_margin = float(checkpoint.get("best_margin", 0.0))

    feature_names = checkpoint.get("feature_names", [])
    hidden_dim = int(train_args.get("hidden_dim", 64))
    dropout = float(train_args.get("dropout", 0.0))
    model = RewardHead(len(feature_names), hidden_dim, dropout)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    prediction_rows = {str(row["id"]): row for row in read_jsonl(Path(args.prediction_jsonl))}
    labels = {}
    if args.labels_jsonl:
        labels = {str(row["id"]): row for row in read_jsonl(Path(args.labels_jsonl))}
    output_rows = build_output_rows(
        model,
        checkpoint["normalizer"],
        prediction_rows,
        labels,
        fallback_mode,
        policy_margin,
        keep_candidates=args.keep_candidates,
    )
    write_jsonl(Path(args.output_jsonl), output_rows)
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "prediction_jsonl": args.prediction_jsonl,
                "output_jsonl": args.output_jsonl,
                "fallback_mode": fallback_mode,
                "policy_margin": policy_margin,
                "rows": len(output_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
