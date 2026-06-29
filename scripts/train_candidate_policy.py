from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from candidate_policy import feature_vector, make_policy, save_policy


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def candidate_label(candidate: dict) -> float:
    if candidate.get("is_chosen"):
        return 1.0
    if candidate.get("is_rejected"):
        return 0.0
    if candidate.get("role") == "chosen":
        return 1.0
    if candidate.get("role") == "rejected":
        return 0.0
    return 0.0


def collect_rows(path: Path):
    for row in read_jsonl(path):
        candidates = row.get("candidates") or []
        if not candidates:
            continue
        chosen = row.get("selected_canonical_prediction") or row.get("canonical_prediction")
        for candidate in candidates:
            text = candidate.get("canonical_prediction") or candidate.get("prediction") or candidate.get("raw_text")
            if not text:
                continue
            item = dict(candidate)
            item["is_chosen"] = text == chosen
            item["is_rejected"] = text != chosen
            item["label"] = 1.0 if item["is_chosen"] else 0.0
            yield item


def fit_linear_policy(samples: list[dict], l2: float) -> dict:
    if not samples:
        raise ValueError("No training samples found")

    X = np.stack([feature_vector(sample) for sample in samples], axis=0)
    y = np.asarray([candidate_label(sample) for sample in samples], dtype=np.float32)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    Xn = (X - mean) / std
    Xn = np.concatenate([np.ones((Xn.shape[0], 1), dtype=np.float32), Xn], axis=1)

    reg = np.eye(Xn.shape[1], dtype=np.float32) * float(l2)
    reg[0, 0] = 0.0
    w = np.linalg.solve(Xn.T @ Xn + reg, Xn.T @ y)

    bias = float(w[0])
    weights = w[1:]
    return make_policy(weights=weights, mean=mean, std=std, bias=bias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    args = parser.parse_args()

    samples = []
    for path_text in args.input_jsonl:
        path = Path(path_text).resolve()
        samples.extend(list(collect_rows(path)))

    policy = fit_linear_policy(samples, l2=args.l2)
    save_policy(Path(args.output_json).resolve(), policy)
    print(json.dumps({"samples": len(samples), "output": str(Path(args.output_json).resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
