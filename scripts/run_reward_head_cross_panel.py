#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def run(cmd, log_path: Path | None = None):
    if log_path is None:
        subprocess.run(cmd, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stdout)
        raise subprocess.CalledProcessError(completed.returncode, cmd)


def read_report(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "total": data.get("total"),
        "canonical_exact": data.get("accuracy", {}).get("canonical_exact_match_accuracy"),
        "raw_exact": data.get("accuracy", {}).get("raw_exact_match_accuracy"),
        "valid_smiles_rate": data.get("accuracy", {}).get("valid_smiles_rate"),
        "mean_tanimoto": data.get("similarity", {}).get("mean_fingerprint_tanimoto"),
        "mean_edit_similarity": data.get("similarity", {}).get("mean_normalized_edit_similarity"),
        "by_source_exact": {
            key: value.get("canonical_exact_match_accuracy")
            for key, value in data.get("by_group", {}).get("source", {}).items()
        },
        "by_eval_panel_exact": {
            key: value.get("canonical_exact_match_accuracy")
            for key, value in data.get("by_group", {}).get("eval_panel", {}).items()
        },
    }


def mean_std(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-prediction-jsonl", required=True)
    parser.add_argument("--train-labels-jsonl", required=True)
    parser.add_argument("--eval-prediction-jsonl", required=True)
    parser.add_argument("--eval-labels-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="20260627,20260628,20260629")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-negatives-per-positive", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--fallback-mode", default="chem_light")
    parser.add_argument("--margin-grid", default="0,0.05,0.1,0.25,0.5,0.75,1,1.5,2")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_candidate_reward_head.py"
    apply_script = script_dir / "apply_candidate_reward_head.py"
    eval_script = script_dir / "evaluate_ocsr_predictions_detailed.py"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    baseline_report = out_dir / "eval_baseline_selected.json"
    run(
        [
            sys.executable,
            str(eval_script),
            "--benchmark-jsonl",
            args.eval_labels_jsonl,
            "--prediction-jsonl",
            args.eval_prediction_jsonl,
            "--report-json",
            str(baseline_report),
            "--details-jsonl",
            str(out_dir / "eval_baseline_selected_details.jsonl"),
            "--group-fields",
            "source,difficulty,task_type,eval_panel",
        ],
        out_dir / "eval_baseline_selected.log",
    )
    baseline = read_report(baseline_report)

    runs = []
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        train_dir = seed_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                str(train_script),
                "--prediction-jsonl",
                args.train_prediction_jsonl,
                "--labels-jsonl",
                args.train_labels_jsonl,
                "--output-dir",
                str(train_dir),
                "--train-fraction",
                str(args.train_fraction),
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--hidden-dim",
                str(args.hidden_dim),
                "--dropout",
                str(args.dropout),
                "--max-negatives-per-positive",
                str(args.max_negatives_per_positive),
                "--fallback-mode",
                args.fallback_mode,
                "--margin-grid",
                args.margin_grid,
                "--log-every",
                str(max(1, args.epochs // 3)),
            ],
            seed_dir / "train.log",
        )

        pred_path = seed_dir / "pred_reward_head_eval.jsonl"
        run(
            [
                sys.executable,
                str(apply_script),
                "--checkpoint",
                str(train_dir / "reward_head.pt"),
                "--prediction-jsonl",
                args.eval_prediction_jsonl,
                "--labels-jsonl",
                args.eval_labels_jsonl,
                "--output-jsonl",
                str(pred_path),
            ],
            seed_dir / "apply.log",
        )

        eval_report_path = seed_dir / "eval_reward_head.json"
        run(
            [
                sys.executable,
                str(eval_script),
                "--benchmark-jsonl",
                args.eval_labels_jsonl,
                "--prediction-jsonl",
                str(pred_path),
                "--report-json",
                str(eval_report_path),
                "--details-jsonl",
                str(seed_dir / "eval_reward_head_details.jsonl"),
                "--group-fields",
                "source,difficulty,task_type,eval_panel",
            ],
            seed_dir / "eval.log",
        )

        train_report = json.loads((train_dir / "reward_head_report.json").read_text(encoding="utf-8"))
        eval_metrics = read_report(eval_report_path)
        runs.append(
            {
                "seed": seed,
                "best_margin": train_report.get("best_margin"),
                "train_sample_count": train_report.get("train_sample_count"),
                "dev_sample_count": train_report.get("dev_sample_count"),
                "train_pair_count": train_report.get("train_pair_count"),
                "dev_pair_count": train_report.get("dev_pair_count"),
                "train_dev_policy_exact": train_report.get("dev_eval", {}).get("metrics", {}).get("policy_exact"),
                "eval": eval_metrics,
                "delta_vs_baseline_exact": eval_metrics["canonical_exact"] - baseline["canonical_exact"],
                "delta_vs_baseline_tanimoto": eval_metrics["mean_tanimoto"] - baseline["mean_tanimoto"],
            }
        )

    summary = {
        "train_prediction_jsonl": args.train_prediction_jsonl,
        "train_labels_jsonl": args.train_labels_jsonl,
        "eval_prediction_jsonl": args.eval_prediction_jsonl,
        "eval_labels_jsonl": args.eval_labels_jsonl,
        "output_dir": str(out_dir),
        "seeds": seeds,
        "baseline": baseline,
        "runs": runs,
        "aggregate": {
            "canonical_exact": mean_std([item["eval"]["canonical_exact"] for item in runs]),
            "mean_tanimoto": mean_std([item["eval"]["mean_tanimoto"] for item in runs]),
            "delta_exact": mean_std([item["delta_vs_baseline_exact"] for item in runs]),
            "delta_tanimoto": mean_std([item["delta_vs_baseline_tanimoto"] for item in runs]),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
