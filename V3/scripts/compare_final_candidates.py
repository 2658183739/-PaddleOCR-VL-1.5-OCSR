import argparse
import json
from pathlib import Path
from statistics import mean


BENCHMARKS = ("legacy_core_dev", "legacy_region_dev")


def load_candidate(label: str, root: Path):
    panels = {}
    for benchmark in BENCHMARKS:
        with (root / benchmark / "report.json").open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        panels[benchmark] = {
            "canonical_exact": float(report["accuracy"]["canonical_exact_match_accuracy"]),
            "valid_rate": float(report["accuracy"]["valid_smiles_rate"]),
        }
    return {
        "label": label,
        "root": str(root),
        "panels": panels,
        "macro_exact": mean(panel["canonical_exact"] for panel in panels.values()),
        "min_valid_rate": min(panel["valid_rate"] for panel in panels.values()),
    }


def choose_winner(baseline, candidate, regression_tolerance, minimum_improvement):
    no_panel_regression = all(
        candidate["panels"][benchmark]["canonical_exact"]
        >= baseline["panels"][benchmark]["canonical_exact"] - regression_tolerance
        for benchmark in BENCHMARKS
    )
    validity_ok = (
        candidate["min_valid_rate"]
        >= baseline["min_valid_rate"] - regression_tolerance
    )
    macro_delta = candidate["macro_exact"] - baseline["macro_exact"]
    meaningful_improvement = macro_delta >= minimum_improvement
    winner = candidate if no_panel_regression and validity_ok and meaningful_improvement else baseline
    return {
        "candidate_no_panel_regression": no_panel_regression,
        "candidate_validity_ok": validity_ok,
        "candidate_macro_delta": macro_delta,
        "candidate_meaningful_improvement": meaningful_improvement,
        "winner": winner,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-label", default="final")
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--candidate-label", default="hard_replay")
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--regression-tolerance", type=float, default=0.005)
    parser.add_argument("--minimum-improvement", type=float, default=0.005)
    args = parser.parse_args()

    baseline = load_candidate(args.baseline_label, Path(args.baseline_root).resolve())
    candidate = load_candidate(args.candidate_label, Path(args.candidate_root).resolve())
    decision = choose_winner(
        baseline,
        candidate,
        regression_tolerance=args.regression_tolerance,
        minimum_improvement=args.minimum_improvement,
    )

    payload = {
        "regression_tolerance": args.regression_tolerance,
        "minimum_improvement": args.minimum_improvement,
        "baseline": baseline,
        "candidate": candidate,
        **decision,
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision["winner"], ensure_ascii=False))


if __name__ == "__main__":
    main()
