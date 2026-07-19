import argparse
import json
from pathlib import Path
from statistics import mean


BENCHMARKS = ("legacy_core_dev", "legacy_region_dev")


def load_report(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_step(path: Path):
    return int(path.name.split("-", 1)[1])


def collect_checkpoints(phase_root: Path):
    results = []
    for checkpoint in sorted(phase_root.glob("checkpoint-*"), key=checkpoint_step):
        panels = {}
        for benchmark in BENCHMARKS:
            report_path = checkpoint / benchmark / "report.json"
            if not report_path.exists():
                break
            report = load_report(report_path)
            panels[benchmark] = {
                "canonical_exact": float(report["accuracy"]["canonical_exact_match_accuracy"]),
                "valid_rate": float(report["accuracy"]["valid_smiles_rate"]),
            }
        if len(panels) != len(BENCHMARKS):
            continue
        results.append(
            {
                "checkpoint": checkpoint.name,
                "step": checkpoint_step(checkpoint),
                "panels": panels,
                "macro_exact": mean(panel["canonical_exact"] for panel in panels.values()),
                "min_valid_rate": min(panel["valid_rate"] for panel in panels.values()),
            }
        )
    if not results:
        raise FileNotFoundError(f"No completely evaluated checkpoints under {phase_root}")
    return results


def select_checkpoint(results, validity_floor: float, tie_tolerance: float):
    eligible = [row for row in results if row["min_valid_rate"] >= validity_floor]
    if not eligible:
        raise RuntimeError("No checkpoint passed the validity floor")
    best_score = max(row["macro_exact"] for row in eligible)
    near_best = [row for row in eligible if row["macro_exact"] >= best_score - tie_tolerance]
    return min(near_best, key=lambda row: row["step"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--validity-floor", type=float, default=0.95)
    parser.add_argument("--tie-tolerance", type=float, default=0.005)
    args = parser.parse_args()

    phase_root = Path(args.eval_root).resolve() / args.phase
    results = collect_checkpoints(phase_root)
    winner = select_checkpoint(results, args.validity_floor, args.tie_tolerance)
    payload = {
        "phase": args.phase,
        "validity_floor": args.validity_floor,
        "tie_tolerance": args.tie_tolerance,
        "checkpoints": results,
        "winner": winner,
    }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(winner, ensure_ascii=False))


if __name__ == "__main__":
    main()
