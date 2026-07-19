import argparse
import json
from pathlib import Path


def run_and_panel(details_path: str):
    parts = Path(details_path.replace("\\", "/")).parts
    marker = parts.index("eval_runs_probes")
    return parts[marker + 1], parts[marker + 3]


def collect(input_dir: Path):
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline_run, baseline_panel = run_and_panel(payload["baseline"])
        candidate_run, candidate_panel = run_and_panel(payload["candidate"])
        if baseline_panel != candidate_panel:
            raise ValueError(f"Panel mismatch in {path}")
        exact = payload["canonical_exact"]
        valid = payload["valid_smiles"]
        rows.append(
            {
                "baseline_run": baseline_run,
                "candidate_run": candidate_run,
                "panel": baseline_panel,
                "n_images": payload["n"],
                "independent_units": exact["independent_units"],
                "resampling_unit": exact["resampling_unit"],
                "exact_delta": exact["delta_mean"],
                "exact_ci95_low": exact["ci95_low"],
                "exact_ci95_high": exact["ci95_high"],
                "probability_exact_delta_gt_zero": exact["probability_delta_gt_zero"],
                "valid_delta": valid["delta_mean"],
                "selection_gate_pass": payload["selection_gate"]["pass"],
                "source_json": str(path),
            }
        )
    return rows


def render_markdown(rows):
    lines = [
        "# Probe paired bootstrap comparisons",
        "",
        "> Each row uses 10,000 paired bootstrap resamples over `structure_id` clusters.",
        "> These are per-seed development comparisons, not a substitute for 4+ seed confirmatory replication.",
        "",
        "| baseline | candidate | panel | images | units | exact delta | 95% CI | valid delta | gate |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['baseline_run']} | {row['candidate_run']} | {row['panel']} | "
            f"{row['n_images']} | {row['independent_units']} | {row['exact_delta']:.6f} | "
            f"[{row['exact_ci95_low']:.6f}, {row['exact_ci95_high']:.6f}] | "
            f"{row['valid_delta']:.6f} | {row['selection_gate_pass']} |"
        )
    lines.extend(
        [
            "",
            "A CI crossing zero means the observed per-seed difference is compatible with both improvement and regression at this development-sample resolution.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="V3/evidence/probe_paired")
    parser.add_argument("--output-json", default="V3/evidence/probe_paired_summary.json")
    parser.add_argument("--output-md", default="V3/evidence/probe_paired_summary.md")
    args = parser.parse_args()

    rows = collect(Path(args.input_dir))
    if not rows:
        raise FileNotFoundError(f"No comparison JSON files found under {args.input_dir}")
    Path(args.output_json).write_text(
        json.dumps({"comparisons": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(args.output_md).write_text(render_markdown(rows), encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
