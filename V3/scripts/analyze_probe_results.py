import argparse
import json
from pathlib import Path
from statistics import mean


CORE_RUNS = {
    "00": ("data_00_s1", "data_00_s2"),
    "10": ("data_10_s1", "data_10_s2"),
    "01": ("data_01_s1", "data_01_s2"),
    "11": ("data_11_s1", "data_11_s2"),
}

DATASET_BY_CONDITION = {
    "00": "./V3/data/sft_materialized/train_v3_a_control.jsonl",
    "10": "./V3/data/sft_materialized/train_v3_d_wild_only.jsonl",
    "01": "./V3/data/sft_materialized/train_v3_e_aug_only.jsonl",
    "11": "./V3/data/sft_materialized/train_v3_b_recommended.jsonl",
    "dose2": "./V3/data/sft_materialized/train_v3_c_real_heavy.jsonl",
}

BENCHMARKS = ("legacy_core_dev", "legacy_region_dev")
DOSE_RUN = "aug_dose2_s1"
WARMSTART_RUN = "warmstart_control_s1"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_checkpoint(run_root: Path):
    checkpoints = []
    for child in run_root.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            try:
                checkpoints.append((int(child.name.split("-", 1)[1]), child))
            except ValueError:
                continue
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint directory under {run_root}")
    return max(checkpoints)[1]


def read_run(eval_root: Path, run_id: str):
    checkpoint = latest_checkpoint(eval_root / run_id)
    panels = {}
    for benchmark in BENCHMARKS:
        report = load_json(checkpoint / benchmark / "report.json")
        panels[benchmark] = {
            "canonical_exact": float(report["accuracy"]["canonical_exact_match_accuracy"]),
            "valid_rate": float(report["accuracy"]["valid_smiles_rate"]),
        }
    return {
        "run_id": run_id,
        "checkpoint": checkpoint.name,
        "panels": panels,
        "macro_exact": mean(panel["canonical_exact"] for panel in panels.values()),
        "min_valid_rate": min(panel["valid_rate"] for panel in panels.values()),
    }


def analyze(eval_root: Path, validity_tolerance: float, dose_selection_margin: float):
    runs = {}
    conditions = {}
    for condition, run_ids in CORE_RUNS.items():
        condition_runs = [read_run(eval_root, run_id) for run_id in run_ids]
        for run in condition_runs:
            runs[run["run_id"]] = run
        conditions[condition] = {
            "run_ids": list(run_ids),
            "mean_macro_exact": mean(run["macro_exact"] for run in condition_runs),
            "seed_range": max(run["macro_exact"] for run in condition_runs)
            - min(run["macro_exact"] for run in condition_runs),
            "min_valid_rate": min(run["min_valid_rate"] for run in condition_runs),
        }

    control_validity = conditions["00"]["min_valid_rate"]
    validity_floor = control_validity - validity_tolerance
    eligible = [
        condition
        for condition, result in conditions.items()
        if result["min_valid_rate"] >= validity_floor
    ]
    if not eligible:
        raise RuntimeError("No condition passed the validity regression gate")

    core_winner = max(
        eligible,
        key=lambda condition: (conditions[condition]["mean_macro_exact"], -int(condition)),
    )
    effects = {
        "wild_main_effect": mean(
            [conditions["10"]["mean_macro_exact"], conditions["11"]["mean_macro_exact"]]
        )
        - mean([conditions["00"]["mean_macro_exact"], conditions["01"]["mean_macro_exact"]]),
        "augmentation_main_effect": mean(
            [conditions["01"]["mean_macro_exact"], conditions["11"]["mean_macro_exact"]]
        )
        - mean([conditions["00"]["mean_macro_exact"], conditions["10"]["mean_macro_exact"]]),
        "interaction": conditions["11"]["mean_macro_exact"]
        - conditions["10"]["mean_macro_exact"]
        - conditions["01"]["mean_macro_exact"]
        + conditions["00"]["mean_macro_exact"],
    }
    diagnostics = {}
    selected_condition = core_winner

    dose_root = eval_root / DOSE_RUN
    if dose_root.exists():
        dose = read_run(eval_root, DOSE_RUN)
        diagnostics["augmentation_dose2"] = {
            **dose,
            "delta_vs_11_s1": dose["macro_exact"] - runs["data_11_s1"]["macro_exact"],
            "passes_validity_gate": dose["min_valid_rate"] >= validity_floor,
            "selection_margin": dose_selection_margin,
        }
        if (
            effects["augmentation_main_effect"] > 0
            and dose["min_valid_rate"] >= validity_floor
            and dose["macro_exact"]
            > conditions[core_winner]["mean_macro_exact"] + dose_selection_margin
        ):
            selected_condition = "dose2"

    warmstart_root = eval_root / WARMSTART_RUN
    if warmstart_root.exists():
        warmstart = read_run(eval_root, WARMSTART_RUN)
        diagnostics["warmstart"] = {
            **warmstart,
            "continuation_run_id": "data_11_s1",
            "continuation_minus_base15": runs["data_11_s1"]["macro_exact"]
            - warmstart["macro_exact"],
        }

    if selected_condition == "dose2":
        selected_metrics = {
            "run_ids": [DOSE_RUN],
            "mean_macro_exact": diagnostics["augmentation_dose2"]["macro_exact"],
            "seed_range": None,
            "min_valid_rate": diagnostics["augmentation_dose2"]["min_valid_rate"],
            "selection_note": "single_seed_dose2_selected_with_positive_main_effect_and_margin",
        }
    else:
        selected_metrics = conditions[selected_condition]

    return {
        "runs": runs,
        "conditions": conditions,
        "effects": effects,
        "diagnostics": diagnostics,
        "validity_floor": validity_floor,
        "winner": {
            "condition": selected_condition,
            "dataset_path": DATASET_BY_CONDITION[selected_condition],
            **selected_metrics,
        },
    }


def write_markdown(result, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# H800 Probe Analysis\n\n")
        handle.write("| Condition | Mean macro exact | Seed range | Min valid rate |\n")
        handle.write("|---|---:|---:|---:|\n")
        for condition in ("00", "10", "01", "11"):
            row = result["conditions"][condition]
            handle.write(
                f"| {condition} | {row['mean_macro_exact']:.6f} | "
                f"{row['seed_range']:.6f} | {row['min_valid_rate']:.6f} |\n"
            )
        handle.write("\n")
        handle.write(f"Selected condition: `{result['winner']['condition']}`\n\n")
        handle.write(f"Final dataset: `{result['winner']['dataset_path']}`\n\n")
        for name, value in result["effects"].items():
            handle.write(f"- {name}: {value:.6f}\n")
        if result["diagnostics"]:
            handle.write("\n## Diagnostics\n\n")
        dose = result["diagnostics"].get("augmentation_dose2")
        if dose:
            handle.write(
                f"- dose2 macro exact: {dose['macro_exact']:.6f}; "
                f"delta vs 11 seed1: {dose['delta_vs_11_s1']:.6f}; "
                f"validity gate: {dose['passes_validity_gate']}\n"
            )
        warmstart = result["diagnostics"].get("warmstart")
        if warmstart:
            handle.write(
                f"- continuation minus base-1.5 warm-start: "
                f"{warmstart['continuation_minus_base15']:.6f}\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--validity-tolerance", type=float, default=0.005)
    parser.add_argument("--dose-selection-margin", type=float, default=0.005)
    args = parser.parse_args()

    result = analyze(
        Path(args.eval_root).resolve(),
        args.validity_tolerance,
        args.dose_selection_margin,
    )
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, Path(args.output_md).resolve())
    print(json.dumps(result["winner"], ensure_ascii=False))


if __name__ == "__main__":
    main()
