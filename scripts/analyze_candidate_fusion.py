import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


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


def canonicalize_smiles(smiles: str):
    text = str(smiles or "").strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def get_ground_truth_smiles(row: dict):
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, dict) and ground_truth.get("smiles"):
        return str(ground_truth["smiles"]).strip()
    for key in ("canonical_smiles", "smiles", "label_summary"):
        if row.get(key):
            return str(row[key]).strip()
    return ""


def parse_run_arg(value: str):
    if "=" not in value:
        raise ValueError(f"Run must be NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Run name is empty in: {value}")
    return name, Path(path)


def load_prediction_run(path: Path):
    rows = {}
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if sample_id:
            rows[str(sample_id)] = row
    return rows


def normalize_candidate(run_name: str, sample_id: str, row: dict, candidate: dict, source: str):
    prediction = str(candidate.get("prediction", "") or "").strip()
    canonical = candidate.get("canonical_prediction") or canonicalize_smiles(prediction)
    score = candidate.get("generation_score")
    if score is None or (isinstance(score, float) and math.isnan(score)):
        score_value = -1_000_000.0
    else:
        try:
            score_value = float(score)
        except Exception:
            score_value = -1_000_000.0
    penalty = candidate.get("smiles_structure_penalty", 1000)
    try:
        penalty_value = float(penalty)
    except Exception:
        penalty_value = 1000.0
    return {
        "id": sample_id,
        "run": run_name,
        "source": source,
        "prediction": prediction,
        "canonical_prediction": canonical,
        "generation_score": score,
        "_score_value": score_value,
        "smiles_structure_penalty": penalty,
        "_penalty_value": penalty_value,
        "prompt_index": candidate.get("prompt_index"),
        "tta_name": candidate.get("tta_name"),
        "tta_index": candidate.get("tta_index"),
        "generation_index": candidate.get("generation_index"),
        "row_vote_count": row.get("vote_count"),
    }


def collect_candidates(run_name: str, sample_id: str, row: dict):
    candidates = []
    selected_candidate = {
        "prediction": row.get("prediction", ""),
        "canonical_prediction": row.get("canonical_prediction"),
        "generation_score": row.get("generation_score"),
        "smiles_structure_penalty": row.get("smiles_structure_penalty", 1000),
        "prompt_index": None,
        "tta_name": "selected",
        "tta_index": None,
        "generation_index": None,
    }
    candidates.append(normalize_candidate(run_name, sample_id, row, selected_candidate, "selected"))
    for candidate in row.get("candidates") or []:
        candidates.append(normalize_candidate(run_name, sample_id, row, candidate, "candidate"))
    return candidates


def _candidate_text(candidate: dict):
    return str(candidate.get("prediction", "") or "").strip()


def _candidate_key_value(candidate: dict):
    return candidate.get("canonical_prediction") or _candidate_text(candidate)


def _valid_candidates(candidates):
    non_empty = [item for item in candidates if _candidate_text(item)]
    valid = [item for item in non_empty if item["canonical_prediction"]]
    return non_empty, valid


def choose_vote_score_candidate(candidates):
    non_empty = [item for item in candidates if item["prediction"]]
    valid = [item for item in non_empty if item["canonical_prediction"]]
    if valid:
        canonical_counts = Counter(item["canonical_prediction"] for item in valid)
        canonical_run_counts = defaultdict(set)
        for item in valid:
            canonical_run_counts[item["canonical_prediction"]].add(item["run"])

        def valid_key(item):
            canonical = item["canonical_prediction"]
            return (
                canonical_counts[canonical],
                len(canonical_run_counts[canonical]),
                item["_score_value"],
                -item["_penalty_value"],
                1 if item["source"] == "selected" else 0,
            )

        best = max(valid, key=valid_key)
        canonical = best["canonical_prediction"]
        return dict(
            best,
            fusion_reason="valid_cross_run_vote_score",
            fusion_vote_count=canonical_counts[canonical],
            fusion_distinct_runs=len(canonical_run_counts[canonical]),
        )

    if non_empty:
        raw_counts = Counter(item["prediction"] for item in non_empty)

        def raw_key(item):
            return (
                raw_counts[item["prediction"]],
                -item["_penalty_value"],
                item["_score_value"],
                1 if item["source"] == "selected" else 0,
            )

        best = max(non_empty, key=raw_key)
        return dict(
            best,
            fusion_reason="non_empty_penalty_score",
            fusion_vote_count=raw_counts[best["prediction"]],
            fusion_distinct_runs=len({item["run"] for item in non_empty if item["prediction"] == best["prediction"]}),
        )

    return {
        "prediction": "",
        "canonical_prediction": None,
        "run": "",
        "source": "",
        "generation_score": None,
        "smiles_structure_penalty": None,
        "fusion_reason": "empty_fallback",
        "fusion_vote_count": 0,
        "fusion_distinct_runs": 0,
    }


def choose_score_candidate(candidates):
    non_empty, valid = _valid_candidates(candidates)
    pool = valid or non_empty
    if not pool:
        return choose_vote_score_candidate(candidates)
    best = max(
        pool,
        key=lambda item: (
            item["_score_value"],
            -item["_penalty_value"],
            1 if item["source"] == "selected" else 0,
        ),
    )
    return dict(
        best,
        fusion_reason="valid_generation_score",
        fusion_vote_count=1,
        fusion_distinct_runs=1,
    )


def choose_penalty_score_candidate(candidates):
    non_empty, valid = _valid_candidates(candidates)
    pool = valid or non_empty
    if not pool:
        return choose_vote_score_candidate(candidates)
    best = max(
        pool,
        key=lambda item: (
            -item["_penalty_value"],
            item["_score_value"],
            1 if item["source"] == "selected" else 0,
        ),
    )
    return dict(
        best,
        fusion_reason="valid_penalty_score",
        fusion_vote_count=1,
        fusion_distinct_runs=1,
    )


def choose_domain_selected_candidate(candidates, target):
    source = str(target.get("source", ""))
    preferred_run = "continue_sft" if source in {"real_world_photo_scan", "uspto"} else "baseline"
    selected = [
        item
        for item in candidates
        if item["source"] == "selected" and item["run"] == preferred_run and _candidate_text(item)
    ]
    if selected:
        best = selected[0]
        return dict(
            best,
            fusion_reason=f"domain_selected_{preferred_run}",
            fusion_vote_count=1,
            fusion_distinct_runs=1,
        )
    return choose_score_candidate(candidates)


def choose_source_weighted_candidate(candidates, target):
    non_empty, valid = _valid_candidates(candidates)
    pool = valid or non_empty
    if not pool:
        return choose_vote_score_candidate(candidates)

    source = str(target.get("source", ""))
    counts = Counter(_candidate_key_value(item) for item in pool)
    run_counts = defaultdict(set)
    for item in pool:
        run_counts[_candidate_key_value(item)].add(item["run"])

    def key(item):
        value = _candidate_key_value(item)
        run_bonus = 0.0
        if source == "uspto" and item["run"] == "continue_sft":
            run_bonus += 0.8
        if source in {"edu_exam", "decimer_handdrawn"} and item["run"] == "baseline":
            run_bonus += 0.5
        if source == "real_world_photo_scan" and item["run"] == "continue_sft":
            run_bonus += 0.25
        selected_bonus = 0.35 if item["source"] == "selected" else 0.0
        return (
            counts[value] + run_bonus + selected_bonus,
            len(run_counts[value]),
            -item["_penalty_value"],
            item["_score_value"],
        )

    best = max(pool, key=key)
    return dict(
        best,
        fusion_reason="source_weighted_vote",
        fusion_vote_count=counts[_candidate_key_value(best)],
        fusion_distinct_runs=len(run_counts[_candidate_key_value(best)]),
    )


def choose_fused_candidate(candidates, target=None, strategy="vote_score"):
    if strategy == "vote_score":
        return choose_vote_score_candidate(candidates)
    if strategy == "score":
        return choose_score_candidate(candidates)
    if strategy == "penalty_score":
        return choose_penalty_score_candidate(candidates)
    if strategy == "domain_selected":
        return choose_domain_selected_candidate(candidates, target or {})
    if strategy == "source_weighted":
        return choose_source_weighted_candidate(candidates, target or {})
    raise ValueError(f"Unknown fusion strategy: {strategy}")


def init_accumulator():
    return {
        "total": 0,
        "selected_correct": 0,
        "candidate_oracle_correct": 0,
        "selected_valid": 0,
    }


def add_to_acc(acc, selected_correct: bool, candidate_oracle: bool, selected_valid: bool):
    acc["total"] += 1
    acc["selected_correct"] += int(selected_correct)
    acc["candidate_oracle_correct"] += int(candidate_oracle)
    acc["selected_valid"] += int(selected_valid)


def finalize_acc(acc):
    total = acc["total"]
    if total == 0:
        return {
            "total": 0,
            "selected_exact": 0.0,
            "candidate_oracle_exact": 0.0,
            "selected_valid_rate": 0.0,
        }
    return {
        "total": total,
        "selected_exact": acc["selected_correct"] / total,
        "candidate_oracle_exact": acc["candidate_oracle_correct"] / total,
        "selected_valid_rate": acc["selected_valid"] / total,
        "candidate_oracle_gain_over_selected": (acc["candidate_oracle_correct"] - acc["selected_correct"]) / total,
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-jsonl", required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Prediction run in NAME=PATH format. Pass multiple times.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-fields", default="source,difficulty,task_type")
    parser.add_argument("--fused-jsonl", default="")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--details-jsonl", default="")
    parser.add_argument(
        "--strategies",
        default="vote_score,score,penalty_score,domain_selected,source_weighted",
        help="Comma-separated fusion strategies to write and summarize.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    benchmark_path = Path(args.benchmark_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for value in args.run:
        name, path = parse_run_arg(value)
        runs.append((name, load_prediction_run(path)))

    benchmark = {str(row["id"]): row for row in read_jsonl(benchmark_path)}
    group_fields = [field.strip() for field in args.group_fields.split(",") if field.strip()]

    run_acc = {name: init_accumulator() for name, _ in runs}
    run_groups = {name: {field: defaultdict(init_accumulator) for field in group_fields} for name, _ in runs}
    combined_acc = init_accumulator()
    combined_groups = {field: defaultdict(init_accumulator) for field in group_fields}
    fused_acc = init_accumulator()
    fused_groups = {field: defaultdict(init_accumulator) for field in group_fields}
    strategy_names = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if "vote_score" not in strategy_names:
        strategy_names.insert(0, "vote_score")
    strategy_acc = {name: init_accumulator() for name in strategy_names}
    strategy_groups = {
        name: {field: defaultdict(init_accumulator) for field in group_fields}
        for name in strategy_names
    }
    strategy_rows = {name: [] for name in strategy_names}

    fused_rows = []
    detail_rows = []

    for sample_id, target in benchmark.items():
        gt_canonical = canonicalize_smiles(get_ground_truth_smiles(target))
        all_candidates = []
        per_run = {}

        for run_name, predictions in runs:
            row = predictions.get(sample_id, {})
            selected_canonical = row.get("canonical_prediction") or canonicalize_smiles(row.get("prediction", ""))
            candidates = collect_candidates(run_name, sample_id, row) if row else []
            all_candidates.extend(candidates)
            candidate_oracle = any(
                item.get("canonical_prediction") == gt_canonical for item in candidates if gt_canonical
            )
            selected_correct = bool(gt_canonical and selected_canonical == gt_canonical)
            selected_valid = bool(selected_canonical)
            add_to_acc(run_acc[run_name], selected_correct, candidate_oracle, selected_valid)
            for field in group_fields:
                group_name = str(target.get(field, "unknown"))
                add_to_acc(run_groups[run_name][field][group_name], selected_correct, candidate_oracle, selected_valid)
            per_run[run_name] = {
                "prediction": row.get("prediction", ""),
                "canonical_prediction": selected_canonical,
                "selected_correct": selected_correct,
                "selected_valid": selected_valid,
                "candidate_oracle_correct": candidate_oracle,
                "candidate_count": len(candidates),
            }

        combined_candidate_oracle = any(
            item.get("canonical_prediction") == gt_canonical for item in all_candidates if gt_canonical
        )
        selected_union_correct = any(value["selected_correct"] for value in per_run.values())
        add_to_acc(combined_acc, selected_union_correct, combined_candidate_oracle, True)
        for field in group_fields:
            group_name = str(target.get(field, "unknown"))
            add_to_acc(combined_groups[field][group_name], selected_union_correct, combined_candidate_oracle, True)

        strategy_details = {}
        for strategy_name in strategy_names:
            fused_strategy = choose_fused_candidate(all_candidates, target, strategy_name)
            fused_strategy_canonical = fused_strategy.get("canonical_prediction")
            fused_strategy_correct = bool(gt_canonical and fused_strategy_canonical == gt_canonical)
            fused_strategy_valid = bool(fused_strategy_canonical)
            add_to_acc(
                strategy_acc[strategy_name],
                fused_strategy_correct,
                fused_strategy_correct,
                fused_strategy_valid,
            )
            for field in group_fields:
                group_name = str(target.get(field, "unknown"))
                add_to_acc(
                    strategy_groups[strategy_name][field][group_name],
                    fused_strategy_correct,
                    fused_strategy_correct,
                    fused_strategy_valid,
                )
            strategy_rows[strategy_name].append(
                {
                    "id": sample_id,
                    "image_path": target.get("image") or target.get("image_path"),
                    "prediction": fused_strategy.get("prediction", ""),
                    "canonical_prediction": fused_strategy_canonical,
                    "fusion_strategy": strategy_name,
                    "fusion_reason": fused_strategy.get("fusion_reason"),
                    "fusion_vote_count": fused_strategy.get("fusion_vote_count"),
                    "fusion_distinct_runs": fused_strategy.get("fusion_distinct_runs"),
                    "fusion_source_run": fused_strategy.get("run"),
                    "fusion_source": fused_strategy.get("source"),
                    "generation_score": fused_strategy.get("generation_score"),
                    "smiles_structure_penalty": fused_strategy.get("smiles_structure_penalty"),
                }
            )
            strategy_details[strategy_name] = {
                "prediction": fused_strategy.get("prediction", ""),
                "canonical_prediction": fused_strategy_canonical,
                "correct": fused_strategy_correct,
                "valid": fused_strategy_valid,
                "reason": fused_strategy.get("fusion_reason"),
                "source_run": fused_strategy.get("run"),
            }

        fused = choose_fused_candidate(all_candidates, target, "vote_score")
        fused_canonical = fused.get("canonical_prediction")
        fused_correct = bool(gt_canonical and fused_canonical == gt_canonical)
        fused_valid = bool(fused_canonical)
        add_to_acc(fused_acc, fused_correct, fused_correct, fused_valid)
        for field in group_fields:
            group_name = str(target.get(field, "unknown"))
            add_to_acc(fused_groups[field][group_name], fused_correct, fused_correct, fused_valid)

        fused_rows.append(
            {
                "id": sample_id,
                "image_path": target.get("image") or target.get("image_path"),
                "prediction": fused.get("prediction", ""),
                "canonical_prediction": fused_canonical,
                "fusion_reason": fused.get("fusion_reason"),
                "fusion_vote_count": fused.get("fusion_vote_count"),
                "fusion_distinct_runs": fused.get("fusion_distinct_runs"),
                "fusion_source_run": fused.get("run"),
                "fusion_source": fused.get("source"),
                "generation_score": fused.get("generation_score"),
                "smiles_structure_penalty": fused.get("smiles_structure_penalty"),
            }
        )
        detail_rows.append(
            {
                "id": sample_id,
                "source": target.get("source"),
                "difficulty": target.get("difficulty"),
                "task_type": target.get("task_type"),
                "image": target.get("image") or target.get("image_path"),
                "ground_truth_canonical": gt_canonical,
                "runs": per_run,
                "selected_union_correct": selected_union_correct,
                "combined_candidate_oracle_correct": combined_candidate_oracle,
                "fused_prediction": fused.get("prediction", ""),
                "fused_canonical_prediction": fused_canonical,
                "fused_correct": fused_correct,
                "fused_reason": fused.get("fusion_reason"),
                "fused_source_run": fused.get("run"),
                "fusion_strategies": strategy_details,
            }
        )

    summary = {
        "total": len(benchmark),
        "runs": {},
        "combined": {
            "selected_union_oracle": finalize_acc(combined_acc),
            "candidate_pool_oracle": finalize_acc(combined_acc)["candidate_oracle_exact"],
            "by_group": {
                field: {group: finalize_acc(acc) for group, acc in groups.items()}
                for field, groups in combined_groups.items()
            },
        },
        "fused": {
            "metrics": finalize_acc(fused_acc),
            "by_group": {
                field: {group: finalize_acc(acc) for group, acc in groups.items()}
                for field, groups in fused_groups.items()
            },
        },
        "fused_strategies": {
            strategy_name: {
                "metrics": finalize_acc(strategy_acc[strategy_name]),
                "by_group": {
                    field: {group: finalize_acc(acc) for group, acc in groups.items()}
                    for field, groups in strategy_groups[strategy_name].items()
                },
            }
            for strategy_name in strategy_names
        },
    }
    for run_name, _ in runs:
        summary["runs"][run_name] = {
            "metrics": finalize_acc(run_acc[run_name]),
            "by_group": {
                field: {group: finalize_acc(acc) for group, acc in groups.items()}
                for field, groups in run_groups[run_name].items()
            },
        }

    summary_path = Path(args.summary_json) if args.summary_json else output_dir / "candidate_fusion_summary.json"
    details_path = Path(args.details_jsonl) if args.details_jsonl else output_dir / "candidate_fusion_details.jsonl"
    fused_path = Path(args.fused_jsonl) if args.fused_jsonl else output_dir / "pred_fused_heuristic.jsonl"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(details_path, detail_rows)
    write_jsonl(fused_path, fused_rows)
    for strategy_name, rows in strategy_rows.items():
        strategy_path = output_dir / f"pred_fused_{strategy_name}.jsonl"
        write_jsonl(strategy_path, rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_json={summary_path}")
    print(f"details_jsonl={details_path}")
    print(f"fused_jsonl={fused_path}")


if __name__ == "__main__":
    main()
