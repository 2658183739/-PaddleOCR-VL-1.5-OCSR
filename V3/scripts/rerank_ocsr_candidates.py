#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


RDLogger.DisableLog("rdApp.*")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def canonicalize(smiles: str, isomeric: bool = True):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def get_ground_truth_smiles(row: dict):
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, dict) and ground_truth.get("smiles"):
        return str(ground_truth["smiles"]).strip()
    for key in ("canonical_smiles", "smiles", "label_summary"):
        if row.get(key):
            return str(row[key]).strip()
    return ""


def mol_props(canonical_smiles: str):
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return {
            "heavy_atoms": 999,
            "hetero_atoms": 999,
            "rings": 999,
            "fragments": 999,
            "formal_charge_abs": 999,
            "has_dot": 1 if "." in str(canonical_smiles) else 0,
        }
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "hetero_atoms": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "fragments": len(Chem.GetMolFrags(mol)),
        "formal_charge_abs": abs(formal_charge),
        "has_dot": 1 if "." in canonical_smiles else 0,
    }


def aggregate_candidates(candidates):
    grouped = defaultdict(list)
    for item in candidates or []:
        canonical = canonicalize(item.get("canonical_prediction") or item.get("prediction"))
        if canonical:
            grouped[canonical].append(item)

    aggregates = []
    for canonical, items in grouped.items():
        scores = [
            float(item["generation_score"])
            for item in items
            if item.get("generation_score") is not None
        ]
        best_item = max(
            items,
            key=lambda item: (
                item.get("generation_score") if item.get("generation_score") is not None else -1_000_000.0,
                -item.get("prompt_index", 99),
                -item.get("tta_index", 99),
                -item.get("generation_index", 99),
            ),
        )
        props = mol_props(canonical)
        aggregates.append(
            {
                "canonical": canonical,
                "nonisomeric": canonicalize(canonical, isomeric=False),
                "items": items,
                "representative": best_item,
                "count": len(items),
                "max_score": max(scores) if scores else -1_000_000.0,
                "mean_score": sum(scores) / len(scores) if scores else -1_000_000.0,
                "min_prompt_index": min(item.get("prompt_index", 99) for item in items),
                "min_tta_index": min(item.get("tta_index", 99) for item in items),
                "min_generation_index": min(item.get("generation_index", 99) for item in items),
                **props,
            }
        )
    return aggregates


def choose_current(aggregates):
    return sorted(
        aggregates,
        key=lambda item: (
            -item["count"],
            -item["max_score"],
            item["representative"].get("smiles_structure_penalty", 0),
            len(item["representative"].get("prediction", "")),
            item["min_prompt_index"],
            item["min_tta_index"],
            item["min_generation_index"],
        ),
    )[0]


def choose_chem_light(aggregates, salt_bonus: float, heavy_penalty: float, stereo_min_ratio: float):
    # The first score gently rescues salts/multi-fragment structures that tie the main vote.
    best = max(
        aggregates,
        key=lambda item: (
            item["count"] + salt_bonus * item["has_dot"],
            item["max_score"] - heavy_penalty * item["heavy_atoms"],
            -item["formal_charge_abs"],
            -item["min_prompt_index"],
        ),
    )

    # If the same non-stereo skeleton has an explicit backslash isomer with enough support,
    # prefer it. This fixes E/Z cases where beam 0 collapses the direction.
    if best.get("nonisomeric"):
        stereo_group = [
            item
            for item in aggregates
            if item.get("nonisomeric") == best["nonisomeric"] and "\\" in item["canonical"]
        ]
        if stereo_group:
            stereo_best = max(
                stereo_group,
                key=lambda item: (
                    item["count"],
                    item["max_score"] - heavy_penalty * item["heavy_atoms"],
                    -item["min_generation_index"],
                ),
            )
            if stereo_best["count"] >= best["count"] * stereo_min_ratio:
                best = stereo_best

    return best


def choose_realworld_soft(aggregates, count_bonus: float, heavy_penalty: float):
    # Cropped real-world exam pages often produce several near-tie candidates.
    # In that setting, hard vote count over-selects longer hallucinated structures.
    return max(
        aggregates,
        key=lambda item: (
            item["max_score"] + count_bonus * item["count"] - heavy_penalty * item["heavy_atoms"],
            -item["formal_charge_abs"],
            -item["fragments"],
            -item["min_prompt_index"],
            -item["min_generation_index"],
        ),
    )


def should_use_realworld_soft(row: dict):
    fields = [
        row.get("id", ""),
        row.get("source", ""),
        row.get("difficulty", ""),
        row.get("task_type", ""),
        row.get("preprocess_variant", ""),
        row.get("preprocess_note", ""),
    ]
    text = " ".join(str(value) for value in fields).lower()
    return "chinese_exam" in text or "exam_q1" in text


def build_output_row(row: dict, chosen: dict, keep_candidates: bool):
    rep = chosen["representative"]
    out = dict(row)
    out["prediction"] = rep.get("prediction") or chosen["canonical"]
    out["canonical_prediction"] = chosen["canonical"]
    out["generation_score"] = rep.get("generation_score")
    out["smiles_structure_penalty"] = rep.get("smiles_structure_penalty")
    out["raw_text"] = rep.get("raw_text", "")
    out["prompt"] = rep.get("prompt", row.get("prompt", ""))
    out["selection_reason"] = "chem_light_candidate_rerank"
    out["vote_count"] = chosen["count"]
    out["rerank_debug"] = {
        "candidate_count": len(row.get("candidates", [])),
        "unique_valid_candidates": None,
        "max_score": chosen["max_score"],
        "mean_score": chosen["mean_score"],
        "has_dot": chosen["has_dot"],
        "heavy_atoms": chosen["heavy_atoms"],
        "fragments": chosen["fragments"],
        "nonisomeric": chosen.get("nonisomeric"),
    }
    if not keep_candidates:
        out.pop("candidates", None)
    return out


def evaluate_rows(rows, labels):
    total = len(rows)
    exact = 0
    valid = 0
    changed = 0
    good_change = 0
    bad_change = 0
    oracle = 0
    preference_rows = []

    for original, reranked, aggregates in rows:
        sample_id = original["id"]
        target = labels.get(sample_id)
        pred = canonicalize(reranked.get("prediction"))
        base_pred = canonicalize(original.get("prediction"))
        exact += pred is not None and pred == target
        valid += pred is not None
        oracle += target in {item["canonical"] for item in aggregates}
        if pred != base_pred:
            changed += 1
            good_change += pred == target and base_pred != target
            bad_change += pred != target and base_pred == target

        if target and base_pred != target and target in {item["canonical"] for item in aggregates}:
            positive = next(item for item in aggregates if item["canonical"] == target)
            negative = next((item for item in aggregates if item["canonical"] == base_pred), None)
            if negative:
                preference_rows.append(
                    {
                        "id": sample_id,
                        "image_path": original.get("image_path"),
                        "positive_smiles": positive["canonical"],
                        "negative_smiles": negative["canonical"],
                        "positive_votes": positive["count"],
                        "negative_votes": negative["count"],
                        "positive_max_score": positive["max_score"],
                        "negative_max_score": negative["max_score"],
                    }
                )

    return {
        "total": total,
        "canonical_exact": exact / total if total else 0.0,
        "valid_smiles_rate": valid / total if total else 0.0,
        "oracle_exact": oracle / total if total else 0.0,
        "changed_predictions": changed,
        "good_changes": good_change,
        "bad_changes": bad_change,
        "preference_pair_count": len(preference_rows),
        "preference_rows": preference_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--labels-jsonl", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--preference-jsonl", default="")
    parser.add_argument(
        "--mode",
        choices=["current", "chem_light", "realworld_soft", "hybrid_realworld_soft"],
        default="chem_light",
    )
    parser.add_argument("--salt-bonus", type=float, default=2.0)
    parser.add_argument("--heavy-penalty", type=float, default=0.02)
    parser.add_argument("--soft-count-bonus", type=float, default=0.0)
    parser.add_argument("--soft-heavy-penalty", type=float, default=0.006)
    parser.add_argument("--stereo-min-ratio", type=float, default=0.25)
    parser.add_argument("--keep-candidates", action="store_true")
    args = parser.parse_args()

    labels = {}
    if args.labels_jsonl:
        for row in read_jsonl(Path(args.labels_jsonl)):
            labels[row["id"]] = canonicalize(get_ground_truth_smiles(row))

    output_rows = []
    eval_rows = []
    for row in read_jsonl(Path(args.prediction_jsonl)):
        aggregates = aggregate_candidates(row.get("candidates", []))
        if not aggregates:
            reranked = dict(row)
            reranked["selection_reason"] = "chem_light_no_valid_candidate"
            if not args.keep_candidates:
                reranked.pop("candidates", None)
        else:
            if args.mode == "current":
                chosen = choose_current(aggregates)
            elif args.mode == "chem_light" or (
                args.mode == "hybrid_realworld_soft" and not should_use_realworld_soft(row)
            ):
                chosen = choose_chem_light(
                    aggregates,
                    salt_bonus=args.salt_bonus,
                    heavy_penalty=args.heavy_penalty,
                    stereo_min_ratio=args.stereo_min_ratio,
                )
            else:
                chosen = choose_realworld_soft(
                    aggregates,
                    count_bonus=args.soft_count_bonus,
                    heavy_penalty=args.soft_heavy_penalty,
                )
            reranked = build_output_row(row, chosen, keep_candidates=args.keep_candidates)
            reranked["selection_reason"] = f"{args.mode}_candidate_rerank"
            reranked["rerank_debug"]["unique_valid_candidates"] = len(aggregates)
        output_rows.append(reranked)
        if labels:
            eval_rows.append((row, reranked, aggregates))

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if labels:
        report = evaluate_rows(eval_rows, labels)
        preference_rows = report.pop("preference_rows")
        if args.report_json:
            report_path = Path(args.report_json)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))

        if args.preference_jsonl:
            pref_path = Path(args.preference_jsonl)
            pref_path.parent.mkdir(parents=True, exist_ok=True)
            with pref_path.open("w", encoding="utf-8") as handle:
                for row in preference_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
