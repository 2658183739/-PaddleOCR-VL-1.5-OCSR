import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

try:
    from rdkit.Chem import rdFingerprintGenerator
except Exception:
    rdFingerprintGenerator = None


RDLogger.DisableLog("rdApp.*")
MORGAN_GENERATOR = (
    rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    if rdFingerprintGenerator is not None
    else None
)


SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|Si|Se|Na|Li|Mg|Ca|Fe|Zn|Cu|Mn|Hg|Ag|Au|Al|As|"
    r"B|C|N|O|P|S|F|I|b|c|n|o|p|s|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|"
    r"\*|\$|\%[0-9]{2}|[0-9])"
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


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
    if isinstance(ground_truth, dict):
        smiles = ground_truth.get("smiles")
        if smiles:
            return str(smiles).strip()
    for key in ("canonical_smiles", "smiles", "label_summary"):
        smiles = row.get(key)
        if smiles:
            return str(smiles).strip()
    return ""


def safe_rate(numerator: float, denominator: float):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def tokenize_smiles(smiles_text: str):
    text = re.sub(r"\s+", "", str(smiles_text or "").strip())
    if not text:
        return []
    tokens = SMILES_TOKEN_PATTERN.findall(text)
    if "".join(tokens) != text:
        return list(text)
    return tokens


def compute_overlap_metrics(gt_tokens, pred_tokens):
    gt_counter = Counter(gt_tokens)
    pred_counter = Counter(pred_tokens)
    overlap = sum((gt_counter & pred_counter).values())
    precision = safe_rate(overlap, len(pred_tokens))
    recall = safe_rate(overlap, len(gt_tokens))
    f1 = safe_rate(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return {
        "overlap": overlap,
        "gt_count": len(gt_tokens),
        "pred_count": len(pred_tokens),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def levenshtein_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        curr = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def normalized_edit_similarity(a: str, b: str):
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(a, b)
    return 1.0 - (dist / max_len)


def fingerprint_tanimoto(gt_smiles: str, pred_smiles: str):
    gt_mol = Chem.MolFromSmiles(gt_smiles) if gt_smiles else None
    pred_mol = Chem.MolFromSmiles(pred_smiles) if pred_smiles else None
    if gt_mol is None or pred_mol is None:
        return None
    if MORGAN_GENERATOR is not None:
        gt_fp = MORGAN_GENERATOR.GetFingerprint(gt_mol)
        pred_fp = MORGAN_GENERATOR.GetFingerprint(pred_mol)
    else:
        gt_fp = AllChem.GetMorganFingerprintAsBitVect(gt_mol, radius=2, nBits=2048)
        pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol, radius=2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(gt_fp, pred_fp))


def init_group_accumulator():
    return {
        "total": 0,
        "missing_predictions": 0,
        "raw_exact": 0,
        "canonical_exact": 0,
        "valid_predictions": 0,
        "token_tp": 0,
        "token_pred": 0,
        "token_gt": 0,
        "token_macro_precision_sum": 0.0,
        "token_macro_recall_sum": 0.0,
        "token_macro_f1_sum": 0.0,
        "edit_similarity_sum": 0.0,
        "fingerprint_tanimoto_sum": 0.0,
        "fingerprint_tanimoto_count": 0,
    }


def finalize_group(acc):
    total = acc["total"]
    token_micro_precision = safe_rate(acc["token_tp"], acc["token_pred"])
    token_micro_recall = safe_rate(acc["token_tp"], acc["token_gt"])
    token_micro_f1 = (
        safe_rate(2 * token_micro_precision * token_micro_recall, token_micro_precision + token_micro_recall)
        if (token_micro_precision + token_micro_recall)
        else 0.0
    )

    return {
        "total": total,
        "missing_predictions": acc["missing_predictions"],
        "raw_exact_match_accuracy": safe_rate(acc["raw_exact"], total),
        "canonical_exact_match_accuracy": safe_rate(acc["canonical_exact"], total),
        "valid_smiles_rate": safe_rate(acc["valid_predictions"], total),
        "token_micro_precision": token_micro_precision,
        "token_micro_recall": token_micro_recall,
        "token_micro_f1": token_micro_f1,
        "token_macro_precision": safe_rate(acc["token_macro_precision_sum"], total),
        "token_macro_recall": safe_rate(acc["token_macro_recall_sum"], total),
        "token_macro_f1": safe_rate(acc["token_macro_f1_sum"], total),
        "mean_normalized_edit_similarity": safe_rate(acc["edit_similarity_sum"], total),
        "mean_fingerprint_tanimoto": safe_rate(
            acc["fingerprint_tanimoto_sum"], acc["fingerprint_tanimoto_count"]
        ),
        "fingerprint_tanimoto_coverage": safe_rate(acc["fingerprint_tanimoto_count"], total),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-jsonl", required=True)
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--report-json", default="")
    parser.add_argument("--details-jsonl", default="")
    parser.add_argument(
        "--group-fields",
        default="source,difficulty,task_type",
        help="Comma-separated benchmark fields to aggregate by.",
    )
    args = parser.parse_args()

    benchmark = {row["id"]: row for row in read_jsonl(Path(args.benchmark_jsonl).resolve())}
    predictions = {row["id"]: row for row in read_jsonl(Path(args.prediction_jsonl).resolve())}

    group_fields = [field.strip() for field in args.group_fields.split(",") if field.strip()]
    groups = {field: defaultdict(init_group_accumulator) for field in group_fields}
    overall = init_group_accumulator()
    detail_rows = []

    for sample_id, target in benchmark.items():
        gt_raw = get_ground_truth_smiles(target)
        gt_canonical = canonicalize_smiles(gt_raw) or gt_raw
        pred_row = predictions.get(sample_id)

        overall["total"] += 1
        for field in group_fields:
            groups[field][str(target.get(field, "unknown"))]["total"] += 1

        if pred_row is None:
            overall["missing_predictions"] += 1
            for field in group_fields:
                groups[field][str(target.get(field, "unknown"))]["missing_predictions"] += 1
            detail_rows.append(
                {
                    "id": sample_id,
                    "missing_prediction": True,
                    "ground_truth": gt_raw,
                    "ground_truth_canonical": gt_canonical,
                }
            )
            continue

        pred_raw = str(pred_row.get("prediction", "")).strip()
        pred_canonical = canonicalize_smiles(pred_raw)

        eval_gt = gt_canonical
        eval_pred = pred_canonical if pred_canonical is not None else pred_raw

        token_metrics = compute_overlap_metrics(
            tokenize_smiles(eval_gt),
            tokenize_smiles(eval_pred),
        )
        edit_similarity = normalized_edit_similarity(eval_gt, eval_pred)
        tanimoto = fingerprint_tanimoto(gt_canonical, pred_canonical) if pred_canonical else None
        raw_exact = pred_raw == gt_raw
        canonical_exact = pred_canonical is not None and pred_canonical == gt_canonical
        valid_prediction = pred_canonical is not None

        if raw_exact:
            overall["raw_exact"] += 1
        if canonical_exact:
            overall["canonical_exact"] += 1
        if valid_prediction:
            overall["valid_predictions"] += 1
        overall["token_tp"] += token_metrics["overlap"]
        overall["token_pred"] += token_metrics["pred_count"]
        overall["token_gt"] += token_metrics["gt_count"]
        overall["token_macro_precision_sum"] += token_metrics["precision"]
        overall["token_macro_recall_sum"] += token_metrics["recall"]
        overall["token_macro_f1_sum"] += token_metrics["f1"]
        overall["edit_similarity_sum"] += edit_similarity
        if tanimoto is not None:
            overall["fingerprint_tanimoto_sum"] += tanimoto
            overall["fingerprint_tanimoto_count"] += 1

        for field in group_fields:
            group_name = str(target.get(field, "unknown"))
            acc = groups[field][group_name]
            if raw_exact:
                acc["raw_exact"] += 1
            if canonical_exact:
                acc["canonical_exact"] += 1
            if valid_prediction:
                acc["valid_predictions"] += 1
            acc["token_tp"] += token_metrics["overlap"]
            acc["token_pred"] += token_metrics["pred_count"]
            acc["token_gt"] += token_metrics["gt_count"]
            acc["token_macro_precision_sum"] += token_metrics["precision"]
            acc["token_macro_recall_sum"] += token_metrics["recall"]
            acc["token_macro_f1_sum"] += token_metrics["f1"]
            acc["edit_similarity_sum"] += edit_similarity
            if tanimoto is not None:
                acc["fingerprint_tanimoto_sum"] += tanimoto
                acc["fingerprint_tanimoto_count"] += 1

        detail_rows.append(
            {
                "id": sample_id,
                "image_path": target.get("image_path", ""),
                "source": target.get("source", "unknown"),
                "difficulty": target.get("difficulty", "unknown"),
                "task_type": target.get("task_type", "unknown"),
                "ground_truth": gt_raw,
                "ground_truth_canonical": gt_canonical,
                "prediction": pred_raw,
                "prediction_canonical": pred_canonical,
                "raw_exact_match": raw_exact,
                "canonical_exact_match": canonical_exact,
                "valid_smiles": valid_prediction,
                "smiles_token_precision": token_metrics["precision"],
                "smiles_token_recall": token_metrics["recall"],
                "smiles_token_f1": token_metrics["f1"],
                "normalized_edit_similarity": edit_similarity,
                "fingerprint_tanimoto": tanimoto,
            }
        )

    report = {
        "total": overall["total"],
        "missing_predictions": overall["missing_predictions"],
        "accuracy": {
            "raw_exact_match_accuracy": safe_rate(overall["raw_exact"], overall["total"]),
            "canonical_exact_match_accuracy": safe_rate(overall["canonical_exact"], overall["total"]),
            "valid_smiles_rate": safe_rate(overall["valid_predictions"], overall["total"]),
        },
        "f1": {
            "smiles_token_micro_precision": finalize_group(overall)["token_micro_precision"],
            "smiles_token_micro_recall": finalize_group(overall)["token_micro_recall"],
            "smiles_token_micro_f1": finalize_group(overall)["token_micro_f1"],
            "smiles_token_macro_precision": finalize_group(overall)["token_macro_precision"],
            "smiles_token_macro_recall": finalize_group(overall)["token_macro_recall"],
            "smiles_token_macro_f1": finalize_group(overall)["token_macro_f1"],
        },
        "similarity": {
            "mean_normalized_edit_similarity": finalize_group(overall)["mean_normalized_edit_similarity"],
            "mean_fingerprint_tanimoto": finalize_group(overall)["mean_fingerprint_tanimoto"],
            "fingerprint_tanimoto_coverage": finalize_group(overall)["fingerprint_tanimoto_coverage"],
        },
        "by_group": {},
    }

    for field in group_fields:
        report["by_group"][field] = {
            group_name: finalize_group(acc)
            for group_name, acc in sorted(groups[field].items())
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.details_jsonl:
        detail_path = Path(args.details_jsonl).resolve()
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with detail_path.open("w", encoding="utf-8") as handle:
            for row in detail_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
