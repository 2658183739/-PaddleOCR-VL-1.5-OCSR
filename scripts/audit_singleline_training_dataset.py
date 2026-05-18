import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def assistant_text(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def user_text(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def resolve_image(jsonl_path: Path, image_value: str) -> Path:
    raw = Path(str(image_value))
    if raw.is_absolute():
        return raw
    return (jsonl_path.parent / raw).resolve()


def load_eval_keys(eval_paths: list[Path]):
    ids = set()
    images = set()
    smiles = Counter()
    for eval_path in eval_paths:
        if not eval_path.exists():
            continue
        for _, row in read_jsonl(eval_path):
            ids.add(str(row.get("id", "")))
            image = row.get("image") or row.get("image_path")
            if image:
                images.add(Path(str(image)).name.lower())
            gt = row.get("ground_truth") or {}
            smile = row.get("canonical_smiles") or row.get("smiles") or gt.get("smiles") or row.get("label_summary")
            if smile:
                smiles[str(smile)] += 1
    return ids, images, smiles


def try_load_rdkit():
    try:
        from rdkit import Chem

        return Chem
    except Exception:
        return None


def audit(train_path: Path, eval_paths: list[Path], limit_invalid_examples: int):
    Chem = try_load_rdkit()
    eval_ids, eval_image_names, eval_smiles = load_eval_keys(eval_paths)

    total = 0
    missing_images = []
    unreadable_images = []
    bad_prompt = []
    empty_outputs = []
    non_smiles_outputs = []
    invalid_smiles = []
    id_counts = Counter()
    source_counts = Counter()
    difficulty_counts = Counter()
    policy_counts = Counter()
    eval_id_overlap = []
    eval_image_name_overlap = []
    eval_smiles_overlap = 0
    unique_smiles = set()

    for line_no, record in read_jsonl(train_path):
        total += 1
        meta = record.get("meta", {})
        record_id = str(meta.get("id", ""))
        source = str(meta.get("source", "unknown"))
        difficulty = str(meta.get("difficulty", "unknown"))
        text = assistant_text(record).strip()
        prompt = user_text(record)

        id_counts[record_id] += 1
        source_counts[source] += 1
        difficulty_counts[difficulty] += 1
        policy_counts[str(meta.get("singleline_policy", "none"))] += 1

        if "canonical SMILES" not in prompt:
            bad_prompt.append({"line": line_no, "id": record_id, "prompt": prompt[:160]})
        if not text:
            empty_outputs.append({"line": line_no, "id": record_id})
        if text.startswith("\\chemfig") or "ssml" in prompt.lower():
            non_smiles_outputs.append({"line": line_no, "id": record_id, "output": text[:160]})

        if Chem is not None and text:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                if len(invalid_smiles) < limit_invalid_examples:
                    invalid_smiles.append({"line": line_no, "id": record_id, "smiles": text})
            else:
                unique_smiles.add(Chem.MolToSmiles(mol, canonical=True))
        elif text:
            unique_smiles.add(text)

        image_values = record.get("images") or []
        if not image_values:
            missing_images.append({"line": line_no, "id": record_id, "image": ""})
        for image in image_values:
            image_path = resolve_image(train_path, image)
            if not image_path.exists():
                missing_images.append({"line": line_no, "id": record_id, "image": str(image)})
            elif image_path.stat().st_size <= 0:
                unreadable_images.append({"line": line_no, "id": record_id, "image": str(image), "reason": "empty_file"})
            else:
                try:
                    with Image.open(image_path) as opened:
                        opened.verify()
                except Exception as exc:
                    unreadable_images.append({"line": line_no, "id": record_id, "image": str(image), "reason": str(exc)})
            image_name = image_path.name.lower()
            if image_name in eval_image_names and len(eval_image_name_overlap) < limit_invalid_examples:
                eval_image_name_overlap.append({"line": line_no, "id": record_id, "image_name": image_name})

        if record_id in eval_ids and len(eval_id_overlap) < limit_invalid_examples:
            eval_id_overlap.append({"line": line_no, "id": record_id})
        if text in eval_smiles:
            eval_smiles_overlap += 1

    duplicate_unique_ids = {key: value for key, value in id_counts.items() if value > 1}
    report = {
        "train_path": str(train_path),
        "eval_paths": [str(path) for path in eval_paths],
        "total": total,
        "unique_ids": len(id_counts),
        "weighted_duplicate_id_count": len(duplicate_unique_ids),
        "max_id_repeat": max(id_counts.values()) if id_counts else 0,
        "unique_outputs_or_canonical_smiles": len(unique_smiles),
        "source_counts": dict(source_counts),
        "difficulty_counts": dict(difficulty_counts),
        "policy_counts": dict(policy_counts),
        "missing_images_count": len(missing_images),
        "unreadable_images_count": len(unreadable_images),
        "bad_prompt_count": len(bad_prompt),
        "empty_output_count": len(empty_outputs),
        "non_smiles_output_count": len(non_smiles_outputs),
        "invalid_smiles_count": len(invalid_smiles) if Chem is not None else None,
        "rdkit_available": Chem is not None,
        "eval_id_overlap_count": len(eval_id_overlap),
        "eval_image_name_overlap_count": len(eval_image_name_overlap),
        "eval_smiles_overlap_weighted_count": eval_smiles_overlap,
        "examples": {
            "missing_images": missing_images[:limit_invalid_examples],
            "unreadable_images": unreadable_images[:limit_invalid_examples],
            "bad_prompt": bad_prompt[:limit_invalid_examples],
            "empty_outputs": empty_outputs[:limit_invalid_examples],
            "non_smiles_outputs": non_smiles_outputs[:limit_invalid_examples],
            "invalid_smiles": invalid_smiles[:limit_invalid_examples],
            "eval_id_overlap": eval_id_overlap[:limit_invalid_examples],
            "eval_image_name_overlap": eval_image_name_overlap[:limit_invalid_examples],
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train", default="V2/data/sft_materialized/train_singleline_rw_messages.jsonl")
    parser.add_argument(
        "--eval",
        nargs="*",
        default=["V2/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl"],
    )
    parser.add_argument("--report", default="V2/reports/singleline_rw_dataset_audit.json")
    parser.add_argument("--limit-invalid-examples", type=int, default=20)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    train_path = (project_root / args.train).resolve()
    eval_paths = [(project_root / path).resolve() for path in args.eval]
    report_path = (project_root / args.report).resolve()

    report = audit(train_path, eval_paths, args.limit_invalid_examples)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
