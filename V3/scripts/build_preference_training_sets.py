import argparse
import json
from pathlib import Path


DEFAULT_PROMPT = "OCR: Output only the canonical SMILES string for the molecule shown in the image."


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


def normalize_image_path(image_path: str, project_root: Path | None, relative_to: Path | None):
    path = Path(str(image_path))
    if project_root and path.is_absolute():
        try:
            return path.relative_to(project_root).as_posix()
        except ValueError:
            return str(path)
    if relative_to and path.is_absolute():
        try:
            return path.relative_to(relative_to).as_posix()
        except ValueError:
            return str(path)
    return str(image_path)


def collect_pairs(paths, project_root: Path | None):
    seen = set()
    rows = []
    for source_path in paths:
        for row in read_jsonl(source_path):
            image = normalize_image_path(row.get("image_path", ""), project_root, None)
            key = (
                row.get("id"),
                image,
                row.get("positive_smiles"),
                row.get("negative_smiles"),
            )
            if key in seen:
                continue
            seen.add(key)
            out = dict(row)
            out["image_path"] = image
            out["source_file"] = source_path.as_posix()
            rows.append(out)
    return rows


def build_sft_row(pair, prompt: str):
    return {
        "messages": [
            {"role": "user", "content": f"<image>{prompt}"},
            {"role": "assistant", "content": pair["positive_smiles"]},
        ],
        "images": [pair["image_path"]],
        "meta": {
            "id": pair.get("id"),
            "source_file": pair.get("source_file"),
            "negative_smiles": pair.get("negative_smiles"),
            "positive_votes": pair.get("positive_votes"),
            "negative_votes": pair.get("negative_votes"),
            "positive_max_score": pair.get("positive_max_score"),
            "negative_max_score": pair.get("negative_max_score"),
            "preference_policy": "candidate_oracle_positive",
        },
    }


def build_dpo_row(pair, prompt: str):
    user = {"role": "user", "content": f"<image>{prompt}"}
    return {
        "id": pair.get("id"),
        "images": [pair["image_path"]],
        "chosen": [
            user,
            {"role": "assistant", "content": pair["positive_smiles"]},
        ],
        "rejected": [
            user,
            {"role": "assistant", "content": pair["negative_smiles"]},
        ],
        "meta": {
            "source_file": pair.get("source_file"),
            "positive_votes": pair.get("positive_votes"),
            "negative_votes": pair.get("negative_votes"),
            "positive_max_score": pair.get("positive_max_score"),
            "negative_max_score": pair.get("negative_max_score"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preference-jsonl", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default="")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else None
    input_paths = [Path(path).resolve() for path in args.preference_jsonl]
    output_dir = Path(args.output_dir)
    pairs = collect_pairs(input_paths, project_root)

    raw_path = output_dir / "preference_pairs_merged.jsonl"
    sft_path = output_dir / "preference_positive_sft_messages.jsonl"
    dpo_path = output_dir / "preference_dpo_chosen_rejected.jsonl"
    report_path = output_dir / "preference_dataset_report.json"

    write_jsonl(raw_path, pairs)
    write_jsonl(sft_path, [build_sft_row(pair, args.prompt) for pair in pairs])
    write_jsonl(dpo_path, [build_dpo_row(pair, args.prompt) for pair in pairs])

    report = {
        "input_files": [path.as_posix() for path in input_paths],
        "pair_count": len(pairs),
        "raw_pairs": raw_path.as_posix(),
        "sft_messages": sft_path.as_posix(),
        "dpo_chosen_rejected": dpo_path.as_posix(),
        "note": "DPO training should wait until this file has enough pairs and includes the target domain.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
