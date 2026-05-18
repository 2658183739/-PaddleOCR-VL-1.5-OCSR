from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_trial(review_root: Path, converter_command: str) -> dict[str, object]:
    template_path = review_root / "review_template.jsonl"
    rows = list(read_jsonl(template_path))

    results = []
    no_converter = 0
    for row in rows:
        result = dict(row)
        if not converter_command.strip():
            result["predicted_smiles"] = ""
            result["rdkit_valid"] = ""
            result["manual_status"] = "no_converter"
            result["notes"] = "未配置 chemfig/ssml 到 SMILES 的转换器，当前仅生成自动化试跑框架。"
            no_converter += 1
        results.append(result)

    write_jsonl(review_root / "trial_results.jsonl", results)
    summary = {
        "total": len(results),
        "no_converter": no_converter,
        "configured_converter": bool(converter_command.strip()),
    }
    (review_root / "trial_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", default="V2/data/eval/edu_chemc_convertibility_trial_v1/review_package")
    parser.add_argument("--converter-command", default="")
    args = parser.parse_args()

    review_root = Path(args.review_root).resolve()
    summary = run_trial(review_root=review_root, converter_command=args.converter_command)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
