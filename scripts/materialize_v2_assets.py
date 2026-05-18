import argparse
import json
import shutil
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


def resolve_source_path(project_root: Path, image_value: str) -> Path:
    raw = Path(str(image_value).strip())
    if raw.is_absolute():
        return raw.resolve()
    return (project_root / raw).resolve()


def bucket_name(message_record: dict) -> str:
    source = message_record.get("meta", {}).get("source", "unknown")
    difficulty = message_record.get("meta", {}).get("difficulty", "unknown")
    safe_source = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(source))
    safe_diff = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(difficulty))
    return f"{safe_source}__{safe_diff}"


def stage_name_from_file(path: Path) -> str:
    name = path.stem
    if name.endswith("_messages"):
        name = name[: -len("_messages")]
    return name


def materialize_dataset(project_root: Path, input_path: Path, assets_root: Path, output_path: Path):
    stage_name = stage_name_from_file(input_path)
    stage_assets_root = assets_root / stage_name
    stage_assets_root.mkdir(parents=True, exist_ok=True)

    rewritten = []
    copied = 0
    reused = 0

    for record in read_jsonl(input_path):
        image_value = record["images"][0]
        normalized = str(image_value)
        if normalized.startswith("./"):
            normalized = normalized[2:]
        source_path = resolve_source_path(project_root, normalized)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source asset: {source_path}")

        group_dir = stage_assets_root / bucket_name(record)
        group_dir.mkdir(parents=True, exist_ok=True)
        dest_path = group_dir / source_path.name
        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)
            copied += 1
        else:
            reused += 1

        new_record = dict(record)
        # IMPORTANT: message jsonl files live under V2/data/sft_materialized/.
        # Use a path relative to that directory, otherwise some loaders will
        # incorrectly join the jsonl directory with a project-root-style path.
        new_record["images"] = [f"../assets/{stage_name}/{bucket_name(record)}/{source_path.name}".replace('\\', '/')]
        rewritten.append(new_record)

    write_jsonl(output_path, rewritten)
    return {
        "stage": stage_name,
        "records": len(rewritten),
        "copied": copied,
        "reused": reused,
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--input", action="append", required=False, default=[])
    parser.add_argument("--assets-root", default="V2/data/assets")
    parser.add_argument("--output-root", default="V2/data/sft_materialized")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    assets_root = (project_root / args.assets_root).resolve()
    output_root = (project_root / args.output_root).resolve()

    input_files = [Path(item).resolve() for item in args.input] if args.input else [
        (project_root / "V2/data/sft/train_phase1_messages.jsonl").resolve(),
        (project_root / "V2/data/sft/train_phase2_messages.jsonl").resolve(),
        (project_root / "V2/data/sft/train_phase3_messages.jsonl").resolve(),
        (project_root / "V2/data/sft/val_messages.jsonl").resolve(),
    ]

    summary = []
    for input_path in input_files:
        output_path = output_root / input_path.name
        result = materialize_dataset(project_root, input_path, assets_root, output_path)
        summary.append(result)
        print(json.dumps(result, ensure_ascii=False))

    summary_path = project_root / "V2/reports/v2_asset_materialization_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
