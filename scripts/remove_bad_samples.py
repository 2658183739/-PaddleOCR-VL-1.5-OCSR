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
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_name(text: str) -> str:
    return Path(text).name.strip()


def remove_from_jsonl(jsonl_path: Path, bad_names: set[str], dry_run: bool):
    records = list(read_jsonl(jsonl_path))
    kept = []
    removed = []

    for record in records:
        image_path = record["images"][0]
        if normalize_name(image_path) in bad_names:
            removed.append(record)
        else:
            kept.append(record)

    if not dry_run:
        write_jsonl(jsonl_path, kept)

    return {
        "jsonl": str(jsonl_path),
        "before": len(records),
        "removed": len(removed),
        "after": len(kept),
        "removed_ids": [r.get("meta", {}).get("id", "unknown") for r in removed[:20]],
    }


def maybe_delete_asset(root: Path, bad_names: set[str], dry_run: bool):
    deleted = []
    for path in root.rglob("*"):
        if path.is_file() and path.name in bad_names:
            deleted.append(str(path))
            if not dry_run:
                path.unlink(missing_ok=True)
    return deleted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--phase", choices=["phase2", "phase3", "both"], default="both")
    parser.add_argument("--delete-assets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("bad", nargs="+", help="Bad image file names or full asset paths")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    bad_names = {normalize_name(item) for item in args.bad}

    targets = []
    if args.phase in {"phase2", "both"}:
        targets.append(project_root / "V2" / "data" / "sft_materialized" / "train_phase2_messages.jsonl")
    if args.phase in {"phase3", "both"}:
        targets.append(project_root / "V2" / "data" / "sft_materialized" / "train_phase3_messages.jsonl")

    summaries = []
    for target in targets:
        if not target.exists():
            raise FileNotFoundError(f"Missing target jsonl: {target}")
        summaries.append(remove_from_jsonl(target, bad_names, dry_run=args.dry_run))

    deleted_assets = []
    if args.delete_assets:
        if args.phase in {"phase2", "both"}:
            deleted_assets.extend(maybe_delete_asset(project_root / "V2" / "data" / "assets" / "train_phase2", bad_names, args.dry_run))
        if args.phase in {"phase3", "both"}:
            deleted_assets.extend(maybe_delete_asset(project_root / "V2" / "data" / "assets" / "train_phase3", bad_names, args.dry_run))

    print(json.dumps({
        "dry_run": args.dry_run,
        "phase": args.phase,
        "bad_names": sorted(bad_names),
        "summaries": summaries,
        "deleted_assets": deleted_assets[:50],
        "deleted_asset_count": len(deleted_assets),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
