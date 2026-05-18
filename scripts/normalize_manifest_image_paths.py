import argparse
import csv
from pathlib import Path


KNOWN_ROOT_MARKERS = [
    "real_world_collection\\",
    "real_world_collection/",
    "real_world_collection_extra\\",
    "real_world_collection_extra/",
    "public_extra_collection\\",
    "public_extra_collection/",
]


def normalize_separators(path_text: str):
    return path_text.replace("\\", "/")


def to_project_relative_path(raw_path: str, project_root: Path, manifest_path: Path):
    text = str(raw_path or "").strip()
    if not text:
        return text

    normalized = normalize_separators(text)
    normalized_lower = normalized.lower()

    for marker in KNOWN_ROOT_MARKERS:
        marker_norm = normalize_separators(marker)
        idx = normalized_lower.find(marker_norm.lower())
        if idx >= 0:
            return normalized[idx:]

    raw = Path(text)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(project_root / raw)
        candidates.append(manifest_path.parent / raw)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            return resolved.relative_to(project_root.resolve()).as_posix()
        except Exception:
            continue

    return normalized


def normalize_manifest(manifest_path: Path, project_root: Path):
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "image_path" not in fieldnames:
        raise ValueError(f"Manifest does not contain image_path column: {manifest_path}")

    changed = 0
    for row in rows:
        before = row.get("image_path", "")
        after = to_project_relative_path(before, project_root, manifest_path)
        if after != before:
            row["image_path"] = after
            changed += 1

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{manifest_path}: normalized {changed} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", action="append", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    for item in args.manifest:
        normalize_manifest(Path(item).resolve(), project_root)


if __name__ == "__main__":
    main()
