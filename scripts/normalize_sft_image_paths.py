import argparse
import json
from pathlib import Path


KNOWN_ROOT_MARKERS = (
    "prepared/",
    "ocsr_evalset_final/",
    "real_world_collection/",
    "real_world_collection_extra/",
    "public_extra_collection/",
    "server_ready/",
    "models/",
)


def normalize_separators(path_text: str):
    return str(path_text or "").replace("\\", "/").strip()


def path_tail_from_known_marker(path_text: str):
    normalized = normalize_separators(path_text)
    lowered = normalized.lower()
    for marker in KNOWN_ROOT_MARKERS:
        marker_lower = marker.lower()
        idx = lowered.find(marker_lower)
        if idx >= 0:
            return normalized[idx:]
    return None


def resolve_asset_path(raw_path: str, dataset_path: Path, project_root: Path):
    text = str(raw_path or "").strip()
    if not text or text.startswith("http") or text.startswith("data:"):
        return text, False

    raw = Path(text)
    tail = path_tail_from_known_marker(text)
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(project_root / raw)
        candidates.append(dataset_path.parent / raw)
        candidates.append(dataset_path.parent.parent / raw)

    if tail:
        tail_path = Path(tail)
        candidates.extend(
            [
                project_root / tail_path,
                dataset_path.parent / tail_path,
                dataset_path.parent.parent / tail_path,
            ]
        )

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        candidate_str = str(resolved)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if resolved.exists():
            return candidate_str, candidate_str != text

    return text, False


def normalize_dataset(dataset_path: Path, project_root: Path, dry_run: bool):
    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    output_lines = []
    changed = 0
    unresolved = 0

    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        row_changed = False

        for info_key, url_key in (("image_info", "image_url"), ("video_info", "video_url")):
            infos = row.get(info_key, [])
            for info in infos:
                before = info.get(url_key, "")
                after, did_change = resolve_asset_path(before, dataset_path, project_root)
                if did_change:
                    info[url_key] = after
                    row_changed = True
                elif before and not before.startswith("http") and not before.startswith("data:") and not Path(before).exists():
                    unresolved += 1

        if row_changed:
            changed += 1
        output_lines.append(json.dumps(row, ensure_ascii=False))

    if changed and not dry_run:
        dataset_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    print(f"{dataset_path}: changed_rows={changed}, unresolved_assets={unresolved}, dry_run={dry_run}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    for item in args.dataset:
        normalize_dataset(Path(item).resolve(), project_root, args.dry_run)


if __name__ == "__main__":
    main()
