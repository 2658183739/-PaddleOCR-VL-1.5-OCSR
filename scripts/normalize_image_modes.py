import argparse
from collections import Counter
from pathlib import Path

from PIL import Image


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            yield path


def normalize_one(path: Path, dry_run: bool) -> tuple[str, str]:
    try:
        with Image.open(path) as img:
            mode = img.mode
            if mode == "RGB":
                return ("ok", mode)

            converted = img.convert("RGB")
            if not dry_run:
                converted.save(path)
            return ("converted", mode)
    except Exception as exc:
        return ("error", f"{type(exc).__name__}: {exc}")


def normalize_root(root: Path, dry_run: bool, limit_errors: int = 20):
    counters = Counter()
    mode_counter = Counter()
    errors = []

    for path in iter_images(root):
        counters["total"] += 1
        status, detail = normalize_one(path, dry_run=dry_run)
        counters[status] += 1
        if status == "ok":
            mode_counter[detail] += 1
        elif status == "converted":
            mode_counter[detail] += 1
        else:
            if len(errors) < limit_errors:
                errors.append((str(path), detail))

    return counters, mode_counter, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, help="Root directory to normalize. Can be passed multiple times.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    args = parser.parse_args()

    for root_value in args.root:
        root = Path(root_value).resolve()
        if not root.exists():
            print(f"[MISSING] {root}")
            continue
        counters, modes, errors = normalize_root(root, dry_run=args.dry_run)
        print(f"=== {root} ===")
        print(f"total={counters['total']} ok={counters['ok']} converted={counters['converted']} error={counters['error']} dry_run={args.dry_run}")
        print(f"modes={dict(modes)}")
        if errors:
            print("errors:")
            for path, detail in errors:
                print(f"  {path} -> {detail}")


if __name__ == "__main__":
    main()
