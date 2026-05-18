import argparse
import shutil
import zipfile
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_streamed(parts: list[Path], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("wb") as dst:
        for part in parts:
            with part.open("rb") as src:
                shutil.copyfileobj(src, dst, length=1024 * 1024 * 16)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="V2/data/EDU-CHMEC-MM23")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    raw_parts = root / "raw_parts"
    raw_unpacked = root / "raw_unpacked"
    ensure_dir(raw_parts)
    ensure_dir(raw_unpacked)

    # Collect all split parts from the current mixed top-level layout.
    part_files = sorted(root.rglob("EDU-CHEMC.zip.*"), key=lambda p: int(p.suffix[1:]))
    if not part_files:
        raise FileNotFoundError(f"No EDU-CHEMC split part files found under: {root}")

    for part in part_files:
        dest = raw_parts / part.name
        if part.resolve() != dest.resolve() and not dest.exists():
            shutil.copy2(part, dest)

    merged_zip = raw_unpacked / "EDU-CHEMC_train_validation_full.zip"
    copy_streamed(sorted(raw_parts.glob("EDU-CHEMC.zip.*"), key=lambda p: int(p.suffix[1:])), merged_zip)

    merged_extract_dir = raw_unpacked / "train_validation"
    if not merged_extract_dir.exists():
        extract_zip(merged_zip, merged_extract_dir)

    # Candidate eval val999 zip if present.
    val999_candidates = list(root.rglob("EDU-CHEMC-val999.zip"))
    if val999_candidates:
        val999_zip = val999_candidates[0]
        dest = raw_parts / val999_zip.name
        if val999_zip.resolve() != dest.resolve() and not dest.exists():
            shutil.copy2(val999_zip, dest)
        val999_extract_dir = raw_unpacked / "val999"
        if not val999_extract_dir.exists():
            extract_zip(dest, val999_extract_dir)

    # Optional top-level test archive bundle.
    test_archives = list(root.glob("test-*.zip"))
    if test_archives:
        test_bundle_dir = raw_unpacked / "test_bundle"
        for archive in test_archives:
            dest = raw_parts / archive.name
            if archive.resolve() != dest.resolve() and not dest.exists():
                shutil.copy2(archive, dest)
            out_dir = test_bundle_dir / archive.stem
            if not out_dir.exists():
                extract_zip(dest, out_dir)

    print(f"root={root}")
    print(f"raw_parts={raw_parts}")
    print(f"raw_unpacked={raw_unpacked}")
    print(f"merged_zip={merged_zip}")


if __name__ == "__main__":
    main()
