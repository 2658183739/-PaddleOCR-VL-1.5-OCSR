from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from pathlib import Path


def try_load_rdkit():
    try:
        from rdkit import Chem

        return Chem
    except Exception:
        return None


def canonicalize(Chem, smiles: str) -> str | None:
    text = str(smiles or "").strip()
    if not text:
        return None
    if Chem is None:
        return text
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def iter_lines(path: Path):
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                yield line.rstrip("\n")
    else:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                yield line.rstrip("\n")


def parse_smiles_from_text(path: Path, smiles_column: str | None):
    first = next(iter_lines(path), None)
    if first is None:
        return

    suffixes = {s.lower() for s in path.suffixes}
    if ".csv" in suffixes or ".tsv" in suffixes:
        delimiter = "\t" if ".tsv" in suffixes else ","
        headers = [cell.strip() for cell in first.split(delimiter)]
        column_index = None
        if smiles_column and smiles_column in headers:
            column_index = headers.index(smiles_column)
        else:
            for candidate in ("canonical_smiles", "smiles", "Smiles", "SMILES"):
                if candidate in headers:
                    column_index = headers.index(candidate)
                    break
        if column_index is None:
            raise ValueError(f"Could not find SMILES column in {path}")
        for line in iter_lines(path):
            if line == first:
                continue
            parts = line.split(delimiter)
            if column_index < len(parts):
                yield parts[column_index].strip()
        return

    if ".jsonl" in suffixes:
        for line in iter_lines(path):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("canonical_smiles", "smiles", "label_summary"):
                if row.get(key):
                    yield str(row[key]).strip()
                    break
        return

    for line in iter_lines(path):
        if line.strip():
            yield line.strip().split()[0]


def sample_smiles(input_path: Path, output_path: Path, source_name: str, limit: int, min_len: int, max_len: int, seed: int, smiles_column: str | None):
    Chem = try_load_rdkit()
    seen = set()
    kept = []
    rng = random.Random(seed)
    for raw in parse_smiles_from_text(input_path, smiles_column):
        canonical = canonicalize(Chem, raw)
        if not canonical:
            continue
        if canonical in seen:
            continue
        if len(canonical) < min_len or len(canonical) > max_len:
            continue
        seen.add(canonical)
        kept.append(canonical)

    rng.shuffle(kept)
    selected = kept[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "smiles", "source", "task_type", "license", "source_url_or_doc"])
        for idx, smiles in enumerate(selected, start=1):
            writer.writerow([
                f"{source_name}_{idx:06d}",
                smiles,
                source_name,
                "molecule_structure_recognition",
                "public_source_sampled",
                input_path.name,
            ])

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "source_name": source_name,
        "rdkit_available": Chem is not None,
        "unique_valid_smiles": len(kept),
        "selected": len(selected),
        "min_len": min_len,
        "max_len": max_len,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--smiles-column", default="")
    args = parser.parse_args()

    sample_smiles(
        input_path=Path(args.input).resolve(),
        output_path=Path(args.output).resolve(),
        source_name=args.source_name,
        limit=args.limit,
        min_len=args.min_len,
        max_len=args.max_len,
        seed=args.seed,
        smiles_column=args.smiles_column or None,
    )


if __name__ == "__main__":
    main()
