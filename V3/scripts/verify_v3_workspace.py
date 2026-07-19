#!/usr/bin/env python3
"""Verify that the assembled V3 workspace is portable and leakage-safe."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import rdkit
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.error")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


@lru_cache(maxsize=100000)
def canonicalize(value: str) -> str | None:
    text = re.sub(r"\s+", "", str(value or ""))
    if not text or "." in text:
        return None
    current = text
    seen: set[str] = set()
    for _ in range(16):
        if current in seen:
            return None
        seen.add(current)
        mol = Chem.MolFromSmiles(current)
        if mol is None or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            return None
        normalized = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if normalized == current:
            return normalized
        current = normalized
    return None


@lru_cache(maxsize=100000)
def murcko_scaffold(value: str) -> str | None:
    canonical = canonicalize(value)
    if canonical is None:
        return None
    mol = Chem.MolFromSmiles(canonical)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def assistant_text(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages", [])):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def ground_truth(row: dict[str, Any]) -> str:
    value = row.get("ground_truth", {})
    return str(value.get("smiles", "")) if isinstance(value, dict) else ""


def resolve_image(raw_text: str, manifest: Path, root: Path) -> Path | None:
    raw = Path(raw_text)
    candidates = [raw] if raw.is_absolute() else [
        root / raw,
        manifest.parent / raw,
        manifest.parent.parent / raw,
        manifest.parent.parent.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--open-image-sample", type=int, default=200)
    args = parser.parse_args()
    root = args.project_root.resolve()
    v3 = root / "V3"

    required_files = [
        v3 / "models/v2_1_export/model-00001-of-00001.safetensors",
        v3 / "models/v2_1_export/config.json",
        v3 / "models/paddleocr_vl_1_5_base/model.safetensors",
        v3 / "data/sft_materialized/dev_legacy_region_strict_messages.jsonl",
    ]
    missing_required = [str(path) for path in required_files if not path.is_file()]

    train_paths = {
        "A": v3 / "data/sft_materialized/train_v3_a_control.jsonl",
        "D_wild_only": v3 / "data/sft_materialized/train_v3_d_wild_only.jsonl",
        "E_aug_only": v3 / "data/sft_materialized/train_v3_e_aug_only.jsonl",
        "B": v3 / "data/sft_materialized/train_v3_b_recommended.jsonl",
        "C": v3 / "data/sft_materialized/train_v3_c_real_heavy.jsonl",
        "hard": v3 / "data/sft_materialized/train_v3_hard_replay_seed.jsonl",
    }
    eval_paths = {
        "legacy_dev_core": v3 / "data/eval/dev_legacy_core_strict/labels.jsonl",
        "legacy_dev_region": v3 / "data/eval/dev_legacy_region_strict/labels.jsonl",
        "locked_wild_strict": v3 / "data/eval/wild_strict_v3/labels.jsonl",
        "locked_wild_scaffold_novel": v3 / "data/eval/wild_strict_scaffold_novel_v3/labels.jsonl",
    }

    eval_smiles: set[str] = set()
    eval_counts: dict[str, int] = {}
    eval_unique_smiles: dict[str, int] = {}
    eval_duplicate_smiles: dict[str, int] = {}
    eval_scaffolds: set[str] = set()
    locked_wild_papers: set[str] = set()
    locked_wild_paper_counts: Counter[str] = Counter()
    missing_eval_images: Counter[str] = Counter()
    image_sample: list[Path] = []
    for name, path in eval_paths.items():
        rows = read_jsonl(path)
        eval_counts[name] = len(rows)
        panel_smiles: list[str] = []
        for row in rows:
            canonical = canonicalize(ground_truth(row))
            if canonical:
                panel_smiles.append(canonical)
                eval_smiles.add(canonical)
                scaffold = murcko_scaffold(canonical)
                if scaffold:
                    eval_scaffolds.add(scaffold)
            if name == "locked_wild_strict" and row.get("paper_group"):
                paper_group = str(row["paper_group"])
                locked_wild_papers.add(paper_group)
                locked_wild_paper_counts[paper_group] += 1
            resolved = resolve_image(str(row.get("image", "")), path, root)
            if resolved is None:
                missing_eval_images[name] += 1
            elif len(image_sample) < args.open_image_sample:
                image_sample.append(resolved)
        eval_unique_smiles[name] = len(set(panel_smiles))
        eval_duplicate_smiles[name] = len(panel_smiles) - len(set(panel_smiles))

    train_counts: dict[str, int] = {}
    train_unique_smiles: dict[str, int] = {}
    train_eval_overlap: dict[str, int] = {}
    train_eval_scaffold_overlap: dict[str, int] = {}
    train_locked_paper_overlap: dict[str, int] = {}
    missing_train_images: Counter[str] = Counter()
    invalid_train_labels: Counter[str] = Counter()
    for name, path in train_paths.items():
        rows = read_jsonl(path)
        train_counts[name] = len(rows)
        canonical_values: set[str] = set()
        scaffold_values: set[str] = set()
        wild_papers: set[str] = set()
        for row in rows:
            canonical = canonicalize(assistant_text(row))
            if canonical is None:
                invalid_train_labels[name] += 1
            else:
                canonical_values.add(canonical)
                scaffold = murcko_scaffold(canonical)
                if scaffold:
                    scaffold_values.add(scaffold)
            meta = row.get("meta", {})
            if meta.get("source") == "molrecbench_wild" and meta.get("paper_group"):
                wild_papers.add(str(meta["paper_group"]))
            image_ref = str((row.get("images") or [""])[0])
            resolved = resolve_image(image_ref, path, root)
            if resolved is None:
                missing_train_images[name] += 1
            elif len(image_sample) < args.open_image_sample:
                image_sample.append(resolved)
        train_unique_smiles[name] = len(canonical_values)
        train_eval_overlap[name] = len(canonical_values & eval_smiles)
        train_eval_scaffold_overlap[name] = len(scaffold_values & eval_scaffolds)
        train_locked_paper_overlap[name] = len(wild_papers & locked_wild_papers)

    bad_open_images = 0
    for path in image_sample:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            bad_open_images += 1

    report = {
        "status": "pass",
        "rdkit_version": rdkit.__version__,
        "missing_required_files": missing_required,
        "train_counts": train_counts,
        "train_unique_canonical_smiles": train_unique_smiles,
        "train_eval_canonical_overlap": train_eval_overlap,
        "train_eval_murcko_scaffold_overlap_diagnostic": train_eval_scaffold_overlap,
        "train_locked_wild_paper_overlap": train_locked_paper_overlap,
        "invalid_train_labels": dict(invalid_train_labels),
        "missing_train_images": dict(missing_train_images),
        "eval_counts": eval_counts,
        "eval_unique_canonical_smiles_by_panel": eval_unique_smiles,
        "eval_duplicate_canonical_smiles_by_panel": eval_duplicate_smiles,
        "locked_wild_paper_count": len(locked_wild_papers),
        "locked_wild_max_images_per_paper": max(locked_wild_paper_counts.values(), default=0),
        "eval_unique_canonical_smiles": len(eval_smiles),
        "missing_eval_images": dict(missing_eval_images),
        "opened_image_sample": len(image_sample),
        "bad_open_images": bad_open_images,
    }
    failures = (
        len(missing_required)
        + sum(train_eval_overlap.values())
        + sum(train_locked_paper_overlap.values())
        + sum(invalid_train_labels.values())
        + sum(missing_train_images.values())
        + sum(missing_eval_images.values())
        + eval_duplicate_smiles.get("locked_wild_strict", 0)
        + int(len(locked_wild_papers) < 60)
        + int(max(locked_wild_paper_counts.values(), default=0) > 5)
        + bad_open_images
    )
    if failures:
        report["status"] = "fail"
        report["failure_count"] = failures
    hash_paths = [
        *required_files,
        *train_paths.values(),
        *eval_paths.values(),
        v3 / "evidence/dataset_build_report.json",
    ]
    report["artifact_sha256"] = {
        str(path.relative_to(v3)).replace("\\", "/"): sha256_file(path)
        for path in hash_paths
        if path.is_file()
    }
    output = v3 / "evidence/workspace_verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
