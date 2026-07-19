#!/usr/bin/env python3
"""Build deterministic V3 OCSR training mixtures and evaluation manifests.

The script keeps canonical-SMILES SFT separate from MolRecBench symbolic labels,
filters train/eval overlap at molecule level, and generates conservative offline
augmentations for real-world training images. It is intentionally idempotent.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageEnhance, ImageFilter

try:
    import rdkit
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.error")
except ImportError:  # pragma: no cover - reported explicitly at runtime
    rdkit = None
    Chem = None
    MurckoScaffold = None


SEED = 20260717
CANONICAL_PROMPT = (
    "<image>OCR: Output only the canonical SMILES string for the molecule "
    "shown in the image."
)
REAL_SOURCES = {"real_world", "auto_weak_real_world"}
HIGH_RISK_DIFFICULTIES = {
    "chinese_exam",
    "degraded_scan",
    "document_embed",
    "handwritten",
    "journal_fig",
    "long_molecule",
    "multi_grid",
    "page_level",
    "photo",
    "scan",
    "stereo_focused",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def assistant_text(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages", [])):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def ground_truth_smiles(row: dict[str, Any]) -> str:
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, dict):
        return str(ground_truth.get("smiles", "")).strip()
    return str(row.get("smiles", "")).strip()


@lru_cache(maxsize=100000)
def canonicalize(smiles: str) -> str | None:
    if Chem is None:
        raise RuntimeError("RDKit is required. Use .conda_rdkit/python.exe on Windows.")
    text = re.sub(r"\s+", "", str(smiles or ""))
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
def murcko_scaffold(smiles: str) -> str | None:
    canonical = canonicalize(smiles)
    if canonical is None or MurckoScaffold is None:
        return None
    mol = Chem.MolFromSmiles(canonical)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def stable_key(seed: int, row: dict[str, Any]) -> str:
    meta = row.get("meta", {})
    identifier = meta.get("id") or row.get("id") or json.dumps(row, sort_keys=True)
    return hashlib.sha256(f"{seed}|{identifier}".encode("utf-8")).hexdigest()


def deterministic_shuffle(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def resolve_image(path_text: str, manifest: Path, project_root: Path) -> Path:
    raw = Path(path_text)
    candidates = [raw] if raw.is_absolute() else [
        project_root / raw,
        manifest.parent / raw,
        manifest.parent.parent / raw,
        manifest.parent.parent.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve image {path_text!r} from {manifest}")


def eval_canonical_set(paths: Iterable[Path]) -> tuple[set[str], Counter[str]]:
    values: set[str] = set()
    status: Counter[str] = Counter()
    for path in paths:
        for row in read_jsonl(path):
            value = canonicalize(ground_truth_smiles(row))
            if value is None:
                status["noncanonical_or_symbolic"] += 1
            else:
                values.add(value)
                status["canonical"] += 1
    return values, status


def paper_group_from_id(identifier: str) -> str:
    match = re.match(r"^(.*)_\d+_figure_", identifier)
    if match:
        return match.group(1)
    if "_figure_" in identifier:
        return identifier.split("_figure_", 1)[0]
    return identifier


def wild_difficulty(row: dict[str, Any]) -> str:
    labels = row.get("hardcase_label") or []
    atom_count = len(row.get("symbols") or [])
    if len(labels) >= 6 or atom_count >= 45:
        return "hard"
    if len(labels) >= 3 or atom_count >= 25:
        return "medium_hard"
    if atom_count >= 12:
        return "medium"
    return "easy"


def wild_atom_bin(row: dict[str, Any]) -> str:
    atom_count = len(row.get("symbols") or [])
    if atom_count <= 5:
        return "atom_001_005"
    if atom_count <= 15:
        return "atom_006_015"
    if atom_count <= 30:
        return "atom_016_030"
    if atom_count <= 60:
        return "atom_031_060"
    return "atom_061_plus"


def make_wild_train_record(row: dict[str, Any], canonical: str, paper_group: str) -> dict[str, Any]:
    filename = Path(str(row.get("local_image_path", row.get("id", "")))).name
    return {
        "images": [f"../assets/molrecbench_wild_v1/images/{filename}"],
        "messages": [
            {"role": "user", "content": CANONICAL_PROMPT},
            {"role": "assistant", "content": canonical},
        ],
        "meta": {
            "id": row.get("id", filename),
            "source": "molrecbench_wild",
            "source_url_or_doc": "https://huggingface.co/datasets/opendatalab/MolRecBench-Wild",
            "paper_group": paper_group,
            "task_type": "molecule_structure_recognition",
            "eval_target": "canonical_smiles",
            "label_policy": "RDKit canonicalized strict MolRecBench subset",
            "v3_filter": "valid_single_molecule_no_dummy_atoms",
            "hardcase_label": row.get("hardcase_label") or [],
            "difficulty": wild_difficulty(row),
            "atom_count": len(row.get("symbols") or []),
            "smiles_length": len(canonical),
        },
    }


def make_wild_eval_record(
    row: dict[str, Any], canonical: str | None, paper_group: str
) -> dict[str, Any]:
    filename = Path(str(row.get("local_image_path", row.get("id", "")))).name
    return {
        "id": row.get("id", filename),
        "source": "molrecbench_wild",
        "paper_group": paper_group,
        "structure_id": hashlib.sha256((canonical or str(row.get("smiles", ""))).encode("utf-8")).hexdigest()[:20],
        "image": f"V3/data/assets/molrecbench_wild_v1/images/{filename}",
        "task_type": "molecule_structure_recognition",
        "image_type": "real_world_article_crop",
        "difficulty": wild_difficulty(row),
        "ground_truth": {"smiles": canonical or str(row.get("smiles", ""))},
        "eval_target": "canonical_smiles" if canonical else "molrecbench_symbolic_smiles",
        "license": "Apache-2.0",
        "source_url_or_doc": "https://huggingface.co/datasets/opendatalab/MolRecBench-Wild",
        "qc_status": "pending_manual_review",
        "benchmark_role": "locked_final_test" if canonical else "locked_symbolic_test",
        "benchmark_track": "wild_strict_v3" if canonical else "wild_symbolic_v3",
        "hardcase_label": row.get("hardcase_label") or [],
        "atom_count": len(row.get("symbols") or []),
        "label_policy": "RDKit canonicalized MolRecBench paper-group holdout" if canonical else "MolRecBench symbolic label",
        "automated_qc": {
            "rdkit_single_molecule": canonical is not None,
            "human_review": "pending",
        },
    }


def build_grouped_wild_split(
    rows: list[dict[str, Any]],
    legacy_dev_smiles: set[str],
    eval_target: int,
    train_cap: int,
    eval_max_per_paper: int,
    eval_min_papers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    strict_pool: list[tuple[dict[str, Any], str, str]] = []
    symbolic_by_group: dict[str, list[dict[str, Any]]] = {}
    stats: Counter[str] = Counter()
    for row in rows:
        identifier = str(row.get("id", ""))
        paper_group = paper_group_from_id(identifier)
        canonical = canonicalize(str(row.get("smiles", "")))
        if canonical is None:
            symbolic_by_group.setdefault(paper_group, []).append(row)
            stats["symbolic_or_invalid"] += 1
            continue
        if canonical in legacy_dev_smiles:
            stats["legacy_dev_molecule_overlap"] += 1
            continue
        strict_pool.append((row, canonical, paper_group))

    groups: dict[str, list[tuple[dict[str, Any], str, str]]] = {}
    for item in strict_pool:
        groups.setdefault(item[2], []).append(item)
    if eval_target <= 0 or eval_target >= len(strict_pool):
        raise ValueError(f"wild eval target must be between 1 and {len(strict_pool)-1}")

    difficulty_total = Counter(wild_difficulty(row) for row, _, _ in strict_pool)
    atom_total = Counter(wild_atom_bin(row) for row, _, _ in strict_pool)
    difficulty_target = {
        key: round(value * eval_target / len(strict_pool)) for key, value in difficulty_total.items()
    }
    atom_target = {
        key: round(value * eval_target / len(strict_pool)) for key, value in atom_total.items()
    }
    selected_groups: set[str] = set()
    selected_canonical: set[str] = set()
    difficulty_count: Counter[str] = Counter()
    atom_count: Counter[str] = Counter()
    selected_size = 0

    selected_preview: dict[str, list[tuple[dict[str, Any], str, str]]] = {}

    def unique_group_preview(
        group_rows: list[tuple[dict[str, Any], str, str]],
    ) -> list[tuple[dict[str, Any], str, str]]:
        preview: list[tuple[dict[str, Any], str, str]] = []
        seen = set(selected_canonical)
        for item in sorted(
            group_rows,
            key=lambda candidate: stable_key(
                SEED + 41, {"id": candidate[0].get("id", "")}
            ),
        ):
            canonical = item[1]
            if canonical in seen:
                continue
            seen.add(canonical)
            preview.append(item)
            if len(preview) >= eval_max_per_paper:
                break
        return preview

    while selected_size < eval_target or len(selected_groups) < eval_min_papers:
        best_group = None
        best_score = None
        for group_name, group_rows in groups.items():
            if group_name in selected_groups:
                continue
            preview = unique_group_preview(group_rows)
            if not preview:
                continue
            group_difficulty = Counter(wild_difficulty(row) for row, _, _ in preview)
            group_atoms = Counter(wild_atom_bin(row) for row, _, _ in preview)
            deficit_gain = sum(
                min(count, max(0, difficulty_target.get(key, 0) - difficulty_count[key]))
                for key, count in group_difficulty.items()
            )
            deficit_gain += 2 * sum(
                min(count, max(0, atom_target.get(key, 0) - atom_count[key]))
                for key, count in group_atoms.items()
            )
            projected = selected_size + len(preview)
            overshoot_penalty = max(0, projected - eval_target) * 0.25
            tie = stable_key(SEED + 37, {"id": group_name})
            score = (deficit_gain - overshoot_penalty, tie)
            if best_score is None or score > best_score:
                best_score = score
                best_group = group_name
        if best_group is None:
            break
        selected_groups.add(best_group)
        chosen = unique_group_preview(groups[best_group])
        selected_preview[best_group] = chosen
        selected_canonical.update(item[1] for item in chosen)
        selected_size += len(chosen)
        difficulty_count.update(wild_difficulty(row) for row, _, _ in chosen)
        atom_count.update(wild_atom_bin(row) for row, _, _ in chosen)

    eval_items = [item for group in sorted(selected_groups) for item in selected_preview[group]]
    eval_canonical = {canonical for _, canonical, _ in eval_items}
    train_items_before_molecule_filter = [
        item for item in strict_pool if item[2] not in selected_groups
    ]
    train_items = [
        item for item in train_items_before_molecule_filter if item[1] not in eval_canonical
    ]
    stats["train_rows_removed_for_locked_molecule_overlap"] = (
        len(train_items_before_molecule_filter) - len(train_items)
    )
    train_items.sort(key=lambda item: stable_key(SEED, {"id": item[0].get("id", "")}))
    if train_cap > 0 and len(train_items) > train_cap:
        stats["valid_train_over_cap"] = len(train_items) - train_cap
        train_items = train_items[:train_cap]

    train_records = [make_wild_train_record(row, canonical, group) for row, canonical, group in train_items]
    eval_records = [make_wild_eval_record(row, canonical, group) for row, canonical, group in eval_items]
    symbolic_records = [
        make_wild_eval_record(row, None, group)
        for group in sorted(selected_groups)
        for row in symbolic_by_group.get(group, [])
    ]
    eval_ids = {str(row.get("id", "")) for row, _, _ in eval_items}
    split_manifest = [
        {
            "id": row.get("id", ""),
            "paper_group": group,
            "split": (
                "locked_eval"
                if str(row.get("id", "")) in eval_ids
                else "locked_eval_excess"
                if group in selected_groups
                else "train"
            ),
            "canonical_smiles": canonical,
        }
        for row, canonical, group in strict_pool
    ]
    stats["strict_pool"] = len(strict_pool)
    stats["paper_groups_total"] = len(groups)
    stats["paper_groups_eval"] = len(selected_groups)
    stats["strict_eval"] = len(eval_records)
    stats["strict_eval_excess_held_out"] = sum(
        1 for row in split_manifest if row["split"] == "locked_eval_excess"
    )
    stats["eval_max_per_paper"] = eval_max_per_paper
    stats["strict_train"] = len(train_records)
    stats["symbolic_eval_same_papers"] = len(symbolic_records)
    return train_records, eval_records, symbolic_records, stats, split_manifest


def build_legacy_dev_messages(
    rows: list[dict[str, Any]], dataset_name: str
) -> tuple[list[dict[str, Any]], Counter[str]]:
    messages: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for row in rows:
        canonical = canonicalize(ground_truth_smiles(row))
        if canonical is None:
            stats["noncanonical_or_multifragment"] += 1
            continue
        messages.append(
            {
                "images": [f"../eval/{dataset_name}/{row.get('image', '')}"],
                "messages": [
                    {"role": "user", "content": CANONICAL_PROMPT},
                    {"role": "assistant", "content": canonical},
                ],
                "meta": {
                    "id": row.get("id", ""),
                    "structure_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20],
                    "source": row.get("source", "unknown"),
                    "difficulty": row.get("difficulty", "unknown"),
                    "task_type": row.get("task_type", "unknown"),
                    "benchmark_role": "legacy_development",
                },
            }
        )
    stats["accepted"] = len(messages)
    return messages, stats


def build_legacy_dev_labels(
    rows: list[dict[str, Any]], dataset_name: str
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for row in rows:
        canonical = canonicalize(ground_truth_smiles(row))
        if canonical is None:
            continue
        item = copy.deepcopy(row)
        item["ground_truth"] = {"smiles": canonical}
        item["structure_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
        item["image"] = f"V3/data/eval/{dataset_name}/{row.get('image', '')}"
        item["benchmark_role"] = "legacy_development"
        labels.append(item)
    return labels


def build_strict_wild_train(
    rows: list[dict[str, Any]], eval_smiles: set[str], cap: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    accepted: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        canonical = canonicalize(assistant_text(row))
        if canonical is None:
            reasons["symbolic_or_invalid_label"] += 1
            continue
        if canonical in eval_smiles:
            reasons["eval_molecule_overlap"] += 1
            continue
        identifier = str(row.get("meta", {}).get("id", ""))
        unique_key = f"{identifier}|{canonical}"
        if unique_key in seen:
            reasons["duplicate_id_and_label"] += 1
            continue
        seen.add(unique_key)
        clean = copy.deepcopy(row)
        clean["messages"] = [
            {"role": "user", "content": CANONICAL_PROMPT},
            {"role": "assistant", "content": canonical},
        ]
        clean.setdefault("meta", {})["eval_target"] = "canonical_smiles"
        clean["meta"]["label_policy"] = "RDKit canonicalized strict MolRecBench subset"
        clean["meta"]["v3_filter"] = "valid_single_molecule_no_dummy_atoms"
        accepted.append(clean)

    accepted.sort(key=lambda row: stable_key(SEED, row))
    if cap > 0 and len(accepted) > cap:
        reasons["valid_but_over_cap"] += len(accepted) - cap
        accepted = accepted[:cap]
    reasons["accepted"] = len(accepted)
    return accepted, reasons


def build_strict_base_train(
    rows: list[dict[str, Any]], eval_smiles: set[str]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    accepted: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for row in rows:
        canonical = canonicalize(assistant_text(row))
        if canonical is None:
            reasons["multifragment_or_invalid_label"] += 1
            continue
        if canonical in eval_smiles:
            reasons["new_eval_molecule_overlap"] += 1
            continue
        clean = copy.deepcopy(row)
        for message in clean.get("messages", []):
            if message.get("role") == "assistant":
                message["content"] = canonical
        accepted.append(clean)
    reasons["accepted"] = len(accepted)
    return accepted, reasons


def border_fill(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    points = [
        rgb.getpixel((0, 0)),
        rgb.getpixel((max(0, width - 1), 0)),
        rgb.getpixel((0, max(0, height - 1))),
        rgb.getpixel((max(0, width - 1), max(0, height - 1))),
    ]
    return tuple(int(sum(point[channel] for point in points) / len(points)) for channel in range(3))


def augment_image(source: Path, destination: Path, seed: int, variant: int) -> dict[str, Any]:
    rng = random.Random(f"{seed}|{source.as_posix()}|{variant}")
    angle = rng.uniform(-3.0, 3.0)
    brightness = rng.uniform(0.82, 1.15)
    contrast = rng.uniform(0.82, 1.18)
    blur = rng.uniform(0.25, 0.75)
    quality = rng.randint(62, 86)

    with Image.open(source) as opened:
        image = opened.convert("RGB")
        image = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=border_fill(image),
        )
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = image.filter(ImageFilter.GaussianBlur(radius=blur))
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, "JPEG", quality=quality, optimize=True)
    return {
        "rotation_deg": round(angle, 3),
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "gaussian_blur_radius": round(blur, 3),
        "jpeg_quality": quality,
    }


def build_real_augmentations(
    base_rows: list[dict[str, Any]],
    source_manifest: Path,
    project_root: Path,
    v3_root: Path,
    variants: int,
) -> tuple[list[list[dict[str, Any]]], Counter[str]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        meta = row.get("meta", {})
        if meta.get("source") not in REAL_SOURCES:
            continue
        image_ref = str((row.get("images") or [""])[0])
        if not image_ref:
            continue
        unique.setdefault(image_ref, row)

    output: list[list[dict[str, Any]]] = [[] for _ in range(variants)]
    stats: Counter[str] = Counter()
    for image_ref, row in sorted(unique.items()):
        source = resolve_image(image_ref, source_manifest, project_root)
        source_id = str(row.get("meta", {}).get("id", source.stem))
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id)[:160]
        for variant in range(1, variants + 1):
            destination = v3_root / "data" / "assets" / "v3_aug_real" / f"{safe_id}_v{variant}.jpg"
            params = augment_image(source, destination, SEED, variant)
            augmented = copy.deepcopy(row)
            augmented["images"] = [f"../assets/v3_aug_real/{destination.name}"]
            augmented.setdefault("meta", {})["id"] = f"{source_id}__v3aug{variant}"
            augmented["meta"]["source"] = "v3_aug_real_world"
            augmented["meta"]["source_parent"] = row.get("meta", {}).get("source")
            augmented["meta"]["augmentation"] = params
            augmented["meta"]["singleline_policy"] = f"v3_offline_aug_{variant}"
            augmented["meta"]["repeat_index"] = 0
            output[variant - 1].append(augmented)
            stats[f"variant_{variant}"] += 1
    stats["unique_source_images"] = len(unique)
    return output, stats


def summarize_train(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sources: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    policies: Counter[str] = Counter()
    unique_images: set[str] = set()
    unique_smiles: set[str] = set()
    for row in rows:
        meta = row.get("meta", {})
        sources[str(meta.get("source", "unknown"))] += 1
        difficulties[str(meta.get("difficulty", "unknown"))] += 1
        policies[str(meta.get("singleline_policy", "unweighted"))] += 1
        image_ref = str((row.get("images") or [""])[0])
        if image_ref:
            unique_images.add(image_ref)
        canonical = canonicalize(assistant_text(row))
        if canonical:
            unique_smiles.add(canonical)
    total = len(rows)
    return {
        "rows": total,
        "unique_image_refs": len(unique_images),
        "unique_canonical_smiles": len(unique_smiles),
        "source_counts": dict(sorted(sources.items())),
        "source_percent": {
            key: round(value * 100.0 / total, 4) for key, value in sorted(sources.items())
        },
        "difficulty_counts": dict(sorted(difficulties.items())),
        "policy_counts": dict(sorted(policies.items())),
    }


def build_hard_replay(rows: list[dict[str, Any]], cap: int = 7000) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        meta = row.get("meta", {})
        target = assistant_text(row)
        is_hard = (
            str(meta.get("difficulty", "")) in HIGH_RISK_DIFFICULTIES
            or bool(meta.get("contains_stereo"))
            or len(target) >= 100
            or str(meta.get("source", "")) in {"molrecbench_wild", "v3_aug_real_world"}
        )
        if is_hard:
            selected.append(row)
    selected.sort(key=lambda row: stable_key(SEED + 91, row))
    return selected[:cap]


def build_wild_eval_manifests(
    source_rows: list[dict[str, Any]], v3_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    strict: list[dict[str, Any]] = []
    symbolic: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for row in source_rows:
        item = copy.deepcopy(row)
        filename = Path(str(row.get("image", ""))).name
        item["image"] = f"V3/data/eval/molrecbench_wild_300/images/{filename}"
        canonical = canonicalize(ground_truth_smiles(row))
        item["automated_qc"] = {
            "rdkit_single_molecule": canonical is not None,
            "human_review": "pending",
        }
        if canonical is None:
            item["benchmark_track"] = "wild_symbolic_v3"
            symbolic.append(item)
            stats["symbolic_or_invalid"] += 1
        else:
            item["ground_truth"]["smiles"] = canonical
            item["eval_target"] = "canonical_smiles"
            item["benchmark_track"] = "wild_strict_v3"
            item["label_policy"] = "RDKit canonicalized MolRecBench held-out subset"
            strict.append(item)
            stats["strict_canonical"] += 1
    write_jsonl(v3_root / "data" / "eval" / "wild_strict_v3" / "labels.jsonl", strict)
    write_jsonl(v3_root / "data" / "eval" / "wild_symbolic_v3" / "labels.jsonl", symbolic)
    return strict, symbolic, stats


def write_qc_sheet(
    path: Path,
    main_rows: list[dict[str, Any]],
    wild_rows: list[dict[str, Any]],
) -> None:
    fields = [
        "panel",
        "benchmark_role",
        "sample_id",
        "source",
        "difficulty",
        "image",
        "label",
        "automated_status",
        "reviewer_1",
        "reviewer_1_decision",
        "reviewer_1_reason",
        "reviewer_2",
        "reviewer_2_decision",
        "reviewer_2_reason",
        "final_decision",
        "review_time",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for panel, role, rows in (
            ("core_767", "legacy_development", main_rows),
            ("wild_strict_v3", "locked_final_test", wild_rows),
        ):
            for row in rows:
                writer.writerow(
                    {
                        "panel": panel,
                        "benchmark_role": role,
                        "sample_id": row.get("id", ""),
                        "source": row.get("source", ""),
                        "difficulty": row.get("difficulty", ""),
                        "image": row.get("image", ""),
                        "label": ground_truth_smiles(row),
                        "automated_status": "pass",
                        "reviewer_1_decision": "pending",
                        "reviewer_2_decision": "pending",
                        "final_decision": "pending",
                    }
                )


def write_mix_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    source_names = sorted(
        {source for summary in summaries.values() for source in summary["source_counts"]}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mixture", "total_rows", *source_names])
        for name, summary in summaries.items():
            writer.writerow(
                [name, summary["rows"], *[summary["source_counts"].get(s, 0) for s in source_names]]
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--wild-cap", type=int, default=1200)
    parser.add_argument("--wild-eval-target", type=int, default=300)
    parser.add_argument("--wild-eval-max-per-paper", type=int, default=5)
    parser.add_argument("--wild-eval-min-papers", type=int, default=60)
    parser.add_argument("--augmentation-variants", type=int, default=2)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    v3_root = project_root / "V3"
    source_data = v3_root / "data/source"
    base_path = source_data / "train_singleline_rw_v2_clean_weighted_a100_messages.jsonl"
    wild_all_path = source_data / "molrecbench_wild_all_annotation.jsonl"
    core_eval_path = v3_root / "data/eval/canonical_smiles_main_v1/annotations/labels.jsonl"
    region_eval_path = v3_root / "data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl"

    required = [base_path, wild_all_path, core_eval_path, region_eval_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing copied V3 source inputs:\n" + "\n".join(missing))
    if Chem is None:
        raise RuntimeError("RDKit is required to build leakage-safe V3 datasets.")

    base_rows = read_jsonl(base_path)
    wild_all_rows = read_jsonl(wild_all_path)
    core_eval_rows = read_jsonl(core_eval_path)
    region_eval_rows = read_jsonl(region_eval_path)

    legacy_dev_smiles, legacy_dev_parse_stats = eval_canonical_set(
        [core_eval_path, region_eval_path]
    )
    strict_wild, strict_eval, symbolic_eval, wild_split_stats, split_manifest = build_grouped_wild_split(
        wild_all_rows,
        legacy_dev_smiles,
        args.wild_eval_target,
        args.wild_cap,
        args.wild_eval_max_per_paper,
        args.wild_eval_min_papers,
    )
    locked_eval_smiles = {
        canonical
        for canonical in (canonicalize(ground_truth_smiles(row)) for row in strict_eval)
        if canonical
    }
    all_heldout_smiles = legacy_dev_smiles | locked_eval_smiles
    strict_base, base_train_stats = build_strict_base_train(base_rows, all_heldout_smiles)
    augmentations, augmentation_stats = build_real_augmentations(
        strict_base,
        base_path,
        project_root,
        v3_root,
        args.augmentation_variants,
    )
    if len(augmentations) < 2:
        raise ValueError("V3 mixtures require at least two augmentation variants")

    mixture_a = deterministic_shuffle(strict_base, SEED + 1)
    mixture_wild_only = deterministic_shuffle(strict_base + strict_wild, SEED + 2)
    mixture_aug_only = deterministic_shuffle(strict_base + augmentations[0], SEED + 3)
    mixture_b = deterministic_shuffle(strict_base + strict_wild + augmentations[0], SEED + 4)
    mixture_c = deterministic_shuffle(
        strict_base + strict_wild + augmentations[0] + augmentations[1], SEED + 5
    )
    hard_replay = deterministic_shuffle(build_hard_replay(mixture_b), SEED + 6)
    mixture_b_scaffolds = {
        scaffold
        for scaffold in (murcko_scaffold(assistant_text(row)) for row in mixture_b)
        if scaffold
    }
    scaffold_novel_eval = [
        row
        for row in strict_eval
        if (scaffold := murcko_scaffold(ground_truth_smiles(row))) is not None
        and scaffold not in mixture_b_scaffolds
    ]

    dev_core, dev_core_stats = build_legacy_dev_messages(
        core_eval_rows, "canonical_smiles_main_v1"
    )
    dev_region, dev_region_stats = build_legacy_dev_messages(
        region_eval_rows, "ocsr_realworld_mixed_eval_v1p1"
    )
    dev_core_labels = build_legacy_dev_labels(core_eval_rows, "canonical_smiles_main_v1")
    dev_region_labels = build_legacy_dev_labels(
        region_eval_rows, "ocsr_realworld_mixed_eval_v1p1"
    )

    output_dir = v3_root / "data" / "sft_materialized"
    outputs = {
        "A_control": ("train_v3_a_control.jsonl", mixture_a),
        "D_wild_only": ("train_v3_d_wild_only.jsonl", mixture_wild_only),
        "E_aug_only": ("train_v3_e_aug_only.jsonl", mixture_aug_only),
        "B_recommended": ("train_v3_b_recommended.jsonl", mixture_b),
        "C_real_heavy": ("train_v3_c_real_heavy.jsonl", mixture_c),
        "hard_replay_seed": ("train_v3_hard_replay_seed.jsonl", hard_replay),
    }
    summaries: dict[str, dict[str, Any]] = {}
    for name, (filename, rows) in outputs.items():
        write_jsonl(output_dir / filename, rows)
        summaries[name] = summarize_train(rows)

    write_jsonl(output_dir / "dev_legacy_core_strict_messages.jsonl", dev_core)
    write_jsonl(output_dir / "dev_legacy_region_strict_messages.jsonl", dev_region)
    write_jsonl(v3_root / "data/eval/dev_legacy_core_strict/labels.jsonl", dev_core_labels)
    write_jsonl(v3_root / "data/eval/dev_legacy_region_strict/labels.jsonl", dev_region_labels)
    write_jsonl(v3_root / "data/eval/wild_strict_v3/labels.jsonl", strict_eval)
    write_jsonl(
        v3_root / "data/eval/wild_strict_scaffold_novel_v3/labels.jsonl",
        scaffold_novel_eval,
    )
    write_jsonl(v3_root / "data/eval/wild_symbolic_v3/labels.jsonl", symbolic_eval)
    write_jsonl(v3_root / "evidence/wild_paper_group_split.jsonl", split_manifest)
    write_qc_sheet(v3_root / "qc" / "eval_manual_review.csv", core_eval_rows, strict_eval)
    write_mix_csv(v3_root / "evidence" / "mixture_counts.csv", summaries)

    report = {
        "seed": SEED,
        "rdkit_version": getattr(rdkit, "__version__", None),
        "inputs": {"base_rows": len(base_rows), "wild_all_rows": len(wild_all_rows)},
        "base_train_filter": dict(base_train_stats),
        "legacy_dev_parse_stats": dict(legacy_dev_parse_stats),
        "legacy_dev_unique_canonical_smiles": len(legacy_dev_smiles),
        "locked_eval_unique_canonical_smiles": len(locked_eval_smiles),
        "wild_paper_group_split": dict(wild_split_stats),
        "dev_manifests": {
            "core": dict(dev_core_stats),
            "region": dict(dev_region_stats),
        },
        "augmentation": dict(augmentation_stats),
        "mixtures": summaries,
        "wild_eval": {
            "strict_rows": len(strict_eval),
            "symbolic_rows": len(symbolic_eval),
            "scaffold_novel_rows": len(scaffold_novel_eval),
            "human_review_status": "pending",
        },
        "leakage_policy": [
            "base and strict MolRecBench train rows are excluded on canonical molecule overlap with all development and locked-test panels",
            "MolRecBench is split by paper_group before row-level sampling so no paper appears in both train and locked test",
            "legacy core and region panels are development-only because they were used during V2-1 tuning",
        ],
    }
    report_path = v3_root / "evidence" / "dataset_build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
