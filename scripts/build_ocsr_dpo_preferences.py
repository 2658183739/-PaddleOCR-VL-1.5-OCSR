from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_PROMPT = "OCR: Output only the canonical SMILES string for the molecule shown in the image."
DEFAULT_TARGET_FIELDS = ("ground_truth.smiles", "canonical_smiles", "smiles")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    items = []
    for chunk in value.split(","):
        text = chunk.strip()
        if text:
            items.append(text)
    return tuple(items)


def normalize_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def get_nested(record: dict, path: str):
    value = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def extract_target_text(record: dict, field_order: tuple[str, ...]) -> tuple[str, str]:
    for field in field_order:
        value = get_nested(record, field)
        if value is None:
            continue
        text = normalize_text(value)
        if text:
            return text, field
    return "", ""


def canonicalize_smiles(smiles_text: str) -> str | None:
    text = str(smiles_text or "").strip()
    if not text:
        return None
    text = re.sub(r"\s+", "", text)
    try:
        from rdkit import Chem
    except Exception:
        return None

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def resolve_existing_path(value: str, project_root: Path | None, benchmark_path: Path | None) -> Path:
    text = normalize_text(value)
    if not text:
        return Path("")
    path = Path(text)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if project_root is not None:
            candidates.append(project_root / path)
        if benchmark_path is not None:
            candidates.append(benchmark_path.parent / path)
            candidates.append(benchmark_path.parent.parent / path)
            candidates.append(benchmark_path.parent.parent.parent / path)
        candidates.append(path)

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except Exception:
            continue

    return path.resolve() if path.is_absolute() else path


def to_output_relative_path(path: Path, output_dir: Path) -> str:
    if not str(path):
        return ""
    try:
        rel = os.path.relpath(path, output_dir)
        return Path(rel).as_posix()
    except Exception:
        return path.as_posix()


def normalize_prompt(prompt: str) -> str:
    text = normalize_text(prompt)
    return text or DEFAULT_PROMPT


def to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def build_candidate(candidate: dict, prompt: str, run_name: str, origin: str, fallback_vote_count: int = 0) -> dict:
    prediction = normalize_text(candidate.get("prediction"))
    raw_text = normalize_text(candidate.get("raw_text"))
    canonical_prediction = candidate.get("canonical_prediction")
    if canonical_prediction is not None:
        canonical_prediction = normalize_text(canonical_prediction)
        canonical_prediction = canonical_prediction or None
    if canonical_prediction is None and prediction:
        canonical_prediction = canonicalize_smiles(prediction)

    prompt_index = candidate.get("prompt_index")
    tta_index = candidate.get("tta_index")
    generation_index = candidate.get("generation_index")
    return {
        "prompt": prompt,
        "prediction": prediction,
        "canonical_prediction": canonical_prediction,
        "raw_text": raw_text,
        "selection_reason": normalize_text(candidate.get("selection_reason")) or origin,
        "vote_count": to_int(candidate.get("vote_count"), fallback_vote_count),
        "prompt_index": to_int(prompt_index, 999 if prompt_index is None else 0),
        "tta_index": to_int(tta_index, 999 if tta_index is None else 0),
        "generation_index": to_int(generation_index, 999 if generation_index is None else 0),
        "origin": origin,
        "run_name": run_name,
    }


def candidate_signature(candidate: dict) -> tuple[str, str, str, str]:
    return (
        normalize_text(candidate.get("prompt")),
        normalize_text(candidate.get("prediction")),
        normalize_text(candidate.get("canonical_prediction")),
        normalize_text(candidate.get("raw_text")),
    )


def collect_prompt_pool(row: dict, prompt: str, run_name: str) -> list[dict]:
    fallback_vote_count = to_int(row.get("vote_count"), 0)
    pool = []

    selected = build_candidate(row, prompt, run_name, "selected", fallback_vote_count=fallback_vote_count)
    if selected["prediction"] or selected["canonical_prediction"] or selected["raw_text"]:
        pool.append(selected)

    for candidate in row.get("candidates") or []:
        if normalize_prompt(candidate.get("prompt")) != prompt:
            continue
        pool.append(build_candidate(candidate, prompt, run_name, "candidate", fallback_vote_count=fallback_vote_count))

    unique = []
    seen = set()
    for candidate in pool:
        signature = candidate_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(candidate)
    return unique


def choose_rejected_candidate(pool: list[dict], gt_canonical: str, allow_invalid_rejected: bool) -> dict | None:
    scored = []
    for candidate in pool:
        prediction = candidate["prediction"]
        if not prediction:
            continue
        canonical_prediction = candidate.get("canonical_prediction") or canonicalize_smiles(prediction)
        if canonical_prediction == gt_canonical:
            continue
        is_valid = canonical_prediction is not None
        if not is_valid and not allow_invalid_rejected:
            continue
        score = (
            1 if is_valid else 0,
            candidate.get("vote_count", 0),
            -abs(len(prediction) - len(gt_canonical)),
            -candidate.get("prompt_index", 999),
            -candidate.get("tta_index", 999),
            -candidate.get("generation_index", 999),
            -len(prediction),
            prediction,
        )
        scored.append((score, {**candidate, "canonical_prediction": canonical_prediction}))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    best = dict(scored[0][1])
    best["negative_kind"] = "valid_wrong" if best.get("canonical_prediction") else "invalid_wrong"
    return best


def load_benchmark_index(
    benchmark_path: Path,
    target_fields: tuple[str, ...],
    include_sources: set[str],
    exclude_sources: set[str],
) -> tuple[dict[str, dict], Counter]:
    index = {}
    source_counts = Counter()
    for row in read_jsonl(benchmark_path):
        rid = normalize_text(row.get("id"))
        if not rid:
            continue
        source = normalize_text(row.get("source") or row.get("benchmark_track") or row.get("dataset") or row.get("eval_target") or "unknown")
        source_counts[source] += 1
        if include_sources and source not in include_sources:
            continue
        if source in exclude_sources:
            continue
        target_text, target_field = extract_target_text(row, target_fields)
        canonical_target = canonicalize_smiles(target_text)
        index[rid] = {
            "id": rid,
            "source": source,
            "target_text": target_text,
            "target_field": target_field,
            "target_canonical": canonical_target,
            "image_path": row.get("image_path") or row.get("image") or "",
            "task_type": row.get("task_type") or "",
            "difficulty": row.get("difficulty") or "",
            "benchmark_track": row.get("benchmark_track") or "",
            "eval_target": row.get("eval_target") or "",
        }
    return index, source_counts


def build_preference_pairs(
    project_root: Path,
    output_dir: Path,
    benchmark_path: Path,
    prediction_paths: list[Path],
    target_fields: tuple[str, ...],
    include_sources: set[str],
    exclude_sources: set[str],
    allow_invalid_rejected: bool,
    max_pairs_per_source: int,
    max_pairs_total: int,
) -> tuple[list[dict], dict]:
    benchmark_index, benchmark_source_counts = load_benchmark_index(
        benchmark_path=benchmark_path,
        target_fields=target_fields,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
    )

    outputs = []
    stats = Counter()
    by_source = Counter()
    by_run = Counter()
    by_negative_kind = Counter()
    by_target_field = Counter()
    per_source_pairs = Counter()

    for pred_path in prediction_paths:
        run_name = pred_path.stem
        rows = list(read_jsonl(pred_path))
        stats["prediction_rows"] += len(rows)
        for row in rows:
            rid = normalize_text(row.get("id"))
            if not rid:
                stats["skipped_missing_id"] += 1
                continue

            bench = benchmark_index.get(rid)
            if bench is None:
                stats["skipped_missing_benchmark"] += 1
                continue

            source = bench["source"]
            if include_sources and source not in include_sources:
                stats["skipped_source_filtered"] += 1
                continue
            if source in exclude_sources:
                stats["skipped_source_excluded"] += 1
                continue

            if max_pairs_total > 0 and len(outputs) >= max_pairs_total:
                stats["skipped_total_cap"] += 1
                break
            if max_pairs_per_source > 0 and per_source_pairs[source] >= max_pairs_per_source:
                stats["skipped_source_cap"] += 1
                continue

            gt_text = bench["target_text"]
            gt_canonical = bench["target_canonical"]
            if not gt_text or not gt_canonical:
                stats["skipped_missing_target"] += 1
                continue

            prompt = normalize_prompt(row.get("prompt") or DEFAULT_PROMPT)
            pool = collect_prompt_pool(row, prompt, run_name)
            rejected = choose_rejected_candidate(pool, gt_canonical, allow_invalid_rejected)
            if rejected is None:
                stats["skipped_no_rejected"] += 1
                continue

            source_image_path = resolve_existing_path(
                str(bench["image_path"] or row.get("image_path") or row.get("image") or ""),
                project_root,
                benchmark_path,
            )
            image_path = to_output_relative_path(source_image_path, output_dir)
            pair_uid = hashlib.sha1(f"{run_name}|{rid}|{prompt}".encode("utf-8")).hexdigest()[:16]

            chosen = {
                "text": gt_canonical,
                "canonical": gt_canonical,
                "origin": "gold",
            }
            pair = {
                "pair_uid": pair_uid,
                "id": rid,
                "run_name": run_name,
                "prediction_file": str(pred_path.as_posix()),
                "source": source,
                "target_field": bench["target_field"],
                "benchmark_track": bench["benchmark_track"],
                "eval_target": bench["eval_target"],
                "task_type": bench["task_type"],
                "difficulty": bench["difficulty"],
                "prompt": prompt,
            "messages": [{"role": "user", "content": f"<image>{prompt}"}],
            "images": [image_path],
            "image": image_path,
            "source_image_path": source_image_path.as_posix(),
            "chosen": chosen["text"],
            "rejected": rejected["prediction"],
                "chosen_canonical": chosen["canonical"],
                "rejected_canonical": rejected.get("canonical_prediction"),
                "rejected_kind": rejected["negative_kind"],
                "candidate_count": len(pool),
                "valid_candidate_count": sum(1 for item in pool if item.get("canonical_prediction")),
                "selected_prediction": normalize_text(row.get("prediction")),
                "selected_canonical_prediction": normalize_text(row.get("canonical_prediction")) or None,
                "selected_selection_reason": normalize_text(row.get("selection_reason")) or None,
                "selected_vote_count": to_int(row.get("vote_count"), 0),
            }
            outputs.append(pair)
            by_source[source] += 1
            by_run[run_name] += 1
            by_negative_kind[rejected["negative_kind"]] += 1
            by_target_field[bench["target_field"]] += 1
            per_source_pairs[source] += 1
            stats["usable_pairs"] += 1

        if max_pairs_total > 0 and len(outputs) >= max_pairs_total:
            break

    summary = {
        "benchmark": str(benchmark_path.as_posix()),
        "prediction_files": [str(path.as_posix()) for path in prediction_paths],
        "target_fields": list(target_fields),
        "include_sources": sorted(include_sources),
        "exclude_sources": sorted(exclude_sources),
        "allow_invalid_rejected": allow_invalid_rejected,
        "max_pairs_per_source": max_pairs_per_source,
        "max_pairs_total": max_pairs_total,
        "totals": dict(stats),
        "benchmark_source_counts": dict(benchmark_source_counts),
        "usable_by_source": dict(by_source),
        "usable_by_run": dict(by_run),
        "usable_by_negative_kind": dict(by_negative_kind),
        "usable_by_target_field": dict(by_target_field),
    }
    return outputs, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--benchmark-jsonl", required=True)
    parser.add_argument("--prediction-jsonl", nargs="+", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--target-field-order", default=",".join(DEFAULT_TARGET_FIELDS))
    parser.add_argument("--include-sources", nargs="*", default=[])
    parser.add_argument("--exclude-sources", nargs="*", default=[])
    parser.add_argument("--allow-invalid-rejected", action="store_true")
    parser.add_argument("--max-pairs-per-source", type=int, default=0)
    parser.add_argument("--max-pairs-total", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    benchmark_path = (project_root / args.benchmark_jsonl).resolve()
    prediction_paths = [(project_root / item).resolve() for item in args.prediction_jsonl]
    output_path = (project_root / args.output_jsonl).resolve()
    output_dir = output_path.parent
    report_path = (project_root / args.report_json).resolve()
    target_fields = tuple(field.strip() for field in split_csv(args.target_field_order) if field.strip()) or DEFAULT_TARGET_FIELDS
    include_sources = {normalize_text(item) for item in args.include_sources if normalize_text(item)}
    exclude_sources = {normalize_text(item) for item in args.exclude_sources if normalize_text(item)}

    pairs, summary = build_preference_pairs(
        project_root=project_root,
        output_dir=output_dir,
        benchmark_path=benchmark_path,
        prediction_paths=prediction_paths,
        target_fields=target_fields,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        allow_invalid_rejected=args.allow_invalid_rejected,
        max_pairs_per_source=max(0, args.max_pairs_per_source),
        max_pairs_total=max(0, args.max_pairs_total),
    )

    write_jsonl(output_path, pairs)
    write_json(report_path, summary)
    print(json.dumps({"output": str(output_path), "report": str(report_path), "pairs": len(pairs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
