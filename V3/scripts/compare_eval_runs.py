#!/usr/bin/env python3
"""Paired comparison of two detailed OCSR evaluation runs.

Uses paired bootstrap confidence intervals and an exact McNemar test so a small
local score increase is not automatically treated as a real model improvement.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        identifier = str(row.get("id", ""))
        if not identifier:
            raise ValueError(f"Missing id at {path}:{line_number}")
        if identifier in rows:
            raise ValueError(f"Duplicate id {identifier!r} in {path}")
        rows[identifier] = row
    return rows


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    baseline: list[float], candidate: list[float], iterations: int, seed: int
) -> dict[str, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired bootstrap requires two non-empty equal-length vectors")
    rng = random.Random(seed)
    size = len(baseline)
    deltas: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(size):
            index = rng.randrange(size)
            total += candidate[index] - baseline[index]
        deltas.append(total / size)
    return {
        "delta_mean": sum(c - b for b, c in zip(baseline, candidate)) / size,
        "ci95_low": percentile(deltas, 0.025),
        "ci95_high": percentile(deltas, 0.975),
        "probability_delta_gt_zero": sum(delta > 0 for delta in deltas) / iterations,
    }


def mcnemar_exact(baseline: list[bool], candidate: list[bool]) -> dict[str, Any]:
    baseline_only = sum(b and not c for b, c in zip(baseline, candidate))
    candidate_only = sum(c and not b for b, c in zip(baseline, candidate))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(baseline_only, candidate_only)
        p_value = min(
            1.0,
            2.0 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant),
        )
    return {
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def metric_summary(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    field: str,
    iterations: int,
    seed: int,
    cluster_field: str,
) -> dict[str, Any]:
    if cluster_field:
        grouped_baseline: dict[str, list[float]] = defaultdict(list)
        grouped_candidate: dict[str, list[float]] = defaultdict(list)
        for index, (baseline_row, candidate_row) in enumerate(zip(baseline_rows, candidate_rows)):
            cluster = str(baseline_row.get(cluster_field) or baseline_row.get("id") or index)
            candidate_cluster = str(candidate_row.get(cluster_field) or candidate_row.get("id") or index)
            if cluster != candidate_cluster:
                raise ValueError(f"Cluster mismatch for paired row {baseline_row.get('id')}")
            grouped_baseline[cluster].append(float(bool(baseline_row.get(field))))
            grouped_candidate[cluster].append(float(bool(candidate_row.get(field))))
        clusters = sorted(grouped_baseline)
        baseline = [sum(grouped_baseline[key]) / len(grouped_baseline[key]) for key in clusters]
        candidate = [sum(grouped_candidate[key]) / len(grouped_candidate[key]) for key in clusters]
        unit = cluster_field
    else:
        baseline = [float(bool(row.get(field))) for row in baseline_rows]
        candidate = [float(bool(row.get(field))) for row in candidate_rows]
        unit = "sample_id"
    return {
        "baseline": sum(baseline) / len(baseline),
        "candidate": sum(candidate) / len(candidate),
        "independent_units": len(baseline),
        "resampling_unit": unit,
        **paired_bootstrap(baseline, candidate, iterations, seed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-details", type=Path, required=True)
    parser.add_argument("--candidate-details", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--cluster-field",
        default="",
        help="Resample independent clusters (for example paper_group or structure_id) instead of images.",
    )
    args = parser.parse_args()

    baseline = read_by_id(args.baseline_details)
    candidate = read_by_id(args.candidate_details)
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if baseline_ids != candidate_ids:
        raise ValueError(
            f"Run IDs differ: baseline_only={len(baseline_ids-candidate_ids)}, "
            f"candidate_only={len(candidate_ids-baseline_ids)}"
        )
    ids = sorted(baseline_ids)
    baseline_rows = [baseline[identifier] for identifier in ids]
    candidate_rows = [candidate[identifier] for identifier in ids]

    report: dict[str, Any] = {
        "n": len(ids),
        "baseline": str(args.baseline_details),
        "candidate": str(args.candidate_details),
        "cluster_field": args.cluster_field,
        "canonical_exact": metric_summary(
            baseline_rows,
            candidate_rows,
            "canonical_exact_match",
            args.bootstrap_iterations,
            args.seed,
            args.cluster_field,
        ),
        "valid_smiles": metric_summary(
            baseline_rows,
            candidate_rows,
            "valid_smiles",
            args.bootstrap_iterations,
            args.seed + 1,
            args.cluster_field,
        ),
    }
    if args.cluster_field:
        report["mcnemar_exact"] = {
            "skipped": True,
            "reason": "Image-level McNemar assumes independent pairs; clustered bootstrap is primary for repeated images/papers.",
        }
    else:
        report["mcnemar_exact"] = mcnemar_exact(
            [bool(row.get("canonical_exact_match")) for row in baseline_rows],
            [bool(row.get("canonical_exact_match")) for row in candidate_rows],
        )

    regressions: list[str] = []
    improvements: list[str] = []
    grouped: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for identifier, base, cand in zip(ids, baseline_rows, candidate_rows):
        base_ok = bool(base.get("canonical_exact_match"))
        cand_ok = bool(cand.get("canonical_exact_match"))
        if base_ok and not cand_ok:
            regressions.append(identifier)
        elif cand_ok and not base_ok:
            improvements.append(identifier)
        for field in ("source", "difficulty", "task_type"):
            value = str(base.get(field, "unknown"))
            bucket = grouped[field][value]
            bucket["n"] = bucket.get("n", 0) + 1
            bucket["baseline_correct"] = bucket.get("baseline_correct", 0) + int(base_ok)
            bucket["candidate_correct"] = bucket.get("candidate_correct", 0) + int(cand_ok)

    report["improvement_ids"] = improvements
    report["regression_ids"] = regressions
    report["transition_counts"] = dict(
        Counter(
            f"{int(bool(base.get('canonical_exact_match')))}->{int(bool(cand.get('canonical_exact_match')))}"
            for base, cand in zip(baseline_rows, candidate_rows)
        )
    )
    report["groups"] = grouped
    report["selection_gate"] = {
        "pass": (
            report["canonical_exact"]["ci95_low"] >= -0.005
            and report["valid_smiles"]["delta_mean"] >= -0.005
            and len(improvements) >= len(regressions)
        ),
        "rule": "exact CI lower bound >= -0.5pp, validity delta >= -0.5pp, descriptive wins >= regressions",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
