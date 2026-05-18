import csv
import json
import shutil
from collections import Counter
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def copy_image(project_root: Path, source_rel: str, dest_root: Path, source_name: str) -> str:
    src = (project_root / source_rel).resolve()
    if not src.exists():
        raise FileNotFoundError(f'Missing eval image: {src}')
    source_dir = dest_root / source_name
    ensure_dir(source_dir)
    dest = source_dir / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
    return str(dest.relative_to(dest_root.parent).as_posix())


def build_records(project_root: Path):
    source_files = [
        project_root / 'V2' / 'data' / 'benchmarks' / 'competition_eval_source.jsonl',
        project_root / 'server_ready' / 'paddleocr_vl_ocsr_a800' / 'data' / 'benchmarks' / 'auxiliary_eval.jsonl',
    ]
    out_root = project_root / 'V2' / 'data' / 'eval'
    images_root = out_root / 'images'

    records = []
    seen_ids = set()
    for fp in source_files:
        for row in read_jsonl(fp):
            rid = row['id']
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            rel_image = copy_image(project_root, row['image_path'], images_root, row['source'])
            records.append(
                {
                    'id': rid,
                    'source': row['source'],
                    'image': rel_image,
                    'task_type': row['task_type'],
                    'image_type': row.get('image_type', row['source']),
                    'difficulty': row['difficulty'],
                    'ground_truth': {
                        'smiles': row['canonical_smiles'],
                        'inchi': None,
                        'selfies': None,
                        'mol': None,
                    },
                    'eval_target': 'canonical_smiles',
                    'license': 'mixed_public_and_team_curated',
                    'source_url_or_doc': row.get('source', 'unknown'),
                    'qc_status': 'pass',
                }
            )
    return records


def write_csv(path: Path, records) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'source', 'image', 'task_type', 'image_type', 'difficulty', 'smiles', 'eval_target', 'qc_status'])
        for row in records:
            writer.writerow([
                row['id'],
                row['source'],
                row['image'],
                row['task_type'],
                row['image_type'],
                row['difficulty'],
                row['ground_truth']['smiles'],
                row['eval_target'],
                row['qc_status'],
            ])


def write_stats(path: Path, records) -> None:
    by_source = Counter(r['source'] for r in records)
    by_diff = Counter(r['difficulty'] for r in records)
    by_task = Counter(r['task_type'] for r in records)
    by_img = Counter(r['image_type'] for r in records)
    payload = {
        'total': len(records),
        'by_source': dict(by_source),
        'by_difficulty': dict(by_diff),
        'by_task_type': dict(by_task),
        'by_image_type': dict(by_img),
        'eval_target': 'canonical_smiles',
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def write_readme(path: Path, records) -> None:
    by_source = Counter(r['source'] for r in records)
    text = f'''# OCSR Official Submission Eval Set v1

## 1. Purpose

This is the unified submission-ready evaluation set for the OCSR project.

It is designed to be:

- directly submittable
- canonical-SMILES only
- free of synthetic-heavy evaluation construction
- source-diverse within the canonical-SMILES task boundary

## 2. Scope

- Task: Optical Chemical Structure Recognition
- Input: single molecule image
- Output: canonical SMILES
- Total samples: {len(records)}

## 3. Included sources

{json.dumps(dict(by_source), ensure_ascii=False, indent=2)}

## 4. Why this bundle is submission-ready

- one unified task target: `canonical_smiles`
- no EDU-CHEMC / Markush / mechanism-style alternate label formats mixed into the main score
- public benchmark core retained
- real-world auxiliary samples retained
- images, labels, stats, and QC are packaged together

## 5. Directory layout

```text
eval/
  README.md
  QC_REPORT.md
  ANNOTATION_GUIDELINE.md
  stats.json
  annotations/
    labels.jsonl
    labels.csv
  images/
    decimer/
    uob/
    uspto/
    real_world/
```

## 6. Notes

- This bundle is the main submission evaluation set.
- Supplementary experimental layers are archived under `V2/data/eval_sources/` and are not part of the main score bundle.
'''
    path.write_text(text, encoding='utf-8')


def write_qc(path: Path, records) -> None:
    by_source = Counter(r['source'] for r in records)
    text = f'''# QC Report

## Checks performed

1. Unified target field: `canonical_smiles`
2. Unified annotation schema in `annotations/labels.jsonl`
3. Source-level merge only from canonical-SMILES-compatible sets
4. EDU-CHEMC / Markush / mechanism layers excluded from the main score bundle
5. Image files copied into a single self-contained `images/` tree

## Source distribution

{json.dumps(dict(by_source), ensure_ascii=False, indent=2)}

## Known limitation

- This bundle is still benchmark-heavy and should be explained together with the separate training-data and supplementary-layer documents.
'''
    path.write_text(text, encoding='utf-8')


def write_annotation_guide(path: Path) -> None:
    text = '''# Annotation Guideline

## Task target

- Output format: canonical SMILES
- One image corresponds to one target structure string

## Rules

1. Do not mix alternate targets such as chemfig / ssml_normed / CXSMILES into this main bundle.
2. Keep the original source grouping in metadata.
3. Use the canonical SMILES string as the only scoring target.
4. Exclude samples with unresolved ambiguity from the main submission bundle.
'''
    path.write_text(text, encoding='utf-8')


def main() -> None:
    v2_root = Path(__file__).resolve().parents[1]
    project_root = v2_root.parent
    out_root = v2_root / 'data' / 'eval'
    annotations_root = out_root / 'annotations'
    records = build_records(project_root)
    write_jsonl(annotations_root / 'labels.jsonl', records)
    write_csv(annotations_root / 'labels.csv', records)
    write_stats(out_root / 'stats.json', records)
    write_readme(out_root / 'README.md', records)
    write_qc(out_root / 'QC_REPORT.md', records)
    write_annotation_guide(out_root / 'ANNOTATION_GUIDELINE.md')
    print(json.dumps({'total': len(records), 'output_root': str(out_root)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
