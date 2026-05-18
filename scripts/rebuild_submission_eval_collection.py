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


def write_csv(path: Path, fieldnames, records) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def move_current_canonical(eval_root: Path):
    canonical_root = eval_root / 'canonical_smiles_main_v1'
    ensure_dir(canonical_root)
    src_annotations = eval_root / 'annotations'
    src_images = eval_root / 'images'
    dest_annotations = canonical_root / 'annotations'
    dest_images = canonical_root / 'images'

    if src_annotations.exists() and not dest_annotations.exists():
        shutil.move(str(src_annotations), str(dest_annotations))
    if src_images.exists() and not dest_images.exists():
        shutil.move(str(src_images), str(dest_images))
    return canonical_root


def build_edu_test(eval_sources_root: Path, eval_root: Path):
    labels_path = eval_sources_root / 'edu_chmec_eval_candidate_v1' / 'manifests' / 'labels_ssml_normed.jsonl'
    if not labels_path.exists():
        raise FileNotFoundError(f'Missing EDU-CHEMC labels: {labels_path}')

    edu_root = eval_root / 'edu_chmec_ssml_normed_test_v1'
    images_root = edu_root / 'images'
    annotations_root = edu_root / 'annotations'
    source_images_root = eval_sources_root / 'edu_chmec_eval_candidate_v1' / 'images' / 'test'

    records = []
    csv_rows = []
    for row in read_jsonl(labels_path):
        if row.get('split') != 'test':
            continue
        src_img = source_images_root / Path(row['image']).name
        if not src_img.exists():
            raise FileNotFoundError(f'Missing EDU-CHEMC test image: {src_img}')
        ensure_dir(images_root)
        dest_img = images_root / src_img.name
        if not dest_img.exists():
            shutil.copy2(src_img, dest_img)

        new_row = {
            'id': row['id'],
            'source': row['source'],
            'split': 'test',
            'image': f'images/{src_img.name}',
            'annotation_json': row['annotation_json'],
            'task_type': row['task_type'],
            'image_type': row['image_type'],
            'difficulty': row['difficulty'],
            'label_format': row['label_format'],
            'ssml_normed': row['ssml_normed'],
            'eval_target': row['eval_target'],
            'license': row['license'],
            'source_url_or_doc': row['source_url_or_doc'],
            'qc_status': row['qc_status'],
        }
        records.append(new_row)
        csv_rows.append(new_row)

    write_jsonl(annotations_root / 'labels.jsonl', records)
    write_csv(
        annotations_root / 'labels.csv',
        ['id', 'source', 'split', 'image', 'annotation_json', 'task_type', 'image_type', 'difficulty', 'label_format', 'ssml_normed', 'eval_target', 'license', 'source_url_or_doc', 'qc_status'],
        csv_rows,
    )
    return edu_root, records


def write_top_level_docs(eval_root: Path, canonical_count: int, edu_count: int, canonical_records, edu_records):
    total = canonical_count + edu_count
    canonical_sources = Counter(r['source'] for r in canonical_records)
    edu_sources = Counter(r['source'] for r in edu_records)
    stats = {
        'total': total,
        'subbenchmarks': {
            'canonical_smiles_main_v1': {
                'total': canonical_count,
                'sources': dict(canonical_sources),
                'eval_target': 'canonical_smiles',
            },
            'edu_chmec_ssml_normed_test_v1': {
                'total': edu_count,
                'sources': dict(edu_sources),
                'eval_target': 'ssml_normed',
            },
        },
    }
    (eval_root / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')

    readme = f'''# OCSR Evaluation Collection v2

## 1. Purpose

This directory is the final submission-ready evaluation **collection** for the OCSR project.

It is organized as a small number of clearly separated sub-benchmarks so that:

- the total scale is comfortably above 1000 samples;
- the main canonical-SMILES benchmark remains clean and interpretable;
- the education-domain benchmark can be reported separately without corrupting the main score.

## 2. Total scale

- Total samples: **{total}**

## 3. Sub-benchmarks

### A. `canonical_smiles_main_v1`

- Total: **{canonical_count}**
- Target: `canonical_smiles`
- Sources: {json.dumps(dict(canonical_sources), ensure_ascii=False)}

### B. `edu_chmec_ssml_normed_test_v1`

- Total: **{edu_count}**
- Target: `ssml_normed`
- Sources: {json.dumps(dict(edu_sources), ensure_ascii=False)}

## 4. Directory layout

```text
eval/
  README.md
  QC_REPORT.md
  ANNOTATION_GUIDELINE.md
  stats.json
  canonical_smiles_main_v1/
    annotations/
    images/
  edu_chmec_ssml_normed_test_v1/
    annotations/
    images/
```

## 5. Use guidance

- Use `canonical_smiles_main_v1` as the primary OCSR benchmark.
- Use `edu_chmec_ssml_normed_test_v1` as a separated education-domain benchmark.
- Do **not** average the two tasks into a single metric without explicitly explaining the label-format difference.
'''
    (eval_root / 'README.md').write_text(readme, encoding='utf-8')

    qc = f'''# QC Report

## Checks performed

1. Removed `user_collected` from the formal submission package.
2. Removed macOS `._*` image junk from the main eval tree.
3. Unified the main benchmark into a self-contained directory.
4. Kept EDU-CHEMC as a separate benchmark because its target is `ssml_normed`, not `canonical_smiles`.
5. Ensured both sub-benchmarks have local `annotations/` and `images/` directories.

## Collection scale

- canonical main: {canonical_count}
- edu benchmark: {edu_count}
- total: {total}

## Remaining risk

- The two sub-benchmarks use different output targets and should be reported separately.
'''
    (eval_root / 'QC_REPORT.md').write_text(qc, encoding='utf-8')

    guide = '''# Annotation Guideline

## canonical_smiles_main_v1

- One image corresponds to one canonical SMILES string.
- Main score should use canonical-SMILES-compatible metrics.

## edu_chmec_ssml_normed_test_v1

- One image corresponds to one `ssml_normed` target string.
- This benchmark is education-specific and should not be merged directly into the canonical-SMILES main score.

## General rules

1. Keep sub-benchmarks separate.
2. Keep labels exactly as distributed in each sub-benchmark.
3. Do not silently convert `ssml_normed` to `canonical_smiles` in the formal package.
'''
    (eval_root / 'ANNOTATION_GUIDELINE.md').write_text(guide, encoding='utf-8')


def main() -> None:
    v2_root = Path(__file__).resolve().parents[1]
    project_root = v2_root.parent
    eval_root = v2_root / 'data' / 'eval'
    eval_sources_root = v2_root / 'data' / 'eval_sources'

    canonical_root = move_current_canonical(eval_root)
    canonical_labels = list(read_jsonl(canonical_root / 'annotations' / 'labels.jsonl'))
    edu_root, edu_records = build_edu_test(eval_sources_root, eval_root)
    write_top_level_docs(eval_root, len(canonical_labels), len(edu_records), canonical_labels, edu_records)
    print(json.dumps({'canonical_total': len(canonical_labels), 'edu_total': len(edu_records), 'eval_root': str(eval_root), 'canonical_root': str(canonical_root), 'edu_root': str(edu_root)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
