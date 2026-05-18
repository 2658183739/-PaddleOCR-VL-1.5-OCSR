# V2-1 Weak-Domain Data Expansion Plan

## Why We Need This

The V2-1 baseline is valid but uneven:

- `uob`: strong, exact about 71.5%.
- `uspto`: usable, exact about 38.5%.
- `real_world`: weak, exact about 13.4%.
- `decimer`: very weak, exact about 2.0%.
- `edu_chemc`: weak on mixed eval, exact about 7.2%.

Therefore, the next data work should not add more easy UOB-style clean renders. It should target hand-drawn structures, real scanned/photo/document images, and education/exam-style molecule depictions.

## Key Principle

Evaluation data and training data must be expanded separately.

- Evaluation set: small, clean, representative, never trained on. Its job is to measure whether we really improved.
- Training set: larger, diverse, augmented, allowed to include synthetic/teacher-labeled/human-verified data. Its job is to change the model.

Never allow exact image, filename, ID, canonical SMILES, or near-duplicate molecule leakage from evaluation into training.

## Evaluation Set Expansion

### Target

Initial internal evaluation set created:

```text
V2-1/data/eval/weak_domain_v2/
```

Current seed size: 577 samples assembled from existing held-out evaluation sources.

Current seed split:

| Domain | Count | Purpose |
| --- | ---: | --- |
| `decimer_handdrawn` | 150 | Measure hand-drawn robustness |
| `real_world_photo_scan` | 212 | Measure photos, scans, crops, low contrast |
| `edu_exam` | 153 | Measure Chinese exam/teaching/question-style images |
| `long_or_stereo` | 62 | Stress long molecules and stereochemistry |

Current build and audit commands:

```bash
python V2-1/scripts/build_weak_domain_eval_v2.py --project-root .
python V2-1/scripts/audit_weak_domain_eval.py --project-root .
```

Current audit report:

```text
V2-1/reports/weak_domain_v2_audit.json
```

Local audit result: no missing images, no unreadable images, no duplicate IDs, no duplicate SMILES under local string-level checking, and no overlap with `train_singleline_rw_messages.jsonl` by ID, image filename, or SMILES. Re-run the audit in the cloud eval environment to get RDKit-backed validation.

Target next size after public/private collection: 800-1200 samples total.

Suggested split:

| Domain | Count | Purpose |
| --- | ---: | --- |
| `decimer_handdrawn` | 250-350 | Measure hand-drawn robustness |
| `real_world_photo_scan` | 250-350 | Measure photos, scans, crops, low contrast |
| `edu_exam` | 150-250 | Measure Chinese exam/teaching/question-style images |
| `document_page_context` | 100-150 | Measure figures embedded in papers/patents/pages |
| `long_or_stereo` | 50-100 | Stress long molecules and stereochemistry |

### What To Include

For `decimer_handdrawn`:

- Hold out DECIMER hand-drawn images that are not used in training.
- Add self-collected handwritten molecules from multiple people if possible.
- Include pen, pencil, marker, uneven line width, imperfect rings, erased strokes.

For `real_world_photo_scan`:

- Phone photos of printed or screen-displayed structures.
- Scans at different DPI and compression levels.
- Crops from patent/article-like documents where license permits.
- Include blur, skew, shadows, low contrast, yellowed paper, and screenshots.

For `edu_exam`:

- Chinese-style chemistry question images containing molecule structures.
- Structures with nearby Chinese text, labels, arrows, option letters, or reaction context.
- Prefer self-made or permission-cleared mock exam pages to avoid copyright problems.

For `document_page_context`:

- Cropped figures from documents with captions, figure numbers, nearby formulas, or tables.
- If one image contains multiple molecules, the task description must define the target clearly. If no target marker exists, crop to one molecule.

### Annotation Format

Use the same schema as current eval files:

```json
{
  "id": "weak_decimer_000001",
  "source": "decimer_handdrawn",
  "image": "images/decimer_handdrawn/weak_decimer_000001.png",
  "task_type": "molecule_structure_recognition",
  "difficulty": "handwritten",
  "ground_truth": {
    "smiles": "canonical SMILES here",
    "inchi": null,
    "selfies": null,
    "mol": null
  },
  "eval_target": "canonical_smiles",
  "license": "CC-BY-4.0 or private_internal or self_collected",
  "source_url_or_doc": "source or collection note",
  "qc_status": "pass"
}
```

### Evaluation QC

Every evaluation label must pass:

- RDKit parses the raw SMILES.
- Canonical SMILES is non-empty.
- Image opens successfully.
- One expected molecule per sample, unless the task description explicitly defines target selection.
- No overlap with training by canonical SMILES.
- A sampled subset is visually checked by a human.

## Training Set Expansion

### Target

Create a new training materialization:

```text
V2-1/data/sft_materialized/train_singleline_rw_v2_messages.jsonl
```

Recommended additional unique images for the next training run:

| Domain | Unique images | Training weight |
| --- | ---: | ---: |
| DECIMER/public hand-drawn | 3000-4500 | repeat 3-4 |
| Private hand-drawn | 500-1500 | repeat 4-5 |
| Real-world photo/scan | 2000-4000 | repeat 4-5 |
| Edu/exam-style | 1000-2500 | repeat 4-5 |
| Patent/article document crops | 2000-5000 | repeat 2-3 |
| Synthetic hard augmentations | 5000-15000 | repeat 1-2 |

Keep UOB and clean USPTO capped. They are already strong enough and should mainly preserve basic competence.

### Import Path Now Available

Use this script to import newly collected public/private weak-domain training manifests:

```bash
python V2-1/scripts/import_weak_domain_training_pool.py \
  --project-root . \
  --manifest V2-1/data/manifests/weak_domain_training_candidates.jsonl \
  --output V2-1/data/sft_materialized/train_weak_domain_pool_messages.jsonl \
  --assets-root V2-1/data/assets/weak_domain_pool
```

Candidate manifest rows should contain at least:

```json
{
  "id": "private_handdrawn_000001",
  "image": "relative/or/absolute/path/to/image.png",
  "smiles": "CCO",
  "source": "private_handdrawn",
  "difficulty": "handwritten",
  "weak_domain": "decimer_handdrawn",
  "license": "private_internal",
  "source_url_or_doc": "collection_batch_202605"
}
```

The importer canonicalizes SMILES when RDKit is available, copies images into `data/assets/weak_domain_pool/`, converts rows to PaddleOCR-VL SFT message format, and filters molecules already present in evaluation labels.

After importing the weak-domain pool, build the next trainable V2-2 dataset:

```bash
python V2-1/scripts/build_singleline_rw_v2_dataset.py --project-root .
```

This creates:

```text
V2-1/data/sft_materialized/train_singleline_rw_v2_messages.jsonl
V2-1/reports/singleline_rw_v2_dataset_summary.json
```

The builder keeps the current V2-1 base training set, adds weak-domain replay examples, and applies evaluation-SMILES filtering again.

### Image Normalization Now Available

Raw private/public images should first go through:

```bash
python V2-1/scripts/prepare_weak_domain_manifest.py \
  --input V2-1/data/incoming/weak_domain/weak_domain_training_candidates.csv \
  --output V2-1/data/manifests/weak_domain_training_candidates.jsonl \
  --image-output-root V2-1/data/incoming/weak_domain/normalized_images
```

This script accepts `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, and `.tiff`; converts images to RGB PNG; applies EXIF transpose; resizes extremely small or large images into a trainable range; adds a small white border; and checks each row has a usable SMILES string.

Collection entrypoint:

```text
V2-1/data/incoming/weak_domain/
```

### What To Add

For `decimer`:

- Add DECIMER hand-drawn images to training, excluding held-out eval molecules.
- Add self-collected hand drawings assigned from known SMILES.
- Render the same molecule in clean form only if needed for paired clean-to-handwritten consistency.

For `real_world`:

- Print public-domain or self-rendered molecules, then photograph them with different phones.
- Create scanned PDF pages, then crop molecule regions.
- Use patent/document-derived public or license-compatible molecule crops.
- Add controlled degradation: blur, JPEG artifacts, perspective warp, shadows, low contrast, paper texture, small crop margins.

For `edu_chemc`:

- Do not train on `chemfig`, `ssml_normed`, or LaTeX as target. Target must remain one-line canonical SMILES.
- Create self-made Chinese exam-style pages using known SMILES rendered into molecule images, then add nearby Chinese text/options/reaction symbols.
- Collect private teacher/student mock questions if permission is available.
- If using public exam screenshots, record provenance and keep them private unless license allows redistribution.

For `document_page_context`:

- Use molecule crops, not full pages, unless target selection is unambiguous.
- If full-page or multi-grid examples are needed, add visual target markers or define rules in the prompt and task description.

### Public Sources To Prioritize

| Source | Use | Notes |
| --- | --- | --- |
| DECIMER hand-drawn molecule images | Training + held-out eval | Open benchmark of 5088 hand-drawn depictions, CC-BY 4.0 |
| MolGrapher / USPTO-30K | Training + eval reference | Real molecule images from patent-style documents; useful for `uspto`, `real_world`, and abbreviation style |
| MolGrapher-Synthetic-300K | Training style inspiration or subset | Synthetic images with molecule/render/image augmentations; large, so sample selectively |
| PatCID | Real-world patent/document crops | Huge patent-derived corpus; sample carefully and verify license/provenance |
| PubChem | SMILES source for rendering | Good for molecule diversity; render yourself with RDKit/CDK/Indigo |
| ChEMBL | Drug-like SMILES source | Good for medicinal chemistry structures and realistic functional groups |

### Private Data Collection

Private data is valuable because it can match the competition's hidden distribution better than public benchmarks.

Recommended collection plan:

1. Hand-drawn collection:
   - Select 500-1000 diverse canonical SMILES from PubChem/ChEMBL.
   - Ask 10-30 people to draw 20-50 assigned molecules each.
   - Capture both scan and phone photo versions.
   - Keep the assigned SMILES as ground truth; do not OCR the label from the drawing.

2. Real-world photo collection:
   - Render molecules to PDF pages with varied fonts, bond widths, captions, Chinese/English surrounding text.
   - Print or display on screen.
   - Photograph under varied lighting, angle, distance, and background.
   - Crop automatically and manually inspect.

3. Edu/exam collection:
   - Generate mock Chinese exam questions ourselves.
   - Include option letters, Chinese descriptors, arrows, reaction conditions, and nearby distractor text.
   - Keep the molecule crop as the supervised image when the prompt expects one molecule.

4. Document crop collection:
   - Use patent/public article pages where redistribution is allowed, or keep restricted materials private.
   - Crop molecule regions and store page/document provenance.
   - Manually verify any teacher-generated label.

### Labeling Workflow

Preferred ground-truth hierarchy:

1. Assigned known SMILES from generation or hand-drawing task.
2. MOL/SDF source converted to canonical SMILES.
3. Existing public dataset label canonicalized by RDKit.
4. Teacher model output only after human/RDKit validation.

Do not use unverified OCR/OCSR output as ground truth.

### Training Recipe

Recommended next experiment:

```text
V2-2: weak-domain replay + hard augmentations
```

Change only the dataset first. Keep model, LoRA config, prompt, dtype, and inference settings the same.

Suggested weighting:

- `decimer`: repeat 4
- `private_handdrawn`: repeat 5
- `real_world`: repeat 5
- `edu_exam`: repeat 5
- `patent_document`: repeat 3
- `molgrapher_synthetic_hard`: repeat 2
- `uob`: repeat 1, cap if needed
- `uspto`: repeat 1
- `clean_synthetic`: cap 1000-1500

Checkpoint selection should use a weighted score:

```text
score = 0.30 * exact_overall
      + 0.20 * tanimoto_overall
      + 0.20 * exact_decimer
      + 0.15 * exact_real_world
      + 0.15 * exact_edu_exam
```

This avoids selecting a checkpoint that improves easy UOB while still failing the real weak domains.

## Anti-Leakage Rules

Before training:

- Build a canonical SMILES set from every eval JSONL.
- Remove any training sample whose canonical SMILES appears in eval.
- Also remove exact image filename/path overlaps.
- Optionally remove near-duplicate images using perceptual hash.

After training:

- Report source-level metrics, not only aggregate metrics.
- Save example predictions for exact hits, close Tanimoto misses, invalid SMILES, and hallucinated/repeated outputs.

## One-Week Practical Plan

Day 1:

- Freeze `V2-1` baseline and download cloud model/eval artifacts.
- Create `weak_domain_v2` schema and split plan.

Day 2-3:

- Import DECIMER and sample public patent/document data.
- Generate 1000-2000 edu/exam-style synthetic pages.

Day 4:

- Collect or generate first private hand-drawn/photo batch.
- Run RDKit and image QC.

Day 5:

- Build `train_singleline_rw_v2_messages.jsonl`.
- Run 20-sample smoke training/inference sanity check.

Day 6-7:

- Train V2-2.
- Evaluate on `canonical_main`, `mixed_v1p1`, and `weak_domain_v2`.
- Compare against V2-1 with source-level metrics.
