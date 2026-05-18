# Submission Ready Checklist (2026-05-18)

This file maps the current `V2-1` workspace to the competition submission requirements.

## 1. What Is Directly Submittable Now

### 1.1 Evaluation Set

There are two candidate submission paths.

#### Recommended conservative path

Submit:

```text
V2-1/data/eval/canonical_smiles_main_v1/
V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/
V2-1/scripts/evaluate_ocsr_predictions_detailed.py
V2-1/configs/prompt.txt
V2-1/runbooks/training_dataset_construction_science_zh.md
V2-1/reports/singleline_rw_dataset_summary.json
V2-1/reports/singleline_rw_dataset_stats.json
V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/README.md
V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/TECHNICAL_REPORT_zh.md
V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/stats.json
```

Reason:

- `canonical_smiles_main_v1` is the cleanest main OCSR benchmark.
- `mixed_eval_v1p1` is the cleanest current “real-world + edu supplement” benchmark we have with documentation.
- Both already have labels, images, and structure.

#### Aggressive collection path

Submit:

```text
V2-1/data/eval/
```

Reason:

- It is already packaged as a large “official submission eval collection”.
- It includes the `canonical_smiles_main_v1` main benchmark plus `edu_chmec_ssml_normed_test_v1`.

Risk:

- It mixes two formal sub-benchmarks with different target spaces:
  - `canonical_smiles`
  - `ssml_normed`
- This is defensible only if the email/report clearly says it is a collection with separately reported scores, not one single unified metric.

Recommendation:

- For the first trial submission, use the conservative path.

### 1.2 Training Data Construction Report

Directly usable sources:

```text
V2-1/runbooks/training_dataset_construction_science_zh.md
V2-1/reports/singleline_rw_dataset_summary.json
V2-1/reports/singleline_rw_dataset_stats.json
V2-1/reports/singleline_rw_dataset_audit.json
V2-1/scripts/build_singleline_rw_sft_dataset.py
V2-1/scripts/audit_singleline_training_dataset.py
V2-1/scripts/summarize_singleline_dataset_stats.py
```

This is enough to form a first Markdown/PDF training-data report.

### 1.3 GitHub Open-Source Project

Directly suitable for GitHub:

```text
V2-1/configs/
V2-1/scripts/
V2-1/runbooks/
V2-1/reports/
V2-1/README.md
V2-1/BASELINE_V2_1.md
V2-1/DATA_COLLECTION_GUIDE_zh.md
V2-1/DATA_EXPANSION_PLAN.md
V2-1/download_public_weak_sources.md
V2-1/USER_DATA_TODO_zh.md
V2-1/SUBMISSION_READY_20260518.md
```

Do not open-source by default:

```text
V2-1/data/assets/train_phase*
V2-1/data/sft_materialized/train_*.jsonl
V2-1/data/public_sources/raw/
V2-1/data/incoming/weak_domain/raw_images/
```

These are training/private/raw assets, not required for public release.

### 1.4 Hugging Face Model

Upload this model folder:

```text
V2-1/outputs/export/
```

The current local flattened export already contains:

- `model-00001-of-00001.safetensors`
- `model.safetensors.index.json`
- `config.json`
- `generation_config.json`
- tokenizer files
- processor/preprocessor files
- remote-code files:
  - `configuration_paddleocr_vl.py`
  - `image_processing_paddleocr_vl.py`
  - `modeling_paddleocr_vl.py`
  - `processing_paddleocr_vl.py`

This is the correct Hugging Face upload target.

## 2. Validation Set Used In Training

Current training-time validation set:

```text
V2-1/data/sft_materialized/val_singleline_v1p1_messages.jsonl
```

This is not the main “competition evaluation set submission”.
It is the internal validation set materialized from:

```text
V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl
```

So:

- Internal validation benchmark actually used in training/eval selection:
  - `ocsr_realworld_mixed_eval_v1p1`
- Main benchmark used for cleaner OCSR reporting:
  - `canonical_smiles_main_v1`

## 3. Model Results Already Available

Available local prediction/evaluation artifacts:

```text
V2-1/eval_runs_export_full/canonical_main/
V2-1/eval_runs_export_full/mixed_v1p1/mixed_v1p1/
```

Current fixed reports:

- `canonical_main/report_fixed.json`
- `mixed_v1p1/mixed_v1p1/report_fixed.json`

These are enough to support a first model card and submission email summary.

## 4. Recommended First-Trial Submission Package

### Evaluation Set email

Send links for:

```text
canonical_smiles_main_v1
ocsr_realworld_mixed_eval_v1p1
evaluation script
dataset description docs
```

### Training Data Report email

Send:

```text
training_dataset_construction_science_zh.md
plus exported PDF if needed
```

### GitHub email

Open-source the code/docs workspace only.

### Hugging Face email

Upload:

```text
V2-1/outputs/export/
```

## 5. What Still Needs Packaging Work

Not blockers for a first trial, but should be improved:

1. Convert training-data report Markdown into a nicer PDF.
2. Write a proper Hugging Face model card.
3. Prepare a smaller GitHub-clean repo if you do not want to publish all local data directories.
4. Add a one-page submission summary in Chinese.

## 6. Practical Recommendation

If the goal is “submit one version now and test the water”:

1. Evaluation set:
   submit `canonical_smiles_main_v1 + ocsr_realworld_mixed_eval_v1p1`.
2. Training data report:
   submit current Markdown report plus code references.
3. GitHub:
   publish code/docs/configs/reports only.
4. Hugging Face:
   upload `V2-1/outputs/export/`.
