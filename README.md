# PaddleOCR-VL-1.5-OCSR

OCR-based molecular structure recognition built on top of `PaddleOCR-VL-1.5`, with a single-line canonical-SMILES training target and a weak-domain evaluation focus.

## What This Repository Contains

This open-source release includes:

- training configs
- dataset build and audit scripts
- inference and evaluation scripts
- runbooks and reproducibility notes
- dataset reports and benchmark notes

This release does **not** include private or large raw training assets by default.

## Current Baseline

The current merged-export baseline was trained with a `single-stage real-weighted LoRA SFT` setup.

Main local evaluation results:

### `canonical_smiles_main_v1`

- canonical exact accuracy: `32.86%`
- token micro F1: `70.35%`
- valid SMILES rate: `71.84%`
- mean fingerprint Tanimoto: `0.6992`

### `ocsr_realworld_mixed_eval_v1p1`

- canonical exact accuracy: `33.77%`
- token micro F1: `70.18%`
- valid SMILES rate: `75.84%`
- mean fingerprint Tanimoto: `0.6849`

Source-level behavior is uneven:

- `uob` is relatively strong
- `uspto` is usable but still behind published OCSR systems
- `real_world`, `decimer`, and `edu_chemc` remain the main weak domains

## Method

The current training line uses:

- image input only
- fixed prompt:
  - `OCR: Output only the canonical SMILES string for the molecule shown in the image.`
- assistant output restricted to canonical SMILES
- deterministic real-world weighting
- exclusion of incompatible label spaces such as `ssml_normed`, `chemfig`, and LaTeX targets from the main line

## Repository Layout

- `configs/`: training/export/prompt configs
- `scripts/`: dataset build, audit, inference, evaluation, and weak-domain tooling
- `runbooks/`: training and dataset construction notes
- `reports/`: dataset and experiment summaries

## Reproducibility

Key scripts:

- build train dataset:
  - `scripts/build_singleline_rw_sft_dataset.py`
- audit dataset:
  - `scripts/audit_singleline_training_dataset.py`
- transformer inference:
  - `scripts/infer_ocsr_transformers.py`
- detailed evaluation:
  - `scripts/evaluate_ocsr_predictions_detailed.py`

## Weak-Domain Work

This release also includes tooling for:

- weak-domain held-out evaluation construction
- weak-domain intake manifests
- weak-domain auto replay generation
- controlled evaluation generation from SMILES

Relevant scripts:

- `scripts/build_weak_domain_eval_v2.py`
- `scripts/audit_weak_domain_eval.py`
- `scripts/import_weak_domain_training_pool.py`
- `scripts/generate_auto_weak_domain_replay.py`
- `scripts/generate_controlled_eval_from_smiles.py`

## Open/Closed Boundary

Open in this repository:

- code
- configs
- runbooks
- reports
- documentation

Not included by default:

- private training images
- full materialized training assets
- private weak-domain raw collections
- heavyweight public raw source archives

## Model

The corresponding merged fine-tuned model should be published separately on Hugging Face.
