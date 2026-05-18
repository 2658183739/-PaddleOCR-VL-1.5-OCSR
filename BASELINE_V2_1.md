# V2-1 Baseline Handoff

This folder is a frozen baseline snapshot copied from `V2` on 2026-05-13.

## Current Status

- Local baseline folder: `V2-1/`
- Training line: Single-stage Real-Weighted LoRA SFT
- Base model: `PaddleOCR-VL-1.5`
- Fine-tuned model to keep: merged LoRA export from cloud
- Important fix in this baseline: `scripts/evaluate_ocsr_predictions_detailed.py` now reads `ground_truth.smiles`, `canonical_smiles`, `smiles`, or `label_summary`, so it will not report fake zero scores on the current JSONL label format.

## Validated Metrics

The following metrics were computed on the cloud from the merged export model:

| Eval set | Total | Valid SMILES | Canonical exact acc | Mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| `canonical_smiles_main_v1` | 767 | 71.84% | 32.86% | 0.6992 |
| `ocsr_realworld_mixed_eval_v1p1` | 770 | 75.84% | 33.77% | 0.6849 |

Source-level breakdown for `canonical_smiles_main_v1`:

| Source | Total | Valid SMILES | Canonical exact acc | Mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| `decimer` | 150 | 65.33% | 2.00% | 0.2536 |
| `real_world` | 217 | 51.61% | 13.36% | 0.5662 |
| `uob` | 200 | 92.50% | 71.50% | 0.9003 |
| `uspto` | 200 | 78.00% | 38.50% | 0.8361 |

Source-level breakdown for `ocsr_realworld_mixed_eval_v1p1`:

| Source | Total | Valid SMILES | Canonical exact acc | Mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| `edu_chemc` | 153 | 85.62% | 7.19% | 0.3022 |
| `real_world` | 217 | 51.61% | 13.36% | 0.5662 |
| `uob` | 200 | 92.50% | 71.50% | 0.9003 |
| `uspto` | 200 | 78.00% | 38.50% | 0.8361 |

Interpretation: the baseline is real and usable. UOB and USPTO are strong; DECIMER hand-drawn structures, EDU-CHEMC, and real-world photos/scans are the main bottlenecks.

## Cloud Artifacts To Download Into V2-1

Download these from the cloud project root `/data/coding/data` into the matching local `V2-1` paths.

### Must Download

| Cloud path | Local target under `V2-1` | Why |
| --- | --- | --- |
| `/data/coding/data/V2/outputs/singleline_rw_lora/export/` | `outputs/export/` or `outputs/singleline_rw_lora/export/` | Merged full model for inference/submission/Hugging Face upload |
| `/data/coding/data/V2/outputs/singleline_rw_lora/train_singleline_rw.log` | `outputs/train_singleline_rw.log` or `outputs/singleline_rw_lora/train_singleline_rw.log` | Training evidence for report and reproducibility |
| `/data/coding/data/V2/outputs/singleline_rw_lora/trainer_state.json` | `outputs/trainer_state.json` or `outputs/singleline_rw_lora/trainer_state.json` | Checkpoint/loss history |
| `/data/coding/data/V2/outputs/singleline_rw_lora/train_results.json` | `outputs/train_results.json` or `outputs/singleline_rw_lora/train_results.json` | Final training metrics |
| `/data/coding/data/V2/outputs/singleline_rw_lora/all_results.json` | `outputs/all_results.json` or `outputs/singleline_rw_lora/all_results.json` | Aggregated training metrics |
| `/data/coding/data/V2/eval_runs_export_full/canonical_main/` | `eval_runs_export_full/canonical_main/` | Main eval predictions and detailed report |
| `/data/coding/data/V2/eval_runs_export_full/mixed_v1p1/` | `eval_runs_export_full/mixed_v1p1/` | Mixed eval predictions and detailed report |

The current local snapshot uses the flatter form:

```text
V2-1/outputs/export/
V2-1/outputs/train_singleline_rw.log
V2-1/outputs/train_results.json
V2-1/outputs/trainer_state.json
```

That layout is fine as long as your run commands point to the actual path.

### Strongly Recommended

| Cloud path | Local target under `V2-1` | Why |
| --- | --- | --- |
| `/data/coding/data/V2/outputs/singleline_rw_lora/checkpoint-1600/` | `outputs/singleline_rw_lora/checkpoint-1600/` | Final LoRA checkpoint before merge |
| `/data/coding/data/V2/outputs/singleline_rw_lora/peft_model-00001-of-00001.safetensors` | `outputs/singleline_rw_lora/peft_model-00001-of-00001.safetensors` | Root LoRA weights |
| `/data/coding/data/V2/outputs/singleline_rw_lora/peft_model.safetensors.index.json` | `outputs/singleline_rw_lora/peft_model.safetensors.index.json` | Root LoRA index |
| `/data/coding/data/V2/outputs/singleline_rw_lora/lora_config.json` | `outputs/singleline_rw_lora/lora_config.json` | LoRA config |
| `/data/coding/data/V2/eval_runs_export_smoke/` | `eval_runs_export_smoke/` | Smoke-test predictions proving the export path works |

### Already Present Locally, But Verify If Re-syncing

| Local path | Expected content |
| --- | --- |
| `data/assets/train_phase3/` | Training images/assets |
| `data/sft_materialized/train_phase3_messages.jsonl` | Phase-3 SFT materialized messages |
| `data/sft_materialized/train_singleline_rw_messages.jsonl` | Active training JSONL |
| `data/sft_materialized/val_singleline_v1p1_messages.jsonl` | Validation JSONL |
| `data/eval/canonical_smiles_main_v1/` | Main evaluation set |
| `data/eval/ocsr_realworld_mixed_eval_v1p1/` | Mixed evaluation set |
| `configs/` | Training/export/prompt configs |
| `scripts/` | Dataset, training, inference, and evaluation utilities |
| `runbooks/` | Training and data-construction notes |
| `reports/` | Dataset audit summaries |

## Suggested Download Command

From a local terminal with SSH access to the cloud container, use `rsync` if available:

```bash
rsync -avP root@<cloud-host>:/data/coding/data/V2/outputs/singleline_rw_lora/export/ "V2-1/outputs/singleline_rw_lora/export/"
rsync -avP root@<cloud-host>:/data/coding/data/V2/outputs/singleline_rw_lora/train_singleline_rw.log "V2-1/outputs/singleline_rw_lora/"
rsync -avP root@<cloud-host>:/data/coding/data/V2/outputs/singleline_rw_lora/trainer_state.json "V2-1/outputs/singleline_rw_lora/"
rsync -avP root@<cloud-host>:/data/coding/data/V2/outputs/singleline_rw_lora/train_results.json "V2-1/outputs/singleline_rw_lora/"
rsync -avP root@<cloud-host>:/data/coding/data/V2/outputs/singleline_rw_lora/all_results.json "V2-1/outputs/singleline_rw_lora/"
rsync -avP root@<cloud-host>:/data/coding/data/V2/eval_runs_export_full/ "V2-1/eval_runs_export_full/"
```

If downloading through AI Studio or a web file browser, preserve the same folder layout.

## Submission Mapping

### Evaluation Set

Use:

- `data/eval/canonical_smiles_main_v1/images/`
- `data/eval/canonical_smiles_main_v1/annotations/labels.jsonl`
- `data/eval/ocsr_realworld_mixed_eval_v1p1/images/`
- `data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl`
- `scripts/evaluate_ocsr_predictions_detailed.py`
- `configs/prompt.txt`
- dataset descriptions from `reports/` and `runbooks/`

Package requirement coverage: images/documents, annotations, task description, evaluation script, dataset description including source, scale, category distribution, and difficulty analysis.

### Training Data Construction Report

Start from:

- `runbooks/training_dataset_construction_science_zh.md`
- `reports/singleline_rw_dataset_summary.json`
- `reports/singleline_rw_dataset_stats.json`
- `reports/singleline_rw_dataset_audit_rdkit.json`
- relevant builders in `scripts/`

The report should explicitly cover collection/synthesis, key code paths, annotation schema, annotation tools or generation tools, de-duplication, leakage checks, RDKit validation, and quality control.

### Open Source Project

Public GitHub should include:

- `configs/`
- `scripts/`
- `runbooks/`
- `reports/`
- `README.md`
- small eval/demo data if license allows
- instructions for training, export, inference, and evaluation

Do not publish private or restricted training data unless license is confirmed.

### Hugging Face Model

Upload the merged export model:

- `outputs/singleline_rw_lora/export/model-00001-of-00001.safetensors`
- `outputs/singleline_rw_lora/export/model.safetensors.index.json`
- `outputs/singleline_rw_lora/export/config.json`
- `outputs/singleline_rw_lora/export/generation_config.json`
- tokenizer, processor, preprocessor, and remote-code files in the same export folder

Model card should include intended use, base model, LoRA/SFT method, training data summary, eval results, limitations, and license notes.

## Reproduction Commands

Train:

```bash
cd /data/coding/data
conda activate torch
bash V2/run_4090_lora_singleline_rw.sh
```

Export merged model:

```bash
cd /data/coding/data
conda activate torch
paddleformers-cli export V2/configs/paddleocr_vl_lora_export_base.yaml \
  model_name_or_path=/data/coding/data/model/PaddleOCR-VL-1.5 \
  output_dir=/data/coding/data/V2/outputs/singleline_rw_lora
```

Evaluate merged export:

```bash
cd /data/coding/data
conda activate eval_torch
python V2/scripts/infer_ocsr_transformers.py \
  --model-dir /data/coding/data/V2/outputs/singleline_rw_lora/export \
  --benchmark-jsonl /data/coding/data/V2/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl \
  --project-root /data/coding/data \
  --output-jsonl /data/coding/data/V2/eval_runs_export_full/canonical_main/pred.jsonl \
  --prompt-file /data/coding/data/V2/configs/prompt.txt \
  --device cuda \
  --torch-dtype bfloat16 \
  --max-new-tokens 256 \
  --min-pixels 50176 \
  --max-pixels 200704
```

Then run:

```bash
python V2/scripts/evaluate_ocsr_predictions_detailed.py \
  --benchmark-jsonl /data/coding/data/V2/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl \
  --prediction-jsonl /data/coding/data/V2/eval_runs_export_full/canonical_main/pred.jsonl \
  --report-json /data/coding/data/V2/eval_runs_export_full/canonical_main/report.json \
  --details-jsonl /data/coding/data/V2/eval_runs_export_full/canonical_main/details.jsonl
```

## Optimization Strategy

Detailed weak-domain data expansion plan and scripts are in:

```text
V2-1/DATA_EXPANSION_PLAN.md
V2-1/scripts/build_weak_domain_eval_v2.py
V2-1/scripts/audit_weak_domain_eval.py
V2-1/scripts/import_weak_domain_training_pool.py
```

An automatic weak-domain replay path is also ready:

```text
V2-1/scripts/generate_auto_weak_domain_replay.py
V2-1/scripts/build_singleline_rw_v2_dataset.py
V2-1/data/sft_materialized/train_weak_domain_auto_messages.jsonl
V2-1/data/sft_materialized/train_singleline_rw_v2_messages.jsonl
```

Public-source and controlled-eval tooling added in this round:

```text
V2-1/download_public_weak_sources.md
V2-1/DATA_COLLECTION_GUIDE_zh.md
V2-1/scripts/generate_controlled_eval_from_smiles.py
V2-1/data/eval_generated/README.md
V2-1/data/eval_generated/source_smiles/generated_eval_seed.csv
```

This round has already optimized five practical directions:

1. Evaluation correctness:
   fixed the OCSR evaluator so current JSONL labels using `ground_truth.smiles` no longer produce fake zero metrics.
2. Weak-domain measurement:
   added `weak_domain_v2` as a dedicated held-out set for DECIMER-style handwritten, real-world photo/scan, edu-exam, and long/stereo stress cases.
3. Data ingestion:
   added a normalized intake path for weak-domain images plus SMILES manifests, with filtering against evaluation molecules.
4. Automatic domain diversification:
   generated auto weak-domain replay samples from existing trainable images to reduce manual collection pressure.
5. V2-2 training readiness:
   added a native V2-1 training config and launcher that build the V2-2 dataset, audit against three eval sets, and write outputs under `V2-1/outputs_v2/`.

### 1. Fix The Weak Domains First

Current errors are concentrated in DECIMER hand-drawn structures, EDU-CHEMC, and real-world photos/scans. Build the next dataset around those failures instead of adding more easy UOB-style rendered samples.

Recommended actions:

- Mine false negatives from `details.jsonl` by source and difficulty.
- Add hard-negative training examples for hand-drawn, scan, photo, multi-grid, page-level, and long-molecule cases.
- Keep UOB/USPTO in training, but cap them so they do not dominate the gradient.
- Add controlled augmentations: blur, JPEG compression, low contrast, perspective warp, uneven illumination, crop margin variation, and handwritten-style line noise.

### 2. Improve Output Discipline

The model still produces unclosed rings, repeated fragments, and overlong continuations. This is mostly a decoding and target-format problem.

Recommended actions:

- Add stop criteria around newline/eos after first SMILES.
- Add repetition penalty and no-repeat n-gram constraints only if they do not hurt exact match.
- Train with stricter assistant targets: one-line canonical SMILES only, no markdown, no chemistry prose.
- Add invalid-output examples where the correct target is still a compact SMILES to teach recovery from noisy images.

### 3. Add Chemistry-Aware Postprocessing

Exact match is strict, but Tanimoto already shows many predictions are structurally close. A lightweight postprocessor can convert close-but-invalid strings into valid candidates.

Recommended actions:

- Generate N candidates with mild sampling or beam search.
- Canonicalize each candidate with RDKit.
- Rank by validity, string likelihood, ring/parenthesis balance, atom valence sanity, and optional image-text confidence.
- Use simple repair heuristics for unmatched parentheses/ring closures only when they improve RDKit validity.

### 4. Use Specialist Distillation

PaddleOCR-VL is a strong document/VLM backbone, but OCSR has specialized SOTA systems. Use them as teachers rather than replacing the pipeline.

Recommended actions:

- Run specialist OCSR teachers such as MolScribe/DECIMER-style recognizers on the unlabeled or weakly labeled pool.
- Keep only high-confidence agreements after RDKit validation.
- Distill teacher outputs into the single-line SMILES format.
- Use disagreement cases as manual review candidates.

### 5. Upgrade Evaluation To A Model-Selection Dashboard

The old evaluator silently read the wrong field and reported fake zero metrics. The next iteration should make this impossible.

Recommended actions:

- Add schema validation before scoring: fail if GT is empty for more than a tiny fraction.
- Report exact, valid rate, token F1, edit similarity, Tanimoto, and per-source breakdown.
- Save top failure examples per source with image path, GT, prediction, canonical forms, and Tanimoto.
- Select checkpoints by weighted score emphasizing real-world/DECIMER, not overall exact only.

### 6. Prepare A Clean Public Story

For the competition, the project story should be "domain-robust OCSR with PaddleOCR-VL, chemistry validation, and real-world difficulty analysis."

Recommended actions:

- Keep training data private if needed, but publish all code and eval data allowed by license.
- Publish the merged model on Hugging Face with a strong model card.
- Include transparent limitations: weak hand-drawn and hard real-world structures in V2-1.
- Show progress using source-level metrics, not just a single aggregate number.
