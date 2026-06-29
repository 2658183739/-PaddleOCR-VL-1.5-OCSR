# Candidate Rerank Route - 2026-06-19

## Purpose

This route improves V2-1 inference without another model run. It reads saved beam/TTA candidates from `pred.jsonl`, selects a better candidate, and writes a normal prediction JSONL that can be evaluated by the existing detailed evaluator.

It is intended for panels where `--save-candidates` was used during inference.

## Script

Local/remote script:

```bash
V2-1/scripts/rerank_ocsr_candidates.py
```

Main mode:

```bash
--mode chem_light
```

The current rule is deliberately small and inspectable:

- aggregate duplicate candidates by canonical SMILES;
- use vote count as the main signal;
- add a small bonus for salt/multi-fragment candidates when votes are close;
- mildly penalize larger molecules as a tie breaker;
- preserve explicit E/Z backslash stereochemistry when a same-skeleton candidate has enough support.

This is not a replacement for model training. It is a cheap candidate selector that exposes how much improvement is available before another LoRA/DPO run.

## Remote Outputs

Original V2-1 UOB80 reranked outputs:

```bash
/root/autodl-fs/outputs_v2/v2_1_original_compare/eval/uob_medium_80/rerank_chem_light_pred.jsonl
/root/autodl-fs/outputs_v2/v2_1_original_compare/eval/uob_medium_80/rerank_chem_light_report.json
/root/autodl-fs/outputs_v2/v2_1_original_compare/eval/uob_medium_80/rerank_chem_light_eval_report.json
/root/autodl-fs/outputs_v2/v2_1_original_compare/eval/uob_medium_80/preference_pairs.jsonl
```

Local copies:

```bash
V2-1/reports/remote_eval_20260619/v2_original_uob80/rerank_chem_light_pred.jsonl
V2-1/reports/remote_eval_20260619/v2_original_uob80/rerank_chem_light_report.json
V2-1/reports/remote_eval_20260619/v2_original_uob80/rerank_chem_light_eval_report.json
V2-1/reports/remote_eval_20260619/v2_original_uob80/preference_pairs.jsonl
```

## Results

Panel: `uob_medium_80`

| source | canonical exact | raw exact | valid | mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| original V2-1 selected | 0.7500 | 0.6750 | 1.0000 | 0.8935 |
| original V2-1 + chem_light rerank | 0.8000 | 0.7250 | 1.0000 | 0.9026 |
| fast90 selected | 0.7125 | 0.6375 | 1.0000 | 0.8685 |
| fast90 + chem_light rerank | 0.7625 | 0.6875 | 1.0000 | 0.8777 |

Rerank diagnostic on original V2-1:

- Changed predictions: 5 / 80
- Good changes: 4
- Bad changes: 0
- Candidate oracle: 0.8625
- Generated preference pairs: 9

Panel: `mixed_uob_uspto_realworld_60`

| source | canonical exact | raw exact | valid | mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| original V2-1 selected | 0.4167 | 0.3500 | 0.9667 | 0.6169 |
| original V2-1 + chem_light rerank | 0.4333 | 0.3667 | 0.9667 | 0.6199 |

Breakdown after rerank:

| group | canonical exact | raw exact | valid | mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| real_world / chinese_exam | 0.0000 | 0.0000 | 1.0000 | 0.1058 |
| uob | 0.8500 | 0.7000 | 1.0000 | 0.9628 |
| uspto | 0.4500 | 0.4000 | 0.9000 | 0.8100 |

Rerank diagnostic on mixed60:

- Changed predictions: 5 / 60
- Good changes: 1
- Bad changes: 0
- Candidate oracle: 0.4667
- Generated preference pairs: 3

Interpretation:

- The reranker is still low risk on a broader non-weak panel.
- The larger bottleneck is candidate generation, especially `real_world/chinese_exam`, where the correct answer is not present in the candidate set.
- DPO from current saved candidates is premature because the preference set is too small and cannot fix absent correct candidates.

## Reproduce

```bash
cd /root/autodl-tmp/data/platform_migration_bundle_20260531
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

OUT=/root/autodl-fs/outputs_v2/v2_1_original_compare/eval/uob_medium_80

python V2-1/scripts/rerank_ocsr_candidates.py \
  --prediction-jsonl "$OUT/pred.jsonl" \
  --output-jsonl "$OUT/rerank_chem_light_pred.jsonl" \
  --labels-jsonl V2-1/reports/fast90_panels_v1/uob_medium_80/annotations/labels.jsonl \
  --report-json "$OUT/rerank_chem_light_report.json" \
  --preference-jsonl "$OUT/preference_pairs.jsonl"

python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
  --benchmark-jsonl V2-1/reports/fast90_panels_v1/uob_medium_80/annotations/labels.jsonl \
  --prediction-jsonl "$OUT/rerank_chem_light_pred.jsonl" \
  --report-json "$OUT/rerank_chem_light_eval_report.json" \
  --details-jsonl "$OUT/rerank_chem_light_details.jsonl"
```

## Next

Do not continue the fast90 LoRA route as the main route. It underperforms original V2-1 and does not add candidate oracle diversity on UOB80.

Recommended next route:

1. Integrate the reranker into candidate-enabled inference as a default postprocess.
2. Run a small high-resolution/prompt/preprocessing probe on `real_world/chinese_exam` before more LoRA training.
3. Add focused real-world/USpto examples only if the probe shows the current model can generate near-correct candidates.
4. Use DPO only after more candidate panels produce a meaningful preference dataset.
