# Main Eval Reward Policy Check - 2026-06-27

## Goal

Check whether the reward-policy candidate reranker from the 770 diagnostic panel still improves the full 1344-sample main evaluation scope:

- `canonical_smiles_main_v1`: 767 samples
- `weak_domain_v2`: 577 samples

## Run

Remote root:

```bash
/root/autodl-tmp/data/platform_migration_bundle_20260531
```

Remote output:

```bash
/root/autodl-fs/outputs_v2/main_eval_with_candidates_20260627_fast_notta
```

Local retrieved reports:

```bash
V2-1/reports/main_eval_with_candidates_20260627_fast_notta
```

Runner:

```bash
V2-1/run_4090_eval_main_with_candidates_v1.sh
```

Decode setting:

- `num_beams=4`
- `num_return_sequences=4`
- `tta_preset=none`
- `min_pixels=50176`
- `max_pixels=200704`
- reward policy: `/root/autodl-fs/outputs_v2/reward_policy_rerank_20260626_split75/policy_raw.json`
- `fallback_mode=chem_light`
- `policy_margin=1.5`

The run reuses existing candidates where available:

- 617 canonical-overlap candidates from the 770 fast/no-TTA region-panel run.
- 577 weak-domain candidates from `V2-1/eval_runs_export_full/weak_domain_v2_beam4/pred.jsonl`.
- 150 canonical samples were newly generated in this run, including an initial 30-sample partial and a 120-sample resumed gap fill.

## Results

| Scope | Variant | Canonical exact | Raw exact | Valid SMILES | Mean Tanimoto |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical 767 | selected | 0.3703 | 0.3220 | 0.9739 | 0.6371 |
| canonical 767 | chem_light | 0.3638 | 0.3194 | 0.9752 | 0.6303 |
| canonical 767 | reward_policy | 0.3755 | 0.3286 | 0.9752 | 0.6397 |
| weak 577 | selected | 0.1300 | 0.1040 | 0.9792 | 0.3938 |
| weak 577 | chem_light | 0.1265 | 0.1005 | 0.9792 | 0.3915 |
| weak 577 | reward_policy | 0.1265 | 0.1040 | 0.9792 | 0.3956 |
| combined 1344 | selected | 0.2671 | 0.2284 | 0.9762 | 0.5323 |
| combined 1344 | chem_light | 0.2619 | 0.2254 | 0.9769 | 0.5275 |
| combined 1344 | reward_policy | 0.2686 | 0.2321 | 0.9769 | 0.5347 |

Combined reward-policy delta:

- vs selected: `+0.00149` canonical exact, `+0.00238` mean Tanimoto
- vs chem_light: `+0.00670` canonical exact, `+0.00717` mean Tanimoto

Internal change stats:

- canonical oracle-valid subset: reward policy changed 103 rows, with 9 good changes and 5 bad changes vs selected.
- weak oracle-valid subset: reward policy changed 106 rows, with 4 good changes and 6 bad changes vs selected.

## Interpretation

Reward-policy reranking is a small net positive on the full 1344 scope, but it is not yet a robust standalone solution. The gain comes mostly from canonical/UOB-style samples. Weak-domain exact match is slightly worse than selected, even though mean Tanimoto improves.

This supports using the reward signal for preference-data generation and conservative reranking, but not replacing the model output with aggressive chemistry-only rerank rules.

## Next Step

Use the candidate pools to build a post-training set:

1. Positive pairs: candidate exactly matching ground truth.
2. Hard negatives: high reward-policy score but wrong canonical SMILES, plus selected-correct/policy-wrong cases.
3. Train a small preference head or DPO-style LoRA on candidate choices before touching the base OCR/VLM model.
4. Validate first on the same 1344 split, then on the 770 diagnostic panel to catch weak-domain regressions.

## Preference Pool

Builder:

```bash
python V2-1/scripts/build_candidate_preference_dataset.py \
  --prediction-jsonl V2-1/reports/main_eval_with_candidates_20260627_fast_notta/combined/pred_selected.jsonl \
  --labels-jsonl V2-1/reports/main_eval_with_candidates_20260627_fast_notta/combined/labels.jsonl \
  --reward-policy-json V2-1/reports/reward_policy_rerank_20260626_split75/policy_raw.json \
  --output-jsonl V2-1/reports/main_eval_with_candidates_20260627_fast_notta/preference_pairs_reward_policy.jsonl \
  --report-json V2-1/reports/main_eval_with_candidates_20260627_fast_notta/preference_pairs_reward_policy_report.json \
  --max-hard-negatives-per-sample 3 \
  --policy-margin 1.5
```

Output:

- `preference_pairs_reward_policy.jsonl`: 1402 pairs
- `preference_pairs_reward_policy_report.json`

Preference pool stats:

- total samples: 1344
- samples with valid candidates: 1313
- samples with an oracle-positive candidate: 521
- samples with at least one generated pair: 516
- no valid candidates: 31

Pair counts:

- `oracle_positive_vs_selected`: 162
- `oracle_positive_vs_hard_negative`: 1210
- `oracle_positive_vs_chem_light`: 25
- `oracle_positive_vs_reward_policy`: 5

## Reward Head Smoke

The first post-training smoke run trains a small candidate-choice reward head over aggregate candidate features. It does not update PaddleOCR-VL.

Scripts:

- `V2-1/scripts/train_candidate_reward_head.py`
- `V2-1/scripts/apply_candidate_reward_head.py`

Artifacts:

- `V2-1/reports/main_eval_with_candidates_20260627_fast_notta/reward_head_smoke/reward_head.pt`
- `V2-1/reports/main_eval_with_candidates_20260627_fast_notta/reward_head_smoke/reward_head_report.json`
- `V2-1/reports/main_eval_with_candidates_20260627_fast_notta/reward_head_smoke/pred_reward_head.jsonl`
- `V2-1/reports/main_eval_with_candidates_20260627_fast_notta/reward_head_smoke/report_reward_head.json`

Training command:

```bash
python V2-1/scripts/train_candidate_reward_head.py \
  --prediction-jsonl V2-1/reports/main_eval_with_candidates_20260627_fast_notta/combined/pred_selected.jsonl \
  --labels-jsonl V2-1/reports/main_eval_with_candidates_20260627_fast_notta/combined/labels.jsonl \
  --output-dir V2-1/reports/main_eval_with_candidates_20260627_fast_notta/reward_head_smoke \
  --train-fraction 0.75 \
  --seed 20260627 \
  --epochs 200 \
  --batch-size 256 \
  --lr 0.001 \
  --hidden-dim 64 \
  --dropout 0.05 \
  --max-negatives-per-positive 8 \
  --fallback-mode chem_light \
  --margin-grid 0,0.05,0.1,0.25,0.5,0.75,1,1.5,2 \
  --log-every 25
```

Smoke train stats:

- train samples with oracle-positive candidates: 366
- dev samples with oracle-positive candidates: 150
- train pairs: 1566
- dev pairs: 622
- best margin on dev: 2.0
- final dev pair accuracy: 0.8971

Dev-only held-out candidate-choice subset:

| Variant | Samples | Canonical exact | Raw exact | Mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| selected | 150 | 0.7533 | 0.6000 | 0.9132 |
| reward head | 150 | 0.7733 | 0.6133 | 0.9147 |

Full 1344 in-sample smoke result:

| Variant | Samples | Canonical exact | Raw exact | Mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| selected | 1344 | 0.2671 | 0.2284 | 0.5323 |
| linear reward policy | 1344 | 0.2686 | 0.2321 | 0.5347 |
| reward head | 1344 | 0.3229 | 0.2746 | 0.5537 |

The full 1344 reward-head score is in-sample and should be treated as an upper-bound smoke result, not a clean generalization estimate. The clean signal is the dev-only subset: `+0.0200` canonical exact and `+0.0014` mean Tanimoto over selected, with weak-domain dev exact flat.

Apply command:

```bash
python V2-1/scripts/apply_candidate_reward_head.py \
  --checkpoint V2-1/reports/main_eval_with_candidates_20260627_fast_notta/reward_head_smoke/reward_head.pt \
  --prediction-jsonl V2-1/reports/main_eval_with_candidates_20260627_fast_notta/combined/pred_selected.jsonl \
  --labels-jsonl V2-1/reports/main_eval_with_candidates_20260627_fast_notta/combined/labels.jsonl \
  --output-jsonl V2-1/reports/main_eval_with_candidates_20260627_fast_notta/reward_head_smoke/pred_reward_head_reapply.jsonl
```

Reapply check:

- `pred_reward_head.jsonl` and `pred_reward_head_reapply.jsonl` have identical SHA-256 prefix: `53034a1c44526cfb`.
