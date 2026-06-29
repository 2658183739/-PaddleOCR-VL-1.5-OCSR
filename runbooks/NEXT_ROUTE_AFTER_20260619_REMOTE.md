# Next Route After 2026-06-19 Remote Experiments

## Current Best Usable Route

Use original V2-1 export plus `chem_light` candidate rerank for candidate-enabled inference.

Known checks:

| panel | selected canonical exact | reranked canonical exact | note |
| --- | ---: | ---: | --- |
| `uob_medium_80` | 0.7500 | 0.8000 | strong, low-risk improvement |
| `mixed_uob_uspto_realworld_60` | 0.4167 | 0.4333 | small gain, no bad changes |
| `realworld20_highpix_notta_probe` | 0.0000 | 0.0000 | no correct candidate generated |

Keep `V2-1/outputs/export` as the baseline model. Do not replace it with `fast90_from_v1_sft`.

## Routes To Stop

- Stop `fast90_from_v1_sft` as the main model route. It underperformed original V2-1 on `uob_medium_80`:
  - original V2-1: 0.7500 canonical exact;
  - fast90: 0.7125 canonical exact.
- Stop high-pixel TTA for real-world as a blind inference fix. It was too slow and high-pixel no-TTA still produced 0 / 20 exact with 0 / 20 oracle.
- Do not run DPO yet for real-world. There are no positive preference pairs because the correct answer is absent from the candidates.

## Next Useful Work

### 1. Integrate Candidate Rerank

Make `V2-1/scripts/rerank_ocsr_candidates.py --mode chem_light` part of the inference pipeline whenever `--save-candidates` is available.

Acceptance check:

- Reproduce UOB80 0.8000 canonical exact.
- Confirm mixed60 does not regress.

### 2. Build A Focused Real-World Data Probe

Target the failure mode directly instead of weak-domain expansion.

Minimum data target:

- 100-300 real-world/chinese-exam-like rendered or collected structures.
- Keep a held-out 20-50 sample validation panel separate from training.
- Include the exact image style that failed: dense structures, exam-style scans, and long/cyclic molecules.

Stop condition:

- If a 20-sample real-world validation probe still has candidate oracle 0 after short targeted SFT, do not scale that training run.

### 3. Run A Short Targeted Continuation Only After Data Exists

Train from original V2-1 export, not from fast90.

Suggested guardrails:

- Output under `/root/autodl-fs/outputs_v2`.
- Save a fixed validation report before and after training.
- Use a short smoke first, then only continue if candidate oracle improves on real-world.

### 4. DPO Later

DPO becomes useful only when saved candidates contain wrong selected answers plus correct alternatives.

Current preference data:

- UOB80: 9 preference pairs.
- mixed60: 3 preference pairs.
- realworld20 high-pixel no-TTA: 0 preference pairs.

This is not enough for a meaningful DPO run.

## Remote Paths

Remote root:

```bash
/root/autodl-tmp/data/platform_migration_bundle_20260531
```

Large-output root:

```bash
/root/autodl-fs/outputs_v2
```

Important reports:

```bash
/root/autodl-fs/outputs_v2/v2_1_original_compare/eval/uob_medium_80
/root/autodl-fs/outputs_v2/v2_1_original_compare/eval/mixed_uob_uspto_realworld_60
/root/autodl-fs/outputs_v2/v2_1_original_compare/eval/realworld20_highpix_notta_probe
```

Local report copies:

```bash
V2-1/reports/remote_eval_20260619/
```
