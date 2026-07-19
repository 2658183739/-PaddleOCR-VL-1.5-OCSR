#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_ROOT="${V3_BACKUP_ROOT:-/root/autodl-fs/V3_results}"
WORKERS="${V3_INFER_WORKERS:-4}"
EVAL_ROOT="V3/eval_runs_probes"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"

python V3/scripts/eval_latest_checkpoints.py \
  --project-root . \
  --phase data_11_s1 \
  --phase data_00_s1 \
  --phase data_10_s1 \
  --phase data_01_s1 \
  --phase data_11_s2 \
  --phase data_10_s2 \
  --phase data_00_s2 \
  --phase data_01_s2 \
  --phase aug_dose2_s1 \
  --workers "$WORKERS" \
  --eval-root "$EVAL_ROOT" \
  --summary-csv V3/evidence/probe_checkpoint_eval_summary.csv \
  --summary-md V3/evidence/probe_checkpoint_eval_summary.md

python V3/scripts/eval_latest_checkpoints.py \
  --project-root . \
  --phase warmstart_control_s1 \
  --workers "$WORKERS" \
  --base-model-dir V3/models/paddleocr_vl_1_5_base \
  --eval-root "$EVAL_ROOT" \
  --summary-csv V3/evidence/probe_checkpoint_eval_summary.csv \
  --summary-md V3/evidence/probe_checkpoint_eval_summary.md

python V3/scripts/analyze_probe_results.py \
  --eval-root "$EVAL_ROOT" \
  --output-json V3/evidence/probe_analysis.json \
  --output-md V3/evidence/probe_analysis.md

python V3/scripts/update_experiment_matrix.py \
  --matrix V3/evidence/experiment_matrix.csv \
  --analysis V3/evidence/probe_analysis.json

# Validate that final's compact checkpoint format preserves independently
# exportable LoRA snapshots before committing to the 1400-step run.
SHARDING_SMOKE=V3/outputs/smoke_h800_sharding_io
if [[ ! -s "$BACKUP_ROOT/evidence/sharding_io_smoke_complete.txt" ]]; then
  bash V3/run_a100_stage.sh probe_b \
    seed=20260717 \
    max_steps=2 \
    do_eval=false \
    eval_steps=999999 \
    save_steps=1 \
    per_device_train_batch_size=32 \
    gradient_accumulation_steps=1 \
    save_checkpoint_format=sharding_io \
    load_checkpoint_format=sharding_io \
    output_dir=./$SHARDING_SMOKE \
    logging_dir=./$SHARDING_SMOKE/visualdl_logs

  python V3/scripts/eval_latest_checkpoints.py \
    --project-root . \
    --phase smoke_h800_sharding_io \
    --all-checkpoints \
    --workers 1 \
    --limit 1 \
    --eval-root V3/eval_runs_sharding_smoke \
    --summary-csv V3/evidence/sharding_io_smoke_summary.csv \
    --summary-md V3/evidence/sharding_io_smoke_summary.md
  date -Iseconds > "$BACKUP_ROOT/evidence/sharding_io_smoke_complete.txt"
fi

mkdir -p "$BACKUP_ROOT/eval_runs_probes" "$BACKUP_ROOT/evidence"
rsync -a "$EVAL_ROOT/" "$BACKUP_ROOT/eval_runs_probes/"
rsync -a V3/evidence/ "$BACKUP_ROOT/evidence/"
date -Iseconds > "$BACKUP_ROOT/evidence/probe_eval_complete.txt"
echo "[DONE] probe generation evaluation and factorial analysis completed"
