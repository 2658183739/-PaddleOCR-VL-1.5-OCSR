#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_ROOT="${V3_BACKUP_ROOT:-/root/autodl-fs/V3_results}"
WORKERS="${V3_INFER_WORKERS:-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${V3_OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
test -s V3/evidence/probe_analysis.json

WINNER_DATASET="$(python -c 'import json; print(json.load(open("V3/evidence/probe_analysis.json", encoding="utf-8"))["winner"]["dataset_path"])')"
VALIDITY_FLOOR="$(python -c 'import json; print(max(0.0, json.load(open("V3/evidence/probe_analysis.json", encoding="utf-8"))["winner"]["min_valid_rate"] - 0.005))')"

if [[ ! -s V3/outputs/final_s1/train_results.json ]]; then
  bash V3/run_a100_stage.sh final \
    train_dataset_path="$WINNER_DATASET" \
    seed=20260717 \
    max_steps=1400 \
    do_eval=false \
    eval_steps=999999 \
    save_steps=200 \
    per_device_train_batch_size=32 \
    gradient_accumulation_steps=1 \
    save_checkpoint_format=sharding_io \
    load_checkpoint_format=sharding_io \
    output_dir=./V3/outputs/final_s1 \
    logging_dir=./V3/outputs/final_s1/visualdl_logs
fi

mkdir -p "$BACKUP_ROOT/outputs/final_s1" "$BACKUP_ROOT/evidence"
rsync -a V3/outputs/final_s1/ "$BACKUP_ROOT/outputs/final_s1/"
rsync -a V3/logs/ "$BACKUP_ROOT/logs/"

python V3/scripts/eval_latest_checkpoints.py \
  --project-root . \
  --phase final_s1 \
  --all-checkpoints \
  --workers "$WORKERS" \
  --eval-root V3/eval_runs_final \
  --summary-csv V3/evidence/final_checkpoint_eval_summary.csv \
  --summary-md V3/evidence/final_checkpoint_eval_summary.md

python V3/scripts/select_best_checkpoint.py \
  --eval-root V3/eval_runs_final \
  --phase final_s1 \
  --validity-floor "$VALIDITY_FLOOR" \
  --output-json V3/evidence/final_checkpoint_selection.json

FINAL_CHECKPOINT="$(python -c 'import json; print(json.load(open("V3/evidence/final_checkpoint_selection.json", encoding="utf-8"))["winner"]["checkpoint"])')"
FINAL_CHECKPOINT_DIR="./V3/outputs/final_s1/$FINAL_CHECKPOINT"

paddleformers-cli export V3/configs/export_selected.yaml \
  model_name_or_path=./V3/models/v2_1_export \
  output_dir="$FINAL_CHECKPOINT_DIR"

python V3/scripts/prepare_runtime_export.py \
  --runtime-dir "$FINAL_CHECKPOINT_DIR/export" \
  --base-model-dir V3/models/v2_1_export
if [[ -d V3/models/final_selected_export ]]; then
  mv V3/models/final_selected_export "V3/models/final_selected_export.previous.$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p V3/models/final_selected_export
rsync -a "$FINAL_CHECKPOINT_DIR/export/" V3/models/final_selected_export/

if [[ ! -s V3/outputs/hard_replay_s1/train_results.json ]]; then
  bash V3/run_a100_stage.sh hard_replay \
    seed=20260718 \
    max_steps=300 \
    do_eval=false \
    eval_steps=999999 \
    save_steps=300 \
    per_device_train_batch_size=32 \
    gradient_accumulation_steps=1 \
    save_checkpoint_format=sharding_io \
    load_checkpoint_format=sharding_io \
    output_dir=./V3/outputs/hard_replay_s1 \
    logging_dir=./V3/outputs/hard_replay_s1/visualdl_logs
fi

python V3/scripts/eval_latest_checkpoints.py \
  --project-root . \
  --phase hard_replay_s1 \
  --workers "$WORKERS" \
  --base-model-dir V3/models/final_selected_export \
  --eval-root V3/eval_runs_hard \
  --summary-csv V3/evidence/hard_replay_eval_summary.csv \
  --summary-md V3/evidence/hard_replay_eval_summary.md

python V3/scripts/compare_final_candidates.py \
  --baseline-root "V3/eval_runs_final/final_s1/$FINAL_CHECKPOINT" \
  --candidate-root V3/eval_runs_hard/hard_replay_s1/checkpoint-300 \
  --output-json V3/evidence/final_vs_hard_replay.json

for panel in legacy_core_dev legacy_region_dev; do
  python V3/scripts/compare_eval_runs.py \
    --baseline-details "V3/eval_runs_final/final_s1/$FINAL_CHECKPOINT/$panel/details.jsonl" \
    --candidate-details "V3/eval_runs_hard/hard_replay_s1/checkpoint-300/$panel/details.jsonl" \
    --cluster-field structure_id \
    --output-json "V3/evidence/final_vs_hard_replay_${panel}_paired.json" \
    > "V3/evidence/final_vs_hard_replay_${panel}_paired.log"
done

WINNER_LABEL="$(python -c 'import json; print(json.load(open("V3/evidence/final_vs_hard_replay.json", encoding="utf-8"))["winner"]["label"])')"
if [[ "$WINNER_LABEL" == "hard_replay" ]]; then
  paddleformers-cli export V3/configs/export_selected.yaml \
    model_name_or_path=./V3/models/final_selected_export \
    output_dir=./V3/outputs/hard_replay_s1/checkpoint-300
  python V3/scripts/prepare_runtime_export.py \
    --runtime-dir V3/outputs/hard_replay_s1/checkpoint-300/export \
    --base-model-dir V3/models/final_selected_export
  WINNER_EXPORT=./V3/outputs/hard_replay_s1/checkpoint-300/export
else
  WINNER_EXPORT=./V3/models/final_selected_export
fi

if [[ -d V3/models/final_best_export ]]; then
  mv V3/models/final_best_export "V3/models/final_best_export.previous.$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p V3/models/final_best_export
rsync -a "$WINNER_EXPORT/" V3/models/final_best_export/
test -s V3/models/final_best_export/config.json

mkdir -p "$BACKUP_ROOT/outputs/hard_replay_s1" "$BACKUP_ROOT/models/final_best_export"
rsync -a V3/outputs/hard_replay_s1/ "$BACKUP_ROOT/outputs/hard_replay_s1/"
rsync -a V3/models/final_best_export/ "$BACKUP_ROOT/models/final_best_export/"
rsync -a V3/eval_runs_final/ "$BACKUP_ROOT/eval_runs_final/"
rsync -a V3/eval_runs_hard/ "$BACKUP_ROOT/eval_runs_hard/"
rsync -a V3/evidence/ "$BACKUP_ROOT/evidence/"
date -Iseconds > "$BACKUP_ROOT/evidence/final_pipeline_complete.txt"
echo "[DONE] final training, checkpoint selection, hard replay, and export completed"
