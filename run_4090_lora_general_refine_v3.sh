#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/V2-1/outputs/export}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./V2-1/outputs_v2/general_refine_from_v1_sft}"

python V2-1/scripts/audit_singleline_training_dataset.py \
  --project-root "$PROJECT_ROOT" \
  --train V2-1/data/sft_materialized/train_singleline_rw_messages.jsonl \
  --eval \
    V2-1/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl \
    V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl \
    V2-1/data/eval/weak_domain_v2/annotations/labels.jsonl \
  --report V2-1/reports/singleline_general_refine_dataset_audit_runtime.json
python V2-1/scripts/summarize_singleline_dataset_stats.py \
  --project-root "$PROJECT_ROOT" \
  --input V2-1/data/sft_materialized/train_singleline_rw_messages.jsonl \
  --report V2-1/reports/singleline_general_refine_dataset_stats.json

mkdir -p "$PROJECT_ROOT/$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(resume_from_checkpoint="$RESUME_FROM_CHECKPOINT")
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FLAGS_allocator_strategy="${FLAGS_allocator_strategy:-auto_growth}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" paddleformers-cli train V2-1/configs/ocsr_lora_general_refine_4090.yaml \
  model_name_or_path="$MODEL_DIR" \
  output_dir="$OUTPUT_DIR" \
  logging_dir="$OUTPUT_DIR/visualdl_logs" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$PROJECT_ROOT/$OUTPUT_DIR/train_general_refine.log"
