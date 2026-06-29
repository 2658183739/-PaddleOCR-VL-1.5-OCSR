#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/PaddleOCR-VL-0.9B}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python V2-1/scripts/build_singleline_rw_v2_dataset.py --project-root "$PROJECT_ROOT"
python V2-1/scripts/audit_singleline_training_dataset.py \
  --project-root "$PROJECT_ROOT" \
  --train V2-1/data/sft_materialized/train_singleline_rw_v2_messages.jsonl \
  --eval \
    V2-1/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl \
    V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl \
    V2-1/data/eval/weak_domain_v2/annotations/labels.jsonl \
  --report V2-1/reports/singleline_rw_v2_dataset_audit_runtime.json
python V2-1/scripts/summarize_singleline_dataset_stats.py \
  --project-root "$PROJECT_ROOT" \
  --input V2-1/data/sft_materialized/train_singleline_rw_v2_messages.jsonl \
  --report V2-1/reports/singleline_rw_v2_dataset_stats.json

mkdir -p "$PROJECT_ROOT/V2-1/outputs_v2/singleline_rw_lora"
EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(resume_from_checkpoint="$RESUME_FROM_CHECKPOINT")
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" paddleformers-cli train V2-1/configs/ocsr_lora_singleline_rw_v2_4090.yaml model_name_or_path="$MODEL_DIR" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$PROJECT_ROOT/V2-1/outputs_v2/singleline_rw_lora/train_singleline_rw_v2.log"
