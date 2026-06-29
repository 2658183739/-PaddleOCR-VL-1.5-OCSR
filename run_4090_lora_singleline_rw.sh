#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/PaddleOCR-VL-0.9B}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python V2/scripts/build_singleline_rw_sft_dataset.py --project-root "$PROJECT_ROOT"
python V2/scripts/audit_singleline_training_dataset.py --project-root "$PROJECT_ROOT" --report V2/reports/singleline_rw_dataset_audit_runtime.json
python V2/scripts/summarize_singleline_dataset_stats.py --project-root "$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/V2/outputs/singleline_rw_lora"
EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(resume_from_checkpoint="$RESUME_FROM_CHECKPOINT")
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" paddleformers-cli train V2/configs/ocsr_lora_singleline_rw_4090.yaml model_name_or_path="$MODEL_DIR" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$PROJECT_ROOT/V2/outputs/singleline_rw_lora/train_singleline_rw.log"
