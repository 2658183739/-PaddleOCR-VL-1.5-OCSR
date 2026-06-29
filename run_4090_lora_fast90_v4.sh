#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/V2-1/outputs/export}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./V2-1/outputs_v2/fast90_from_v1_sft}"
MAX_STEPS="${MAX_STEPS:-320}"
SAVE_STEPS="${SAVE_STEPS:-80}"

if [[ "$OUTPUT_DIR" = /* ]]; then
  OUTPUT_PATH="$OUTPUT_DIR"
else
  OUTPUT_PATH="$PROJECT_ROOT/$OUTPUT_DIR"
fi

python V2-1/scripts/build_fast90_eval_panels.py --project-root "$PROJECT_ROOT" --output-root V2-1/reports/fast90_panels_v1

mkdir -p "$OUTPUT_PATH"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FLAGS_allocator_strategy="${FLAGS_allocator_strategy:-auto_growth}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" paddleformers-cli train V2-1/configs/ocsr_lora_fast90_4090.yaml \
  model_name_or_path="$MODEL_DIR" \
  output_dir="$OUTPUT_DIR" \
  logging_dir="$OUTPUT_DIR/visualdl_logs" \
  max_steps="$MAX_STEPS" \
  save_steps="$SAVE_STEPS" \
  overwrite_output_dir=true \
  2>&1 | tee -a "$OUTPUT_PATH/train_fast90.log"
