#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGE="${1:-}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -n "$STAGE" ]]; then
  shift
fi

case "$STAGE" in
  probe_a) CONFIG="V3/configs/probe_a_control_a100.yaml" ;;
  probe_d) CONFIG="V3/configs/probe_d_wild_only_a100.yaml" ;;
  probe_e) CONFIG="V3/configs/probe_e_aug_only_a100.yaml" ;;
  probe_b) CONFIG="V3/configs/probe_b_recommended_a100.yaml" ;;
  probe_c) CONFIG="V3/configs/probe_c_real_heavy_a100.yaml" ;;
  probe_base15) CONFIG="V3/configs/probe_base15_recommended_a100.yaml" ;;
  final) CONFIG="V3/configs/final_continue_a100.yaml" ;;
  hard_replay) CONFIG="V3/configs/hard_replay_a100.yaml" ;;
  *)
    echo "Usage: bash V3/run_a100_stage.sh {probe_a|probe_d|probe_e|probe_b|probe_c|probe_base15|final|hard_replay}" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
command -v paddleformers-cli >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
test -s V3/models/v2_1_export/model-00001-of-00001.safetensors
test -s V3/models/paddleocr_vl_1_5_base/model.safetensors
test -s V3/data/sft_materialized/train_v3_b_recommended.jsonl
test -s V3/data/sft_materialized/dev_legacy_region_strict_messages.jsonl
mkdir -p "V3/logs"

LOG="V3/logs/${STAGE}_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" paddleformers-cli train "$CONFIG" "$@" 2>&1 | tee "$LOG"
