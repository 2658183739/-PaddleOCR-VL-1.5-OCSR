#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_ROOT="${V3_BACKUP_ROOT:-/root/autodl-fs/V3_results}"
# Beam4 keeps four hypotheses per token and does not fit four model replicas
# in an 80GB H800. Use one worker by default; greedy final evaluation may use 4.
WORKERS="${V3_INFER_WORKERS:-1}"
MODEL_DIR="V3/models/final_best_export"
OUT="V3/eval_runs_generation/beam4_return4"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
test -s "$MODEL_DIR/config.json"
test -s V3/evidence/final_vs_hard_replay.json
mkdir -p "$OUT/legacy_core_dev" "$OUT/legacy_region_dev" "$BACKUP_ROOT/evidence"

run_panel() {
  local name="$1"
  local labels="$2"
  local panel="$OUT/$name"

  if [[ ! -s "$panel/report.json" ]]; then
    python V3/scripts/run_sharded_inference.py \
      --model-dir "$MODEL_DIR" \
      --benchmark-jsonl "$labels" \
      --project-root "$PROJECT_ROOT" \
      --output-jsonl "$panel/pred.jsonl" \
      --prompt-file V3/configs/prompt.txt \
      --device cuda \
      --torch-dtype bfloat16 \
      --max-new-tokens 256 \
      --min-pixels 50176 \
      --max-pixels 200704 \
      --workers "$WORKERS" \
      --num-beams 4 \
      --num-return-sequences 4 \
      --save-candidates

    python V3/scripts/evaluate_ocsr_predictions_detailed.py \
      --benchmark-jsonl "$labels" \
      --prediction-jsonl "$panel/pred.jsonl" \
      --report-json "$panel/report.json" \
      --details-jsonl "$panel/details.jsonl"
  fi
}

run_panel legacy_core_dev V3/data/eval/dev_legacy_core_strict/labels.jsonl
run_panel legacy_region_dev V3/data/eval/dev_legacy_region_strict/labels.jsonl

GREEDY_ROOT="$(python -c 'import json; print(json.load(open("V3/evidence/final_vs_hard_replay.json", encoding="utf-8"))["winner"]["root"])')"
python V3/scripts/compare_final_candidates.py \
  --baseline-label greedy \
  --baseline-root "$GREEDY_ROOT" \
  --candidate-label beam4_return4 \
  --candidate-root "$OUT" \
  --output-json V3/evidence/generation_policy_selection.json

for panel in legacy_core_dev legacy_region_dev; do
  python V3/scripts/compare_eval_runs.py \
    --baseline-details "$GREEDY_ROOT/$panel/details.jsonl" \
    --candidate-details "$OUT/$panel/details.jsonl" \
    --cluster-field structure_id \
    --output-json "V3/evidence/generation_policy_${panel}_paired.json" \
    > "V3/evidence/generation_policy_${panel}_paired.log"
done

mkdir -p "$BACKUP_ROOT/eval_runs_generation"
rsync -a V3/eval_runs_generation/ "$BACKUP_ROOT/eval_runs_generation/"
rsync -a V3/evidence/ "$BACKUP_ROOT/evidence/"
date -Iseconds > "$BACKUP_ROOT/evidence/generation_ablation_complete.txt"
echo "[DONE] greedy vs beam4/return4 generation ablation completed"
