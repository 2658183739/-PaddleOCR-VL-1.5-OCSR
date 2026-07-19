#!/usr/bin/env bash
set -euo pipefail

if [[ "${UNLOCK_FINAL_TEST:-}" != "FINAL_MODEL_SELECTED" ]]; then
  echo "Refusing to read locked test. Freeze the final model, then set UNLOCK_FINAL_TEST=FINAL_MODEL_SELECTED." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/V3/models/final_selected_export}"
GREEDY_WORKERS="${V3_GREEDY_WORKERS:-4}"
BEAM_WORKERS="${V3_BEAM_WORKERS:-${V3_INFER_WORKERS:-1}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$PROJECT_ROOT/V3/eval_runs_locked/$STAMP"

cd "$PROJECT_ROOT"
test -s "$MODEL_DIR/config.json"
test -s V3/data/eval/wild_strict_v3/labels.jsonl
test -s V3/evidence/generation_policy_selection.json
mkdir -p \
  "$OUT/wild_strict" \
  "$OUT/wild_scaffold_novel" \
  "$OUT/wild_symbolic" \
  "$OUT/private_photo"

sha256sum \
  "$MODEL_DIR/config.json" \
  "$MODEL_DIR"/model*.safetensors \
  V3/data/eval/wild_strict_v3/labels.jsonl \
  V3/data/eval/wild_strict_scaffold_novel_v3/labels.jsonl \
  V3/data/eval/wild_symbolic_v3/labels.jsonl \
  V3/configs/prompt.txt \
  V3/configs/prompt_symbolic.txt \
  V3/evidence/generation_policy_selection.json > "$OUT/locked_test_manifest.sha256"

GENERATION_ARGS=()
GENERATION_LABEL="$(python -c 'import json; print(json.load(open("V3/evidence/generation_policy_selection.json", encoding="utf-8"))["winner"]["label"])')"
CANONICAL_WORKERS="$GREEDY_WORKERS"
if [[ "$GENERATION_LABEL" == "beam4_return4" || "$GENERATION_LABEL" == "beam4_chem_light" ]]; then
  GENERATION_ARGS=(--num-beams 4 --num-return-sequences 4 --save-candidates)
  CANONICAL_WORKERS="$BEAM_WORKERS"
fi

WILD_PRED="$OUT/wild_strict/pred.jsonl"
if [[ "$GENERATION_LABEL" == "beam4_chem_light" ]]; then
  WILD_PRED="$OUT/wild_strict/pred_beam_raw.jsonl"
fi

python V3/scripts/run_sharded_inference.py \
  --model-dir "$MODEL_DIR" \
  --benchmark-jsonl V3/data/eval/wild_strict_v3/labels.jsonl \
  --project-root "$PROJECT_ROOT" \
  --output-jsonl "$WILD_PRED" \
  --prompt-file V3/configs/prompt.txt \
  --device cuda \
  --torch-dtype bfloat16 \
  --max-new-tokens 256 \
  --min-pixels 50176 \
  --max-pixels 200704 \
  --workers "$CANONICAL_WORKERS" \
  "${GENERATION_ARGS[@]}"

if [[ "$GENERATION_LABEL" == "beam4_chem_light" ]]; then
  python V3/scripts/rerank_ocsr_candidates.py \
    --prediction-jsonl "$WILD_PRED" \
    --output-jsonl "$OUT/wild_strict/pred.jsonl" \
    --mode chem_light
fi

python V3/scripts/evaluate_ocsr_predictions_detailed.py \
  --benchmark-jsonl V3/data/eval/wild_strict_v3/labels.jsonl \
  --prediction-jsonl "$OUT/wild_strict/pred.jsonl" \
  --report-json "$OUT/wild_strict/report.json" \
  --details-jsonl "$OUT/wild_strict/details.jsonl"

# The scaffold-novel panel is a strict subset, so reuse the frozen predictions.
python V3/scripts/evaluate_ocsr_predictions_detailed.py \
  --benchmark-jsonl V3/data/eval/wild_strict_scaffold_novel_v3/labels.jsonl \
  --prediction-jsonl "$OUT/wild_strict/pred.jsonl" \
  --report-json "$OUT/wild_scaffold_novel/report.json" \
  --details-jsonl "$OUT/wild_scaffold_novel/details.jsonl"

# Symbolic labels are intentionally not RDKit-canonical targets. They were not
# used to select the canonical decoder policy, so freeze this separate track to
# greedy rather than transferring an unvalidated beam/reranker decision.
printf '%s\n' \
  '{' \
  '  "label": "greedy",' \
  '  "selected_on_locked_data": false,' \
  '  "reason": "symbolic is a separate transcription track without a development decoder ablation"' \
  '}' > "$OUT/wild_symbolic/generation_policy.json"

python V3/scripts/run_sharded_inference.py \
  --model-dir "$MODEL_DIR" \
  --benchmark-jsonl V3/data/eval/wild_symbolic_v3/labels.jsonl \
  --project-root "$PROJECT_ROOT" \
  --output-jsonl "$OUT/wild_symbolic/pred.jsonl" \
  --prompt-file V3/configs/prompt_symbolic.txt \
  --device cuda \
  --torch-dtype bfloat16 \
  --max-new-tokens 256 \
  --min-pixels 50176 \
  --max-pixels 200704 \
  --workers "$GREEDY_WORKERS"

python V3/scripts/evaluate_symbolic_predictions.py \
  --benchmark-jsonl V3/data/eval/wild_symbolic_v3/labels.jsonl \
  --prediction-jsonl "$OUT/wild_symbolic/pred.jsonl" \
  --report-json "$OUT/wild_symbolic/report.json" \
  --details-jsonl "$OUT/wild_symbolic/details.jsonl"

PRIVATE_LABELS="V3/data/eval/private_photo_v3/labels.jsonl"
if [[ -s "$PRIVATE_LABELS" ]]; then
  python V3/scripts/run_sharded_inference.py \
    --model-dir "$MODEL_DIR" \
    --benchmark-jsonl "$PRIVATE_LABELS" \
    --project-root "$PROJECT_ROOT" \
    --output-jsonl "$OUT/private_photo/pred.jsonl" \
    --prompt-file V3/configs/prompt.txt \
    --device cuda \
    --torch-dtype bfloat16 \
    --max-new-tokens 256 \
    --min-pixels 50176 \
    --max-pixels 200704 \
    --workers "$CANONICAL_WORKERS" \
    "${GENERATION_ARGS[@]}"

  python V3/scripts/evaluate_ocsr_predictions_detailed.py \
    --benchmark-jsonl "$PRIVATE_LABELS" \
    --prediction-jsonl "$OUT/private_photo/pred.jsonl" \
    --report-json "$OUT/private_photo/report.json" \
    --details-jsonl "$OUT/private_photo/details.jsonl"
fi

echo "Locked test completed once at: $OUT"
