#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_ROOT="${V3_BACKUP_ROOT:-/root/autodl-fs/V3_results}"
BEAM_ROOT="V3/eval_runs_generation/beam4_return4"
OUT="V3/eval_runs_generation/beam4_chem_light"

cd "$PROJECT_ROOT"
test -s V3/evidence/generation_policy_selection.json
test -s "$BEAM_ROOT/legacy_core_dev/pred.jsonl"
test -s "$BEAM_ROOT/legacy_region_dev/pred.jsonl"

run_panel() {
  local name="$1"
  local labels="$2"
  local panel="$OUT/$name"
  mkdir -p "$panel"
  if [[ ! -s "$panel/report.json" ]]; then
    python V3/scripts/rerank_ocsr_candidates.py \
      --prediction-jsonl "$BEAM_ROOT/$name/pred.jsonl" \
      --output-jsonl "$panel/pred.jsonl" \
      --labels-jsonl "$labels" \
      --report-json "$panel/rerank_report.json" \
      --mode chem_light

    python V3/scripts/evaluate_ocsr_predictions_detailed.py \
      --benchmark-jsonl "$labels" \
      --prediction-jsonl "$panel/pred.jsonl" \
      --report-json "$panel/report.json" \
      --details-jsonl "$panel/details.jsonl"
  fi
}

run_panel legacy_core_dev V3/data/eval/dev_legacy_core_strict/labels.jsonl
run_panel legacy_region_dev V3/data/eval/dev_legacy_region_strict/labels.jsonl

CURRENT_LABEL="$(python -c 'import json; print(json.load(open("V3/evidence/generation_policy_selection.json", encoding="utf-8"))["winner"]["label"])')"
CURRENT_ROOT="$(python -c 'import json; print(json.load(open("V3/evidence/generation_policy_selection.json", encoding="utf-8"))["winner"]["root"])')"
cp V3/evidence/generation_policy_selection.json V3/evidence/generation_policy_beam_selection.json

python V3/scripts/compare_final_candidates.py \
  --baseline-label "$CURRENT_LABEL" \
  --baseline-root "$CURRENT_ROOT" \
  --candidate-label beam4_chem_light \
  --candidate-root "$OUT" \
  --output-json V3/evidence/generation_policy_selection.json

for panel in legacy_core_dev legacy_region_dev; do
  python V3/scripts/compare_eval_runs.py \
    --baseline-details "$CURRENT_ROOT/$panel/details.jsonl" \
    --candidate-details "$OUT/$panel/details.jsonl" \
    --cluster-field structure_id \
    --output-json "V3/evidence/generation_policy_chem_light_${panel}_paired.json" \
    > "V3/evidence/generation_policy_chem_light_${panel}_paired.log"
done

mkdir -p "$BACKUP_ROOT/eval_runs_generation" "$BACKUP_ROOT/evidence"
rsync -a V3/eval_runs_generation/ "$BACKUP_ROOT/eval_runs_generation/"
rsync -a V3/evidence/ "$BACKUP_ROOT/evidence/"
date -Iseconds > "$BACKUP_ROOT/evidence/candidate_rerank_complete.txt"
echo "[DONE] fixed beam candidate pool chem-light rerank completed"
