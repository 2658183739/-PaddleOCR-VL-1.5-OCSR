#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_ROOT="${V3_BACKUP_ROOT:-/root/autodl-fs/V3_results}"
EVAL_ROOT="V3/eval_runs_probes"
OUT="V3/evidence/probe_paired"

cd "$PROJECT_ROOT"
mkdir -p "$OUT" "$BACKUP_ROOT/evidence"

compare_panel() {
  local baseline="$1"
  local candidate="$2"
  local panel="$3"
  local output="$OUT/${candidate}_vs_${baseline}_${panel}.json"
  if [[ -s "$output" ]]; then
    return
  fi
  python V3/scripts/compare_eval_runs.py \
    --baseline-details "$EVAL_ROOT/$baseline/checkpoint-250/$panel/details.jsonl" \
    --candidate-details "$EVAL_ROOT/$candidate/checkpoint-250/$panel/details.jsonl" \
    --cluster-field structure_id \
    --bootstrap-iterations 10000 \
    --output-json "$output" \
    > "$OUT/${candidate}_vs_${baseline}_${panel}.log"
}

for panel in legacy_core_dev legacy_region_dev; do
  for candidate in data_10_s1 data_01_s1 data_11_s1; do
    compare_panel data_00_s1 "$candidate" "$panel"
  done
  for candidate in data_10_s2 data_01_s2 data_11_s2; do
    compare_panel data_00_s2 "$candidate" "$panel"
  done
  compare_panel data_11_s1 aug_dose2_s1 "$panel"
  compare_panel warmstart_control_s1 data_11_s1 "$panel"
done

python V3/scripts/summarize_probe_pairwise.py
rsync -a V3/evidence/probe_paired/ "$BACKUP_ROOT/evidence/probe_paired/"
rsync -a V3/evidence/probe_paired_summary.json "$BACKUP_ROOT/evidence/"
rsync -a V3/evidence/probe_paired_summary.md "$BACKUP_ROOT/evidence/"
date -Iseconds > "$BACKUP_ROOT/evidence/probe_pairwise_complete.txt"
echo "[DONE] probe paired bootstrap comparisons completed"
