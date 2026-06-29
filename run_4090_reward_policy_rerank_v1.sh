#!/usr/bin/env bash
set -euo pipefail

cd "${REMOTE_ROOT:-/root/autodl-tmp/data/platform_migration_bundle_20260531}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONUNBUFFERED=1

LABELS="${LABELS:-V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl}"
PRED_JSONL="${PRED_JSONL:-/root/autodl-fs/outputs_v2/full_eval_region_panel_v1_fast_notta/ocsr_realworld_mixed_eval_v1p1/merged/pred.jsonl}"
OUT_DIR="${OUT_DIR:-/root/autodl-fs/outputs_v2/reward_policy_rerank_v1}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.75}"
POLICY_MARGIN="${POLICY_MARGIN:-1.5}"
EPOCHS="${EPOCHS:-500}"
SEED="${SEED:-20260626}"

mkdir -p "$OUT_DIR"
printf "\n=== reward policy rerank start %s ===\n" "$(date '+%F %T %Z')" | tee "$OUT_DIR/run.log"
echo "LABELS=$LABELS" | tee -a "$OUT_DIR/run.log"
echo "PRED_JSONL=$PRED_JSONL" | tee -a "$OUT_DIR/run.log"
echo "OUT_DIR=$OUT_DIR" | tee -a "$OUT_DIR/run.log"
echo "TRAIN_FRACTION=$TRAIN_FRACTION POLICY_MARGIN=$POLICY_MARGIN EPOCHS=$EPOCHS SEED=$SEED" | tee -a "$OUT_DIR/run.log"

python V2-1/scripts/reward_policy_reranker.py \
  --prediction-jsonl "$PRED_JSONL" \
  --labels-jsonl "$LABELS" \
  --output-jsonl "$OUT_DIR/pred_policy_raw.jsonl" \
  --policy-json "$OUT_DIR/policy_raw.json" \
  --report-json "$OUT_DIR/report_policy_raw.json" \
  --details-jsonl "$OUT_DIR/details_policy_raw.jsonl" \
  --epochs "$EPOCHS" \
  --train-fraction "$TRAIN_FRACTION" \
  --seed "$SEED" \
  --fallback-mode none \
  --policy-margin 0 2>&1 | tee -a "$OUT_DIR/run.log"

python V2-1/scripts/reward_policy_reranker.py \
  --prediction-jsonl "$PRED_JSONL" \
  --labels-jsonl "$LABELS" \
  --load-policy-json "$OUT_DIR/policy_raw.json" \
  --output-jsonl "$OUT_DIR/pred_hybrid.jsonl" \
  --report-json "$OUT_DIR/report_hybrid.json" \
  --details-jsonl "$OUT_DIR/details_hybrid.jsonl" \
  --fallback-mode chem_light \
  --policy-margin "$POLICY_MARGIN" 2>&1 | tee -a "$OUT_DIR/run.log"

python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
  --benchmark-jsonl "$LABELS" \
  --prediction-jsonl "$OUT_DIR/pred_hybrid.jsonl" \
  --report-json "$OUT_DIR/eval_hybrid.json" \
  --details-jsonl "$OUT_DIR/eval_details_hybrid.jsonl" 2>&1 | tee -a "$OUT_DIR/run.log"

python - <<'PY' "$OUT_DIR"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
report = json.loads((out / "report_hybrid.json").read_text(encoding="utf-8"))
eval_report = json.loads((out / "eval_hybrid.json").read_text(encoding="utf-8"))
summary = {
    "policy_margin": report.get("policy_margin"),
    "internal_metrics": report.get("all", {}).get("metrics", {}),
    "dev_oracle_subset": report.get("dev_oracle_subset", {}).get("metrics", {}),
    "eval_accuracy": eval_report.get("accuracy", {}),
    "eval_similarity": eval_report.get("similarity", {}),
    "by_source_canonical_exact": {
        key: value.get("canonical_exact_match_accuracy")
        for key, value in eval_report.get("by_group", {}).get("source", {}).items()
    },
}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

printf "\n=== reward policy rerank end %s ===\n" "$(date '+%F %T %Z')" | tee -a "$OUT_DIR/run.log"
