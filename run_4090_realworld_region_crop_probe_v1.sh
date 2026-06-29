#!/usr/bin/env bash
set -euo pipefail

cd "${REMOTE_ROOT:-/root/autodl-tmp/data/platform_migration_bundle_20260531}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONUNBUFFERED=1

OUT_ROOT="${OUT_ROOT:-/root/autodl-fs/outputs_v2/realworld_region_crop_probe_v1}"
LABELS="${LABELS:-/root/autodl-fs/outputs_v2/v2_1_original_compare/eval/realworld20_highpix_notta_probe/labels.jsonl}"
PANEL_ROOT="${PANEL_ROOT:-/root/autodl-fs/outputs_v2/realworld_region_crop_probe_v1/panel}"
MODEL_DIR="${MODEL_DIR:-V2-1/outputs/export}"
VARIANTS="${VARIANTS:-exam_q1_panel,exam_q1_trim,exam_q1_trim_gray}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
MIN_PIXELS="${MIN_PIXELS:-50176}"
SOFT_COUNT_BONUS="${SOFT_COUNT_BONUS:-0.0}"
SOFT_HEAVY_PENALTY="${SOFT_HEAVY_PENALTY:-0.006}"

mkdir -p "$OUT_ROOT"

printf '\n=== realworld region crop probe start %s ===\n' "$(date '+%F %T %Z')" | tee "$OUT_ROOT/run.log"
df -h /root/autodl-tmp /root/autodl-fs | tee -a "$OUT_ROOT/run.log"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$OUT_ROOT/run.log" || true

python V2-1/scripts/build_realworld_region_crop_probe.py \
  --project-root . \
  --labels-jsonl "$LABELS" \
  --output-root "$PANEL_ROOT" \
  --variants "$VARIANTS" | tee "$OUT_ROOT/manifest_build.log"

IFS=',' read -ra VARIANT_ARRAY <<< "$VARIANTS"
for variant in "${VARIANT_ARRAY[@]}"; do
  variant="$(echo "$variant" | xargs)"
  [ -n "$variant" ] || continue

  LABEL_FILE="$PANEL_ROOT/$variant/annotations/labels.jsonl"
  OUT="$OUT_ROOT/$variant"
  mkdir -p "$OUT"
  printf '\n=== variant %s start %s ===\n' "$variant" "$(date '+%F %T %Z')" | tee -a "$OUT_ROOT/run.log"

  python V2-1/scripts/infer_ocsr_transformers.py \
    --model-dir "$MODEL_DIR" \
    --benchmark-jsonl "$LABEL_FILE" \
    --project-root . \
    --output-jsonl "$OUT/pred.jsonl" \
    --prompt-list-file V2-1/configs/prompt_rank.txt \
    --num-beams 4 \
    --num-return-sequences 4 \
    --repetition-penalty 1.05 \
    --no-repeat-ngram-size 8 \
    --tta-preset none \
    --save-candidates \
    --device cuda \
    --torch-dtype bfloat16 \
    --min-pixels "$MIN_PIXELS" \
    --max-pixels "$MAX_PIXELS" 2>&1 | tee "$OUT/infer.log"

  python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
    --benchmark-jsonl "$LABEL_FILE" \
    --prediction-jsonl "$OUT/pred.jsonl" \
    --report-json "$OUT/report.json" \
    --details-jsonl "$OUT/details.jsonl" 2>&1 | tee "$OUT/eval.log"

  python V2-1/scripts/rerank_ocsr_candidates.py \
    --prediction-jsonl "$OUT/pred.jsonl" \
    --output-jsonl "$OUT/rerank_chem_light_pred.jsonl" \
    --labels-jsonl "$LABEL_FILE" \
    --report-json "$OUT/rerank_chem_light_report.json" \
    --preference-jsonl "$OUT/preference_pairs.jsonl" 2>&1 | tee "$OUT/rerank.log"

  python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
    --benchmark-jsonl "$LABEL_FILE" \
    --prediction-jsonl "$OUT/rerank_chem_light_pred.jsonl" \
    --report-json "$OUT/rerank_chem_light_eval_report.json" \
    --details-jsonl "$OUT/rerank_chem_light_details.jsonl" 2>&1 | tee "$OUT/rerank_eval.log"

  python V2-1/scripts/rerank_ocsr_candidates.py \
    --prediction-jsonl "$OUT/pred.jsonl" \
    --output-jsonl "$OUT/rerank_realworld_soft_pred.jsonl" \
    --labels-jsonl "$LABEL_FILE" \
    --report-json "$OUT/rerank_realworld_soft_report.json" \
    --preference-jsonl "$OUT/preference_pairs_realworld_soft.jsonl" \
    --mode realworld_soft \
    --soft-count-bonus "$SOFT_COUNT_BONUS" \
    --soft-heavy-penalty "$SOFT_HEAVY_PENALTY" 2>&1 | tee "$OUT/rerank_realworld_soft.log"

  python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
    --benchmark-jsonl "$LABEL_FILE" \
    --prediction-jsonl "$OUT/rerank_realworld_soft_pred.jsonl" \
    --report-json "$OUT/rerank_realworld_soft_eval_report.json" \
    --details-jsonl "$OUT/rerank_realworld_soft_details.jsonl" 2>&1 | tee "$OUT/rerank_realworld_soft_eval.log"

  printf '\n=== variant %s chem_light report ===\n' "$variant" | tee -a "$OUT_ROOT/run.log"
  cat "$OUT/rerank_chem_light_report.json" | tee -a "$OUT_ROOT/run.log"
  printf '\n=== variant %s realworld_soft report ===\n' "$variant" | tee -a "$OUT_ROOT/run.log"
  cat "$OUT/rerank_realworld_soft_report.json" | tee -a "$OUT_ROOT/run.log"
done

python - <<'PY' | tee "$OUT_ROOT/summary.json"
import json
import os
from pathlib import Path

out_root = Path(os.environ.get("OUT_ROOT", "/root/autodl-fs/outputs_v2/realworld_region_crop_probe_v1"))
rows = []
for variant_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
    variant = variant_dir.name
    report_path = variant_dir / "report.json"
    chem_eval_path = variant_dir / "rerank_chem_light_eval_report.json"
    chem_diag_path = variant_dir / "rerank_chem_light_report.json"
    soft_eval_path = variant_dir / "rerank_realworld_soft_eval_report.json"
    soft_diag_path = variant_dir / "rerank_realworld_soft_report.json"
    if not report_path.exists():
        continue
    selected = json.loads(report_path.read_text(encoding="utf-8"))
    chem_eval = json.loads(chem_eval_path.read_text(encoding="utf-8")) if chem_eval_path.exists() else {}
    chem_diag = json.loads(chem_diag_path.read_text(encoding="utf-8")) if chem_diag_path.exists() else {}
    soft_eval = json.loads(soft_eval_path.read_text(encoding="utf-8")) if soft_eval_path.exists() else {}
    soft_diag = json.loads(soft_diag_path.read_text(encoding="utf-8")) if soft_diag_path.exists() else {}

    def source_metric(payload, key):
        try:
            return payload["by_group"]["source"]["real_world"][key]
        except Exception:
            return None

    rows.append(
        {
            "variant": variant,
            "selected_canonical_exact": source_metric(selected, "canonical_exact_match_accuracy"),
            "selected_raw_exact": source_metric(selected, "raw_exact_match_accuracy"),
            "selected_valid_smiles": source_metric(selected, "valid_smiles_rate"),
            "selected_tanimoto": source_metric(selected, "mean_fingerprint_tanimoto"),
            "chem_light_canonical_exact": source_metric(chem_eval, "canonical_exact_match_accuracy"),
            "chem_light_raw_exact": source_metric(chem_eval, "raw_exact_match_accuracy"),
            "chem_light_valid_smiles": source_metric(chem_eval, "valid_smiles_rate"),
            "chem_light_tanimoto": source_metric(chem_eval, "mean_fingerprint_tanimoto"),
            "chem_light_oracle_exact": chem_diag.get("oracle_exact"),
            "chem_light_changed_predictions": chem_diag.get("changed_predictions"),
            "chem_light_good_changes": chem_diag.get("good_changes"),
            "chem_light_bad_changes": chem_diag.get("bad_changes"),
            "chem_light_preference_pair_count": chem_diag.get("preference_pair_count"),
            "realworld_soft_canonical_exact": source_metric(soft_eval, "canonical_exact_match_accuracy"),
            "realworld_soft_raw_exact": source_metric(soft_eval, "raw_exact_match_accuracy"),
            "realworld_soft_valid_smiles": source_metric(soft_eval, "valid_smiles_rate"),
            "realworld_soft_tanimoto": source_metric(soft_eval, "mean_fingerprint_tanimoto"),
            "realworld_soft_oracle_exact": soft_diag.get("oracle_exact"),
            "realworld_soft_changed_predictions": soft_diag.get("changed_predictions"),
            "realworld_soft_good_changes": soft_diag.get("good_changes"),
            "realworld_soft_bad_changes": soft_diag.get("bad_changes"),
            "realworld_soft_preference_pair_count": soft_diag.get("preference_pair_count"),
        }
    )
print(json.dumps({"variants": rows}, ensure_ascii=False, indent=2))
PY

printf '\n=== realworld region crop probe end %s ===\n' "$(date '+%F %T %Z')" | tee -a "$OUT_ROOT/run.log"
df -h /root/autodl-tmp /root/autodl-fs | tee -a "$OUT_ROOT/run.log"
