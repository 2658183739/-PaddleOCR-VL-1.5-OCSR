#!/usr/bin/env bash
set -euo pipefail

cd "${REMOTE_ROOT:-/root/autodl-tmp/data/platform_migration_bundle_20260531}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-base}"

export PYTHONUNBUFFERED=1

MODEL_DIR="${MODEL_DIR:-V2-1/outputs/export}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-fs/outputs_v2/weak_layout_crop_20260627}"
BASE_OUT_ROOT="${BASE_OUT_ROOT:-/root/autodl-fs/outputs_v2/main_eval_with_candidates_20260627_fast_notta}"
BENCHMARK_LABELS="${BENCHMARK_LABELS:-$BASE_OUT_ROOT/combined/labels.jsonl}"
BASE_CANDIDATE_PRED="${BASE_CANDIDATE_PRED:-$BASE_OUT_ROOT/combined/pred_selected.jsonl}"
PAIR_REWARD_PRED="${PAIR_REWARD_PRED:-}"
PAIR_REWARD_HEAD_CKPT="${PAIR_REWARD_HEAD_CKPT:-V2-1/reports/pair_reward_head_smoke_20260627/train770_pair_seed20260627_e80/reward_head.pt}"
PAIR_FALLBACK_MODE="${PAIR_FALLBACK_MODE:-chem_light}"
PAIR_POLICY_MARGIN="${PAIR_POLICY_MARGIN:-0.5}"
PAIR_MARGIN_GRID="${PAIR_MARGIN_GRID:-0.00,0.05,0.10,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.75,1.00}"
PAIR_GROUP_MARGIN_FIELDS="${PAIR_GROUP_MARGIN_FIELDS:-source,difficulty,task_type,source+difficulty,source+difficulty+task_type}"
MERGE_REFERENCE_RUN="${MERGE_REFERENCE_RUN:-base}"
VARIANTS="${VARIANTS:-layout_primary,layout_target_tight,layout_primary_trim,layout_primary_gray,layout_auto_structure,layout_wide,layout_top_left}"
MIN_PIXELS="${MIN_PIXELS:-50176}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
TTA_PRESET="${TTA_PRESET:-none}"
NUM_BEAMS="${NUM_BEAMS:-4}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-4}"
PROMPT_LIST_FILE="${PROMPT_LIST_FILE:-V2-1/configs/prompt_weak_layout_rank.txt}"

mkdir -p "$OUT_ROOT"

printf '\n=== weak layout crop candidates start %s ===\n' "$(date '+%F %T %Z')" | tee "$OUT_ROOT/run.log"
echo "REMOTE_ROOT=$(pwd)" | tee -a "$OUT_ROOT/run.log"
echo "MODEL_DIR=$MODEL_DIR" | tee -a "$OUT_ROOT/run.log"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$OUT_ROOT/run.log"
echo "BENCHMARK_LABELS=$BENCHMARK_LABELS" | tee -a "$OUT_ROOT/run.log"
echo "BASE_CANDIDATE_PRED=$BASE_CANDIDATE_PRED" | tee -a "$OUT_ROOT/run.log"
echo "PAIR_REWARD_PRED=$PAIR_REWARD_PRED" | tee -a "$OUT_ROOT/run.log"
echo "PAIR_REWARD_HEAD_CKPT=$PAIR_REWARD_HEAD_CKPT" | tee -a "$OUT_ROOT/run.log"
echo "PAIR_FALLBACK_MODE=$PAIR_FALLBACK_MODE PAIR_POLICY_MARGIN=$PAIR_POLICY_MARGIN" | tee -a "$OUT_ROOT/run.log"
echo "PAIR_MARGIN_GRID=$PAIR_MARGIN_GRID" | tee -a "$OUT_ROOT/run.log"
echo "PAIR_GROUP_MARGIN_FIELDS=$PAIR_GROUP_MARGIN_FIELDS" | tee -a "$OUT_ROOT/run.log"
echo "MERGE_REFERENCE_RUN=$MERGE_REFERENCE_RUN" | tee -a "$OUT_ROOT/run.log"
echo "VARIANTS=$VARIANTS" | tee -a "$OUT_ROOT/run.log"
echo "TTA_PRESET=$TTA_PRESET NUM_BEAMS=$NUM_BEAMS NUM_RETURN_SEQUENCES=$NUM_RETURN_SEQUENCES" | tee -a "$OUT_ROOT/run.log"
df -h /root/autodl-tmp /root/autodl-fs | tee -a "$OUT_ROOT/run.log"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$OUT_ROOT/run.log" || true

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "missing MODEL_DIR: $MODEL_DIR" | tee -a "$OUT_ROOT/run.log"
  exit 2
fi
if [[ ! -s "$BENCHMARK_LABELS" ]]; then
  echo "missing BENCHMARK_LABELS: $BENCHMARK_LABELS" | tee -a "$OUT_ROOT/run.log"
  exit 2
fi
if [[ ! -s "$BASE_CANDIDATE_PRED" ]]; then
  echo "missing BASE_CANDIDATE_PRED: $BASE_CANDIDATE_PRED" | tee -a "$OUT_ROOT/run.log"
  exit 2
fi
if [[ ! -s "$PROMPT_LIST_FILE" ]]; then
  echo "missing PROMPT_LIST_FILE: $PROMPT_LIST_FILE" | tee -a "$OUT_ROOT/run.log"
  exit 2
fi

python V2-1/scripts/build_weak_layout_crop_panels.py \
  --project-root . \
  --labels-jsonl "$BENCHMARK_LABELS" \
  --output-root "$OUT_ROOT/crops" \
  --variants "$VARIANTS" 2>&1 | tee "$OUT_ROOT/build_crops.log"

IFS=',' read -r -a VARIANT_ARRAY <<< "$VARIANTS"

run_variant() {
  local variant="$1"
  local label_file="$OUT_ROOT/crops/$variant/annotations/labels.jsonl"
  local out_dir="$OUT_ROOT/$variant"
  mkdir -p "$out_dir/parts"

  if [[ ! -s "$label_file" ]]; then
    echo "missing crop label file for $variant: $label_file" | tee -a "$OUT_ROOT/run.log"
    return 2
  fi

  python V2-1/scripts/eval_jsonl_resume.py \
    --benchmark-jsonl "$label_file" \
    --prediction-glob "$out_dir/parts/*.jsonl" \
    --merged-jsonl "$out_dir/pred.jsonl" \
    --remaining-jsonl "$out_dir/remaining.jsonl" \
    --status-json "$out_dir/status.json" | tee "$out_dir/status_before.log"

  local remaining
  remaining="$(python - <<'PY' "$out_dir/status.json"
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["remaining"])
PY
)"

  if [[ "$remaining" -gt 0 ]]; then
    local part="$out_dir/parts/part_$(date '+%Y%m%d_%H%M%S').jsonl"
    printf '\n=== %s infer remaining=%s %s ===\n' "$variant" "$remaining" "$(date '+%F %T %Z')" | tee -a "$OUT_ROOT/run.log"
    python V2-1/scripts/infer_ocsr_transformers.py \
      --model-dir "$MODEL_DIR" \
      --benchmark-jsonl "$out_dir/remaining.jsonl" \
      --project-root . \
      --output-jsonl "$part" \
      --prompt-list-file "$PROMPT_LIST_FILE" \
      --num-beams "$NUM_BEAMS" \
      --num-return-sequences "$NUM_RETURN_SEQUENCES" \
      --repetition-penalty 1.05 \
      --no-repeat-ngram-size 8 \
      --tta-preset "$TTA_PRESET" \
      --save-candidates \
      --device cuda \
      --torch-dtype bfloat16 \
      --min-pixels "$MIN_PIXELS" \
      --max-pixels "$MAX_PIXELS" 2>&1 | tee "$out_dir/infer_$(date '+%Y%m%d_%H%M%S').log"
  fi

  python V2-1/scripts/eval_jsonl_resume.py \
    --benchmark-jsonl "$label_file" \
    --prediction-glob "$out_dir/parts/*.jsonl" \
    --merged-jsonl "$out_dir/pred.jsonl" \
    --remaining-jsonl "$out_dir/remaining.jsonl" \
    --status-json "$out_dir/status.json" | tee "$out_dir/status_after.log"

  local after_remaining
  after_remaining="$(python - <<'PY' "$out_dir/status.json"
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["remaining"])
PY
)"
  if [[ "$after_remaining" -ne 0 ]]; then
    echo "variant $variant still has remaining=$after_remaining; skip scoring until complete" | tee -a "$OUT_ROOT/run.log"
    return 0
  fi

  python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
    --benchmark-jsonl "$label_file" \
    --prediction-jsonl "$out_dir/pred.jsonl" \
    --report-json "$out_dir/report_selected.json" \
    --details-jsonl "$out_dir/details_selected.jsonl" \
    --group-fields source,difficulty,task_type,eval_panel,preprocess_domain 2>&1 | tee "$out_dir/eval_selected.log"
}

for variant in "${VARIANT_ARRAY[@]}"; do
  variant="$(echo "$variant" | xargs)"
  [[ -z "$variant" ]] && continue
  run_variant "$variant"
done

FUSION_ARGS=(
  --benchmark-jsonl "$BENCHMARK_LABELS"
  --output-dir "$OUT_ROOT/fusion"
  --group-fields source,difficulty,task_type,eval_panel
  --strategies vote_score,score,penalty_score,source_weighted
)

if [[ -s "$BASE_CANDIDATE_PRED" ]]; then
  FUSION_ARGS+=(--run "base=$BASE_CANDIDATE_PRED")
else
  echo "skip base fusion run: missing $BASE_CANDIDATE_PRED" | tee -a "$OUT_ROOT/run.log"
fi

if [[ -n "$PAIR_REWARD_PRED" && -s "$PAIR_REWARD_PRED" ]]; then
  FUSION_ARGS+=(--run "pair_reward=$PAIR_REWARD_PRED")
fi

for variant in "${VARIANT_ARRAY[@]}"; do
  variant="$(echo "$variant" | xargs)"
  pred="$OUT_ROOT/$variant/pred.jsonl"
  if [[ -s "$pred" ]]; then
    FUSION_ARGS+=(--run "$variant=$pred")
  else
    echo "skip fusion run for $variant: missing $pred" | tee -a "$OUT_ROOT/run.log"
  fi
done

python V2-1/scripts/analyze_candidate_fusion.py "${FUSION_ARGS[@]}" 2>&1 | tee "$OUT_ROOT/fusion.log"

MERGED_POOL="$OUT_ROOT/fusion/cross_run_candidate_pool.jsonl"
MERGE_ARGS=(
  --benchmark-jsonl "$BENCHMARK_LABELS"
  --output-jsonl "$MERGED_POOL"
  --reference-run "$MERGE_REFERENCE_RUN"
)

if [[ -s "$BASE_CANDIDATE_PRED" ]]; then
  MERGE_ARGS+=(--run "base=$BASE_CANDIDATE_PRED")
fi

if [[ -n "$PAIR_REWARD_PRED" && -s "$PAIR_REWARD_PRED" ]]; then
  MERGE_ARGS+=(--run "pair_reward=$PAIR_REWARD_PRED")
fi

for variant in "${VARIANT_ARRAY[@]}"; do
  variant="$(echo "$variant" | xargs)"
  pred="$OUT_ROOT/$variant/pred.jsonl"
  if [[ -s "$pred" ]]; then
    MERGE_ARGS+=(--run "$variant=$pred")
  fi
done

python V2-1/scripts/merge_prediction_candidate_runs.py "${MERGE_ARGS[@]}" 2>&1 | tee "$OUT_ROOT/merge_candidate_pool.log"

if [[ -s "$PAIR_REWARD_HEAD_CKPT" && -s "$MERGED_POOL" ]]; then
  python V2-1/scripts/apply_candidate_reward_head.py \
    --checkpoint "$PAIR_REWARD_HEAD_CKPT" \
    --prediction-jsonl "$MERGED_POOL" \
    --labels-jsonl "$BENCHMARK_LABELS" \
    --output-jsonl "$OUT_ROOT/fusion/pred_cross_run_pair_reward.jsonl" \
    --fallback-mode "$PAIR_FALLBACK_MODE" \
    --policy-margin "$PAIR_POLICY_MARGIN" 2>&1 | tee "$OUT_ROOT/fusion/apply_cross_run_pair_reward.log"

  python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
    --benchmark-jsonl "$BENCHMARK_LABELS" \
    --prediction-jsonl "$OUT_ROOT/fusion/pred_cross_run_pair_reward.jsonl" \
    --report-json "$OUT_ROOT/fusion/report_cross_run_pair_reward.json" \
    --details-jsonl "$OUT_ROOT/fusion/details_cross_run_pair_reward.jsonl" \
    --group-fields source,difficulty,task_type,eval_panel 2>&1 | tee "$OUT_ROOT/fusion/eval_cross_run_pair_reward.log"

  python V2-1/scripts/sweep_candidate_reward_head_margin.py \
    --checkpoint "$PAIR_REWARD_HEAD_CKPT" \
    --prediction-jsonl "$MERGED_POOL" \
    --labels-jsonl "$BENCHMARK_LABELS" \
    --output-dir "$OUT_ROOT/fusion/pair_reward_margin_sweep" \
    --fallback-mode "$PAIR_FALLBACK_MODE" \
    --margin-grid "$PAIR_MARGIN_GRID" \
    --group-fields source,difficulty,task_type,eval_panel,source+difficulty,source+difficulty+task_type \
    --group-margin-fields "$PAIR_GROUP_MARGIN_FIELDS" 2>&1 | tee "$OUT_ROOT/fusion/pair_reward_margin_sweep.log"
else
  echo "skip cross-run pair reward: missing checkpoint or merged pool" | tee -a "$OUT_ROOT/run.log"
fi

python - <<'PY' "$OUT_ROOT"
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary = {
    "out_root": str(out_root),
    "crop_manifest": str(out_root / "crops" / "manifest.json"),
    "variant_reports": {},
    "fusion_summary": None,
}
for report in sorted(out_root.glob("layout_*/report_selected.json")):
    data = json.loads(report.read_text(encoding="utf-8"))
    acc = data.get("accuracy", {})
    sim = data.get("similarity", {})
    summary["variant_reports"][report.parent.name] = {
        "total": data.get("total"),
        "canonical_exact": acc.get("canonical_exact_match_accuracy"),
        "raw_exact": acc.get("raw_exact_match_accuracy"),
        "valid_smiles_rate": acc.get("valid_smiles_rate"),
        "mean_tanimoto": sim.get("mean_fingerprint_tanimoto"),
    }
fusion_path = out_root / "fusion" / "candidate_fusion_summary.json"
if fusion_path.exists():
    summary["fusion_summary"] = json.loads(fusion_path.read_text(encoding="utf-8"))
pair_report_path = out_root / "fusion" / "report_cross_run_pair_reward.json"
if pair_report_path.exists():
    data = json.loads(pair_report_path.read_text(encoding="utf-8"))
    acc = data.get("accuracy", {})
    sim = data.get("similarity", {})
    summary["cross_run_pair_reward"] = {
        "total": data.get("total"),
        "canonical_exact": acc.get("canonical_exact_match_accuracy"),
        "raw_exact": acc.get("raw_exact_match_accuracy"),
        "valid_smiles_rate": acc.get("valid_smiles_rate"),
        "mean_tanimoto": sim.get("mean_fingerprint_tanimoto"),
    }
sweep_path = out_root / "fusion" / "pair_reward_margin_sweep" / "margin_sweep_summary.json"
if sweep_path.exists():
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    summary["cross_run_pair_reward_margin_sweep"] = sweep
    best_entry = dict(sweep["best"])
    best_entry["strategy"] = "global_margin"
    for item in sweep.get("group_margin_results", []):
        overall = item.get("overall", {})
        candidate = {
            "strategy": "group_margin",
            "group_field": item.get("group_field"),
            "prediction_jsonl": item.get("prediction_jsonl"),
            "report_json": item.get("report_json"),
            "canonical_exact": overall.get("canonical_exact"),
            "raw_exact": overall.get("raw_exact"),
            "valid_smiles_rate": overall.get("valid_smiles_rate"),
            "mean_tanimoto": overall.get("mean_tanimoto"),
        }
        best_key = (
            best_entry.get("canonical_exact") or 0.0,
            best_entry.get("raw_exact") or 0.0,
            best_entry.get("mean_tanimoto") or 0.0,
            best_entry.get("valid_smiles_rate") or 0.0,
        )
        candidate_key = (
            candidate.get("canonical_exact") or 0.0,
            candidate.get("raw_exact") or 0.0,
            candidate.get("mean_tanimoto") or 0.0,
            candidate.get("valid_smiles_rate") or 0.0,
        )
        if candidate_key > best_key:
            best_entry = candidate
    best_report_path = Path(best_entry["report_json"])
    if best_report_path.exists():
        data = json.loads(best_report_path.read_text(encoding="utf-8"))
        acc = data.get("accuracy", {})
        sim = data.get("similarity", {})
        summary["cross_run_pair_reward_best"] = {
            "strategy": best_entry.get("strategy"),
            "group_field": best_entry.get("group_field"),
            "margin": best_entry.get("margin"),
            "prediction_jsonl": best_entry.get("prediction_jsonl"),
            "report_json": best_entry.get("report_json"),
            "total": data.get("total"),
            "canonical_exact": acc.get("canonical_exact_match_accuracy"),
            "raw_exact": acc.get("raw_exact_match_accuracy"),
            "valid_smiles_rate": acc.get("valid_smiles_rate"),
            "mean_tanimoto": sim.get("mean_fingerprint_tanimoto"),
        }
(out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

printf '\n=== weak layout crop candidates end %s ===\n' "$(date '+%F %T %Z')" | tee -a "$OUT_ROOT/run.log"
df -h /root/autodl-tmp /root/autodl-fs | tee -a "$OUT_ROOT/run.log"
