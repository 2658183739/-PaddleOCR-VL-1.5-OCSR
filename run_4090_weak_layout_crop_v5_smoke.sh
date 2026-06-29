#!/usr/bin/env bash
set -euo pipefail

cd "${REMOTE_ROOT:-/root/autodl-tmp/data/platform_migration_bundle_20260531}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-base}"

export PYTHONUNBUFFERED=1

MODEL_DIR="${MODEL_DIR:-V2-1/outputs/export}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-fs/outputs_v2/weak_layout_crop_v5_20260628}"
V4_OUT_ROOT="${V4_OUT_ROOT:-/root/autodl-fs/outputs_v2/weak_layout_crop_v4_20260627}"
BASE_OUT_ROOT="${BASE_OUT_ROOT:-/root/autodl-fs/outputs_v2/main_eval_with_candidates_20260627_fast_notta}"
REGION_OUT_ROOT="${REGION_OUT_ROOT:-/root/autodl-fs/outputs_v2/region_panel_770_fast_notta}"
BENCHMARK_LABELS="${BENCHMARK_LABELS:-$BASE_OUT_ROOT/combined/labels.jsonl}"
BASE_CANDIDATE_PRED="${BASE_CANDIDATE_PRED:-$BASE_OUT_ROOT/combined/pred_selected.jsonl}"
MAIN_REWARD_PRED="${MAIN_REWARD_PRED:-/root/autodl-fs/outputs_v2/reward_head_margin0_sweep_20260627/pred_reward_head_m0p00.jsonl}"
REGION_LABELS="${REGION_LABELS:-$REGION_OUT_ROOT/labels.jsonl}"
REGION_CANDIDATE_PRED="${REGION_CANDIDATE_PRED:-$REGION_OUT_ROOT/pred_selected.jsonl}"
V4_POOL="${V4_POOL:-$V4_OUT_ROOT/fusion/cross_run_candidate_pool.jsonl}"
VARIANTS="${VARIANTS:-layout_upper_structure_only}"
PROMPT_LIST_FILE="${PROMPT_LIST_FILE:-V2-1/configs/prompt_weak_layout_rank.txt}"
MIN_PIXELS="${MIN_PIXELS:-50176}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
TTA_PRESET="${TTA_PRESET:-none}"
NUM_BEAMS="${NUM_BEAMS:-4}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-4}"

TRAIN_FRACTION="${TRAIN_FRACTION:-0.75}"
SEED="${SEED:-20260632}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.0008}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.05}"
HARD_NEGATIVE_LOSS_WEIGHT="${HARD_NEGATIVE_LOSS_WEIGHT:-0.15}"
CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-policy}"
SELECTION_EVERY="${SELECTION_EVERY:-1}"
FALLBACK_MODES="${FALLBACK_MODES:-chem_light,selected,none}"
MARGIN_GRID="${MARGIN_GRID:-0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.35,0.5}"
SAMPLE_WEIGHT_RULES="${SAMPLE_WEIGHT_RULES:-eval_panel:weak_domain_v2=1.35,difficulty:document_embed=2.2,difficulty:journal_fig=2.2,difficulty:multi_grid=2.2,source:real_world_photo_scan=1.15}"
ROUTE_GROUP_FIELDS="${ROUTE_GROUP_FIELDS:-source+difficulty+task_type,source+difficulty,difficulty+task_type,source,difficulty,task_type}"

mkdir -p "$OUT_ROOT"

printf '\n=== weak layout crop v5 smoke start %s ===\n' "$(date '+%F %T %Z')" | tee "$OUT_ROOT/run.log"
echo "REMOTE_ROOT=$(pwd)" | tee -a "$OUT_ROOT/run.log"
echo "MODEL_DIR=$MODEL_DIR" | tee -a "$OUT_ROOT/run.log"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$OUT_ROOT/run.log"
echo "V4_OUT_ROOT=$V4_OUT_ROOT" | tee -a "$OUT_ROOT/run.log"
echo "V4_POOL=$V4_POOL" | tee -a "$OUT_ROOT/run.log"
echo "BENCHMARK_LABELS=$BENCHMARK_LABELS" | tee -a "$OUT_ROOT/run.log"
echo "BASE_CANDIDATE_PRED=$BASE_CANDIDATE_PRED" | tee -a "$OUT_ROOT/run.log"
echo "MAIN_REWARD_PRED=$MAIN_REWARD_PRED" | tee -a "$OUT_ROOT/run.log"
echo "VARIANTS=$VARIANTS" | tee -a "$OUT_ROOT/run.log"
echo "SAMPLE_WEIGHT_RULES=$SAMPLE_WEIGHT_RULES" | tee -a "$OUT_ROOT/run.log"
df -h /root/autodl-tmp /root/autodl-fs | tee -a "$OUT_ROOT/run.log"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$OUT_ROOT/run.log" || true

for required in "$MODEL_DIR" "$BENCHMARK_LABELS" "$BASE_CANDIDATE_PRED" "$V4_POOL" "$PROMPT_LIST_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" | tee -a "$OUT_ROOT/run.log"
    exit 2
  fi
done

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
  --run "v4_pool=$V4_POOL"
)
MERGE_ARGS=(
  --benchmark-jsonl "$BENCHMARK_LABELS"
  --output-jsonl "$OUT_ROOT/fusion/cross_run_candidate_pool_v5.jsonl"
  --reference-run v4_pool
  --run "v4_pool=$V4_POOL"
)

for variant in "${VARIANT_ARRAY[@]}"; do
  variant="$(echo "$variant" | xargs)"
  pred="$OUT_ROOT/$variant/pred.jsonl"
  if [[ -s "$pred" ]]; then
    FUSION_ARGS+=(--run "$variant=$pred")
    MERGE_ARGS+=(--run "$variant=$pred")
  else
    echo "skip fusion/merge run for $variant: missing $pred" | tee -a "$OUT_ROOT/run.log"
  fi
done

python V2-1/scripts/analyze_candidate_fusion.py "${FUSION_ARGS[@]}" 2>&1 | tee "$OUT_ROOT/fusion.log"
python V2-1/scripts/merge_prediction_candidate_runs.py "${MERGE_ARGS[@]}" 2>&1 | tee "$OUT_ROOT/merge_candidate_pool.log"

TRAIN_POOL="$OUT_ROOT/fusion/cross_run_candidate_pool_v5.jsonl"
TRAIN_DIR="$OUT_ROOT/candidate_choice_reward_v5/train_seed${SEED}_e${EPOCHS}"
mkdir -p "$(dirname "$TRAIN_DIR")"

python V2-1/scripts/train_candidate_choice_reward_head.py \
  --prediction-jsonl "$TRAIN_POOL" \
  --labels-jsonl "$BENCHMARK_LABELS" \
  --output-dir "$TRAIN_DIR" \
  --train-fraction "$TRAIN_FRACTION" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --hidden-dim "$HIDDEN_DIM" \
  --dropout "$DROPOUT" \
  --hard-negative-loss-weight "$HARD_NEGATIVE_LOSS_WEIGHT" \
  --checkpoint-selection "$CHECKPOINT_SELECTION" \
  --selection-every "$SELECTION_EVERY" \
  --fallback-modes "$FALLBACK_MODES" \
  --margin-grid "$MARGIN_GRID" \
  --sample-weight-rules "$SAMPLE_WEIGHT_RULES" 2>&1 | tee "$OUT_ROOT/train_candidate_choice.log"

python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
  --benchmark-jsonl "$BENCHMARK_LABELS" \
  --prediction-jsonl "$TRAIN_DIR/pred_choice_reward_head.jsonl" \
  --report-json "$TRAIN_DIR/report_choice_reward_head_1344.json" \
  --details-jsonl "$TRAIN_DIR/details_choice_reward_head_1344.jsonl" \
  --group-fields source,difficulty,task_type,eval_panel 2>&1 | tee "$TRAIN_DIR/eval_choice_reward_1344.log"

if [[ -s "$REGION_LABELS" && -s "$REGION_CANDIDATE_PRED" ]]; then
  python V2-1/scripts/apply_candidate_reward_head.py \
    --checkpoint "$TRAIN_DIR/reward_head.pt" \
    --prediction-jsonl "$REGION_CANDIDATE_PRED" \
    --output-jsonl "$TRAIN_DIR/pred_choice_reward_head_region770.jsonl" \
    --labels-jsonl "$REGION_LABELS" 2>&1 | tee "$TRAIN_DIR/apply_choice_reward_region770.log"

  python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
    --benchmark-jsonl "$REGION_LABELS" \
    --prediction-jsonl "$TRAIN_DIR/pred_choice_reward_head_region770.jsonl" \
    --report-json "$TRAIN_DIR/report_choice_reward_head_region770.json" \
    --details-jsonl "$TRAIN_DIR/details_choice_reward_head_region770.jsonl" \
    --group-fields source,difficulty,task_type 2>&1 | tee "$TRAIN_DIR/eval_choice_reward_region770.log"
fi

if [[ -s "$MAIN_REWARD_PRED" ]]; then
  python V2-1/scripts/route_prediction_runs_by_group.py \
    --labels-jsonl "$BENCHMARK_LABELS" \
    --run "main=$MAIN_REWARD_PRED" \
    --run "choice=$TRAIN_DIR/pred_choice_reward_head.jsonl" \
    --output-dir "$TRAIN_DIR/route_vs_main" \
    --group-fields "$ROUTE_GROUP_FIELDS" 2>&1 | tee "$TRAIN_DIR/route_vs_main.log"

  python V2-1/scripts/crossval_route_prediction_runs_by_group.py \
    --labels-jsonl "$BENCHMARK_LABELS" \
    --run "main=$MAIN_REWARD_PRED" \
    --run "choice=$TRAIN_DIR/pred_choice_reward_head.jsonl" \
    --fallback-run main \
    --output-dir "$TRAIN_DIR/crossval_route_vs_main" \
    --group-fields "$ROUTE_GROUP_FIELDS" \
    --write-full-label-route 2>&1 | tee "$TRAIN_DIR/crossval_route_vs_main.log"
fi

python - <<'PY' "$OUT_ROOT" "$TRAIN_DIR" "$TRAIN_POOL" "$MAIN_REWARD_PRED"
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
train_dir = Path(sys.argv[2])
train_pool = sys.argv[3]
main_reward_pred = sys.argv[4]

def load(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def metric(path):
    data = load(path)
    if not data:
        return None
    acc = data.get("accuracy", {})
    sim = data.get("similarity", {})
    return {
        "total": data.get("total"),
        "canonical_exact": acc.get("canonical_exact_match_accuracy"),
        "raw_exact": acc.get("raw_exact_match_accuracy"),
        "valid_smiles_rate": acc.get("valid_smiles_rate"),
        "mean_tanimoto": sim.get("mean_fingerprint_tanimoto"),
        "by_eval_panel_exact": {
            key: value.get("canonical_exact_match_accuracy")
            for key, value in data.get("by_group", {}).get("eval_panel", {}).items()
        },
        "by_difficulty_exact": {
            key: value.get("canonical_exact_match_accuracy")
            for key, value in data.get("by_group", {}).get("difficulty", {}).items()
        },
    }

summary = {
    "out_root": str(out_root),
    "train_dir": str(train_dir),
    "train_pool": train_pool,
    "main_reward_pred": main_reward_pred,
    "fusion_summary": load(out_root / "fusion" / "candidate_fusion_summary.json"),
    "train_report": load(train_dir / "choice_reward_head_report.json"),
    "eval_1344": metric(train_dir / "report_choice_reward_head_1344.json"),
    "eval_region770": metric(train_dir / "report_choice_reward_head_region770.json"),
    "route_vs_main": load(train_dir / "route_vs_main" / "route_summary.json"),
    "crossval_route_vs_main": load(train_dir / "crossval_route_vs_main" / "crossval_route_summary.json"),
}
(out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

printf '\n=== weak layout crop v5 smoke end %s ===\n' "$(date '+%F %T %Z')" | tee -a "$OUT_ROOT/run.log"
df -h /root/autodl-tmp /root/autodl-fs | tee -a "$OUT_ROOT/run.log"
