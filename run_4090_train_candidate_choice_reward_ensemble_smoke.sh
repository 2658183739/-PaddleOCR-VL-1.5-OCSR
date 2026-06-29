#!/usr/bin/env bash
set -euo pipefail

cd "${REMOTE_ROOT:-/root/autodl-tmp/data/platform_migration_bundle_20260531}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-base}"

export PYTHONUNBUFFERED=1

BASE_OUT_ROOT="${BASE_OUT_ROOT:-/root/autodl-fs/outputs_v2/main_eval_with_candidates_20260627_fast_notta}"
REGION_OUT_ROOT="${REGION_OUT_ROOT:-/root/autodl-fs/outputs_v2/region_panel_770_fast_notta}"
CROP_OUT_ROOT="${CROP_OUT_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-fs/outputs_v2/candidate_choice_reward_ensemble_smoke_20260628}"
OUTPUTS_ROOT="$(dirname "$BASE_OUT_ROOT")"

BENCHMARK_LABELS="${BENCHMARK_LABELS:-$BASE_OUT_ROOT/combined/labels.jsonl}"
BASE_CANDIDATE_PRED="${BASE_CANDIDATE_PRED:-$BASE_OUT_ROOT/combined/pred_selected.jsonl}"
MAIN_REWARD_PRED="${MAIN_REWARD_PRED:-}"
VARIANTS="${VARIANTS:-layout_primary,layout_target_tight,layout_auto_structure,layout_textless_right,layout_structure_core_tight,layout_panel_a_core,layout_mid_structure_only}"
MERGED_POOL="${MERGED_POOL:-$OUT_ROOT/candidate_pool.jsonl}"
MERGE_REFERENCE_RUN="${MERGE_REFERENCE_RUN:-base}"
TRAIN_POOL="${TRAIN_POOL:-}"

SEEDS="${SEEDS:-20260631,20260632,20260633}"
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
SAMPLE_WEIGHT_RULES="${SAMPLE_WEIGHT_RULES:-eval_panel:weak_domain_v2=1.35,difficulty:document_embed=2.0,difficulty:journal_fig=2.0,difficulty:multi_grid=2.0,source:real_world_photo_scan=1.15}"
ROUTE_GROUP_FIELDS="${ROUTE_GROUP_FIELDS:-source+difficulty+task_type,source+difficulty,difficulty+task_type,source,difficulty,task_type,eval_panel}"

mkdir -p "$OUT_ROOT"

if [[ -z "$MAIN_REWARD_PRED" ]]; then
  if [[ -s "$OUTPUTS_ROOT/reward_head_margin0_sweep_20260627/pred_reward_head_m0p00.jsonl" ]]; then
    MAIN_REWARD_PRED="$OUTPUTS_ROOT/reward_head_margin0_sweep_20260627/pred_reward_head_m0p00.jsonl"
  elif [[ -s "$BASE_OUT_ROOT/reward_head_margin0_sweep_20260627/pred_reward_head_m0p00.jsonl" ]]; then
    MAIN_REWARD_PRED="$BASE_OUT_ROOT/reward_head_margin0_sweep_20260627/pred_reward_head_m0p00.jsonl"
  fi
fi

printf '\n=== candidate-choice ensemble smoke start %s ===\n' "$(date '+%F %T %Z')" | tee "$OUT_ROOT/run.log"
echo "REMOTE_ROOT=$(pwd)" | tee -a "$OUT_ROOT/run.log"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$OUT_ROOT/run.log"
echo "BENCHMARK_LABELS=$BENCHMARK_LABELS" | tee -a "$OUT_ROOT/run.log"
echo "BASE_CANDIDATE_PRED=$BASE_CANDIDATE_PRED" | tee -a "$OUT_ROOT/run.log"
echo "CROP_OUT_ROOT=$CROP_OUT_ROOT" | tee -a "$OUT_ROOT/run.log"
echo "TRAIN_POOL=$TRAIN_POOL" | tee -a "$OUT_ROOT/run.log"
echo "MAIN_REWARD_PRED=$MAIN_REWARD_PRED" | tee -a "$OUT_ROOT/run.log"
echo "SEEDS=$SEEDS EPOCHS=$EPOCHS" | tee -a "$OUT_ROOT/run.log"

if [[ -z "$TRAIN_POOL" ]]; then
  if [[ -n "$CROP_OUT_ROOT" && -d "$CROP_OUT_ROOT" ]]; then
    IFS=',' read -r -a VARIANT_ARRAY <<< "$VARIANTS"
    MERGE_ARGS=(
      --benchmark-jsonl "$BENCHMARK_LABELS"
      --output-jsonl "$MERGED_POOL"
      --reference-run "$MERGE_REFERENCE_RUN"
      --run "base=$BASE_CANDIDATE_PRED"
    )
    CROP_RUN_COUNT=0
    for variant in "${VARIANT_ARRAY[@]}"; do
      variant="$(echo "$variant" | xargs)"
      [[ -z "$variant" ]] && continue
      pred="$CROP_OUT_ROOT/$variant/pred.jsonl"
      if [[ -s "$pred" ]]; then
        MERGE_ARGS+=(--run "$variant=$pred")
        CROP_RUN_COUNT=$((CROP_RUN_COUNT + 1))
      else
        echo "skip missing crop prediction for $variant: $pred" | tee -a "$OUT_ROOT/run.log"
      fi
    done
    if [[ "$CROP_RUN_COUNT" -gt 0 ]]; then
      python V2-1/scripts/merge_prediction_candidate_runs.py "${MERGE_ARGS[@]}" 2>&1 | tee "$OUT_ROOT/merge_candidate_pool.log"
      TRAIN_POOL="$MERGED_POOL"
    fi
  fi
  TRAIN_POOL="${TRAIN_POOL:-$BASE_CANDIDATE_PRED}"
fi
echo "resolved TRAIN_POOL=$TRAIN_POOL" | tee -a "$OUT_ROOT/run.log"

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
CHECKPOINTS=()
for seed in "${SEED_ARRAY[@]}"; do
  seed="$(echo "$seed" | xargs)"
  [[ -z "$seed" ]] && continue
  TRAIN_DIR="$OUT_ROOT/train_seed${seed}_e${EPOCHS}"
  if [[ ! -s "$TRAIN_DIR/reward_head.pt" ]]; then
    python V2-1/scripts/train_candidate_choice_reward_head.py \
      --prediction-jsonl "$TRAIN_POOL" \
      --labels-jsonl "$BENCHMARK_LABELS" \
      --output-dir "$TRAIN_DIR" \
      --train-fraction 0.75 \
      --seed "$seed" \
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
      --sample-weight-rules "$SAMPLE_WEIGHT_RULES" 2>&1 | tee "$OUT_ROOT/train_seed${seed}.log"
  else
    echo "reuse checkpoint $TRAIN_DIR/reward_head.pt" | tee -a "$OUT_ROOT/run.log"
  fi
  CHECKPOINTS+=("$TRAIN_DIR/reward_head.pt")

  if [[ ! -s "$TRAIN_DIR/report_choice_reward_head_1344.json" ]]; then
    python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
      --benchmark-jsonl "$BENCHMARK_LABELS" \
      --prediction-jsonl "$TRAIN_DIR/pred_choice_reward_head.jsonl" \
      --report-json "$TRAIN_DIR/report_choice_reward_head_1344.json" \
      --details-jsonl "$TRAIN_DIR/details_choice_reward_head_1344.jsonl" \
      --group-fields source,difficulty,task_type,eval_panel 2>&1 | tee "$TRAIN_DIR/eval_choice_reward_1344.log"
  fi
done

CHECKPOINT_LIST="$(IFS=','; echo "${CHECKPOINTS[*]}")"
python V2-1/scripts/apply_candidate_reward_head_ensemble.py \
  --checkpoints "$CHECKPOINT_LIST" \
  --prediction-jsonl "$TRAIN_POOL" \
  --output-jsonl "$OUT_ROOT/pred_choice_reward_head_ensemble.jsonl" \
  --margin-mode mean 2>&1 | tee "$OUT_ROOT/apply_ensemble.log"

python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
  --benchmark-jsonl "$BENCHMARK_LABELS" \
  --prediction-jsonl "$OUT_ROOT/pred_choice_reward_head_ensemble.jsonl" \
  --report-json "$OUT_ROOT/report_choice_reward_head_ensemble_1344.json" \
  --details-jsonl "$OUT_ROOT/details_choice_reward_head_ensemble_1344.jsonl" \
  --group-fields source,difficulty,task_type,eval_panel 2>&1 | tee "$OUT_ROOT/eval_ensemble_1344.log"

if [[ -s "$MAIN_REWARD_PRED" ]]; then
  python V2-1/scripts/route_prediction_runs_by_group.py \
    --labels-jsonl "$BENCHMARK_LABELS" \
    --run "main=$MAIN_REWARD_PRED" \
    --run "ensemble=$OUT_ROOT/pred_choice_reward_head_ensemble.jsonl" \
    --output-dir "$OUT_ROOT/route_vs_main" \
    --group-fields "$ROUTE_GROUP_FIELDS" 2>&1 | tee "$OUT_ROOT/route_vs_main.log"

  python V2-1/scripts/crossval_route_prediction_runs_by_group.py \
    --labels-jsonl "$BENCHMARK_LABELS" \
    --run "main=$MAIN_REWARD_PRED" \
    --run "ensemble=$OUT_ROOT/pred_choice_reward_head_ensemble.jsonl" \
    --fallback-run main \
    --output-dir "$OUT_ROOT/crossval_route_vs_main" \
    --group-fields "$ROUTE_GROUP_FIELDS" \
    --write-full-label-route 2>&1 | tee "$OUT_ROOT/crossval_route_vs_main.log"
else
  echo "skip route_vs_main; missing MAIN_REWARD_PRED=$MAIN_REWARD_PRED" | tee -a "$OUT_ROOT/run.log"
fi

python - <<'PY' "$OUT_ROOT" "$TRAIN_POOL" "$MAIN_REWARD_PRED" "$CHECKPOINT_LIST"
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
train_pool = sys.argv[2]
main_reward_pred = sys.argv[3]
checkpoint_list = [item for item in sys.argv[4].split(",") if item]

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
    "train_pool": train_pool,
    "main_reward_pred": main_reward_pred,
    "checkpoints": checkpoint_list,
    "eval_1344": metric(out_root / "report_choice_reward_head_ensemble_1344.json"),
    "route_vs_main": load(out_root / "route_vs_main" / "route_summary.json"),
    "crossval_route_vs_main": load(out_root / "crossval_route_vs_main" / "crossval_route_summary.json"),
}
(out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

printf '\n=== candidate-choice ensemble smoke end %s ===\n' "$(date '+%F %T %Z')" | tee -a "$OUT_ROOT/run.log"
