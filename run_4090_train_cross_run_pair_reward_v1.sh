#!/usr/bin/env bash
set -euo pipefail

cd "${REMOTE_ROOT:-/root/autodl-tmp/data/platform_migration_bundle_20260531}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-base}"

export PYTHONUNBUFFERED=1

BASE_OUT_ROOT="${BASE_OUT_ROOT:-/root/autodl-fs/outputs_v2/main_eval_with_candidates_20260627_fast_notta}"
CROP_OUT_ROOT="${CROP_OUT_ROOT:-/root/autodl-fs/outputs_v2/weak_layout_crop_20260627}"
OUT_ROOT="${OUT_ROOT:-$CROP_OUT_ROOT/cross_run_pair_reward_v1}"
BENCHMARK_LABELS="${BENCHMARK_LABELS:-$BASE_OUT_ROOT/combined/labels.jsonl}"
BASE_CANDIDATE_PRED="${BASE_CANDIDATE_PRED:-$BASE_OUT_ROOT/combined/pred_selected.jsonl}"
PAIR_REWARD_PRED="${PAIR_REWARD_PRED:-}"
VARIANTS="${VARIANTS:-layout_primary,layout_target_tight,layout_primary_trim,layout_primary_gray,layout_auto_structure,layout_wide,layout_top_left}"
MERGED_POOL="${MERGED_POOL:-$CROP_OUT_ROOT/fusion/cross_run_candidate_pool.jsonl}"
MERGE_REFERENCE_RUN="${MERGE_REFERENCE_RUN:-base}"
AUGMENT_STEREO_VARIANTS="${AUGMENT_STEREO_VARIANTS:-1}"
STEREO_SCORE_DELTA="${STEREO_SCORE_DELTA:--0.15}"
STEREO_PENALTY_DELTA="${STEREO_PENALTY_DELTA:-0.0}"

MAX_HARD_NEGATIVES_PER_SAMPLE="${MAX_HARD_NEGATIVES_PER_SAMPLE:-4}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.75}"
SEED="${SEED:-20260627}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
PAIR_TYPE_WEIGHTS="${PAIR_TYPE_WEIGHTS:-oracle_positive_vs_selected=2.5,oracle_positive_vs_reward_policy=2.5,oracle_positive_vs_chem_light=1.75,selected_correct_guard_vs_chem_light=3.0,selected_correct_guard_vs_reward_policy=3.0,default=1.0}"
EVAL_MARGIN_GRID="${EVAL_MARGIN_GRID:-0.00,0.05,0.10,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.75,1.00}"
EVAL_GROUP_MARGIN_FIELDS="${EVAL_GROUP_MARGIN_FIELDS:-source,difficulty,task_type,source+difficulty,source+difficulty+task_type}"

mkdir -p "$OUT_ROOT"

printf '\n=== cross-run pair reward train start %s ===\n' "$(date '+%F %T %Z')" | tee "$OUT_ROOT/run.log"
echo "REMOTE_ROOT=$(pwd)" | tee -a "$OUT_ROOT/run.log"
echo "BASE_OUT_ROOT=$BASE_OUT_ROOT" | tee -a "$OUT_ROOT/run.log"
echo "CROP_OUT_ROOT=$CROP_OUT_ROOT" | tee -a "$OUT_ROOT/run.log"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$OUT_ROOT/run.log"
echo "BENCHMARK_LABELS=$BENCHMARK_LABELS" | tee -a "$OUT_ROOT/run.log"
echo "BASE_CANDIDATE_PRED=$BASE_CANDIDATE_PRED" | tee -a "$OUT_ROOT/run.log"
echo "PAIR_REWARD_PRED=$PAIR_REWARD_PRED" | tee -a "$OUT_ROOT/run.log"
echo "VARIANTS=$VARIANTS" | tee -a "$OUT_ROOT/run.log"
echo "MERGED_POOL=$MERGED_POOL" | tee -a "$OUT_ROOT/run.log"
echo "AUGMENT_STEREO_VARIANTS=$AUGMENT_STEREO_VARIANTS" | tee -a "$OUT_ROOT/run.log"
echo "STEREO_SCORE_DELTA=$STEREO_SCORE_DELTA STEREO_PENALTY_DELTA=$STEREO_PENALTY_DELTA" | tee -a "$OUT_ROOT/run.log"
echo "PAIR_TYPE_WEIGHTS=$PAIR_TYPE_WEIGHTS" | tee -a "$OUT_ROOT/run.log"
echo "EVAL_MARGIN_GRID=$EVAL_MARGIN_GRID" | tee -a "$OUT_ROOT/run.log"
echo "EVAL_GROUP_MARGIN_FIELDS=$EVAL_GROUP_MARGIN_FIELDS" | tee -a "$OUT_ROOT/run.log"
df -h /root/autodl-tmp /root/autodl-fs | tee -a "$OUT_ROOT/run.log"

if [[ ! -s "$BENCHMARK_LABELS" ]]; then
  echo "missing BENCHMARK_LABELS: $BENCHMARK_LABELS" | tee -a "$OUT_ROOT/run.log"
  exit 2
fi
if [[ ! -s "$BASE_CANDIDATE_PRED" ]]; then
  echo "missing BASE_CANDIDATE_PRED: $BASE_CANDIDATE_PRED" | tee -a "$OUT_ROOT/run.log"
  exit 2
fi

IFS=',' read -r -a VARIANT_ARRAY <<< "$VARIANTS"

MERGE_ARGS=(
  --benchmark-jsonl "$BENCHMARK_LABELS"
  --output-jsonl "$MERGED_POOL"
  --reference-run "$MERGE_REFERENCE_RUN"
  --run "base=$BASE_CANDIDATE_PRED"
)

if [[ -n "$PAIR_REWARD_PRED" && -s "$PAIR_REWARD_PRED" ]]; then
  MERGE_ARGS+=(--run "pair_reward=$PAIR_REWARD_PRED")
fi

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

if [[ "$CROP_RUN_COUNT" -eq 0 ]]; then
  echo "no crop prediction runs found under $CROP_OUT_ROOT; run run_4090_weak_layout_crop_candidates_v1.sh first" | tee -a "$OUT_ROOT/run.log"
  exit 2
fi

python V2-1/scripts/merge_prediction_candidate_runs.py "${MERGE_ARGS[@]}" 2>&1 | tee "$OUT_ROOT/merge_candidate_pool.log"

TRAIN_POOL="$MERGED_POOL"
if [[ "$AUGMENT_STEREO_VARIANTS" == "1" ]]; then
  TRAIN_POOL="$OUT_ROOT/cross_run_candidate_pool_stereo_aug.jsonl"
  python V2-1/scripts/augment_candidate_stereo_variants.py \
    --input-jsonl "$MERGED_POOL" \
    --output-jsonl "$TRAIN_POOL" \
    --include-selected \
    --score-delta "$STEREO_SCORE_DELTA" \
    --penalty-delta "$STEREO_PENALTY_DELTA" 2>&1 | tee "$OUT_ROOT/augment_stereo_variants.log"
fi

PREFERENCE_JSONL="$OUT_ROOT/preference_pairs.jsonl"
PREFERENCE_REPORT="$OUT_ROOT/preference_pairs_report.json"

python V2-1/scripts/build_candidate_preference_dataset.py \
  --prediction-jsonl "$TRAIN_POOL" \
  --labels-jsonl "$BENCHMARK_LABELS" \
  --output-jsonl "$PREFERENCE_JSONL" \
  --report-json "$PREFERENCE_REPORT" \
  --max-hard-negatives-per-sample "$MAX_HARD_NEGATIVES_PER_SAMPLE" 2>&1 | tee "$OUT_ROOT/build_preference_pairs.log"

PAIR_COUNT="$(python - <<'PY' "$PREFERENCE_JSONL"
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    print(sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
PY
)"
echo "PAIR_COUNT=$PAIR_COUNT" | tee -a "$OUT_ROOT/run.log"
if [[ "$PAIR_COUNT" -le 0 ]]; then
  echo "no preference pairs; stop" | tee -a "$OUT_ROOT/run.log"
  exit 2
fi

TRAIN_DIR="$OUT_ROOT/train_seed${SEED}_e${EPOCHS}"
python V2-1/scripts/train_candidate_pair_reward_head.py \
  --pair-jsonl "$PREFERENCE_JSONL" \
  --prediction-jsonl "$TRAIN_POOL" \
  --labels-jsonl "$BENCHMARK_LABELS" \
  --output-dir "$TRAIN_DIR" \
  --train-fraction "$TRAIN_FRACTION" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --fallback-mode chem_light \
  --pair-type-weights "$PAIR_TYPE_WEIGHTS" 2>&1 | tee "$OUT_ROOT/train_pair_reward.log"

python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
  --benchmark-jsonl "$BENCHMARK_LABELS" \
  --prediction-jsonl "$TRAIN_DIR/pred_pair_reward_head.jsonl" \
  --report-json "$TRAIN_DIR/report_pair_reward_head.json" \
  --details-jsonl "$TRAIN_DIR/details_pair_reward_head.jsonl" \
  --group-fields source,difficulty,task_type,eval_panel 2>&1 | tee "$TRAIN_DIR/eval_pair_reward.log"

python V2-1/scripts/sweep_candidate_reward_head_margin.py \
  --checkpoint "$TRAIN_DIR/reward_head.pt" \
  --prediction-jsonl "$TRAIN_POOL" \
  --labels-jsonl "$BENCHMARK_LABELS" \
  --output-dir "$TRAIN_DIR/margin_sweep" \
  --fallback-mode chem_light \
  --margin-grid "$EVAL_MARGIN_GRID" \
  --group-fields source,difficulty,task_type,eval_panel,source+difficulty,source+difficulty+task_type \
  --group-margin-fields "$EVAL_GROUP_MARGIN_FIELDS" 2>&1 | tee "$TRAIN_DIR/margin_sweep.log"

python - <<'PY' "$OUT_ROOT" "$TRAIN_DIR" "$PREFERENCE_REPORT" "$TRAIN_POOL"
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
train_dir = Path(sys.argv[2])
preference_report_path = Path(sys.argv[3])
train_pool_path = Path(sys.argv[4])

summary = {
    "out_root": str(out_root),
    "train_dir": str(train_dir),
    "train_pool": None,
    "preference_report": None,
    "train_report": None,
    "eval_report": None,
    "margin_sweep": None,
    "best_eval_report": None,
}
summary["train_pool"] = str(train_pool_path)
if preference_report_path.exists():
    summary["preference_report"] = json.loads(preference_report_path.read_text(encoding="utf-8"))
train_report_path = train_dir / "pair_reward_head_report.json"
if train_report_path.exists():
    summary["train_report"] = json.loads(train_report_path.read_text(encoding="utf-8"))
eval_report_path = train_dir / "report_pair_reward_head.json"
if eval_report_path.exists():
    data = json.loads(eval_report_path.read_text(encoding="utf-8"))
    acc = data.get("accuracy", {})
    sim = data.get("similarity", {})
    summary["eval_report"] = {
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
sweep_path = train_dir / "margin_sweep" / "margin_sweep_summary.json"
if sweep_path.exists():
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    summary["margin_sweep"] = sweep
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
        summary["best_eval_report"] = {
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
            "by_eval_panel_exact": {
                key: value.get("canonical_exact_match_accuracy")
                for key, value in data.get("by_group", {}).get("eval_panel", {}).items()
            },
            "by_difficulty_exact": {
                key: value.get("canonical_exact_match_accuracy")
                for key, value in data.get("by_group", {}).get("difficulty", {}).items()
            },
        }
(out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

printf '\n=== cross-run pair reward train end %s ===\n' "$(date '+%F %T %Z')" | tee -a "$OUT_ROOT/run.log"
df -h /root/autodl-tmp /root/autodl-fs | tee -a "$OUT_ROOT/run.log"
