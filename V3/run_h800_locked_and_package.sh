#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_ROOT="${V3_BACKUP_ROOT:-/root/autodl-fs/V3_results}"

cd "$PROJECT_ROOT"
test -s V3/models/final_best_export/config.json
test -s "$BACKUP_ROOT/evidence/final_pipeline_complete.txt"

if [[ ! -s V3/evidence/generation_policy_selection.json ]]; then
  bash V3/run_h800_generation_ablation.sh
fi

# This is CPU-only and reuses the frozen beam candidate pool. It must run before
# locked test so the final selector is chosen only on development data.
if [[ ! -s "$BACKUP_ROOT/evidence/candidate_rerank_complete.txt" ]]; then
  bash V3/run_h800_candidate_rerank.sh
fi

MODEL_DIR="$PROJECT_ROOT/V3/models/final_best_export" \
V3_INFER_WORKERS="${V3_INFER_WORKERS:-1}" \
UNLOCK_FINAL_TEST=FINAL_MODEL_SELECTED \
bash V3/run_locked_final_test.sh

python V3/scripts/audit_release_readiness.py --project-root .
python V3/scripts/build_final_report.py --project-root .
python V3/scripts/collect_training_artifacts.py --project-root .
python -m unittest discover -s V3/tests -v

mkdir -p "$BACKUP_ROOT/eval_runs_locked" "$BACKUP_ROOT/evidence"
rsync -a V3/eval_runs_locked/ "$BACKUP_ROOT/eval_runs_locked/"
rsync -a V3/evidence/ "$BACKUP_ROOT/evidence/"
python -m pip freeze > "$BACKUP_ROOT/evidence/pip-freeze-final.txt"
nvidia-smi > "$BACKUP_ROOT/evidence/nvidia-smi-final.txt"
mkdir -p V3/evidence/runtime V3/logs
rsync -a "$BACKUP_ROOT/evidence/pip-freeze-final.txt" V3/evidence/runtime/
rsync -a "$BACKUP_ROOT/evidence/nvidia-smi-final.txt" V3/evidence/runtime/
rsync -a "$BACKUP_ROOT/logs/" V3/logs/

STAMP="$(date +%Y%m%d_%H%M%S)"
PACKAGE="$BACKUP_ROOT/v3_final_model_scripts_evidence_$STAMP.tar.gz"
tar -czf "$PACKAGE" \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  V3/README.md \
  V3/MODEL_CARD_zh.md \
  V3/DATASET_CARD_zh.md \
  V3/TRAINING_DATA_AND_FINETUNING_REPORT_zh.md \
  V3/OFFICIAL_FEEDBACK_RESPONSE_zh.md \
  V3/CONTRIBUTING.md \
  V3/REPRODUCTION_GUIDE_zh.md \
  V3/RELEASE_UPLOAD_GUIDE_zh.md \
  V3/SUBMISSION_CHECKLIST_zh.md \
  V3/MISSING_CONTENT_AND_FIXES_zh.md \
  V3/DECISION_AND_48H_PLAN_zh.md \
  V3/configs \
  V3/data/source \
  V3/data/sft_materialized \
  V3/data/eval/dev_legacy_core_strict \
  V3/data/eval/dev_legacy_region_strict \
  V3/data/eval/wild_strict_v3 \
  V3/data/eval/wild_strict_scaffold_novel_v3 \
  V3/data/eval/wild_symbolic_v3 \
  V3/demo \
  V3/qc \
  V3/scripts \
  V3/tests \
  V3/runbooks \
  V3/evidence \
  V3/eval_runs_locked \
  V3/logs \
  V3/models/final_best_export \
  V3/setup_h800_environment.sh \
  V3/run_a100_stage.sh \
  V3/run_h800_probes.sh \
  V3/run_h800_probe_eval.sh \
  V3/run_h800_probe_pairwise.sh \
  V3/run_h800_final.sh \
  V3/run_h800_generation_ablation.sh \
  V3/run_h800_candidate_rerank.sh \
  V3/run_h800_locked_and_package.sh \
  V3/run_h800_pipeline.sh \
  V3/run_locked_final_test.sh

sha256sum "$PACKAGE" > "$PACKAGE.sha256"
printf '%s\n' "$PACKAGE" > "$BACKUP_ROOT/evidence/final_package_path.txt"
date -Iseconds > "$BACKUP_ROOT/evidence/all_pipeline_complete.txt"
echo "[DONE] locked test and final package completed: $PACKAGE"
