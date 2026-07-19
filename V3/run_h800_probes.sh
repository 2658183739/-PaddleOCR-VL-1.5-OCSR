#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_ROOT="${V3_BACKUP_ROOT:-/root/autodl-fs/V3_results}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${V3_OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
mkdir -p V3/outputs V3/logs "$BACKUP_ROOT/outputs" "$BACKUP_ROOT/logs" "$BACKUP_ROOT/evidence"

nvidia-smi > "$BACKUP_ROOT/evidence/nvidia-smi.txt"
python --version > "$BACKUP_ROOT/evidence/python-version.txt" 2>&1
python -m pip freeze > "$BACKUP_ROOT/evidence/pip-freeze.txt"
cp -f V3/evidence/experiment_matrix.csv "$BACKUP_ROOT/evidence/experiment_matrix_planned.csv"

run_probe() {
  local stage="$1"
  local run_id="$2"
  local seed="$3"
  local output_dir="./V3/outputs/$run_id"

  if [[ -s "$output_dir/train_results.json" ]]; then
    echo "[SKIP] completed run: $run_id"
    return
  fi

  echo "[RUN] stage=$stage run_id=$run_id seed=$seed"
  bash V3/run_a100_stage.sh "$stage" \
    seed="$seed" \
    max_steps=250 \
    do_eval=false \
    eval_steps=999999 \
    save_steps=250 \
    per_device_train_batch_size=32 \
    gradient_accumulation_steps=1 \
    output_dir="$output_dir" \
    logging_dir="$output_dir/visualdl_logs"

  rsync -a "$output_dir/" "$BACKUP_ROOT/outputs/$run_id/"
  rsync -a V3/logs/ "$BACKUP_ROOT/logs/"
}

# Balanced order within each seed reduces time-order confounding.
run_probe probe_b data_11_s1 20260717
run_probe probe_a data_00_s1 20260717
run_probe probe_d data_10_s1 20260717
run_probe probe_e data_01_s1 20260717

run_probe probe_b data_11_s2 20260718
run_probe probe_d data_10_s2 20260718
run_probe probe_a data_00_s2 20260718
run_probe probe_e data_01_s2 20260718

# Planned warm-start and augmentation-dose diagnostics.
run_probe probe_base15 warmstart_control_s1 20260717
run_probe probe_c aug_dose2_s1 20260717

date -Iseconds > "$BACKUP_ROOT/evidence/probes_complete.txt"
echo "[DONE] all H800 probe runs completed"
