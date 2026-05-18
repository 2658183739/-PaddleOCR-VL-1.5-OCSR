#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN_ARGS=()
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN_ARGS+=(--dry-run)
fi

cd "$PROJECT_ROOT"

"$PYTHON_BIN" V2/scripts/normalize_image_modes.py \
  --root "$PROJECT_ROOT/V2/data/assets/train_phase1" \
  --root "$PROJECT_ROOT/V2/data/assets/train_phase2" \
  --root "$PROJECT_ROOT/V2/data/assets/train_phase3" \
  --root "$PROJECT_ROOT/V2/data/eval/canonical_smiles_main_v1/images" \
  "${DRY_RUN_ARGS[@]}"
