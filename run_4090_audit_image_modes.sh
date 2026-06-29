#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source /data/miniconda/etc/profile.d/conda.sh
conda activate torch
cd "$PROJECT_ROOT"

python "$PROJECT_ROOT/V2/scripts/audit_phase_image_modes.py" "$@"
