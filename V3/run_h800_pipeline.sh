#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${V3_OMP_NUM_THREADS:-1}"
bash V3/run_h800_probes.sh
bash V3/run_h800_probe_eval.sh
bash V3/run_h800_probe_pairwise.sh
bash V3/run_h800_final.sh
bash V3/run_h800_generation_ablation.sh
bash V3/run_h800_candidate_rerank.sh
bash V3/run_h800_locked_and_package.sh
