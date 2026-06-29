#!/usr/bin/env bash
set -euo pipefail

VARIANTS="${VARIANTS:-layout_primary,layout_target_tight,layout_auto_structure,layout_textless_right,layout_structure_core_tight,layout_panel_a_core,layout_mid_structure_only}"

export VARIANTS
exec bash V2-1/run_4090_weak_layout_crop_candidates_v1.sh "$@"
