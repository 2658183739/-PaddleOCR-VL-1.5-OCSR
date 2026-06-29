#!/usr/bin/env bash
set -euo pipefail

VARIANTS="${VARIANTS:-layout_primary,layout_target_tight,layout_primary_trim,layout_primary_gray,layout_auto_structure,layout_wide,layout_top_left,layout_full,layout_center,layout_right,layout_lower,layout_first_row,layout_structure_band,layout_top_left_tiny,layout_wide_trim,layout_wide_auto_structure,layout_target_gray,layout_focus_lower_right,layout_focus_mid_right,layout_captionless_lower,layout_panel_a_zoom,layout_panel_a_square}"

export VARIANTS
exec bash V2-1/run_4090_weak_layout_crop_candidates_v1.sh "$@"
