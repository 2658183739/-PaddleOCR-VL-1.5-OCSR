#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

for name in phase1_lora phase2_lora phase3_lora; do
  target="$PROJECT_ROOT/V2/outputs/$name"
  if [[ -d "$target" ]]; then
    rm -rf "$target"
  fi
  mkdir -p "$target"
  echo "reset $target"
done
