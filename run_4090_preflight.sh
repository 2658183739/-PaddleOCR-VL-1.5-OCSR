#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

python V2/scripts/build_v2_ocsr_messages_dataset.py --project-root "$PROJECT_ROOT"
python V2/scripts/materialize_v2_assets.py --project-root "$PROJECT_ROOT"

echo "V2 preflight complete at $PROJECT_ROOT"
