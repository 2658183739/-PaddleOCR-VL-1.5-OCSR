#!/usr/bin/env bash
set -euo pipefail

cd "${REMOTE_ROOT:-/root/autodl-tmp/data/platform_migration_bundle_20260531}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONUNBUFFERED=1

LABELS="${LABELS:-V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl}"
MODEL_DIR="${MODEL_DIR:-V2-1/outputs/export}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-fs/outputs_v2/full_eval_region_panel_v1/ocsr_realworld_mixed_eval_v1p1}"
MIN_PIXELS="${MIN_PIXELS:-50176}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
GENERAL_TTA="${GENERAL_TTA:-light}"
CHINESE_TTA="${CHINESE_TTA:-none}"
NUM_BEAMS="${NUM_BEAMS:-4}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-4}"

mkdir -p "$OUT_ROOT"/{splits,general/parts,chinese_exam_panel/parts,merged}

printf '\n=== full region-panel eval start %s ===\n' "$(date '+%F %T %Z')" | tee "$OUT_ROOT/run.log"
echo "LABELS=$LABELS" | tee -a "$OUT_ROOT/run.log"
echo "MODEL_DIR=$MODEL_DIR" | tee -a "$OUT_ROOT/run.log"
df -h /root/autodl-tmp /root/autodl-fs | tee -a "$OUT_ROOT/run.log"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$OUT_ROOT/run.log" || true

python - <<'PY' "$LABELS" "$OUT_ROOT"
import json
import sys
from collections import Counter
from pathlib import Path

labels = Path(sys.argv[1])
out_root = Path(sys.argv[2])
project_root = Path.cwd().resolve()
rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines() if line.strip()]
general = []
chinese = []

def get_image_ref(row):
    if str(row.get("image", "")).strip():
        return str(row["image"])
    if str(row.get("image_path", "")).strip():
        return str(row["image_path"])
    raise KeyError(f"missing image field for row {row.get('id')}")

def stable_row(row):
    raw = Path(get_image_ref(row))
    candidates = [raw] if raw.is_absolute() else [
        project_root / raw,
        labels.parent / raw,
        labels.parent.parent / raw,
        labels.parent.parent.parent / raw,
    ]
    image_path = next((path.resolve() for path in candidates if path.exists()), None)
    if image_path is None:
        raise FileNotFoundError(str(raw))
    out = dict(row)
    try:
        out["image"] = image_path.relative_to(project_root).as_posix()
    except ValueError:
        out["image"] = image_path.as_posix()
    out.pop("image_path", None)
    return out

for row in rows:
    row = stable_row(row)
    task = str(row.get("task_type", ""))
    difficulty = str(row.get("difficulty", ""))
    sample_id = str(row.get("id", ""))
    if "chinese_exam" in task or "chinese_exam" in difficulty or "extra_chinese_exam" in sample_id:
        chinese.append(row)
    else:
        general.append(row)

def write(path, out_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

write(out_root / "splits" / "general" / "annotations" / "labels.jsonl", general)
write(out_root / "splits" / "chinese_exam_raw" / "annotations" / "labels.jsonl", chinese)
summary = {
    "source_labels": str(labels),
    "total": len(rows),
    "general": len(general),
    "chinese_exam": len(chinese),
    "source": dict(Counter(str(row.get("source", "")) for row in rows)),
    "task_type": dict(Counter(str(row.get("task_type", "")) for row in rows)),
}
(out_root / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

CHINESE_RAW="$OUT_ROOT/splits/chinese_exam_raw/annotations/labels.jsonl"
CHINESE_PANEL_ROOT="$OUT_ROOT/splits/chinese_exam_panel"
if [[ -s "$CHINESE_RAW" ]]; then
  python V2-1/scripts/build_realworld_region_crop_probe.py \
    --project-root . \
    --labels-jsonl "$CHINESE_RAW" \
    --output-root "$CHINESE_PANEL_ROOT" \
    --variants exam_q1_panel | tee "$OUT_ROOT/chinese_panel_manifest.log"
fi

run_panel() {
  local name="$1"
  local label_file="$2"
  local tta="$3"
  local out_dir="$OUT_ROOT/$name"
  mkdir -p "$out_dir/parts"

  python V2-1/scripts/eval_jsonl_resume.py \
    --benchmark-jsonl "$label_file" \
    --prediction-glob "$out_dir/parts/*.jsonl" \
    --merged-jsonl "$out_dir/pred.jsonl" \
    --remaining-jsonl "$out_dir/remaining.jsonl" \
    --status-json "$out_dir/status.json" | tee "$out_dir/status_before.log"

  local remaining
  remaining="$(python - <<'PY' "$out_dir/status.json"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["remaining"])
PY
)"
  if [[ "$remaining" -gt 0 ]]; then
    local part="$out_dir/parts/part_$(date '+%Y%m%d_%H%M%S').jsonl"
    printf '\n=== %s infer remaining=%s tta=%s %s ===\n' "$name" "$remaining" "$tta" "$(date '+%F %T %Z')" | tee -a "$OUT_ROOT/run.log"
    python V2-1/scripts/infer_ocsr_transformers.py \
      --model-dir "$MODEL_DIR" \
      --benchmark-jsonl "$out_dir/remaining.jsonl" \
      --project-root . \
      --output-jsonl "$part" \
      --prompt-list-file V2-1/configs/prompt_rank.txt \
      --num-beams "$NUM_BEAMS" \
      --num-return-sequences "$NUM_RETURN_SEQUENCES" \
      --repetition-penalty 1.05 \
      --no-repeat-ngram-size 8 \
      --tta-preset "$tta" \
      --save-candidates \
      --device cuda \
      --torch-dtype bfloat16 \
      --min-pixels "$MIN_PIXELS" \
      --max-pixels "$MAX_PIXELS" 2>&1 | tee "$out_dir/infer_$(date '+%Y%m%d_%H%M%S').log"
  fi

  python V2-1/scripts/eval_jsonl_resume.py \
    --benchmark-jsonl "$label_file" \
    --prediction-glob "$out_dir/parts/*.jsonl" \
    --merged-jsonl "$out_dir/pred.jsonl" \
    --remaining-jsonl "$out_dir/remaining.jsonl" \
    --status-json "$out_dir/status.json" | tee "$out_dir/status_after.log"
}

GENERAL_LABELS="$OUT_ROOT/splits/general/annotations/labels.jsonl"
CHINESE_LABELS="$CHINESE_PANEL_ROOT/exam_q1_panel/annotations/labels.jsonl"

if [[ -s "$CHINESE_LABELS" ]]; then
  run_panel chinese_exam_panel "$CHINESE_LABELS" "$CHINESE_TTA"
fi
if [[ -s "$GENERAL_LABELS" ]]; then
  run_panel general "$GENERAL_LABELS" "$GENERAL_TTA"
fi

python - <<'PY' "$LABELS" "$OUT_ROOT"
import json
import sys
from pathlib import Path

labels = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
out_root = Path(sys.argv[2])
preds = {}
for name in ["general", "chinese_exam_panel"]:
    path = out_root / name / "pred.jsonl"
    if not path.exists():
        continue
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            preds[row["id"]] = row
merged = out_root / "merged" / "pred.jsonl"
merged.parent.mkdir(parents=True, exist_ok=True)
missing = []
with merged.open("w", encoding="utf-8") as handle:
    for row in labels:
        sample_id = row["id"]
        if sample_id not in preds:
            missing.append(sample_id)
            continue
        handle.write(json.dumps(preds[sample_id], ensure_ascii=False) + "\n")
status = {"total": len(labels), "merged": len(preds), "missing": missing}
(out_root / "merged" / "merge_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(status, ensure_ascii=False, indent=2))
if missing:
    raise SystemExit(3)
PY

python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
  --benchmark-jsonl "$LABELS" \
  --prediction-jsonl "$OUT_ROOT/merged/pred.jsonl" \
  --report-json "$OUT_ROOT/merged/report.json" \
  --details-jsonl "$OUT_ROOT/merged/details.jsonl" 2>&1 | tee "$OUT_ROOT/merged/eval.log"

python V2-1/scripts/rerank_ocsr_candidates.py \
  --prediction-jsonl "$OUT_ROOT/merged/pred.jsonl" \
  --output-jsonl "$OUT_ROOT/merged/rerank_chem_light_pred.jsonl" \
  --labels-jsonl "$LABELS" \
  --report-json "$OUT_ROOT/merged/rerank_chem_light_report.json" \
  --preference-jsonl "$OUT_ROOT/merged/preference_pairs.jsonl" \
  --mode chem_light 2>&1 | tee "$OUT_ROOT/merged/rerank.log"

python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
  --benchmark-jsonl "$LABELS" \
  --prediction-jsonl "$OUT_ROOT/merged/rerank_chem_light_pred.jsonl" \
  --report-json "$OUT_ROOT/merged/rerank_chem_light_eval_report.json" \
  --details-jsonl "$OUT_ROOT/merged/rerank_chem_light_details.jsonl" 2>&1 | tee "$OUT_ROOT/merged/rerank_eval.log"

python - <<'PY' "$OUT_ROOT"
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
selected = json.loads((out / "merged" / "report.json").read_text(encoding="utf-8"))
rerank = json.loads((out / "merged" / "rerank_chem_light_eval_report.json").read_text(encoding="utf-8"))
diag = json.loads((out / "merged" / "rerank_chem_light_report.json").read_text(encoding="utf-8"))
summary = {
    "selected": selected["accuracy"],
    "selected_similarity": selected["similarity"],
    "rerank": rerank["accuracy"],
    "rerank_similarity": rerank["similarity"],
    "rerank_diag": diag,
    "by_source_rerank": {
        key: value["canonical_exact_match_accuracy"]
        for key, value in rerank.get("by_group", {}).get("source", {}).items()
    },
    "by_task_type_rerank": {
        key: value["canonical_exact_match_accuracy"]
        for key, value in rerank.get("by_group", {}).get("task_type", {}).items()
    },
}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

printf '\n=== full region-panel eval end %s ===\n' "$(date '+%F %T %Z')" | tee -a "$OUT_ROOT/run.log"
df -h /root/autodl-tmp /root/autodl-fs | tee -a "$OUT_ROOT/run.log"
