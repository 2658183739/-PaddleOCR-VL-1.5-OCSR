#!/usr/bin/env bash
set -euo pipefail

cd "${REMOTE_ROOT:-/root/autodl-tmp/data/platform_migration_bundle_20260531}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONUNBUFFERED=1

MODEL_DIR="${MODEL_DIR:-V2-1/outputs/export}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-fs/outputs_v2/main_eval_with_candidates_20260627_fast_notta}"
POLICY_JSON="${POLICY_JSON:-/root/autodl-fs/outputs_v2/reward_policy_rerank_20260626_split75/policy_raw.json}"
POLICY_MARGIN="${POLICY_MARGIN:-1.5}"
MIN_PIXELS="${MIN_PIXELS:-50176}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
TTA_PRESET="${TTA_PRESET:-none}"
NUM_BEAMS="${NUM_BEAMS:-4}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-4}"

CANONICAL_LABELS="${CANONICAL_LABELS:-V2-1/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl}"
WEAK_LABELS="${WEAK_LABELS:-V2-1/data/eval/weak_domain_v2/annotations/labels.jsonl}"

mkdir -p "$OUT_ROOT"/combined

printf '\n=== main eval with candidates start %s ===\n' "$(date '+%F %T %Z')" | tee "$OUT_ROOT/run.log"
echo "MODEL_DIR=$MODEL_DIR" | tee -a "$OUT_ROOT/run.log"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$OUT_ROOT/run.log"
echo "POLICY_JSON=$POLICY_JSON" | tee -a "$OUT_ROOT/run.log"
echo "POLICY_MARGIN=$POLICY_MARGIN" | tee -a "$OUT_ROOT/run.log"
echo "TTA_PRESET=$TTA_PRESET NUM_BEAMS=$NUM_BEAMS NUM_RETURN_SEQUENCES=$NUM_RETURN_SEQUENCES" | tee -a "$OUT_ROOT/run.log"
df -h /root/autodl-tmp /root/autodl-fs | tee -a "$OUT_ROOT/run.log"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$OUT_ROOT/run.log" || true

prepare_panel_labels() {
  local name="$1"
  local source_labels="$2"
  local stable_labels="$3"
  mkdir -p "$(dirname "$stable_labels")"

  python - <<'PY' "$source_labels" "$stable_labels"
import json
import sys
from pathlib import Path

source_labels = Path(sys.argv[1]).resolve()
stable_labels = Path(sys.argv[2]).resolve()
project_root = Path.cwd().resolve()

def image_ref(row):
    if str(row.get("image", "")).strip():
        return str(row["image"])
    if str(row.get("image_path", "")).strip():
        return str(row["image_path"])
    raise KeyError(f"missing image field for row {row.get('id')}")

rows = []
missing = []
for row in [json.loads(line) for line in source_labels.read_text(encoding="utf-8").splitlines() if line.strip()]:
    raw = Path(image_ref(row))
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [
            project_root / raw,
            source_labels.parent / raw,
            source_labels.parent.parent / raw,
            source_labels.parent.parent.parent / raw,
        ]
    resolved = next((path.resolve() for path in candidates if path.exists()), None)
    if resolved is None:
        missing.append({"id": row.get("id"), "image": str(raw)})
        continue
    out = dict(row)
    try:
        out["image"] = resolved.relative_to(project_root).as_posix()
    except ValueError:
        out["image"] = resolved.as_posix()
    out.pop("image_path", None)
    rows.append(out)

stable_labels.parent.mkdir(parents=True, exist_ok=True)
with stable_labels.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "source_labels": str(source_labels),
    "stable_labels": str(stable_labels),
    "total": len(rows) + len(missing),
    "written": len(rows),
    "missing_images": missing[:20],
    "missing_count": len(missing),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
if missing:
    raise SystemExit(2)
PY
}

run_panel() {
  local name="$1"
  local label_file="$2"
  local out_dir="$OUT_ROOT/$name"
  mkdir -p "$out_dir/parts"

  if [[ ! -s "$label_file" ]]; then
    echo "missing label file: $label_file" | tee -a "$OUT_ROOT/run.log"
    return 2
  fi

  python V2-1/scripts/eval_jsonl_resume.py \
    --benchmark-jsonl "$label_file" \
    --prediction-glob "$out_dir/parts/*.jsonl" \
    --merged-jsonl "$out_dir/pred.jsonl" \
    --remaining-jsonl "$out_dir/remaining.jsonl" \
    --status-json "$out_dir/status.json" | tee "$out_dir/status_before.log"

  local remaining
  remaining="$(python - <<'PY' "$out_dir/status.json"
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["remaining"])
PY
)"

  if [[ "$remaining" -gt 0 ]]; then
    local part="$out_dir/parts/part_$(date '+%Y%m%d_%H%M%S').jsonl"
    printf '\n=== %s infer remaining=%s tta=%s %s ===\n' "$name" "$remaining" "$TTA_PRESET" "$(date '+%F %T %Z')" | tee -a "$OUT_ROOT/run.log"
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
      --tta-preset "$TTA_PRESET" \
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

  local after_remaining
  after_remaining="$(python - <<'PY' "$out_dir/status.json"
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["remaining"])
PY
)"
  if [[ "$after_remaining" -ne 0 ]]; then
    echo "panel $name still has remaining=$after_remaining; skip scoring until complete" | tee -a "$OUT_ROOT/run.log"
    return 0
  fi

  python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
    --benchmark-jsonl "$label_file" \
    --prediction-jsonl "$out_dir/pred.jsonl" \
    --report-json "$out_dir/report_selected.json" \
    --details-jsonl "$out_dir/details_selected.jsonl" 2>&1 | tee "$out_dir/eval_selected.log"

  python V2-1/scripts/rerank_ocsr_candidates.py \
    --prediction-jsonl "$out_dir/pred.jsonl" \
    --output-jsonl "$out_dir/pred_chem_light.jsonl" \
    --labels-jsonl "$label_file" \
    --report-json "$out_dir/report_chem_light_internal.json" \
    --preference-jsonl "$out_dir/preference_pairs_chem_light.jsonl" \
    --mode chem_light 2>&1 | tee "$out_dir/rerank_chem_light.log"

  python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
    --benchmark-jsonl "$label_file" \
    --prediction-jsonl "$out_dir/pred_chem_light.jsonl" \
    --report-json "$out_dir/report_chem_light.json" \
    --details-jsonl "$out_dir/details_chem_light.jsonl" 2>&1 | tee "$out_dir/eval_chem_light.log"

  if [[ -s "$POLICY_JSON" ]]; then
    python V2-1/scripts/reward_policy_reranker.py \
      --prediction-jsonl "$out_dir/pred.jsonl" \
      --labels-jsonl "$label_file" \
      --load-policy-json "$POLICY_JSON" \
      --output-jsonl "$out_dir/pred_reward_policy.jsonl" \
      --report-json "$out_dir/report_reward_policy_internal.json" \
      --details-jsonl "$out_dir/details_reward_policy_internal.jsonl" \
      --fallback-mode chem_light \
      --policy-margin "$POLICY_MARGIN" 2>&1 | tee "$out_dir/rerank_reward_policy.log"

    python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
      --benchmark-jsonl "$label_file" \
      --prediction-jsonl "$out_dir/pred_reward_policy.jsonl" \
      --report-json "$out_dir/report_reward_policy.json" \
      --details-jsonl "$out_dir/details_reward_policy.jsonl" 2>&1 | tee "$out_dir/eval_reward_policy.log"
  else
    echo "skip reward policy: missing $POLICY_JSON" | tee -a "$OUT_ROOT/run.log"
  fi
}

CANONICAL_STABLE_LABELS="$OUT_ROOT/splits/canonical_smiles_main_v1/annotations/labels.jsonl"
WEAK_STABLE_LABELS="$OUT_ROOT/splits/weak_domain_v2/annotations/labels.jsonl"

prepare_panel_labels canonical_smiles_main_v1 "$CANONICAL_LABELS" "$CANONICAL_STABLE_LABELS" | tee "$OUT_ROOT/prepare_canonical_smiles_main_v1.log"
prepare_panel_labels weak_domain_v2 "$WEAK_LABELS" "$WEAK_STABLE_LABELS" | tee "$OUT_ROOT/prepare_weak_domain_v2.log"

run_panel canonical_smiles_main_v1 "$CANONICAL_STABLE_LABELS"
run_panel weak_domain_v2 "$WEAK_STABLE_LABELS"

python - <<'PY' "$OUT_ROOT"
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
panels = ["canonical_smiles_main_v1", "weak_domain_v2"]
variants = {
    "selected": "pred.jsonl",
    "chem_light": "pred_chem_light.jsonl",
    "reward_policy": "pred_reward_policy.jsonl",
}

combined = out_root / "combined"
combined.mkdir(parents=True, exist_ok=True)

label_out = combined / "labels.jsonl"
label_rows = []
panel_label_counts = {}
for panel in panels:
    status_path = out_root / panel / "status.json"
    if not status_path.exists():
        continue
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("remaining") != 0:
        continue
    label_path = Path(status["benchmark"])
    rows = [json.loads(line) for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    panel_label_counts[panel] = len(rows)
    for row in rows:
        out = dict(row)
        out["id"] = f"{panel}::{row['id']}"
        out["eval_panel"] = panel
        label_rows.append(out)

with label_out.open("w", encoding="utf-8") as handle:
    for row in label_rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

combined_status = {
    "panels": panels,
    "panel_label_counts": panel_label_counts,
    "total": len(label_rows),
    "labels_jsonl": str(label_out),
    "variant_predictions": {},
}

for variant, file_name in variants.items():
    rows = []
    complete = True
    for panel in panels:
        pred_path = out_root / panel / file_name
        if panel not in panel_label_counts or not pred_path.exists():
            complete = False
            continue
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["id"] = f"{panel}::{row['id']}"
            row["eval_panel"] = panel
            rows.append(row)
    if complete and len(rows) == len(label_rows):
        out_path = combined / f"pred_{variant}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        combined_status["variant_predictions"][variant] = str(out_path)
    else:
        combined_status["variant_predictions"][variant] = None

(combined / "combined_status.json").write_text(
    json.dumps(combined_status, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(combined_status, ensure_ascii=False, indent=2))
PY

if [[ -s "$OUT_ROOT/combined/pred_selected.jsonl" ]]; then
  python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
    --benchmark-jsonl "$OUT_ROOT/combined/labels.jsonl" \
    --prediction-jsonl "$OUT_ROOT/combined/pred_selected.jsonl" \
    --report-json "$OUT_ROOT/combined/report_selected.json" \
    --details-jsonl "$OUT_ROOT/combined/details_selected.jsonl" \
    --group-fields source,difficulty,task_type,eval_panel 2>&1 | tee "$OUT_ROOT/combined/eval_selected.log"
fi

if [[ -s "$OUT_ROOT/combined/pred_chem_light.jsonl" ]]; then
  python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
    --benchmark-jsonl "$OUT_ROOT/combined/labels.jsonl" \
    --prediction-jsonl "$OUT_ROOT/combined/pred_chem_light.jsonl" \
    --report-json "$OUT_ROOT/combined/report_chem_light.json" \
    --details-jsonl "$OUT_ROOT/combined/details_chem_light.jsonl" \
    --group-fields source,difficulty,task_type,eval_panel 2>&1 | tee "$OUT_ROOT/combined/eval_chem_light.log"
fi

if [[ -s "$OUT_ROOT/combined/pred_reward_policy.jsonl" ]]; then
  python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
    --benchmark-jsonl "$OUT_ROOT/combined/labels.jsonl" \
    --prediction-jsonl "$OUT_ROOT/combined/pred_reward_policy.jsonl" \
    --report-json "$OUT_ROOT/combined/report_reward_policy.json" \
    --details-jsonl "$OUT_ROOT/combined/details_reward_policy.jsonl" \
    --group-fields source,difficulty,task_type,eval_panel 2>&1 | tee "$OUT_ROOT/combined/eval_reward_policy.log"
fi

python - <<'PY' "$OUT_ROOT" "$POLICY_JSON" "$POLICY_MARGIN"
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
policy_json = sys.argv[2]
policy_margin = sys.argv[3]
variants = ["selected", "chem_light", "reward_policy"]
panels = ["canonical_smiles_main_v1", "weak_domain_v2", "combined"]

def read_report(path):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    acc = data.get("accuracy", {})
    sim = data.get("similarity", {})
    return {
        "total": data.get("total"),
        "missing_predictions": data.get("missing_predictions"),
        "canonical_exact": acc.get("canonical_exact_match_accuracy"),
        "raw_exact": acc.get("raw_exact_match_accuracy"),
        "valid_smiles_rate": acc.get("valid_smiles_rate"),
        "mean_tanimoto": sim.get("mean_fingerprint_tanimoto"),
        "mean_edit_similarity": sim.get("mean_normalized_edit_similarity"),
        "tanimoto_coverage": sim.get("fingerprint_tanimoto_coverage"),
        "by_source_exact": {
            key: value.get("canonical_exact_match_accuracy")
            for key, value in data.get("by_group", {}).get("source", {}).items()
        },
        "by_task_type_exact": {
            key: value.get("canonical_exact_match_accuracy")
            for key, value in data.get("by_group", {}).get("task_type", {}).items()
        },
        "by_eval_panel_exact": {
            key: value.get("canonical_exact_match_accuracy")
            for key, value in data.get("by_group", {}).get("eval_panel", {}).items()
        },
    }

summary = {
    "out_root": str(out_root),
    "policy_json": policy_json,
    "policy_margin": policy_margin,
    "variants": {},
}
for panel in panels:
    summary["variants"][panel] = {}
    for variant in variants:
        report = out_root / panel / f"report_{variant}.json"
        summary["variants"][panel][variant] = read_report(report)

combined = summary["variants"].get("combined", {})
selected = combined.get("selected")
for variant in ["chem_light", "reward_policy"]:
    current = combined.get(variant)
    if selected and current:
        current["delta_vs_selected_exact"] = current["canonical_exact"] - selected["canonical_exact"]
        current["delta_vs_selected_tanimoto"] = current["mean_tanimoto"] - selected["mean_tanimoto"]
chem = combined.get("chem_light")
reward = combined.get("reward_policy")
if chem and reward:
    reward["delta_vs_chem_light_exact"] = reward["canonical_exact"] - chem["canonical_exact"]
    reward["delta_vs_chem_light_tanimoto"] = reward["mean_tanimoto"] - chem["mean_tanimoto"]

(out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

printf '\n=== main eval with candidates end %s ===\n' "$(date '+%F %T %Z')" | tee -a "$OUT_ROOT/run.log"
df -h /root/autodl-tmp /root/autodl-fs | tee -a "$OUT_ROOT/run.log"
