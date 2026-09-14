#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU ASSIGNMENT_DIR RUNROOT MIN_FREE_MIB" >&2
  exit 2
fi
GPU="$1"
ASSIGN="$2"
RUNROOT="$3"
MIN_FREE_MIB="$4"

ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_b24"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
DAPS_PY="$ROOT/conda-envs/daps/bin/python"
HARD_CEILING_MIB=52452

case "$GPU" in
  0) EXPECTED_UUID=GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3 ;;
  1) EXPECTED_UUID=GPU-883c037a-34d2-48c4-467f-9a352fd8fdff ;;
  2) EXPECTED_UUID=GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe ;;
  3) EXPECTED_UUID=GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a ;;
  *) echo "STOP|invalid_gpu:$GPU"; exit 2 ;;
esac
[[ "$MIN_FREE_MIB" =~ ^[0-9]+$ ]] || { echo "STOP|bad_min_free:$MIN_FREE_MIB"; exit 2; }
(( MIN_FREE_MIB <= HARD_CEILING_MIB )) || { echo "STOP|min_free_exceeds_hard_ceiling:$MIN_FREE_MIB"; exit 2; }
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$UUID" == "$EXPECTED_UUID" ]] || { echo "STOP|gpu_uuid_mismatch:$GPU:$UUID"; exit 2; }
[[ -d "$ASSIGN" ]] || { echo "STOP|missing_assignment_dir:$ASSIGN"; exit 2; }
mkdir -p "$RUNROOT/workers/gpu${GPU}"
WORKER="$RUNROOT/workers/gpu${GPU}"
export PYTHONPATH="$REPO:$REPO/scripts/b22${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

mapfile -t ROWFILES < <(find "$ASSIGN" -maxdepth 1 -type f -name 'row*.csv' | sort)
TOTAL=${#ROWFILES[@]}
DONE=0
WAIT_TOTAL=0

for ROWCSV in "${ROWFILES[@]}"; do
  BASENAME=$(basename "$ROWCSV" .csv)
  IMGDIR="$WORKER/$BASENAME"
  if [[ -f "$IMGDIR/methods/SMOKE_COMPLETE.json" ]]; then
    STATUS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$IMGDIR/methods/SMOKE_COMPLETE.json")
    [[ "$STATUS" == "PASS" ]] || { echo "STOP|existing_nonpass:$IMGDIR"; exit 3; }
    DONE=$((DONE+1))
    echo "PILOT_REUSE|gpu=$GPU|row=$BASENAME|done=$DONE/$TOTAL"
    continue
  fi
  [[ ! -e "$IMGDIR" ]] || { echo "STOP|partial_image_dir_requires_manual_review:$IMGDIR"; exit 3; }
  mkdir -p "$IMGDIR"
  cp "$ROWCSV" "$IMGDIR/role_row.csv"
  "$PY" - "$ROWCSV" > "$IMGDIR/role.env" <<'PY'
import csv,sys
with open(sys.argv[1],newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
if len(rows)!=1: raise SystemExit('expected one assignment row')
r=rows[0]
print(f"IMAGE_ID={r['image_id']}")
print(f"MEASUREMENT_SEED={r['dev_measurement_seed']}")
print(f"CLASS_LABEL={r['class_label']}")
PY
  # shellcheck disable=SC1090
  source "$IMGDIR/role.env"

  WAIT_START=$(date +%s)
  while true; do
    FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ "$FREE" =~ ^[0-9]+$ ]] || { echo "STOP|cannot_parse_free:$FREE"; exit 3; }
    if (( FREE >= MIN_FREE_MIB )); then break; fi
    echo "PILOT_WAIT_FOR_FIT|gpu=$GPU|image=$IMAGE_ID|free_mib=$FREE|required_mib=$MIN_FREE_MIB"
    sleep 60
  done
  NOW=$(date +%s); WAIT_TOTAL=$((WAIT_TOTAL + NOW - WAIT_START))

  echo "PILOT_IMAGE_START|gpu=$GPU|image=$IMAGE_ID|class=$CLASS_LABEL|free_mib=$FREE"
  CUDA_VISIBLE_DEVICES="$GPU" "$DAPS_PY" "$REPO/scripts/b24/generate_b24_locked_input.py" \
    --image-id "$IMAGE_ID" --measurement-seed "$MEASUREMENT_SEED" --output "$IMGDIR/input" \
    > "$IMGDIR/input_generation.log" 2>&1
  [[ -f "$IMGDIR/input/input_manifest.json" ]] || { echo "STOP|missing_input_manifest:$IMGDIR"; exit 4; }

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$REPO/scripts/b24/run_b24_3_np_branching.py" \
    --input-manifest "$IMGDIR/input/input_manifest.json" \
    --role-row "$IMGDIR/role_row.csv" \
    --output-root "$IMGDIR/methods" \
    --physical-gpu "$GPU" \
    > "$IMGDIR/run.log" 2>&1
  [[ -f "$IMGDIR/methods/SMOKE_COMPLETE.json" ]] || { echo "STOP|missing_method_summary:$IMGDIR"; exit 5; }
  STATUS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$IMGDIR/methods/SMOKE_COMPLETE.json")
  [[ "$STATUS" == "PASS" ]] || { echo "STOP|method_nonpass:$IMGDIR"; exit 5; }
  DONE=$((DONE+1))
  echo "PILOT_IMAGE_COMPLETE|gpu=$GPU|image=$IMAGE_ID|class=$CLASS_LABEL|done=$DONE/$TOTAL"
done

"$PY" - "$WORKER/WORKER_COMPLETE.json" "$GPU" "$TOTAL" "$DONE" "$WAIT_TOTAL" "$MIN_FREE_MIB" <<'PY'
import json,sys
out,gpu,total,done,wait,gate=sys.argv[1:]
p={"schema_version":"b24.pilot16-worker.v1","status":"PASS","gpu":int(gpu),"assigned":int(total),"completed":int(done),"fit_wait_seconds":int(wait),"min_free_mib":int(gate)}
open(out,'w').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY

echo "B24_3_PILOT_WORKER_PASS|gpu=$GPU|completed=$DONE/$TOTAL|wait_s=$WAIT_TOTAL"
