#!/usr/bin/env bash
set -u -o pipefail

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
CTRL_PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
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
[[ -x "$CTRL_PY" && -x "$DAPS_PY" ]] || { echo "STOP|missing_python_env"; exit 2; }

WORKER="$RUNROOT/workers/gpu${GPU}"
mkdir -p "$WORKER"
export PYTHONPATH="$REPO:$REPO/scripts/b22${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

wait_for_fit() {
  local image="$1"
  local start now free
  start=$(date +%s)
  while true; do
    free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ "$free" =~ ^[0-9]+$ ]] || { echo "STOP|cannot_parse_free:$free"; return 1; }
    if (( free >= MIN_FREE_MIB )); then
      now=$(date +%s)
      WAIT_LAST=$((now-start))
      FREE_LAST="$free"
      return 0
    fi
    echo "DEV80_WAIT_FOR_FIT|gpu=$GPU|image=$image|free_mib=$free|required_mib=$MIN_FREE_MIB"
    sleep 60
  done
}

mapfile -t TASKFILES < <(find "$ASSIGN" -maxdepth 1 -type f -name 'task*.json' | sort)
TOTAL=${#TASKFILES[@]}
DONE=0
FAILED=0
WAIT_TOTAL=0
FAILED_LIST="$WORKER/FAILED_TASKS.tsv"
printf 'task_json\timage_id\tstage\trc\n' > "$FAILED_LIST"

for TASKJSON in "${TASKFILES[@]}"; do
  read -r TASK_INDEX IMAGE_ID CLASS_LABEL PILOT INPUT_MODE RUN_NP OUTPUT_SUBDIR SOURCE_INPUT < <(
    "$CTRL_PY" - "$TASKJSON" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(p['task_index'], p['image_id'], p['class_label'], int(bool(p['pilot16'])), p['input_mode'], int(bool(p['run_np'])), p['output_subdir'], p.get('source_input_manifest') or '-')
PY
  )
  IMGDIR="$WORKER/$OUTPUT_SUBDIR"
  COMPLETE="$IMGDIR/IMAGE_COMPLETE.json"
  if [[ -f "$COMPLETE" ]]; then
    STATUS=$("$CTRL_PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$COMPLETE")
    if [[ "$STATUS" == "PASS" ]]; then
      DONE=$((DONE+1))
      echo "DEV80_REUSE|gpu=$GPU|image=$IMAGE_ID|done=$DONE/$TOTAL"
      continue
    fi
    echo "STOP|existing_nonpass_completion:$COMPLETE"
    printf '%s\t%s\texisting_nonpass\t99\n' "$TASKJSON" "$IMAGE_ID" >> "$FAILED_LIST"
    FAILED=$((FAILED+1))
    continue
  fi
  if [[ -e "$IMGDIR" ]]; then
    echo "DEV80_SKIP_PARTIAL|gpu=$GPU|image=$IMAGE_ID|path=$IMGDIR"
    printf '%s\t%s\tpartial_requires_review\t98\n' "$TASKJSON" "$IMAGE_ID" >> "$FAILED_LIST"
    FAILED=$((FAILED+1))
    continue
  fi

  mkdir -p "$IMGDIR/logs"
  cp "$TASKJSON" "$IMGDIR/task.json"
  ROLE_ROW="$IMGDIR/role_row.csv"
  if ! "$CTRL_PY" - "$TASKJSON" "$ROLE_ROW" <<'PY'
import csv,json,sys
p=json.load(open(sys.argv[1])); row=p['role_row']
with open(sys.argv[2],'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(row.keys()),lineterminator='\n'); w.writeheader(); w.writerow(row)
PY
  then
    rc=$?
    printf '%s\t%s\trole_row\t%s\n' "$TASKJSON" "$IMAGE_ID" "$rc" >> "$FAILED_LIST"
    FAILED=$((FAILED+1)); continue
  fi

  INPUT_MANIFEST=""
  if [[ "$INPUT_MODE" == "REUSE_PILOT16" ]]; then
    INPUT_MANIFEST="$SOURCE_INPUT"
    if [[ ! -f "$INPUT_MANIFEST" ]]; then
      echo "DEV80_TASK_FAIL|gpu=$GPU|image=$IMAGE_ID|stage=source_input_missing"
      printf '%s\t%s\tsource_input_missing\t97\n' "$TASKJSON" "$IMAGE_ID" >> "$FAILED_LIST"
      FAILED=$((FAILED+1)); continue
    fi
    "$CTRL_PY" - "$INPUT_MANIFEST" "$IMGDIR/INPUT_REUSE.json" <<'PY'
import hashlib,json,sys
p=sys.argv[1]
h=hashlib.sha256(open(p,'rb').read()).hexdigest()
open(sys.argv[2],'w').write(json.dumps({'source_input_manifest':p,'source_manifest_sha256':h,'measurement_generation_performed':False},indent=2,sort_keys=True)+'\n')
PY
  elif [[ "$INPUT_MODE" == "GENERATE_FROZEN_DEV" ]]; then
    if ! wait_for_fit "$IMAGE_ID"; then
      printf '%s\t%s\tinput_wait\t96\n' "$TASKJSON" "$IMAGE_ID" >> "$FAILED_LIST"
      FAILED=$((FAILED+1)); continue
    fi
    WAIT_TOTAL=$((WAIT_TOTAL + WAIT_LAST))
    MEAS_SEED=$("$CTRL_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["measurement_seed"])' "$TASKJSON")
    if CUDA_VISIBLE_DEVICES="$GPU" "$DAPS_PY" "$REPO/scripts/b24/generate_b24_locked_input.py" \
      --image-id "$IMAGE_ID" --measurement-seed "$MEAS_SEED" --output "$IMGDIR/input" \
      > "$IMGDIR/logs/input_generation.log" 2>&1; then
      INPUT_MANIFEST="$IMGDIR/input/input_manifest.json"
    else
      rc=$?
      echo "DEV80_TASK_FAIL|gpu=$GPU|image=$IMAGE_ID|stage=input_generation|rc=$rc"
      printf '%s\t%s\tinput_generation\t%s\n' "$TASKJSON" "$IMAGE_ID" "$rc" >> "$FAILED_LIST"
      FAILED=$((FAILED+1)); continue
    fi
  else
    echo "DEV80_TASK_FAIL|gpu=$GPU|image=$IMAGE_ID|stage=bad_input_mode:$INPUT_MODE"
    printf '%s\t%s\tbad_input_mode\t95\n' "$TASKJSON" "$IMAGE_ID" >> "$FAILED_LIST"
    FAILED=$((FAILED+1)); continue
  fi

  if ! wait_for_fit "$IMAGE_ID"; then
    printf '%s\t%s\texecution_wait\t94\n' "$TASKJSON" "$IMAGE_ID" >> "$FAILED_LIST"
    FAILED=$((FAILED+1)); continue
  fi
  WAIT_TOTAL=$((WAIT_TOTAL + WAIT_LAST))
  echo "DEV80_IMAGE_START|gpu=$GPU|task=$TASK_INDEX|image=$IMAGE_ID|class=$CLASS_LABEL|pilot=$PILOT|run_np=$RUN_NP|free_mib=$FREE_LAST"

  if CUDA_VISIBLE_DEVICES="$GPU" "$CTRL_PY" "$REPO/scripts/b24/run_b24_3_dev80_image.py" \
      --task-json "$TASKJSON" \
      --input-manifest "$INPUT_MANIFEST" \
      --role-row "$ROLE_ROW" \
      --task-root "$IMGDIR" \
      --physical-gpu "$GPU" \
      --min-free-mib "$MIN_FREE_MIB" \
      > "$IMGDIR/run.log" 2>&1; then
    if [[ ! -f "$COMPLETE" ]]; then
      echo "DEV80_TASK_FAIL|gpu=$GPU|image=$IMAGE_ID|stage=missing_completion"
      printf '%s\t%s\tmissing_completion\t93\n' "$TASKJSON" "$IMAGE_ID" >> "$FAILED_LIST"
      FAILED=$((FAILED+1)); continue
    fi
    DONE=$((DONE+1))
    echo "DEV80_IMAGE_COMPLETE|gpu=$GPU|image=$IMAGE_ID|class=$CLASS_LABEL|done=$DONE/$TOTAL"
  else
    rc=$?
    echo "DEV80_TASK_FAIL|gpu=$GPU|image=$IMAGE_ID|stage=image_execution|rc=$rc|log=$IMGDIR/run.log"
    printf '%s\t%s\timage_execution\t%s\n' "$TASKJSON" "$IMAGE_ID" "$rc" >> "$FAILED_LIST"
    FAILED=$((FAILED+1))
    continue
  fi
done

STATUS=PASS
if (( FAILED > 0 || DONE != TOTAL )); then STATUS=PARTIAL_FAILURE; fi
"$CTRL_PY" - "$WORKER/WORKER_COMPLETE.json" "$GPU" "$TOTAL" "$DONE" "$FAILED" "$WAIT_TOTAL" "$MIN_FREE_MIB" "$STATUS" "$FAILED_LIST" <<'PY'
import json,sys
out,gpu,total,done,failed,wait,gate,status,failed_list=sys.argv[1:]
p={
 'schema_version':'b24.dev80-worker.v1','status':status,'gpu':int(gpu),
 'assigned':int(total),'completed':int(done),'failed':int(failed),
 'fit_wait_seconds':int(wait),'min_free_mib':int(gate),
 'failed_tasks_tsv':failed_list,'confirmation_exposed':False,
}
open(out,'w').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY

echo "B24_3_DEV80_WORKER_DONE|gpu=$GPU|status=$STATUS|completed=$DONE/$TOTAL|failed=$FAILED|wait_s=$WAIT_TOTAL"
if [[ "$STATUS" != "PASS" ]]; then exit 7; fi
