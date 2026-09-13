#!/usr/bin/env bash
set -u -o pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU ASSIGNMENT_DIR RUNROOT MIN_FREE_MIB" >&2
  exit 2
fi
GPU="$1"; ASSIGN="$2"; RUNROOT="$3"; MIN_FREE_MIB="$4"
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_b24"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
HARD_CEILING_MIB=52452

case "$GPU" in
  0) EXPECTED_UUID=GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3 ;;
  1) EXPECTED_UUID=GPU-883c037a-34d2-48c4-467f-9a352fd8fdff ;;
  2) EXPECTED_UUID=GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe ;;
  3) EXPECTED_UUID=GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a ;;
  *) echo "STOP|invalid_gpu:$GPU"; exit 2 ;;
esac
[[ "$MIN_FREE_MIB" =~ ^[0-9]+$ ]] || { echo "STOP|bad_min_free:$MIN_FREE_MIB"; exit 2; }
(( MIN_FREE_MIB <= HARD_CEILING_MIB )) || { echo "STOP|min_free_exceeds_hard_ceiling"; exit 2; }
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$UUID" == "$EXPECTED_UUID" ]] || { echo "STOP|gpu_uuid_mismatch:$GPU:$UUID"; exit 2; }
[[ -d "$ASSIGN" ]] || { echo "STOP|missing_assignment_dir:$ASSIGN"; exit 2; }

WORKER="$RUNROOT/workers/gpu${GPU}"
mkdir -p "$WORKER"
export PYTHONPATH="$REPO:$REPO/scripts/b22${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

mapfile -t TASKFILES < <(find "$ASSIGN" -maxdepth 1 -type f -name 'task*.json' | sort)
TOTAL=${#TASKFILES[@]}; DONE=0; FAILED=0; WAIT_TOTAL=0
FAILED_LIST="$WORKER/FAILED_TASKS.tsv"
printf 'task_json\timage_id\tstage\trc\n' > "$FAILED_LIST"

for TASKJSON in "${TASKFILES[@]}"; do
  read -r IMAGE_ID CLASS_LABEL OUTPUT_SUBDIR INPUT_MANIFEST ROLE_ROW < <(
    "$PY" - "$TASKJSON" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(p['image_id'],p['class_label'],p['output_subdir'],p['input_manifest'],p['role_row'])
PY
  )
  IMGDIR="$WORKER/$OUTPUT_SUBDIR"
  COMPLETE="$IMGDIR/PE3_COMPLETE.json"
  if [[ -f "$COMPLETE" ]]; then
    STATUS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$COMPLETE")
    if [[ "$STATUS" == "PASS" ]]; then
      DONE=$((DONE+1)); echo "PE3_REUSE|gpu=$GPU|image=$IMAGE_ID|done=$DONE/$TOTAL"; continue
    fi
    printf '%s\t%s\texisting_nonpass\t99\n' "$TASKJSON" "$IMAGE_ID" >> "$FAILED_LIST"
    FAILED=$((FAILED+1)); continue
  fi
  if [[ -e "$IMGDIR" ]]; then
    echo "PE3_SKIP_PARTIAL|gpu=$GPU|image=$IMAGE_ID|path=$IMGDIR"
    printf '%s\t%s\tpartial_requires_review\t98\n' "$TASKJSON" "$IMAGE_ID" >> "$FAILED_LIST"
    FAILED=$((FAILED+1)); continue
  fi
  [[ -f "$INPUT_MANIFEST" && -f "$ROLE_ROW" ]] || {
    printf '%s\t%s\tmissing_source\t97\n' "$TASKJSON" "$IMAGE_ID" >> "$FAILED_LIST"
    FAILED=$((FAILED+1)); continue
  }
  mkdir -p "$IMGDIR"
  cp "$TASKJSON" "$IMGDIR/task.json"

  WAIT_START=$(date +%s)
  while true; do
    FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ "$FREE" =~ ^[0-9]+$ ]] || { printf '%s\t%s\tfree_parse\t96\n' "$TASKJSON" "$IMAGE_ID" >> "$FAILED_LIST"; FAILED=$((FAILED+1)); break; }
    if (( FREE >= MIN_FREE_MIB )); then break; fi
    echo "PE3_WAIT_FOR_FIT|gpu=$GPU|image=$IMAGE_ID|free_mib=$FREE|required_mib=$MIN_FREE_MIB"
    sleep 60
  done
  if (( FREE < MIN_FREE_MIB )); then continue; fi
  NOW=$(date +%s); WAIT_TOTAL=$((WAIT_TOTAL + NOW - WAIT_START))

  echo "PE3_IMAGE_START|gpu=$GPU|image=$IMAGE_ID|class=$CLASS_LABEL|free_mib=$FREE"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$REPO/scripts/b24/run_b24_3_pe3_dev80.py" \
    --input-manifest "$INPUT_MANIFEST" \
    --role-row "$ROLE_ROW" \
    --output-root "$IMGDIR/methods" \
    --physical-gpu "$GPU" \
    --arms NP_PE3_SCORE,NP_PE3_RANDOM \
    > "$IMGDIR/run.log" 2>&1
  RC=$?
  set -e
  if (( RC != 0 )); then
    echo "PE3_TASK_FAIL|gpu=$GPU|image=$IMAGE_ID|rc=$RC"
    printf '%s\t%s\tmethod\t%s\n' "$TASKJSON" "$IMAGE_ID" "$RC" >> "$FAILED_LIST"
    FAILED=$((FAILED+1)); continue
  fi

  SUMMARY="$IMGDIR/methods/SMOKE_COMPLETE.json"
  set +e
  "$PY" - "$SUMMARY" "$TASKJSON" "$COMPLETE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); task=json.load(open(sys.argv[2]))
expected=['NP_PE3_SCORE','NP_PE3_RANDOM']
if s.get('status')!='PASS' or s.get('arms')!=expected: raise SystemExit(2)
for arm in expected:
    a=s['arm_results'][arm]
    if int(a['total_unet_evals'])!=8800 or int(a['proposal_unet_evals'])!=8796 or int(a['terminal_count'])!=4:
        raise SystemExit(3)
out={
  'schema_version':'b24.pe3-image-complete.v1','status':'PASS',
  'image_id':task['image_id'],'class_label':task['class_label'],
  'fresh_baseline_class':task['fresh_baseline_class'],
  'source_dev80_task':task['source_dev80_task'],
  'source_np4_result':task['np4_result'],
  'baseline_metrics':task['baseline_metrics'],
  'input_manifest':task['input_manifest'],
  'pe3_summary':sys.argv[1],
  'pe3_results':{arm:str(__import__('pathlib').Path(sys.argv[1]).parent/arm/'result.json') for arm in expected},
  'measurement_generation_performed':False,'confirmation_exposed':False,
}
open(sys.argv[3],'w').write(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
  VRC=$?
  set -e
  if (( VRC != 0 )); then
    printf '%s\t%s\tvalidation\t%s\n' "$TASKJSON" "$IMAGE_ID" "$VRC" >> "$FAILED_LIST"
    FAILED=$((FAILED+1)); continue
  fi
  DONE=$((DONE+1))
  echo "PE3_IMAGE_COMPLETE|gpu=$GPU|image=$IMAGE_ID|class=$CLASS_LABEL|done=$DONE/$TOTAL"
done

STATUS=PASS
(( FAILED == 0 && DONE == TOTAL )) || STATUS=FAIL
"$PY" - "$WORKER/WORKER_COMPLETE.json" "$GPU" "$TOTAL" "$DONE" "$FAILED" "$WAIT_TOTAL" "$MIN_FREE_MIB" "$STATUS" <<'PY'
import json,sys
out,gpu,total,done,failed,wait,gate,status=sys.argv[1:]
p={"schema_version":"b24.pe3-worker.v1","status":status,"gpu":int(gpu),"assigned":int(total),"completed":int(done),"failed":int(failed),"fit_wait_seconds":int(wait),"min_free_mib":int(gate),"measurement_generation_performed":False,"confirmation_exposed":False}
open(out,'w').write(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY

echo "B24_3_PE3_WORKER_$STATUS|gpu=$GPU|completed=$DONE/$TOTAL|failed=$FAILED|wait_s=$WAIT_TOTAL"
[[ "$STATUS" == "PASS" ]]
