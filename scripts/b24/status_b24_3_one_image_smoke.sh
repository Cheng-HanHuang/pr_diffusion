#!/usr/bin/env bash
set -euo pipefail
ROOT=/egr/research-pac/huang248
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
PTR="$OUTROOT/B24_3_ONE_IMAGE_LATEST_RUN.txt"
[[ -f "$PTR" ]] || { echo "STOP|missing_pointer:$PTR"; exit 2; }
RUN=$(cat "$PTR")
[[ -d "$RUN" ]] || { echo "STOP|missing_run:$RUN"; exit 2; }
PID=""
[[ -f "$RUN/runner.pid" ]] && PID=$(cat "$RUN/runner.pid")
COMPLETE="$RUN/methods/SMOKE_COMPLETE.json"
if [[ -f "$COMPLETE" ]]; then
  echo "B24_3_ONE_IMAGE_STATUS|state=COMPLETE|run=$RUN|pid=$PID"
  /egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python - "$COMPLETE" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(json.dumps({
  'status':p.get('status'),
  'image_id':p.get('image_id'),
  'class_label':p.get('class_label'),
  'arms':p.get('arms'),
  'max_b24_process_gpu_mib':p.get('max_b24_process_gpu_mib'),
  'max_torch_reserved_mib':p.get('max_torch_reserved_mib'),
  'recommended_pilot_min_free_mib':p.get('recommended_pilot_min_free_mib'),
  'total_wall_seconds_excluding_model_load':p.get('total_wall_seconds_excluding_model_load'),
  'arm_results':p.get('arm_results'),
},sort_keys=True))
PY
  exit 0
fi
if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
  STATE=RUNNING
else
  STATE=STOPPED_WITHOUT_SUMMARY
fi
echo "B24_3_ONE_IMAGE_STATUS|state=$STATE|run=$RUN|pid=$PID"
if [[ -f "$RUN/run.log" ]]; then
  echo "===== run.log tail ====="
  tail -40 "$RUN/run.log"
fi
