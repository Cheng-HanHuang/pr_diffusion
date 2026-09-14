#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then echo "usage: $0 PE3_RUN DEV80_RUN GPU" >&2; exit 2; fi
PE3RUN="$1"; DEV80RUN="$2"; GPU="$3"
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_b24"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
OUT="$PE3RUN/compute_audit"
mkdir -p "$OUT"

case "$GPU" in 0|1|2|3) ;; *) echo "STOP|bad_gpu:$GPU"; exit 2;; esac
WORKER="$PE3RUN/workers/gpu${GPU}/WORKER_COMPLETE.json"
PIDFILE="$PE3RUN/gpu${GPU}.pid"

while [[ ! -f "$WORKER" ]]; do
  if [[ -f "$PIDFILE" ]]; then
    PID=$(cat "$PIDFILE")
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "STOP|pe3_worker_ended_before_completion|gpu=$GPU|pid=$PID"; exit 3
    fi
  fi
  echo "FLOP_AUDIT_WAIT_PE3_WORKER|gpu=$GPU"
  sleep 60
done
STATUS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$WORKER")
[[ "$STATUS" == "PASS" ]] || { echo "STOP|pe3_worker_nonpass:$WORKER:$STATUS"; exit 3; }

while true; do
  FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "$FREE" =~ ^[0-9]+$ ]] || { echo "STOP|cannot_parse_free:$FREE"; exit 3; }
  (( FREE >= 10240 )) && break
  echo "FLOP_AUDIT_WAIT_FOR_FIT|gpu=$GPU|free_mib=$FREE|required_mib=10240"
  sleep 60
done

export PYTHONPATH="$REPO:$REPO/scripts/b22${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$REPO/scripts/b24/run_b24_3_cross_family_flop_audit.py" \
  --dev80-run "$DEV80RUN" --gpu "$GPU" --output "$OUT/run" \
  > "$OUT/audit.log" 2>&1

[[ -f "$OUT/run/CROSS_FAMILY_FLOP_AUDIT.json" ]] || { echo "STOP|missing_flop_audit_json"; exit 4; }
"$PY" - "$OUT/run/CROSS_FAMILY_FLOP_AUDIT.json" "$OUT/AUDIT_COMPLETE.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
if p.get('status')!='PASS' or p.get('confirmation_exposed') is not False: raise SystemExit(2)
out={'schema_version':'b24.cross-family-flop-audit-complete.v1','status':'PASS','audit_json':sys.argv[1],'calibration_image_id':p['calibration_image_id'],'confirmation_exposed':False}
open(sys.argv[2],'w').write(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY

echo "B24_3_FLOP_AUDIT_PASS|gpu=$GPU|json=$OUT/run/CROSS_FAMILY_FLOP_AUDIT.json"
