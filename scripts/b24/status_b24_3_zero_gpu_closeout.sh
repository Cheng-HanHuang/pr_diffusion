#!/usr/bin/env bash
set -euo pipefail
ROOT=/egr/research-pac/huang248
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
PTR="$OUTROOT/B24_3_ZERO_GPU_CLOSEOUT_LATEST_RUN.txt"
[[ -f "$PTR" ]] || { echo "B24_3_ZERO_GPU_CLOSEOUT_STATUS|state=NO_RUN|pointer=$PTR"; exit 0; }
RUN=$(cat "$PTR")
[[ -d "$RUN" ]] || { echo "B24_3_ZERO_GPU_CLOSEOUT_STATUS|state=MISSING_RUN|run=$RUN"; exit 1; }
C="$RUN/B24_3_DEV_CLOSEOUT.json"
[[ -f "$C" ]] || { echo "B24_3_ZERO_GPU_CLOSEOUT_STATUS|state=INCOMPLETE|run=$RUN"; exit 1; }
python - "$C" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(f"B24_3_ZERO_GPU_CLOSEOUT_STATUS|state={p.get('status')}|decision={p.get('decision')}|confirmation_exposed={p.get('confirmation_exposed')}|gpu_work={p.get('gpu_work_performed')}")
print(json.dumps(p,sort_keys=True))
PY
echo "REPORT|$RUN/B24_3_DEV_CLOSEOUT.md"
echo "FRESH2|$RUN/FRESH2_SUMMARY.json"
echo "COMPLEMENTARITY|$RUN/COMPLEMENTARITY.json"
echo "COMPUTE|$RUN/COMPUTE_CLOSEOUT.json"
