#!/usr/bin/env bash
set -euo pipefail
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
PTR="$OUTROOT/B24_3_EPP321_LATEST_RUN.txt"
[[ -f "$PTR" ]] || { echo "B24_3_EPP321_STATUS|state=NO_RUN"; exit 0; }
RUN=$(cat "$PTR")
[[ -d "$RUN" ]] || { echo "B24_3_EPP321_STATUS|state=MISSING_RUN|run=$RUN"; exit 1; }
TOTAL=16
DONE=$(find "$RUN/workers" -type f -path '*/methods/SMOKE_COMPLETE.json' 2>/dev/null | wc -l)
echo "B24_3_EPP321_STATUS|run=$RUN|completed=$DONE/$TOTAL"
for GPU in 0 1 2 3; do
  PIDFILE="$RUN/gpu${GPU}.pid"; WC="$RUN/workers/gpu${GPU}/WORKER_COMPLETE.json"
  if [[ -f "$WC" ]]; then
    echo "GPU${GPU}|state=COMPLETE|summary=$(cat "$WC")"
  elif [[ -f "$PIDFILE" ]]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then echo "GPU${GPU}|state=RUNNING|pid=$PID"; else echo "GPU${GPU}|state=EXITED_NO_SUMMARY|pid=$PID"; fi
  else
    echo "GPU${GPU}|state=NO_PID"
  fi
done
if (( DONE == TOTAL )); then
  echo "B24_3_EPP321_GATE|ALL_16_COMPLETE"
  CUDA_VISIBLE_DEVICES="" "$PY" "$REPO/scripts/b24/summarize_b24_3_epp321_refinement.py" --refinement-run "$RUN"
else
  echo "B24_3_EPP321_GATE|WAIT"
fi
