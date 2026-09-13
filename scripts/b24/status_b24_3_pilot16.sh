#!/usr/bin/env bash
set -euo pipefail
ROOT=/egr/research-pac/huang248
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
PTR="$OUTROOT/B24_3_PILOT16_LATEST_RUN.txt"
[[ -f "$PTR" ]] || { echo "STOP|missing_pointer:$PTR"; exit 2; }
RUN=$(cat "$PTR")
[[ -d "$RUN" ]] || { echo "STOP|missing_run:$RUN"; exit 2; }

CAL=0
if [[ -f "$RUN/PILOT16_MANIFEST.json" ]]; then
  CAL=1
fi
NEW_DONE=$(find "$RUN/workers" -type f -path '*/methods/SMOKE_COMPLETE.json' 2>/dev/null | wc -l || true)
TOTAL_DONE=$((CAL + NEW_DONE))

echo "B24_3_PILOT16_STATUS|run=$RUN|completed=$TOTAL_DONE/16|calibration_reuse=$CAL|new_completed=$NEW_DONE/15"
ALL_WORKERS=1
for GPU in 0 1 2 3; do
  PID=""; [[ -f "$RUN/gpu${GPU}.pid" ]] && PID=$(cat "$RUN/gpu${GPU}.pid")
  WC="$RUN/workers/gpu${GPU}/WORKER_COMPLETE.json"
  if [[ -f "$WC" ]]; then
    echo "GPU${GPU}|state=COMPLETE|pid=$PID|summary=$(cat "$WC")"
  elif [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
    echo "GPU${GPU}|state=RUNNING|pid=$PID"
    ALL_WORKERS=0
  elif [[ -f "$RUN/gpu${GPU}.pid" ]]; then
    echo "GPU${GPU}|state=STOPPED_WITHOUT_SUMMARY|pid=$PID"
    ALL_WORKERS=0
  else
    ASSIGN="$RUN/assignments/gpu${GPU}"
    COUNT=0; [[ -d "$ASSIGN" ]] && COUNT=$(find "$ASSIGN" -maxdepth 1 -type f -name 'row*.csv' | wc -l)
    if (( COUNT == 0 )); then echo "GPU${GPU}|state=NO_ASSIGNMENT"; else echo "GPU${GPU}|state=NOT_LAUNCHED|assigned=$COUNT"; ALL_WORKERS=0; fi
  fi
done

if (( TOTAL_DONE == 16 && ALL_WORKERS == 1 )); then
  echo "B24_3_PILOT16_GATE|ALL_16_COMPLETE"
else
  echo "B24_3_PILOT16_GATE|RUNNING_OR_INCOMPLETE"
  for GPU in 0 1 2 3; do
    if [[ -f "$RUN/gpu${GPU}.log" ]]; then
      echo "===== GPU $GPU tail ====="
      tail -12 "$RUN/gpu${GPU}.log"
    fi
  done
fi
