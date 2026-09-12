#!/usr/bin/env bash
set -euo pipefail
ROOT=/egr/research-pac/huang248/outputs/pr_diffusion/b24
LATEST="$ROOT/LATEST_CONTROL_RUN.txt"
if [ ! -f "$LATEST" ]; then
  echo "B24_STATUS|NO_CONTROL_RUN"
  exit 0
fi
RUNROOT=$(cat "$LATEST")
echo "B24_STATUS|runroot=$RUNROOT"
for GPU in 0 1 2 3; do
  PIDFILE="$RUNROOT/pids/gpu${GPU}.pid"
  LOG="$RUNROOT/logs/gpu${GPU}.log"
  if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then STATE=RUNNING; else STATE=STOPPED; fi
  else
    PID=-
    if [ -f "$LOG" ]; then STATE=DRY_OR_NO_PID; else STATE=MISSING; fi
  fi
  LAST=$(tail -n 1 "$LOG" 2>/dev/null | tr '\n' ' ' || true)
  echo "GPU$GPU|state=$STATE|pid=$PID|last=$LAST"
done
