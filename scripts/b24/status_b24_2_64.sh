#!/usr/bin/env bash
set -euo pipefail
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
LATEST="$OUTROOT/B24_2_64_LATEST_RUN.txt"

RUNROOT=${1:-}
if [[ -z "$RUNROOT" ]]; then
  [[ -f "$LATEST" ]] || { echo "B24_2_64_STATUS|NO_RUN"; exit 0; }
  RUNROOT=$(cat "$LATEST")
fi

echo "B24_2_64_STATUS|runroot=$RUNROOT"
all_done=1
for GPU in 0 1 2 3; do
  PIDFILE="$RUNROOT/pids/gpu${GPU}.pid"
  LOG="$RUNROOT/logs/gpu${GPU}.log"
  SHARD="$RUNROOT/shard${GPU}/SHARD_COMPLETE.json"
  if [[ -f "$SHARD" ]]; then
    STATE=COMPLETE
  elif [[ -f "$PIDFILE" ]]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then STATE=RUNNING; all_done=0; else STATE=STOPPED_WITHOUT_SUMMARY; all_done=0; fi
  else
    PID=-; STATE=MISSING; all_done=0
  fi
  completed=$(find "$RUNROOT/shard${GPU}" -name IMAGE_COMPLETE.json -type f 2>/dev/null | wc -l | tr -d ' ')
  last=$(tail -n 1 "$LOG" 2>/dev/null | tr '\n' ' ' || true)
  echo "GPU$GPU|state=$STATE|completed=$completed/16|last=$last"
  if [[ "$STATE" == STOPPED_WITHOUT_SUMMARY ]]; then
    tail -40 "$LOG" 2>/dev/null || true
  fi
done

if [[ "$all_done" -eq 1 ]]; then
  if [[ ! -f "$RUNROOT/B24_2_64_SUMMARY.json" ]]; then
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
    "$PY" "$REPO/scripts/b24/analyze_b24_2_64.py" --runroot "$RUNROOT"
  fi
  cat "$RUNROOT/B24_2_64_SUMMARY.json"
  echo "B24_2_64_GATE|COMPLETE_FOR_PREVALENCE_REVIEW"
else
  echo "B24_2_64_GATE|RUNNING_OR_INCOMPLETE"
fi
