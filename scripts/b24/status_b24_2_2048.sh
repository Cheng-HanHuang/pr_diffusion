#!/usr/bin/env bash
set -euo pipefail
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
LATEST="$OUTROOT/B24_2_2048_LATEST_RUN.txt"

RUNROOT=${1:-}
if [[ -z "$RUNROOT" ]]; then
  [[ -f "$LATEST" ]] || { echo "B24_2_2048_STATUS|NO_RUN"; exit 0; }
  RUNROOT=$(cat "$LATEST")
fi
[[ -f "$RUNROOT/LAUNCH.json" ]] || { echo "STOP|missing_launch:$RUNROOT/LAUNCH.json"; exit 2; }
PARENT=$(python - "$RUNROOT/LAUNCH.json" <<'PY'
import json,sys
from pathlib import Path
v=json.loads(Path(sys.argv[1]).read_text())
print(v['parent256_runroot'])
PY
)

echo "B24_2_2048_STATUS|runroot=$RUNROOT|parent256=$PARENT"
all_done=1
parent_ext=0
new_done=0
for GPU in 0 1 2 3; do
  PARENT_SUM="$PARENT/shard${GPU}/SHARD_COMPLETE.json"
  if [[ -f "$PARENT_SUM" ]]; then
    PSTATE=COMPLETE
  else
    PSTATE=INCOMPLETE
  fi
  pc=$(find "$PARENT/shard${GPU}" -name IMAGE_COMPLETE.json -type f 2>/dev/null | wc -l | tr -d ' ')
  parent_ext=$((parent_ext + pc))

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
  new_done=$((new_done + completed))
  last=$(tail -n 1 "$LOG" 2>/dev/null | tr '\n' ' ' || true)
  echo "GPU$GPU|parent256=$PSTATE|parent_ext_completed=$pc/48|state=$STATE|new_completed=$completed/448|last=$last"
  if [[ "$STATE" == STOPPED_WITHOUT_SUMMARY ]]; then
    tail -30 "$LOG" 2>/dev/null || true
  fi
done

# The fixed first-64 checkpoint plus completed 256-extension rows plus rows>=256.
cumulative=$((64 + parent_ext + new_done))
echo "CUMULATIVE_PROGRESS|valid_completed=$cumulative/2048|parent64=64|parent256_extension=$parent_ext/192|new2048_extension=$new_done/1792"
if [[ "$all_done" -eq 1 ]]; then
  echo "B24_2_2048_GATE|ALL_EXTENSION_SHARDS_COMPLETE|cumulative_target=2048"
else
  echo "B24_2_2048_GATE|RUNNING_OR_INCOMPLETE"
fi
