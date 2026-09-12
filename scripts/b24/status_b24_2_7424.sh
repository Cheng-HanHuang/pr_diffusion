#!/usr/bin/env bash
set -euo pipefail
ROOT=/egr/research-pac/huang248
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
LATEST="$OUTROOT/B24_2_7424_LATEST_RUN.txt"

count_completions() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return 0
  fi
  find "$dir" -name IMAGE_COMPLETE.json -type f 2>/dev/null | wc -l | tr -d ' '
}

RUNROOT=${1:-}
if [[ -z "$RUNROOT" ]]; then
  [[ -f "$LATEST" ]] || { echo "B24_2_7424_STATUS|NO_RUN"; exit 0; }
  RUNROOT=$(cat "$LATEST")
fi
[[ -f "$RUNROOT/LAUNCH.json" ]] || { echo "STOP|missing_launch:$RUNROOT/LAUNCH.json"; exit 2; }
PARENT=$("$PY" - "$RUNROOT/LAUNCH.json" <<'PY'
import json,sys
from pathlib import Path
v=json.loads(Path(sys.argv[1]).read_text())
print(v['parent6144_runroot'])
PY
)

echo "B24_2_7424_STATUS|runroot=$RUNROOT|parent6144=$PARENT"
all_done=1
new_done=0
for GPU in 0 1 2 3; do
  PARENT_SUM="$PARENT/shard${GPU}/SHARD_COMPLETE.json"
  if [[ -f "$PARENT_SUM" ]]; then PSTATE=COMPLETE; else PSTATE=INCOMPLETE; fi

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
  completed=$(count_completions "$RUNROOT/shard${GPU}")
  new_done=$((new_done + completed))
  last=$(tail -n 1 "$LOG" 2>/dev/null | tr '\n' ' ' || true)
  echo "GPU$GPU|parent6144=$PSTATE|state=$STATE|new_completed=$completed/320|last=$last"
  if [[ "$STATE" == STOPPED_WITHOUT_SUMMARY ]]; then tail -30 "$LOG" 2>/dev/null || true; fi
done

cumulative=$((6144 + new_done))
echo "CUMULATIVE_PROGRESS|valid_completed=$cumulative/7424|parent6144=6144|new7424_extension=$new_done/1280"
if [[ "$all_done" -eq 1 ]]; then
  echo "B24_2_7424_GATE|ALL_EXTENSION_SHARDS_COMPLETE|cumulative_target=7424"
else
  echo "B24_2_7424_GATE|RUNNING_OR_INCOMPLETE"
fi
