#!/usr/bin/env bash
set -euo pipefail
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
RUNROOT="${1:-}"
if [[ -z "$RUNROOT" ]]; then
  [[ -s "$OUTROOT/B24_1_LATEST_RUN.txt" ]] || { echo "STOP|no B24.1 latest run"; exit 2; }
  RUNROOT=$(cat "$OUTROOT/B24_1_LATEST_RUN.txt")
fi
[[ -d "$RUNROOT" ]] || { echo "STOP|missing runroot:$RUNROOT"; exit 2; }

echo "B24_1_STATUS|runroot=$RUNROOT"
RUNNING=0
for method in daps sitcom; do
  pidfile="$RUNROOT/pids/$method.pid"
  pid=""
  [[ -s "$pidfile" ]] && pid=$(cat "$pidfile")
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "RUNNING|method=$method|pid=$pid"
    RUNNING=1
  elif [[ -s "$RUNROOT/$method/METHOD_SUMMARY.json" ]]; then
    echo "COMPLETE|method=$method|pid=${pid:-UNKNOWN}"
  else
    echo "STOPPED_WITHOUT_SUMMARY|method=$method|pid=${pid:-UNKNOWN}|log=$RUNROOT/logs/$method.log"
    tail -n 20 "$RUNROOT/logs/$method.log" 2>/dev/null || true
  fi
done

if (( RUNNING )); then
  echo "B24_1_NOT_READY_FOR_GATE"
  exit 0
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
set +e
"$PY" "$REPO/scripts/b24/analyze_b24_1_smoke.py" --run-root "$RUNROOT"
RC=$?
set -e
if [[ $RC -eq 0 ]]; then
  echo "B24_1_GATE|PASS|next=64_IMAGE_BASELINE_PREPARATION"
elif [[ $RC -eq 3 ]]; then
  echo "B24_1_GATE|INCOMPLETE"
else
  echo "B24_1_GATE|FAIL|review_before_baseline"
fi
exit "$RC"
