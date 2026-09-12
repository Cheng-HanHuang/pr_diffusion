#!/usr/bin/env bash
set -euo pipefail

# Generic B24 control plane. Scientific B24.2 dispatch stays fail-closed until
# the dedicated baseline backend is committed and authorized.

ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  echo "usage: $0 --stage B24.0|B24.1|B24.2 --manifest PATH --mode serial|batched [--dry-run]" >&2
}

STAGE=""
MANIFEST=""
MODE=""
DRY=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) usage; exit 2 ;;
  esac
done

[ -n "$STAGE" ] && [ -n "$MANIFEST" ] && [ -n "$MODE" ] || { usage; exit 2; }
[ "$MODE" = serial ] || [ "$MODE" = batched ] || { echo "STOP: bad mode" >&2; exit 2; }
[ -x "$PY" ] || { echo "STOP: missing prdiff_ffhq python: $PY" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "STOP: manifest does not exist: $MANIFEST" >&2; exit 4; }

if [ "$STAGE" = "B24.0" ] && [ "$DRY" -ne 1 ]; then
  echo "STOP: B24.0 permits dry rendering only" >&2
  exit 3
fi

# B24.1 has its own dedicated validated smoke launcher. B24.2 must use the
# dedicated scale launcher once committed; this scaffold must not pretend to
# launch scientific work.
if [ "$DRY" -ne 1 ] && { [ "$STAGE" = "B24.1" ] || [ "$STAGE" = "B24.2" ]; }; then
  echo "STOP: generic B24 control-plane scientific dispatch is disabled; use the stage-specific validated launcher" >&2
  exit 5
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="$OUTROOT/control/${STAGE}_${MODE}_${STAMP}"
mkdir -p "$RUNROOT/logs" "$RUNROOT/pids"

for GPU in 0 1 2 3; do
  CMD=(
    "$PY" "$REPO/scripts/b24/run_b24_worker.py"
    --manifest "$MANIFEST"
    --shard "$GPU"
    --gpu "$GPU"
    --mode "$MODE"
    --stage "$STAGE"
    --output-root "$RUNROOT/shard${GPU}"
  )
  [ "$DRY" -eq 1 ] && CMD+=(--dry-run)

  if [ "$DRY" -eq 1 ]; then
    "${CMD[@]}" >"$RUNROOT/logs/gpu${GPU}.log" 2>&1
  else
    nohup "${CMD[@]}" >"$RUNROOT/logs/gpu${GPU}.log" 2>&1 &
    PID=$!
    printf '%s\n' "$PID" >"$RUNROOT/pids/gpu${GPU}.pid"
  fi
done

printf '%s\n' "$RUNROOT" >"$OUTROOT/LATEST_CONTROL_RUN.txt"
if [ "$DRY" -eq 1 ]; then
  echo "B24_CONTROL_DRY_RUN|stage=$STAGE|mode=$MODE|runroot=$RUNROOT|shards=4"
else
  echo "B24_CONTROL_LAUNCHED|stage=$STAGE|mode=$MODE|runroot=$RUNROOT|shards=4"
fi
