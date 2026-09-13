#!/usr/bin/env bash
set -euo pipefail
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
PTR="$OUTROOT/B24_3_PE3_LATEST_RUN.txt"

[[ -f "$PTR" ]] || { echo "B24_3_PE3_STATUS|state=NO_RUN|pointer=$PTR"; exit 0; }
RUN=$(cat "$PTR"); [[ -d "$RUN" ]] || { echo "B24_3_PE3_STATUS|state=MISSING_RUN|run=$RUN"; exit 1; }
COMPLETED=$(find "$RUN/workers" -type f -name PE3_COMPLETE.json 2>/dev/null | wc -l)
FAILED=0
for g in 0 1 2 3; do
  F="$RUN/workers/gpu${g}/FAILED_TASKS.tsv"
  if [[ -f "$F" ]]; then N=$(tail -n +2 "$F" | sed '/^[[:space:]]*$/d' | wc -l); FAILED=$((FAILED+N)); fi
done
echo "B24_3_PE3_STATUS|run=$RUN|completed=$COMPLETED/80|failed_records=$FAILED"

WORKERS_PASS=1
for g in 0 1 2 3; do
  W="$RUN/workers/gpu${g}/WORKER_COMPLETE.json"; PIDF="$RUN/gpu${g}.pid"
  if [[ -f "$W" ]]; then
    S=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$W")
    echo "GPU${g}|state=COMPLETE|summary=$(cat "$W")"
    [[ "$S" == "PASS" ]] || WORKERS_PASS=0
  elif [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "GPU${g}|state=RUNNING|pid=$(cat "$PIDF")"
    WORKERS_PASS=0
  else
    echo "GPU${g}|state=STOPPED_OR_NOT_STARTED"
    WORKERS_PASS=0
  fi
done

AUDIT_COMPLETE="$RUN/compute_audit/AUDIT_COMPLETE.json"
AUDIT_PASS=0
if [[ -f "$AUDIT_COMPLETE" ]]; then
  S=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$AUDIT_COMPLETE")
  echo "FLOP_AUDIT|state=COMPLETE|summary=$(cat "$AUDIT_COMPLETE")"
  [[ "$S" == "PASS" ]] && AUDIT_PASS=1
elif [[ -f "$RUN/flop_audit.pid" ]] && kill -0 "$(cat "$RUN/flop_audit.pid")" 2>/dev/null; then
  echo "FLOP_AUDIT|state=RUNNING|pid=$(cat "$RUN/flop_audit.pid")"
else
  echo "FLOP_AUDIT|state=STOPPED_OR_NOT_STARTED"
  [[ -f "$RUN/flop_audit.log" ]] && tail -n 30 "$RUN/flop_audit.log"
fi

if [[ "$COMPLETED" -eq 80 && "$FAILED" -eq 0 && "$WORKERS_PASS" -eq 1 && "$AUDIT_PASS" -eq 1 ]]; then
  if [[ ! -f "$RUN/PE3_DEV80_SUMMARY.json" ]]; then
    "$PY" "$REPO/scripts/b24/summarize_b24_3_pe3.py" --run "$RUN"
  fi
  echo "B24_3_PE3_GATE|ALL_80_AND_FLOP_AUDIT_COMPLETE"
  cat "$RUN/PE3_DEV80_SUMMARY.json"
  echo "FLOP_AUDIT_JSON|$RUN/compute_audit/run/CROSS_FAMILY_FLOP_AUDIT.json"
  cat "$RUN/compute_audit/run/CROSS_FAMILY_FLOP_AUDIT.json"
else
  echo "B24_3_PE3_GATE|NOT_COMPLETE"
fi
