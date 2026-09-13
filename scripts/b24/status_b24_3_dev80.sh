#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
PTR="$OUTROOT/B24_3_DEV80_LATEST_RUN.txt"

[[ -f "$PTR" ]] || { echo "B24_3_DEV80_STATUS|state=NO_RUN|pointer=$PTR"; exit 0; }
RUN=$(cat "$PTR")
[[ -d "$RUN" ]] || { echo "B24_3_DEV80_STATUS|state=MISSING_RUN|run=$RUN"; exit 1; }

COMPLETED=$(find "$RUN/workers" -type f -name IMAGE_COMPLETE.json 2>/dev/null | wc -l)
FAILED_TASKS=0
for g in 0 1 2 3; do
  F="$RUN/workers/gpu${g}/FAILED_TASKS.tsv"
  if [[ -f "$F" ]]; then
    N=$(tail -n +2 "$F" | sed '/^[[:space:]]*$/d' | wc -l)
    FAILED_TASKS=$((FAILED_TASKS + N))
  fi
done

echo "B24_3_DEV80_STATUS|run=$RUN|completed=$COMPLETED/80|failed_records=$FAILED_TASKS"
ALL_WORKERS_PASS=1
for g in 0 1 2 3; do
  PIDFILE="$RUN/gpu${g}.pid"
  WC="$RUN/workers/gpu${g}/WORKER_COMPLETE.json"
  if [[ -f "$WC" ]]; then
    echo -n "GPU${g}|state=COMPLETE|summary="
    cat "$WC"
    STATUS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$WC")
    [[ "$STATUS" == "PASS" ]] || ALL_WORKERS_PASS=0
  elif [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "GPU${g}|state=RUNNING|pid=$(cat "$PIDFILE")"
    ALL_WORKERS_PASS=0
  else
    echo "GPU${g}|state=INCOMPLETE"
    ALL_WORKERS_PASS=0
  fi
done

if [[ "$COMPLETED" -eq 80 && "$ALL_WORKERS_PASS" -eq 1 ]]; then
  echo "B24_3_DEV80_GATE|ALL_80_COMPLETE"
  if [[ ! -f "$RUN/DEV80_SUMMARY.json" ]]; then
    CUDA_VISIBLE_DEVICES="" "$PY" "$REPO/scripts/b24/summarize_b24_3_dev80.py" --run "$RUN" > "$RUN/summarize.log" 2>&1
  fi
  "$PY" - "$RUN/DEV80_SUMMARY.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(json.dumps({
 'status':p['status'],
 'image_count':p['image_count'],
 'fresh_class_counts':p['fresh_class_counts'],
 'epp321_vs_np4_selected':p['epp321_vs_np4_selected'],
 'epp321_vs_np4_oracle':p['epp321_vs_np4_oracle'],
 'fresh_both_baselines_fail_count':p['fresh_both_baselines_fail_count'],
 'epp321_selected_rescues_where_both_fresh_baseline_oracles_fail':p['epp321_selected_rescues_where_both_fresh_baseline_oracles_fail'],
 'np4_selected_rescues_where_both_fresh_baseline_oracles_fail':p['np4_selected_rescues_where_both_fresh_baseline_oracles_fail'],
 'method_distributions':p['method_distributions'],
 'next':p['next'],
},sort_keys=True))
PY
  echo "SUMMARY_JSON|$RUN/DEV80_SUMMARY.json"
  echo "PER_IMAGE_CSV|$RUN/DEV80_PER_IMAGE.csv"
else
  echo "B24_3_DEV80_GATE|NOT_COMPLETE"
  if (( FAILED_TASKS > 0 )); then
    echo "FAILED_TASKS_BEGIN"
    for g in 0 1 2 3; do
      F="$RUN/workers/gpu${g}/FAILED_TASKS.tsv"
      [[ -f "$F" ]] && cat "$F"
    done
    echo "FAILED_TASKS_END"
  fi
fi
