#!/usr/bin/env bash
# CPU-only scientific analysis and visual failure atlas for a validated B22.2 run.
set -u

REPO_ROOT=${B22_REPO_ROOT:-/egr/research-pac/huang248/pr_diffusion_b22}
DAPS_PY=${B22_DAPS_PY:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
RUN_ROOT=${1:-}
OUTPUT_ROOT=${B22_OUTPUT_ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/b22_baselines}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${2:-$OUTPUT_ROOT/B22_3_scientific_analysis_$STAMP}
LOG=${OUT}_launcher.log
ARCHIVE=${OUT}.tar.gz

if [[ -z "$RUN_ROOT" ]]; then
  echo "usage: bash scripts/b22/run_b22_3_analysis.sh <B22_2_run_root> [output_dir]" >&2
  exit 2
fi
if [[ ! -d "$RUN_ROOT/full" ]]; then
  echo "STOP: missing full stage under $RUN_ROOT" >&2
  exit 1
fi
if [[ -e "$OUT" || -e "$OUT.tmp" || -e "$ARCHIVE" ]]; then
  echo "STOP: refusing to overwrite $OUT, $OUT.tmp, or $ARCHIVE" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUT")"

run_step() {
  local name=$1
  shift
  echo "[RUN ] $name"
  "$@"
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[OK  ] $name"
  else
    echo "[FAIL] $name rc=$rc"
  fi
  return "$rc"
}

{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=$REPO_ROOT"
  echo "repo_head=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "run_root=$RUN_ROOT"
  echo "analysis_out=$OUT"

  export PYTHONPATH="$REPO_ROOT/scripts/b22${PYTHONPATH:+:$PYTHONPATH}"

  run_step scientific-analysis \
    "$DAPS_PY" -u "$REPO_ROOT/scripts/b22/analyze_b22_3_panel.py" \
      --run_root "$RUN_ROOT" \
      --out "$OUT" \
      --bootstrap_reps 100000 \
      --bootstrap_seed 5401 || exit 1

  run_step failure-atlas \
    "$DAPS_PY" -u "$REPO_ROOT/scripts/b22/render_b22_3_failure_atlas.py" \
      --run_root "$RUN_ROOT" \
      --analysis_dir "$OUT" || exit 1

  cat > "$OUT/FINAL_STATUS.txt" <<EOF
B22.3 CPU scientific analysis: COMPLETE
input_run_root=$RUN_ROOT
analysis_out=$OUT
GPU work performed=NO
scientific interpretation status=PENDING EXECUTION-LEAD VISUAL REVIEW
EOF

  run_step archive \
    tar -C "$(dirname "$OUT")" -czf "$ARCHIVE" "$(basename "$OUT")" || exit 1

  echo "B22.3 analysis completed."
  echo "analysis_out=$OUT"
  echo "return_archive=$ARCHIVE"
} > "$LOG" 2>&1
rc=$?
cat "$LOG"
exit "$rc"
