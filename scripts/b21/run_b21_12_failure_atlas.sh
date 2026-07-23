#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
PYTHON_BIN=${PYTHON_BIN:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
IMAGE_ROOT=${IMAGE_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
B21_11=${B21_11:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_11_fresh2_final_val100_meas5401}
ROWS=${ROWS:-$B21_11/analysis_theta0.7/fresh2_final_rows.csv}
OUT=${OUT:-$B21_11/b21_12_failure_atlas}
REPORT=${REPORT:-$OUT/b21_12_failure_atlas.md}
TILE_SIZE=${TILE_SIZE:-192}
GOOD_THRESHOLD=${GOOD_THRESHOLD:-25.0}
ANALYZER="$REPO/scripts/b21/analyze_b21_12_failure_atlas.py"

cd "$REPO"
mkdir -p "$OUT"

[[ -x "$PYTHON_BIN" ]] || { echo "[fatal] PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$ANALYZER" ]] || { echo "[fatal] missing analyzer: $ANALYZER" >&2; exit 2; }
[[ -s "$ROWS" ]] || { echo "[fatal] missing B21.11 rows: $ROWS" >&2; exit 2; }
[[ "$(wc -l < "$ROWS")" -eq 101 ]] || { echo "[fatal] expected 101 CSV lines in $ROWS" >&2; exit 2; }
[[ "$GOOD_THRESHOLD" == "25.0" ]] || { echo "[fatal] GOOD_THRESHOLD is frozen to 25.0" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=""
set -o pipefail

"$PYTHON_BIN" "$ANALYZER" \
  --rows "$ROWS" \
  --outdir "$OUT" \
  --image-root "$IMAGE_ROOT" \
  --tile-size "$TILE_SIZE" \
  --good-threshold "$GOOD_THRESHOLD" \
  --report "$REPORT" \
  2>&1 | tee "$OUT/analyzer_stdout.txt"

status=${PIPESTATUS[0]}
if [[ "$status" -ne 0 ]]; then
  echo "[fatal] B21.12 analyzer failed with exit code $status" >&2
  exit "$status"
fi

for path in \
  "$OUT/failure_atlas_summary.json" \
  "$OUT/failure_atlas_rows.csv" \
  "$OUT/persistent_failure_rows.csv" \
  "$OUT/fresh2_rescue_rows.csv" \
  "$OUT/protected_fresh1_success_rows.csv" \
  "$OUT/manual_failure_labels_template.csv" \
  "$OUT/sheets/persistent_failure.png" \
  "$OUT/sheets/fresh2_rescue.png" \
  "$OUT/sheets/protected_fresh1_success.png" \
  "$REPORT"
do
  [[ -s "$path" ]] || { echo "[fatal] missing artifact: $path" >&2; exit 3; }
done

case_count=$(find "$OUT/cases" -maxdepth 1 -type f -name '*.png' | wc -l)
[[ "$case_count" -eq 27 ]] || { echo "[fatal] expected 27 case panels, found $case_count" >&2; exit 3; }

echo "[done] B21.12 zero-GPU atlas complete"
echo "[summary] $OUT/failure_atlas_summary.json"
echo "[report] $REPORT"
