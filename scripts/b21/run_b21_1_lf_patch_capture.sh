#!/usr/bin/env bash
set -euo pipefail

# B21.1 LF guidance patch capture.
# This is a documentation/diff-capture task; it does not launch reconstruction jobs.

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
OUT=${OUT:-$B21_BASE/B21_1_lf_patch_capture}
UPSTREAM_REF=${UPSTREAM_REF:-e7a77d0}
CLEAN_CHECK_DIR=${CLEAN_CHECK_DIR:-/tmp/daps_${UPSTREAM_REF}_check_${USER}}

cd "$REPO"
mkdir -p "$OUT"

python scripts/b21/capture_lf_patch.py \
  --repo_root "$REPO" \
  --daps_root "external/daps" \
  --upstream_ref "$UPSTREAM_REF" \
  --diff_paths "sampler.py,posterior_sample.py" \
  --patch_path "$REPO/docs/b21/patches/daps_b20_lf_guidance.patch" \
  --report_path "$REPO/docs/b21/b21_1_lf_patch_capture.md" \
  --outdir "$OUT" \
  --clean_check_dir "$CLEAN_CHECK_DIR" \
  --refresh_clean_check_dir

echo "B21.1 report: $REPO/docs/b21/b21_1_lf_patch_capture.md"
echo "B21.1 patch:  $REPO/docs/b21/patches/daps_b20_lf_guidance.patch"
echo "B21.1 artifacts: $OUT"
