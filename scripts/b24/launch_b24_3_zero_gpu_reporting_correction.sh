#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
PY="$ROOT/conda-envs/daps/bin/python"
SOURCE="$OUTROOT/B24_3_zero_gpu_closeout_20260914T044033Z"
SOURCE_ARCHIVE="$SOURCE.tar.gz"
SOURCE_SIDECAR="$SOURCE_ARCHIVE.sha256"
EXPECTED_SOURCE_ARCHIVE_SHA=fe4ac6e0ac5c6554973cc171067f52fe0b729e6a33ded7503594fd424da3b45d

[[ -x "$PY" ]] || { echo "STOP|missing_python:$PY"; exit 2; }
[[ -d "$SOURCE" ]] || { echo "STOP|missing_source_capsule:$SOURCE"; exit 2; }
[[ -f "$SOURCE_ARCHIVE" && -f "$SOURCE_SIDECAR" ]] || { echo "STOP|missing_source_archive_or_sidecar"; exit 2; }

ACTUAL_SOURCE_ARCHIVE_SHA=$(sha256sum "$SOURCE_ARCHIVE" | awk '{print $1}')
[[ "$ACTUAL_SOURCE_ARCHIVE_SHA" == "$EXPECTED_SOURCE_ARCHIVE_SHA" ]] || {
  echo "STOP|source_archive_sha_mismatch|expected=$EXPECTED_SOURCE_ARCHIVE_SHA|actual=$ACTUAL_SOURCE_ARCHIVE_SHA"
  exit 2
}
sha256sum -c "$SOURCE_SIDECAR"
(
  cd "$SOURCE"
  sha256sum -c SHA256SUMS.txt
)
echo "B24_3_REPORTING_CORRECTION_SOURCE_CAPSULE_PASS|sha256=$ACTUAL_SOURCE_ARCHIVE_SHA"

[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  echo "STOP|B24_worktree_dirty"
  git -C "$REPO" status --short
  exit 3
}
git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD)
REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || {
  echo "STOP|local_not_ancestor|local=$LOCAL|remote=$REMOTE"
  exit 4
}
if [[ "$LOCAL" != "$REMOTE" ]]; then
  git -C "$REPO" merge --ff-only "origin/$BRANCH"
fi
HEAD=$(git -C "$REPO" rev-parse HEAD)

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="" "$PY" -m py_compile \
  scripts/b24/correct_b24_3_zero_gpu_closeout.py \
  scripts/b24/test_b24_3_zero_gpu_reporting_correction.py
CUDA_VISIBLE_DEVICES="" "$PY" scripts/b24/test_b24_3_zero_gpu_reporting_correction.py
CUDA_VISIBLE_DEVICES="" "$PY" - <<'PY'
import torch
assert not torch.cuda.is_available(), 'CUDA unexpectedly visible in reporting-only correction'
print('B24_3_ZERO_GPU_REPORTING_CORRECTION_ENV_PASS')
PY

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$OUTROOT/B24_3_zero_gpu_closeout_corrected_${STAMP}"
[[ ! -e "$RUN" ]] || { echo "STOP|successor_exists:$RUN"; exit 5; }

CUDA_VISIBLE_DEVICES="" "$PY" "$REPO/scripts/b24/correct_b24_3_zero_gpu_closeout.py" \
  --source "$SOURCE" \
  --output "$RUN"

"$PY" - "$RUN" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1])
c=json.load(open(r/'B24_3_DEV_CLOSEOUT.json'))
k=json.load(open(r/'CORRECTION_CHECKS.json'))
if c.get('status')!='PASS' or c.get('decision')!='STOP_B24_METHOD_REFINEMENT':
    raise SystemExit('corrected closeout decision drift')
if c.get('confirmation_exposed') is not False or c.get('gpu_work_performed') is not False or c.get('measurement_generation_performed') is not False:
    raise SystemExit('corrected closeout scope violation')
if k.get('status')!='PASS' or k.get('nested_oracle_inequality',{}).get('pass') is not True:
    raise SystemExit('correction checks failed')
shared=c['complementarity_summary']['shared_failure_subset']
expected_success={'DAPS1':0,'FRESH2_SELECTED':0,'SITCOM1':0,'NP4_SELECTED':2,'EPP321_SELECTED':2,'PE3_SCORE_SELECTED':1,'PE3_RANDOM_SELECTED':0}
expected_exclusive={'DAPS1':0,'FRESH2_SELECTED':0,'SITCOM1':0,'NP4_SELECTED':1,'EPP321_SELECTED':2,'PE3_SCORE_SELECTED':0,'PE3_RANDOM_SELECTED':0}
if shared['count']!=10 or shared['good25_success_counts']!=expected_success or shared['exclusive_good25_counts_among_compared_executable_methods']!=expected_exclusive:
    raise SystemExit(f'shared-failure correction drift: {shared}')
print('B24_3_ZERO_GPU_REPORTING_CORRECTION_GATE_PASS')
PY

cat > "$RUN/LAUNCH_IDENTITY.txt" <<EOF
correction_head=$HEAD
source_capsule=$SOURCE
source_capsule_archive_sha256=$EXPECTED_SOURCE_ARCHIVE_SHA
correction_type=ZERO_GPU_REPORTING_ONLY
gpu_work=0
measurement_generation=0
reconstruction=0
confirmation_exposed=0
method_refinement_decision=STOP_B24_METHOD_REFINEMENT
EOF

(
  cd "$RUN"
  rm -f SHA256SUMS.txt
  find . -maxdepth 1 -type f ! -name 'SHA256SUMS.txt' -printf '%f\n' | sort | xargs sha256sum > SHA256SUMS.txt
)

ARCHIVE="$RUN.tar.gz"
tar -C "$(dirname "$RUN")" -czf "$ARCHIVE" "$(basename "$RUN")"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
printf '%s\n' "$RUN" > "$OUTROOT/B24_3_ZERO_GPU_CLOSEOUT_CORRECTED_LATEST_RUN.txt"

cat "$RUN/CORRECTION_CHECKS.json"
cat "$RUN/B24_3_DEV_CLOSEOUT.json"
echo "CORRECTED_REPORT|$RUN/B24_3_DEV_CLOSEOUT.md"
echo "CORRECTED_COMPLEMENTARITY|$RUN/COMPLEMENTARITY.json"
echo "CORRECTION_CHECKS|$RUN/CORRECTION_CHECKS.json"
echo "CORRECTION_METADATA|$RUN/CORRECTION_METADATA.json"
echo "CORRECTED_ARCHIVE|$ARCHIVE"
echo "CORRECTED_ARCHIVE_SHA256|$ARCHIVE.sha256"
echo "B24_3_ZERO_GPU_REPORTING_CORRECTION_COMPLETE|run=$RUN|head=$HEAD|source_sha256=$EXPECTED_SOURCE_ARCHIVE_SHA"
