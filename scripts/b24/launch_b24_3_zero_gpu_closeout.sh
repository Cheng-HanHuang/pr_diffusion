#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
DAPS_PY="$ROOT/conda-envs/daps/bin/python"
DEV_PTR="$OUTROOT/B24_3_DEV80_LATEST_RUN.txt"
PE3_PTR="$OUTROOT/B24_3_PE3_LATEST_RUN.txt"

[[ -x "$DAPS_PY" ]] || { echo "STOP|missing_daps_python:$DAPS_PY"; exit 2; }
[[ -f "$DEV_PTR" && -f "$PE3_PTR" ]] || { echo "STOP|missing_source_pointer"; exit 2; }
DEV80=$(cat "$DEV_PTR")
PE3=$(cat "$PE3_PTR")
[[ -d "$DEV80" && -d "$PE3" ]] || { echo "STOP|missing_source_run"; exit 2; }

"$DAPS_PY" - "$DEV80" "$PE3" <<'PY'
import json,pathlib,sys
D=pathlib.Path(sys.argv[1]); P=pathlib.Path(sys.argv[2])
ds=json.load(open(D/'DEV80_SUMMARY.json')); ps=json.load(open(P/'PE3_DEV80_SUMMARY.json'))
if ds.get('status')!='PASS' or ds.get('image_count')!=80 or ds.get('confirmation_exposed') is not False:
    raise SystemExit('bad DEV80 source')
if ps.get('status')!='PASS' or ps.get('image_count')!=80 or ps.get('confirmation_exposed') is not False:
    raise SystemExit('bad PE3 source')
if ps.get('decision')!='STOP_B24_METHOD_REFINEMENT' or ps.get('passing_arms')!=[]:
    raise SystemExit('PE3 frozen stop verdict drift')
a=P/'compute_audit/run/CROSS_FAMILY_FLOP_AUDIT.json'
if not a.is_file(): raise SystemExit(f'missing compute audit {a}')
av=json.load(open(a))
if av.get('status')!='PASS' or av.get('confirmation_exposed') is not False:
    raise SystemExit('bad compute audit source')
print('B24_3_CLOSEOUT_SOURCE_GATE_PASS')
PY

[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "STOP|B24_worktree_dirty"; git -C "$REPO" status --short; exit 3; }
git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD)
REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || { echo "STOP|local_not_ancestor|local=$LOCAL|remote=$REMOTE"; exit 4; }
if [[ "$LOCAL" != "$REMOTE" ]]; then git -C "$REPO" merge --ff-only "origin/$BRANCH"; fi
HEAD=$(git -C "$REPO" rev-parse HEAD)

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="" "$DAPS_PY" -m py_compile \
  scripts/b24/closeout_b24_3_zero_gpu.py \
  scripts/b24/test_b24_3_zero_gpu_closeout.py
CUDA_VISIBLE_DEVICES="" "$DAPS_PY" scripts/b24/test_b24_3_zero_gpu_closeout.py

# Explicitly assert that CUDA is hidden in the exact execution environment.
CUDA_VISIBLE_DEVICES="" "$DAPS_PY" - <<'PY'
import torch
assert not torch.cuda.is_available(), 'CUDA unexpectedly visible in zero-GPU closeout'
print('B24_3_ZERO_GPU_ENV_PASS')
PY

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$OUTROOT/B24_3_zero_gpu_closeout_${STAMP}"
[[ ! -e "$RUN" ]] || { echo "STOP|run_exists:$RUN"; exit 5; }

CUDA_VISIBLE_DEVICES="" "$DAPS_PY" "$REPO/scripts/b24/closeout_b24_3_zero_gpu.py" \
  --dev80-run "$DEV80" \
  --pe3-run "$PE3" \
  --output "$RUN"

"$DAPS_PY" - "$RUN" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); p=json.load(open(r/'B24_3_DEV_CLOSEOUT.json'))
if p.get('status')!='PASS' or p.get('decision')!='STOP_B24_METHOD_REFINEMENT': raise SystemExit('closeout verdict drift')
if p.get('confirmation_exposed') is not False or p.get('gpu_work_performed') is not False or p.get('measurement_generation_performed') is not False:
    raise SystemExit('closeout scope violation')
required=['FRESH2_PER_IMAGE.csv','FRESH2_SUMMARY.json','DEV80_CLOSEOUT_PER_IMAGE.csv','PAIRWISE_EXECUTABLE.csv','HARD_SUBSET.csv','COMPLEMENTARITY.json','COMPUTE_CLOSEOUT.json','B24_3_DEV_CLOSEOUT.md','B24_3_DEV_CLOSEOUT.json','SHA256SUMS.txt']
for name in required:
    if not (r/name).is_file(): raise SystemExit(f'missing closeout artifact {name}')
print('B24_3_ZERO_GPU_CLOSEOUT_GATE_PASS')
PY

cat > "$RUN/LAUNCH_IDENTITY.txt" <<EOF
head=$HEAD
source_dev80_run=$DEV80
source_pe3_run=$PE3
gpu_work=0
measurement_generation=0
confirmation_exposed=0
fresh2_theta=0.7
method_refinement_decision=STOP_B24_METHOD_REFINEMENT
EOF

# Recompute checksums after LAUNCH_IDENTITY is added.
(
  cd "$RUN"
  rm -f SHA256SUMS.txt
  find . -maxdepth 1 -type f ! -name 'SHA256SUMS.txt' -printf '%f\n' | sort | xargs sha256sum > SHA256SUMS.txt
)

ARCHIVE="$RUN.tar.gz"
tar -C "$(dirname "$RUN")" -czf "$ARCHIVE" "$(basename "$RUN")"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
printf '%s\n' "$RUN" > "$OUTROOT/B24_3_ZERO_GPU_CLOSEOUT_LATEST_RUN.txt"

cat "$RUN/B24_3_DEV_CLOSEOUT.json"
echo "REPORT|$RUN/B24_3_DEV_CLOSEOUT.md"
echo "FRESH2|$RUN/FRESH2_SUMMARY.json"
echo "COMPLEMENTARITY|$RUN/COMPLEMENTARITY.json"
echo "COMPUTE|$RUN/COMPUTE_CLOSEOUT.json"
echo "ARCHIVE|$ARCHIVE"
echo "ARCHIVE_SHA256|$ARCHIVE.sha256"
echo "B24_3_ZERO_GPU_CLOSEOUT_COMPLETE|run=$RUN|head=$HEAD"
