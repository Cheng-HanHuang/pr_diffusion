#!/usr/bin/env bash
set -euo pipefail

# B24.0 zero-GPU closeout. This script never invokes a model or nvidia-smi.
# It creates the isolated PAC worktree, imports the exact populated PRE_B23
# registry, runs pure-CPU tests/validation, dry-renders future manifests, then
# commits/pushes only PRE_B24 plus compact zero-GPU evidence.

ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
B24WT="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
BASE=27505e6328157ac9296c95dc5e611cbeef80de98
PRE23="$CONTROL/manifests/b23/PRE_B23_EXPOSURE.csv"
PRE23_SHA=a513cb4e3b79b39700ff1d623cb4b2eaf496bc2d6d0fe58bd963709e6a56d288
PY="$ROOT/conda-envs/daps/bin/python"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="$OUTROOT/B24_0_closeout_$STAMP"
mkdir -p "$RUNROOT"
LOG="$RUNROOT/B24_0_closeout.log"
RESULTS="$RUNROOT/ZERO_GPU_STEP_RESULTS.tsv"
: >"$LOG"
printf 'step\tstatus\trc\n' >"$RESULTS"

short() { echo "$*"; }
step() {
  local name="$1"; shift
  if "$@" >>"$LOG" 2>&1; then
    printf '%s\tPASS\t0\n' "$name" >>"$RESULTS"
    short "PASS|$name"
  else
    rc=$?
    printf '%s\tFAIL\t%s\n' "$name" "$rc" >>"$RESULTS"
    short "FAIL|$name|rc=$rc|log=$LOG"
    exit "$rc"
  fi
}

short "B24_0_CLOSEOUT|zero_gpu=YES|runroot=$RUNROOT"

# Read-only verification of the B23 worktree plus shared-repository fetch/worktree metadata.
git -C "$CONTROL" rev-parse --git-dir >/dev/null
[ ! -e "$B24WT" ] || { short "STOP|B24 worktree already exists:$B24WT"; exit 20; }
[ -f "$PRE23" ] || { short "STOP|missing populated PRE_B23:$PRE23"; exit 21; }
ACTUAL_PRE23=$(sha256sum "$PRE23" | awk '{print $1}')
[ "$ACTUAL_PRE23" = "$PRE23_SHA" ] || { short "STOP|PRE_B23_SHA|$ACTUAL_PRE23"; exit 22; }
[ -z "$(git -C "$CONTROL" status --short --untracked-files=no)" ] || {
  short "STOP|B23 tracked worktree is dirty; do not use it as B24 control repo"; exit 23;
}

step fetch_b24 git -C "$CONTROL" fetch origin \
  "+refs/heads/codex/b23-execution:refs/remotes/origin/codex/b23-execution" \
  "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
REMOTE_BASE=$(git -C "$CONTROL" rev-parse origin/codex/b23-execution)
[ "$REMOTE_BASE" = "$BASE" ] || { short "STOP|remote_b23_head=$REMOTE_BASE|expected=$BASE"; exit 24; }
step ancestry git -C "$CONTROL" merge-base --is-ancestor "$BASE" "origin/$BRANCH"

if git -C "$CONTROL" show-ref --verify --quiet "refs/heads/$BRANCH"; then
  short "STOP|local branch already exists; refusing ambiguous worktree reuse:$BRANCH"
  exit 25
fi
step add_worktree git -C "$CONTROL" worktree add -b "$BRANCH" "$B24WT" "origin/$BRANCH"

cd "$B24WT"
HEAD=$(git rev-parse HEAD)
REMOTE_HEAD=$(git rev-parse "origin/$BRANCH")
[ "$HEAD" = "$REMOTE_HEAD" ] || { short "STOP|new worktree head does not equal fetched remote head"; exit 26; }
short "WORKTREE|head=$HEAD|branch=$(git branch --show-current)"

step compile "$PY" -m py_compile \
  prdiffusion/b24_protocol.py \
  scripts/b24/build_pre_b24_exposure.py \
  scripts/b24/render_b24_baseline_manifest.py \
  scripts/b24/validate_b24_0.py \
  scripts/b24/run_b24_worker.py

step tests env CUDA_VISIBLE_DEVICES= "$PY" -m unittest tests.b24.test_b24_protocol
step exposure env CUDA_VISIBLE_DEVICES= "$PY" scripts/b24/build_pre_b24_exposure.py \
  --pre-b23 "$PRE23" --out manifests/b24/PRE_B24_EXPOSURE.csv
step validate env CUDA_VISIBLE_DEVICES= "$PY" scripts/b24/validate_b24_0.py
step render64 env CUDA_VISIBLE_DEVICES= "$PY" scripts/b24/render_b24_baseline_manifest.py \
  --count 64 --out "$RUNROOT/B24_2_baseline_64.future.json"
step render256 env CUDA_VISIBLE_DEVICES= "$PY" scripts/b24/render_b24_baseline_manifest.py \
  --count 256 --out "$RUNROOT/B24_2_baseline_256.future.json"
step control_dry64 env CUDA_VISIBLE_DEVICES= bash scripts/b24/launch_b24_nohup.sh \
  --stage B24.0 --manifest "$RUNROOT/B24_2_baseline_64.future.json" --mode serial --dry-run
step control_dry256 env CUDA_VISIBLE_DEVICES= bash scripts/b24/launch_b24_nohup.sh \
  --stage B24.0 --manifest "$RUNROOT/B24_2_baseline_256.future.json" --mode batched --dry-run

PRE24_SHA=$(sha256sum manifests/b24/PRE_B24_EXPOSURE.csv | awk '{print $1}')
PRE24_COUNT=$(tail -n +2 manifests/b24/PRE_B24_EXPOSURE.csv | wc -l | tr -d ' ')
MAN64_SHA=$(sha256sum "$RUNROOT/B24_2_baseline_64.future.json" | awk '{print $1}')
MAN256_SHA=$(sha256sum "$RUNROOT/B24_2_baseline_256.future.json" | awk '{print $1}')

cat >"$RUNROOT/B24_0_CHECKPOINT.json" <<EOF
{
  "schema_version": 1,
  "stage": "B24.0",
  "gpu_work_performed": false,
  "model_work_performed": false,
  "measurement_generation_performed": false,
  "reconstruction_performed": false,
  "scientific_screening_rows_populated": false,
  "worktree_head_before_generated_exposure": "$HEAD",
  "required_base": "$BASE",
  "pre_b23_sha256": "$PRE23_SHA",
  "pre_b24_sha256": "$PRE24_SHA",
  "pre_b24_row_count": $PRE24_COUNT,
  "future_manifest_64_file_sha256": "$MAN64_SHA",
  "future_manifest_256_file_sha256": "$MAN256_SHA",
  "future_scientific_execution_authorized": false
}
EOF

EVDIR="docs/b24/evidence/B24_0_closeout_$STAMP"
mkdir -p "$EVDIR"
cp "$RUNROOT/B24_0_CHECKPOINT.json" "$EVDIR/B24_0_CHECKPOINT.json"
cp "$RESULTS" "$EVDIR/ZERO_GPU_STEP_RESULTS.tsv"
cat >"$EVDIR/README.md" <<EOF
# B24.0 zero-GPU closeout

- PAC run root: \`$RUNROOT\`
- source B24 head: \`$HEAD\`
- populated PRE_B23 SHA-256: \`$PRE23_SHA\`
- PRE_B24 SHA-256: \`$PRE24_SHA\`
- PRE_B24 rows: \`$PRE24_COUNT\`
- GPU/model/measurement/reconstruction work: **NO**
- future B24.1/B24.2 authorization: **NO**

Full log and future dry-rendered manifests remain on PAC and are not scientific results.
EOF

# Before staging, the only repo changes may be PRE_B24 and this exact evidence directory.
UNEXPECTED=$(git status --porcelain | awk -v ev="$EVDIR/" '
  $2 == "manifests/b24/PRE_B24_EXPOSURE.csv" {next}
  index($2, ev) == 1 {next}
  {print}
')
[ -z "$UNEXPECTED" ] || { short "STOP|unexpected repo changes"; printf '%s\n' "$UNEXPECTED"; exit 27; }

git add manifests/b24/PRE_B24_EXPOSURE.csv "$EVDIR"
step staged_diff git diff --cached --check
step commit git commit -m "B24.0: publish exposure freeze and zero-GPU closeout"
FINAL_HEAD=$(git rev-parse HEAD)
step push git push origin "HEAD:refs/heads/$BRANCH"

short "B24_0_COMPLETE|head=$FINAL_HEAD|pre_b24_sha256=$PRE24_SHA|rows=$PRE24_COUNT|gpu_work=NO"
short "EVIDENCE|$EVDIR"
short "RESULTS|$RESULTS"
short "LOG|$LOG"
short "STOP|B24.1_AND_B24.2_NOT_AUTHORIZED"
