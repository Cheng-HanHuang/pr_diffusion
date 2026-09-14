#!/usr/bin/env bash
set -euo pipefail
ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
PARENT64="$OUTROOT/B24_2_64_20260826T040303Z"
PARENT_MANIFEST="$PARENT64/B24_2_baseline_64.json"
LATEST="$OUTROOT/B24_2_256_LATEST_RUN.txt"
export TORCH_HOME="$ROOT/models/torch_cache"

RUNROOT=${1:-}
if [[ -z "$RUNROOT" ]]; then
  [[ -f "$LATEST" ]] || { echo "STOP|missing_latest_256_pointer"; exit 2; }
  RUNROOT=$(cat "$LATEST")
fi
MANIFEST="$RUNROOT/B24_2_baseline_256.json"
[[ -d "$RUNROOT" && -f "$MANIFEST" && -f "$PARENT_MANIFEST" ]] || { echo "STOP|missing_run_or_manifest|runroot=$RUNROOT"; exit 2; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "STOP|B24_worktree_dirty"; exit 3; }

git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD); REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || { echo "STOP|local_not_ancestor"; exit 4; }
if [[ "$LOCAL" != "$REMOTE" ]]; then git -C "$REPO" merge --ff-only "origin/$BRANCH"; fi
HEAD=$(git -C "$REPO" rev-parse HEAD)
cd "$REPO"; export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONDONTWRITEBYTECODE=1
"$PY" -m py_compile scripts/b24/run_b24_2_256_extension_shard.py

"$PY" - "$PARENT_MANIFEST" "$MANIFEST" <<'PY'
import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text()); q=json.loads(Path(sys.argv[2]).read_text())
assert len(p['rows'])==64 and len(q['rows'])==256 and q['rows'][:64]==p['rows']
print('PREFIX_READY|parent=64|target=256')
PY

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
launched=0
for GPU in 0 1 2 3; do
  SHARDROOT="$RUNROOT/shard${GPU}"
  SUMMARY="$SHARDROOT/SHARD_COMPLETE.json"
  PIDFILE="$RUNROOT/pids/gpu${GPU}.pid"
  LOG="$RUNROOT/logs/gpu${GPU}.log"
  if [[ -f "$SUMMARY" ]]; then
    echo "SHARD_ALREADY_COMPLETE|gpu=$GPU"
    continue
  fi
  if [[ -f "$PIDFILE" ]]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
      echo "SHARD_STILL_RUNNING|gpu=$GPU|pid=$PID"
      continue
    fi
    mv "$PIDFILE" "$RUNROOT/pids/gpu${GPU}.pre_resume_$STAMP.pid"
  fi
  if [[ -f "$LOG" ]]; then mv "$LOG" "$RUNROOT/logs/gpu${GPU}.pre_resume_$STAMP.log"; fi
  [[ -d "$SHARDROOT" ]] || { echo "STOP|missing_partial_shard_root|gpu=$GPU|path=$SHARDROOT"; exit 20; }
  nohup env PYTHONPATH="$PYTHONPATH" PYTHONDONTWRITEBYTECODE=1 TORCH_HOME="$TORCH_HOME" \
    "$PY" scripts/b24/run_b24_2_256_extension_shard.py \
      --manifest "$MANIFEST" --parent-manifest "$PARENT_MANIFEST" \
      --shard "$GPU" --gpu "$GPU" --repo "$REPO" --output-root "$SHARDROOT" --resume \
    >"$RUNROOT/logs/gpu${GPU}.log" 2>&1 &
  NEWPID=$!; printf '%s\n' "$NEWPID" >"$RUNROOT/pids/gpu${GPU}.pid"
  echo "SHARD_RESUMED|gpu=$GPU|pid=$NEWPID|head=$HEAD"
  launched=$((launched+1))
done

echo "B24_2_256_RESUME|runroot=$RUNROOT|launched_shards=$launched"
echo "STATUS|bash $REPO/scripts/b24/status_b24_2_256.sh $RUNROOT"
