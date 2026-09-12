#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
PARENT_POINTER="$OUTROOT/B24_2_256_LATEST_RUN.txt"
RUN2048_POINTER="$OUTROOT/B24_2_2048_LATEST_RUN.txt"
GATE_MIB=10240

[[ -x "$PY" ]] || { echo "STOP|missing_python:$PY"; exit 2; }
[[ -f "$PARENT_POINTER" ]] || { echo "STOP|missing_parent_pointer:$PARENT_POINTER"; exit 2; }
[[ -f "$RUN2048_POINTER" ]] || { echo "STOP|missing_2048_pointer:$RUN2048_POINTER"; exit 2; }
PARENT256=$(cat "$PARENT_POINTER")
RUN2048=$(cat "$RUN2048_POINTER")
[[ -d "$PARENT256" ]] || { echo "STOP|missing_parent256:$PARENT256"; exit 2; }
[[ -d "$RUN2048" ]] || { echo "STOP|missing_run2048:$RUN2048"; exit 2; }

# Do not mutate or restart the three healthy 2048 shards.
for GPU in 0 1 3; do
  PIDFILE="$RUN2048/pids/gpu${GPU}.pid"
  [[ -f "$PIDFILE" ]] || { echo "STOP|missing_healthy_pidfile|gpu=$GPU"; exit 3; }
  PID=$(cat "$PIDFILE")
  kill -0 "$PID" 2>/dev/null || { echo "STOP|healthy_shard_not_live|gpu=$GPU|pid=$PID"; exit 3; }
  echo "HEALTHY_SHARD_UNTOUCHED|gpu=$GPU|pid=$PID"
done

# GPU 2 must not already have a live parent or 2048 worker before repair.
for PIDFILE in "$PARENT256/pids/gpu2.pid" "$RUN2048/pids/gpu2.pid"; do
  if [[ -f "$PIDFILE" ]]; then
    PID=$(cat "$PIDFILE" 2>/dev/null || true)
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
      echo "STOP|gpu2_worker_already_live|pid=$PID|pidfile=$PIDFILE"
      exit 4
    fi
  fi
done

[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  echo "STOP|B24_worktree_dirty"; git -C "$REPO" status --short; exit 5;
}
git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD)
REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || {
  echo "STOP|local_not_ancestor|local=$LOCAL|remote=$REMOTE"; exit 6;
}
if [[ "$LOCAL" != "$REMOTE" ]]; then
  git -C "$REPO" merge --ff-only "origin/$BRANCH"
fi
HEAD=$(git -C "$REPO" rev-parse HEAD)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME="$ROOT/models/torch_cache"
export B24_BASELINE_MIN_FREE_MIB="$GATE_MIB"

"$PY" -m py_compile scripts/b24/recover_b24_2_gpu2_handoff.py

count_completions() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    printf '0\n'
    return 0
  fi
  find "$dir" -name IMAGE_COMPLETE.json -type f -print 2>/dev/null | wc -l | tr -d ' '
}

PARENT_DONE=$(count_completions "$PARENT256/shard2")
NEW_DONE=$(count_completions "$RUN2048/shard2")
echo "GPU2_REPAIR_PREFLIGHT|head=$HEAD|parent256_completed=$PARENT_DONE/48|new2048_completed=$NEW_DONE/448|gate_mib=$GATE_MIB"

# Preserve the original failed 2048 handoff log. The supervisor and eventual
# 2048 shard-2 child then share a fresh gpu2.log for simple status monitoring.
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$RUN2048/logs/gpu2.log"
if [[ -f "$LOG" ]]; then
  mv "$LOG" "$RUN2048/logs/gpu2.pre_repair_${STAMP}.log"
fi
mkdir -p "$RUN2048/logs" "$RUN2048/pids" "$PARENT256/pids"

nohup env \
  PYTHONPATH="$PYTHONPATH" \
  PYTHONDONTWRITEBYTECODE=1 \
  TORCH_HOME="$TORCH_HOME" \
  B24_BASELINE_MIN_FREE_MIB="$GATE_MIB" \
  "$PY" scripts/b24/recover_b24_2_gpu2_handoff.py \
    --parent256 "$PARENT256" \
    --run2048 "$RUN2048" \
    --repo "$REPO" \
    --gate-mib "$GATE_MIB" \
  >"$LOG" 2>&1 &
SUP_PID=$!
printf '%s\n' "$SUP_PID" >"$PARENT256/pids/gpu2.pid"
printf '%s\n' "$SUP_PID" >"$RUN2048/pids/gpu2.pid"

echo "GPU2_REPAIR_LAUNCHED|supervisor_pid=$SUP_PID|parent256=$PARENT256|run2048=$RUN2048|gate_mib=$GATE_MIB"
echo "GPUS_0_1_3|UNTOUCHED_AND_RUNNING"
echo "STATUS|bash $REPO/scripts/b24/status_b24_2_2048.sh $RUN2048"
echo "LOG|$LOG"
