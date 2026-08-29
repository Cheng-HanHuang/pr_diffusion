#!/usr/bin/env bash
set -euo pipefail
ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
LATEST="$OUTROOT/B24_2_6144_LATEST_RUN.txt"
export TORCH_HOME="$ROOT/models/torch_cache"

RUNROOT=${1:-}
if [[ -z "$RUNROOT" ]]; then
  [[ -f "$LATEST" ]] || { echo "STOP|no_6144_run"; exit 2; }
  RUNROOT=$(cat "$LATEST")
fi
[[ -f "$RUNROOT/LAUNCH.json" ]] || { echo "STOP|missing_launch:$RUNROOT/LAUNCH.json"; exit 2; }

readarray -t META < <("$PY" - "$RUNROOT/LAUNCH.json" <<'PY'
import json,sys
from pathlib import Path
v=json.loads(Path(sys.argv[1]).read_text())
print(v['manifest_path'])
print(v['parent2048_runroot'])
print(v['baseline_min_free_mib'])
PY
)
MANIFEST=${META[0]}
PARENT=${META[1]}
GATE=${META[2]}
[[ "$GATE" == "10240" ]] || { echo "STOP|unexpected_gate:$GATE"; exit 3; }

[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  echo "STOP|B24_worktree_dirty"; git -C "$REPO" status --short; exit 3;
}
git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD)
REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || {
  echo "STOP|local_not_ancestor|local=$LOCAL|remote=$REMOTE"; exit 4;
}
if [[ "$LOCAL" != "$REMOTE" ]]; then git -C "$REPO" merge --ff-only "origin/$BRANCH"; fi
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export B24_BASELINE_MIN_FREE_MIB="$GATE"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for GPU in 0 1 2 3; do
  SHARD_SUM="$RUNROOT/shard${GPU}/SHARD_COMPLETE.json"
  PIDFILE="$RUNROOT/pids/gpu${GPU}.pid"
  LOG="$RUNROOT/logs/gpu${GPU}.log"
  if [[ -f "$SHARD_SUM" ]]; then
    echo "RESUME_SKIP|gpu=$GPU|reason=complete"
    continue
  fi
  if [[ -f "$PIDFILE" ]]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
      echo "RESUME_SKIP|gpu=$GPU|reason=running|pid=$PID"
      continue
    fi
    mv "$PIDFILE" "$RUNROOT/pids/gpu${GPU}.pre_resume_$STAMP.pid"
  fi
  if [[ -f "$LOG" ]]; then
    mv "$LOG" "$RUNROOT/logs/gpu${GPU}.pre_resume_$STAMP.log"
  fi
  MODE=()
  if [[ -d "$RUNROOT/shard${GPU}" ]]; then MODE=(--resume); fi
  nohup env \
    PYTHONPATH="$PYTHONPATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME="$TORCH_HOME" \
    B24_BASELINE_MIN_FREE_MIB="$GATE" \
    "$PY" scripts/b24/run_b24_2_6144_extension_shard.py \
      --manifest "$MANIFEST" --parent-runroot "$PARENT" \
      --shard "$GPU" --gpu "$GPU" --repo "$REPO" \
      --output-root "$RUNROOT/shard${GPU}" "${MODE[@]}" \
    >"$LOG" 2>&1 &
  PID=$!
  printf '%s\n' "$PID" >"$PIDFILE"
  echo "RESUMED|gpu=$GPU|pid=$PID|mode=${MODE[*]:-fresh}"
done

echo "STATUS|bash $REPO/scripts/b24/status_b24_2_6144.sh $RUNROOT"
