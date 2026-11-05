#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
BRANCH=codex/prdiff-track-r-a40-xpk-gated-autopilot-20261105

EXPECTED="${1:?usage: $0 EXPECTED_COMMIT_SHA}"
PRIOR_PROJECT_H="${PRIOR_PROJECT_H:-247.80191737622505}"
OTHER_XPK_H="${OTHER_XPK_H:-0}"
MAX_GPUS=2

mkdir -p "$BASE"

if [ -n "$(git -C "$WT" status --porcelain)" ]; then
  echo "STOP_DIRTY_WORKTREE"
  exit 8
fi

git -C "$WT" fetch origin "$BRANCH"
REMOTE="$(git -C "$WT" rev-parse "origin/$BRANCH")"
if [ "$REMOTE" != "$EXPECTED" ]; then
  echo "STOP_REMOTE_HEAD_MOVED expected=$EXPECTED remote=$REMOTE"
  exit 8
fi

git -C "$WT" checkout --detach "$EXPECTED"
test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED"

# Re-running the same launch wrapper resumes the exact pinned run when possible.
RUN=""
if [ -s "$BASE/LAST_XPK_A49_RUN.txt" ]; then
  CANDIDATE="$(cat "$BASE/LAST_XPK_A49_RUN.txt")"
  if [ -d "$CANDIDATE" ] && [ -f "$CANDIDATE/XPK_A49_CONTROLLER_STATE.json" ]; then
    STATE_HEAD="$("$PY" - "$CANDIDATE/XPK_A49_CONTROLLER_STATE.json" <<'PY'
import json,sys
try:
    print(json.load(open(sys.argv[1])).get("source_head",""))
except Exception:
    print("")
PY
)"
    if [ "$STATE_HEAD" = "$EXPECTED" ]; then
      if [ -f "$CANDIDATE/XPK_A49_TERMINAL.json" ]; then
        echo "XP_K_A49_ALREADY_TERMINAL"
        cat "$CANDIDATE/XPK_A49_TERMINAL.json"
        exit 0
      fi
      RUN="$CANDIDATE"
    fi
  fi
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [ -z "$RUN" ]; then
  RUN="$BASE/xp_a41_xpk_autopilot_${STAMP}_${EXPECTED:0:12}"
fi
LOG="$BASE/xp_a49_xpk_controller_${STAMP}_${EXPECTED:0:12}.log"

printf '%s\n' "$RUN" > "$BASE/LAST_XPK_A49_RUN.txt"
printf '%s\n' "$LOG" > "$BASE/LAST_XPK_A49_LOG.txt"

export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

nohup "$PY" -u -m scripts.track_r.xp_a49_xpk_one_shot_autopilot \
  --expected-head "$EXPECTED" \
  --branch "$BRANCH" \
  --run "$RUN" \
  --prior-project-h "$PRIOR_PROJECT_H" \
  --other-xpk-h "$OTHER_XPK_H" \
  --max-gpus "$MAX_GPUS" \
  --wait-seconds 60 \
  --push-retry-seconds 120 \
  >"$LOG" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "$PID" > "$BASE/LAST_XPK_A49_PID.txt"

echo "XP_K_A49_ONE_SHOT_LAUNCHED"
echo "HEAD=$EXPECTED"
echo "PID=$PID"
echo "RUN=$RUN"
echo "LOG=$LOG"
echo "PRIOR_PROJECT_H=$PRIOR_PROJECT_H"
echo "OTHER_XPK_H=$OTHER_XPK_H"
echo "MAX_GPUS=$MAX_GPUS"
