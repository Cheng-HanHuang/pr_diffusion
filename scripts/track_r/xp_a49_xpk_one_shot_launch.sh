#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
BRANCH=codex/prdiff-track-r-a40-xpk-gated-autopilot-20261105

EXPECTED="${1:?usage: $0 EXPECTED_COMMIT_SHA}"
DEFAULT_PRIOR_PROJECT_H=247.80191737622505
PRIOR_EXPLICIT=0
OTHER_EXPLICIT=0
if [ "${PRIOR_PROJECT_H+x}" = x ]; then PRIOR_EXPLICIT=1; fi
if [ "${OTHER_XPK_H+x}" = x ]; then OTHER_EXPLICIT=1; fi
if [ "$PRIOR_EXPLICIT" -ne "$OTHER_EXPLICIT" ]; then
  echo "STOP_SET_BOTH_PRIOR_PROJECT_H_AND_OTHER_XPK_H_OR_NEITHER"
  exit 8
fi
PRIOR_PROJECT_H="${PRIOR_PROJECT_H:-$DEFAULT_PRIOR_PROJECT_H}"
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

# Failed/recovered A49 attempts still consume physical GPU occupancy. When the
# operator has not supplied a newer accepted ledger explicitly, carry all
# previous A49 journal charges into BOTH the XP-K cap and project baseline.
AUTO_PRIOR_A49_H="$(PYTHONPATH="$WT" "$PY" - "$BASE" "$RUN" <<'PY'
import json,sys,time
from pathlib import Path
from scripts.track_r.xp_a47_xpk_gpu_ledger_cpu import charge
base=Path(sys.argv[1]); current=Path(sys.argv[2]).resolve()
total=0.0
for d in sorted(base.glob("xp_a41_xpk_autopilot_*")):
    try:
        if d.resolve()==current:
            continue
    except Exception:
        continue
    p=d/"XPK_GPU_EVENTS.jsonl"
    if not p.is_file() or p.is_symlink():
        continue
    rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    total += float(charge(rows,time.time())["physical_gpu_h"])
print(f"{total:.12f}")
PY
)"
if [ "$PRIOR_EXPLICIT" -eq 0 ]; then
  OTHER_XPK_H="$AUTO_PRIOR_A49_H"
  PRIOR_PROJECT_H="$("$PY" - "$DEFAULT_PRIOR_PROJECT_H" "$AUTO_PRIOR_A49_H" <<'PY'
import sys
print(f"{float(sys.argv[1])+float(sys.argv[2]):.12f}")
PY
)"
fi

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
echo "AUTO_PRIOR_A49_H=$AUTO_PRIOR_A49_H"
echo "LEDGER_INPUTS_EXPLICIT=$PRIOR_EXPLICIT"
echo "MAX_GPUS=$MAX_GPUS"
