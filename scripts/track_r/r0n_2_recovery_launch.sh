#!/usr/bin/env bash
# N2-only recovery. Reads immutable N2 CPU-prepared PNGs, never reruns N0/N1/N3.
# Usage: bash scripts/track_r/r0n_2_recovery_launch.sh <exact-GitHub-HEAD>
set -u

ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
CACHE="$ROOT/outputs/.cache/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
BRANCH=codex/prdiff-track-r-r0n-20261023
EXPECTED="${1:-}"
TASK="$BASE/r0n_n2prep_20261024T061111Z/N2_GT_FREE_CANDIDATES.json"
TASK_SHA=7be734de4cd2e6f395add521717506be9260c37708f8902e248c1848b68b705b

if [ -z "$EXPECTED" ];then echo STOP_N2_RECOVERY_NO_HEAD; exit 3;fi
if [ ! -x "$PY" ];then echo STOP_N2_RECOVERY_PYTHON_MISSING;exit 3;fi
if [ -n "$(git -C "$WT" status --porcelain)" ];then
 echo STOP_N2_RECOVERY_DIRTY_WORKTREE;exit 4
fi
git -C "$WT" fetch origin "$BRANCH" || exit $?
if [ "$(git -C "$WT" rev-parse "origin/$BRANCH")" != "$EXPECTED" ];then
 echo STOP_N2_RECOVERY_REMOTE_HEAD_CHANGED;exit 4
fi
git -C "$WT" checkout --detach "$EXPECTED" || exit $?
if [ -n "$(git -C "$WT" status --porcelain)" ];then
 echo STOP_N2_RECOVERY_DIRTY_AFTER_CHECKOUT;exit 4
fi

mkdir -p "$BASE/logs" "$CACHE/home" "$CACHE/tmp" "$CACHE/config" \
 "$CACHE/data" "$CACHE/state" "$CACHE/torch" "$CACHE/hf" "$CACHE/pip"
export PYTHONPATH="$WT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export R0N_GIT_ORIGINAL_HOME="${HOME:-}"
export HOME="$CACHE/home" TMPDIR="$CACHE/tmp"
export XDG_CACHE_HOME="$CACHE" XDG_CONFIG_HOME="$CACHE/config"
export XDG_DATA_HOME="$CACHE/data" XDG_STATE_HOME="$CACHE/state"
export TORCH_HOME="$CACHE/torch" HF_HOME="$CACHE/hf" PIP_CACHE_DIR="$CACHE/pip"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$WT" || exit 5

CUDA_VISIBLE_DEVICES="" "$PY" - <<'PY'
import ast
from pathlib import Path
for name in ("scripts/track_r/r0n_2_clip_orientation.py",
             "scripts/track_r/r0n_2_recovery_autopilot.py",
             "prdiffusion/track_r/r0n_clip_compat.py"):
    ast.parse(Path(name).read_text(),filename=name)
print("R0N_N2_RECOVERY_SYNTAX_PASS")
PY
RC=$?
if [ "$RC" -ne 0 ];then echo STOP_N2_RECOVERY_SYNTAX;exit "$RC";fi
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q -p no:cacheprovider tests/track_r
RC=$?
echo "R0N_N2_RECOVERY_CPU_TEST_RC=$RC"
if [ "$RC" -ne 0 ];then echo STOP_N2_RECOVERY_TESTS;exit "$RC";fi

if [ "$(sha256sum "$TASK" | awk '{print $1}')" != "$TASK_SHA" ];then
 echo STOP_N2_RECOVERY_EXISTING_TASKS_SHA;exit 7
fi
echo "R0N_N2_FROZEN_TASK_SHA_PASS=$TASK_SHA"

# Choose physical GPU with the most actual free VRAM and >= 40 GiB headroom.
GPU=""
BEST_FREE=0
if command -v nvidia-smi >/dev/null 2>&1;then
 while IFS=',' read -r INDEX FREE TOTAL;do
   INDEX="${INDEX//[[:space:]]/}"
   FREE="${FREE//[[:space:]]/}"
   if [[ "$INDEX" =~ ^[0-9]+$ && "$FREE" =~ ^[0-9]+$ ]] &&
      [ "$INDEX" -le 3 ] && [ "$FREE" -ge 40960 ] &&
      [ "$FREE" -gt "$BEST_FREE" ];then
       GPU="$INDEX";BEST_FREE="$FREE"
   fi
 done < <(nvidia-smi --query-gpu=index,memory.free,memory.total \
              --format=csv,noheader,nounits)
fi
if [ -z "$GPU" ];then
 echo "STOP_N2_RECOVERY_NO_GPU_WITH_40_GIB_FREE";exit 8
fi
echo "R0N_N2_SELECTED_GPU=$GPU FREE_MIB=$BEST_FREE"

# Publishing can be unavailable but must not block the scientific result.
if GIT_TERMINAL_PROMPT=0 timeout 20s git push --dry-run origin \
     "HEAD:refs/heads/$BRANCH" >/dev/null 2>&1;then
 echo R0N_N2_GIT_PUSH_PREFLIGHT_PASS
else
 echo R0N_N2_GIT_PUSH_PREFLIGHT_WARNING_LOCAL_RETURN_WILL_BE_PRESERVED
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$BASE/r0n_n2_recovery_$STAMP"
LOG="$BASE/logs/r0n_n2_recovery_$STAMP.log"
if [ -e "$RUN" ];then echo STOP_N2_RECOVERY_OUTPUT_EXISTS;exit 9;fi

CUDA_VISIBLE_DEVICES="" nohup "$PY" -u \
 scripts/track_r/r0n_2_recovery_autopilot.py \
 --expected-head "$EXPECTED" --out "$RUN" \
 --gpu "$GPU" --budget-seconds 870 > "$LOG" 2>&1 &
PID=$!
printf '%s\n' "$RUN" > "$BASE/LATEST_R0N_N2_RECOVERY_RUN"
printf '%s\n' "$PID" > "$BASE/logs/r0n_n2_recovery_$STAMP.pid"
echo "N2_RECOVERY_LAUNCHED_PID=$PID"
echo "N2_RECOVERY_RUN=$RUN"
echo "N2_RECOVERY_LOG=$LOG"
