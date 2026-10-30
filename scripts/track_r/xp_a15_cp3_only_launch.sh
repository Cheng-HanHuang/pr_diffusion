#!/usr/bin/env bash
# New-terminal, one-shot CP3-only launcher (no G2/G3/G4/G5 GPU generation).
set -euo pipefail
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$ROOT/pr_diffusion_track_r_xp_a13"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
CP2_BRANCH="codex/prdiff-track-r-xp-a15-cp2-20261030T185023Z"
PUB="$ROOT/pr_diffusion_track_r_cp2_publish_20261030T185023Z"
[[ "$HEAD" =~ ^[a-f0-9]{40}$ ]] || { echo STOP_CP3_HEAD_SHA40;exit 8; }
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -x "$PY"
test -d "$PUB"
test "$(git -C "$PUB" branch --show-current)" = "$CP2_BRANCH"
LOCAL=$(git -C "$PUB" rev-parse HEAD)
case "$LOCAL" in ef1cf10*) ;; *) echo STOP_CP2_LOCAL_HEAD;exit 8;; esac
REMOTE=$(git -C "$PUB" -c core.sshCommand='ssh -o BatchMode=yes -o ConnectTimeout=12' ls-remote git@github.com:Cheng-HanHuang/pr_diffusion.git "refs/heads/$CP2_BRANCH" | cut -f1)
test "$LOCAL" = "$REMOTE" || { echo STOP_CP2_REMOTE_NOT_PUBLISHED;exit 8; }
if test -e "$BASE/XP_A15_CP3_ONLY_LATEST_LOG.txt"; then
 echo STOP_CP3_ALREADY_LAUNCHED;exit 8
fi
if pgrep -af '[x]p_a15_cp2_d4_cpu.py|[x]p_a15_cp2_audit_cpu.py|[x]p_a15_cp3_score_cpu.py|[x]p_a15_cp3_protocol_cpu.py|[x]p_a15_cp3_only_pipeline.sh';then
 echo STOP_COMPETING_A15_SCORING;exit 8
fi
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd "$WT"
echo "===== CP3-ONLY CPU SOURCE AND REGRESSION CHECKS ====="
bash -n scripts/track_r/xp_a15_cp3_only_pipeline.sh
bash -n scripts/track_r/xp_a15_cp3_only_launch.sh
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a15_cp23.py' -v
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$BASE/XP_A15_CP3_ONLY_$STAMP.log"
SESSION="prdiff_a15_cp3_only_$STAMP"
test ! -e "$LOG"
printf -v CMD '%q ' bash scripts/track_r/xp_a15_cp3_only_pipeline.sh "$HEAD" "$STAMP"
tmux new-session -d -s "$SESSION" \
 "cd '$WT'; env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 $CMD > '$LOG' 2>&1"
printf '%s\n' "$SESSION" > "$BASE/XP_A15_CP3_ONLY_LATEST_SESSION.txt"
printf '%s\n' "$LOG" > "$BASE/XP_A15_CP3_ONLY_LATEST_LOG.txt"
echo "A15_CP3_ONLY_CPU_LAUNCHED"
echo "CP2_REMOTELY_PUBLISHED_SHA=$LOCAL"
echo "CP3_SESSION=$SESSION"
echo "CP3_LOG=$LOG"
echo "CP3_NOT_AUTHORIZED_FOR_RETURN_UNTIL_FINAL_INDEPENDENT_AUDIT"
