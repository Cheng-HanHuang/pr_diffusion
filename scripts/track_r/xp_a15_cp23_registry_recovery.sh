#!/usr/bin/env bash
# Fail-closed recovery ONLY from A15's initial CP2 registry-path failure.
# Preserves the old log, old session marker, all original GPU receipts,
# and both checkpoint embargo/release gates. No new GPU generation.
set -euo pipefail
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$ROOT/pr_diffusion_track_r_xp_a13"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
OLD="956ecd174ca4e54e0bffe23fafebbe0a87a4b0c6"
REG="988aaed5fb61099ca43cbe950a181a6c38ab74c7"
MARK="$BASE/XP_A15_CP23_REGISTRY_RECOVERY_APPLIED.txt"
[[ "$HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo STOP_A15_RECOVERY_HEAD; exit 8; }
test -x "$PY"
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test ! -e "$MARK"
test -f "$BASE/XP_A15_CP23_LATEST_LOG.txt"
test -f "$BASE/XP_A15_CP23_LATEST_SESSION.txt"
PREV_LOG=$(cat "$BASE/XP_A15_CP23_LATEST_LOG.txt")
PREV_SESSION=$(cat "$BASE/XP_A15_CP23_LATEST_SESSION.txt")
case "$PREV_LOG" in "$BASE"/XP_A15_CP23_*.log) ;; *) echo STOP_PREVIOUS_LOG_PATH;exit 8;; esac
[[ "$PREV_SESSION" =~ ^prdiff_a15_cp23_[0-9]{8}T[0-9]{6}Z$ ]] || {
 echo STOP_PREVIOUS_SESSION_NAME;exit 8;
}
test -f "$PREV_LOG"
grep -Fq "f31901c808954f2bee724ab8a3834e2b65ef25c9:docs/track_r/theory/PREDICTIONS_REGISTRY.md" "$PREV_LOG"
grep -Fq "ACTION NEEDED: A15_CP2_D4" "$PREV_LOG"
if tmux has-session -t "$PREV_SESSION" 2>/dev/null; then
 echo STOP_A15_PREVIOUS_SESSION_STILL_ACTIVE; exit 8
fi
if pgrep -af '[x]p_a15_cp2_d4_cpu.py|[x]p_a15_cp2_audit_cpu.py|[x]p_a15_cp3_score_cpu.py|[x]p_a15_cp3_protocol_cpu.py|[x]p_a15_cp23_cpu_pipeline.sh'; then
 echo STOP_A15_CONCURRENT_ANALYSIS;exit 8
fi
if find "$BASE" -maxdepth 1 -type d -name 'xp_a15_cp2_d4_*' -print -quit | grep -q .;then
 echo STOP_CP2_OUTPUT_ALREADY_EXISTS; exit 8
fi
test "$(git -C "$WT" show "$REG:docs/track_r/theory/PREDICTIONS_REGISTRY.md" | grep -Fc '| v4 |')" = "1"
test "$(git -C "$WT" show "$REG:docs/track_r/theory/PREDICTIONS_REGISTRY.md" | grep -Fc 'b0ff25bc98ef7ba294db3f33c4df4f1f9835951890ebd879afedf807c9d93da3')" -ge 1
# New code must carry the historical registry SHA as a literal.
grep -Fq 'PREREG_HEAD="988aaed5fb61099ca43cbe950a181a6c38ab74c7"' "$WT/scripts/track_r/xp_a15_cp2_d4_cpu.py"
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$WT" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
cd "$WT"
echo "===== A15 RECOVERY CPU SOURCE CHECKS ====="
bash -n scripts/track_r/xp_a15_cp23_cpu_pipeline.sh
bash -n scripts/track_r/xp_a15_cp2_publish.sh
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a15_cp23.py' -v
echo "===== VERIFY PATCHED REGISTRATION PRIOR TO NEW RUN ====="
"$PY" - <<'PY'
from pathlib import Path
from scripts.track_r.xp_a15_cp2_d4_cpu import frozen, PREREG_HEAD
assert PREREG_HEAD == "988aaed5fb61099ca43cbe950a181a6c38ab74c7"
a=frozen(Path.cwd())
assert a["v4"]["status"]=="FROZEN_BEFORE_XP_D4_UNBLINDING"
print("A15_CP2_V4_FROZEN_REGISTRY_RECOVERY_PASS")
PY
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SESSION="prdiff_a15_cp23_$STAMP"
LOG="$BASE/XP_A15_CP23_$STAMP.log"
test ! -e "$LOG"
if tmux has-session -t "$SESSION" 2>/dev/null;then
 echo STOP_A15_SESSION_NAME_DUPLICATE;exit 8
fi
printf -v CMD '%q ' bash scripts/track_r/xp_a15_cp23_cpu_pipeline.sh "$HEAD" "$STAMP"
tmux new-session -d -s "$SESSION" \
 "cd '$WT'; env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 $CMD > '$LOG' 2>&1"
# Keep original log and marker as immutable source audit references.
{
 echo "status=A15_REGISTRY_SOURCE_ONLY_RECOVERY_LAUNCHED"
 echo "previous_code_head=$OLD"
 echo "previous_log=$PREV_LOG"
 echo "previous_session=$PREV_SESSION"
 echo "registry_snapshot=$REG"
 echo "recovery_code_head=$HEAD"
 echo "recovery_log=$LOG"
 echo "recovery_session=$SESSION"
} > "$MARK"
printf '%s\n' "$SESSION" > "$BASE/XP_A15_CP23_LATEST_SESSION.txt"
printf '%s\n' "$LOG" > "$BASE/XP_A15_CP23_LATEST_LOG.txt"
echo "A15_CP23_REGISTRY_RECOVERY_LAUNCHED"
echo "SESSION=$SESSION"
echo "LOG=$LOG"
echo "NO_PLANNER_RETURN_UNTIL_CP2_AUDIT_AND_CP3_RELEASE"
