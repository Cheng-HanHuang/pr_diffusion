#!/usr/bin/env bash
# A15: checkpoint-only CP2 & CP3 CPU analysis, no inverse-solving GPU.
# CP2 publishes ONLY after independent exact U-statistic/protocol audit.
# CP3 remains local/provisional until NFE+SSIM/LPIPS+independent audit.
set -euo pipefail
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$ROOT/pr_diffusion_track_r_xp_a13"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
[[ "$HEAD" =~ ^[a-f0-9]{40}$ ]] || { echo STOP_A15_EXPECTED_SHA40;exit 8; }
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -x "$PY"
if test -e "$BASE/XP_A15_CP23_LATEST_SESSION.txt"; then
 echo STOP_A15_CP23_ALREADY_LAUNCHED;exit 8
fi
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$WT" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd "$WT"
echo "===== CPU SYNTAX/REGRESSION CHECK ====="
for file in scripts/track_r/xp_a15_cp23_cpu_autopilot.sh \
 scripts/track_r/xp_a15_cp23_cpu_pipeline.sh \
 scripts/track_r/xp_a15_cp2_publish.sh; do
 bash -n "$file"
done
"$PY" - <<'PY'
import ast
from pathlib import Path
for f in (
 "xp_a15_cp2_d4_cpu.py","xp_a15_cp2_audit_cpu.py",
 "xp_a15_cp3_score_cpu.py","xp_a15_cp3_protocol_cpu.py"):
 p=Path("scripts/track_r")/f
 ast.parse(p.read_text(),filename=str(p))
 print("A15_SOURCE_SYNTAX_PASS",f)
PY
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a15_cp23.py' -v
echo "===== START CHECKPOINT-SILENT DETACHED CPU PIPELINE ====="
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$BASE/XP_A15_CP23_$STAMP.log"
SESSION=prdiff_a15_cp23_$STAMP
test ! -e "$LOG"
printf -v COMMAND '%q ' bash scripts/track_r/xp_a15_cp23_cpu_pipeline.sh "$HEAD" "$STAMP"
tmux new-session -d -s "$SESSION" \
 "cd '$WT'; env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 $COMMAND > '$LOG' 2>&1"
printf '%s\n' "$SESSION" > "$BASE/XP_A15_CP23_LATEST_SESSION.txt"
printf '%s\n' "$LOG" > "$BASE/XP_A15_CP23_LATEST_LOG.txt"
echo "A15_CPU_AUTOPILOT_LAUNCHED"
echo "SESSION=$SESSION"
echo "LOG=$LOG"
echo "CP2_IS_NOT_RETURNED_UNTIL_CP2_COMMITTED_AND_PUSHED"
echo "CP3_IS_NOT_RETURNED_UNTIL_ITS_INDEPENDENT_ANALYSIS_IS_COMPLETE"
